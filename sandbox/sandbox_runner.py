"""Sandbox runner supporting Docker and local execution modes.

Two-tier Docker isolation model:

  BUILD sandbox — network enabled (dependency fetching), workspace rw.
                  Dependency caches (/root/.m2, /root/.cache/pip, …) are
                  written to a shared *host temp directory* so the TEST
                  tier can reuse them without network access.
  TEST  sandbox — network disabled, read-only rootfs, tmpfs for /tmp.
                  The shared cache volume is mounted read-only so the
                  test harness can resolve pre-fetched dependencies.
                  Prevents LLM-generated test code from exfiltrating data
                  or mutating the host outside the workspace.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config.settings import SandboxConfig, SandboxTier, SandboxType
from tools.terminal_tools import CommandResult

logger = logging.getLogger(__name__)


@dataclass
class SandboxInfo:
    sandbox_id: str
    sandbox_type: SandboxType
    tier: SandboxTier
    workspace_path: Path
    status: str = "created"


class SandboxBase(ABC):
    """Abstract sandbox interface."""

    @abstractmethod
    async def create(
        self,
        workspace_path: Path,
        language_name: str = "python",
        tier: SandboxTier = SandboxTier.BUILD,
        cache_dir: Path | None = None,
    ) -> SandboxInfo:
        ...

    @abstractmethod
    async def execute(self, sandbox_id: str, command: str) -> CommandResult:
        ...

    @abstractmethod
    async def destroy(self, sandbox_id: str) -> None:
        ...

    @abstractmethod
    async def copy_to(self, sandbox_id: str, src: Path, dest: str) -> None:
        ...

    @abstractmethod
    async def copy_from(self, sandbox_id: str, src: str, dest: Path) -> None:
        ...


class DockerSandbox(SandboxBase):
    """Docker-based sandbox for isolated code execution."""

    def __init__(self, config: SandboxConfig) -> None:
        self.config = config
        self._containers: dict[str, Any] = {}

    async def create(
        self,
        workspace_path: Path,
        language_name: str = "python",
        tier: SandboxTier = SandboxTier.BUILD,
        cache_dir: Path | None = None,
    ) -> SandboxInfo:
        import docker
        client = docker.from_env()

        # Auto-detect image from language profile when not explicitly configured
        image = self.config.image
        from core.language import get_language_profile
        profile = get_language_profile(language_name)
        if not image:
            image = profile.docker_image

        # ── Build volume mapping ──────────────────────────────────────
        volumes: dict[str, dict[str, str]] = {
            str(workspace_path.resolve()): {
                "bind": "/workspace",
                "mode": "rw",
            },
        }

        # Mount the shared cache directory for every cache_path the
        # language profile declares.  Both tiers get rw — tools like Maven
        # need write access for lock files, logs, and plugin metadata even
        # during test execution.  Isolation is enforced by network_disabled
        # and read_only rootfs, not by the cache mount mode.
        if cache_dir and profile.cache_paths:
            for container_path in profile.cache_paths:
                # One host sub-directory per container-path to avoid collisions
                host_subdir = cache_dir / container_path.strip("/").replace("/", "_")
                host_subdir.mkdir(parents=True, exist_ok=True)
                volumes[str(host_subdir)] = {
                    "bind": container_path,
                    "mode": "rw",
                }

        # ── Tier-specific container configuration ─────────────────────
        if tier == SandboxTier.TEST:
            # TEST tier: no network, read-only rootfs, tmpfs for /tmp.
            # Workspace is mounted rw so the test-agent fix loop can still
            # write updated files from the host side;  the read-only rootfs
            # prevents the *generated code* from writing outside /workspace
            # and /tmp.
            container = client.containers.run(
                image,
                command="sleep infinity",
                detach=True,
                mem_limit=self.config.memory_limit,
                nano_cpus=int(self.config.cpu_limit * 1e9),
                network_disabled=True,
                read_only=True,
                volumes=volumes,
                tmpfs={"/tmp": "size=256m"},
                working_dir="/workspace",
                remove=True,
            )
            logger.info("Created TEST sandbox (network=none, rootfs=ro, tmpfs=/tmp)")
        else:
            # BUILD tier: network access allowed (needs to fetch deps).
            container = client.containers.run(
                image,
                command="sleep infinity",
                detach=True,
                mem_limit=self.config.memory_limit,
                nano_cpus=int(self.config.cpu_limit * 1e9),
                network_disabled=False,
                volumes=volumes,
                working_dir="/workspace",
                remove=True,
            )
            logger.info("Created BUILD sandbox (network=on)")

        sandbox_id = container.id[:12]
        self._containers[sandbox_id] = container

        return SandboxInfo(
            sandbox_id=sandbox_id,
            sandbox_type=SandboxType.DOCKER,
            tier=tier,
            workspace_path=workspace_path,
            status="running",
        )

    async def execute(self, sandbox_id: str, command: str) -> CommandResult:
        container = self._containers.get(sandbox_id)
        if not container:
            return CommandResult(exit_code=-1, stdout="", stderr="Sandbox not found")

        try:
            exec_result = container.exec_run(
                ["bash", "-c", command],
                workdir="/workspace",
                demux=True,
            )
            stdout = exec_result.output[0].decode("utf-8") if exec_result.output[0] else ""
            stderr = exec_result.output[1].decode("utf-8") if exec_result.output[1] else ""
            return CommandResult(
                exit_code=exec_result.exit_code,
                stdout=stdout,
                stderr=stderr,
            )
        except Exception as e:
            return CommandResult(exit_code=-1, stdout="", stderr=str(e))

    async def destroy(self, sandbox_id: str) -> None:
        container = self._containers.pop(sandbox_id, None)
        if container:
            try:
                container.stop(timeout=5)
            except Exception:
                container.kill()
            logger.info(f"Destroyed Docker sandbox {sandbox_id}")

    async def copy_to(self, sandbox_id: str, src: Path, dest: str) -> None:
        container = self._containers.get(sandbox_id)
        if container:
            import tarfile, io
            tar_stream = io.BytesIO()
            with tarfile.open(fileobj=tar_stream, mode="w") as tar:
                tar.add(str(src), arcname=Path(dest).name)
            tar_stream.seek(0)
            container.put_archive(str(Path(dest).parent), tar_stream)

    async def copy_from(self, sandbox_id: str, src: str, dest: Path) -> None:
        container = self._containers.get(sandbox_id)
        if container:
            import tarfile, io
            bits, _ = container.get_archive(src)
            tar_stream = io.BytesIO()
            for chunk in bits:
                tar_stream.write(chunk)
            tar_stream.seek(0)
            with tarfile.open(fileobj=tar_stream) as tar:
                tar.extractall(path=str(dest))


class LocalSandbox(SandboxBase):
    """Local sandbox for development/testing (no isolation)."""

    def __init__(self) -> None:
        self._workspaces: dict[str, Path] = {}
        self._counter = 0

    async def create(self, workspace_path: Path, language_name: str = "python", tier: SandboxTier = SandboxTier.BUILD, cache_dir: Path | None = None) -> SandboxInfo:
        self._counter += 1
        sandbox_id = f"local-{tier.value}-{self._counter}"
        self._workspaces[sandbox_id] = workspace_path
        logger.warning(
            "LocalSandbox provides NO isolation — generated code runs directly "
            "on the host.  Use Docker sandbox for production."
        )
        return SandboxInfo(
            sandbox_id=sandbox_id,
            sandbox_type=SandboxType.LOCAL,
            tier=tier,
            workspace_path=workspace_path,
            status="running",
        )

    async def execute(self, sandbox_id: str, command: str) -> CommandResult:
        workspace = self._workspaces.get(sandbox_id)
        if not workspace:
            return CommandResult(exit_code=-1, stdout="", stderr="Sandbox not found")

        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=str(workspace),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        return CommandResult(
            exit_code=proc.returncode or 0,
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
        )

    async def destroy(self, sandbox_id: str) -> None:
        self._workspaces.pop(sandbox_id, None)

    async def copy_to(self, sandbox_id: str, src: Path, dest: str) -> None:
        workspace = self._workspaces.get(sandbox_id)
        if workspace:
            target = workspace / dest
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)

    async def copy_from(self, sandbox_id: str, src: str, dest: Path) -> None:
        workspace = self._workspaces.get(sandbox_id)
        if workspace:
            source = workspace / src
            if source.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)


class SandboxManager:
    """Factory and lifecycle manager for sandboxes."""

    def __init__(self, config: SandboxConfig) -> None:
        self.config = config
        self._active: dict[str, tuple[SandboxBase, SandboxInfo]] = {}
        # Shared directory for dependency caches between BUILD ↔ TEST tiers.
        # Created lazily on first Docker sandbox, cleaned up in destroy_all().
        self._cache_dir: Path | None = None

    def _ensure_cache_dir(self) -> Path:
        """Return (and lazily create) the shared host-side cache directory."""
        if self._cache_dir is None:
            self._cache_dir = Path(tempfile.mkdtemp(prefix="mac_cache_"))
        return self._cache_dir

    def _create_sandbox_backend(self) -> SandboxBase:
        if self.config.sandbox_type == SandboxType.DOCKER:
            return DockerSandbox(self.config)
        return LocalSandbox()

    async def create_sandbox(
        self,
        workspace_path: Path,
        language_name: str = "python",
        tier: SandboxTier = SandboxTier.BUILD,
    ) -> SandboxInfo:
        backend = self._create_sandbox_backend()
        cache_dir = (
            self._ensure_cache_dir()
            if self.config.sandbox_type == SandboxType.DOCKER
            else None
        )
        info = await backend.create(
            workspace_path,
            language_name=language_name,
            tier=tier,
            cache_dir=cache_dir,
        )
        self._active[info.sandbox_id] = (backend, info)
        return info

    async def execute(self, sandbox_id: str, command: str) -> CommandResult:
        if sandbox_id not in self._active:
            return CommandResult(exit_code=-1, stdout="", stderr="Unknown sandbox")
        backend, _ = self._active[sandbox_id]
        return await backend.execute(sandbox_id, command)

    async def destroy_sandbox(self, sandbox_id: str) -> None:
        if sandbox_id in self._active:
            backend, _ = self._active.pop(sandbox_id)
            await backend.destroy(sandbox_id)

    async def destroy_all(self) -> None:
        for sid in list(self._active.keys()):
            await self.destroy_sandbox(sid)
        # Clean up the shared cache directory
        if self._cache_dir and self._cache_dir.exists():
            shutil.rmtree(self._cache_dir, ignore_errors=True)
            self._cache_dir = None
