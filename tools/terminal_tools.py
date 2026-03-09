"""Terminal/command execution tools for agents."""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from core.language import LanguageProfile, PYTHON

if TYPE_CHECKING:
    from sandbox.sandbox_runner import SandboxManager

logger = logging.getLogger(__name__)


@dataclass
class CommandResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


# Commands that use a shared build directory (target/, bin/, obj/, node_modules/.cache).
# Concurrent invocations of these tools corrupt each other's build artifacts,
# so they must be serialized via _build_lock.
# - mvn/gradle: Maven target/, Gradle build/
# - dotnet: bin/, obj/
# - cargo: target/
# - go: shared module cache and build cache under GOPATH/GOMODCACHE
# - npx/tsc/node: node_modules/.cache, dist/, concurrent npx installs race
_EXCLUSIVE_BUILD_COMMANDS = frozenset({
    "mvn", "gradle", "dotnet", "cargo",
    "go", "npx", "tsc", "node",
})


class TerminalTools:
    """Provides command execution capabilities — locally or inside a Docker sandbox.

    When a ``sandbox_manager`` and ``sandbox_id`` are provided, commands are
    routed through the Docker container instead of running on the host.  This
    gives full isolation (filesystem, network, resource limits) for generated
    code execution without any agent code changes.
    """

    # Class-level lock shared across all TerminalTools instances.
    # Prevents concurrent build/test commands from fighting over shared
    # build directories (Maven target/, Cargo target/, dotnet bin/obj/).
    _build_lock: asyncio.Lock | None = None

    @classmethod
    def _get_build_lock(cls) -> asyncio.Lock:
        if cls._build_lock is None:
            cls._build_lock = asyncio.Lock()
        return cls._build_lock

    def __init__(
        self,
        working_dir: Path,
        timeout: int = 120,
        language: LanguageProfile | None = None,
        sandbox_manager: SandboxManager | None = None,
        sandbox_id: str | None = None,
    ) -> None:
        self.working_dir = working_dir
        self.timeout = timeout
        self._lang = language or PYTHON
        self._allowed_commands = set(self._lang.allowed_commands)
        self._sandbox_manager = sandbox_manager
        self._sandbox_id = sandbox_id

    @property
    def is_sandboxed(self) -> bool:
        return self._sandbox_manager is not None and self._sandbox_id is not None

    async def run_command(self, command: str) -> CommandResult:
        """Execute a command in the working directory.

        Uses execvp-style execution (no shell) to prevent command injection.
        The command string is split via shlex so options and quoted args work,
        but shell metacharacters (;, &&, |, >, ` etc.) are never interpreted.

        For compiled-language build tools (mvn, gradle, cargo, dotnet) that
        share a mutable build directory, execution is serialized via a class-
        level asyncio.Lock to prevent concurrent builds from corrupting each
        other's artifacts.
        """
        try:
            parts = shlex.split(command)
        except ValueError as e:
            return CommandResult(exit_code=1, stdout="", stderr=f"Invalid command: {e}")

        if not parts:
            return CommandResult(exit_code=1, stdout="", stderr="Empty command")

        base_cmd = parts[0]
        if base_cmd not in self._allowed_commands:
            logger.warning(f"Blocked command: {base_cmd}")
            return CommandResult(
                exit_code=1,
                stdout="",
                stderr=f"Command not allowed: {base_cmd}",
            )

        # Serialize build tools that share mutable state (target/, bin/, obj/)
        if base_cmd in _EXCLUSIVE_BUILD_COMMANDS:
            async with self._get_build_lock():
                return await self._execute(parts)
        else:
            return await self._execute(parts)

    async def _execute(self, parts: list[str]) -> CommandResult:
        """Route to sandbox or local execution."""
        if self.is_sandboxed:
            return await self._sandbox_execute(parts)
        return await self._local_execute(parts)

    async def _sandbox_execute(self, parts: list[str]) -> CommandResult:
        """Execute command inside the Docker sandbox container."""
        assert self._sandbox_manager is not None and self._sandbox_id is not None

        # Build env-var prefix for the container
        # Inside Docker the workspace is mounted at /workspace
        env_parts = ["PYTHONPATH=/workspace"]
        if self._lang.source_root:
            env_parts.insert(0, f"PYTHONPATH=/workspace/{self._lang.source_root}:/workspace")
            env_parts = env_parts[:1]  # keep only the combined one

        env_export = " && ".join(f"export {e}" for e in env_parts)
        cmd_str = " ".join(shlex.quote(p) for p in parts)
        full_cmd = f"{env_export} && cd /workspace && {cmd_str}"

        try:
            result = await asyncio.wait_for(
                self._sandbox_manager.execute(self._sandbox_id, full_cmd),
                timeout=self.timeout,
            )
            return result
        except asyncio.TimeoutError:
            return CommandResult(
                exit_code=-1,
                stdout="",
                stderr=f"Sandbox command timed out after {self.timeout}s",
                timed_out=True,
            )
        except Exception as e:
            return CommandResult(exit_code=-1, stdout="", stderr=f"Sandbox error: {e}")

    async def _local_execute(self, parts: list[str]) -> CommandResult:
        """Low-level subprocess execution on the host."""
        # Build environment — include source root in PYTHONPATH so imports resolve
        env = {**os.environ, "PYTHONPATH": str(self.working_dir)}
        if self._lang.source_root:
            src_dir = str(self.working_dir / self._lang.source_root)
            env["PYTHONPATH"] = f"{src_dir}{os.pathsep}{env['PYTHONPATH']}"

        # Reduce host state bleed: hint to build tools to prefer local/cached
        # artifacts where possible.  This doesn't fully isolate (only Docker
        # sandbox does), but discourages generated code from silently
        # downloading arbitrary dependencies during test runs.
        #
        # NOTE: We intentionally do NOT force Maven offline (-o) or Go
        # -mod=readonly here — first builds in a fresh workspace legitimately
        # need to fetch dependencies.  Instead we use "prefer offline" hints
        # that still allow downloads when the cache is empty.
        env.setdefault("npm_config_prefer_offline", "true")
        # Go: default to -mod=mod (allow go.sum updates) but don't duplicate
        existing_goflags = env.get("GOFLAGS", "")
        if "-mod=" not in existing_goflags:
            env["GOFLAGS"] = f"{existing_goflags} -mod=mod".strip()

        try:
            proc = await asyncio.create_subprocess_exec(
                *parts,
                cwd=str(self.working_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=self.timeout
                )
                return CommandResult(
                    exit_code=proc.returncode or 0,
                    stdout=stdout.decode("utf-8", errors="replace"),
                    stderr=stderr.decode("utf-8", errors="replace"),
                )
            except asyncio.TimeoutError:
                proc.kill()
                return CommandResult(
                    exit_code=-1,
                    stdout="",
                    stderr=f"Command timed out after {self.timeout}s",
                    timed_out=True,
                )
        except Exception as e:
            return CommandResult(exit_code=-1, stdout="", stderr=str(e))

    async def run_tests(self, test_path: str = "") -> CommandResult:
        """Run tests using the language-appropriate test runner.

        When running locally (no sandbox), emits a warning because generated
        test code could make outbound network calls, access the host
        filesystem, or interact with local services.  In sandbox mode, Docker
        network/filesystem isolation handles this automatically.
        """
        if not self.is_sandboxed:
            logger.warning(
                "Running tests on HOST without sandbox isolation — generated "
                "test code may access network, filesystem, or local services. "
                "Consider enabling Docker sandbox for production use."
            )
        cmd = self._lang.test_command
        if test_path:
            cmd += f" {test_path}"
        return await self.run_command(cmd)

    async def run_linter(self, file_path: str = ".") -> CommandResult:
        """Run linter for the configured language."""
        if not self._lang.lint_command:
            return CommandResult(exit_code=0, stdout="No linter configured", stderr="")
        return await self.run_command(f"{self._lang.lint_command} {file_path}")

    async def run_type_check(self, file_path: str = ".") -> CommandResult:
        """Run type checker for the configured language."""
        if not self._lang.type_check_command:
            return CommandResult(exit_code=0, stdout="No type checker configured", stderr="")
        return await self.run_command(f"{self._lang.type_check_command} {file_path}")

    async def run_security_scan(self, file_path: str = ".") -> CommandResult:
        """Run security scanner for the configured language."""
        if not self._lang.security_scan_command:
            return CommandResult(exit_code=0, stdout="No security scanner configured", stderr="")
        return await self.run_command(f"{self._lang.security_scan_command}")

    async def run_build(self) -> CommandResult:
        """Run build command for the configured language."""
        if not self._lang.build_command:
            return CommandResult(exit_code=0, stdout="No build step required", stderr="")
        return await self.run_command(self._lang.build_command)
