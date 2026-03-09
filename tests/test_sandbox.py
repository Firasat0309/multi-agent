"""Tests for the two-tier sandbox isolation model."""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from config.settings import SandboxConfig, SandboxTier, SandboxType
from sandbox.sandbox_runner import (
    DockerSandbox,
    LocalSandbox,
    SandboxInfo,
    SandboxManager,
)


def _run(coro):
    """Helper to run async code in sync tests."""
    return asyncio.run(coro)


# ── SandboxTier enum ─────────────────────────────────────────────────────────


class TestSandboxTier:
    def test_tier_values(self):
        assert SandboxTier.BUILD == "build"
        assert SandboxTier.TEST == "test"


# ── SandboxInfo includes tier ────────────────────────────────────────────────


class TestSandboxInfo:
    def test_info_has_tier_field(self):
        info = SandboxInfo(
            sandbox_id="abc",
            sandbox_type=SandboxType.DOCKER,
            tier=SandboxTier.TEST,
            workspace_path=Path("/tmp"),
        )
        assert info.tier == SandboxTier.TEST


# ── LocalSandbox ─────────────────────────────────────────────────────────────


class TestLocalSandbox:
    @pytest.fixture
    def sandbox(self):
        return LocalSandbox()

    def test_create_build_tier(self, sandbox, tmp_path):
        info = _run(sandbox.create(tmp_path, tier=SandboxTier.BUILD))
        assert info.tier == SandboxTier.BUILD
        assert info.sandbox_type == SandboxType.LOCAL
        assert "build" in info.sandbox_id

    def test_create_test_tier(self, sandbox, tmp_path):
        info = _run(sandbox.create(tmp_path, tier=SandboxTier.TEST))
        assert info.tier == SandboxTier.TEST
        assert "test" in info.sandbox_id

    def test_creates_unique_ids(self, sandbox, tmp_path):
        info1 = _run(sandbox.create(tmp_path, tier=SandboxTier.BUILD))
        info2 = _run(sandbox.create(tmp_path, tier=SandboxTier.TEST))
        assert info1.sandbox_id != info2.sandbox_id


# ── DockerSandbox (mocked docker client) ─────────────────────────────────────


class TestDockerSandboxTiers:
    """Verify Docker container configuration differs by tier."""

    def _make_sandbox(self, **overrides):
        defaults = dict(
            sandbox_type=SandboxType.DOCKER,
            image="python:3.12-slim",
            memory_limit="512m",
            cpu_limit=1.0,
            timeout_seconds=300,
            network_enabled=False,
        )
        defaults.update(overrides)
        return DockerSandbox(SandboxConfig(**defaults))

    def _mock_docker_module(self):
        """Create a mock docker module with containers.run returning a container mock."""
        mock_container = MagicMock()
        mock_container.id = "abcdef123456"
        mock_module = MagicMock()
        mock_module.from_env.return_value.containers.run.return_value = mock_container
        return mock_module

    def test_build_tier_has_network(self, tmp_path):
        sandbox = self._make_sandbox()
        mock_docker = self._mock_docker_module()

        with patch.dict("sys.modules", {"docker": mock_docker}):
            info = _run(sandbox.create(tmp_path, tier=SandboxTier.BUILD))

            call_kwargs = mock_docker.from_env.return_value.containers.run.call_args
            assert call_kwargs.kwargs["network_disabled"] is False
            assert "read_only" not in call_kwargs.kwargs
            assert info.tier == SandboxTier.BUILD

    def test_test_tier_no_network(self, tmp_path):
        sandbox = self._make_sandbox()
        mock_docker = self._mock_docker_module()

        with patch.dict("sys.modules", {"docker": mock_docker}):
            info = _run(sandbox.create(tmp_path, tier=SandboxTier.TEST))

            call_kwargs = mock_docker.from_env.return_value.containers.run.call_args
            assert call_kwargs.kwargs["network_disabled"] is True
            assert info.tier == SandboxTier.TEST

    def test_test_tier_read_only_rootfs(self, tmp_path):
        sandbox = self._make_sandbox()
        mock_docker = self._mock_docker_module()

        with patch.dict("sys.modules", {"docker": mock_docker}):
            _run(sandbox.create(tmp_path, tier=SandboxTier.TEST))

            call_kwargs = mock_docker.from_env.return_value.containers.run.call_args
            assert call_kwargs.kwargs["read_only"] is True

    def test_test_tier_has_tmpfs(self, tmp_path):
        sandbox = self._make_sandbox()
        mock_docker = self._mock_docker_module()

        with patch.dict("sys.modules", {"docker": mock_docker}):
            _run(sandbox.create(tmp_path, tier=SandboxTier.TEST))

            call_kwargs = mock_docker.from_env.return_value.containers.run.call_args
            assert "/tmp" in call_kwargs.kwargs["tmpfs"]

    def test_build_tier_mounts_cache_rw(self, tmp_path):
        sandbox = self._make_sandbox(image="maven:3.9-eclipse-temurin-21-alpine")
        mock_docker = self._mock_docker_module()
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        with patch.dict("sys.modules", {"docker": mock_docker}):
            _run(sandbox.create(
                tmp_path, language_name="java",
                tier=SandboxTier.BUILD, cache_dir=cache_dir,
            ))

            call_kwargs = mock_docker.from_env.return_value.containers.run.call_args
            volumes = call_kwargs.kwargs["volumes"]
            # /root/.m2 should get a rw mount
            m2_mount = [v for k, v in volumes.items() if v["bind"] == "/root/.m2"]
            assert len(m2_mount) == 1
            assert m2_mount[0]["mode"] == "rw"

    def test_test_tier_mounts_cache_rw(self, tmp_path):
        """Cache is rw in both tiers — Maven/pip need write access for logs and locks."""
        sandbox = self._make_sandbox(image="maven:3.9-eclipse-temurin-21-alpine")
        mock_docker = self._mock_docker_module()
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        with patch.dict("sys.modules", {"docker": mock_docker}):
            _run(sandbox.create(
                tmp_path, language_name="java",
                tier=SandboxTier.TEST, cache_dir=cache_dir,
            ))

            call_kwargs = mock_docker.from_env.return_value.containers.run.call_args
            volumes = call_kwargs.kwargs["volumes"]
            m2_mount = [v for k, v in volumes.items() if v["bind"] == "/root/.m2"]
            assert len(m2_mount) == 1
            assert m2_mount[0]["mode"] == "rw"

    def test_no_cache_mount_without_cache_dir(self, tmp_path):
        sandbox = self._make_sandbox(image="maven:3.9-eclipse-temurin-21-alpine")
        mock_docker = self._mock_docker_module()

        with patch.dict("sys.modules", {"docker": mock_docker}):
            _run(sandbox.create(tmp_path, language_name="java", tier=SandboxTier.BUILD))

            call_kwargs = mock_docker.from_env.return_value.containers.run.call_args
            volumes = call_kwargs.kwargs["volumes"]
            # Only the workspace mount should be present
            binds = [v["bind"] for v in volumes.values()]
            assert "/root/.m2" not in binds


# ── SandboxManager ───────────────────────────────────────────────────────────


class TestSandboxManager:
    def test_create_both_tiers(self, tmp_path):
        config = SandboxConfig(sandbox_type=SandboxType.LOCAL)
        manager = SandboxManager(config)

        build = _run(manager.create_sandbox(tmp_path, tier=SandboxTier.BUILD))
        test = _run(manager.create_sandbox(tmp_path, tier=SandboxTier.TEST))

        assert build.tier == SandboxTier.BUILD
        assert test.tier == SandboxTier.TEST
        assert build.sandbox_id != test.sandbox_id

        # Both are tracked
        assert build.sandbox_id in manager._active
        assert test.sandbox_id in manager._active

    def test_destroy_all(self, tmp_path):
        config = SandboxConfig(sandbox_type=SandboxType.LOCAL)
        manager = SandboxManager(config)

        _run(manager.create_sandbox(tmp_path, tier=SandboxTier.BUILD))
        _run(manager.create_sandbox(tmp_path, tier=SandboxTier.TEST))
        assert len(manager._active) == 2

        _run(manager.destroy_all())
        assert len(manager._active) == 0

    def test_docker_manager_creates_cache_dir(self, tmp_path):
        config = SandboxConfig(sandbox_type=SandboxType.DOCKER)
        manager = SandboxManager(config)

        cache = manager._ensure_cache_dir()
        assert cache.exists()
        assert cache.is_dir()

        # Idempotent
        assert manager._ensure_cache_dir() is cache

        # Cleanup on destroy_all
        _run(manager.destroy_all())
        assert manager._cache_dir is None

    def test_local_manager_no_cache_dir(self, tmp_path):
        config = SandboxConfig(sandbox_type=SandboxType.LOCAL)
        manager = SandboxManager(config)

        _run(manager.create_sandbox(tmp_path, tier=SandboxTier.BUILD))
        assert manager._cache_dir is None


# ── Settings ─────────────────────────────────────────────────────────────────


class TestSettingsHostExecution:
    def test_allow_host_execution_default_false(self):
        from config.settings import Settings
        s = Settings()
        assert s.allow_host_execution is False

    def test_allow_host_execution_can_be_set(self):
        from config.settings import Settings
        s = Settings(allow_host_execution=True)
        assert s.allow_host_execution is True
