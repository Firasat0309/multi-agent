"""Sandbox lifecycle management — create and destroy Docker sandboxes for a pipeline run."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from config.settings import SandboxConfig, SandboxTier, SandboxType, Settings

if TYPE_CHECKING:
    from core.language import LanguageProfile
    from sandbox.sandbox_runner import SandboxManager

logger = logging.getLogger(__name__)


class SandboxOrchestrator:
    """Manages the two-tier sandbox lifecycle for a single pipeline run.

    Two-tier model:
    - BUILD sandbox: network-enabled for fetching dependencies (pip, mvn, npm, …)
    - TEST  sandbox: no-network, read-only rootfs — prevents LLM-generated test
                     code from exfiltrating data or mutating the host filesystem.

    Usage::

        orch = SandboxOrchestrator(settings)
        result = await orch.setup(lang_profile)
        try:
            # use result.sandbox_manager, result.build_id, result.test_id
            ...
        finally:
            await orch.teardown()
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._manager: SandboxManager | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    async def setup(self, lang_profile: LanguageProfile) -> SandboxSetupResult:
        """Create BUILD and TEST sandboxes.

        Returns ``SandboxSetupResult`` with manager and sandbox IDs.
        On failure, raises if ``allow_host_execution`` is False; otherwise
        logs a warning and returns a result with ``manager=None``.
        """
        if self._settings.sandbox.sandbox_type != SandboxType.DOCKER:
            logger.warning(
                "Running in LOCAL mode with NO sandbox isolation.  "
                "Use Docker for production workloads."
            )
            return SandboxSetupResult(manager=None, build_id=None, test_id=None)

        from sandbox.sandbox_runner import SandboxManager

        try:
            manager = SandboxManager(self._settings.sandbox)
            build_info = await manager.create_sandbox(
                self._settings.workspace_dir,
                language_name=lang_profile.name,
                tier=SandboxTier.BUILD,
            )
            test_info = await manager.create_sandbox(
                self._settings.workspace_dir,
                language_name=lang_profile.name,
                tier=SandboxTier.TEST,
            )
            self._manager = manager
            logger.info(
                "Sandboxes created: BUILD=%s TEST=%s",
                build_info.sandbox_id, test_info.sandbox_id,
            )
            return SandboxSetupResult(
                manager=manager,
                build_id=build_info.sandbox_id,
                test_id=test_info.sandbox_id,
            )
        except Exception as e:
            if self._settings.allow_host_execution:
                logger.warning(
                    "Docker sandbox unavailable (%s); falling back to host execution", e
                )
                return SandboxSetupResult(manager=None, build_id=None, test_id=None)
            raise SandboxUnavailableError(
                f"Docker sandbox unavailable: {e}.  "
                "Either start Docker or pass --allow-host-execution (NOT recommended)."
            ) from e

    async def teardown(self) -> None:
        """Destroy all sandboxes created by this orchestrator (non-blocking on error)."""
        if self._manager is None:
            return
        try:
            await self._manager.destroy_all()
            logger.info("Sandboxes destroyed")
        except Exception as e:
            logger.warning("Sandbox teardown failed (non-critical): %s", e)
        finally:
            self._manager = None


class SandboxUnavailableError(RuntimeError):
    """Raised when Docker is unavailable and host execution is not allowed."""


class SandboxSetupResult:
    """Holds sandbox manager and IDs returned by ``SandboxOrchestrator.setup()``."""

    __slots__ = ("manager", "build_id", "test_id")

    def __init__(
        self,
        manager: SandboxManager | None,
        build_id: str | None,
        test_id: str | None,
    ) -> None:
        self.manager = manager
        self.build_id = build_id
        self.test_id = test_id
