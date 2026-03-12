"""Pipeline facade — public entry point for generation and enhancement runs."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config.settings import Settings
from core.llm_client import LLMClient, LLMConfigError
from core.live_console import LiveConsole, LiveConsoleHandler
from core.models import RepositoryBlueprint, ChangePlan, RepoAnalysis, TokenCost

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    success: bool
    workspace_path: Path
    blueprint: RepositoryBlueprint | None = None
    change_plan: ChangePlan | None = None
    repo_analysis: RepoAnalysis | None = None
    task_stats: dict[str, int] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    token_cost: TokenCost | None = None


class Pipeline:
    """Thin facade: manages console/logging lifecycle, then delegates to
    ``RunPipeline`` (new projects) or ``EnhancePipeline`` (modifications).

    All orchestration logic lives in those two focused classes; this class
    exists only to own the cross-cutting concerns that wrap every run:
      - LiveConsole start/stop
      - Per-run file log attachment/detachment
      - LLMClient construction
    """

    def __init__(self, settings: Settings | None = None, interactive: bool = True) -> None:
        from config.settings import get_settings
        self.settings = settings or get_settings()
        self.interactive = interactive
        self._live: LiveConsole | None = None
        self.llm = LLMClient(self.settings.llm)

    # ── Public API ────────────────────────────────────────────────────────────

    async def run(self, user_prompt: str) -> PipelineResult:
        """Generate a new project from *user_prompt*."""
        from core.pipeline_run import RunPipeline

        start_time = time.monotonic()
        self._start_live()
        fh = self._start_file_logging(self.settings.workspace_dir)
        try:
            return await RunPipeline(self.settings, self.llm, self._live).execute(
                user_prompt, start_time,
            )
        finally:
            self._stop_live()
            self._stop_file_logging(fh)

    async def enhance(self, user_prompt: str) -> PipelineResult:
        """Modify an existing repository according to *user_prompt*."""
        from core.pipeline_enhance import EnhancePipeline

        start_time = time.monotonic()
        self._start_live()
        fh = self._start_file_logging(self.settings.workspace_dir)
        try:
            return await EnhancePipeline(
                self.settings, self.llm, self._live, self.interactive,
            ).execute(user_prompt, start_time)
        finally:
            self._stop_live()
            self._stop_file_logging(fh)

    async def run_fullstack(
        self,
        user_prompt: str,
        figma_url: str = "",
    ) -> PipelineResult:
        """Generate a complete fullstack project (backend + frontend) from *user_prompt*.

        Optionally accepts a *figma_url* for design parsing.  The pipeline
        generates:
          - `workspace/backend/`  — the backend API (via RunPipeline)
          - `workspace/frontend/` — the React/Next.js/Vue UI (via FrontendPipeline)
        with a shared OpenAPI contract as the handshake between the two.
        """
        from core.pipeline_fullstack import FullstackPipeline

        start_time = time.monotonic()
        self._start_live()
        fh = self._start_file_logging(self.settings.workspace_dir)
        try:
            return await FullstackPipeline(
                self.settings, self.llm, self._live, self.interactive,
            ).execute(user_prompt, start_time, figma_url=figma_url)
        finally:
            self._stop_live()
            self._stop_file_logging(fh)

    # ── Console / logging lifecycle ───────────────────────────────────────────

    def _start_live(self) -> None:
        if not self.interactive:
            return
        self._live = LiveConsole()
        handler = LiveConsoleHandler(self._live)
        handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(handler)
        self._live.start()

    def _stop_live(self) -> None:
        if self._live:
            self._live.stop()
            self._live = None

    def _start_file_logging(self, workspace: Path) -> logging.FileHandler:
        workspace.mkdir(parents=True, exist_ok=True)
        log_path = workspace / "run.log"
        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        ))
        logging.getLogger().addHandler(fh)
        return fh

    def _stop_file_logging(self, fh: logging.FileHandler) -> None:
        logging.getLogger().removeHandler(fh)
        fh.close()
