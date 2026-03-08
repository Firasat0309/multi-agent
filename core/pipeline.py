"""Main pipeline: orchestrates the full code generation flow from prompt to repository."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agents.architect_agent import ArchitectAgent
from agents.planner_agent import PlannerAgent
from config.settings import Settings
from core.agent_manager import AgentManager
from core.llm_client import LLMClient
from core.models import RepositoryBlueprint
from core.observability import record_task_completion, start_metrics_server, setup_tracing
from core.repository_manager import RepositoryManager
from core.task_engine import TaskGraph
from memory.dependency_graph import DependencyGraphStore
from memory.embedding_store import EmbeddingStore
from memory.repo_index import RepoIndexStore
from sandbox.sandbox_runner import SandboxManager

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    success: bool
    workspace_path: Path
    blueprint: RepositoryBlueprint | None = None
    task_stats: dict[str, int] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0


class Pipeline:
    """End-to-end pipeline: prompt -> architecture -> tasks -> agents -> repository."""

    def __init__(self, settings: Settings | None = None) -> None:
        from config.settings import get_settings
        self.settings = settings or get_settings()
        self.llm = LLMClient(self.settings.llm)

    async def run(self, user_prompt: str) -> PipelineResult:
        """Execute the full generation pipeline."""
        start_time = time.monotonic()
        errors: list[str] = []

        logger.info("=" * 60)
        logger.info("Starting code generation pipeline")
        logger.info(f"Prompt: {user_prompt[:100]}...")
        logger.info("=" * 60)

        # Initialize observability (non-blocking)
        try:
            if self.settings.observability.enable_tracing:
                setup_tracing(self.settings.observability.otlp_endpoint)
        except Exception:
            logger.warning("Failed to initialize observability, continuing without it")

        # ── Phase 1: Architecture ─────────────────────────────────────────
        logger.info("[Phase 1] Designing architecture...")
        repo_manager = RepositoryManager(self.settings.workspace_dir)
        architect = ArchitectAgent(llm_client=self.llm, repo_manager=repo_manager)

        try:
            blueprint = await architect.design_architecture(user_prompt)
        except Exception as e:
            logger.exception("Architecture design failed")
            return PipelineResult(
                success=False,
                workspace_path=self.settings.workspace_dir,
                errors=[f"Architecture design failed: {e}"],
                elapsed_seconds=time.monotonic() - start_time,
            )

        logger.info(
            f"Blueprint: {blueprint.name} with {len(blueprint.file_blueprints)} files"
        )

        # Initialize workspace
        repo_manager.initialize(blueprint)

        # ── Phase 2: Planning ─────────────────────────────────────────────
        logger.info("[Phase 2] Building task graph...")
        planner = PlannerAgent(llm_client=self.llm, repo_manager=repo_manager)

        try:
            task_graph = await planner.create_task_graph(blueprint)
        except Exception as e:
            logger.exception("Task planning failed")
            return PipelineResult(
                success=False,
                workspace_path=self.settings.workspace_dir,
                blueprint=blueprint,
                errors=[f"Task planning failed: {e}"],
                elapsed_seconds=time.monotonic() - start_time,
            )

        logger.info(f"Task graph: {len(task_graph.tasks)} tasks")

        # ── Phase 3: Execution ────────────────────────────────────────────
        logger.info("[Phase 3] Executing tasks...")
        agent_manager = AgentManager(
            settings=self.settings,
            llm_client=self.llm,
            repo_manager=repo_manager,
            blueprint=blueprint,
        )

        try:
            exec_result = await agent_manager.execute_graph(task_graph)
        except Exception as e:
            logger.exception("Task execution failed")
            return PipelineResult(
                success=False,
                workspace_path=self.settings.workspace_dir,
                blueprint=blueprint,
                errors=[f"Task execution failed: {e}"],
                elapsed_seconds=time.monotonic() - start_time,
            )

        # ── Phase 4: Finalize ─────────────────────────────────────────────
        logger.info("[Phase 4] Finalizing repository...")

        # Index all generated files into memory stores
        try:
            self._index_workspace(repo_manager)
        except Exception as e:
            logger.warning(f"Indexing failed (non-critical): {e}")

        elapsed = time.monotonic() - start_time
        stats = exec_result.get("stats", {})
        success = stats.get("failed", 0) == 0 and stats.get("blocked", 0) == 0

        logger.info("=" * 60)
        logger.info(f"Pipeline {'SUCCEEDED' if success else 'COMPLETED WITH ISSUES'}")
        logger.info(f"Stats: {stats}")
        logger.info(f"Elapsed: {elapsed:.1f}s")
        logger.info(f"Workspace: {self.settings.workspace_dir}")
        logger.info("=" * 60)

        return PipelineResult(
            success=success,
            workspace_path=self.settings.workspace_dir,
            blueprint=blueprint,
            task_stats=stats,
            metrics=exec_result.get("metrics", {}),
            errors=errors,
            elapsed_seconds=elapsed,
        )

    def _index_workspace(self, repo_manager: RepositoryManager) -> None:
        """Index all generated files into memory stores."""
        index_store = RepoIndexStore(self.settings.workspace_dir)
        dep_store = DependencyGraphStore(self.settings.workspace_dir)
        embedding_store = EmbeddingStore(
            persist_dir=self.settings.memory.chroma_persist_dir
        )

        for file_info in repo_manager.get_repo_index().files:
            index_store.update_file(file_info)

            # Update dependency graph
            for imp in file_info.imports:
                dep_store.add_dependency(file_info.path, imp)

            # Index embeddings
            content = repo_manager.read_file(file_info.path)
            if content:
                embedding_store.index_file(file_info.path, content)

        dep_store.save()
        index_store.save()
