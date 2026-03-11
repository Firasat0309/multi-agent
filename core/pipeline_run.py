"""Generation pipeline — new project from prompt to repository."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from agents.architect_agent import ArchitectAgent
from agents.planner_agent import PlannerAgent
from config.settings import Settings, SandboxType
from core.agent_manager import AgentManager
from core.event_bus import EventBus
from core.language import detect_language_from_blueprint
from core.llm_client import LLMClient, LLMConfigError, calculate_cost
from core.models import TokenCost
from core.observability import setup_tracing
from core.repository_manager import RepositoryManager
from core.run_reporter import RunReporter
from core.sandbox_orchestrator import SandboxOrchestrator, SandboxUnavailableError
from core.pipeline_definition import GENERATE_PIPELINE
from core.state_machine import LifecycleEngine
from core.task_engine import TaskGraph
from core.tier_scheduler import TierScheduler
from core.workspace_indexer import index_workspace
from memory.dependency_graph import DependencyGraphStore
from memory.embedding_store import EmbeddingStore

if TYPE_CHECKING:
    from core.live_console import LiveConsole
    from core.pipeline import PipelineResult

logger = logging.getLogger(__name__)


class RunPipeline:
    """Executes the new-project generation workflow.

    Phases:
      1. Architecture — ArchitectAgent designs the blueprint
      2. Planning     — PlannerAgent builds the lifecycle task graph
      3. Execution    — AgentManager runs all tasks inside a sandbox
      4. Finalise     — index workspace, write run_report.json
    """

    def __init__(
        self,
        settings: Settings,
        llm: LLMClient,
        live: LiveConsole | None,
    ) -> None:
        self._settings = settings
        self._llm = llm
        self._live = live

    async def execute(self, user_prompt: str, start_time: float) -> PipelineResult:
        from core.pipeline import PipelineResult

        errors: list[str] = []

        # ── Observability ─────────────────────────────────────────────────────
        try:
            if self._settings.observability.enable_tracing:
                setup_tracing(self._settings.observability.otlp_endpoint)
        except Exception:
            logger.warning("Failed to initialise observability, continuing without it")

        # ── Phase 1: Architecture ─────────────────────────────────────────────
        self._phase("Architecture Design", "running")
        logger.info("[Phase 1] Designing architecture...")

        repo_manager = RepositoryManager(self._settings.workspace_dir)
        architect = ArchitectAgent(llm_client=self._llm, repo_manager=repo_manager)

        try:
            blueprint = await architect.design_architecture(user_prompt)
        except (LLMConfigError, Exception) as e:
            is_config = isinstance(e, LLMConfigError)
            if not is_config:
                logger.exception("Architecture design failed")
            else:
                logger.error(str(e))
            self._fail_phase("Architecture Design", str(e))
            return PipelineResult(
                success=False,
                workspace_path=self._settings.workspace_dir,
                errors=[f"Architecture design failed: {e}"],
                elapsed_seconds=time.monotonic() - start_time,
            )

        lang_profile = detect_language_from_blueprint(blueprint.tech_stack)
        logger.info(
            "Blueprint: %s | Language: %s | %d files",
            blueprint.name, lang_profile.display_name, len(blueprint.file_blueprints),
        )
        if self._live:
            self._live.set_blueprint(
                name=blueprint.name,
                language=lang_profile.display_name,
                files=len(blueprint.file_blueprints),
                style=blueprint.architecture_style,
            )
        self._complete_phase("Architecture Design")
        repo_manager.initialize(blueprint)

        # ── Phase 2: Planning ─────────────────────────────────────────────────
        self._phase("Task Planning", "running")
        logger.info("[Phase 2] Building lifecycle plan...")

        planner = PlannerAgent(llm_client=self._llm, repo_manager=repo_manager)
        try:
            # Compiled languages (Java, Go, Rust, C#, TypeScript) need more review-fix
            # cycles because initial code generation is more likely to contain syntax
            # errors that require multiple LLM passes to fully resolve.
            is_compiled = bool(lang_profile.build_command)
            review_fixes = 4 if is_compiled else 3
            lifecycle_engine, global_graph = await planner.create_lifecycle_plan(
                blueprint,
                max_review_fixes=review_fixes,
            )
        except (LLMConfigError, Exception) as e:
            is_config = isinstance(e, LLMConfigError)
            if not is_config:
                logger.exception("Task planning failed")
            else:
                logger.error(str(e))
            self._fail_phase("Task Planning", str(e))
            return PipelineResult(
                success=False,
                workspace_path=self._settings.workspace_dir,
                blueprint=blueprint,
                errors=[f"Task planning failed: {e}"],
                elapsed_seconds=time.monotonic() - start_time,
            )

        logger.info(
            "Lifecycle engine: %d files, global DAG: %d tasks",
            len(lifecycle_engine.get_stats()), len(global_graph.tasks),
        )
        self._complete_phase("Task Planning")
        if self._live:
            for task in global_graph.tasks.values():
                self._live.update_task(task.task_id, task.description, task.status.value)

        # ── Phase 3: Execution ────────────────────────────────────────────────
        self._phase("Code Generation & Review", "running")
        logger.info("[Phase 3] Executing tasks...")

        sandbox = SandboxOrchestrator(self._settings)
        try:
            sb = await sandbox.setup(lang_profile)
        except SandboxUnavailableError as e:
            self._fail_phase("Code Generation & Review", str(e))
            return PipelineResult(
                success=False,
                workspace_path=self._settings.workspace_dir,
                blueprint=blueprint,
                errors=[str(e)],
                elapsed_seconds=time.monotonic() - start_time,
            )

        run_dep_store = DependencyGraphStore(self._settings.workspace_dir)
        run_embedding_store = EmbeddingStore(
            persist_dir=self._settings.memory.chroma_persist_dir,
            embedding_model=self._settings.memory.embedding_model,
        )
        repo_manager._embedding_store = run_embedding_store

        event_bus = EventBus()
        agent_manager = AgentManager(
            settings=self._settings,
            llm_client=self._llm,
            repo_manager=repo_manager,
            blueprint=blueprint,
            live_console=self._live,
            sandbox_manager=sb.manager,
            build_sandbox_id=sb.build_id,
            test_sandbox_id=sb.test_id,
            dep_store=run_dep_store,
            embedding_store=run_embedding_store,
            event_bus=event_bus,
        )

        # Compute dependency tiers for incremental build verification.
        # Foundational files (models, interfaces) compile first; dependent
        # files (services, controllers) generate only after their deps pass.
        tier_scheduler = TierScheduler()
        known_paths = {fb.path for fb in blueprint.file_blueprints}
        file_deps = {
            fb.path: [d for d in fb.depends_on if d in known_paths]
            for fb in blueprint.file_blueprints
        }
        tiers = tier_scheduler.compute_tiers(
            file_paths=[fb.path for fb in blueprint.file_blueprints],
            file_deps=file_deps,
        )

        # Use checkpoint mode for compiled languages — build verification
        # happens at repo level between tiers instead of per-file.
        if is_compiled:
            lifecycle_engine.checkpoint_mode = True

        try:
            exec_result = await agent_manager.execute_with_checkpoints(
                lifecycle_engine,
                global_graph,
                tiers=tiers,
                pipeline_def=GENERATE_PIPELINE,
            )
        except Exception as e:
            logger.exception("Task execution failed")
            self._fail_phase("Code Generation & Review", str(e))
            return PipelineResult(
                success=False,
                workspace_path=self._settings.workspace_dir,
                blueprint=blueprint,
                errors=[f"Task execution failed: {e}"],
                elapsed_seconds=time.monotonic() - start_time,
            )
        finally:
            await sandbox.teardown()

        self._complete_phase("Code Generation & Review")

        # ── Phase 4: Finalise ─────────────────────────────────────────────────
        self._phase("Finalize", "running")
        logger.info("[Phase 4] Finalising repository...")

        try:
            index_workspace(repo_manager, self._settings)
        except Exception as e:
            logger.warning("Indexing failed (non-critical): %s", e)

        elapsed = time.monotonic() - start_time
        stats = exec_result.get("stats", {})

        # Code success: all files generated, reviewed, and built correctly.
        # Test failures (tests_degraded) are a quality signal, not a hard gate —
        # the generated code itself is valid even when generated tests don't all pass.
        code_success = (
            stats.get("failed", 0) == 0
            and stats.get("blocked", 0) == 0
            and stats.get("lifecycle_failed", 0) == 0
        )
        tests_passed = stats.get("lifecycle_tests_degraded", 0) == 0
        # Overall success requires code to generate correctly; test quality is
        # reported separately so a run with working code is never masked as
        # a total failure just because generated tests are imperfect.
        success = code_success

        self._complete_phase("Finalize")
        logger.info(
            "Pipeline %s | code_success=%s tests_passed=%s | stats=%s | elapsed=%.1fs",
            "SUCCEEDED" if success else "COMPLETED WITH ISSUES",
            code_success, tests_passed, stats, elapsed,
        )

        token_cost = self._build_token_cost()
        RunReporter(self._settings.workspace_dir).write_run_report(
            prompt=user_prompt,
            blueprint=blueprint,
            task_graph=global_graph,
            stats=stats,
            elapsed=elapsed,
            success=success,
            code_success=code_success,
            tests_passed=tests_passed,
            token_cost=token_cost,
        )

        return PipelineResult(
            success=success,
            workspace_path=self._settings.workspace_dir,
            blueprint=blueprint,
            task_stats=stats,
            metrics=exec_result.get("metrics", {}),
            errors=errors,
            elapsed_seconds=elapsed,
            token_cost=token_cost,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_token_cost(self) -> TokenCost:
        cost = calculate_cost(
            self._llm.config.model,
            self._llm.total_input_tokens,
            self._llm.total_output_tokens,
        )
        return TokenCost(
            input_tokens=self._llm.total_input_tokens,
            output_tokens=self._llm.total_output_tokens,
            model=self._llm.config.model,
            cost_usd=cost,
        )

    def _phase(self, name: str, status: str) -> None:
        if self._live:
            self._live.set_phase(name, status)

    def _complete_phase(self, name: str) -> None:
        if self._live:
            self._live.complete_phase(name)

    def _fail_phase(self, name: str, msg: str) -> None:
        if self._live:
            self._live.fail_phase(name, msg)
