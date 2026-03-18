"""Generation pipeline — new project from prompt to repository."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from agents.architect_agent import ArchitectAgent
from agents.planner_agent import PlannerAgent
from config.settings import Settings
from core.agent_manager import AgentManager
from core.event_bus import EventBus
from core.language import detect_language_from_blueprint
from core.llm_client import LLMClient, LLMConfigError, calculate_cost
from core.models import APIContract, TokenCost
from core.run_reporter import RunReporter
from core.sandbox_orchestrator import SandboxOrchestrator, SandboxUnavailableError
from core.pipeline_definition import GENERATE_PIPELINE
from core.tier_scheduler import TierScheduler
from core.workspace_indexer import index_workspace
from memory.dependency_graph import DependencyGraphStore
from memory.embedding_store import EmbeddingStore
from core.mcp_client import MCPClient
from core.observability import setup_tracing
from core.repository_manager import RepositoryManager

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
        root_write_lock: asyncio.Lock | None = None,
        api_contract: APIContract | None = None,
    ) -> None:
        self._settings = settings
        self._llm = llm
        self._live = live
        self._root_write_lock = root_write_lock
        self._api_contract = api_contract

    async def execute(self, user_prompt: str, start_time: float, *, resume: bool = False) -> PipelineResult:
        from core.pipeline import PipelineResult

        # ── Resume fast path ──────────────────────────────────────────────────
        if resume:
            return await self._execute_resume(user_prompt, start_time)

        errors: list[str] = []

        # ── Observability ─────────────────────────────────────────────────────
        try:
            if self._settings.observability.enable_tracing:
                setup_tracing(self._settings.observability.otlp_endpoint)
        except Exception:
            logger.warning("Failed to initialise observability, continuing without it")

        mcp_client: MCPClient | None = None
        if self._settings.mcp_server_command:
            try:
                mcp_client = MCPClient(self._settings.mcp_server_command)
                await mcp_client.initialize()
            except Exception as e:
                logger.error("Failed to initialize MCP client in run pipeline: %s", e)
                mcp_client = None

        # ── Phase 1: Architecture ─────────────────────────────────────────────
        self._phase("Architecture Design", "running")
        logger.info("[Phase 1] Designing architecture...")

        repo_manager = RepositoryManager(self._settings.workspace_dir)
        if self._root_write_lock is not None:
            repo_manager._root_write_lock = self._root_write_lock
        architect = ArchitectAgent(
            llm_client=self._llm,
            repo_manager=repo_manager,
            mcp_client=mcp_client,
        )

        # Timeout the architecture LLM call so the pipeline never hangs
        # indefinitely if the API endpoint stalls.
        _arch_timeout = self._settings.phase_timeout_seconds
        try:
            blueprint = await asyncio.wait_for(
                architect.design_architecture(user_prompt),
                timeout=_arch_timeout,
            )
        except asyncio.TimeoutError:
            msg = (
                f"Architecture design timed out after {_arch_timeout}s. "
                "The LLM provider may be overloaded — try again later."
            )
            logger.error(msg)
            self._fail_phase("Architecture Design", msg)
            return PipelineResult(
                success=False,
                workspace_path=self._settings.workspace_dir,
                errors=[msg],
                elapsed_seconds=time.monotonic() - start_time,
            )
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
        try:
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

            planner = PlannerAgent(llm_client=self._llm, repo_manager=repo_manager, mcp_client=mcp_client)
            try:
                is_compiled = bool(lang_profile.build_command)
                review_fixes = 2  # 2 review-fix cycles per file; pipeline only fails on build
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

            lc_stats = lifecycle_engine.get_stats()
            lc_file_count = sum(lc_stats.values())
            logger.info(
                "Lifecycle engine: %d files (%s), global DAG: %d tasks",
                lc_file_count,
                ", ".join(f"{v} {k}" for k, v in lc_stats.items()),
                len(global_graph.tasks),
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

            # Pre-warm the embedding model in a background thread.
            # Loading SentenceTransformer makes HuggingFace network HEAD requests
            # on first use; doing it now avoids blocking the event loop during the
            # first file generation context build.
            asyncio.ensure_future(asyncio.to_thread(run_embedding_store._ensure_client))

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
                mcp_client=mcp_client,
                api_contract=self._api_contract,
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
                from core.pipeline_executor import PipelineExecutor
                
                executor = PipelineExecutor(
                    agent_manager=agent_manager,
                    settings=self._settings,
                    lang_profile=lang_profile,
                    event_bus=event_bus,
                )
                
                exec_result = await executor.execute(
                    lifecycle_engine,
                    global_graph,
                    tiers=tiers,
                    pipeline_def=GENERATE_PIPELINE.with_retries(
                        self._settings.build_checkpoint_retries
                    ),
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
            # Also check checkpoint results — if any build checkpoint failed, the
            # code is not reliable even if lifecycle states look OK.
            checkpoints_passed = all(
                cr.get("passed", True)
                for cr in exec_result.get("checkpoint_results", [])
            )
            code_success = (
                stats.get("failed", 0) == 0
                and stats.get("blocked", 0) == 0
                and stats.get("lifecycle_failed", 0) == 0
                and checkpoints_passed
            )
            tests_passed = stats.get("lifecycle_tests_degraded", 0) == 0

            # Security and integration checkpoints are soft gates — failures
            # result in DEGRADED quality, not pipeline failure.
            sec_ck = exec_result.get("security_checkpoint", {})
            int_ck = exec_result.get("integration_checkpoint", {})
            security_passed = sec_ck.get("passed", True)
            integration_passed = int_ck.get("passed", True)
            quality_passed = security_passed and integration_passed

            # Overall success requires code to generate correctly; security and
            # integration are quality signals reported separately.
            success = code_success

            self._complete_phase("Finalize")
            logger.info(
                "Pipeline %s | code_success=%s tests_passed=%s "
                "security_passed=%s integration_passed=%s | stats=%s | elapsed=%.1fs",
                "SUCCEEDED" if success else "COMPLETED WITH ISSUES",
                code_success,
                tests_passed,
                security_passed,
                integration_passed,
                stats,
                elapsed,
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
        except Exception as flow_err:
            logger.error("Unhandled error in run pipeline: %e", flow_err)
            raise
        finally:
            # Ensure sandbox and MCP client are cleaned up even on failure.
            try:
                if "sandbox" in locals():
                    await sandbox.teardown()
            except Exception:
                logger.warning("Error during sandbox teardown", exc_info=True)
            if mcp_client:
                await mcp_client.close()

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

    async def _execute_resume(self, user_prompt: str, start_time: float) -> "PipelineResult":
        """Resume a previous run from the last checkpoint.

        Loads the saved lifecycle state and blueprint, skips Architecture +
        Planning, and re-enters the Execution phase.  Files that already
        PASSED are kept; FAILED / intermediate files are reset to PENDING.
        """
        from core.pipeline import PipelineResult
        from core.state_machine import LifecycleEngine
        from core.models import RepositoryBlueprint, FileBlueprint
        import json as _json

        workspace = self._settings.workspace_dir
        state_path = workspace / ".pipeline_state.json"
        blueprint_path = workspace / "file_blueprints.json"

        # ── Validate resume prerequisites ─────────────────────────────────────
        if not state_path.exists():
            logger.error("No .pipeline_state.json found in %s — cannot resume", workspace)
            return PipelineResult(
                success=False,
                workspace_path=workspace,
                errors=["Cannot resume: no .pipeline_state.json found. Run without --resume first."],
                elapsed_seconds=time.monotonic() - start_time,
            )

        if not blueprint_path.exists():
            logger.error("No file_blueprints.json found — cannot resume")
            return PipelineResult(
                success=False,
                workspace_path=workspace,
                errors=["Cannot resume: no file_blueprints.json found."],
                elapsed_seconds=time.monotonic() - start_time,
            )

        # ── Load blueprint ────────────────────────────────────────────────────
        self._phase("Resume: Loading State", "running")
        logger.info("[Resume] Loading lifecycle state from %s", state_path)

        repo_manager = RepositoryManager(workspace)
        if self._root_write_lock is not None:
            repo_manager._root_write_lock = self._root_write_lock

        try:
            bp_raw = _json.loads(blueprint_path.read_text(encoding="utf-8"))
            arch_path = workspace / "architecture.md"
            arch_doc = arch_path.read_text(encoding="utf-8") if arch_path.exists() else ""
            blueprint = RepositoryBlueprint(
                name=bp_raw.get("name", "resumed"),
                description=bp_raw.get("description", ""),
                architecture_style=bp_raw.get("architecture_style", ""),
                tech_stack=bp_raw.get("tech_stack", {}),
                folder_structure=bp_raw.get("folder_structure", []),
                file_blueprints=[
                    FileBlueprint(
                        path=f["path"],
                        purpose=f.get("purpose", ""),
                        depends_on=f.get("depends_on", []),
                        exports=f.get("exports", []),
                        language=f.get("language", ""),
                        layer=f.get("layer", ""),
                    )
                    for f in bp_raw.get("files", [])
                ],
                architecture_doc=arch_doc,
            )
        except Exception as e:
            logger.error("Failed to load blueprint: %s", e)
            return PipelineResult(
                success=False,
                workspace_path=workspace,
                errors=[f"Resume failed: could not load blueprint: {e}"],
                elapsed_seconds=time.monotonic() - start_time,
            )

        lang_profile = detect_language_from_blueprint(blueprint.tech_stack)

        # ── Load lifecycle state ──────────────────────────────────────────────
        try:
            lifecycle_engine = LifecycleEngine.load_state(str(state_path))
        except Exception as e:
            logger.error("Failed to load lifecycle state: %s", e)
            return PipelineResult(
                success=False,
                workspace_path=workspace,
                errors=[f"Resume failed: could not load pipeline state: {e}"],
                elapsed_seconds=time.monotonic() - start_time,
            )

        summary = lifecycle_engine.get_results_summary()
        logger.info(
            "[Resume] Loaded state: %d total, %d passed, %d failed, %d to retry",
            summary["total_files"], summary["passed"], summary["failed"],
            summary["total_files"] - summary["passed"] - summary.get("degraded", 0),
        )

        if summary["passed"] + summary.get("degraded", 0) == summary["total_files"]:
            logger.info("[Resume] All files already passed — nothing to do")
            self._complete_phase("Resume: Loading State")
            return PipelineResult(
                success=True,
                workspace_path=workspace,
                blueprint=blueprint,
                elapsed_seconds=time.monotonic() - start_time,
            )

        self._complete_phase("Resume: Loading State")

        # ── Setup sandbox + agent manager ─────────────────────────────────────
        self._phase("Code Generation & Review (Resume)", "running")

        mcp_client: MCPClient | None = None
        if self._settings.mcp_server_command:
            try:
                mcp_client = MCPClient(self._settings.mcp_server_command)
                await mcp_client.initialize()
            except Exception as e:
                logger.error("Failed to initialize MCP client: %s", e)
                mcp_client = None

        repo_manager.initialize(blueprint)

        sandbox = SandboxOrchestrator(self._settings)
        try:
            sb = await sandbox.setup(lang_profile)
        except SandboxUnavailableError as e:
            self._fail_phase("Code Generation & Review (Resume)", str(e))
            return PipelineResult(
                success=False,
                workspace_path=workspace,
                blueprint=blueprint,
                errors=[str(e)],
                elapsed_seconds=time.monotonic() - start_time,
            )

        try:
            run_dep_store = DependencyGraphStore(workspace)
            run_embedding_store = EmbeddingStore(
                persist_dir=self._settings.memory.chroma_persist_dir,
                embedding_model=self._settings.memory.embedding_model,
            )
            repo_manager._embedding_store = run_embedding_store
            asyncio.ensure_future(asyncio.to_thread(run_embedding_store._ensure_client))

            event_bus = EventBus()
            from core.task_engine import TaskGraph
            # Build a minimal global graph (advisory tasks still need to run)
            planner = PlannerAgent(llm_client=self._llm, repo_manager=repo_manager, mcp_client=mcp_client)
            try:
                _, global_graph = await planner.create_lifecycle_plan(blueprint, max_review_fixes=2)
            except Exception:
                logger.warning("[Resume] Could not re-create global graph — using empty graph")
                global_graph = TaskGraph()

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
                mcp_client=mcp_client,
                api_contract=self._api_contract,
            )

            # Recompute tiers
            is_compiled = bool(lang_profile.build_command)
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

            if is_compiled:
                lifecycle_engine.checkpoint_mode = True

            errors: list[str] = []
            try:
                from core.pipeline_executor import PipelineExecutor

                executor = PipelineExecutor(
                    agent_manager=agent_manager,
                    settings=self._settings,
                    lang_profile=lang_profile,
                    event_bus=event_bus,
                )

                exec_result = await executor.execute(
                    lifecycle_engine,
                    global_graph,
                    tiers=tiers,
                    pipeline_def=GENERATE_PIPELINE.with_retries(
                        self._settings.build_checkpoint_retries,
                    ),
                )
            except Exception as e:
                logger.exception("[Resume] Task execution failed")
                self._fail_phase("Code Generation & Review (Resume)", str(e))
                return PipelineResult(
                    success=False,
                    workspace_path=workspace,
                    blueprint=blueprint,
                    errors=[f"Resume execution failed: {e}"],
                    elapsed_seconds=time.monotonic() - start_time,
                )

            self._complete_phase("Code Generation & Review (Resume)")

            # ── Finalise ──────────────────────────────────────────────────────
            try:
                index_workspace(repo_manager, self._settings)
            except Exception as e:
                logger.warning("Indexing failed (non-critical): %s", e)

            elapsed = time.monotonic() - start_time
            stats = exec_result.get("stats", {})

            checkpoints_passed = all(
                cr.get("passed", True)
                for cr in exec_result.get("checkpoint_results", [])
            )
            code_success = (
                stats.get("failed", 0) == 0
                and stats.get("blocked", 0) == 0
                and stats.get("lifecycle_failed", 0) == 0
                and checkpoints_passed
            )
            success = code_success

            logger.info(
                "[Resume] Pipeline %s | stats=%s | elapsed=%.1fs",
                "SUCCEEDED" if success else "COMPLETED WITH ISSUES",
                stats, elapsed,
            )

            token_cost = self._build_token_cost()
            RunReporter(workspace).write_run_report(
                prompt=user_prompt,
                blueprint=blueprint,
                task_graph=global_graph,
                stats=stats,
                elapsed=elapsed,
                success=success,
                code_success=code_success,
                tests_passed=stats.get("lifecycle_tests_degraded", 0) == 0,
                token_cost=token_cost,
            )

            return PipelineResult(
                success=success,
                workspace_path=workspace,
                blueprint=blueprint,
                task_stats=stats,
                metrics=exec_result.get("metrics", {}),
                errors=errors,
                elapsed_seconds=elapsed,
                token_cost=token_cost,
            )
        except Exception as flow_err:
            logger.error("Unhandled error in resume pipeline: %s", flow_err)
            raise
        finally:
            try:
                await sandbox.teardown()
            except Exception:
                logger.warning("Error during sandbox teardown", exc_info=True)
            if mcp_client:
                await mcp_client.close()
