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
from agents.repository_analyzer_agent import RepositoryAnalyzerAgent
from agents.change_planner_agent import ChangePlannerAgent
from config.settings import Settings
from core.agent_manager import AgentManager
from core.language import detect_language_from_blueprint
from core.live_console import LiveConsole, LiveConsoleHandler
from core.llm_client import LLMClient, LLMConfigError, calculate_cost
from core.models import RepositoryBlueprint, ChangePlan, RepoAnalysis, TokenCost
from core.plan_approver import PlanApprover, PlanPendingApprovalError
from core.observability import record_task_completion, start_metrics_server, setup_tracing
from core.repository_manager import RepositoryManager
from core.run_reporter import RunReporter
from core.sandbox_orchestrator import SandboxOrchestrator, SandboxUnavailableError
from core.state_machine import LifecycleEngine
from core.task_engine import TaskGraph, ModificationTaskGraphBuilder
from core.workspace_snapshot import WorkspaceSnapshot
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
    change_plan: ChangePlan | None = None
    repo_analysis: RepoAnalysis | None = None
    task_stats: dict[str, int] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    token_cost: TokenCost | None = None


class Pipeline:
    """End-to-end pipeline: prompt -> architecture -> tasks -> agents -> repository."""

    def __init__(self, settings: Settings | None = None, interactive: bool = True) -> None:
        from config.settings import get_settings
        self.settings = settings or get_settings()
        self.interactive = interactive
        self._live: LiveConsole | None = None
        try:
            self.llm = LLMClient(self.settings.llm)
        except LLMConfigError as e:
            # Re-raise with full context - will be caught in run()
            raise

    def _start_live(self) -> None:
        if not self.interactive:
            return
        self._live = LiveConsole()
        # Route all logging to the live console
        handler = LiveConsoleHandler(self._live)
        handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(handler)
        self._live.start()

    def _stop_live(self) -> None:
        if self._live:
            self._live.stop()
            self._live = None

    async def run(self, user_prompt: str) -> PipelineResult:
        """Execute the full generation pipeline."""
        start_time = time.monotonic()
        errors: list[str] = []
        self._file_handler: logging.FileHandler | None = None

        self._start_live()

        try:
            return await self._run_inner(user_prompt, start_time, errors)
        finally:
            self._stop_live()
            # Ensure file handler is closed even on early exit / exception
            if self._file_handler:
                self._stop_file_logging(self._file_handler)
                self._file_handler = None

    def _start_file_logging(self, workspace: Path) -> logging.FileHandler:
        """Attach a file handler so all logs are persisted to workspace/run.log."""
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

    async def _run_inner(
        self, user_prompt: str, start_time: float, errors: list[str]
    ) -> PipelineResult:
        file_handler = self._start_file_logging(self.settings.workspace_dir)
        self._file_handler = file_handler
        logger.info("Starting code generation pipeline")
        logger.info(f"Prompt: {user_prompt[:100]}...")

        # Initialize observability (non-blocking)
        try:
            if self.settings.observability.enable_tracing:
                setup_tracing(self.settings.observability.otlp_endpoint)
        except Exception:
            logger.warning("Failed to initialize observability, continuing without it")

        # ── Phase 1: Architecture ─────────────────────────────────────────
        if self._live:
            self._live.set_phase("Architecture Design", "running")
        logger.info("[Phase 1] Designing architecture...")
        repo_manager = RepositoryManager(self.settings.workspace_dir)
        architect = ArchitectAgent(llm_client=self.llm, repo_manager=repo_manager)

        try:
            blueprint = await architect.design_architecture(user_prompt)
        except LLMConfigError as e:
            logger.error(str(e))
            if self._live:
                self._live.fail_phase("Architecture Design", str(e))
            return PipelineResult(
                success=False,
                workspace_path=self.settings.workspace_dir,
                errors=[str(e)],
                elapsed_seconds=time.monotonic() - start_time,
            )
        except Exception as e:
            logger.exception("Architecture design failed")
            if self._live:
                self._live.fail_phase("Architecture Design", str(e))
            return PipelineResult(
                success=False,
                workspace_path=self.settings.workspace_dir,
                errors=[f"Architecture design failed: {e}"],
                elapsed_seconds=time.monotonic() - start_time,
            )

        # Detect language from the generated blueprint
        lang_profile = detect_language_from_blueprint(blueprint.tech_stack)
        logger.info(
            f"Blueprint: {blueprint.name} | Language: {lang_profile.display_name} | "
            f"{len(blueprint.file_blueprints)} files"
        )

        if self._live:
            self._live.set_blueprint(
                name=blueprint.name,
                language=lang_profile.display_name,
                files=len(blueprint.file_blueprints),
                style=blueprint.architecture_style,
            )
            self._live.complete_phase("Architecture Design")

        # Initialize workspace
        repo_manager.initialize(blueprint)

        # ── Phase 2: Planning ─────────────────────────────────────────────
        if self._live:
            self._live.set_phase("Task Planning", "running")

        logger.info("[Phase 2] Building lifecycle plan...")
        planner = PlannerAgent(llm_client=self.llm, repo_manager=repo_manager)

        lifecycle_engine: LifecycleEngine | None = None
        task_graph: TaskGraph | None = None
        global_graph: TaskGraph | None = None

        try:
            lifecycle_engine, global_graph = await planner.create_lifecycle_plan(blueprint)
            task_graph = global_graph  # used for reporting / live display
        except LLMConfigError as e:
            logger.error(str(e))
            if self._live:
                self._live.fail_phase("Task Planning", str(e))
            return PipelineResult(
                success=False,
                workspace_path=self.settings.workspace_dir,
                blueprint=blueprint,
                errors=[str(e)],
                elapsed_seconds=time.monotonic() - start_time,
            )
        except Exception as e:
            logger.exception("Task planning failed")
            if self._live:
                self._live.fail_phase("Task Planning", str(e))
            return PipelineResult(
                success=False,
                workspace_path=self.settings.workspace_dir,
                blueprint=blueprint,
                errors=[f"Task planning failed: {e}"],
                elapsed_seconds=time.monotonic() - start_time,
            )

        assert task_graph is not None
        logger.info(f"Task graph: {len(task_graph.tasks)} tasks")
        logger.info(
            "Lifecycle engine: %d files, global DAG: %d tasks",
            len(lifecycle_engine.get_stats()), len(task_graph.tasks),
        )

        if self._live:
            self._live.complete_phase("Task Planning")
            # Register all tasks in the live display
            for task in task_graph.tasks.values():
                self._live.update_task(task.task_id, task.description, task.status.value)

        # ── Phase 3: Execution ────────────────────────────────────────────
        if self._live:
            self._live.set_phase("Code Generation & Review", "running")
        logger.info("[Phase 3] Executing tasks...")

        # Spin up sandbox(es) when Docker mode is requested.
        # Two-tier model: BUILD sandbox (network for deps) + TEST sandbox
        # (no network, read-only rootfs — prevents data exfiltration by
        # LLM-generated test code).
        from config.settings import SandboxTier, SandboxType
        sandbox_manager: SandboxManager | None = None
        build_sandbox_id: str | None = None
        test_sandbox_id: str | None = None

        if self.settings.sandbox.sandbox_type == SandboxType.DOCKER:
            try:
                sandbox_manager = SandboxManager(self.settings.sandbox)
                build_info = await sandbox_manager.create_sandbox(
                    self.settings.workspace_dir,
                    language_name=lang_profile.name,
                    tier=SandboxTier.BUILD,
                )
                build_sandbox_id = build_info.sandbox_id
                logger.info("Docker BUILD sandbox created: %s", build_sandbox_id)

                test_info = await sandbox_manager.create_sandbox(
                    self.settings.workspace_dir,
                    language_name=lang_profile.name,
                    tier=SandboxTier.TEST,
                )
                test_sandbox_id = test_info.sandbox_id
                logger.info("Docker TEST sandbox created: %s (network=none)", test_sandbox_id)
            except Exception as e:
                if self.settings.allow_host_execution:
                    logger.warning(
                        "Docker sandbox unavailable (%s), falling back to host "
                        "execution (--allow-host-execution is set)", e,
                    )
                    sandbox_manager = None
                else:
                    msg = (
                        f"Docker sandbox unavailable: {e}.  "
                        "Either install/start Docker or pass --allow-host-execution "
                        "to run without isolation (NOT recommended for untrusted prompts)."
                    )
                    logger.error(msg)
                    if self._live:
                        self._live.fail_phase("Code Generation & Review", msg)
                    return PipelineResult(
                        success=False,
                        workspace_path=self.settings.workspace_dir,
                        blueprint=blueprint,
                        errors=[msg],
                        elapsed_seconds=time.monotonic() - start_time,
                    )
        else:
            # Explicit --sandbox local
            logger.warning(
                "Running in LOCAL mode with NO sandbox isolation.  Generated "
                "code will execute directly on the host.  Use Docker sandbox "
                "for production workloads."
            )

        # Memory stores for fresh generation start empty but are populated
        # incrementally as files are generated and indexed.  Even within a
        # single run the dep_store lets later phases (module review, arch
        # review) know transitive impact without a full filesystem rescan.
        run_dep_store = DependencyGraphStore(self.settings.workspace_dir)
        run_embedding_store = EmbeddingStore(
            persist_dir=self.settings.memory.chroma_persist_dir,
            embedding_model=self.settings.memory.embedding_model,
        )
        # Wire incremental embedding into repo_manager so every write_file()
        # call indexes the new content immediately instead of only at finalization.
        repo_manager._embedding_store = run_embedding_store

        agent_manager = AgentManager(
            settings=self.settings,
            llm_client=self.llm,
            repo_manager=repo_manager,
            blueprint=blueprint,
            live_console=self._live,
            sandbox_manager=sandbox_manager,
            build_sandbox_id=build_sandbox_id,
            test_sandbox_id=test_sandbox_id,
            dep_store=run_dep_store,
            embedding_store=run_embedding_store,
        )

        try:
            exec_result = await agent_manager.execute_with_lifecycle(
                lifecycle_engine, global_graph,
            )
        except Exception as e:
            logger.exception("Task execution failed")
            if self._live:
                self._live.fail_phase("Code Generation & Review", str(e))
            return PipelineResult(
                success=False,
                workspace_path=self.settings.workspace_dir,
                blueprint=blueprint,
                errors=[f"Task execution failed: {e}"],
                elapsed_seconds=time.monotonic() - start_time,
            )
        finally:
            # Always destroy the sandbox to free resources
            if sandbox_manager:
                try:
                    await sandbox_manager.destroy_all()
                    logger.info("Sandbox destroyed")
                except Exception:
                    logger.warning("Sandbox teardown failed (non-critical)")

        if self._live:
            self._live.complete_phase("Code Generation & Review")

        # ── Phase 4: Finalize ─────────────────────────────────────────────
        if self._live:
            self._live.set_phase("Finalize", "running")
        logger.info("[Phase 4] Finalizing repository...")

        # Index all generated files into memory stores
        try:
            self._index_workspace(repo_manager)
        except Exception as e:
            logger.warning(f"Indexing failed (non-critical): {e}")

        elapsed = time.monotonic() - start_time
        stats = exec_result.get("stats", {})
        success = (
            stats.get("failed", 0) == 0
            and stats.get("blocked", 0) == 0
            and stats.get("lifecycle_failed", 0) == 0
        )

        if self._live:
            self._live.complete_phase("Finalize")

        logger.info(f"Pipeline {'SUCCEEDED' if success else 'COMPLETED WITH ISSUES'}")
        logger.info(f"Stats: {stats}")
        logger.info(f"Elapsed: {elapsed:.1f}s | Workspace: {self.settings.workspace_dir}")

        # Build cost summary and write report
        token_cost = self._build_token_cost()
        RunReporter(self.settings.workspace_dir).write_run_report(
            prompt=user_prompt,
            blueprint=blueprint,
            task_graph=task_graph,
            stats=stats,
            elapsed=elapsed,
            success=success,
            token_cost=token_cost,
        )

        self._stop_file_logging(file_handler)
        self._file_handler = None

        return PipelineResult(
            success=success,
            workspace_path=self.settings.workspace_dir,
            blueprint=blueprint,
            task_stats=stats,
            metrics=exec_result.get("metrics", {}),
            errors=errors,
            elapsed_seconds=elapsed,
            token_cost=token_cost,
        )

    def _build_token_cost(self) -> TokenCost:
        """Build a TokenCost from the LLM client's cumulative counters."""
        cost = calculate_cost(
            self.llm.config.model,
            self.llm.total_input_tokens,
            self.llm.total_output_tokens,
        )
        return TokenCost(
            input_tokens=self.llm.total_input_tokens,
            output_tokens=self.llm.total_output_tokens,
            model=self.llm.config.model,
            cost_usd=cost,
        )

    def _index_workspace(self, repo_manager: RepositoryManager) -> None:
        """Index all generated files into memory stores."""
        index_store = RepoIndexStore(self.settings.workspace_dir)
        dep_store = DependencyGraphStore(self.settings.workspace_dir)
        embedding_store = EmbeddingStore(
            persist_dir=self.settings.memory.chroma_persist_dir,
            embedding_model=self.settings.memory.embedding_model,
        )

        repo_index = repo_manager.get_repo_index()
        known_files = {f.path for f in repo_index.files}
        lang = repo_manager._lang_profile

        for file_info in repo_index.files:
            index_store.update_file(file_info)

            # Update dependency graph — resolve raw import strings to file paths
            # so that graph queries (impact analysis, ordering) work correctly.
            for imp in file_info.imports:
                resolved = lang.resolve_import_to_path(imp, known_files)
                if resolved:
                    dep_store.add_dependency(file_info.path, resolved)
                else:
                    logger.debug("Unresolved import '%s' in %s", imp, file_info.path)

            # Index embeddings
            content = repo_manager.read_file(file_info.path)
            if content:
                embedding_store.index_file(file_info.path, content)

        dep_store.save()
        index_store.save()

    # ── Modification pipeline ─────────────────────────────────────────

    async def enhance(self, user_prompt: str) -> PipelineResult:
        """Modify an existing repository based on a user request.

        Unlike ``run()`` which creates a new project from scratch, this
        method follows the modification workflow:

        1. Repository Analysis — scan existing files, build index & deps
        2. Change Planning — LLM plans targeted modifications
        3. Task DAG — build modification task graph from the plan
        4. Execution — agents perform targeted edits, reviews, tests
        5. Finalization — re-index and report
        """
        start_time = time.monotonic()
        errors: list[str] = []
        self._file_handler: logging.FileHandler | None = None

        self._start_live()

        try:
            return await self._enhance_inner(user_prompt, start_time, errors)
        finally:
            self._stop_live()
            if self._file_handler:
                self._stop_file_logging(self._file_handler)
                self._file_handler = None

    async def _enhance_inner(
        self, user_prompt: str, start_time: float, errors: list[str],
    ) -> PipelineResult:
        file_handler = self._start_file_logging(self.settings.workspace_dir)
        self._file_handler = file_handler
        logger.info("Starting repository modification pipeline")
        logger.info(f"Prompt: {user_prompt[:100]}...")

        repo_manager = RepositoryManager(self.settings.workspace_dir)

        # ── Phase 1: Repository Analysis ──────────────────────────────
        if self._live:
            self._live.set_phase("Repository Analysis", "running")
        logger.info("[Phase 1] Analyzing existing repository...")

        analyzer = RepositoryAnalyzerAgent(
            llm_client=self.llm, repo_manager=repo_manager,
        )

        try:
            repo_analysis = await analyzer.analyze_repository(self.settings.workspace_dir)
        except Exception as e:
            logger.exception("Repository analysis failed")
            if self._live:
                self._live.fail_phase("Repository Analysis", str(e))
            return PipelineResult(
                success=False,
                workspace_path=self.settings.workspace_dir,
                errors=[f"Repository analysis failed: {e}"],
                elapsed_seconds=time.monotonic() - start_time,
            )

        lang_profile = detect_language_from_blueprint(repo_analysis.tech_stack)
        logger.info(
            "Analysis: %d modules | Language: %s | Arch: %s",
            len(repo_analysis.modules), lang_profile.display_name,
            repo_analysis.architecture_style,
        )

        if self._live:
            self._live.set_blueprint(
                name=f"(modify) {self.settings.workspace_dir.name}",
                language=lang_profile.display_name,
                files=len(repo_analysis.modules),
                style=repo_analysis.architecture_style,
            )
            self._live.complete_phase("Repository Analysis")

        # Scan existing files into the repo index
        repo_manager.scan_existing_repo()

        # ── Load persisted memory stores ──────────────────────────────
        # Stores are loaded immediately after scanning so the dependency graph
        # and vector index built from previous runs (or the initial generation
        # run) are available to the change planner, task builder, and all agents.
        dep_store = DependencyGraphStore(self.settings.workspace_dir)
        embedding_store = EmbeddingStore(
            persist_dir=self.settings.memory.chroma_persist_dir,
            embedding_model=self.settings.memory.embedding_model,
        )
        # Wire incremental embedding into repo_manager so every write_file()
        # call during modification indexes the new content immediately.
        repo_manager._embedding_store = embedding_store

        # Sync the in-memory graph from the freshly scanned repo index so the
        # graph always reflects the current state on disk, not just prior runs.
        repo_index = repo_manager.get_repo_index()
        known_files = {f.path for f in repo_index.files}
        for file_info in repo_index.files:
            for imp in file_info.imports:
                resolved = lang_profile.resolve_import_to_path(imp, known_files)
                if resolved:
                    dep_store.add_dependency(file_info.path, resolved)
                else:
                    logger.debug("Unresolved import '%s' in %s", imp, file_info.path)
        dep_store.save()
        logger.info(
            "Memory stores loaded: dep_graph=%d nodes, chroma_dir=%s",
            len(dep_store.get_graph().nodes()),
            self.settings.memory.chroma_persist_dir,
        )

        # Build a minimal blueprint from the analysis (for context builder compat)
        from core.models import FileBlueprint
        blueprint = RepositoryBlueprint(
            name=self.settings.workspace_dir.name,
            description=repo_analysis.summary,
            architecture_style=repo_analysis.architecture_style,
            tech_stack=repo_analysis.tech_stack,
            file_blueprints=[
                FileBlueprint(
                    path=m.file,
                    purpose=f"Module: {m.name}",
                    exports=m.functions + m.classes,
                    language=repo_analysis.tech_stack.get("language", "python"),
                    layer=m.layer,
                )
                for m in repo_analysis.modules
            ],
        )

        # ── Phase 2: Change Planning ─────────────────────────────────
        if self._live:
            self._live.set_phase("Change Planning", "running")
        logger.info("[Phase 2] Planning targeted changes...")

        planner = ChangePlannerAgent(
            llm_client=self.llm, repo_manager=repo_manager,
        )

        # Provide key file contents to the planner for precision
        file_contents = repo_manager.read_all_source_files()

        try:
            change_plan = await planner.plan_changes(
                user_request=user_prompt,
                repo_analysis=repo_analysis,
                file_contents=file_contents,
            )
        except Exception as e:
            logger.exception("Change planning failed")
            if self._live:
                self._live.fail_phase("Change Planning", str(e))
            return PipelineResult(
                success=False,
                workspace_path=self.settings.workspace_dir,
                repo_analysis=repo_analysis,
                errors=[f"Change planning failed: {e}"],
                elapsed_seconds=time.monotonic() - start_time,
            )

        logger.info(
            "Change plan: %s | %d modifications, %d new files",
            change_plan.summary, len(change_plan.changes), len(change_plan.new_files),
        )
        for c in change_plan.changes:
            logger.info("  → %s %s: %s", c.type.value, c.file, c.description)
        for nf in change_plan.new_files:
            logger.info("  + NEW %s: %s", nf.path, nf.purpose)

        if self._live:
            self._live.complete_phase("Change Planning")

        # ── Approval gate (optional) ───────────────────────────────────
        if self.settings.require_plan_approval:
            approver = PlanApprover(
                interactive=self.interactive,
                workspace=self.settings.workspace_dir,
            )
            try:
                approved = approver.display_and_approve(change_plan)
            except PlanPendingApprovalError as e:
                logger.info("Plan pending human approval: %s", e)
                return PipelineResult(
                    success=False,
                    workspace_path=self.settings.workspace_dir,
                    repo_analysis=repo_analysis,
                    errors=[str(e)],
                    elapsed_seconds=time.monotonic() - start_time,
                )
            if not approved:
                logger.info("Change plan rejected by user")
                return PipelineResult(
                    success=False,
                    workspace_path=self.settings.workspace_dir,
                    repo_analysis=repo_analysis,
                    errors=["Change plan rejected by user"],
                    elapsed_seconds=time.monotonic() - start_time,
                )

        # ── Phase 3: Build modification task DAG ──────────────────────
        if self._live:
            self._live.set_phase("Task Planning", "running")
        logger.info("[Phase 3] Building modification task graph...")

        mod_builder = ModificationTaskGraphBuilder(dep_store=dep_store)
        task_graph = mod_builder.build_from_change_plan(change_plan, blueprint)

        logger.info("Modification task graph: %d tasks", len(task_graph.tasks))

        if self._live:
            self._live.complete_phase("Task Planning")
            for task in task_graph.tasks.values():
                self._live.update_task(task.task_id, task.description, task.status.value)

        # ── Phase 4: Execute modifications ────────────────────────────
        if self._live:
            self._live.set_phase("Code Modification & Review", "running")
        logger.info("[Phase 4] Executing targeted modifications...")

        # No sandbox needed for modification — edits are targeted, not full builds
        # (Sandboxes are still used for test execution if Docker is available)
        from config.settings import SandboxTier, SandboxType
        sandbox_manager = None
        build_sandbox_id = None
        test_sandbox_id = None

        if self.settings.sandbox.sandbox_type == SandboxType.DOCKER:
            try:
                sandbox_manager = SandboxManager(self.settings.sandbox)
                build_info = await sandbox_manager.create_sandbox(
                    self.settings.workspace_dir,
                    language_name=lang_profile.name,
                    tier=SandboxTier.BUILD,
                )
                build_sandbox_id = build_info.sandbox_id
                test_info = await sandbox_manager.create_sandbox(
                    self.settings.workspace_dir,
                    language_name=lang_profile.name,
                    tier=SandboxTier.TEST,
                )
                test_sandbox_id = test_info.sandbox_id
            except Exception as e:
                if self.settings.allow_host_execution:
                    logger.warning("Docker sandbox unavailable (%s), using host execution", e)
                    sandbox_manager = None
                else:
                    msg = f"Docker sandbox unavailable: {e}. Pass --allow-host-execution to run without isolation."
                    logger.error(msg)
                    if self._live:
                        self._live.fail_phase("Code Modification & Review", msg)
                    return PipelineResult(
                        success=False,
                        workspace_path=self.settings.workspace_dir,
                        change_plan=change_plan,
                        repo_analysis=repo_analysis,
                        errors=[msg],
                        elapsed_seconds=time.monotonic() - start_time,
                    )

        agent_manager = AgentManager(
            settings=self.settings,
            llm_client=self.llm,
            repo_manager=repo_manager,
            blueprint=blueprint,
            live_console=self._live,
            sandbox_manager=sandbox_manager,
            build_sandbox_id=build_sandbox_id,
            test_sandbox_id=test_sandbox_id,
            dep_store=dep_store,
            embedding_store=embedding_store,
        )

        # ── Snapshot → execute → commit (or auto-restore on failure) ──────
        changed_files: list[str] = []
        diff_stats: dict[str, int] = {"lines_added": 0, "lines_removed": 0}
        try:
            async with WorkspaceSnapshot(self.settings.workspace_dir) as snap:
                try:
                    exec_result = await agent_manager.execute_graph(task_graph)
                    changed_files = snap.get_changed_files()
                    diff_stats = snap.compute_diff_stats()
                    snap.commit()
                except Exception:
                    logger.error("Modification failed — restoring workspace from snapshot")
                    raise
                finally:
                    if sandbox_manager:
                        try:
                            await sandbox_manager.destroy_all()
                        except Exception:
                            logger.warning("Sandbox teardown failed (non-critical)")
        except Exception as e:
            logger.exception("Modification execution failed")
            if self._live:
                self._live.fail_phase("Code Modification & Review", str(e))
            return PipelineResult(
                success=False,
                workspace_path=self.settings.workspace_dir,
                blueprint=blueprint,
                change_plan=change_plan,
                repo_analysis=repo_analysis,
                errors=[f"Modification execution failed: {e}"],
                elapsed_seconds=time.monotonic() - start_time,
            )

        if self._live:
            self._live.complete_phase("Code Modification & Review")

        # ── Phase 5: Finalize ─────────────────────────────────────────
        if self._live:
            self._live.set_phase("Finalize", "running")
        logger.info("[Phase 5] Finalizing modifications...")

        try:
            self._index_workspace(repo_manager)
        except Exception as e:
            logger.warning(f"Re-indexing failed (non-critical): {e}")

        elapsed = time.monotonic() - start_time
        stats = exec_result.get("stats", {})
        success = stats.get("failed", 0) == 0 and stats.get("blocked", 0) == 0

        if self._live:
            self._live.complete_phase("Finalize")

        logger.info(f"Modification pipeline {'SUCCEEDED' if success else 'COMPLETED WITH ISSUES'}")
        logger.info(f"Stats: {stats}")
        logger.info(f"Elapsed: {elapsed:.1f}s | Workspace: {self.settings.workspace_dir}")

        # Build cost summary and write report
        token_cost = self._build_token_cost()
        RunReporter(self.settings.workspace_dir).write_modify_report(
            prompt=user_prompt,
            change_plan=change_plan,
            task_graph=task_graph,
            stats=stats,
            elapsed=elapsed,
            success=success,
            changed_files=changed_files,
            diff_stats=diff_stats,
            token_cost=token_cost,
        )

        self._stop_file_logging(file_handler)
        self._file_handler = None

        return PipelineResult(
            success=success,
            workspace_path=self.settings.workspace_dir,
            blueprint=blueprint,
            change_plan=change_plan,
            repo_analysis=repo_analysis,
            task_stats=stats,
            metrics=exec_result.get("metrics", {}),
            errors=errors,
            elapsed_seconds=elapsed,
            token_cost=token_cost,
        )

