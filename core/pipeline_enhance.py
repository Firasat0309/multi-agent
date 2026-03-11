"""Enhancement pipeline — targeted modification of an existing repository."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from agents.change_planner_agent import ChangePlannerAgent
from agents.repository_analyzer_agent import RepositoryAnalyzerAgent
from config.settings import Settings
from core.agent_manager import AgentManager
from core.event_bus import EventBus
from core.language import detect_language_from_blueprint
from core.llm_client import LLMClient, calculate_cost
from core.models import ChangeAction, ChangeActionType, FileBlueprint, RepositoryBlueprint, TokenCost
from core.plan_approver import PlanApprover, PlanPendingApprovalError
from core.repository_manager import RepositoryManager
from core.run_reporter import RunReporter
from core.sandbox_orchestrator import SandboxOrchestrator, SandboxUnavailableError
from core.task_engine import ModificationTaskGraphBuilder
from core.workspace_indexer import index_workspace
from core.workspace_snapshot import WorkspaceSnapshot
from memory.dependency_graph import DependencyGraphStore
from memory.embedding_store import EmbeddingStore

if TYPE_CHECKING:
    from core.live_console import LiveConsole
    from core.pipeline import PipelineResult

logger = logging.getLogger(__name__)


class EnhancePipeline:
    """Executes the repository-modification workflow.

    Phases:
      1. Analysis      — RepositoryAnalyzerAgent scans existing files
      2. Change Plan   — ChangePlannerAgent produces a structured diff plan
      3. Task DAG      — ModificationTaskGraphBuilder builds execution order
      4. Execution     — AgentManager performs targeted edits inside a snapshot
      5. Finalise      — re-index workspace, write modify_report.json
    """

    def __init__(
        self,
        settings: Settings,
        llm: LLMClient,
        live: LiveConsole | None,
        interactive: bool = True,
    ) -> None:
        self._settings = settings
        self._llm = llm
        self._live = live
        self._interactive = interactive

    async def execute(self, user_prompt: str, start_time: float) -> PipelineResult:
        from core.pipeline import PipelineResult

        errors: list[str] = []
        repo_manager = RepositoryManager(self._settings.workspace_dir)

        # ── Phase 1: Repository Analysis ─────────────────────────────────────
        self._phase("Repository Analysis", "running")
        logger.info("[Phase 1] Analysing existing repository...")

        analyzer = RepositoryAnalyzerAgent(
            llm_client=self._llm, repo_manager=repo_manager,
        )
        try:
            repo_analysis = await analyzer.analyze_repository(self._settings.workspace_dir)
        except Exception as e:
            logger.exception("Repository analysis failed")
            self._fail_phase("Repository Analysis", str(e))
            return PipelineResult(
                success=False,
                workspace_path=self._settings.workspace_dir,
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
                name=f"(modify) {self._settings.workspace_dir.name}",
                language=lang_profile.display_name,
                files=len(repo_analysis.modules),
                style=repo_analysis.architecture_style,
            )
        self._complete_phase("Repository Analysis")

        repo_manager.scan_existing_repo()

        # ── Load & sync memory stores ─────────────────────────────────────────
        dep_store = DependencyGraphStore(self._settings.workspace_dir)
        embedding_store = EmbeddingStore(
            persist_dir=self._settings.memory.chroma_persist_dir,
            embedding_model=self._settings.memory.embedding_model,
        )
        repo_manager._embedding_store = embedding_store

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
            "Memory stores loaded: dep_graph=%d nodes",
            len(dep_store.get_graph().nodes()),
        )

        # Minimal blueprint for context-builder compatibility
        blueprint = RepositoryBlueprint(
            name=self._settings.workspace_dir.name,
            description=repo_analysis.summary,
            architecture_style=repo_analysis.architecture_style,
            tech_stack=repo_analysis.tech_stack,
            architecture_doc=repo_analysis.summary,
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

        # ── Phase 2: Change Planning ──────────────────────────────────────────
        self._phase("Change Planning", "running")
        logger.info("[Phase 2] Planning targeted changes...")

        planner = ChangePlannerAgent(llm_client=self._llm, repo_manager=repo_manager)
        try:
            change_plan = await planner.plan_changes(
                user_request=user_prompt,
                repo_analysis=repo_analysis,
            )
        except Exception as e:
            logger.exception("Change planning failed")
            self._fail_phase("Change Planning", str(e))
            return PipelineResult(
                success=False,
                workspace_path=self._settings.workspace_dir,
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
        self._complete_phase("Change Planning")

        # ── Approval gate ─────────────────────────────────────────────────────
        if self._settings.require_plan_approval:
            approver = PlanApprover(
                interactive=self._interactive,
                workspace=self._settings.workspace_dir,
            )
            try:
                approved = approver.display_and_approve(change_plan)
            except PlanPendingApprovalError as e:
                logger.info("Plan pending human approval: %s", e)
                return PipelineResult(
                    success=False,
                    workspace_path=self._settings.workspace_dir,
                    repo_analysis=repo_analysis,
                    errors=[str(e)],
                    elapsed_seconds=time.monotonic() - start_time,
                )
            if not approved:
                logger.info("Change plan rejected by user")
                return PipelineResult(
                    success=False,
                    workspace_path=self._settings.workspace_dir,
                    repo_analysis=repo_analysis,
                    errors=["Change plan rejected by user"],
                    elapsed_seconds=time.monotonic() - start_time,
                )

        # ── Reconcile new_files against workspace ────────────────────────────
        # The LLM sometimes mis-classifies existing files as new (e.g. pom.xml,
        # application.properties that are already present in the repo).  A
        # GENERATE_FILE task on an existing file overwrites it from scratch and
        # fails if the file has no blueprint entry.  The correct treatment is a
        # MODIFY_FILE task, so we demote any already-existing "new" files to
        # a change action and remove them from new_files.
        workspace = self._settings.workspace_dir
        truly_new: list[FileBlueprint] = []
        for nf in change_plan.new_files:
            if (workspace / nf.path).exists():
                logger.warning(
                    "new_files entry '%s' already exists on disk — demoting to modify change",
                    nf.path,
                )
                # Only add a modify action if one isn't already there to avoid
                # generating two tasks for the same file.
                already_in_changes = any(c.file == nf.path for c in change_plan.changes)
                if not already_in_changes:
                    # Use ADD_FUNCTION (generic "add something") rather than
                    # CREATE_FILE, since the file already exists — PatchAgent
                    # receives change_type as a hint in its prompt and
                    # CREATE_FILE on an existing file is contradictory.
                    change_plan.changes.append(ChangeAction(
                        type=ChangeActionType.ADD_FUNCTION,
                        file=nf.path,
                        description=nf.purpose,
                    ))
            else:
                truly_new.append(nf)

        if len(truly_new) != len(change_plan.new_files):
            logger.info(
                "new_files reconciled: %d truly new, %d demoted to modify",
                len(truly_new), len(change_plan.new_files) - len(truly_new),
            )
        change_plan.new_files = truly_new

        # Enrich the repo blueprint with genuinely new files so the context
        # builder can supply file_blueprint to CoderAgent for GENERATE_FILE tasks.
        if change_plan.new_files:
            blueprint.file_blueprints = blueprint.file_blueprints + change_plan.new_files
            logger.info(
                "Blueprint enriched with %d new file(s) from change plan",
                len(change_plan.new_files),
            )

        # ── Phase 3: Build modification task DAG ─────────────────────────────
        self._phase("Task Planning", "running")
        logger.info("[Phase 3] Building modification task graph...")

        mod_builder = ModificationTaskGraphBuilder(dep_store=dep_store)
        task_graph = mod_builder.build_from_change_plan(change_plan, blueprint)
        logger.info("Modification task graph: %d tasks", len(task_graph.tasks))

        self._complete_phase("Task Planning")
        if self._live:
            for task in task_graph.tasks.values():
                self._live.update_task(task.task_id, task.description, task.status.value)

        # ── Phase 4: Execute modifications ────────────────────────────────────
        self._phase("Code Modification & Review", "running")
        logger.info("[Phase 4] Executing targeted modifications...")

        sandbox = SandboxOrchestrator(self._settings)
        try:
            sb = await sandbox.setup(lang_profile)
        except SandboxUnavailableError as e:
            self._fail_phase("Code Modification & Review", str(e))
            return PipelineResult(
                success=False,
                workspace_path=self._settings.workspace_dir,
                change_plan=change_plan,
                repo_analysis=repo_analysis,
                errors=[str(e)],
                elapsed_seconds=time.monotonic() - start_time,
            )

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
            dep_store=dep_store,
            embedding_store=embedding_store,
            event_bus=event_bus,
        )

        changed_files: list[str] = []
        diff_stats: dict[str, int] = {"lines_added": 0, "lines_removed": 0}
        try:
            async with WorkspaceSnapshot(self._settings.workspace_dir) as snap:
                try:
                    exec_result = await agent_manager.execute_graph(task_graph)
                    changed_files = snap.get_changed_files()
                    diff_stats = snap.compute_diff_stats()
                    snap.commit()
                except Exception:
                    logger.error("Modification failed — restoring workspace from snapshot")
                    raise
                finally:
                    await sandbox.teardown()
        except Exception as e:
            logger.exception("Modification execution failed")
            self._fail_phase("Code Modification & Review", str(e))
            return PipelineResult(
                success=False,
                workspace_path=self._settings.workspace_dir,
                blueprint=blueprint,
                change_plan=change_plan,
                repo_analysis=repo_analysis,
                errors=[f"Modification execution failed: {e}"],
                elapsed_seconds=time.monotonic() - start_time,
            )

        self._complete_phase("Code Modification & Review")

        # ── Phase 5: Finalise ─────────────────────────────────────────────────
        self._phase("Finalize", "running")
        logger.info("[Phase 5] Finalising modifications...")

        try:
            index_workspace(repo_manager, self._settings)
        except Exception as e:
            logger.warning("Re-indexing failed (non-critical): %s", e)

        elapsed = time.monotonic() - start_time
        stats = exec_result.get("stats", {})
        success = stats.get("failed", 0) == 0 and stats.get("blocked", 0) == 0

        self._complete_phase("Finalize")
        logger.info(
            "Modification pipeline %s | stats=%s | elapsed=%.1fs",
            "SUCCEEDED" if success else "COMPLETED WITH ISSUES", stats, elapsed,
        )

        token_cost = self._build_token_cost()
        RunReporter(self._settings.workspace_dir).write_modify_report(
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

        return PipelineResult(
            success=success,
            workspace_path=self._settings.workspace_dir,
            blueprint=blueprint,
            change_plan=change_plan,
            repo_analysis=repo_analysis,
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
