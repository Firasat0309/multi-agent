"""Lifecycle orchestrator — drives per-file FSM phases via LifecycleEngine.

Single responsibility: given a ``LifecycleEngine`` and a global ``TaskGraph``,
advance every file through its Generate → Review → Fix → Build → Test lifecycle,
then hand off to ``TaskDispatcher`` for the global DAG (security, deploy, docs).

``AgentManager`` keeps public shim methods so that all existing callers remain
unchanged.  This module owns the *implementation*.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, TYPE_CHECKING

from core.context_builder import ContextBuilder
from core.event_bus import AgentEvent, BusEventType
from core.models import Task, TaskType
from core.observability import record_agent_end, record_agent_start
from core.state_machine import EventType, FilePhase

if TYPE_CHECKING:
    from core.agent_manager import AgentManager
    from core.pipeline_definition import PipelineDefinition
    from core.state_machine import LifecycleEngine
    from core.task_engine import TaskGraph
    from core.tier_scheduler import Tier

logger = logging.getLogger(__name__)


class LifecycleOrchestrator:
    """Runs the event-sourced per-file lifecycle and the subsequent global DAG.

    Two public entry points mirror the original ``AgentManager`` methods:

    * :meth:`execute_with_lifecycle` — pure lifecycle FSM mode.
    * :meth:`execute_with_checkpoints` — tier-scheduled mode that delegates to
      :class:`~core.pipeline_executor.PipelineExecutor`.
    """

    def __init__(self, am: AgentManager) -> None:
        self._am = am

    # ── Public entry points ───────────────────────────────────────────────────

    async def execute_with_lifecycle(
        self,
        engine: LifecycleEngine,
        global_graph: TaskGraph,
    ) -> dict[str, Any]:
        """Phase 1: drive per-file lifecycles; Phase 2: run the global DAG.

        Phase 1 completes when all files have reached a terminal state.  Phase 2
        then marks the sentinel task as completed (unblocking the global DAG) and
        delegates graph execution to :meth:`~AgentManager.execute_graph`.
        """
        start_time = time.monotonic()
        semaphore = asyncio.Semaphore(self._am.settings.max_concurrent_agents)
        phase_timeout: float = float(self._am.settings.phase_timeout_seconds)

        in_flight: set[str] = set()
        running_tasks: set[asyncio.Task[None]] = set()

        async def run_file_phase(path: str, phase: FilePhase) -> None:
            async with semaphore:
                try:
                    await asyncio.wait_for(
                        self._execute_lifecycle_phase(engine, path, phase),
                        timeout=phase_timeout,
                    )
                except asyncio.TimeoutError:
                    logger.error(
                        "[%s] %s timed out after %ds — marking FAILED",
                        path, phase.value, int(phase_timeout),
                    )
                    try:
                        engine.process_event(path, EventType.RETRIES_EXHAUSTED)
                    except Exception:
                        pass
                    self._am._metrics["tasks_failed"] += 1
                finally:
                    in_flight.discard(path)

        while not engine.all_terminal():
            actionable = engine.get_actionable_files()
            to_dispatch = [
                (path, phase) for path, phase in actionable
                if path not in in_flight
            ]
            for path, phase in to_dispatch:
                in_flight.add(path)
                t = asyncio.create_task(
                    run_file_phase(path, phase), name=f"lifecycle:{path}"
                )
                running_tasks.add(t)
                t.add_done_callback(running_tasks.discard)

            if not running_tasks:
                logger.error(
                    "Lifecycle stall: no actionable files and none in-flight. "
                    "Phase counts: %s",
                    engine.get_stats(),
                )
                break

            await asyncio.wait(running_tasks, return_when=asyncio.FIRST_COMPLETED)

        for t in list(running_tasks):
            t.cancel()
        if running_tasks:
            await asyncio.gather(*list(running_tasks), return_exceptions=True)

        # ── Phase 2: Global DAG ───────────────────────────────────────────────
        logger.info("All file lifecycles terminal — running global DAG")
        sentinel = next(
            (t for t in global_graph.tasks.values() if t.metadata.get("sentinel")),
            None,
        )
        if sentinel:
            global_graph.mark_completed(sentinel.task_id)
        else:
            logger.warning(
                "Sentinel task not found in global DAG — global phases may stall"
            )

        # Delegate graph execution back through AgentManager so the metrics
        # dict remains the single source of truth.
        await self._am.execute_graph(global_graph)

        elapsed = time.monotonic() - start_time
        self._am.repo.save_repo_index()
        try:
            self._am.repo.rebuild_dependency_graph()
        except Exception as exc:
            logger.warning("Dependency graph rebuild failed (non-critical): %s", exc)

        lifecycle_summary = engine.get_results_summary()
        global_stats = global_graph.get_stats()

        logger.info(
            "Lifecycle execution complete in %.1fs. Files: %d passed, %d failed. "
            "Global tasks: %s",
            elapsed,
            lifecycle_summary["passed"],
            lifecycle_summary["failed"],
            global_stats,
        )

        return {
            "stats": {
                **global_stats,
                "lifecycle_passed": lifecycle_summary["passed"],
                "lifecycle_failed": lifecycle_summary["failed"],
                "lifecycle_total_fixes": lifecycle_summary["total_fix_cycles"],
                "lifecycle_tests_degraded": lifecycle_summary["tests_degraded"],
            },
            "metrics": self._am._metrics,
            "elapsed_seconds": elapsed,
            "lifecycle_summary": lifecycle_summary,
        }

    async def execute_with_checkpoints(
        self,
        engine: LifecycleEngine,
        global_graph: TaskGraph,
        *,
        tiers: list[Tier] | None = None,
        pipeline_def: Any = None,
    ) -> dict[str, Any]:
        """Tier-scheduled execution with repo-level build checkpoints.

        Delegates entirely to :class:`~core.pipeline_executor.PipelineExecutor`
        which owns the tier-blocking and checkpoint-retry logic.
        """
        from core.pipeline_executor import PipelineExecutor

        executor = PipelineExecutor(
            agent_manager=self._am,
            settings=self._am.settings,
            lang_profile=self._am._lang,
            event_bus=self._am._event_bus,
        )
        return await executor.execute(
            engine,
            global_graph,
            pipeline_def=pipeline_def,
            tiers=tiers,
        )

    # ── Per-file phase execution ──────────────────────────────────────────────

    async def _execute_lifecycle_phase(
        self,
        engine: LifecycleEngine,
        file_path: str,
        phase: FilePhase,
    ) -> None:
        """Execute one lifecycle phase for *file_path* and drive the FSM forward.

        Maps the current ``FilePhase`` to the appropriate ``TaskType``, runs the
        agent, then fires the correct ``EventType`` based on the result so the
        state machine can transition to the next phase.
        """
        lc = engine.get_lifecycle(file_path)

        # PENDING is a virtual phase — fire DEPS_MET to advance to GENERATING.
        if phase == FilePhase.PENDING:
            engine.process_event(file_path, EventType.DEPS_MET)
            phase = lc.phase  # now GENERATING

        gen_task_type = TaskType(lc.generation_task_type)
        gen_verb = "Modify" if gen_task_type == TaskType.MODIFY_FILE else "Generate"

        phase_config: dict[FilePhase, dict[str, Any]] = {
            FilePhase.GENERATING: {
                "task_type": gen_task_type,
                "success_event": EventType.CODE_GENERATED,
                "failure_event": EventType.RETRIES_EXHAUSTED,
                "description": f"{gen_verb} {file_path}",
            },
            FilePhase.REVIEWING: {
                "task_type": TaskType.REVIEW_FILE,
                "success_event": EventType.REVIEW_PASSED,
                "failure_event": EventType.REVIEW_FAILED,
                "description": f"Review {file_path}",
            },
            FilePhase.FIXING: {
                "task_type": TaskType.FIX_CODE,
                # Even on fix-agent failure fire FIX_APPLIED to re-enter the
                # review cycle rather than hard-failing the file.
                "success_event": EventType.FIX_APPLIED,
                "failure_event": EventType.FIX_APPLIED,
                "description": f"Fix {file_path} ({lc.fix_trigger} issues)",
            },
            FilePhase.BUILDING: {
                "task_type": TaskType.VERIFY_BUILD,
                "success_event": EventType.BUILD_PASSED,
                "failure_event": EventType.BUILD_FAILED,
                "description": f"Verify build for {file_path}",
            },
            FilePhase.TESTING: {
                "task_type": TaskType.GENERATE_TEST,
                "success_event": EventType.TEST_PASSED,
                "failure_event": EventType.TEST_FAILED,
                "description": f"Test {file_path}",
            },
        }

        config = phase_config.get(phase)
        if config is None:
            logger.warning("No action for phase %s on %s", phase.value, file_path)
            return

        task_meta = self._build_lifecycle_metadata(lc)
        task_meta.update(lc.change_metadata)
        task = Task(
            task_id=0,
            task_type=config["task_type"],
            file=file_path,
            description=config["description"],
            metadata=task_meta,
        )

        if self._am._live:
            self._am._live.log(f"[cyan]Lifecycle:[/cyan] {config['description']}")

        context_builder = ContextBuilder(
            workspace_dir=self._am.repo.workspace,
            blueprint=self._am.blueprint,
            repo_index=self._am.repo.get_repo_index(),
            dep_store=self._am._dep_store,
            embedding_store=self._am._embedding_store,
        )
        # Run in thread — SentenceTransformer loads synchronously on first call.
        context = await asyncio.to_thread(context_builder.build, task)

        try:
            agent = self._am._create_agent(config["task_type"])
            record_agent_start()
            try:
                result = await agent.execute(context)
            finally:
                record_agent_end()

            agent_name = agent.role.value
            if agent_name not in self._am._metrics["agent_metrics"]:
                self._am._metrics["agent_metrics"][agent_name] = []
            self._am._metrics["agent_metrics"][agent_name].append(agent.get_metrics())

            # Review is special: task success only means the agent ran without
            # crashing; result.metrics["passed"] carries the actual verdict.
            if config["task_type"] == TaskType.REVIEW_FILE:
                review_passed = result.success and result.metrics.get("passed", False)
                if review_passed:
                    event_data = {"output": result.output}
                    engine.process_event(file_path, EventType.REVIEW_PASSED, event_data)
                    if self._am._event_bus:
                        await self._am._event_bus.publish(AgentEvent(
                            type=BusEventType.REVIEW_PASSED,
                            task_type=config["task_type"].value,
                            file_path=file_path,
                            agent_name=agent_name,
                        ))
                else:
                    event_data = {"findings": result.errors, "output": result.output}
                    engine.process_event(file_path, EventType.REVIEW_FAILED, event_data)
                    if self._am._event_bus:
                        await self._am._event_bus.publish(AgentEvent(
                            type=BusEventType.REVIEW_FAILED,
                            task_type=config["task_type"].value,
                            file_path=file_path,
                            agent_name=agent_name,
                            data={"findings": result.errors},
                        ))
                self._am._metrics["tasks_completed"] += 1

            elif result.success:
                self._am._metrics["tasks_completed"] += 1
                event_data = self._extract_event_data(result, config["task_type"])
                engine.process_event(file_path, config["success_event"], event_data)
                logger.info("[%s] %s succeeded", file_path, phase.value)
                if self._am._embedding_store and result.files_modified:
                    for fp in result.files_modified:
                        try:
                            content = self._am.repo.read_file(fp)
                            self._am._embedding_store.index_file(fp, content)
                        except Exception:
                            logger.debug("Embedding update skipped for %s", fp)
                        if self._am._event_bus:
                            await self._am._event_bus.publish(AgentEvent(
                                type=BusEventType.FILE_WRITTEN,
                                task_type=config["task_type"].value,
                                file_path=fp,
                                agent_name=agent_name,
                            ))
                if self._am._event_bus:
                    bus_type = (
                        BusEventType.TEST_PASSED
                        if config["task_type"] == TaskType.GENERATE_TEST
                        else BusEventType.TASK_COMPLETED
                    )
                    await self._am._event_bus.publish(AgentEvent(
                        type=bus_type,
                        task_type=config["task_type"].value,
                        file_path=file_path,
                        agent_name=agent_name,
                    ))
                if self._am._live:
                    self._am._live.log(
                        f"[green]Done:[/green] {config['description']}"
                    )

            else:
                event_data = self._extract_event_data(result, config["task_type"])
                engine.process_event(file_path, config["failure_event"], event_data)
                logger.warning(
                    "[%s] %s failed: %s", file_path, phase.value, result.errors
                )
                if self._am._event_bus:
                    bus_type = (
                        BusEventType.TEST_FAILED
                        if config["task_type"] == TaskType.GENERATE_TEST
                        else BusEventType.TASK_FAILED
                    )
                    await self._am._event_bus.publish(AgentEvent(
                        type=bus_type,
                        task_type=config["task_type"].value,
                        file_path=file_path,
                        agent_name=agent_name,
                        data={"errors": result.errors},
                    ))
                if self._am._live:
                    self._am._live.log(
                        f"[yellow]Issue:[/yellow] {config['description']} — "
                        f"transitioning via {config['failure_event'].value}"
                    )

        except Exception:
            logger.exception("[%s] %s error", file_path, phase.value)
            # Degrade gracefully: review/fix exceptions must not hard-fail a
            # file — only generation-level failures result in FAILED state.
            if phase == FilePhase.REVIEWING:
                engine.process_event(file_path, EventType.REVIEW_PASSED)
            elif phase == FilePhase.FIXING:
                engine.process_event(file_path, EventType.FIX_APPLIED)
            else:
                engine.process_event(file_path, EventType.RETRIES_EXHAUSTED)
                self._am._metrics["tasks_failed"] += 1

    # ── Static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _build_lifecycle_metadata(lc: Any) -> dict[str, Any]:
        """Build task metadata from lifecycle state for downstream agents."""
        meta: dict[str, Any] = {}
        if lc.fix_trigger == "review":
            meta["review_errors"] = lc.review_findings
            meta["review_output"] = lc.review_output
            meta["fix_trigger"] = "review"
        elif lc.fix_trigger == "test":
            meta["test_errors"] = lc.test_errors
            meta["fix_trigger"] = "test"
            meta["test_fix_target"] = lc.test_fix_target
        elif lc.fix_trigger == "build":
            meta["build_errors"] = lc.build_errors
            meta["fix_trigger"] = "build"
        return meta

    @staticmethod
    def _extract_event_data(result: Any, task_type: TaskType) -> dict[str, Any]:
        """Extract event data from agent result for lifecycle FSM transitions."""
        data: dict[str, Any] = {}
        if task_type == TaskType.GENERATE_TEST:
            if not result.success:
                data["errors"] = "\n".join(result.errors) if result.errors else result.output
        if task_type == TaskType.VERIFY_BUILD:
            data["errors"] = "\n".join(result.errors) if result.errors else result.output
        return data
