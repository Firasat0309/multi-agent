"""Lifecycle orchestrator — drives per-file FSM phases via LifecycleEngine.

Single responsibility: given a ``LifecycleEngine`` and a global ``TaskGraph``,
advance every file through its Generate → Review → Fix → Build → Test lifecycle,
then hand off to ``TaskDispatcher`` for the global DAG (security, deploy, docs).

``AgentManager._execute_lifecycle_phase`` owns the canonical per-file execution
logic; this orchestrator focuses exclusively on the state-machine event loop
and tier coordination.
"""

from __future__ import annotations

import asyncio
import logging
import time
import warnings
from typing import Any, TYPE_CHECKING

from core.models import TaskType
from core.state_machine import EventType, FilePhase

if TYPE_CHECKING:
    from core.agent_manager import AgentManager
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

        .. deprecated::
            Prefer calling :class:`~core.pipeline_executor.PipelineExecutor`
            directly.  This method is kept for backward compatibility only.
        """
        warnings.warn(
            "LifecycleOrchestrator.execute_with_lifecycle is deprecated. "
            "Use PipelineExecutor.execute() directly.",
            DeprecationWarning,
            stacklevel=2,
        )
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

        .. deprecated::
            Prefer calling :class:`~core.pipeline_executor.PipelineExecutor`
            directly.  This method is kept for backward compatibility only.
        """
        warnings.warn(
            "LifecycleOrchestrator.execute_with_checkpoints is deprecated. "
            "Use PipelineExecutor.execute() directly.",
            DeprecationWarning,
            stacklevel=2,
        )
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
        """Execute one lifecycle phase — delegates to AgentManager.

        All implementation lives in :meth:`AgentManager._execute_lifecycle_phase`;
        this shim exists so ``execute_with_lifecycle`` routes through a single
        stable entry point without the caller needing to reference AgentManager
        directly.
        """
        await self._am._execute_lifecycle_phase(engine, file_path, phase)

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
