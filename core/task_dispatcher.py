"""Task graph executor — dispatches individual tasks to the right agents.

Single responsibility: given a ``TaskGraph``, execute every node by selecting
the appropriate agent via ``AgentManager._create_agent``, managing retries,
updating metrics, and publishing event-bus events.

``AgentManager`` holds the public delegation shim so all existing call-sites
continue to work without modification.  This module owns the *implementation*.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, TYPE_CHECKING

from core.context_builder import ContextBuilder
from core.event_bus import AgentEvent, BusEventType
from core.models import Task, TaskStatus, TaskType
from core.observability import record_agent_end, record_agent_start, record_task_completion

if TYPE_CHECKING:
    from core.agent_manager import AgentManager
    from core.task_engine import TaskGraph

logger = logging.getLogger(__name__)


class TaskDispatcher:
    """Executes a ``TaskGraph`` node-by-node, respecting declared dependencies.

    Callers should only use :meth:`execute_graph`; the ``_execute_task`` helper
    is an implementation detail and must not be called directly by pipelines.
    """

    def __init__(self, am: AgentManager) -> None:
        self._am = am

    # ── Public API ────────────────────────────────────────────────────────────

    async def execute_graph(self, task_graph: TaskGraph) -> dict[str, Any]:
        """Execute all tasks in *task_graph* while respecting their dependency order.

        Tasks whose dependencies have all completed are eligible to run
        concurrently up to ``settings.max_concurrent_agents``.  A deadlock
        guard detects the state where no tasks are ready and none are running,
        marks all remaining tasks as FAILED, and breaks the loop so the caller
        receives a result rather than hanging forever.
        """
        start_time = time.monotonic()
        semaphore = asyncio.Semaphore(self._am.settings.max_concurrent_agents)

        while task_graph.has_remaining_tasks():
            ready_tasks = task_graph.get_ready_tasks()
            if not ready_tasks:
                stats = task_graph.get_stats()
                if stats.get("in_progress", 0) == 0:
                    logger.error(
                        "TaskDispatcher: deadlock — no ready tasks and none in progress"
                    )
                    for t in task_graph.tasks.values():
                        if t.status in (TaskStatus.PENDING, TaskStatus.READY):
                            task_graph.mark_failed(t.task_id)
                            self._am._metrics["tasks_failed"] += 1
                    break
                await asyncio.sleep(0.5)
                continue

            logger.info("TaskDispatcher: dispatching %d task(s)", len(ready_tasks))

            async def run_task(task: Task) -> None:
                async with semaphore:
                    await self._execute_task(task, task_graph)

            await asyncio.gather(*[run_task(t) for t in ready_tasks])

        elapsed = time.monotonic() - start_time
        self._am._metrics["total_time"] = elapsed

        self._am.repo.save_repo_index()
        try:
            self._am.repo.rebuild_dependency_graph()
        except Exception as exc:
            logger.warning("Dependency graph rebuild failed (non-critical): %s", exc)

        stats = task_graph.get_stats()
        logger.info("TaskDispatcher: complete in %.1fs — %s", elapsed, stats)

        return {
            "stats": stats,
            "metrics": self._am._metrics,
            "elapsed_seconds": elapsed,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _execute_task(self, task: Task, task_graph: TaskGraph) -> None:
        """Execute a single task with up to ``task.max_retries`` attempts.

        Context is rebuilt on every retry so the agent always sees the current
        workspace state (not a stale snapshot from a previous failed attempt).
        File-level locking prevents concurrent writes to the same path.
        """
        task.status = TaskStatus.IN_PROGRESS
        agent_name = ""
        task_start = time.monotonic()

        live = self._am._live
        if live:
            live.update_task(task.task_id, task.description, "in_progress")
            live.log(f"[cyan]Starting:[/cyan] {task.description}")

        # FIX_CODE shortcut: skip entirely if the preceding review already passed.
        if task.task_type == TaskType.FIX_CODE and "review_task_id" in task.metadata:
            review_task = task_graph.get_task(task.metadata["review_task_id"])
            if review_task and review_task.result:
                if review_task.result.metrics.get("passed", False):
                    logger.info(
                        "FIX_CODE task %d skipped — review passed for %s",
                        task.task_id, task.file,
                    )
                    task_graph.mark_completed(task.task_id)
                    self._am._metrics["tasks_completed"] += 1
                    if live:
                        live.update_task(task.task_id, task.description, "completed")
                    return
                task.metadata["review_errors"] = review_task.result.errors
                task.metadata["review_output"] = review_task.result.output

        # Per-file lock guards concurrent writes/reads to the same path.
        file_lock = (
            self._am._file_locks.lock_for(task.file)
            if task.file and task.file != "*"
            else None
        )

        for attempt in range(task.max_retries):
            context_builder = ContextBuilder(
                workspace_dir=self._am.repo.workspace,
                blueprint=self._am.blueprint,
                repo_index=self._am.repo.get_repo_index(),
                dep_store=self._am._dep_store,
                embedding_store=self._am._embedding_store,
                api_contract=self._am._api_contract,
            )
            context = await asyncio.to_thread(context_builder.build, task)

            try:
                agent = self._am._create_agent(task.task_type)
                agent_name = agent.role.value
                task.assigned_agent = f"{agent_name}-{id(agent)}"

                if live:
                    live.update_task(
                        task.task_id, task.description, "in_progress", agent_name
                    )

                record_agent_start()
                try:
                    if file_lock:
                        async with file_lock:
                            result = await agent.execute(context)
                    else:
                        result = await agent.execute(context)
                finally:
                    record_agent_end()

                task.result = result

                if agent_name not in self._am._metrics["agent_metrics"]:
                    self._am._metrics["agent_metrics"][agent_name] = []
                self._am._metrics["agent_metrics"][agent_name].append(agent.get_metrics())

                if result.success:
                    elapsed = time.monotonic() - task_start
                    task_graph.mark_completed(task.task_id)
                    self._am._metrics["tasks_completed"] += 1
                    record_task_completion(task.task_type.value, "completed", elapsed)
                    logger.info("Task %d completed: %s", task.task_id, result.output)
                    # Incremental embedding update so subsequent context builds
                    # see newly written files via semantic search.
                    if self._am._embedding_store and result.files_modified:
                        for fp in result.files_modified:
                            try:
                                content = self._am.repo.read_file(fp)
                                self._am._embedding_store.index_file(fp, content)
                            except Exception:
                                logger.debug(
                                    "Embedding update skipped for %s", fp
                                )
                            if self._am._event_bus:
                                await self._am._event_bus.publish(AgentEvent(
                                    type=BusEventType.FILE_WRITTEN,
                                    task_id=task.task_id,
                                    task_type=task.task_type.value,
                                    file_path=fp,
                                    agent_name=agent_name,
                                ))
                    if self._am._event_bus:
                        await self._am._event_bus.publish(AgentEvent(
                            type=BusEventType.TASK_COMPLETED,
                            task_id=task.task_id,
                            task_type=task.task_type.value,
                            file_path=task.file,
                            agent_name=agent_name,
                        ))
                    if live:
                        live.update_task(
                            task.task_id, task.description, "completed", agent_name
                        )
                        live.log(f"[green]Done:[/green] {task.description}")
                    return
                else:
                    logger.warning(
                        "Task %d attempt %d failed: %s",
                        task.task_id, attempt + 1, result.errors,
                    )
                    if live:
                        live.log(
                            f"[yellow]Retry {attempt + 1}:[/yellow] {task.description}"
                        )
                    task.retry_count += 1

            except Exception:
                logger.exception(
                    "Task %d error on attempt %d", task.task_id, attempt + 1
                )
                task.retry_count += 1

        # All retries exhausted.
        elapsed = time.monotonic() - task_start
        task_graph.mark_failed(task.task_id)
        self._am._metrics["tasks_failed"] += 1
        record_task_completion(task.task_type.value, "failed", elapsed)
        logger.error(
            "Task %d failed after %d attempt(s)", task.task_id, task.max_retries
        )
        if self._am._event_bus:
            await self._am._event_bus.publish(AgentEvent(
                type=BusEventType.TASK_FAILED,
                task_id=task.task_id,
                task_type=task.task_type.value,
                file_path=task.file,
                agent_name=agent_name,
            ))
        if live:
            live.update_task(task.task_id, task.description, "failed", agent_name)
            live.log(f"[red]Failed:[/red] {task.description}")
