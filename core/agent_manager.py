"""Agent manager - orchestrates task execution across agents."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, TYPE_CHECKING

from agents.architect_agent import ArchitectAgent
from agents.base_agent import BaseAgent
from agents.build_verifier_agent import BuildVerifierAgent
from agents.coder_agent import CoderAgent
from agents.deploy_agent import DeployAgent
from agents.integration_test_agent import IntegrationTestAgent
from agents.patch_agent import PatchAgent
from agents.reviewer_agent import ReviewerAgent
from agents.security_agent import SecurityAgent
from agents.test_agent import TestAgent
from agents.planner_agent import PlannerAgent
from agents.writer_agent import WriterAgent
from config.settings import Settings
from core.context_builder import ContextBuilder
from core.event_bus import AgentEvent, BusEventType, EventBus
from core.file_lock_manager import FileLockManager
from core.language import detect_language_from_blueprint
from core.llm_client import LLMClient
from core.observability import record_agent_end, record_agent_start, record_task_completion
from core.models import (
    RepositoryBlueprint,
    Task,
    TaskStatus,
    TaskType,
)
from core.repository_manager import RepositoryManager
from core.state_machine import EventType, FilePhase, LifecycleEngine
from core.task_engine import TaskGraph
from tools.terminal_tools import TerminalTools

if TYPE_CHECKING:
    from core.live_console import LiveConsole
    from memory.dependency_graph import DependencyGraphStore
    from memory.embedding_store import EmbeddingStore
    from sandbox.sandbox_runner import SandboxManager

logger = logging.getLogger(__name__)

# Mapping from task type to agent role
TASK_AGENT_MAP: dict[TaskType, type[BaseAgent]] = {
    TaskType.GENERATE_FILE: CoderAgent,
    TaskType.REVIEW_FILE: ReviewerAgent,
    TaskType.REVIEW_MODULE: ReviewerAgent,
    TaskType.REVIEW_ARCHITECTURE: ReviewerAgent,
    TaskType.GENERATE_TEST: TestAgent,
    TaskType.SECURITY_SCAN: SecurityAgent,
    TaskType.GENERATE_DEPLOY: DeployAgent,
    TaskType.GENERATE_DOCS: WriterAgent,
    TaskType.FIX_CODE: CoderAgent,
    TaskType.MODIFY_FILE: PatchAgent,                            # surgical patch
    TaskType.GENERATE_INTEGRATION_TEST: IntegrationTestAgent,
    TaskType.DESIGN_ARCHITECTURE: ArchitectAgent,
    TaskType.CREATE_PLAN: PlannerAgent,
    TaskType.VERIFY_BUILD: BuildVerifierAgent,
}


class AgentManager:
    """Coordinates task execution across specialized agents."""

    def __init__(
        self,
        settings: Settings,
        llm_client: LLMClient,
        repo_manager: RepositoryManager,
        blueprint: RepositoryBlueprint,
        live_console: LiveConsole | None = None,
        sandbox_manager: SandboxManager | None = None,
        build_sandbox_id: str | None = None,
        test_sandbox_id: str | None = None,
        dep_store: DependencyGraphStore | None = None,
        embedding_store: EmbeddingStore | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self.settings = settings
        self.llm = llm_client
        self.repo = repo_manager
        self.blueprint = blueprint
        self._live = live_console
        self._lang = detect_language_from_blueprint(blueprint.tech_stack)
        self._dep_store = dep_store
        self._embedding_store = embedding_store
        self._event_bus = event_bus

        # Two-tier terminal tools: build (network-capable) and test (isolated).
        # TestAgent gets the test terminal; all other agents that need a
        # terminal (SecurityAgent, etc.) get the build terminal.
        self.build_terminal = TerminalTools(
            repo_manager.workspace,
            language=self._lang,
            sandbox_manager=sandbox_manager,
            sandbox_id=build_sandbox_id,
        )
        self.test_terminal = TerminalTools(
            repo_manager.workspace,
            language=self._lang,
            sandbox_manager=sandbox_manager,
            sandbox_id=test_sandbox_id,
        )
        if sandbox_manager and build_sandbox_id:
            logger.info(
                "Build commands routed to sandbox %s; test execution to sandbox %s",
                build_sandbox_id, test_sandbox_id,
            )
        self._metrics: dict[str, Any] = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_time": 0.0,
            "agent_metrics": {},
        }
        self._file_locks = FileLockManager()

    def _create_agent(self, task_type: TaskType) -> BaseAgent:
        agent_cls = TASK_AGENT_MAP.get(task_type)
        if agent_cls is None:
            raise ValueError(f"No agent registered for task type: {task_type}")

        # TestAgent / IntegrationTestAgent → test terminal (no-network sandbox).
        # SecurityAgent → build terminal (bandit needs the full env).
        if agent_cls in (TestAgent, IntegrationTestAgent):
            return agent_cls(
                llm_client=self.llm,
                repo_manager=self.repo,
                terminal=self.test_terminal,
            )
        if agent_cls is SecurityAgent:
            return agent_cls(
                llm_client=self.llm,
                repo_manager=self.repo,
                terminal=self.build_terminal,
            )
        if agent_cls is BuildVerifierAgent:
            return agent_cls(
                llm_client=self.llm,
                repo_manager=self.repo,
                terminal=self.build_terminal,
            )
        return agent_cls(llm_client=self.llm, repo_manager=self.repo)

    async def execute_graph(self, task_graph: TaskGraph) -> dict[str, Any]:
        """Execute all tasks in the graph respecting dependencies."""
        start_time = time.monotonic()
        semaphore = asyncio.Semaphore(self.settings.max_concurrent_agents)

        while task_graph.has_remaining_tasks():
            ready_tasks = task_graph.get_ready_tasks()
            if not ready_tasks:
                # Check for deadlock
                stats = task_graph.get_stats()
                if stats.get("in_progress", 0) == 0:
                    logger.error("Deadlock detected - no ready tasks and none in progress")
                    # Mark abandoned PENDING/READY tasks as FAILED so stats
                    # accurately reflect the incomplete run; without this the
                    # pipeline success-check sees failed=0 and reports success.
                    for t in task_graph.tasks.values():
                        if t.status in (TaskStatus.PENDING, TaskStatus.READY):
                            task_graph.mark_failed(t.task_id)
                            self._metrics["tasks_failed"] += 1
                    break
                await asyncio.sleep(0.5)
                continue

            logger.info(f"Dispatching {len(ready_tasks)} task(s)")

            # Execute ready tasks concurrently (up to limit)
            async def run_task(task: Task) -> None:
                async with semaphore:
                    await self._execute_task(task, task_graph)

            await asyncio.gather(*[run_task(t) for t in ready_tasks])

        elapsed = time.monotonic() - start_time
        self._metrics["total_time"] = elapsed

        # Save final repo index
        self.repo.save_repo_index()

        # Rebuild dependency graph from actual imports (not static blueprint)
        try:
            self.repo.rebuild_dependency_graph()
        except Exception as e:
            logger.warning(f"Dependency graph rebuild failed (non-critical): {e}")

        stats = task_graph.get_stats()
        logger.info(f"Execution complete in {elapsed:.1f}s. Stats: {stats}")

        return {
            "stats": stats,
            "metrics": self._metrics,
            "elapsed_seconds": elapsed,
        }

    async def _execute_task(self, task: Task, task_graph: TaskGraph) -> None:
        """Execute a single task with retry logic."""
        task.status = TaskStatus.IN_PROGRESS
        agent_name = ""
        task_start = time.monotonic()

        if self._live:
            self._live.update_task(task.task_id, task.description, "in_progress")
            self._live.log(f"[cyan]Starting:[/cyan] {task.description}")

        # For FIX_CODE tasks, pull review findings from the review task's result.
        # If the review passed, skip the fix entirely — no LLM call needed.
        if task.task_type == TaskType.FIX_CODE and "review_task_id" in task.metadata:
            review_task = task_graph.get_task(task.metadata["review_task_id"])
            if review_task and review_task.result:
                if review_task.result.metrics.get("passed", False):
                    logger.info(
                        "FIX_CODE task %d skipped — review passed for %s",
                        task.task_id, task.file,
                    )
                    task_graph.mark_completed(task.task_id)
                    self._metrics["tasks_completed"] += 1
                    if self._live:
                        self._live.update_task(
                            task.task_id, task.description, "completed",
                        )
                    return
                task.metadata["review_errors"] = review_task.result.errors
                task.metadata["review_output"] = review_task.result.output

        # Acquire a per-file lock for write tasks to prevent concurrent
        # modifications to the same file from racing.  Read-only tasks
        # (REVIEW_*, SECURITY_SCAN) also acquire the lock so they never read
        # a file while it is being written by another agent.
        file_lock = (
            self._file_locks.lock_for(task.file)
            if task.file and task.file != "*"
            else None
        )

        for attempt in range(task.max_retries):
            # Rebuild context on EVERY attempt so the agent sees the current
            # state of the filesystem, not a stale snapshot from before the
            # first (possibly failed) attempt.
            context_builder = ContextBuilder(
                workspace_dir=self.repo.workspace,
                blueprint=self.blueprint,
                repo_index=self.repo.get_repo_index(),
                dep_store=self._dep_store,
                embedding_store=self._embedding_store,
            )
            context = context_builder.build(task)

            try:
                agent = self._create_agent(task.task_type)
                agent_name = agent.role.value
                task.assigned_agent = f"{agent_name}-{id(agent)}"

                if self._live:
                    self._live.update_task(
                        task.task_id, task.description, "in_progress", agent_name,
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

                # Track agent metrics
                if agent_name not in self._metrics["agent_metrics"]:
                    self._metrics["agent_metrics"][agent_name] = []
                self._metrics["agent_metrics"][agent_name].append(agent.get_metrics())

                if result.success:
                    elapsed = time.monotonic() - task_start
                    task_graph.mark_completed(task.task_id)
                    self._metrics["tasks_completed"] += 1
                    record_task_completion(task.task_type.value, "completed", elapsed)
                    logger.info(f"Task {task.task_id} completed: {result.output}")
                    # Incremental embedding update — keep vector index fresh
                    # so subsequent agents' semantic searches see new content.
                    if self._embedding_store and result.files_modified:
                        for fp in result.files_modified:
                            try:
                                content = self.repo.read_file(fp)
                                self._embedding_store.index_file(fp, content)
                            except Exception:
                                logger.debug("Embedding update skipped for %s", fp)
                            if self._event_bus:
                                await self._event_bus.publish(AgentEvent(
                                    type=BusEventType.FILE_WRITTEN,
                                    task_id=task.task_id,
                                    task_type=task.task_type.value,
                                    file_path=fp,
                                    agent_name=agent_name,
                                ))
                    if self._event_bus:
                        await self._event_bus.publish(AgentEvent(
                            type=BusEventType.TASK_COMPLETED,
                            task_id=task.task_id,
                            task_type=task.task_type.value,
                            file_path=task.file,
                            agent_name=agent_name,
                        ))
                    if self._live:
                        self._live.update_task(
                            task.task_id, task.description, "completed", agent_name,
                        )
                        self._live.log(f"[green]Done:[/green] {task.description}")
                    return
                else:
                    logger.warning(
                        f"Task {task.task_id} attempt {attempt + 1} failed: {result.errors}"
                    )
                    if self._live:
                        self._live.log(
                            f"[yellow]Retry {attempt + 1}:[/yellow] {task.description}"
                        )
                    task.retry_count += 1

            except Exception as e:
                logger.exception(f"Task {task.task_id} error on attempt {attempt + 1}")
                task.retry_count += 1

        # All retries exhausted
        elapsed = time.monotonic() - task_start
        task_graph.mark_failed(task.task_id)
        self._metrics["tasks_failed"] += 1
        record_task_completion(task.task_type.value, "failed", elapsed)
        logger.error(f"Task {task.task_id} failed after {task.max_retries} attempts")
        if self._event_bus:
            await self._event_bus.publish(AgentEvent(
                type=BusEventType.TASK_FAILED,
                task_id=task.task_id,
                task_type=task.task_type.value,
                file_path=task.file,
                agent_name=agent_name,
            ))
        if self._live:
            self._live.update_task(task.task_id, task.description, "failed", agent_name)
            self._live.log(f"[red]Failed:[/red] {task.description}")

    # ── Lifecycle-mode execution ─────────────────────────────────────

    async def execute_with_lifecycle(
        self,
        engine: LifecycleEngine,
        global_graph: TaskGraph,
    ) -> dict[str, Any]:
        """Execute using the event-sourced lifecycle engine.

        Phase 1: Run per-file lifecycles (Generate → Review → Fix → Test)
                 driven by LifecycleEngine events.
        Phase 2: Mark sentinel task completed, then execute the global DAG
                 (security, module review, architecture review, deploy, docs).

        Returns the same result format as ``execute_graph()``.
        """
        start_time = time.monotonic()
        semaphore = asyncio.Semaphore(self.settings.max_concurrent_agents)
        phase_timeout: float = float(self.settings.phase_timeout_seconds)

        # in_flight      — file paths whose current phase is actively executing.
        # running_tasks  — the asyncio.Task objects backing those coroutines.
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
                    self._metrics["tasks_failed"] += 1
                finally:
                    in_flight.discard(path)

        while not engine.all_terminal():
            # Start a task for every file that just became actionable.
            actionable = engine.get_actionable_files()
            to_dispatch = [
                (path, phase) for path, phase in actionable
                if path not in in_flight
            ]
            for path, phase in to_dispatch:
                in_flight.add(path)
                t = asyncio.create_task(
                    run_file_phase(path, phase), name=f"lifecycle:{path}",
                )
                running_tasks.add(t)
                t.add_done_callback(running_tasks.discard)

            if not running_tasks:
                # Nothing running and nothing new — genuine stall (e.g. dep cycle).
                logger.error(
                    "Lifecycle stall: no actionable files and none in-flight. "
                    "Phase counts: %s", engine.get_stats(),
                )
                break

            # Wait for any single task to finish before re-evaluating the engine.
            # FIRST_COMPLETED means a file whose deps just cleared can start
            # immediately rather than waiting for the whole current batch to drain.
            await asyncio.wait(running_tasks, return_when=asyncio.FIRST_COMPLETED)

        # Cancel any tasks still in flight (only reached after a stall break).
        for t in list(running_tasks):
            t.cancel()
        if running_tasks:
            await asyncio.gather(*list(running_tasks), return_exceptions=True)

        # ── Phase 2: Global DAG ──────────────────────────────────────
        logger.info("All file lifecycles terminal — running global DAG")

        # Mark the sentinel task as completed to unblock the global DAG.
        # Look it up by metadata tag rather than hard-coding ID=1 so the lookup
        # remains correct even if the task graph changes ordering.
        sentinel = next(
            (t for t in global_graph.tasks.values() if t.metadata.get("sentinel")),
            None,
        )
        if sentinel:
            global_graph.mark_completed(sentinel.task_id)
        else:
            logger.warning("Sentinel task not found in global DAG — global phases may stall")

        # Reuse the existing graph executor for the global tasks
        await self.execute_graph(global_graph)

        elapsed = time.monotonic() - start_time

        # Save final repo index
        self.repo.save_repo_index()
        try:
            self.repo.rebuild_dependency_graph()
        except Exception as e:
            logger.warning(f"Dependency graph rebuild failed (non-critical): {e}")

        lifecycle_summary = engine.get_results_summary()
        global_stats = global_graph.get_stats()

        logger.info(
            "Lifecycle execution complete in %.1fs. Files: %d passed, %d failed. "
            "Global tasks: %s",
            elapsed, lifecycle_summary["passed"], lifecycle_summary["failed"],
            global_stats,
        )

        return {
            "stats": {
                **global_stats,
                "lifecycle_passed": lifecycle_summary["passed"],
                "lifecycle_failed": lifecycle_summary["failed"],
                "lifecycle_total_fixes": lifecycle_summary["total_fix_cycles"],
                # Files whose code is valid but whose tests didn't fully pass.
                "lifecycle_tests_degraded": lifecycle_summary["tests_degraded"],
            },
            "metrics": self._metrics,
            "elapsed_seconds": elapsed,
            "lifecycle_summary": lifecycle_summary,
        }

    async def _execute_lifecycle_phase(
        self,
        engine: LifecycleEngine,
        file_path: str,
        phase: FilePhase,
    ) -> None:
        """Execute one lifecycle phase for a single file.

        Maps FilePhase → agent type, builds context, runs the agent, then
        processes the result as an EventType to drive the state machine.
        """
        lc = engine.get_lifecycle(file_path)

        # ── PENDING: fire DEPS_MET to move to GENERATING ──────────
        if phase == FilePhase.PENDING:
            engine.process_event(file_path, EventType.DEPS_MET)
            phase = lc.phase  # now GENERATING

        # Map phase → task type + event on success/failure
        phase_config: dict[FilePhase, dict[str, Any]] = {
            FilePhase.GENERATING: {
                "task_type": TaskType.GENERATE_FILE,
                "success_event": EventType.CODE_GENERATED,
                "failure_event": EventType.RETRIES_EXHAUSTED,
                "description": f"Generate {file_path}",
            },
            FilePhase.REVIEWING: {
                "task_type": TaskType.REVIEW_FILE,
                "success_event": EventType.REVIEW_PASSED,
                "failure_event": EventType.REVIEW_FAILED,
                "description": f"Review {file_path}",
            },
            FilePhase.FIXING: {
                "task_type": TaskType.FIX_CODE,
                "success_event": EventType.FIX_APPLIED,
                "failure_event": EventType.RETRIES_EXHAUSTED,
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

        # Build a synthetic Task for the context builder
        task = Task(
            task_id=0,
            task_type=config["task_type"],
            file=file_path,
            description=config["description"],
            metadata=self._build_lifecycle_metadata(lc),
        )

        if self._live:
            self._live.log(f"[cyan]Lifecycle:[/cyan] {config['description']}")

        # Build context and execute
        context_builder = ContextBuilder(
            workspace_dir=self.repo.workspace,
            blueprint=self.blueprint,
            repo_index=self.repo.get_repo_index(),
            dep_store=self._dep_store,
            embedding_store=self._embedding_store,
        )
        context = context_builder.build(task)

        try:
            agent = self._create_agent(config["task_type"])
            record_agent_start()
            try:
                result = await agent.execute(context)
            finally:
                record_agent_end()

            # Track metrics
            agent_name = agent.role.value
            if agent_name not in self._metrics["agent_metrics"]:
                self._metrics["agent_metrics"][agent_name] = []
            self._metrics["agent_metrics"][agent_name].append(agent.get_metrics())

            # Determine success/failure for lifecycle events.
            # Review is special: ReviewerAgent always returns success=True (task
            # completed), but uses result.metrics["passed"] to indicate the
            # actual review verdict.
            if config["task_type"] == TaskType.REVIEW_FILE:
                # Treat a task-level failure (e.g. JSON parse error) as a
                # review failure rather than silently passing the file through.
                # Also change the default to False so an empty metrics dict
                # (result of a ValidationError) doesn't produce a false pass.
                review_passed = result.success and result.metrics.get("passed", False)
                if review_passed:
                    event_data = {"output": result.output}
                    engine.process_event(file_path, EventType.REVIEW_PASSED, event_data)
                    if self._event_bus:
                        await self._event_bus.publish(AgentEvent(
                            type=BusEventType.REVIEW_PASSED,
                            task_type=config["task_type"].value,
                            file_path=file_path,
                            agent_name=agent_name,
                        ))
                else:
                    event_data = {"findings": result.errors, "output": result.output}
                    engine.process_event(file_path, EventType.REVIEW_FAILED, event_data)
                    if self._event_bus:
                        await self._event_bus.publish(AgentEvent(
                            type=BusEventType.REVIEW_FAILED,
                            task_type=config["task_type"].value,
                            file_path=file_path,
                            agent_name=agent_name,
                            data={"findings": result.errors},
                        ))
                self._metrics["tasks_completed"] += 1
            elif result.success:
                self._metrics["tasks_completed"] += 1
                event_data = self._extract_event_data(result, config["task_type"])
                engine.process_event(file_path, config["success_event"], event_data)
                logger.info("[%s] %s succeeded", file_path, phase.value)
                # Incremental embedding update for lifecycle path
                if self._embedding_store and result.files_modified:
                    for fp in result.files_modified:
                        try:
                            content = self.repo.read_file(fp)
                            self._embedding_store.index_file(fp, content)
                        except Exception:
                            logger.debug("Embedding update skipped for %s", fp)
                        if self._event_bus:
                            await self._event_bus.publish(AgentEvent(
                                type=BusEventType.FILE_WRITTEN,
                                task_type=config["task_type"].value,
                                file_path=fp,
                                agent_name=agent_name,
                            ))
                if self._event_bus:
                    bus_type = (
                        BusEventType.TEST_PASSED
                        if config["task_type"] == TaskType.GENERATE_TEST
                        else BusEventType.TASK_COMPLETED
                    )
                    await self._event_bus.publish(AgentEvent(
                        type=bus_type,
                        task_type=config["task_type"].value,
                        file_path=file_path,
                        agent_name=agent_name,
                    ))
                if self._live:
                    self._live.log(f"[green]Done:[/green] {config['description']}")
            else:
                event_data = self._extract_event_data(result, config["task_type"])
                engine.process_event(file_path, config["failure_event"], event_data)
                logger.warning("[%s] %s failed: %s", file_path, phase.value, result.errors)
                if self._event_bus:
                    bus_type = (
                        BusEventType.TEST_FAILED
                        if config["task_type"] == TaskType.GENERATE_TEST
                        else BusEventType.TASK_FAILED
                    )
                    await self._event_bus.publish(AgentEvent(
                        type=bus_type,
                        task_type=config["task_type"].value,
                        file_path=file_path,
                        agent_name=agent_name,
                        data={"errors": result.errors},
                    ))
                if self._live:
                    self._live.log(
                        f"[yellow]Issue:[/yellow] {config['description']} — "
                        f"transitioning via {config['failure_event'].value}"
                    )

        except Exception as e:
            logger.exception("[%s] %s error", file_path, phase.value)
            engine.process_event(file_path, EventType.RETRIES_EXHAUSTED)
            self._metrics["tasks_failed"] += 1

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
        """Extract event data from agent result for lifecycle transitions."""
        data: dict[str, Any] = {}
        if task_type == TaskType.GENERATE_TEST:
            if not result.success:
                data["errors"] = "\n".join(result.errors) if result.errors else result.output
        if task_type == TaskType.VERIFY_BUILD:
            data["errors"] = "\n".join(result.errors) if result.errors else result.output
        return data
