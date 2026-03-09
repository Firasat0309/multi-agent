"""Agent manager - orchestrates task execution across agents."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, TYPE_CHECKING

from agents.base_agent import BaseAgent
from agents.coder_agent import CoderAgent
from agents.deploy_agent import DeployAgent
from agents.reviewer_agent import ReviewerAgent
from agents.security_agent import SecurityAgent
from agents.test_agent import TestAgent
from agents.writer_agent import WriterAgent
from config.settings import Settings
from core.context_builder import ContextBuilder
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
from core.task_engine import TaskGraph
from tools.terminal_tools import TerminalTools

if TYPE_CHECKING:
    from core.live_console import LiveConsole

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
    ) -> None:
        self.settings = settings
        self.llm = llm_client
        self.repo = repo_manager
        self.blueprint = blueprint
        self._live = live_console
        self._lang = detect_language_from_blueprint(blueprint.tech_stack)
        self.terminal = TerminalTools(
            repo_manager.workspace, language=self._lang,
        )
        self._metrics: dict[str, Any] = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_time": 0.0,
            "agent_metrics": {},
        }

    def _create_agent(self, task_type: TaskType) -> BaseAgent:
        agent_cls = TASK_AGENT_MAP.get(task_type)
        if agent_cls is None:
            raise ValueError(f"No agent registered for task type: {task_type}")

        # Some agents need terminal tools
        if agent_cls in (TestAgent, SecurityAgent):
            return agent_cls(
                llm_client=self.llm,
                repo_manager=self.repo,
                terminal=self.terminal,
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

        # For FIX_CODE tasks, pull review findings from the review task's result
        if task.task_type == TaskType.FIX_CODE and "review_task_id" in task.metadata:
            review_task = task_graph.get_task(task.metadata["review_task_id"])
            if review_task and review_task.result:
                task.metadata["review_errors"] = review_task.result.errors
                task.metadata["review_output"] = review_task.result.output

        context_builder = ContextBuilder(
            workspace_dir=self.repo.workspace,
            blueprint=self.blueprint,
            repo_index=self.repo.get_repo_index(),
        )
        context = context_builder.build(task)

        for attempt in range(task.max_retries):
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
        if self._live:
            self._live.update_task(task.task_id, task.description, "failed", agent_name)
            self._live.log(f"[red]Failed:[/red] {task.description}")
