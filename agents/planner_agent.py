"""Planner agent - converts a blueprint into an executable task DAG."""

from __future__ import annotations

import logging

from agents.base_agent import BaseAgent
from core.models import AgentContext, AgentRole, RepositoryBlueprint, TaskResult
from core.state_machine import LifecycleEngine
from core.task_engine import TaskGraph, LifecyclePlanBuilder

logger = logging.getLogger(__name__)


class PlannerAgent(BaseAgent):
    role = AgentRole.PLANNER

    async def execute(self, context: AgentContext) -> TaskResult:
        """Build a lifecycle plan from context.blueprint."""
        try:
            _, global_graph = await self.create_lifecycle_plan(context.blueprint)
            stats = global_graph.get_stats()
            return TaskResult(
                success=True,
                output=f"Plan created for {len(context.blueprint.file_blueprints)} files. Global tasks: {stats}",
                metrics=self.get_metrics(),
            )
        except Exception as exc:
            logger.exception("PlannerAgent.execute failed")
            return TaskResult(success=False, errors=[str(exc)])

    async def create_task_graph(self, blueprint: RepositoryBlueprint) -> TaskGraph:
        """Build the task graph via the lifecycle plan builder.

        Delegates to ``create_lifecycle_plan()`` and returns the global task
        graph.  Kept for backward compatibility with code that calls this method.
        """
        _, global_graph = await self.create_lifecycle_plan(blueprint)
        return global_graph

    async def create_lifecycle_plan(
        self,
        blueprint: RepositoryBlueprint,
        *,
        max_review_fixes: int = 3,
        max_test_fixes: int = 3,
    ) -> tuple[LifecycleEngine, TaskGraph]:
        """Build an event-sourced lifecycle engine + global task graph.

        Returns:
            (lifecycle_engine, global_task_graph)
        """
        logger.info(
            "Building lifecycle plan for %d files (max_review_fixes=%d, max_test_fixes=%d)",
            len(blueprint.file_blueprints), max_review_fixes, max_test_fixes,
        )

        builder = LifecyclePlanBuilder()
        engine, global_graph = builder.build(
            blueprint,
            max_review_fixes=max_review_fixes,
            max_test_fixes=max_test_fixes,
        )

        order = global_graph.get_execution_order()
        logger.info(f"Global task execution order: {order}")
        logger.info(f"Global task stats: {global_graph.get_stats()}")

        return engine, global_graph
