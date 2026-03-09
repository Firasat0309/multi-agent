"""Planner agent - converts a blueprint into an executable task DAG."""

from __future__ import annotations

import logging

from agents.base_agent import BaseAgent
from core.llm_client import LLMClient
from core.models import AgentContext, AgentRole, RepositoryBlueprint, TaskResult
from core.repository_manager import RepositoryManager
from core.state_machine import LifecycleEngine
from core.task_engine import TaskGraph, TaskGraphBuilder, LifecyclePlanBuilder

logger = logging.getLogger(__name__)


class PlannerAgent(BaseAgent):
    role = AgentRole.PLANNER

    async def execute(self, context: AgentContext) -> TaskResult:
        """Not used directly - use create_task_graph instead."""
        return TaskResult(success=False, errors=["Use create_task_graph() method"])

    async def create_task_graph(self, blueprint: RepositoryBlueprint) -> TaskGraph:
        """Build the legacy task DAG from the repository blueprint."""
        logger.info(f"Building task graph for {len(blueprint.file_blueprints)} files")

        builder = TaskGraphBuilder()
        graph = builder.build_from_blueprint(blueprint)

        order = graph.get_execution_order()
        logger.info(f"Task execution order: {order}")
        logger.info(f"Task stats: {graph.get_stats()}")

        return graph

    async def create_lifecycle_plan(
        self,
        blueprint: RepositoryBlueprint,
        *,
        max_review_fixes: int = 2,
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
