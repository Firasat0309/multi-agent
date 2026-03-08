"""Planner agent - converts a blueprint into an executable task DAG."""

from __future__ import annotations

import logging

from agents.base_agent import BaseAgent
from core.llm_client import LLMClient
from core.models import AgentContext, AgentRole, RepositoryBlueprint, TaskResult
from core.repository_manager import RepositoryManager
from core.task_engine import TaskGraph, TaskGraphBuilder

logger = logging.getLogger(__name__)


class PlannerAgent(BaseAgent):
    role = AgentRole.PLANNER

    async def execute(self, context: AgentContext) -> TaskResult:
        """Not used directly - use create_task_graph instead."""
        return TaskResult(success=False, errors=["Use create_task_graph() method"])

    async def create_task_graph(self, blueprint: RepositoryBlueprint) -> TaskGraph:
        """Build the task DAG from the repository blueprint."""
        logger.info(f"Building task graph for {len(blueprint.file_blueprints)} files")

        builder = TaskGraphBuilder()
        graph = builder.build_from_blueprint(blueprint)

        order = graph.get_execution_order()
        logger.info(f"Task execution order: {order}")
        logger.info(f"Task stats: {graph.get_stats()}")

        return graph
