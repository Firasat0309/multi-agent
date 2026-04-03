"""Core task DAG data structure with topological execution ordering.

Extracted from ``task_engine.py`` to keep the DAG data structure decoupled
from the various plan-builder classes that populate it.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import networkx as nx

from core.models import Task, TaskStatus

logger = logging.getLogger(__name__)


class TaskGraph:
    """Manages the task DAG and provides topological execution ordering."""

    def __init__(self) -> None:
        self._graph = nx.DiGraph()
        self._tasks: dict[int, Task] = {}
        self._next_id = 1

    @property
    def tasks(self) -> dict[int, Task]:
        return dict(self._tasks)

    def add_task(self, task: Task) -> None:
        self._tasks[task.task_id] = task
        self._graph.add_node(task.task_id)
        for dep_id in task.dependencies:
            self._graph.add_edge(dep_id, task.task_id)
        self._next_id = max(self._next_id, task.task_id + 1)

    def get_ready_tasks(self) -> list[Task]:
        """Return tasks whose dependencies are all completed."""
        ready = []
        for task_id, task in self._tasks.items():
            if task.status != TaskStatus.PENDING:
                continue
            deps_met = all(
                self._tasks[d].status == TaskStatus.COMPLETED
                for d in task.dependencies
                if d in self._tasks
            )
            if deps_met:
                task.status = TaskStatus.READY
                ready.append(task)
        return ready

    def get_execution_order(self) -> list[int]:
        """Return task IDs in topological order."""
        try:
            return list(nx.topological_sort(self._graph))
        except nx.NetworkXUnfeasible:
            logger.error("Cycle detected in task graph!")
            raise ValueError("Task graph contains a cycle")

    def mark_completed(self, task_id: int) -> None:
        if task_id in self._tasks:
            self._tasks[task_id].status = TaskStatus.COMPLETED

    def mark_failed(self, task_id: int) -> None:
        if task_id in self._tasks:
            task = self._tasks[task_id]
            task.status = TaskStatus.FAILED
            # Block downstream tasks
            for downstream in nx.descendants(self._graph, task_id):
                if downstream in self._tasks:
                    self._tasks[downstream].status = TaskStatus.BLOCKED

    def has_remaining_tasks(self) -> bool:
        return any(
            t.status in (TaskStatus.PENDING, TaskStatus.READY, TaskStatus.IN_PROGRESS)
            for t in self._tasks.values()
        )

    def get_task(self, task_id: int) -> Task | None:
        return self._tasks.get(task_id)

    def get_stats(self) -> dict[str, int]:
        stats: dict[str, int] = defaultdict(int)
        for task in self._tasks.values():
            stats[task.status.value] += 1
        return dict(stats)

    def validate(self) -> list[str]:
        """Validate the task graph for issues."""
        errors: list[str] = []
        if not nx.is_directed_acyclic_graph(self._graph):
            errors.append("Task graph contains cycles")
        for task_id, task in self._tasks.items():
            for dep in task.dependencies:
                if dep not in self._tasks:
                    errors.append(f"Task {task_id} depends on missing task {dep}")
        return errors
