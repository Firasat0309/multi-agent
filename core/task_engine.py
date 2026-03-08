"""Task DAG engine with topological execution ordering."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Iterator

import networkx as nx

from core.models import Task, TaskStatus, TaskType, RepositoryBlueprint, FileBlueprint

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


class TaskGraphBuilder:
    """Builds a task graph from a repository blueprint."""

    def __init__(self) -> None:
        self._next_id = 1

    def _alloc_id(self) -> int:
        tid = self._next_id
        self._next_id += 1
        return tid

    def build_from_blueprint(self, blueprint: RepositoryBlueprint) -> TaskGraph:
        graph = TaskGraph()
        file_task_map: dict[str, int] = {}

        # Phase 1: Generate code files (respecting dependency order)
        for fb in blueprint.file_blueprints:
            deps = [file_task_map[d] for d in fb.depends_on if d in file_task_map]
            task = Task(
                task_id=self._alloc_id(),
                task_type=TaskType.GENERATE_FILE,
                file=fb.path,
                description=f"Generate {fb.path}: {fb.purpose}",
                dependencies=deps,
            )
            graph.add_task(task)
            file_task_map[fb.path] = task.task_id

        # Phase 2: File-level reviews
        review_tasks: list[int] = []
        for fb in blueprint.file_blueprints:
            gen_task_id = file_task_map[fb.path]
            task = Task(
                task_id=self._alloc_id(),
                task_type=TaskType.REVIEW_FILE,
                file=fb.path,
                description=f"Review {fb.path}",
                dependencies=[gen_task_id],
            )
            graph.add_task(task)
            review_tasks.append(task.task_id)

        # Phase 3: Generate tests
        test_tasks: list[int] = []
        for fb in blueprint.file_blueprints:
            if fb.layer in ("test", "config", "deploy"):
                continue
            gen_task_id = file_task_map[fb.path]
            task = Task(
                task_id=self._alloc_id(),
                task_type=TaskType.GENERATE_TEST,
                file=fb.path,
                description=f"Generate tests for {fb.path}",
                dependencies=[gen_task_id],
            )
            graph.add_task(task)
            test_tasks.append(task.task_id)

        # Phase 4: Security scan (after all code generated)
        all_gen_ids = list(file_task_map.values())
        sec_task = Task(
            task_id=self._alloc_id(),
            task_type=TaskType.SECURITY_SCAN,
            file="*",
            description="Security scan of entire codebase",
            dependencies=all_gen_ids,
        )
        graph.add_task(sec_task)

        # Phase 5: Module-level review (after file reviews)
        mod_review = Task(
            task_id=self._alloc_id(),
            task_type=TaskType.REVIEW_MODULE,
            file="*",
            description="Module consistency review",
            dependencies=review_tasks,
        )
        graph.add_task(mod_review)

        # Phase 6: Architecture review
        arch_review = Task(
            task_id=self._alloc_id(),
            task_type=TaskType.REVIEW_ARCHITECTURE,
            file="*",
            description="Architecture review for dependency cycles and layer violations",
            dependencies=[mod_review.task_id, sec_task.task_id],
        )
        graph.add_task(arch_review)

        # Phase 7: Deployment artifacts
        deploy_task = Task(
            task_id=self._alloc_id(),
            task_type=TaskType.GENERATE_DEPLOY,
            file="deploy/",
            description="Generate Dockerfile and Kubernetes manifests",
            dependencies=[arch_review.task_id],
        )
        graph.add_task(deploy_task)

        # Phase 8: Documentation
        docs_task = Task(
            task_id=self._alloc_id(),
            task_type=TaskType.GENERATE_DOCS,
            file="docs/",
            description="Generate README, changelog, API docs",
            dependencies=[arch_review.task_id],
        )
        graph.add_task(docs_task)

        errors = graph.validate()
        if errors:
            raise ValueError(f"Task graph validation failed: {errors}")

        logger.info(f"Built task graph with {len(graph.tasks)} tasks")
        return graph
