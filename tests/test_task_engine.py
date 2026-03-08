"""Tests for the task DAG engine."""

import pytest
from core.models import (
    FileBlueprint,
    RepositoryBlueprint,
    Task,
    TaskStatus,
    TaskType,
)
from core.task_engine import TaskGraph, TaskGraphBuilder


class TestTaskGraph:
    def test_add_and_get_task(self):
        graph = TaskGraph()
        task = Task(task_id=1, task_type=TaskType.GENERATE_FILE, file="a.py", description="gen a")
        graph.add_task(task)
        assert graph.get_task(1) == task

    def test_get_ready_tasks_no_deps(self):
        graph = TaskGraph()
        t1 = Task(task_id=1, task_type=TaskType.GENERATE_FILE, file="a.py", description="gen a")
        t2 = Task(task_id=2, task_type=TaskType.GENERATE_FILE, file="b.py", description="gen b")
        graph.add_task(t1)
        graph.add_task(t2)
        ready = graph.get_ready_tasks()
        assert len(ready) == 2

    def test_get_ready_tasks_with_deps(self):
        graph = TaskGraph()
        t1 = Task(task_id=1, task_type=TaskType.GENERATE_FILE, file="a.py", description="gen a")
        t2 = Task(
            task_id=2,
            task_type=TaskType.GENERATE_FILE,
            file="b.py",
            description="gen b",
            dependencies=[1],
        )
        graph.add_task(t1)
        graph.add_task(t2)

        ready = graph.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].task_id == 1

    def test_topological_order(self):
        graph = TaskGraph()
        t1 = Task(task_id=1, task_type=TaskType.GENERATE_FILE, file="a.py", description="a")
        t2 = Task(task_id=2, task_type=TaskType.GENERATE_FILE, file="b.py", description="b", dependencies=[1])
        t3 = Task(task_id=3, task_type=TaskType.GENERATE_FILE, file="c.py", description="c", dependencies=[2])
        graph.add_task(t1)
        graph.add_task(t2)
        graph.add_task(t3)

        order = graph.get_execution_order()
        assert order.index(1) < order.index(2) < order.index(3)

    def test_mark_completed_unblocks_dependents(self):
        graph = TaskGraph()
        t1 = Task(task_id=1, task_type=TaskType.GENERATE_FILE, file="a.py", description="a")
        t2 = Task(task_id=2, task_type=TaskType.GENERATE_FILE, file="b.py", description="b", dependencies=[1])
        graph.add_task(t1)
        graph.add_task(t2)

        graph.mark_completed(1)
        ready = graph.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].task_id == 2

    def test_mark_failed_blocks_downstream(self):
        graph = TaskGraph()
        t1 = Task(task_id=1, task_type=TaskType.GENERATE_FILE, file="a.py", description="a")
        t2 = Task(task_id=2, task_type=TaskType.GENERATE_FILE, file="b.py", description="b", dependencies=[1])
        graph.add_task(t1)
        graph.add_task(t2)

        graph.mark_failed(1)
        assert t2.status == TaskStatus.BLOCKED

    def test_has_remaining_tasks(self):
        graph = TaskGraph()
        t1 = Task(task_id=1, task_type=TaskType.GENERATE_FILE, file="a.py", description="a")
        graph.add_task(t1)
        assert graph.has_remaining_tasks()
        graph.mark_completed(1)
        assert not graph.has_remaining_tasks()

    def test_validate_missing_dependency(self):
        graph = TaskGraph()
        t1 = Task(task_id=2, task_type=TaskType.GENERATE_FILE, file="a.py", description="a", dependencies=[999])
        graph.add_task(t1)
        errors = graph.validate()
        assert len(errors) > 0

    def test_stats(self):
        graph = TaskGraph()
        t1 = Task(task_id=1, task_type=TaskType.GENERATE_FILE, file="a.py", description="a")
        t2 = Task(task_id=2, task_type=TaskType.GENERATE_FILE, file="b.py", description="b")
        graph.add_task(t1)
        graph.add_task(t2)
        graph.mark_completed(1)
        stats = graph.get_stats()
        assert stats["completed"] == 1
        assert stats["pending"] == 1


class TestTaskGraphBuilder:
    def test_build_from_blueprint(self):
        blueprint = RepositoryBlueprint(
            name="test",
            description="test project",
            architecture_style="REST",
            file_blueprints=[
                FileBlueprint(path="models/user.py", purpose="User model", layer="model"),
                FileBlueprint(
                    path="services/user_service.py",
                    purpose="User service",
                    depends_on=["models/user.py"],
                    layer="service",
                ),
                FileBlueprint(
                    path="controllers/user_controller.py",
                    purpose="User controller",
                    depends_on=["services/user_service.py"],
                    layer="controller",
                ),
            ],
        )

        builder = TaskGraphBuilder()
        graph = builder.build_from_blueprint(blueprint)

        # Should have: 3 generate + 3 review + 3 test + 1 security + 1 module review + 1 arch review + 1 deploy + 1 docs
        assert len(graph.tasks) > 3
        errors = graph.validate()
        assert len(errors) == 0

        order = graph.get_execution_order()
        assert len(order) == len(graph.tasks)
