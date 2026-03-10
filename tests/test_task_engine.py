"""Tests for the task DAG engine."""

import pytest
from core.models import (
    FileBlueprint,
    RepositoryBlueprint,
    Task,
    TaskStatus,
    TaskType,
)
from core.task_engine import TaskGraph, TaskGraphBuilder, LifecyclePlanBuilder


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
    """TaskGraphBuilder no longer provides build_from_blueprint() — tests migrated to LifecyclePlanBuilder."""

    def test_builder_instantiation(self):
        """TaskGraphBuilder can still be instantiated (used internally by ModificationTaskGraphBuilder)."""
        builder = TaskGraphBuilder()
        assert builder is not None

    def test_lifecycle_replaces_legacy_build_from_blueprint(self):
        """LifecyclePlanBuilder.build() replaces the removed build_from_blueprint()."""
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

        builder = LifecyclePlanBuilder()
        engine, global_graph = builder.build(blueprint)

        # Engine has all 3 files
        assert engine.has_file("models/user.py")
        assert engine.has_file("services/user_service.py")
        assert engine.has_file("controllers/user_controller.py")

        # Global graph is valid
        assert len(global_graph.tasks) > 3
        errors = global_graph.validate()
        assert len(errors) == 0

        order = global_graph.get_execution_order()
        assert len(order) == len(global_graph.tasks)


class TestLifecyclePlanBuilder:
    def _make_blueprint(self) -> RepositoryBlueprint:
        return RepositoryBlueprint(
            name="test",
            description="test project",
            architecture_style="REST",
            file_blueprints=[
                FileBlueprint(path="models/User.java", purpose="User entity", layer="model"),
                FileBlueprint(
                    path="repos/UserRepo.java",
                    purpose="User repository",
                    depends_on=["models/User.java"],
                    layer="repository",
                ),
                FileBlueprint(
                    path="services/UserService.java",
                    purpose="User service",
                    depends_on=["repos/UserRepo.java"],
                    layer="service",
                ),
                FileBlueprint(path="pom.xml", purpose="Maven config", layer="config"),
            ],
        )

    def test_build_returns_engine_and_graph(self):
        builder = LifecyclePlanBuilder()
        engine, global_graph = builder.build(self._make_blueprint())

        # Engine should have 4 files
        assert engine.has_file("models/User.java")
        assert engine.has_file("repos/UserRepo.java")
        assert engine.has_file("services/UserService.java")
        assert engine.has_file("pom.xml")

        # Global graph should have: sentinel + security + module review
        # + module fixes (3, config excluded) + arch review + deploy + docs
        assert len(global_graph.tasks) > 5
        assert len(global_graph.validate()) == 0

    def test_config_files_skip_testing(self):
        builder = LifecyclePlanBuilder()
        engine, _ = builder.build(self._make_blueprint())

        from core.state_machine import EventType, FilePhase

        # pom.xml should auto-pass testing
        engine.process_event("pom.xml", EventType.DEPS_MET)
        engine.process_event("pom.xml", EventType.CODE_GENERATED)
        engine.process_event("pom.xml", EventType.REVIEW_PASSED)

        lc = engine.get_lifecycle("pom.xml")
        assert lc.phase == FilePhase.PASSED

    def test_file_deps_respected(self):
        builder = LifecyclePlanBuilder()
        engine, _ = builder.build(self._make_blueprint())

        actionable = engine.get_actionable_files()
        paths = [p for p, _ in actionable]

        # models/User.java and pom.xml have no deps → actionable
        assert "models/User.java" in paths
        assert "pom.xml" in paths
        # services/UserService.java depends on repos → not actionable
        assert "services/UserService.java" not in paths

    def test_sentinel_task_exists(self):
        builder = LifecyclePlanBuilder()
        _, global_graph = builder.build(self._make_blueprint())

        sentinel = global_graph.get_task(1)
        assert sentinel is not None
        assert "sentinel" in sentinel.description.lower()

    def test_global_graph_has_all_phases(self):
        builder = LifecyclePlanBuilder()
        _, global_graph = builder.build(self._make_blueprint())

        task_types = {t.task_type for t in global_graph.tasks.values()}
        assert TaskType.SECURITY_SCAN in task_types
        assert TaskType.REVIEW_MODULE in task_types
        assert TaskType.REVIEW_ARCHITECTURE in task_types
        assert TaskType.GENERATE_DEPLOY in task_types
        assert TaskType.GENERATE_DOCS in task_types
        assert TaskType.FIX_CODE in task_types  # module fix tasks

    def test_custom_fix_limits(self):
        builder = LifecyclePlanBuilder()
        engine, _ = builder.build(
            self._make_blueprint(),
            max_review_fixes=5,
            max_test_fixes=7,
        )

        lc = engine.get_lifecycle("models/User.java")
        assert lc.max_review_fixes == 5
        assert lc.max_test_fixes == 7
