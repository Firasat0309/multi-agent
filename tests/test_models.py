"""Tests for core domain models."""

import pytest
from core.models import (
    FileBlueprint,
    FileIndex,
    RepositoryBlueprint,
    RepositoryIndex,
    ReviewFinding,
    ReviewLevel,
    Task,
    TaskResult,
    TaskStatus,
    TaskType,
)


class TestTask:
    def test_task_default_status(self):
        task = Task(task_id=1, task_type=TaskType.GENERATE_FILE, file="test.py", description="test")
        assert task.status == TaskStatus.PENDING
        assert task.retry_count == 0
        assert task.assigned_agent is None

    def test_task_with_dependencies(self):
        task = Task(
            task_id=2,
            task_type=TaskType.GENERATE_FILE,
            file="svc.py",
            description="generate service",
            dependencies=[1],
        )
        assert task.dependencies == [1]


class TestRepositoryIndex:
    def test_add_and_get_file(self):
        index = RepositoryIndex()
        fi = FileIndex(path="models/user.py", exports=["User"], imports=[], classes=["User"])
        index.add_or_update(fi)
        assert index.get_file("models/user.py") == fi

    def test_update_existing_file(self):
        index = RepositoryIndex()
        fi1 = FileIndex(path="svc.py", exports=["foo"])
        index.add_or_update(fi1)
        fi2 = FileIndex(path="svc.py", exports=["foo", "bar"])
        index.add_or_update(fi2)
        assert len(index.files) == 1
        assert index.get_file("svc.py").exports == ["foo", "bar"]

    def test_get_missing_file(self):
        index = RepositoryIndex()
        assert index.get_file("missing.py") is None


class TestRepositoryBlueprint:
    def test_blueprint_creation(self):
        bp = RepositoryBlueprint(
            name="test-project",
            description="A test project",
            architecture_style="REST",
            tech_stack={"db": "postgresql"},
            folder_structure=["models", "services"],
            file_blueprints=[
                FileBlueprint(
                    path="models/user.py",
                    purpose="User model",
                    layer="model",
                ),
            ],
        )
        assert bp.name == "test-project"
        assert len(bp.file_blueprints) == 1
        assert bp.file_blueprints[0].layer == "model"
