"""Tests for the context builder."""

import pytest

from core.context_builder import ContextBuilder
from core.models import (
    FileBlueprint,
    RepositoryBlueprint,
    RepositoryIndex,
    FileIndex,
    Task,
    TaskType,
)


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "src" / "models").mkdir(parents=True)
    (ws / "src" / "services").mkdir(parents=True)
    (ws / "src" / "models" / "user.py").write_text("class User: pass\n", encoding="utf-8")
    return ws


@pytest.fixture
def blueprint():
    return RepositoryBlueprint(
        name="test",
        description="test",
        architecture_style="REST",
        file_blueprints=[
            FileBlueprint(path="models/user.py", purpose="User model", layer="model"),
            FileBlueprint(
                path="services/user_service.py",
                purpose="User service",
                depends_on=["models/user.py"],
                layer="service",
            ),
        ],
        architecture_doc="REST API",
    )


@pytest.fixture
def repo_index():
    idx = RepositoryIndex()
    idx.add_or_update(FileIndex(
        path="models/user.py",
        exports=["User"],
        imports=[],
        classes=["User"],
    ))
    return idx


class TestContextBuilder:
    def test_build_context_for_codegen(self, workspace, blueprint, repo_index):
        builder = ContextBuilder(workspace, blueprint, repo_index)
        task = Task(
            task_id=1,
            task_type=TaskType.GENERATE_FILE,
            file="services/user_service.py",
            description="Generate user service",
        )
        ctx = builder.build(task)

        assert ctx.task == task
        assert ctx.file_blueprint is not None
        assert ctx.file_blueprint.path == "services/user_service.py"
        assert "models/user.py" in ctx.related_files
        assert ctx.architecture_summary == "REST API"

    def test_build_context_for_review(self, workspace, blueprint, repo_index):
        builder = ContextBuilder(workspace, blueprint, repo_index)
        task = Task(
            task_id=2,
            task_type=TaskType.REVIEW_FILE,
            file="models/user.py",
            description="Review user model",
        )
        ctx = builder.build(task)
        assert "models/user.py" in ctx.related_files

    def test_build_context_missing_blueprint(self, workspace, blueprint, repo_index):
        builder = ContextBuilder(workspace, blueprint, repo_index)
        task = Task(
            task_id=3,
            task_type=TaskType.GENERATE_FILE,
            file="nonexistent.py",
            description="Generate unknown file",
        )
        ctx = builder.build(task)
        assert ctx.file_blueprint is None
