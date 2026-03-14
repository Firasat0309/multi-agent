"""Tests for repository manager."""

import pytest

from core.models import FileBlueprint, RepositoryBlueprint
from core.repository_manager import RepositoryManager


@pytest.fixture
def workspace(tmp_path):
    return tmp_path / "workspace"


@pytest.fixture
def repo_manager(workspace):
    return RepositoryManager(workspace)


@pytest.fixture
def blueprint():
    return RepositoryBlueprint(
        name="test-project",
        description="Test project",
        architecture_style="REST",
        tech_stack={"db": "postgresql"},
        folder_structure=["models", "services", "controllers"],
        file_blueprints=[
            FileBlueprint(path="models/user.py", purpose="User model", layer="model"),
        ],
        architecture_doc="# Test Architecture\nREST API with PostgreSQL",
    )


class TestRepositoryManager:
    def test_initialize_creates_structure(self, repo_manager, blueprint, workspace):
        repo_manager.initialize(blueprint)
        assert (workspace / "src" / "models").exists()
        assert (workspace / "src" / "services").exists()
        assert (workspace / "src" / "controllers").exists()
        assert (workspace / "architecture.md").exists()
        assert (workspace / "file_blueprints.json").exists()
        assert (workspace / "dependency_graph.json").exists()

    def test_write_and_read_file(self, repo_manager, blueprint, workspace):
        repo_manager.initialize(blueprint)
        content = "class User:\n    pass\n"
        repo_manager.write_file("models/user.py", content)
        assert repo_manager.read_file("models/user.py") == content

    def test_write_creates_init_files(self, repo_manager, blueprint, workspace):
        repo_manager.initialize(blueprint)
        repo_manager.write_file("models/user.py", "class User: pass\n")
        assert (workspace / "src" / "models" / "__init__.py").exists()

    def test_list_files(self, repo_manager, blueprint, workspace):
        repo_manager.initialize(blueprint)
        repo_manager.write_file("models/user.py", "class User: pass\n")
        repo_manager.write_file("services/user_service.py", "def create_user(): pass\n")
        files = repo_manager.list_files()
        assert len(files) >= 2

    def test_search_code(self, repo_manager, blueprint, workspace):
        repo_manager.initialize(blueprint)
        repo_manager.write_file("models/user.py", "class User:\n    name: str\n")
        results = repo_manager.search_code("User")
        assert len(results) > 0
        assert results[0]["file"] == "models/user.py"

    def test_indexing(self, repo_manager, blueprint, workspace):
        repo_manager.initialize(blueprint)
        repo_manager.write_file(
            "models/user.py",
            "from pydantic import BaseModel\n\nclass User(BaseModel):\n    name: str\n",
        )
        index = repo_manager.get_repo_index()
        fi = index.get_file("models/user.py")
        assert fi is not None
        assert "User" in fi.classes
        assert "User" in fi.exports

    def test_read_nonexistent_file(self, repo_manager, blueprint):
        repo_manager.initialize(blueprint)
        assert repo_manager.read_file("nonexistent.py") is None

    def test_save_repo_index(self, repo_manager, blueprint, workspace):
        repo_manager.initialize(blueprint)
        repo_manager.write_file("models/user.py", "class User: pass\n")
        repo_manager.save_repo_index()
        assert (workspace / "repo_index.json").exists()

    def test_path_traversal_blocked(self, repo_manager, blueprint):
        """write_file must raise ValueError for paths that escape the workspace."""
        repo_manager.initialize(blueprint)
        import pytest
        with pytest.raises(ValueError, match="Path traversal blocked"):
            repo_manager.write_file("../../etc/passwd", "malicious")

    def test_path_traversal_blocked_absolute(self, repo_manager, blueprint):
        """Absolute paths that escape the workspace are also rejected."""
        import pytest
        repo_manager.initialize(blueprint)
        with pytest.raises((ValueError, Exception)):
            repo_manager.write_file("/etc/passwd", "malicious")

    def test_normal_write_not_blocked(self, repo_manager, blueprint):
        """Legitimate nested writes must still succeed."""
        repo_manager.initialize(blueprint)
        repo_manager.write_file("models/deep/nested/file.py", "x = 1\n")
        assert repo_manager.read_file("models/deep/nested/file.py") == "x = 1\n"
