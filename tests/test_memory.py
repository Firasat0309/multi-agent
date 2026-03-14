"""Tests for memory systems."""

import pytest

from core.models import FileIndex
from memory.repo_index import RepoIndexStore
from memory.dependency_graph import DependencyGraphStore


@pytest.fixture
def workspace(tmp_path):
    return tmp_path / "workspace"


class TestRepoIndexStore:
    def test_save_and_load(self, workspace):
        workspace.mkdir()
        store = RepoIndexStore(workspace)
        fi = FileIndex(
            path="models/user.py",
            exports=["User"],
            imports=["from pydantic import BaseModel"],
            classes=["User"],
            functions=[],
            checksum="abc123",
        )
        store.update_file(fi)

        # Reload from disk
        store2 = RepoIndexStore(workspace)
        loaded = store2.get_file_info("models/user.py")
        assert loaded is not None
        assert loaded.exports == ["User"]
        assert loaded.checksum == "abc123"

    def test_query_exports(self, workspace):
        workspace.mkdir()
        store = RepoIndexStore(workspace)
        store.update_file(FileIndex(path="a.py", exports=["Foo", "Bar"]))
        store.update_file(FileIndex(path="b.py", exports=["Baz"]))
        assert store.query_exports("Foo") == ["a.py"]
        assert store.query_exports("Baz") == ["b.py"]
        assert store.query_exports("Missing") == []


class TestDependencyGraphStore:
    def test_add_and_query(self, workspace):
        workspace.mkdir()
        store = DependencyGraphStore(workspace)
        store.add_dependency("service.py", "model.py")
        store.add_dependency("controller.py", "service.py")

        assert store.get_dependencies("controller.py") == ["service.py"]
        assert store.get_dependents("service.py") == ["controller.py"]

    def test_transitive_dependencies(self, workspace):
        workspace.mkdir()
        store = DependencyGraphStore(workspace)
        store.add_dependency("controller.py", "service.py")
        store.add_dependency("service.py", "model.py")

        transitive = store.get_transitive_dependencies("controller.py")
        assert "service.py" in transitive
        assert "model.py" in transitive

    def test_detect_cycles(self, workspace):
        workspace.mkdir()
        store = DependencyGraphStore(workspace)
        store.add_dependency("a.py", "b.py")
        store.add_dependency("b.py", "a.py")
        cycles = store.detect_cycles()
        assert len(cycles) > 0

    def test_save_and_load(self, workspace):
        workspace.mkdir()
        store = DependencyGraphStore(workspace)
        store.add_dependency("a.py", "b.py")
        store.save()

        store2 = DependencyGraphStore(workspace)
        assert store2.get_dependencies("a.py") == ["b.py"]

    def test_get_layers(self, workspace):
        workspace.mkdir()
        store = DependencyGraphStore(workspace)
        store.add_dependency("controller.py", "service.py")
        store.add_dependency("service.py", "model.py")
        layers = store.get_layers()
        assert layers["model.py"] == 0
        assert layers["service.py"] > layers["model.py"]
        assert layers["controller.py"] > layers["service.py"]
