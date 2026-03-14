"""Tests for developer tools."""

import pytest

from tools.file_tools import FileTools
from tools.code_search import CodeSearch


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def file_tools(workspace):
    return FileTools(workspace)


@pytest.fixture
def code_search(workspace):
    return CodeSearch(workspace)


class TestFileTools:
    def test_write_and_read(self, file_tools, workspace):
        file_tools.write_file("test.py", "print('hello')")
        content = file_tools.read_file("test.py")
        assert content == "print('hello')"

    def test_read_nonexistent(self, file_tools):
        with pytest.raises(FileNotFoundError):
            file_tools.read_file("nonexistent.py")

    def test_edit_file(self, file_tools):
        file_tools.write_file("test.py", "x = 1\ny = 2\n")
        file_tools.edit_file("test.py", "x = 1", "x = 42")
        content = file_tools.read_file("test.py")
        assert "x = 42" in content

    def test_edit_string_not_found(self, file_tools):
        file_tools.write_file("test.py", "x = 1\n")
        with pytest.raises(ValueError):
            file_tools.edit_file("test.py", "z = 999", "z = 0")

    def test_list_files(self, file_tools, workspace):
        file_tools.write_file("a.py", "a")
        file_tools.write_file("sub/b.py", "b")
        files = file_tools.list_files()
        assert len(files) >= 2

    def test_path_traversal_blocked(self, file_tools):
        with pytest.raises(PermissionError):
            file_tools.read_file("../../etc/passwd")

    def test_file_exists(self, file_tools):
        file_tools.write_file("test.py", "x")
        assert file_tools.file_exists("test.py")
        assert not file_tools.file_exists("missing.py")


class TestCodeSearch:
    def test_search(self, code_search, workspace):
        (workspace / "test.py").write_text("class User:\n    name: str\n", encoding="utf-8")
        results = code_search.search("User")
        assert len(results) > 0
        assert results[0].content == "class User:"

    def test_find_definition(self, code_search, workspace):
        (workspace / "test.py").write_text("def create_user():\n    pass\n", encoding="utf-8")
        results = code_search.find_definition("create_user")
        assert len(results) == 1

    def test_find_usages(self, code_search, workspace):
        (workspace / "a.py").write_text("from b import User\nx = User()\n", encoding="utf-8")
        results = code_search.find_usages("User")
        assert len(results) >= 2

    def test_search_no_results(self, code_search, workspace):
        (workspace / "test.py").write_text("x = 1\n", encoding="utf-8")
        results = code_search.search("nonexistent_symbol_xyz")
        assert len(results) == 0
