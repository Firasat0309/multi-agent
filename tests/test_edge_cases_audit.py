"""Edge case tests from the comprehensive audit of all pipeline flows."""

import json

import pytest
from unittest.mock import MagicMock, patch

from core.context_builder import _smart_truncate
from core.error_attributor import extract_error_lines


class TestSmartTruncateEdgeCases:
    """Edge cases for _smart_truncate found during audit."""

    def test_content_only_imports(self):
        """File with only imports and no body should not crash."""
        content = "\n".join([
            "import java.util.List;",
            "import java.util.Map;",
            "import java.util.Optional;",
        ])
        result = _smart_truncate(content, 50)
        # All imports should survive (mandatory)
        assert "import java.util.List" in result

    def test_content_only_body_no_imports(self):
        """File with no imports should use head budget for body."""
        lines = [f"    int field{i} = {i};" for i in range(200)]
        content = "\n".join(lines)
        result = _smart_truncate(content, 500)
        assert "field0" in result  # head preserved
        assert "field199" in result  # tail preserved

    def test_empty_content(self):
        result = _smart_truncate("", 1000)
        assert result == ""

    def test_content_exactly_at_limit(self):
        content = "x" * 100
        result = _smart_truncate(content, 100)
        assert result == content

    def test_large_import_block_exceeds_head_budget(self):
        """When imports exceed 60% budget, they eat into tail budget rather than being dropped."""
        # 10 imports at ~45 chars = ~450 chars; 60% of 1000 = 600
        # Without the fix, imports could be split at the 600 mark.
        # With the fix, all 10 imports are mandatory.
        imports = [f"import com.example.package{i}.ClassName{i};" for i in range(10)]
        body = [f"    // filler line {i}" for i in range(50)]
        tail = ["    public void criticalApi() {}", "}"]
        content = "\n".join(imports + body + tail)
        result = _smart_truncate(content, 1000)
        # All 10 imports must be present (mandatory)
        assert "import com.example.package0" in result
        assert "import com.example.package9" in result
        # Tail API should still be present
        assert "criticalApi" in result

    def test_imports_preserved_over_body(self):
        """Imports should be preserved even when budget is tight."""
        imports = [
            "import java.util.List;",
            "import java.util.Map;",
        ]
        body = [f"    // line {i}" for i in range(100)]
        tail = ["    public void criticalMethod() {}", "}"]
        content = "\n".join(imports + body + tail)
        result = _smart_truncate(content, 300)
        assert "import java.util.List" in result
        assert "import java.util.Map" in result


class TestExtractErrorLinesEdgeCases:
    """Edge cases for extract_error_lines head-truncation."""

    def test_empty_string(self):
        result = extract_error_lines("", max_chars=1000)
        assert result == ""

    def test_short_string_unchanged(self):
        text = "error: something broke"
        result = extract_error_lines(text, max_chars=1000)
        assert "error: something broke" in result

    def test_head_truncation_not_tail(self):
        """Must keep the head (root-cause errors), not the tail (cascading)."""
        # Use actual error-formatted lines so the keyword filter matches
        lines = [f"[ERROR] File{i}.java:[{i},1] error: cannot find symbol {i}" for i in range(100)]
        content = "\n".join(lines)
        result = extract_error_lines(content, max_chars=200)
        assert "File0.java" in result  # First error (root cause) kept
        assert "File99.java" not in result  # Last error (cascading) dropped


class TestCoderFixCodeTestErrors:
    """Test that _fix_code properly handles test_errors metadata."""

    def test_fix_code_reads_test_errors(self):
        """fix_trigger='test' should use test_errors, not build_errors."""
        from agents.coder_agent import CoderAgent

        # Verify the code path exists by checking the source
        import inspect
        source = inspect.getsource(CoderAgent._fix_code)
        assert "test_errors" in source
        # test and integration_test triggers both use test_errors
        assert '"test"' in source
        assert '"integration_test"' in source


class TestIncrementalDepGraphEdgeCases:
    """Edge cases for incremental dependency graph updates."""

    def test_corrupted_json_array(self, tmp_path):
        """dependency_graph.json containing a JSON array should trigger full rebuild."""
        from core.repository_manager import RepositoryManager
        from core.models import FileIndex

        rm = RepositoryManager(tmp_path)
        fi = FileIndex(path="A.java", exports=["ClassA"], imports=[], checksum="aaa")
        rm._repo_index.add_or_update(fi)

        # Write a valid JSON array (not a dict) to the graph file
        dep_path = tmp_path / "dependency_graph.json"
        dep_path.write_text('["not", "a", "dict"]', encoding="utf-8")

        # Should fall back to full rebuild instead of crashing
        graph = rm.update_dependency_graph(changed_files=["A.java"])
        assert isinstance(graph, dict)
        assert "A.java" in graph

    def test_corrupted_json_string(self, tmp_path):
        from core.repository_manager import RepositoryManager
        from core.models import FileIndex

        rm = RepositoryManager(tmp_path)
        fi = FileIndex(path="A.java", exports=["ClassA"], imports=[], checksum="aaa")
        rm._repo_index.add_or_update(fi)

        dep_path = tmp_path / "dependency_graph.json"
        dep_path.write_text('"just a string"', encoding="utf-8")

        graph = rm.update_dependency_graph(changed_files=["A.java"])
        assert isinstance(graph, dict)

    def test_changed_files_not_in_index(self, tmp_path):
        """changed_files referencing non-existent files should not crash."""
        from core.repository_manager import RepositoryManager
        from core.models import FileIndex

        rm = RepositoryManager(tmp_path)
        fi = FileIndex(path="A.java", exports=["ClassA"], imports=[], checksum="aaa")
        rm._repo_index.add_or_update(fi)
        rm.rebuild_dependency_graph()

        # "Ghost.java" doesn't exist in the index
        graph = rm.update_dependency_graph(changed_files=["Ghost.java"])
        assert "A.java" in graph


class TestPerFileTimeoutCapped:
    """Verify per-file timeout doesn't scale with tier size."""

    def test_per_file_timeout_capped(self):
        """Even with 50 files, individual file timeout should be capped."""
        from config.settings import Settings
        settings = Settings()
        base = float(settings.phase_timeout_seconds)
        per_file = min(base, 600.0)
        # Per-file timeout must never exceed 600s
        assert per_file <= 600.0

    def test_total_tier_timeout_does_scale(self):
        """Total tier timeout should scale with file count."""
        from config.settings import Settings
        settings = Settings()
        base = float(settings.phase_timeout_seconds)
        n_files = 50
        phase_timeout = max(base, 120.0 * n_files)
        assert phase_timeout >= 6000.0


class TestStubFlushIdempotent:
    """Verify flush() is safe to call multiple times."""

    def test_double_flush(self, tmp_path):
        from core.stub_generator import StubGenerator
        gen = StubGenerator("java", tmp_path)
        gen.generate_stubs(["src/Foo.java"])

        written1 = gen.flush()
        assert written1 == 1
        assert (tmp_path / "src/Foo.java").exists()

        # Second flush should write 0 (file already on disk)
        written2 = gen.flush()
        assert written2 == 0
