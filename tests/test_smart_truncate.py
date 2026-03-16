"""Tests for smart truncation and timeout scaling."""

import pytest
from core.context_builder import _smart_truncate


class TestSmartTruncate:
    def test_small_file_unchanged(self):
        content = "import foo\n\nclass Bar:\n    pass\n"
        result = _smart_truncate(content, 8000)
        assert result == content

    def test_preserves_imports(self):
        lines = ["import java.util.List;", "import java.util.Optional;", ""]
        lines += [f"// line {i}" for i in range(500)]
        content = "\n".join(lines)
        result = _smart_truncate(content, 200)
        assert "import java.util.List" in result

    def test_preserves_tail(self):
        """Critical methods at end of file must survive truncation."""
        lines = ["package com.example;", ""]
        lines += [f"    String field{i};" for i in range(200)]
        lines += ["    void criticalMethod();", "}"]
        content = "\n".join(lines)
        result = _smart_truncate(content, 1000)
        assert "criticalMethod" in result

    def test_omitted_marker_present(self):
        lines = [f"line {i}" for i in range(500)]
        content = "\n".join(lines)
        result = _smart_truncate(content, 500)
        assert "omitted" in result

    def test_respects_budget(self):
        lines = [f"line {i} with some padding text" for i in range(500)]
        content = "\n".join(lines)
        result = _smart_truncate(content, 2000)
        # Allow small overshoot from the marker line
        assert len(result) < 2200

    def test_head_and_tail_no_overlap(self):
        """When content is just over the limit, ensure no duplicate lines."""
        content = "\n".join(f"line {i}" for i in range(100))
        result = _smart_truncate(content, len(content) - 50)
        lines = result.split("\n")
        non_marker = [l for l in lines if "omitted" not in l]
        # No duplicate content lines
        assert len(non_marker) == len(set(non_marker))

    def test_python_imports_preserved(self):
        lines = [
            "from typing import List, Optional",
            "from dataclasses import dataclass",
            "",
        ]
        lines += [f"    field{i}: str" for i in range(200)]
        lines += ["    def critical_method(self) -> None: ..."]
        content = "\n".join(lines)
        result = _smart_truncate(content, 800)
        assert "from typing" in result
        assert "critical_method" in result

    def test_go_imports_preserved(self):
        lines = [
            'package handlers',
            '',
            'import (',
            '    "net/http"',
            ')',
            '',
        ]
        lines += [f"// comment {i}" for i in range(200)]
        lines += ["func CriticalHandler(w http.ResponseWriter) {}"]
        content = "\n".join(lines)
        result = _smart_truncate(content, 800)
        assert "package handlers" in result
        assert "CriticalHandler" in result
