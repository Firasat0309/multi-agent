"""Code search tools for agents to explore the repository."""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    file: str
    line: int
    content: str
    context_before: list[str]
    context_after: list[str]


class CodeSearch:
    """Provides code search capabilities over the workspace."""

    def __init__(self, workspace_root: Path) -> None:
        self.root = workspace_root

    def search(
        self,
        query: str,
        file_pattern: str = "**/*",
        max_results: int = 50,
        context_lines: int = 2,
    ) -> list[SearchResult]:
        """Search for a string or regex across files."""
        results: list[SearchResult] = []
        try:
            pattern = re.compile(query)
        except re.error:
            pattern = re.compile(re.escape(query))

        for py_file in self.root.rglob(file_pattern):
            if not py_file.is_file():
                continue
            try:
                lines = py_file.read_text(encoding="utf-8").splitlines()
            except Exception:
                continue

            for i, line in enumerate(lines):
                if pattern.search(line):
                    start = max(0, i - context_lines)
                    end = min(len(lines), i + context_lines + 1)
                    results.append(SearchResult(
                        file=str(py_file.relative_to(self.root)),
                        line=i + 1,
                        content=line.strip(),
                        context_before=lines[start:i],
                        context_after=lines[i + 1:end],
                    ))
                    if len(results) >= max_results:
                        return results
        return results

    def find_definition(self, symbol: str) -> list[SearchResult]:
        """Find class, function, type, or struct definitions across languages."""
        # Covers Python (class/def), Java/C# (class/interface/enum),
        # Go (func/type), TypeScript (class/function/const), Rust (fn/struct/enum)
        pattern = rf"(?:class|def|func|function|type|struct|enum|interface|const|let|pub\s+fn)\s+{re.escape(symbol)}\b"
        return self.search(pattern)

    def find_usages(self, symbol: str) -> list[SearchResult]:
        """Find all usages of a symbol."""
        return self.search(rf"\b{re.escape(symbol)}\b")

    def find_imports(self, module: str) -> list[SearchResult]:
        """Find import statements for a module."""
        return self.search(rf"(from|import)\s+.*{re.escape(module)}")

    def find_function(self, function_name: str) -> list[SearchResult]:
        """Find a specific function or method definition across all files.

        Returns SearchResult with context lines so the caller knows
        the exact file and location.
        """
        pattern = (
            rf"(?:def|func|function|pub\s+fn|pub\s+async\s+fn)\s+{re.escape(function_name)}\s*\("
        )
        return self.search(pattern, context_lines=5)

    def find_class(self, class_name: str) -> list[SearchResult]:
        """Find a specific class/struct/interface definition."""
        pattern = (
            rf"(?:class|struct|interface|enum|record|type)\s+{re.escape(class_name)}\b"
        )
        return self.search(pattern, context_lines=5)

    def find_module(self, module_name: str) -> list[SearchResult]:
        """Find files that constitute a module (by filename or content)."""
        # Search by filename first
        results: list[SearchResult] = []
        for path in self.root.rglob("*"):
            if path.is_file() and module_name in path.stem:
                try:
                    first_line = path.read_text(encoding="utf-8").splitlines()[0]
                except Exception:
                    first_line = ""
                results.append(SearchResult(
                    file=str(path.relative_to(self.root)),
                    line=1,
                    content=first_line.strip(),
                    context_before=[],
                    context_after=[],
                ))
        return results

    def get_file_summary(self, file_path: str, max_lines: int = 10) -> dict[str, Any]:
        """Get a summary of a file: its imports, classes, functions, and first N lines.

        Returns a dict with keys: path, imports, classes, functions, first_lines.
        """
        from typing import Any
        resolved = self.root / file_path
        if not resolved.is_file():
            return {"path": file_path, "error": "File not found"}
        try:
            content = resolved.read_text(encoding="utf-8")
        except Exception:
            return {"path": file_path, "error": "Could not read file"}

        lines = content.splitlines()
        imports: list[str] = []
        classes: list[str] = []
        functions: list[str] = []

        for line in lines:
            stripped = line.strip()
            if re.match(r"^(?:import|from|use|using)\s+", stripped):
                imports.append(stripped)
            m = re.match(r"(?:class|struct|interface|enum|type)\s+(\w+)", stripped)
            if m:
                classes.append(m.group(1))
            m = re.match(r"(?:def|func|function|fn)\s+(\w+)", stripped)
            if m:
                functions.append(m.group(1))

        return {
            "path": file_path,
            "line_count": len(lines),
            "imports": imports,
            "classes": classes,
            "functions": functions,
            "first_lines": lines[:max_lines],
        }
