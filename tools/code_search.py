"""Code search tools for agents to explore the repository."""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from pathlib import Path

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
