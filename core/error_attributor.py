"""Compiler error attribution — maps build output to individual source files.

When a repo-level build fails, the raw compiler output contains errors from
potentially many files.  ``ErrorAttributor`` parses that output and returns a
mapping of ``{file_path: [error_messages]}`` so the orchestrator can dispatch
targeted fix tasks to the correct files.

Each language has its own error format:
  - Java/Maven:   ``[ERROR] /path/File.java:[line,col] error: ...``
  - Go:           ``./pkg/file.go:line:col: ...``
  - TypeScript:   ``src/file.ts(line,col): error TS...: ...``
  - Rust/Cargo:   ``error[E0308]: ... --> src/file.rs:line:col``
  - C#/dotnet:    ``File.cs(line,col): error CS...: ...``

The ``LLMErrorAttributor`` provides a fallback that uses the LLM to parse
non-standard or ambiguous compiler output.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import PurePosixPath

logger = logging.getLogger(__name__)


@dataclass
class AttributedError:
    """A single compiler error attributed to a specific file."""

    file_path: str
    line: int | None = None
    column: int | None = None
    message: str = ""
    severity: str = "error"  # "error" | "warning"


@dataclass
class AttributionResult:
    """Complete result of attributing build errors to files."""

    errors_by_file: dict[str, list[AttributedError]] = field(default_factory=dict)
    unattributed_errors: list[str] = field(default_factory=list)

    @property
    def affected_files(self) -> list[str]:
        """Files that have at least one attributed error."""
        return list(self.errors_by_file.keys())

    @property
    def total_errors(self) -> int:
        return sum(len(errs) for errs in self.errors_by_file.values())

    def errors_for_file(self, file_path: str) -> list[str]:
        """Return human-readable error messages for a single file."""
        return [e.message for e in self.errors_by_file.get(file_path, [])]

    def summary_for_file(self, file_path: str) -> str:
        """Return a consolidated error summary for a single file."""
        errors = self.errors_by_file.get(file_path, [])
        if not errors:
            return ""
        lines = []
        for e in errors:
            loc = f":{e.line}" if e.line else ""
            lines.append(f"  {e.file_path}{loc}: {e.message}")
        return "\n".join(lines)


class BaseErrorAttributor(ABC):
    """Abstract base for error attributors."""

    @abstractmethod
    def attribute(
        self,
        build_output: str,
        known_files: set[str] | None = None,
    ) -> AttributionResult:
        """Parse build output and attribute errors to files.

        Args:
            build_output: Raw compiler/build tool stdout+stderr.
            known_files: Set of workspace-relative file paths.  When provided,
                         only errors pointing to known files are attributed;
                         others go to ``unattributed_errors``.

        Returns:
            ``AttributionResult`` with errors grouped by file.
        """


class CompilerErrorAttributor(BaseErrorAttributor):
    """Regex-based attributor supporting common compiler output formats.

    Handles Java/Maven, Go, TypeScript, Rust, and C# error formats.
    Falls back to a generic ``file:line`` pattern for unknown formats.
    """

    # Patterns: each yields named groups (file, line, col?, message?)
    # The number of groups varies by pattern — see _extract_match_fields().
    _PATTERNS: list[tuple[str, re.Pattern[str], int]] = [
        # Java/Maven: [ERROR] /abs/path/File.java:[10,5] error: msg
        # Groups: (file, line, col, message)
        ("java_maven", re.compile(
            r"\[ERROR\]\s+(?:[A-Za-z]:)?[/\\]?(.+?\.java):\[(\d+),(\d+)\]\s*(.*)",
        ), 4),
        # javac standalone: File.java:10: error: msg
        # Groups: (file, line, message) — no column
        ("javac", re.compile(
            r"(.+?\.java):(\d+):\s*(?:error|warning):\s*(.*)",
        ), 3),
        # Go: ./pkg/file.go:10:5: msg
        # Groups: (file, line, col, message)
        ("go", re.compile(
            r"\.?/?(.+?\.go):(\d+):(\d+):\s*(.*)",
        ), 4),
        # TypeScript: src/file.ts(10,5): error TS1234: msg
        # Groups: (file, line, col, message)
        ("typescript", re.compile(
            r"(.+?\.tsx?)\((\d+),(\d+)\):\s*(?:error|warning)\s+\w+:\s*(.*)",
        ), 4),
        # Rust/Cargo: error[E0308]: msg \n --> src/file.rs:10:5
        # Groups: (file, line, col) — message comes from pending_rust_msg
        ("rust_location", re.compile(
            r"-->\s+(.+?\.rs):(\d+):(\d+)",
        ), 3),
        # C#/dotnet: File.cs(10,5): error CS1234: msg
        # Groups: (file, line, col, message)
        ("csharp", re.compile(
            r"(.+?\.cs)\((\d+),(\d+)\):\s*(?:error|warning)\s+\w+:\s*(.*)",
        ), 4),
        # Generic: file.ext:line:col: message  OR  file.ext:line: message
        # Groups: (file, line, col?, message)
        ("generic", re.compile(
            r"([^\s:]+\.\w+):(\d+)(?::(\d+))?:\s*(.*)",
        ), 4),
    ]

    # Rust has a special pattern where the error message comes before the location
    _RUST_ERROR_PATTERN = re.compile(r"^error(?:\[E\d+\])?: (.+)")

    def attribute(
        self,
        build_output: str,
        known_files: set[str] | None = None,
    ) -> AttributionResult:
        result = AttributionResult()
        lines = build_output.splitlines()
        pending_rust_msg: str | None = None

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Check for Rust error message line (comes before location)
            rust_msg = self._RUST_ERROR_PATTERN.match(stripped)
            if rust_msg:
                pending_rust_msg = rust_msg.group(1)
                continue

            attributed = False
            for pattern_name, pattern, expected_groups in self._PATTERNS:
                match = pattern.search(stripped)
                if not match:
                    continue

                groups = match.groups()
                raw_path = groups[0]
                line_no = int(groups[1]) if len(groups) > 1 and groups[1] else None

                # Extract col and message based on expected group count
                if expected_groups == 4:
                    col = int(groups[2]) if groups[2] else None
                    msg = groups[3] if groups[3] else stripped
                elif expected_groups == 3 and pattern_name == "javac":
                    # javac: (file, line, message) — no column
                    col = None
                    msg = groups[2] if groups[2] else stripped
                elif expected_groups == 3 and pattern_name == "rust_location":
                    # rust: (file, line, col) — message from pending_rust_msg
                    col = int(groups[2]) if groups[2] else None
                    msg = pending_rust_msg or stripped
                else:
                    col = None
                    msg = stripped

                # Normalize path
                file_path = self._normalize_path(raw_path)

                # Filter to known files if provided
                if known_files and file_path not in known_files:
                    # Try to find a matching known file by suffix
                    resolved = self._resolve_to_known(file_path, known_files)
                    if resolved:
                        file_path = resolved
                    else:
                        # Could be an external/generated file — skip
                        continue

                error = AttributedError(
                    file_path=file_path,
                    line=line_no,
                    column=col,
                    message=msg.strip(),
                )
                result.errors_by_file.setdefault(file_path, []).append(error)
                attributed = True
                pending_rust_msg = None
                break

            if not attributed and self._looks_like_error(stripped):
                result.unattributed_errors.append(stripped)

        # Deduplicate errors per file (same line+message)
        for path in result.errors_by_file:
            result.errors_by_file[path] = self._deduplicate(result.errors_by_file[path])

        if result.errors_by_file:
            logger.info(
                "Build errors attributed: %d errors across %d files",
                result.total_errors, len(result.affected_files),
            )
        if result.unattributed_errors:
            logger.debug(
                "%d unattributed error lines", len(result.unattributed_errors),
            )

        return result

    @staticmethod
    def _normalize_path(raw_path: str) -> str:
        """Normalize a file path from compiler output to workspace-relative."""
        # Remove drive letters and leading separators
        path = raw_path.replace("\\", "/")

        # Strip common source prefixes to get workspace-relative
        for prefix in ("./", "../"):
            if path.startswith(prefix):
                path = path[len(prefix):]

        # Remove absolute path components if they contain typical workspace markers
        for marker in ("/src/main/java/", "/src/", "/app/"):
            idx = path.find(marker)
            if idx >= 0:
                path = path[idx + 1:]  # keep the marker but remove prefix
                break

        return path

    @staticmethod
    def _resolve_to_known(
        file_path: str, known_files: set[str]
    ) -> str | None:
        """Try to match a compiler path to a known workspace file by suffix."""
        # Exact match
        if file_path in known_files:
            return file_path

        # Suffix match (e.g., com/example/User.java → src/main/java/com/example/User.java)
        # Collect ALL candidates; ambiguous results are discarded to avoid
        # silently attributing errors to the wrong file.
        suffix_matches = [k for k in known_files if k.endswith(file_path)]
        if len(suffix_matches) == 1:
            return suffix_matches[0]
        if len(suffix_matches) > 1:
            logger.debug(
                "Ambiguous suffix attribution for %s — candidates: %s",
                file_path, suffix_matches,
            )
            return None

        # Basename match — only when there is exactly one file with that name.
        # If two files share the same basename (e.g. src/models/User.java and
        # src/dto/User.java) we cannot determine which one is at fault.
        basename = PurePosixPath(file_path).name
        basename_matches = [
            k for k in known_files
            if k.endswith(f"/{basename}") or k == basename
        ]
        if len(basename_matches) == 1:
            return basename_matches[0]
        if len(basename_matches) > 1:
            logger.debug(
                "Ambiguous basename attribution for %s — candidates: %s",
                file_path, basename_matches,
            )
            return None

        return None

    @staticmethod
    def _looks_like_error(line: str) -> bool:
        """Heuristic: does this line look like a compiler error?"""
        lower = line.lower()
        return any(kw in lower for kw in (
            "error", "cannot find symbol", "undefined reference",
            "unresolved", "fatal", "failed to compile",
        ))

    @staticmethod
    def _deduplicate(errors: list[AttributedError]) -> list[AttributedError]:
        """Remove duplicate errors (same file+line+message)."""
        seen: set[tuple[str, int | None, str]] = set()
        unique: list[AttributedError] = []
        for e in errors:
            key = (e.file_path, e.line, e.message)
            if key not in seen:
                seen.add(key)
                unique.append(e)
        return unique
