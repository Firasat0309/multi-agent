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


def extract_error_lines(build_output: str, *, max_chars: int = 4000) -> str:
    """Extract only error-relevant lines from build output.

    Build tools (Maven, Gradle, npm, tsc, cargo, go) can produce huge
    output filled with download progress, info lines, and warnings.
    This function extracts just the lines that matter for diagnosing
    failures:

    **Java/Maven/Gradle:**
      ``[ERROR]`` lines, ``BUILD FAILURE`` section, compilation errors.

    **npm/Node/Next.js:**
      ``ERR!`` lines, ``error TS`` lines, ``Module not found``,
      ``SyntaxError``, ``ERESOLVE``, ``Could not resolve``.

    **Go/Rust/C#:**
      ``error:``, ``error[E]``, ``undefined reference``, etc.

    Returns a compact string of deduplicated error lines, capped at
    *max_chars* to fit within LLM context budgets.
    """
    lines = build_output.splitlines()
    error_lines: list[str] = []
    in_failure_section = False
    seen: set[str] = set()

    # Keywords that mark a line as error-relevant (matched case-insensitively)
    _ERROR_KEYWORDS = (
        # Java / Maven / Gradle
        "[error]", "error:", "error[", "cannot find symbol",
        "build failure", "compilation failure", "failed to execute",
        "does not exist", "cannot resolve", "non-resolvable",
        "undefined reference", "unresolved",
        # npm / Node / Next.js / TypeScript
        "err!", "errno", "eresolve", "module not found",
        "syntaxerror", "typeerror", "referenceerror",
        "could not resolve", "cannot find module",
        "failed to compile", "build error",
        "error ts", "error:", "is not assignable to",
        "has no exported member", "property .* does not exist",
        # Go
        "cannot load", "no required module",
        # Rust / Cargo
        "aborting due to", "could not compile",
    )

    # Patterns that mark the start of a "capture everything" section
    _FAILURE_MARKERS = (
        "BUILD FAILURE",        # Maven
        "FAILED",               # Gradle
        "npm ERR!",             # npm
        "error: aborting",      # Rust
        "FAIL",                 # Go test
        "Failed to compile",    # Next.js / webpack
    )

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Once we hit a failure summary marker, capture everything remaining
        if not in_failure_section:
            for marker in _FAILURE_MARKERS:
                if marker in stripped:
                    in_failure_section = True
                    break

        lower = stripped.lower()
        is_error = (
            in_failure_section
            or any(kw in lower for kw in _ERROR_KEYWORDS)
        )

        if is_error and stripped not in seen:
            seen.add(stripped)
            error_lines.append(stripped)

    if not error_lines:
        # No recognizable error lines — return the last portion of output
        # which typically has the most relevant info.
        return build_output[-max_chars:].strip()

    result = "\n".join(error_lines)
    if len(result) > max_chars:
        # Keep the tail (most relevant errors are at the end)
        result = result[-max_chars:]
    return result


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
        # Next.js/webpack: ./src/file.tsx
        # Module not found: Can't resolve './Bar' in '/abs/path/src/components'
        # Groups: (file, message) — no line/col
        ("nextjs_module", re.compile(
            r"\./(.+?\.tsx?)\s*$",
        ), 1),
        # npm/webpack inline: ERROR in ./src/file.tsx 10:5
        # Groups: (file, line, col)
        ("webpack_error", re.compile(
            r"ERROR in \./(.+?\.tsx?)\s+(\d+):(\d+)",
        ), 3),
        # Generic: file.ext:line:col: message  OR  file.ext:line: message
        # Groups: (file, line, col?, message)
        ("generic", re.compile(
            r"([^\s:]+\.\w+):(\d+)(?::(\d+))?:\s*(.*)",
        ), 4),
    ]

    # Rust has a special pattern where the error message comes before the location
    _RUST_ERROR_PATTERN = re.compile(r"^error(?:\[E\d+\])?: (.+)")

    # Build-level errors that are not file-specific but indicate a failed build.
    # These are captured as unattributed errors so the caller knows the build failed
    # even when no file-specific errors could be parsed.
    _BUILD_LEVEL_ERRORS: list[re.Pattern[str]] = [
        # Java / Maven
        re.compile(r"\[ERROR\]\s+.*Could not resolve dependencies.*", re.IGNORECASE),
        re.compile(r"\[ERROR\]\s+.*Failed to execute goal.*", re.IGNORECASE),
        re.compile(r"\[ERROR\]\s+.*Compilation failure.*", re.IGNORECASE),
        re.compile(r"\[ERROR\]\s+.*BUILD FAILURE.*"),
        re.compile(r"\[ERROR\]\s+.*Non-resolvable parent POM.*", re.IGNORECASE),
        re.compile(r"\[ERROR\]\s+.*Plugin .+ or one of its dependencies could not be resolved.*", re.IGNORECASE),
        re.compile(r"\[ERROR\]\s+.*package .+ does not exist.*", re.IGNORECASE),
        re.compile(r"\[ERROR\]\s+.*cannot find symbol.*", re.IGNORECASE),
        # npm / Node / Next.js
        re.compile(r"npm ERR!\s+.*", re.IGNORECASE),
        re.compile(r"ERR!\s+.*ERESOLVE.*", re.IGNORECASE),
        re.compile(r"Module not found:\s+.*", re.IGNORECASE),
        re.compile(r"Failed to compile.*", re.IGNORECASE),
        re.compile(r"Error:\s+Cannot find module.*", re.IGNORECASE),
        re.compile(r"SyntaxError:\s+.*", re.IGNORECASE),
        re.compile(r"TypeError:\s+.*", re.IGNORECASE),
    ]

    def attribute(
        self,
        build_output: str,
        known_files: set[str] | None = None,
    ) -> AttributionResult:
        result = AttributionResult()
        raw_lines = build_output.splitlines()
        pending_rust_msg: str | None = None

        # Pre-strip and filter empty lines, keeping index for lookahead
        lines: list[str] = []
        for raw_line in raw_lines:
            s = raw_line.strip()
            if s:
                lines.append(s)

        i = 0
        while i < len(lines):
            stripped = lines[i]

            # Check for build-level errors (not file-specific)
            for pat in self._BUILD_LEVEL_ERRORS:
                if pat.search(stripped):
                    result.unattributed_errors.append(stripped)
                    break

            # Check for Rust error message line (comes before location)
            rust_msg = self._RUST_ERROR_PATTERN.match(stripped)
            if rust_msg:
                pending_rust_msg = rust_msg.group(1)
                i += 1
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
                elif expected_groups == 3 and pattern_name == "webpack_error":
                    # webpack: (file, line, col) — message is the full line
                    col = int(groups[2]) if groups[2] else None
                    msg = stripped
                elif expected_groups == 1 and pattern_name == "nextjs_module":
                    # Next.js/webpack: file path on one line, error on next.
                    # e.g.:  ./src/LoginForm.tsx
                    #        Module not found: Can't resolve '../../store/auth'
                    # Look ahead to grab the actual error message.
                    line_no = None
                    col = None
                    msg = stripped
                    if i + 1 < len(lines) and not lines[i + 1].startswith("./"):
                        msg = lines[i + 1]
                        i += 1  # consume the lookahead line
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

            i += 1

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
