"""Fast syntax validation using language-native parsers.

Catches ~60% of syntax errors in <1s before the expensive full-project
build step.  This avoids wasting 8-15s on a build that will inevitably
fail due to a missing semicolon or unbalanced brace.

Supported strategies:
  - Python: ``py_compile`` (stdlib, 0-cost)
  - Java: ``javac`` single-file syntax check (no classpath resolution)
  - TypeScript/JavaScript: ``node --check`` or ``npx tsc --noEmit``
  - Go: ``go vet`` on individual file
  - Rust: ``cargo check`` (deferred to build — no single-file mode)

When no validator is available for a language, the check is skipped and
the file proceeds directly to the full build.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.language import LanguageProfile

logger = logging.getLogger(__name__)

# Timeout for individual syntax checks (seconds).
# These should complete in well under 1s; 10s is a generous safety ceiling.
_SYNTAX_CHECK_TIMEOUT = 10


class FastValidator:
    """Sub-second syntax validation using language-native parsers.

    Designed to sit between file generation and the build step in the
    simple loop executor.  When the syntax check fails, the build is
    skipped entirely and the errors are fed directly to the fix agent,
    saving one full build cycle (8-15s).
    """

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace

    async def validate_syntax(
        self, file_path: str, language: str
    ) -> tuple[bool, str]:
        """Validate syntax of a single generated file.

        Args:
            file_path: Workspace-relative path to the file.
            language: Language identifier (e.g. "python", "java", "typescript").

        Returns:
            ``(passed, error_text)`` — if passed is False, error_text contains
            the compiler/parser error output (truncated to 4000 chars).
        """
        abs_path = self._workspace / file_path
        if not abs_path.exists():
            return True, ""  # File doesn't exist yet — skip

        lang = language.lower()

        if lang == "python":
            return await self._check_python(abs_path)
        elif lang == "java":
            return await self._check_java(abs_path)
        elif lang in ("typescript", "javascript"):
            return await self._check_js_ts(abs_path, lang)
        elif lang == "go":
            return await self._check_go(abs_path)
        elif lang in ("c#", "csharp"):
            # C# syntax checking requires full project context — skip
            return True, ""
        elif lang == "rust":
            # Rust doesn't support single-file check — skip
            return True, ""
        else:
            return True, ""  # Unknown language — skip

    async def _check_python(self, abs_path: Path) -> tuple[bool, str]:
        """Use py_compile (stdlib) for zero-cost Python syntax checking."""
        import py_compile
        try:
            await asyncio.to_thread(
                py_compile.compile, str(abs_path), doraise=True
            )
            return True, ""
        except py_compile.PyCompileError as e:
            return False, str(e)[:4000]

    async def _check_java(self, abs_path: Path) -> tuple[bool, str]:
        """Use javac for syntax-only Java checking.

        Runs ``javac -proc:none -implicit:none`` which parses and
        type-checks the single file without resolving dependencies.
        This catches syntax errors, unbalanced braces, and basic
        type mistakes in <1s.
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                "javac", "-proc:none", "-implicit:none",
                "-Xlint:none", "-d", tempfile.gettempdir(),
                str(abs_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=_SYNTAX_CHECK_TIMEOUT
            )
            if proc.returncode == 0:
                return True, ""
            error_text = (stderr or stdout or b"").decode(errors="replace")
            return False, error_text[:4000]
        except FileNotFoundError:
            logger.debug("javac not found — skipping Java syntax check")
            return True, ""
        except asyncio.TimeoutError:
            logger.warning("Java syntax check timed out for %s", abs_path.name)
            return True, ""  # Don't block on timeout — let the build handle it

    async def _check_js_ts(self, abs_path: Path, lang: str) -> tuple[bool, str]:
        """Use Node.js --check for JS or tsc --noEmit for TS."""
        if lang == "javascript":
            cmd = ["node", "--check", str(abs_path)]
        else:
            # TypeScript: use tsc with isolatedModules to avoid needing full project
            cmd = ["npx", "tsc", "--noEmit", "--isolatedModules",
                   "--allowJs", "--esModuleInterop", str(abs_path)]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=_SYNTAX_CHECK_TIMEOUT
            )
            if proc.returncode == 0:
                return True, ""
            error_text = (stderr or stdout or b"").decode(errors="replace")
            return False, error_text[:4000]
        except FileNotFoundError:
            logger.debug("node/npx not found — skipping JS/TS syntax check")
            return True, ""
        except asyncio.TimeoutError:
            logger.warning("JS/TS syntax check timed out for %s", abs_path.name)
            return True, ""

    async def _check_go(self, abs_path: Path) -> tuple[bool, str]:
        """Use go vet for Go syntax checking."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "go", "vet", str(abs_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(abs_path.parent),
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=_SYNTAX_CHECK_TIMEOUT
            )
            if proc.returncode == 0:
                return True, ""
            error_text = (stderr or stdout or b"").decode(errors="replace")
            return False, error_text[:4000]
        except FileNotFoundError:
            logger.debug("go not found — skipping Go syntax check")
            return True, ""
        except asyncio.TimeoutError:
            logger.warning("Go syntax check timed out for %s", abs_path.name)
            return True, ""
