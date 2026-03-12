"""Import validation for modified source files.

Validates that imports declared in a file can be resolved against the set
of known workspace files.  Called after every MODIFY_FILE / GENERATE_FILE
write to surface broken imports immediately rather than at test time.

Only flags imports that look like *internal* workspace imports — stdlib and
third-party packages are not checked (we have no manifest of installed
packages at validation time).
"""

from __future__ import annotations

import ast
import logging
import posixpath
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.language import LanguageProfile

logger = logging.getLogger(__name__)


class ImportValidator:
    """Validates that all imports in a modified file are resolvable.

    For Python: uses ``ast.parse()`` to extract imports exactly (no regex).
    For Java:   regex over ``import`` statements.
    For Go:     regex over quoted import paths.

    Only internal imports are checked — i.e. imports whose top-level
    package name matches a known workspace directory.  Third-party or
    stdlib imports are silently ignored.

    Returns a list of unresolvable import strings so the caller can report
    them without failing the write.
    """

    def validate(
        self,
        file_path: str,
        content: str,
        known_files: set[str],
        lang: "LanguageProfile",
    ) -> list[str]:
        """Return a list of internal imports that cannot be resolved.

        ``known_files`` should be a set of workspace-relative paths (using
        forward slashes) that currently exist on disk.
        """
        if not known_files:
            return []

        try:
            if lang.name == "python":
                return self._validate_python(content, known_files)
            if lang.name == "java":
                return self._validate_java(content, known_files)
            if lang.name == "go":
                return self._validate_go(content, known_files)
            if lang.name in ("typescript", "tsx"):
                return self._validate_typescript(content, file_path, known_files)
        except Exception as exc:
            logger.debug(
                "Import validation error for %s (%s): %s",
                file_path, lang.name, exc,
            )
        return []

    # ── Language-specific validators ─────────────────────────────────────────

    def _validate_python(self, content: str, known_files: set[str]) -> list[str]:
        try:
            tree = ast.parse(content)
        except SyntaxError:
            # Broken syntax is reported elsewhere; skip import validation
            return []

        # Build the set of top-level workspace packages so we can distinguish
        # internal from third-party imports.
        internal_tops = _internal_top_levels(known_files, ".py")
        broken: list[str] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if top not in internal_tops:
                        continue  # third-party or stdlib
                    candidate = alias.name.replace(".", "/") + ".py"
                    pkg_init = alias.name.replace(".", "/") + "/__init__.py"
                    if candidate not in known_files and pkg_init not in known_files:
                        broken.append(f"import {alias.name}")

            elif isinstance(node, ast.ImportFrom):
                if node.level and node.level > 0:
                    continue  # relative import — skip
                if not node.module:
                    continue
                top = node.module.split(".")[0]
                if top not in internal_tops:
                    continue  # third-party or stdlib
                candidate = node.module.replace(".", "/") + ".py"
                pkg_init = node.module.replace(".", "/") + "/__init__.py"
                if candidate not in known_files and pkg_init not in known_files:
                    broken.append(f"from {node.module} import ...")

        return broken

    def _validate_java(self, content: str, known_files: set[str]) -> list[str]:
        internal_tops = _internal_top_levels(known_files, ".java")
        broken: list[str] = []

        for m in re.finditer(r"^\s*import\s+([\w.]+)\s*;", content, re.MULTILINE):
            imp = m.group(1)
            top = imp.split(".")[0]
            if top not in internal_tops:
                continue
            candidate = imp.replace(".", "/") + ".java"
            if candidate not in known_files:
                broken.append(imp)

        return broken

    def _validate_typescript(
        self, content: str, file_path: str, known_files: set[str]
    ) -> list[str]:
        """Check relative TypeScript/TSX imports against the known workspace file set.

        Only relative imports (starting with ``./`` or ``../``) are checked.
        Bare specifiers (``react``, ``@/components/Foo``) are always skipped
        since we cannot resolve package.json aliases or node_modules here.
        """
        file_path = file_path.replace("\\", "/")
        file_dir = "/".join(file_path.split("/")[:-1])

        # Match: import ... from '...'  /  import '...'  /  require('...')
        imp_re = re.compile(
            r"""(?:import\s+(?:type\s+)?(?:[^'"\n;]+?\s+from\s+)?"""
            r"""['\"]([^'\"]+)['\"]|require\s*\(\s*['\"]([^'\"]+)['\"]\s*\))"""
        )
        broken: list[str] = []
        for m in imp_re.finditer(content):
            imp = m.group(1) or m.group(2)
            if not imp.startswith("."):  # skip node_modules and @-aliases
                continue
            # Resolve relative path
            raw = posixpath.normpath(file_dir + "/" + imp if file_dir else imp)
            raw = raw.lstrip("/")
            # Try all common extensions and index files
            candidates = [
                raw + ".ts",
                raw + ".tsx",
                raw + ".js",
                raw + ".jsx",
                raw + "/index.ts",
                raw + "/index.tsx",
                raw + "/index.js",
            ]
            # Also accept the path as-is (e.g. already has extension)
            candidates.append(raw)
            if not any(c in known_files for c in candidates):
                broken.append(imp)
        return broken

    def _validate_go(self, content: str, known_files: set[str]) -> list[str]:
        # Match quoted import paths inside import blocks or single-line imports
        internal_tops = _internal_top_levels(known_files, ".go")
        broken: list[str] = []

        for m in re.finditer(r'"([\w][^"]+)"', content):
            path = m.group(1)
            # Only check paths that look like relative workspace imports
            # (no dots in the first segment, no external domain-style paths)
            top = path.split("/")[0]
            if top not in internal_tops:
                continue
            # Accept if there's any file under this path
            if not any(f.startswith(path + "/") or f == path + ".go" for f in known_files):
                broken.append(path)

        return broken


# ── Helpers ───────────────────────────────────────────────────────────────────

def _internal_top_levels(known_files: set[str], ext: str) -> set[str]:
    """Return top-level directory names that contain workspace source files."""
    tops: set[str] = set()
    for f in known_files:
        if not f.endswith(ext):
            continue
        parts = f.split("/")
        if len(parts) >= 2:
            tops.add(parts[0])
        elif len(parts) == 1:
            # Root-level file — module name is filename without extension
            tops.add(parts[0].removesuffix(ext))
    return tops
