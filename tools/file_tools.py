"""File manipulation tools for agents."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class FileTools:
    """Provides file read/write/list operations scoped to a workspace."""

    def __init__(self, workspace_root: Path) -> None:
        self.root = workspace_root

    def read_file(self, path: str) -> str:
        """Read a file relative to workspace root."""
        resolved = self._resolve(path)
        if not resolved.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return resolved.read_text(encoding="utf-8")

    def write_file(self, path: str, content: str) -> str:
        """Write content to a file, creating parent directories as needed."""
        resolved = self._resolve(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        return f"Written {len(content)} bytes to {path}"

    def edit_file(self, path: str, old: str, new: str) -> str:
        """Replace a string in a file."""
        resolved = self._resolve(path)
        if not resolved.exists():
            raise FileNotFoundError(f"File not found: {path}")
        content = resolved.read_text(encoding="utf-8")
        if old not in content:
            raise ValueError(f"String not found in {path}: {old[:50]}...")
        content = content.replace(old, new, 1)
        resolved.write_text(content, encoding="utf-8")
        return f"Edited {path}"

    def insert_after(self, path: str, anchor: str, new_content: str) -> str:
        """Insert new content after a matching anchor line.

        Finds the first line containing *anchor* and inserts *new_content*
        on the line(s) immediately following it.
        """
        resolved = self._resolve(path)
        if not resolved.exists():
            raise FileNotFoundError(f"File not found: {path}")
        lines = resolved.read_text(encoding="utf-8").splitlines(keepends=True)
        for i, line in enumerate(lines):
            if anchor in line:
                lines.insert(i + 1, new_content if new_content.endswith("\n") else new_content + "\n")
                resolved.write_text("".join(lines), encoding="utf-8")
                return f"Inserted after line {i + 1} in {path}"
        raise ValueError(f"Anchor not found in {path}: {anchor[:50]}...")

    def insert_before(self, path: str, anchor: str, new_content: str) -> str:
        """Insert new content before a matching anchor line."""
        resolved = self._resolve(path)
        if not resolved.exists():
            raise FileNotFoundError(f"File not found: {path}")
        lines = resolved.read_text(encoding="utf-8").splitlines(keepends=True)
        for i, line in enumerate(lines):
            if anchor in line:
                lines.insert(i, new_content if new_content.endswith("\n") else new_content + "\n")
                resolved.write_text("".join(lines), encoding="utf-8")
                return f"Inserted before line {i + 1} in {path}"
        raise ValueError(f"Anchor not found in {path}: {anchor[:50]}...")

    def insert_at_line(self, path: str, line_number: int, new_content: str) -> str:
        """Insert content at a specific line number (1-based)."""
        resolved = self._resolve(path)
        if not resolved.exists():
            raise FileNotFoundError(f"File not found: {path}")
        lines = resolved.read_text(encoding="utf-8").splitlines(keepends=True)
        idx = max(0, min(line_number - 1, len(lines)))
        lines.insert(idx, new_content if new_content.endswith("\n") else new_content + "\n")
        resolved.write_text("".join(lines), encoding="utf-8")
        return f"Inserted at line {line_number} in {path}"

    def append_to_file(self, path: str, content: str) -> str:
        """Append content to the end of a file."""
        resolved = self._resolve(path)
        if not resolved.exists():
            raise FileNotFoundError(f"File not found: {path}")
        existing = resolved.read_text(encoding="utf-8")
        if not existing.endswith("\n"):
            content = "\n" + content
        resolved.write_text(existing + content, encoding="utf-8")
        return f"Appended to {path}"

    def apply_patch(self, path: str, patch: str) -> str:
        """Apply a unified diff patch to a file.

        The *patch* should be a unified diff (lines starting with +/- and
        context lines) as produced by ``diff -u`` or an LLM generating a
        patch.  This uses a lightweight application strategy:
        - Lines starting with '-' are removed (must match existing)
        - Lines starting with '+' are inserted
        - Context lines (space prefix) are matched for anchoring

        Falls back to full-file replacement if the patch format is not
        recognized (i.e.  the LLM just returned full file content).
        """
        resolved = self._resolve(path)
        if not resolved.exists():
            raise FileNotFoundError(f"File not found: {path}")

        patch_lines = patch.splitlines()
        # Quick heuristic: if the patch has +/- lines, treat as diff
        has_diff_markers = any(
            line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
            for line in patch_lines[:20]
        )
        if not has_diff_markers:
            # Not a real diff — treat as full file replacement
            resolved.write_text(patch, encoding="utf-8")
            return f"Replaced {path} (no diff markers detected)"

        current = resolved.read_text(encoding="utf-8")
        current_lines = current.splitlines()
        result_lines: list[str] = list(current_lines)

        # Simple line-match strategy: find context lines to anchor insertions
        offset = 0
        for line in patch_lines:
            if line.startswith("@@") or line.startswith("---") or line.startswith("+++"):
                continue
            if line.startswith("-"):
                removed = line[1:]
                # Find and remove the matching line
                for i in range(max(0, offset), len(result_lines)):
                    if result_lines[i].rstrip() == removed.rstrip():
                        result_lines.pop(i)
                        offset = i
                        break
            elif line.startswith("+"):
                added = line[1:]
                result_lines.insert(offset, added)
                offset += 1
            elif line.startswith(" "):
                # Context line — advance offset
                ctx = line[1:]
                for i in range(max(0, offset), len(result_lines)):
                    if result_lines[i].rstrip() == ctx.rstrip():
                        offset = i + 1
                        break

        resolved.write_text("\n".join(result_lines) + "\n", encoding="utf-8")
        return f"Patched {path}"

    def list_files(self, directory: str = "", pattern: str = "**/*") -> list[str]:
        """List files matching a glob pattern."""
        target = self._resolve(directory)
        if not target.exists():
            return []
        return [
            str(p.relative_to(self.root))
            for p in target.glob(pattern)
            if p.is_file()
        ]

    def file_exists(self, path: str) -> bool:
        return self._resolve(path).exists()

    def _resolve(self, path: str) -> Path:
        """Resolve a path relative to workspace root, preventing path traversal."""
        resolved = (self.root / path).resolve()
        if not str(resolved).startswith(str(self.root.resolve())):
            raise PermissionError(f"Path traversal detected: {path}")
        return resolved
