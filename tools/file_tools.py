"""File manipulation tools for agents."""

from __future__ import annotations

import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class PatchError(ValueError):
    """Raised when a unified diff patch cannot be applied cleanly."""


def _write_atomic(path: Path, content: str) -> None:
    """Write *content* to *path* atomically via a temp-file + rename.

    Prevents partial writes from being observed by concurrent readers.  On
    Windows, ``os.replace`` is atomic on NTFS for same-volume moves.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=".~", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


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
        _write_atomic(resolved, content)
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
        _write_atomic(resolved, content)
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
                _write_atomic(resolved, "".join(lines))
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
                _write_atomic(resolved, "".join(lines))
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
        _write_atomic(resolved, "".join(lines))
        return f"Inserted at line {line_number} in {path}"

    def append_to_file(self, path: str, content: str) -> str:
        """Append content to the end of a file."""
        resolved = self._resolve(path)
        if not resolved.exists():
            raise FileNotFoundError(f"File not found: {path}")
        existing = resolved.read_text(encoding="utf-8")
        if not existing.endswith("\n"):
            content = "\n" + content
        _write_atomic(resolved, existing + content)
        return f"Appended to {path}"

    def apply_patch(self, file_path: str, patch_text: str) -> bool:
        """Apply a unified diff patch to a file.

        Returns ``True`` if applied cleanly, ``False`` if the patch does not
        apply (context mismatch, missing file, malformed diff).  Never writes
        if validation / application fails — the original file is left intact.
        """
        path = self._resolve(file_path)
        if not path.exists():
            logger.warning("apply_patch: file not found: %s", file_path)
            return False
        original = path.read_text(encoding="utf-8")
        try:
            patched = self._apply_unified_diff(original, patch_text)
        except PatchError as e:
            logger.warning("Patch failed to apply to %s: %s", file_path, e)
            return False
        _write_atomic(path, patched)
        return True

    def validate_patch(self, file_path: str, patch_text: str) -> tuple[bool, str]:
        """Dry-run validate a patch without writing.

        Returns ``(True, "")`` when the patch applies cleanly, or
        ``(False, error_message)`` otherwise.
        """
        path = self._resolve(file_path)
        if not path.exists():
            return False, f"File not found: {file_path}"
        original = path.read_text(encoding="utf-8")
        try:
            self._apply_unified_diff(original, patch_text)
            return True, ""
        except PatchError as e:
            return False, str(e)
        except Exception as e:  # pragma: no cover
            return False, f"Unexpected error during patch validation: {e}"

    @staticmethod
    def _apply_unified_diff(original: str, patch_text: str) -> str:
        """Apply a standard unified diff to *original* and return the patched text.

        Parses every ``@@`` hunk header, verifies context lines match, removes
        ``-`` lines, inserts ``+`` lines, and copies all unchanged lines.
        Raises ``PatchError`` on any mismatch so the caller can decide whether
        to fall back to a full-file rewrite.
        """
        orig_lines = original.splitlines(keepends=True)
        # Normalise: ensure every line ends with newline for uniform handling
        if orig_lines and not orig_lines[-1].endswith("\n"):
            orig_lines[-1] += "\n"

        patch_lines = patch_text.splitlines()

        result: list[str] = []
        orig_pos = 0  # 0-based cursor into orig_lines
        i = 0

        while i < len(patch_lines):
            line = patch_lines[i]

            # Skip file-header lines (--- / +++)
            if line.startswith("---") or line.startswith("+++"):
                i += 1
                continue

            if line.startswith("@@"):
                m = re.match(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
                if not m:
                    raise PatchError(f"Malformed hunk header: {line!r}")
                orig_hunk_start = int(m.group(1))  # 1-based

                # Copy unchanged lines that precede this hunk
                target_pos = orig_hunk_start - 1  # 0-based
                while orig_pos < target_pos:
                    if orig_pos >= len(orig_lines):
                        raise PatchError(
                            f"Hunk starts at line {orig_hunk_start} but file only has "
                            f"{len(orig_lines)} lines"
                        )
                    result.append(orig_lines[orig_pos])
                    orig_pos += 1

                i += 1  # advance past @@ header line

                # Apply lines inside this hunk
                while i < len(patch_lines):
                    hl = patch_lines[i]
                    if hl.startswith("@@") or hl.startswith("---") or hl.startswith("+++"):
                        break  # start of next hunk or file header

                    if hl.startswith(" "):  # context line
                        expected = hl[1:]
                        if orig_pos >= len(orig_lines):
                            raise PatchError(
                                f"Context line expected at position {orig_pos + 1} "
                                f"but file has only {len(orig_lines)} lines"
                            )
                        actual = orig_lines[orig_pos]
                        if actual.rstrip("\n") != expected.rstrip("\n"):
                            raise PatchError(
                                f"Context mismatch at line {orig_pos + 1}: "
                                f"expected {expected!r}, got {actual!r}"
                            )
                        result.append(actual)
                        orig_pos += 1

                    elif hl.startswith("-"):  # remove line
                        expected = hl[1:]
                        if orig_pos >= len(orig_lines):
                            raise PatchError(
                                f"Remove line at position {orig_pos + 1} past end of file"
                            )
                        actual = orig_lines[orig_pos]
                        if actual.rstrip("\n") != expected.rstrip("\n"):
                            raise PatchError(
                                f"Remove mismatch at line {orig_pos + 1}: "
                                f"expected {expected!r}, got {actual!r}"
                            )
                        orig_pos += 1  # skip — don't copy to result

                    elif hl.startswith("+"):  # add line
                        added = hl[1:]
                        if not added.endswith("\n"):
                            added += "\n"
                        result.append(added)

                    elif hl.startswith("\\"):  # "\ No newline at end of file"
                        if result:
                            result[-1] = result[-1].rstrip("\n")

                    # Other lines in hunk body (blank etc.) — skip silently
                    i += 1

            else:
                i += 1  # skip non-hunk top-level lines

        # Copy remaining lines after the last hunk
        while orig_pos < len(orig_lines):
            result.append(orig_lines[orig_pos])
            orig_pos += 1

        return "".join(result)

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
        root_resolved = self.root.resolve()
        resolved = (self.root / path).resolve()
        if not (resolved == root_resolved or resolved.is_relative_to(root_resolved)):
            raise PermissionError(f"Path traversal detected: {path}")
        return resolved
