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
