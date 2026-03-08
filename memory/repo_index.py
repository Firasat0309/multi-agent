"""Filesystem-based repository index (memory layer)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from core.models import FileIndex, RepositoryIndex

logger = logging.getLogger(__name__)


class RepoIndexStore:
    """Persistent filesystem-based repository index."""

    def __init__(self, workspace_dir: Path) -> None:
        self.index_path = workspace_dir / "repo_index.json"
        self._index = RepositoryIndex()
        self._load()

    def _load(self) -> None:
        if not self.index_path.exists():
            return
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
            for f in data.get("files", []):
                self._index.files.append(FileIndex(
                    path=f["path"],
                    exports=f.get("exports", []),
                    imports=f.get("imports", []),
                    classes=f.get("classes", []),
                    functions=f.get("functions", []),
                    checksum=f.get("checksum", ""),
                ))
        except Exception as e:
            logger.warning(f"Failed to load repo index: {e}")

    def save(self) -> None:
        data = {
            "files": [
                {
                    "path": f.path,
                    "exports": f.exports,
                    "imports": f.imports,
                    "classes": f.classes,
                    "functions": f.functions,
                    "checksum": f.checksum,
                }
                for f in self._index.files
            ]
        }
        self.index_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def get_index(self) -> RepositoryIndex:
        return self._index

    def update_file(self, file_index: FileIndex) -> None:
        self._index.add_or_update(file_index)
        self.save()

    def query_exports(self, symbol: str) -> list[str]:
        """Find which files export a symbol."""
        return [f.path for f in self._index.files if symbol in f.exports]

    def query_importers(self, module_path: str) -> list[str]:
        """Find which files import from a given module."""
        module = module_path.replace("/", ".").removesuffix(".py")
        return [
            f.path for f in self._index.files
            if any(module in imp for imp in f.imports)
        ]

    def get_file_info(self, path: str) -> FileIndex | None:
        return self._index.get_file(path)
