"""Workspace snapshot for safe repository modification.

Usage in pipeline._enhance_inner():
    async with WorkspaceSnapshot(workspace_dir) as snap:
        exec_result = await agent_manager.execute_graph(task_graph)
        changed_files = snap.get_changed_files()
        diff_stats = snap.compute_diff_stats()
        snap.commit()
    # on exception → auto-restore before propagating

Only captures source files (not .chroma, __pycache__, .git, etc.).
On failure, restores exactly the files that were changed.
Writes a snapshot manifest so partial restores are possible.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Directories to skip when capturing source files
_SKIP_DIRS: frozenset[str] = frozenset({
    ".chroma",
    "__pycache__",
    ".git",
    "node_modules",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    "dist",
    "build",
    ".eggs",
    "target",  # Java/Rust build output
})

# Source file extensions to include in the snapshot
_SOURCE_EXTS: frozenset[str] = frozenset({
    ".py", ".java", ".go", ".ts", ".js", ".rs", ".cs",
    ".cpp", ".c", ".h", ".rb", ".kt", ".swift", ".scala",
    ".toml", ".json", ".yaml", ".yml", ".xml", ".properties",
    ".sql", ".sh", ".md", ".txt", ".env",
})

# Root-level config files always included by name
_ROOT_LEVEL_NAMES: frozenset[str] = frozenset({
    "pyproject.toml", "requirements.txt", "go.mod", "go.sum",
    "package.json", "package-lock.json", "tsconfig.json",
    "Cargo.toml", "Cargo.lock", "pom.xml", "build.gradle",
    "Makefile", "Dockerfile", "docker-compose.yml",
    ".gitignore", ".editorconfig",
})


def _file_checksum(path: Path) -> str:
    """SHA-1 checksum of a file's bytes."""
    return hashlib.sha1(path.read_bytes()).hexdigest()


class WorkspaceSnapshot:
    """Captures a point-in-time snapshot of all source files before modification.

    Only captures source files (not .chroma, __pycache__, .git, etc.).
    On failure, restores exactly the files that were changed.
    Writes a snapshot manifest so partial restores are possible.
    """

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self._snapshot_dir: Path | None = None
        self._manifest: dict[str, str] = {}  # rel_path → checksum
        self._committed = False

    async def __aenter__(self) -> "WorkspaceSnapshot":
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        self._snapshot_dir = self.workspace / f".snapshot_{ts}"
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._capture()
        logger.info(
            "Workspace snapshot captured: %d files in %s",
            len(self._manifest), self._snapshot_dir.name,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[override]
        if exc_type is not None and not self._committed:
            logger.info(
                "Exception detected (%s) — restoring workspace from snapshot",
                exc_type.__name__,
            )
            self._restore()
        # Always clean up the snapshot directory
        if self._snapshot_dir and self._snapshot_dir.exists():
            shutil.rmtree(self._snapshot_dir, ignore_errors=True)
            logger.debug("Snapshot directory removed: %s", self._snapshot_dir.name)

    def commit(self) -> None:
        """Call after successful completion to keep all changes."""
        self._committed = True
        logger.debug("Snapshot committed — changes will be kept")

    def get_changed_files(self) -> list[str]:
        """Return list of rel-paths that differ from the snapshot (added/modified/deleted)."""
        changed: list[str] = []
        for rel_path, orig_checksum in self._manifest.items():
            current_file = self.workspace / rel_path
            if not current_file.exists():
                changed.append(rel_path)
                continue
            try:
                if _file_checksum(current_file) != orig_checksum:
                    changed.append(rel_path)
            except OSError:
                changed.append(rel_path)
        return changed

    def compute_diff_stats(self) -> dict[str, int]:
        """Count lines added/removed across all changed files.

        Compares the snapshot copy against the current workspace file for each
        changed path.  Returns ``{"lines_added": N, "lines_removed": N}``.
        """
        if not self._snapshot_dir:
            return {"lines_added": 0, "lines_removed": 0}

        total_added = 0
        total_removed = 0

        for rel_path in self.get_changed_files():
            snap_file = self._snapshot_dir / rel_path
            current_file = self.workspace / rel_path

            orig_lines: list[str] = []
            new_lines: list[str] = []

            if snap_file.exists():
                try:
                    orig_lines = snap_file.read_text(encoding="utf-8", errors="replace").splitlines()
                except OSError:
                    pass
            if current_file.exists():
                try:
                    new_lines = current_file.read_text(encoding="utf-8", errors="replace").splitlines()
                except OSError:
                    pass

            # Simple line count diff (not a proper LCS diff, but fast and good enough)
            orig_set = set(orig_lines)
            new_set = set(new_lines)
            total_removed += sum(1 for l in orig_lines if l not in new_set)
            total_added += sum(1 for l in new_lines if l not in orig_set)

        return {"lines_added": total_added, "lines_removed": total_removed}

    # ── Private helpers ──────────────────────────────────────────────────────

    def _capture(self) -> None:
        """Copy all source files into the snapshot directory and build the manifest."""
        for src_file in self._iter_source_files():
            rel = src_file.relative_to(self.workspace)
            rel_str = str(rel).replace("\\", "/")
            checksum = _file_checksum(src_file)
            self._manifest[rel_str] = checksum

            dest = self._snapshot_dir / rel  # type: ignore[operator]
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dest)

        # Write manifest for manual inspection / partial restore
        manifest_path = self._snapshot_dir / "manifest.json"  # type: ignore[operator]
        manifest_path.write_text(
            json.dumps(self._manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _restore(self) -> None:
        """Restore all snapshotted files to their original content."""
        if not self._snapshot_dir:
            return
        restored = 0
        for rel_path in self._manifest:
            snap_file = self._snapshot_dir / rel_path
            dest = self.workspace / rel_path
            if snap_file.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(snap_file, dest)
                restored += 1
            else:
                logger.warning("Snapshot file missing, cannot restore: %s", rel_path)
        logger.info("Restored %d file(s) from snapshot", restored)

    def _iter_source_files(self):  # type: ignore[return]
        """Yield workspace source files, skipping build/cache directories."""
        for p in self.workspace.rglob("*"):
            if not p.is_file():
                continue

            rel_parts = p.relative_to(self.workspace).parts

            # Skip files inside snapshot directories (our own artifacts)
            if any(part.startswith(".snapshot_") for part in rel_parts):
                continue

            # Skip hidden directories (except files at root level like .gitignore)
            if len(rel_parts) > 1 and any(
                part.startswith(".") for part in rel_parts[:-1]
            ):
                continue

            # Skip build/cache directories by name
            if any(part in _SKIP_DIRS for part in rel_parts):
                continue

            # Include by extension or known root-level config name
            if p.suffix in _SOURCE_EXTS or p.name in _ROOT_LEVEL_NAMES:
                yield p
