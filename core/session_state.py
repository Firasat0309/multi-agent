"""Session state persistence for resumable pipeline runs.

When a pipeline run is interrupted (crash, timeout, Ctrl-C), the session
state can be saved to disk and loaded on the next run to skip files that
already passed their generate→build→fix cycle.

Inspired by Claude Code's conversation persistence that survives terminal
restarts and resumes exactly where the user left off.

Usage::

    from core.session_state import SessionState

    # At pipeline start:
    session = SessionState.load_or_create(workspace, run_id)

    # During execution:
    session.mark_file_passed("src/models/User.java")
    session.mark_file_failed("src/services/OrderService.java", "build error: ...")
    session.save()

    # On resume:
    session = SessionState.load_or_create(workspace, run_id)
    if session.is_file_passed("src/models/User.java"):
        skip  # already done
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SESSION_DIR = ".multi-agent"
_SESSION_FILE = "session_state.json"


class FileStatus:
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class FileState:
    """Per-file progress within a session."""
    path: str
    status: str = FileStatus.PENDING
    attempts: int = 0
    last_error: str = ""
    completed_at: float = 0.0


@dataclass
class SessionState:
    """Persistent session state for resumable pipeline runs."""
    run_id: str
    workspace: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    files: dict[str, FileState] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    # Tier tracking: which tiers have been fully completed
    completed_tiers: list[int] = field(default_factory=list)

    # ── File state management ────────────────────────────────────────

    def mark_file_passed(self, path: str) -> None:
        """Mark a file as successfully processed."""
        fs = self.files.get(path) or FileState(path=path)
        fs.status = FileStatus.PASSED
        fs.completed_at = time.time()
        fs.attempts += 1
        self.files[path] = fs
        self.updated_at = time.time()

    def mark_file_failed(self, path: str, error: str = "") -> None:
        """Mark a file as failed."""
        fs = self.files.get(path) or FileState(path=path)
        fs.status = FileStatus.FAILED
        fs.last_error = error[:500]
        fs.attempts += 1
        self.files[path] = fs
        self.updated_at = time.time()

    def mark_file_skipped(self, path: str) -> None:
        """Mark a file as intentionally skipped."""
        fs = self.files.get(path) or FileState(path=path)
        fs.status = FileStatus.SKIPPED
        self.files[path] = fs
        self.updated_at = time.time()

    def is_file_passed(self, path: str) -> bool:
        """Check if a file has already passed."""
        fs = self.files.get(path)
        return fs is not None and fs.status == FileStatus.PASSED

    def is_file_done(self, path: str) -> bool:
        """Check if a file is in a terminal state (passed, failed, skipped)."""
        fs = self.files.get(path)
        return fs is not None and fs.status in (FileStatus.PASSED, FileStatus.FAILED, FileStatus.SKIPPED)

    def pending_files(self, all_paths: list[str]) -> list[str]:
        """Return files from *all_paths* that have not yet passed."""
        return [p for p in all_paths if not self.is_file_passed(p)]

    def mark_tier_complete(self, tier: int) -> None:
        """Record that a tier has been fully completed."""
        if tier not in self.completed_tiers:
            self.completed_tiers.append(tier)
            self.updated_at = time.time()

    def is_tier_complete(self, tier: int) -> bool:
        return tier in self.completed_tiers

    @property
    def summary(self) -> dict[str, int]:
        """Count files by status."""
        counts: dict[str, int] = {}
        for fs in self.files.values():
            counts[fs.status] = counts.get(fs.status, 0) + 1
        return counts

    # ── Persistence ──────────────────────────────────────────────────

    def save(self) -> Path:
        """Save session state to workspace/.multi-agent/session_state.json."""
        self.updated_at = time.time()
        session_dir = Path(self.workspace) / _SESSION_DIR
        session_dir.mkdir(parents=True, exist_ok=True)
        path = session_dir / _SESSION_FILE

        data = {
            "run_id": self.run_id,
            "workspace": self.workspace,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_tiers": self.completed_tiers,
            "metadata": self.metadata,
            "files": {
                k: asdict(v) for k, v in self.files.items()
            },
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.debug("Session state saved: %s (%d files)", path, len(self.files))
        return path

    @classmethod
    def load_or_create(cls, workspace: str, run_id: str) -> "SessionState":
        """Load existing session state or create a new one.

        If a session file exists for the same *run_id*, it is loaded so
        already-passed files can be skipped.  Otherwise a fresh session
        is created.
        """
        path = Path(workspace) / _SESSION_DIR / _SESSION_FILE
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("run_id") == run_id:
                    session = cls(
                        run_id=data["run_id"],
                        workspace=data["workspace"],
                        created_at=data.get("created_at", time.time()),
                        updated_at=data.get("updated_at", time.time()),
                        completed_tiers=data.get("completed_tiers", []),
                        metadata=data.get("metadata", {}),
                    )
                    for k, v in data.get("files", {}).items():
                        session.files[k] = FileState(**v)
                    passed = sum(1 for f in session.files.values() if f.status == FileStatus.PASSED)
                    logger.info(
                        "Resumed session %s: %d files (%d passed)",
                        run_id, len(session.files), passed,
                    )
                    return session
                logger.info("Session file exists but run_id mismatch — creating new session")
            except Exception as exc:
                logger.warning("Failed to load session state: %s — creating new", exc)

        return cls(run_id=run_id, workspace=workspace)

    @classmethod
    def clear(cls, workspace: str) -> None:
        """Remove session state file."""
        path = Path(workspace) / _SESSION_DIR / _SESSION_FILE
        if path.exists():
            path.unlink()
            logger.info("Session state cleared: %s", path)
