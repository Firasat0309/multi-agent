"""Session persistence — save and resume pipeline state across crashes.

A pipeline run can take 30+ minutes.  If it crashes at tier 5 of 8, all
progress is lost.  This module saves checkpoints after each tier so the
executor can resume from the last successful checkpoint.

Architecture
------------
Each pipeline run creates a session directory containing:
  - ``session.json`` — run metadata (prompt, settings, start time)
  - ``checkpoint_tier_N.json`` — state snapshot after tier N completes
  - ``events.jsonl`` — append-only event log (one JSON line per event)
  - ``cost.json`` — accumulated LLM cost tracking

Resume flow:
  1. ``PipelineSession.load_latest_checkpoint()`` finds the most recent tier
  2. ``LifecycleEngine`` is reconstructed from the checkpoint state
  3. ``SimpleLoopExecutor.execute()`` skips completed tiers

Usage::

    session = PipelineSession(session_dir=Path("workspace/.session"))
    session.save_metadata(prompt="Build a REST API", settings={...})

    # After each tier:
    session.save_checkpoint(engine, completed_tier=2, event_log=events)

    # On restart:
    checkpoint = session.load_latest_checkpoint()
    if checkpoint:
        engine = reconstruct_engine(checkpoint)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.errors import SessionError

logger = logging.getLogger(__name__)


@dataclass
class CheckpointData:
    """Serializable snapshot of pipeline state after a tier completes."""

    completed_tier: int
    timestamp: float
    file_states: dict[str, dict[str, Any]]
    # Summary metrics for quick inspection without loading full state
    files_passed: int = 0
    files_failed: int = 0
    files_degraded: int = 0
    files_pending: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "completed_tier": self.completed_tier,
            "timestamp": self.timestamp,
            "file_states": self.file_states,
            "files_passed": self.files_passed,
            "files_failed": self.files_failed,
            "files_degraded": self.files_degraded,
            "files_pending": self.files_pending,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointData:
        return cls(
            completed_tier=data["completed_tier"],
            timestamp=data.get("timestamp", 0.0),
            file_states=data.get("file_states", {}),
            files_passed=data.get("files_passed", 0),
            files_failed=data.get("files_failed", 0),
            files_degraded=data.get("files_degraded", 0),
            files_pending=data.get("files_pending", 0),
        )


@dataclass
class CostSnapshot:
    """Accumulated LLM cost at a point in time."""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    calls_by_agent: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": self.total_cost_usd,
            "calls_by_agent": self.calls_by_agent,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CostSnapshot:
        return cls(
            total_input_tokens=data.get("total_input_tokens", 0),
            total_output_tokens=data.get("total_output_tokens", 0),
            total_cost_usd=data.get("total_cost_usd", 0.0),
            calls_by_agent=data.get("calls_by_agent", {}),
        )


class PipelineSession:
    """Manages persistent session state for a single pipeline run.

    Parameters
    ----------
    session_dir : Path
        Directory for this session's checkpoint files.
        Created automatically if it doesn't exist.
    """

    def __init__(self, session_dir: Path) -> None:
        self._dir = session_dir
        self._events_path = session_dir / "events.jsonl"
        self._cost_path = session_dir / "cost.json"
        self._metadata_path = session_dir / "session.json"

    # ── Lifecycle ────────────────────────────────────────────────────

    def initialize(self) -> None:
        """Create the session directory and metadata file."""
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise SessionError(f"Cannot create session directory: {exc}") from exc

    # ── Metadata ─────────────────────────────────────────────────────

    def save_metadata(
        self,
        *,
        prompt: str,
        mode: str = "generate",
        settings_summary: dict[str, Any] | None = None,
    ) -> None:
        """Save run-level metadata (prompt, mode, config snapshot)."""
        self.initialize()
        metadata = {
            "prompt": prompt,
            "mode": mode,
            "start_time": time.time(),
            "settings": settings_summary or {},
        }
        try:
            self._metadata_path.write_text(json.dumps(metadata, indent=2))
        except OSError as exc:
            raise SessionError(f"Cannot save session metadata: {exc}") from exc

    def load_metadata(self) -> dict[str, Any] | None:
        """Load run-level metadata, or None if no session exists."""
        if not self._metadata_path.exists():
            return None
        try:
            return json.loads(self._metadata_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not load session metadata: %s", exc)
            return None

    # ── Checkpoints ──────────────────────────────────────────────────

    def save_checkpoint(
        self,
        engine: Any,  # LifecycleEngine — TYPE_CHECKING import would cause cycle
        completed_tier: int,
    ) -> None:
        """Save a checkpoint snapshot after a tier completes.

        The checkpoint contains the phase and fix counts for every file,
        enabling reconstruction of the LifecycleEngine on resume.
        """
        self.initialize()

        file_states: dict[str, dict[str, Any]] = {}
        # Count every phase individually — previously, non-terminal phases
        # (generating, reviewing, fixing, building, testing) were all lumped
        # into "pending", making checkpoint metrics systematically wrong and
        # potentially causing resume logic to re-execute already-tested files.
        phase_counts: dict[str, int] = {}

        for path in engine._lifecycles:
            lc = engine.get_lifecycle(path)
            file_states[path] = {
                "phase": lc.phase.value,
                "review_fix_count": lc.review_fix_count,
                "test_fix_count": lc.test_fix_count,
                "build_fix_count": lc.build_fix_count,
                "fix_trigger": lc.fix_trigger,
                "tests_generated": lc.tests_generated,
                "generation_task_type": lc.generation_task_type,
            }
            phase_val = lc.phase.value
            phase_counts[phase_val] = phase_counts.get(phase_val, 0) + 1

        # "pending" for checkpoint summary = all non-terminal, non-passed phases
        # This includes: pending, generating, reviewing, fixing, building, testing
        _terminal = {"passed", "failed", "degraded"}
        files_pending = sum(
            count for phase, count in phase_counts.items() if phase not in _terminal
        )

        checkpoint = CheckpointData(
            completed_tier=completed_tier,
            timestamp=time.time(),
            file_states=file_states,
            files_passed=phase_counts.get("passed", 0),
            files_failed=phase_counts.get("failed", 0),
            files_degraded=phase_counts.get("degraded", 0),
            files_pending=files_pending,
        )

        checkpoint_path = self._dir / f"checkpoint_tier_{completed_tier}.json"
        try:
            checkpoint_path.write_text(json.dumps(checkpoint.to_dict(), indent=2))
            logger.info(
                "Saved checkpoint: tier=%d, passed=%d, failed=%d, degraded=%d, pending=%d",
                completed_tier,
                checkpoint.files_passed,
                checkpoint.files_failed,
                checkpoint.files_degraded,
                checkpoint.files_pending,
            )
        except OSError as exc:
            raise SessionError(
                f"Cannot save checkpoint for tier {completed_tier}: {exc}"
            ) from exc

    def load_latest_checkpoint(self) -> CheckpointData | None:
        """Load the most recent checkpoint, or None if no checkpoints exist."""
        if not self._dir.exists():
            return None

        checkpoints = sorted(
            self._dir.glob("checkpoint_tier_*.json"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )
        if not checkpoints:
            return None

        latest = checkpoints[-1]
        try:
            data = json.loads(latest.read_text())
            checkpoint = CheckpointData.from_dict(data)
            logger.info(
                "Loaded checkpoint: tier=%d (from %s)",
                checkpoint.completed_tier,
                latest.name,
            )
            return checkpoint
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            logger.warning("Could not load checkpoint %s: %s", latest, exc)
            return None

    def list_checkpoints(self) -> list[int]:
        """Return sorted list of completed tier numbers."""
        if not self._dir.exists():
            return []
        return sorted(
            int(p.stem.split("_")[-1])
            for p in self._dir.glob("checkpoint_tier_*.json")
        )

    # ── Event log ────────────────────────────────────────────────────

    def append_event(self, event_data: dict[str, Any]) -> None:
        """Append a single event to the JSONL event log."""
        self.initialize()
        try:
            with open(self._events_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event_data) + "\n")
        except OSError:
            pass  # Event logging is best-effort

    def load_events(self) -> list[dict[str, Any]]:
        """Load all events from the JSONL log."""
        if not self._events_path.exists():
            return []
        events = []
        try:
            with open(self._events_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        events.append(json.loads(line))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not load event log: %s", exc)
        return events

    # ── Cost tracking ────────────────────────────────────────────────

    def save_cost(self, cost: CostSnapshot) -> None:
        """Persist accumulated cost data."""
        self.initialize()
        try:
            self._cost_path.write_text(json.dumps(cost.to_dict(), indent=2))
        except OSError:
            pass  # Cost saving is best-effort

    def load_cost(self) -> CostSnapshot | None:
        """Load persisted cost data, or None."""
        if not self._cost_path.exists():
            return None
        try:
            data = json.loads(self._cost_path.read_text())
            return CostSnapshot.from_dict(data)
        except (OSError, json.JSONDecodeError):
            return None

    # ── Cleanup ──────────────────────────────────────────────────────

    @property
    def session_dir(self) -> Path:
        return self._dir

    @property
    def exists(self) -> bool:
        return self._dir.exists() and self._metadata_path.exists()
