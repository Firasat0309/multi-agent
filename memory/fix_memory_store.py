"""Persistent fix-attempt memory for the simple generate-build-fix loop."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class FixMemoryStore:
    """Stores recent fix attempts and resolutions across runs.

    The simple loop uses this to avoid repeating the same failed fix strategy
    on the next run for the same file.
    """

    def __init__(
        self,
        workspace: Path,
        *,
        max_errors_per_file: int = 5,
        max_resolutions_per_file: int = 3,
    ) -> None:
        self._path = workspace / ".simple_loop_memory.json"
        self._max_errors = max_errors_per_file
        self._max_resolutions = max_resolutions_per_file
        self._lock = asyncio.Lock()
        self._state = self._load_state()

    def _load_state(self) -> dict[str, Any]:
        if not self._path.exists():
            return {"version": 1, "files": {}}
        try:
            state = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(state, dict) and isinstance(state.get("files"), dict):
                return state
        except Exception:
            logger.warning("Fix memory load failed — starting fresh", exc_info=True)
        return {"version": 1, "files": {}}

    def _save_state(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._state, indent=2), encoding="utf-8")

    async def record_attempt(
        self,
        file_path: str,
        *,
        errors_text: str,
        error_hash: str,
        referenced_files: list[str] | None = None,
    ) -> None:
        async with self._lock:
            file_state = self._state.setdefault("files", {}).setdefault(
                file_path,
                {"recent_errors": [], "recent_resolutions": []},
            )
            file_state["recent_errors"].append({
                "timestamp": time.time(),
                "error_hash": error_hash,
                "errors_text": errors_text[:1200],
                "referenced_files": referenced_files or [],
            })
            file_state["recent_errors"] = file_state["recent_errors"][-self._max_errors :]
            await asyncio.to_thread(self._save_state)

    async def record_resolution(self, file_path: str, summary: str) -> None:
        async with self._lock:
            file_state = self._state.setdefault("files", {}).setdefault(
                file_path,
                {"recent_errors": [], "recent_resolutions": []},
            )
            file_state["recent_resolutions"].append({
                "timestamp": time.time(),
                "summary": summary[:600],
            })
            file_state["recent_resolutions"] = file_state["recent_resolutions"][-self._max_resolutions :]
            await asyncio.to_thread(self._save_state)

    async def get_summary(self, file_path: str) -> str:
        async with self._lock:
            file_state = self._state.get("files", {}).get(file_path)
            if not file_state:
                return ""

            lines: list[str] = []
            recent_errors = file_state.get("recent_errors", [])[-3:]
            if recent_errors:
                lines.append("PERSISTENT FIX MEMORY — recent failed attempts:")
                for entry in recent_errors:
                    lines.append(
                        f"  - {entry.get('error_hash', 'unknown')}: {entry.get('errors_text', '')[:240]}"
                    )
                    referenced_files = entry.get("referenced_files", [])
                    if referenced_files:
                        lines.append(
                            "    Referenced files: " + ", ".join(referenced_files[:5])
                        )

            recent_resolutions = file_state.get("recent_resolutions", [])[-2:]
            if recent_resolutions:
                lines.append("PERSISTENT FIX MEMORY — recent successful resolutions:")
                for entry in recent_resolutions:
                    lines.append(f"  - {entry.get('summary', '')}")

            return "\n".join(lines)