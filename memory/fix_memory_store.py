"""Persistent fix-attempt memory for the simple generate-build-fix loop."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Error pattern categories for semantic matching ───────────────────────────
# Each category maps to a set of regex patterns that match common compiler errors.
# This enables cross-file fix hints even when the exact error_hash differs.

_ERROR_CATEGORIES: dict[str, list[re.Pattern[str]]] = {
    "type_mismatch": [
        re.compile(r"incompatible types|type mismatch|cannot convert|cannot be applied", re.I),
        re.compile(r"expected .+ but (found|got)", re.I),
    ],
    "missing_import": [
        re.compile(r"cannot find symbol|cannot resolve|undefined reference", re.I),
        re.compile(r"no such module|module not found|import .+ could not be resolved", re.I),
    ],
    "null_reference": [
        re.compile(r"null pointer|NullPointerException|cannot read propert", re.I),
        re.compile(r"optional.*must be unwrapped|not null assertion", re.I),
    ],
    "missing_method": [
        re.compile(r"method .+ (does not exist|is not defined|cannot be found)", re.I),
        re.compile(r"has no (member|attribute|method|property)", re.I),
    ],
    "constructor_error": [
        re.compile(r"constructor .+ (is not defined|cannot be applied)", re.I),
        re.compile(r"no suitable constructor|cannot instantiate", re.I),
    ],
    "syntax_error": [
        re.compile(r"syntax error|unexpected token|expected .+ before", re.I),
        re.compile(r"unterminated|unclosed|missing (semicolon|bracket|brace)", re.I),
    ],
    "dependency_error": [
        re.compile(r"package .+ does not exist|cannot find.*package", re.I),
        re.compile(r"dependency .+ not found|unresolved dependency", re.I),
    ],
}


def _categorize_error(errors_text: str) -> list[str]:
    """Return a list of error categories matching the given error text."""
    categories = []
    for cat, patterns in _ERROR_CATEGORIES.items():
        if any(p.search(errors_text) for p in patterns):
            categories.append(cat)
    return categories


class FixMemoryStore:
    """Stores recent fix attempts and resolutions across runs.

    The simple loop uses this to avoid repeating the same failed fix strategy
    on the next run for the same file.

    Cross-file learning: when a file is fixed successfully, the resolution
    pattern is stored globally so that OTHER files hitting the same error
    class can benefit from the solution without re-discovering it.
    """

    def __init__(
        self,
        workspace: Path,
        *,
        max_errors_per_file: int = 5,
        max_resolutions_per_file: int = 3,
        max_global_patterns: int = 20,
    ) -> None:
        self._path = workspace / ".simple_loop_memory.json"
        self._max_errors = max_errors_per_file
        self._max_resolutions = max_resolutions_per_file
        self._max_global = max_global_patterns
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
            categories = _categorize_error(errors_text)
            file_state["recent_errors"].append({
                "timestamp": time.time(),
                "error_hash": error_hash,
                "errors_text": errors_text[:2400],
                "referenced_files": referenced_files or [],
                "categories": categories,
            })
            file_state["recent_errors"] = file_state["recent_errors"][-self._max_errors :]
            await asyncio.to_thread(self._save_state)

    async def record_resolution(
        self, file_path: str, summary: str, *, error_hash: str = "",
        errors_text: str = "",
    ) -> None:
        async with self._lock:
            file_state = self._state.setdefault("files", {}).setdefault(
                file_path,
                {"recent_errors": [], "recent_resolutions": []},
            )
            categories = _categorize_error(errors_text) if errors_text else []
            file_state["recent_resolutions"].append({
                "timestamp": time.time(),
                "summary": summary[:600],
                "categories": categories,
            })
            file_state["recent_resolutions"] = file_state["recent_resolutions"][-self._max_resolutions :]

            # Cross-file learning: store the resolution globally keyed by
            # error_hash so other files with the same error class can benefit.
            if error_hash and summary:
                global_patterns = self._state.setdefault("global_patterns", [])
                global_patterns.append({
                    "timestamp": time.time(),
                    "error_hash": error_hash,
                    "resolution": summary[:400],
                    "source_file": file_path,
                    "categories": categories,
                })
                # Keep only recent patterns
                self._state["global_patterns"] = global_patterns[-self._max_global :]

            await asyncio.to_thread(self._save_state)

    async def get_cross_file_hints(self, error_hash: str) -> str:
        """Get resolution hints from OTHER files that fixed the same error class.

        This is the key cross-file learning mechanism: if FileA had error X
        and was fixed with approach Y, when FileB hits the same error X,
        the fix agent sees Y as a suggested approach.
        """
        async with self._lock:
            patterns = self._state.get("global_patterns", [])
            matches = [
                p for p in patterns
                if p.get("error_hash") == error_hash
            ]
            if not matches:
                return ""
            hints = []
            for p in matches[-3:]:  # Last 3 matches
                hints.append(
                    f"  - {p.get('source_file', 'unknown')}: {p.get('resolution', '')}"
                )
            return (
                "CROSS-FILE FIX HINTS — other files resolved this same error type:\n"
                + "\n".join(hints)
            )

    async def get_semantic_hints(self, errors_text: str) -> str:
        """Get resolution hints based on semantic error category matching.

        Unlike ``get_cross_file_hints`` which matches exact error_hash,
        this method categorizes the error (type_mismatch, missing_import,
        etc.) and retrieves resolutions from ANY file that fixed a similar
        category of error.  This produces broader, more useful hints.

        Gated behind the ``ENHANCED_FIX_MEMORY`` feature flag.
        """
        categories = _categorize_error(errors_text)
        if not categories:
            return ""

        async with self._lock:
            patterns = self._state.get("global_patterns", [])
            matches = []
            for p in patterns:
                p_cats = p.get("categories", [])
                if any(c in p_cats for c in categories):
                    matches.append(p)

            if not matches:
                return ""

            # Deduplicate by resolution text
            seen: set[str] = set()
            unique: list[dict[str, Any]] = []
            for m in matches:
                res = m.get("resolution", "")
                if res not in seen:
                    seen.add(res)
                    unique.append(m)

            hints = []
            for p in unique[-5:]:  # Last 5 unique matches
                cats = ", ".join(p.get("categories", []))
                hints.append(
                    f"  - [{cats}] {p.get('source_file', '?')}: {p.get('resolution', '')}"
                )

            matched_cats = ", ".join(categories)
            return (
                f"SEMANTIC FIX HINTS (error categories: {matched_cats}):\n"
                + "\n".join(hints)
            )

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