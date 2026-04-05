"""Fix loop — extracted from SimpleLoopExecutor for single-responsibility.

Encapsulates the generate → build → fix → escalate cycle for a single file,
including error history tracking, escalation strategies, and cross-file learning.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any

from core.feature_flags import feature

logger = logging.getLogger(__name__)


@dataclass
class ErrorRecord:
    """A single build/review error observation."""
    errors_text: str
    error_hash: str
    attempt: int
    referenced_files: list[str] = field(default_factory=list)
    error_signatures: list[str] = field(default_factory=list)


class FixLoop:
    """Manages error history and escalation strategy for a single file.

    Responsibilities:
      - Track error history across fix iterations
      - Detect stalled progress (same hash, oscillating hashes, similar text)
      - Select escalation strategy based on stall count
      - Build enriched fix metadata with memory, cross-file hints, escalation
    """

    def __init__(self, file_path: str, max_attempts: int = 5, max_error_chars: int = 8000) -> None:
        self.file_path = file_path
        self.max_attempts = max_attempts
        self.max_error_chars = max_error_chars
        self.error_history: list[ErrorRecord] = []
        self.fix_count = 0

    # ── Error recording ─────────────────────────────────────────────────

    def record_error(
        self,
        errors_text: str,
        attempt: int,
        referenced_files: list[str] | None = None,
        error_signatures: list[str] | None = None,
    ) -> ErrorRecord:
        """Record a build/review error and return the record."""
        error_hash = hashlib.md5(errors_text.encode()).hexdigest()[:8]
        record = ErrorRecord(
            errors_text=errors_text,
            error_hash=error_hash,
            attempt=attempt,
            referenced_files=referenced_files or [],
            error_signatures=error_signatures or [],
        )
        self.error_history.append(record)
        return record

    def record_review_findings(self, findings: list[str], attempt: int) -> ErrorRecord:
        """Record code review findings as error context."""
        review_text = (
            "🔍 CODE REVIEW found semantic issues "
            "(these may NOT show up as compiler errors):\n"
            + "\n".join(f"  - {f}" for f in findings)
        )
        return self.record_error(review_text, attempt)

    # ── Stall detection ─────────────────────────────────────────────────

    def is_stalled(self) -> bool:
        """Detect whether the fix loop is making no progress."""
        if len(self.error_history) < 2:
            return False
        return self._detect_stall()[0]

    @property
    def stall_count(self) -> int:
        """Number of consecutive stalled attempts."""
        return self._detect_stall()[1]

    def _detect_stall(self) -> tuple[bool, int]:
        """Return (is_stalled, consecutive_stall_count)."""
        if len(self.error_history) < 2:
            return False, 0

        prev = self.error_history[-2]
        curr = self.error_history[-1]

        # Exact same errors
        if curr.error_hash == prev.error_hash:
            count = self._count_consecutive_stalls()
            return True, count

        # Oscillating between two error states
        if len(self.error_history) >= 3:
            recent_hashes = [h.error_hash for h in self.error_history[-3:]]
            if len(set(recent_hashes)) <= 2:
                logger.warning(
                    "[%s] Oscillating between %d error states",
                    self.file_path, len(set(recent_hashes)),
                )
                return True, self._count_consecutive_stalls()

        # Fuzzy similarity check (>60% line overlap)
        prev_lines = set(prev.errors_text.splitlines())
        curr_lines = set(curr.errors_text.splitlines())
        if prev_lines and curr_lines:
            overlap = len(prev_lines & curr_lines)
            total = max(len(prev_lines), len(curr_lines))
            if total > 0 and overlap / total > 0.6:
                logger.warning(
                    "[%s] Error similarity %.0f%%",
                    self.file_path, 100 * overlap / total,
                )
                return True, self._count_consecutive_stalls()

        return False, 0

    def _count_consecutive_stalls(self) -> int:
        """Count how many consecutive attempts produced the same error hash."""
        count = 0
        for i in range(len(self.error_history) - 1, 0, -1):
            if self.error_history[i].error_hash == self.error_history[i - 1].error_hash:
                count += 1
            else:
                break
        return count

    # ── Escalation ──────────────────────────────────────────────────────

    def get_escalation_prefix(self) -> str:
        """Get the escalation prefix for the current stall level, or empty string."""
        stalled, stall_n = self._detect_stall()
        if not stalled:
            return ""

        if stall_n >= 3:
            return (
                f"⚠️ FIX LOOP STALLED ({stall_n + 1} identical attempts). "
                "RADICAL SIMPLIFICATION REQUIRED:\n"
                "- Strip the failing section to the SIMPLEST possible "
                "implementation that compiles\n"
                "- Use stub/TODO implementations for complex logic\n"
                "- Remove any clever abstractions — use plain, direct code\n"
                "- If a dependency is causing issues, remove it and "
                "hard-code the value\n\n"
            )
        elif stall_n >= 2:
            return (
                f"⚠️ FIX LOOP STALLED ({stall_n + 1} identical attempts). "
                "REWRITE FROM SCRATCH:\n"
                "- Do NOT patch the existing code — delete the failing "
                "method/block entirely and rewrite it\n"
                "- Re-read the dependency signatures in Related Files "
                "and match them EXACTLY\n"
                "- If the approach is fundamentally wrong, use an "
                "alternative algorithm or pattern\n\n"
            )
        else:
            return (
                "⚠️ PREVIOUS FIX ATTEMPT DID NOT RESOLVE THE ERRORS — "
                "THE SAME BUILD ERRORS PERSIST.\n"
                "You MUST try a COMPLETELY DIFFERENT approach:\n"
                "- If you changed a method call, check the actual method "
                "signature in the Related Files section\n"
                "- If an import is wrong, check the actual package path\n"
                "- If a type mismatch, read the full error to understand "
                "which types are incompatible\n\n"
            )

    # ── Fix metadata assembly ───────────────────────────────────────────

    def build_fix_metadata(
        self,
        *,
        persistent_memory: str = "",
        cross_file_hints: str = "",
    ) -> dict[str, Any]:
        """Assemble the full fix metadata dict for the CoderAgent.

        Combines: latest errors, escalation prefix, error history summary,
        persistent memory, cross-file hints, and streaming-fix signatures.
        """
        latest = self.error_history[-1] if self.error_history else None
        if not latest:
            return {}

        errors_text = latest.errors_text
        referenced_files = latest.referenced_files

        # Apply escalation prefix if stalled
        escalation = self.get_escalation_prefix()
        if escalation:
            errors_text = escalation + errors_text

        # Build error history summary
        history_summary = ""
        if len(self.error_history) >= 2:
            prev_attempts = self.error_history[:-1]
            history_lines = []
            for h in prev_attempts[-3:]:
                history_lines.append(
                    f"  Attempt {h.attempt}: {h.errors_text[:300]}..."
                )
            history_summary = (
                "PREVIOUS FIX ATTEMPTS (learn from these — do NOT repeat the same fix):\n"
                + "\n".join(history_lines)
            )

        if persistent_memory:
            history_summary = (
                history_summary + "\n\n" if history_summary else ""
            ) + persistent_memory

        if cross_file_hints:
            history_summary = (
                history_summary + "\n\n" if history_summary else ""
            ) + cross_file_hints

        metadata: dict[str, Any] = {
            "build_errors": errors_text + ("\n\n" + history_summary if history_summary else ""),
            "fix_trigger": "build",
            "fix_attempt": latest.attempt,
            "max_fix_attempts": self.max_attempts,
        }
        if referenced_files:
            metadata["referenced_files"] = referenced_files
        if escalation:
            metadata["escalate_fix"] = True

        # Streaming fix: inject known bad patterns
        if feature("STREAMING_FIX"):
            all_sigs: list[str] = []
            for h in self.error_history:
                all_sigs.extend(h.error_signatures)
            if all_sigs:
                seen_sigs: set[str] = set()
                unique_sigs = []
                for s in all_sigs:
                    if s not in seen_sigs:
                        seen_sigs.add(s)
                        unique_sigs.append(s)
                metadata["known_bad_patterns"] = unique_sigs[:5]
                metadata["build_errors"] = (
                    metadata["build_errors"]
                    + "\n\n⚠️ KNOWN BAD PATTERNS (do NOT reproduce these in your fix):\n"
                    + "\n".join(f"  • {s}" for s in unique_sigs[:5])
                )

        return metadata

    @property
    def latest_error_hash(self) -> str:
        """Hash of the most recent error, or empty string."""
        return self.error_history[-1].error_hash if self.error_history else ""

    @property
    def latest_errors_text(self) -> str:
        """Text of the most recent error, or empty string."""
        return self.error_history[-1].errors_text if self.error_history else ""

    @property
    def error_history_dicts(self) -> list[dict[str, Any]]:
        """Convert error history to plain dicts (for backward compatibility)."""
        return [
            {
                "errors_text": r.errors_text,
                "error_hash": r.error_hash,
                "attempt": r.attempt,
                "referenced_files": r.referenced_files,
                "error_signatures": r.error_signatures,
            }
            for r in self.error_history
        ]
