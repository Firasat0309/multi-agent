"""Shared conversation context — cross-agent knowledge accumulation.

In the reference codebase (Claude Code), all agents operate within a shared
conversation that automatically accumulates insights. In our multi-agent system,
agents run in isolation — each starts fresh.

``SharedContext`` bridges this gap by providing a thread-safe store where agents
can deposit and retrieve insights, decisions, and learned facts. This context
is injected into agent prompts so downstream agents benefit from upstream
discoveries.

Usage::

    ctx = SharedContext()

    # Agent A discovers something
    ctx.add_insight("architect", "Using hexagonal architecture for isolation")
    ctx.add_decision("architect", "Spring Boot 3.2 with Java 21", reason="LTS")

    # Agent B consumes accumulated context
    prompt_prefix = ctx.build_context_prompt(for_agent="coder")
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Insight:
    """A single piece of knowledge from an agent."""
    agent: str
    content: str
    category: str = "general"
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class Decision:
    """A recorded architectural or implementation decision."""
    agent: str
    decision: str
    reason: str = ""
    timestamp: float = field(default_factory=time.time)


class SharedContext:
    """Thread-safe cross-agent context accumulator.

    Collects insights and decisions from all agents throughout a pipeline run.
    Each agent can query the accumulated context to inform its own prompts.
    """

    def __init__(self, max_insights: int = 200, max_decisions: int = 50) -> None:
        self._lock = threading.Lock()
        self._insights: list[Insight] = []
        self._decisions: list[Decision] = []
        self._file_notes: dict[str, list[str]] = {}  # path → notes
        self._max_insights = max_insights
        self._max_decisions = max_decisions

    # ── Accumulation ─────────────────────────────────────────────────

    def add_insight(self, agent: str, content: str, category: str = "general") -> None:
        """Record an insight from an agent."""
        with self._lock:
            self._insights.append(Insight(agent=agent, content=content, category=category))
            if len(self._insights) > self._max_insights:
                self._insights = self._insights[-self._max_insights:]

    def add_decision(self, agent: str, decision: str, reason: str = "") -> None:
        """Record an architectural/implementation decision."""
        with self._lock:
            self._decisions.append(Decision(agent=agent, decision=decision, reason=reason))
            if len(self._decisions) > self._max_decisions:
                self._decisions = self._decisions[-self._max_decisions:]

    def add_file_note(self, file_path: str, note: str) -> None:
        """Attach a note to a specific file (e.g., 'uses singleton pattern')."""
        with self._lock:
            self._file_notes.setdefault(file_path, []).append(note)

    # ── Querying ─────────────────────────────────────────────────────

    def get_insights(
        self,
        *,
        category: str | None = None,
        agent: str | None = None,
        limit: int = 50,
    ) -> list[Insight]:
        """Retrieve insights, optionally filtered."""
        with self._lock:
            results = self._insights
            if category:
                results = [i for i in results if i.category == category]
            if agent:
                results = [i for i in results if i.agent == agent]
            return results[-limit:]

    def get_decisions(self, limit: int = 30) -> list[Decision]:
        """Retrieve all recorded decisions."""
        with self._lock:
            return self._decisions[-limit:]

    def get_file_notes(self, file_path: str) -> list[str]:
        """Get notes attached to a specific file."""
        with self._lock:
            return list(self._file_notes.get(file_path, []))

    # ── Prompt building ──────────────────────────────────────────────

    def build_context_prompt(
        self,
        *,
        for_agent: str = "",
        relevant_files: list[str] | None = None,
        max_chars: int = 4000,
    ) -> str:
        """Build a context block suitable for injection into an agent prompt.

        Args:
            for_agent: The consuming agent's name (excluded from insights
                to avoid self-referencing).
            relevant_files: If provided, include file-specific notes for
                these paths.
            max_chars: Maximum characters for the context block.
        """
        parts: list[str] = []

        # Decisions (always included — they're compact and high-value)
        decisions = self.get_decisions()
        if decisions:
            dec_lines = []
            for d in decisions:
                line = f"  - [{d.agent}] {d.decision}"
                if d.reason:
                    line += f" (reason: {d.reason})"
                dec_lines.append(line)
            parts.append("DECISIONS MADE:\n" + "\n".join(dec_lines))

        # Insights from other agents
        insights = self.get_insights()
        if for_agent:
            insights = [i for i in insights if i.agent != for_agent]
        if insights:
            ins_lines = [f"  - [{i.agent}/{i.category}] {i.content}" for i in insights[-20:]]
            parts.append("INSIGHTS FROM OTHER AGENTS:\n" + "\n".join(ins_lines))

        # File-specific notes
        if relevant_files:
            file_parts = []
            for fp in relevant_files:
                notes = self.get_file_notes(fp)
                if notes:
                    file_parts.append(f"  {fp}: " + "; ".join(notes))
            if file_parts:
                parts.append("FILE NOTES:\n" + "\n".join(file_parts))

        result = "\n\n".join(parts)
        if len(result) > max_chars:
            result = result[:max_chars] + "\n... [context truncated]"
        return result

    # ── Serialization ────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize context for persistence."""
        with self._lock:
            return {
                "insights": [
                    {"agent": i.agent, "content": i.content, "category": i.category}
                    for i in self._insights
                ],
                "decisions": [
                    {"agent": d.agent, "decision": d.decision, "reason": d.reason}
                    for d in self._decisions
                ],
                "file_notes": dict(self._file_notes),
            }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SharedContext:
        """Restore from serialized form."""
        ctx = cls()
        for i in data.get("insights", []):
            ctx.add_insight(i["agent"], i["content"], i.get("category", "general"))
        for d in data.get("decisions", []):
            ctx.add_decision(d["agent"], d["decision"], d.get("reason", ""))
        for path, notes in data.get("file_notes", {}).items():
            for note in notes:
                ctx.add_file_note(path, note)
        return ctx

    def __len__(self) -> int:
        with self._lock:
            return len(self._insights) + len(self._decisions)

    @property
    def summary(self) -> str:
        """Brief summary of accumulated context."""
        with self._lock:
            return (
                f"{len(self._insights)} insights, "
                f"{len(self._decisions)} decisions, "
                f"{len(self._file_notes)} annotated files"
            )
