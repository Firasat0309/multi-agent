"""Tests for loop_guards wiring into the agentic loop.

Verifies that:
1. build_budget_nudge() produces correct nudge text for all three cases
2. build_end_turn_reminder() returns (text, False) or (None, True)
3. check_stagnation() correctly tracks stagnant iterations
4. BaseAgent._check_stagnation delegates to loop_guards.check_stagnation
5. The agentic loop properly calls the extracted functions
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.loop_guards import build_budget_nudge, build_end_turn_reminder, check_stagnation


def _run(coro):
    return asyncio.run(coro)


# ── build_budget_nudge ────────────────────────────────────────────────────────


class TestBuildBudgetNudge:
    """Unit tests for the budget guard nudge builder."""

    @staticmethod
    def _path_in_written(target, written):
        return target in written

    def test_single_target_nudge_fires_past_halfway(self):
        """Nudge fires when past halfway, target not written, no write call."""
        result = build_budget_nudge(
            target_file="src/main.py",
            iteration=5,
            max_iterations=10,
            files_written=[],
            has_write_call=False,
            consecutive_nudges=0,
            max_consecutive_nudges=2,
            path_in_written_fn=self._path_in_written,
        )
        assert result is not None
        assert "src/main.py" in result
        assert "WARNING" in result

    def test_single_target_no_nudge_before_halfway(self):
        """No nudge before halfway point."""
        result = build_budget_nudge(
            target_file="src/main.py",
            iteration=2,
            max_iterations=10,
            files_written=[],
            has_write_call=False,
            consecutive_nudges=0,
            max_consecutive_nudges=2,
            path_in_written_fn=self._path_in_written,
        )
        assert result is None

    def test_single_target_no_nudge_when_file_written(self):
        """No nudge when target file already written."""
        result = build_budget_nudge(
            target_file="src/main.py",
            iteration=7,
            max_iterations=10,
            files_written=["src/main.py"],
            has_write_call=False,
            consecutive_nudges=0,
            max_consecutive_nudges=2,
            path_in_written_fn=self._path_in_written,
        )
        assert result is None

    def test_single_target_no_nudge_when_has_write_call(self):
        """No nudge when current turn has a write_file call."""
        result = build_budget_nudge(
            target_file="src/main.py",
            iteration=7,
            max_iterations=10,
            files_written=[],
            has_write_call=True,
            consecutive_nudges=0,
            max_consecutive_nudges=2,
            path_in_written_fn=self._path_in_written,
        )
        assert result is None

    def test_single_target_no_nudge_at_max_consecutive(self):
        """No nudge when consecutive nudge cap reached."""
        result = build_budget_nudge(
            target_file="src/main.py",
            iteration=7,
            max_iterations=10,
            files_written=[],
            has_write_call=False,
            consecutive_nudges=2,
            max_consecutive_nudges=2,
            path_in_written_fn=self._path_in_written,
        )
        assert result is None

    def test_multi_file_nudge_fires(self):
        """Multi-file nudge fires when no target, files written, past halfway."""
        result = build_budget_nudge(
            target_file=None,
            iteration=5,
            max_iterations=10,
            files_written=["a.py", "b.py"],
            has_write_call=False,
            consecutive_nudges=0,
            max_consecutive_nudges=2,
            path_in_written_fn=self._path_in_written,
        )
        assert result is not None
        assert "2 file(s)" in result

    def test_zero_writes_nudge_fires(self):
        """Zero-writes nudge fires when no target, no files, past 1/3."""
        result = build_budget_nudge(
            target_file=None,
            iteration=4,
            max_iterations=10,
            files_written=[],
            has_write_call=False,
            consecutive_nudges=0,
            max_consecutive_nudges=2,
            path_in_written_fn=self._path_in_written,
        )
        assert result is not None
        assert "have NOT written any files" in result

    def test_no_nudge_when_no_target_early(self):
        """No nudge when no target and iteration is early."""
        result = build_budget_nudge(
            target_file=None,
            iteration=1,
            max_iterations=10,
            files_written=[],
            has_write_call=False,
            consecutive_nudges=0,
            max_consecutive_nudges=2,
            path_in_written_fn=self._path_in_written,
        )
        assert result is None


# ── build_end_turn_reminder ───────────────────────────────────────────────────


class TestBuildEndTurnReminder:
    @staticmethod
    def _extract_code(text):
        if "```" in text:
            return "some code"
        return None

    def test_returns_reminder_when_under_max(self):
        reminder, exhausted = build_end_turn_reminder(
            target_file="src/app.py",
            response_content="Here is the code...",
            end_turn_reminders=0,
            max_reminders=2,
            extract_code_block_fn=self._extract_code,
        )
        assert reminder is not None
        assert "write_file" in reminder
        assert not exhausted

    def test_returns_reminder_with_code_hint(self):
        reminder, exhausted = build_end_turn_reminder(
            target_file="src/app.py",
            response_content="```python\nprint('hello')\n```",
            end_turn_reminders=0,
            max_reminders=2,
            extract_code_block_fn=self._extract_code,
        )
        assert reminder is not None
        assert "already wrote the code" in reminder
        assert not exhausted

    def test_returns_exhausted_at_max(self):
        reminder, exhausted = build_end_turn_reminder(
            target_file="src/app.py",
            response_content="whatever",
            end_turn_reminders=2,
            max_reminders=2,
            extract_code_block_fn=self._extract_code,
        )
        assert reminder is None
        assert exhausted

    def test_incremental_reminders(self):
        """Calling with 0, then 1 should both return reminders."""
        r0, _ = build_end_turn_reminder("f.py", "", 0, 2, self._extract_code)
        r1, _ = build_end_turn_reminder("f.py", "", 1, 2, self._extract_code)
        assert r0 is not None
        assert r1 is not None


# ── check_stagnation ─────────────────────────────────────────────────────────


@dataclass
class FakeTC:
    name: str


class TestCheckStagnation:
    def test_resets_on_successful_write(self):
        tcs = [FakeTC("write_file")]
        results = ["Written 500 bytes"]
        new_count, stop = check_stagnation(tcs, results, 5, 2, "test", ["a.py"])
        assert new_count == 0
        assert not stop

    def test_does_not_reset_on_failed_write(self):
        tcs = [FakeTC("write_file")]
        results = ["Error: permission denied"]
        new_count, stop = check_stagnation(tcs, results, 0, 2, "test", ["a.py"])
        assert new_count > 0

    def test_half_increment_on_read_calls(self):
        tcs = [FakeTC("read_file")]
        results = ["content of file"]
        new_count, stop = check_stagnation(tcs, results, 0, 2, "test", ["a.py"])
        assert new_count == 0.5
        assert not stop

    def test_full_increment_on_idle(self):
        tcs = [FakeTC("some_other_tool")]
        results = ["ok"]
        new_count, stop = check_stagnation(tcs, results, 0, 2, "test", ["a.py"])
        assert new_count == 1
        assert not stop

    def test_stops_at_limit(self):
        tcs = [FakeTC("some_other_tool")]
        results = ["ok"]
        new_count, stop = check_stagnation(tcs, results, 1, 2, "test", ["a.py"])
        assert new_count == 2
        assert stop

    def test_half_rate_needs_double_limit(self):
        """Read-only stagnation at half rate needs more iterations to trigger."""
        tcs = [FakeTC("read_file")]
        results = ["content"]
        count = 0.0
        for _ in range(3):
            count, stop = check_stagnation(tcs, results, count, 2, "test", ["a.py"])
            if stop:
                break
        # At half rate: 0.5, 1.0, 1.5 — should NOT have stopped yet
        assert not stop
        assert count == 1.5
        # One more should trigger
        count, stop = check_stagnation(tcs, results, count, 2, "test", ["a.py"])
        assert stop


# ── BaseAgent._check_stagnation delegation ────────────────────────────────────


class TestBaseAgentDelegation:
    def test_check_stagnation_delegates(self):
        """BaseAgent._check_stagnation should produce identical results."""
        from agents.base_agent import BaseAgent

        tcs = [FakeTC("write_file")]
        results = ["Written 100 bytes"]
        module_result = check_stagnation(tcs, results, 3.0, 2, "test", ["a.py"])
        agent_result = BaseAgent._check_stagnation(tcs, results, 3.0, 2, "test", ["a.py"])
        assert module_result == agent_result

    def test_check_stagnation_delegates_stop(self):
        from agents.base_agent import BaseAgent

        tcs = [FakeTC("other")]
        results = ["ok"]
        module_result = check_stagnation(tcs, results, 1, 2, "test", ["a.py"])
        agent_result = BaseAgent._check_stagnation(tcs, results, 1, 2, "test", ["a.py"])
        assert module_result == agent_result
        assert module_result[1] is True  # should_stop
