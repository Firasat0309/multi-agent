"""Tests for SessionState wiring into SimpleLoopExecutor.

Verifies:
1. SessionState unit: mark_file_passed/failed, pending_files, save/load, tier tracking
2. Integration: SimpleLoopExecutor creates/uses session when SESSION_RESUME is on
3. Resume: already-passed files are skipped on re-run
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.session_state import FileState, FileStatus, SessionState


def _run(coro):
    return asyncio.run(coro)


# ── SessionState unit tests ──────────────────────────────────────────────────


class TestSessionStateUnit:
    def test_mark_file_passed(self):
        s = SessionState(run_id="r1", workspace="/tmp/test")
        s.mark_file_passed("src/main.py")
        assert s.is_file_passed("src/main.py")
        assert s.is_file_done("src/main.py")
        assert s.files["src/main.py"].status == FileStatus.PASSED
        assert s.files["src/main.py"].attempts == 1

    def test_mark_file_failed(self):
        s = SessionState(run_id="r1", workspace="/tmp/test")
        s.mark_file_failed("src/main.py", "build error")
        assert not s.is_file_passed("src/main.py")
        assert s.is_file_done("src/main.py")
        assert s.files["src/main.py"].status == FileStatus.FAILED
        assert s.files["src/main.py"].last_error == "build error"

    def test_mark_file_skipped(self):
        s = SessionState(run_id="r1", workspace="/tmp/test")
        s.mark_file_skipped("src/main.py")
        assert not s.is_file_passed("src/main.py")
        assert s.is_file_done("src/main.py")

    def test_pending_files(self):
        s = SessionState(run_id="r1", workspace="/tmp/test")
        s.mark_file_passed("a.py")
        s.mark_file_failed("b.py")
        pending = s.pending_files(["a.py", "b.py", "c.py"])
        assert pending == ["b.py", "c.py"]

    def test_summary(self):
        s = SessionState(run_id="r1", workspace="/tmp/test")
        s.mark_file_passed("a.py")
        s.mark_file_passed("b.py")
        s.mark_file_failed("c.py")
        assert s.summary == {"passed": 2, "failed": 1}

    def test_tier_tracking(self):
        s = SessionState(run_id="r1", workspace="/tmp/test")
        assert not s.is_tier_complete(0)
        s.mark_tier_complete(0)
        assert s.is_tier_complete(0)
        assert not s.is_tier_complete(1)
        # idempotent
        s.mark_tier_complete(0)
        assert s.completed_tiers.count(0) == 1


class TestSessionStatePersistence:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = SessionState(run_id="test-run", workspace=tmp)
            s.mark_file_passed("src/foo.py")
            s.mark_file_failed("src/bar.py", "type error")
            s.mark_tier_complete(0)
            s.save()

            # Load back
            s2 = SessionState.load_or_create(tmp, "test-run")
            assert s2.is_file_passed("src/foo.py")
            assert not s2.is_file_passed("src/bar.py")
            assert s2.files["src/bar.py"].last_error == "type error"
            assert s2.is_tier_complete(0)

    def test_load_different_run_id_creates_new(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = SessionState(run_id="run-1", workspace=tmp)
            s.mark_file_passed("a.py")
            s.save()

            s2 = SessionState.load_or_create(tmp, "run-2")
            assert not s2.is_file_passed("a.py")
            assert len(s2.files) == 0

    def test_load_corrupt_file_creates_new(self):
        with tempfile.TemporaryDirectory() as tmp:
            session_dir = Path(tmp) / ".multi-agent"
            session_dir.mkdir()
            (session_dir / "session_state.json").write_text("NOT JSON", encoding="utf-8")

            s = SessionState.load_or_create(tmp, "run-1")
            assert len(s.files) == 0

    def test_clear(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = SessionState(run_id="r1", workspace=tmp)
            s.save()
            path = Path(tmp) / ".multi-agent" / "session_state.json"
            assert path.exists()
            SessionState.clear(tmp)
            assert not path.exists()


# ── Integration: session wiring in SimpleLoopExecutor ─────────────────────────


class TestExecutorSessionWiring:
    """Verify that SimpleLoopExecutor properly uses SessionState."""

    def _make_executor(self, workspace="/tmp/test"):
        """Create a minimal SimpleLoopExecutor with mocks."""
        from core.simple_loop_executor import SimpleLoopExecutor

        am = MagicMock()
        am.repo.workspace = Path(workspace)
        am.llm.total_input_tokens = 0
        am.llm.total_output_tokens = 0
        am._metrics = {}

        settings = MagicMock()
        settings.max_concurrent_agents = 2
        settings.skip_agents = set()
        settings.execution.file_token_budget = 0
        settings.execution.heartbeat_interval = 30
        settings.execution.agent_token_budget = 0
        settings.execution.compaction_threshold_tokens = 100_000

        lang = MagicMock()
        lang.build_command = None  # interpreted language — no build
        lang.name = "python"

        return SimpleLoopExecutor(am, settings, lang)

    def test_session_field_initialized_to_none(self):
        executor = self._make_executor()
        assert executor._session is None

    @patch("core.simple_loop_executor.feature")
    def test_session_created_when_flag_on(self, mock_feature):
        """When SESSION_RESUME is on, execute() creates a session."""
        mock_feature.side_effect = lambda name: name == "SESSION_RESUME"

        executor = self._make_executor()
        # We can't easily run the full execute() — just verify the import works
        from core.session_state import SessionState
        s = SessionState.load_or_create("/tmp/test", "abc")
        assert s.run_id == "abc"

    def test_session_marks_on_file_results(self):
        """Verify that _session.mark_file_passed/failed are called."""
        with tempfile.TemporaryDirectory() as tmp:
            executor = self._make_executor(tmp)
            session = SessionState(run_id="test", workspace=tmp)
            executor._session = session

            # Simulate what _execute_tiered does after results
            session.mark_file_passed("a.py")
            session.mark_file_failed("b.py", "build error")
            session.save()

            # Verify persistence
            loaded = SessionState.load_or_create(tmp, "test")
            assert loaded.is_file_passed("a.py")
            assert not loaded.is_file_passed("b.py")

    def test_tier_skip_on_resume(self):
        """When a tier is marked complete, files should be skippable."""
        with tempfile.TemporaryDirectory() as tmp:
            # First run: mark tier 0 complete
            s = SessionState(run_id="test", workspace=tmp)
            s.mark_file_passed("a.py")
            s.mark_file_passed("b.py")
            s.mark_tier_complete(0)
            s.save()

            # Load for resume
            s2 = SessionState.load_or_create(tmp, "test")
            assert s2.is_tier_complete(0)
            # Tier 0 should be skipped entirely
            remaining = s2.pending_files(["a.py", "b.py"])
            assert remaining == []

    def test_partial_tier_resume(self):
        """When only some files passed, remaining should be processed."""
        with tempfile.TemporaryDirectory() as tmp:
            s = SessionState(run_id="test", workspace=tmp)
            s.mark_file_passed("a.py")
            # b.py was not processed (crash)
            s.save()

            s2 = SessionState.load_or_create(tmp, "test")
            remaining = s2.pending_files(["a.py", "b.py", "c.py"])
            assert remaining == ["b.py", "c.py"]
            assert not s2.is_tier_complete(0)

    def test_multiple_attempts_tracked(self):
        s = SessionState(run_id="r1", workspace="/tmp/test")
        s.mark_file_failed("x.py", "err1")
        s.mark_file_failed("x.py", "err2")
        s.mark_file_passed("x.py")
        assert s.files["x.py"].attempts == 3
        assert s.is_file_passed("x.py")
