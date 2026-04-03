"""Property-based tests for the FileLifecycle state machine.

Uses hypothesis to generate random event sequences and verify that
the state machine never enters an impossible state.
"""

from __future__ import annotations

import pytest

from core.state_machine import (
    EventType,
    FileLifecycle,
    FilePhase,
)

# All events that can drive transitions
ALL_EVENTS = list(EventType)

# Terminal phases — no transitions out of these
TERMINAL_PHASES = {FilePhase.PASSED, FilePhase.FAILED, FilePhase.DEGRADED}

hypothesis = pytest.importorskip("hypothesis")
given = hypothesis.given
settings = hypothesis.settings
st = pytest.importorskip("hypothesis.strategies")


class TestStateMachineProperties:
    """Property-based tests for FileLifecycle invariants."""

    @given(
        events=st.lists(
            st.sampled_from(ALL_EVENTS),
            min_size=1,
            max_size=30,
        )
    )
    @settings(max_examples=200)
    def test_never_exits_terminal_state(self, events: list[EventType]) -> None:
        """Once a FileLifecycle reaches a terminal phase, no event should move
        it to a non-terminal phase."""
        lc = FileLifecycle("test.py", max_review_fixes=2, max_test_fixes=2, max_build_fixes=2)

        terminal_reached_at: FilePhase | None = None
        for ev in events:
            if lc.is_terminal:
                terminal_reached_at = lc.phase
                try:
                    lc.process_event(ev)
                except ValueError:
                    pass  # invalid transition — fine
                # Must remain in a terminal phase
                assert lc.phase in TERMINAL_PHASES, (
                    f"Escaped terminal {terminal_reached_at} via {ev.value} → {lc.phase.value}"
                )
            else:
                try:
                    lc.process_event(ev)
                except ValueError:
                    pass  # invalid transition is OK

    @given(
        events=st.lists(
            st.sampled_from(ALL_EVENTS),
            min_size=1,
            max_size=30,
        )
    )
    @settings(max_examples=200)
    def test_phase_always_valid(self, events: list[EventType]) -> None:
        """Phase is always a member of FilePhase after any sequence of events."""
        lc = FileLifecycle("test.py")
        for ev in events:
            try:
                lc.process_event(ev)
            except ValueError:
                pass
            assert isinstance(lc.phase, FilePhase)

    @given(
        events=st.lists(
            st.sampled_from(ALL_EVENTS),
            min_size=1,
            max_size=30,
        )
    )
    @settings(max_examples=200)
    def test_event_log_monotonic(self, events: list[EventType]) -> None:
        """Event log timestamps are non-decreasing."""
        lc = FileLifecycle("test.py")
        for ev in events:
            try:
                lc.process_event(ev)
            except ValueError:
                pass

        timestamps = [e.timestamp for e in lc.event_log]
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1], "Event log timestamps must be monotonic"

    @given(
        events=st.lists(
            st.sampled_from(ALL_EVENTS),
            min_size=1,
            max_size=30,
        )
    )
    @settings(max_examples=200)
    def test_fix_counts_bounded(self, events: list[EventType]) -> None:
        """Fix counts never exceed max + 1 (they increment then trigger limit)."""
        max_rf = 2
        max_tf = 2
        max_bf = 2
        lc = FileLifecycle(
            "test.py",
            max_review_fixes=max_rf,
            max_test_fixes=max_tf,
            max_build_fixes=max_bf,
        )
        for ev in events:
            try:
                lc.process_event(ev)
            except ValueError:
                pass

        assert lc.review_fix_count <= max_rf + 1
        assert lc.test_fix_count <= max_tf + 1
        assert lc.build_fix_count <= max_bf + 1


class TestStateMachineHappyPath:
    """Deterministic tests verifying canonical lifecycle paths."""

    def test_full_happy_path(self) -> None:
        """PENDING → GENERATING → REVIEWING → BUILDING → TESTING → PASSED."""
        lc = FileLifecycle("app.py")
        assert lc.phase == FilePhase.PENDING

        lc.process_event(EventType.DEPS_MET)
        assert lc.phase == FilePhase.GENERATING

        lc.process_event(EventType.CODE_GENERATED)
        assert lc.phase == FilePhase.REVIEWING

        lc.process_event(EventType.REVIEW_PASSED)
        assert lc.phase == FilePhase.BUILDING

        lc.process_event(EventType.BUILD_PASSED)
        assert lc.phase == FilePhase.TESTING

        lc.process_event(EventType.TEST_PASSED)
        assert lc.phase == FilePhase.PASSED
        assert lc.is_terminal

    def test_review_fix_cycle(self) -> None:
        """Review failure triggers a fix cycle back through review."""
        lc = FileLifecycle("app.py")
        lc.process_event(EventType.DEPS_MET)
        lc.process_event(EventType.CODE_GENERATED)
        assert lc.phase == FilePhase.REVIEWING

        lc.process_event(EventType.REVIEW_FAILED, data={"findings": ["missing docstring"]})
        assert lc.phase == FilePhase.FIXING
        assert lc.fix_trigger == "review"

        lc.process_event(EventType.FIX_APPLIED)
        assert lc.phase == FilePhase.REVIEWING

    def test_retries_exhausted_always_fails(self) -> None:
        """RETRIES_EXHAUSTED from any non-terminal phase → FAILED."""
        for phase in FilePhase:
            if phase in TERMINAL_PHASES:
                continue
            lc = FileLifecycle("x.py")
            lc.phase = phase  # force phase for testing
            lc.process_event(EventType.RETRIES_EXHAUSTED)
            assert lc.phase == FilePhase.FAILED
