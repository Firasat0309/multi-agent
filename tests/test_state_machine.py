"""Tests for the event-sourced state machine."""

import pytest
from core.state_machine import (
    Event,
    EventType,
    FileLifecycle,
    FilePhase,
    LifecycleEngine,
)


# ── FileLifecycle: happy path ───────────────────────────────────────────

class TestFileLifecycleHappyPath:
    """File flows through Generate → Review(pass) → Test(pass) → PASSED."""

    def test_initial_state_is_pending(self):
        lc = FileLifecycle("app.java")
        assert lc.phase == FilePhase.PENDING
        assert not lc.is_terminal

    def test_full_happy_path(self):
        lc = FileLifecycle("app.java")
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

    def test_event_log_captures_all_transitions(self):
        lc = FileLifecycle("app.java")
        lc.process_event(EventType.DEPS_MET)
        lc.process_event(EventType.CODE_GENERATED)
        lc.process_event(EventType.REVIEW_PASSED)
        lc.process_event(EventType.BUILD_PASSED)
        lc.process_event(EventType.TEST_PASSED)

        assert len(lc.event_log) == 5
        assert lc.event_log[0].event_type == EventType.DEPS_MET
        assert lc.event_log[0].phase_before == FilePhase.PENDING
        assert lc.event_log[0].phase_after == FilePhase.GENERATING
        assert lc.event_log[-1].phase_after == FilePhase.PASSED

    def test_zero_fix_counts_on_happy_path(self):
        lc = FileLifecycle("app.java")
        for evt in [EventType.DEPS_MET, EventType.CODE_GENERATED,
                     EventType.REVIEW_PASSED, EventType.BUILD_PASSED,
                     EventType.TEST_PASSED]:
            lc.process_event(evt)
        assert lc.review_fix_count == 0
        assert lc.test_fix_count == 0
        assert lc.total_fix_count == 0


# ── FileLifecycle: review fix cycle ─────────────────────────────────────

class TestReviewFixCycle:
    """REVIEWING → FIXING → REVIEWING cycle."""

    def test_review_fail_routes_to_fixing(self):
        lc = FileLifecycle("svc.java")
        lc.process_event(EventType.DEPS_MET)
        lc.process_event(EventType.CODE_GENERATED)
        lc.process_event(EventType.REVIEW_FAILED, {"findings": ["null check missing"]})

        assert lc.phase == FilePhase.FIXING
        assert lc.fix_trigger == "review"
        assert lc.review_fix_count == 1
        assert lc.review_findings == ["null check missing"]

    def test_fix_applied_routes_back_to_reviewing(self):
        lc = FileLifecycle("svc.java")
        lc.process_event(EventType.DEPS_MET)
        lc.process_event(EventType.CODE_GENERATED)
        lc.process_event(EventType.REVIEW_FAILED)
        lc.process_event(EventType.FIX_APPLIED)

        assert lc.phase == FilePhase.REVIEWING

    def test_review_pass_after_fix_goes_to_building(self):
        lc = FileLifecycle("svc.java")
        lc.process_event(EventType.DEPS_MET)
        lc.process_event(EventType.CODE_GENERATED)
        lc.process_event(EventType.REVIEW_FAILED)
        lc.process_event(EventType.FIX_APPLIED)
        lc.process_event(EventType.REVIEW_PASSED)

        assert lc.phase == FilePhase.BUILDING

    def test_max_review_fixes_proceeds_to_building(self):
        lc = FileLifecycle("svc.java", max_review_fixes=2)
        lc.process_event(EventType.DEPS_MET)
        lc.process_event(EventType.CODE_GENERATED)

        # Fix cycle 1
        lc.process_event(EventType.REVIEW_FAILED)
        lc.process_event(EventType.FIX_APPLIED)
        # Fix cycle 2
        lc.process_event(EventType.REVIEW_FAILED)
        lc.process_event(EventType.FIX_APPLIED)
        # 3rd failure exceeds limit → proceed to build verification
        lc.process_event(EventType.REVIEW_FAILED)

        assert lc.phase == FilePhase.BUILDING
        assert lc.review_fix_count == 3


# ── FileLifecycle: test fix cycle ───────────────────────────────────────

class TestTestFixCycle:
    """TESTING → FIXING → TESTING cycle with alternating fix targets."""

    def test_test_fail_routes_to_fixing(self):
        lc = FileLifecycle("svc.java")
        lc.process_event(EventType.DEPS_MET)
        lc.process_event(EventType.CODE_GENERATED)
        lc.process_event(EventType.REVIEW_PASSED)
        lc.process_event(EventType.BUILD_PASSED)
        lc.process_event(EventType.TEST_FAILED, {"errors": "NullPointerException"})

        assert lc.phase == FilePhase.FIXING
        assert lc.fix_trigger == "test"
        assert lc.test_fix_count == 1
        assert lc.test_errors == "NullPointerException"

    def test_fix_applied_routes_back_to_testing(self):
        lc = FileLifecycle("svc.java")
        for evt in [EventType.DEPS_MET, EventType.CODE_GENERATED,
                     EventType.REVIEW_PASSED, EventType.BUILD_PASSED]:
            lc.process_event(evt)
        lc.process_event(EventType.TEST_FAILED)
        lc.process_event(EventType.FIX_APPLIED)

        assert lc.phase == FilePhase.TESTING

    def test_alternating_fix_targets(self):
        lc = FileLifecycle("svc.java", max_test_fixes=6)
        for evt in [EventType.DEPS_MET, EventType.CODE_GENERATED,
                     EventType.REVIEW_PASSED, EventType.BUILD_PASSED]:
            lc.process_event(evt)

        assert lc.test_fix_target == "test"

        # Cycle 1: fix test
        lc.process_event(EventType.TEST_FAILED)
        assert lc.test_fix_target == "test"
        lc.process_event(EventType.FIX_APPLIED)
        # After fix_applied from test, target alternates
        assert lc.test_fix_target == "source"

        # Cycle 2: fix source
        lc.process_event(EventType.TEST_FAILED)
        lc.process_event(EventType.FIX_APPLIED)
        assert lc.test_fix_target == "test"

    def test_max_test_fixes_marks_failed(self):
        lc = FileLifecycle("svc.java", max_test_fixes=2)
        for evt in [EventType.DEPS_MET, EventType.CODE_GENERATED,
                     EventType.REVIEW_PASSED, EventType.BUILD_PASSED]:
            lc.process_event(evt)

        # Fix cycles 1, 2
        lc.process_event(EventType.TEST_FAILED)
        lc.process_event(EventType.FIX_APPLIED)
        lc.process_event(EventType.TEST_FAILED)
        lc.process_event(EventType.FIX_APPLIED)

        # 3rd failure exceeds limit → mark as failed
        lc.process_event(EventType.TEST_FAILED)

        assert lc.phase == FilePhase.FAILED
        assert lc.test_fix_count == 3

    def test_full_test_fix_then_pass(self):
        lc = FileLifecycle("svc.java")
        for evt in [EventType.DEPS_MET, EventType.CODE_GENERATED,
                     EventType.REVIEW_PASSED, EventType.BUILD_PASSED]:
            lc.process_event(evt)

        # First test fails, fix, re-test passes
        lc.process_event(EventType.TEST_FAILED)
        lc.process_event(EventType.FIX_APPLIED)
        lc.process_event(EventType.TEST_PASSED)

        assert lc.phase == FilePhase.PASSED
        assert lc.test_fix_count == 1


# ── FileLifecycle: build fix cycle ──────────────────────────────────────

class TestBuildFixCycle:
    """BUILDING → FIXING → BUILDING cycle."""

    def _reach_building(self, lc: FileLifecycle) -> None:
        """Advance lifecycle to the BUILDING phase via the happy path."""
        lc.process_event(EventType.DEPS_MET)
        lc.process_event(EventType.CODE_GENERATED)
        lc.process_event(EventType.REVIEW_PASSED)

    def test_build_fail_routes_to_fixing(self):
        lc = FileLifecycle("svc.java")
        self._reach_building(lc)
        lc.process_event(EventType.BUILD_FAILED, {"errors": "CompilationError: ';' expected"})

        assert lc.phase == FilePhase.FIXING
        assert lc.fix_trigger == "build"
        assert lc.build_fix_count == 1
        assert lc.build_errors == "CompilationError: ';' expected"

    def test_fix_applied_routes_back_to_building(self):
        lc = FileLifecycle("svc.java")
        self._reach_building(lc)
        lc.process_event(EventType.BUILD_FAILED)
        lc.process_event(EventType.FIX_APPLIED)

        assert lc.phase == FilePhase.BUILDING

    def test_build_pass_after_fix_goes_to_testing(self):
        lc = FileLifecycle("svc.java")
        self._reach_building(lc)
        lc.process_event(EventType.BUILD_FAILED)
        lc.process_event(EventType.FIX_APPLIED)
        lc.process_event(EventType.BUILD_PASSED)

        assert lc.phase == FilePhase.TESTING

    def test_max_build_fixes_marks_failed(self):
        lc = FileLifecycle("svc.java", max_build_fixes=2)
        self._reach_building(lc)

        # Fix cycles 1, 2
        lc.process_event(EventType.BUILD_FAILED)
        lc.process_event(EventType.FIX_APPLIED)
        lc.process_event(EventType.BUILD_FAILED)
        lc.process_event(EventType.FIX_APPLIED)

        # 3rd failure exceeds limit → mark as failed
        lc.process_event(EventType.BUILD_FAILED)

        assert lc.phase == FilePhase.FAILED
        assert lc.build_fix_count == 3

    def test_full_build_fix_then_pass(self):
        lc = FileLifecycle("svc.java")
        self._reach_building(lc)

        # Build fails, fix, re-build passes, test passes
        lc.process_event(EventType.BUILD_FAILED)
        lc.process_event(EventType.FIX_APPLIED)
        lc.process_event(EventType.BUILD_PASSED)
        lc.process_event(EventType.TEST_PASSED)

        assert lc.phase == FilePhase.PASSED
        assert lc.build_fix_count == 1


# ── FileLifecycle: combined review + test fixes ─────────────────────────

class TestCombinedFixCycles:
    def test_review_fix_then_test_fix_then_pass(self):
        lc = FileLifecycle("svc.java")
        lc.process_event(EventType.DEPS_MET)
        lc.process_event(EventType.CODE_GENERATED)

        # Review fails, fix, review passes
        lc.process_event(EventType.REVIEW_FAILED)
        lc.process_event(EventType.FIX_APPLIED)
        lc.process_event(EventType.REVIEW_PASSED)

        # Build passes
        lc.process_event(EventType.BUILD_PASSED)

        # Test fails, fix, test passes
        lc.process_event(EventType.TEST_FAILED)
        lc.process_event(EventType.FIX_APPLIED)
        lc.process_event(EventType.TEST_PASSED)

        assert lc.phase == FilePhase.PASSED
        assert lc.review_fix_count == 1
        assert lc.test_fix_count == 1
        assert lc.total_fix_count == 2
        assert len(lc.event_log) == 9


# ── FileLifecycle: error handling ───────────────────────────────────────

class TestLifecycleErrors:
    def test_invalid_transition_raises(self):
        lc = FileLifecycle("app.java")
        with pytest.raises(ValueError, match="No transition"):
            lc.process_event(EventType.CODE_GENERATED)  # can't generate from PENDING

    def test_retries_exhausted_from_any_phase(self):
        lc = FileLifecycle("app.java")
        lc.process_event(EventType.DEPS_MET)
        lc.process_event(EventType.RETRIES_EXHAUSTED)
        assert lc.phase == FilePhase.FAILED
        assert lc.is_terminal

    def test_fix_applied_without_trigger_raises(self):
        lc = FileLifecycle("app.java")
        lc.process_event(EventType.DEPS_MET)
        lc.process_event(EventType.CODE_GENERATED)
        lc.process_event(EventType.REVIEW_FAILED)
        lc.fix_trigger = ""  # corrupt state
        with pytest.raises(ValueError, match="unknown fix_trigger"):
            lc.process_event(EventType.FIX_APPLIED)


# ── Event immutability ──────────────────────────────────────────────────

class TestEventImmutability:
    def test_event_is_frozen(self):
        lc = FileLifecycle("app.java")
        lc.process_event(EventType.DEPS_MET)
        event = lc.event_log[0]
        with pytest.raises(AttributeError):
            event.event_type = EventType.CODE_GENERATED  # type: ignore[misc]

    def test_event_repr(self):
        lc = FileLifecycle("app.java")
        lc.process_event(EventType.DEPS_MET)
        r = repr(lc.event_log[0])
        assert "deps_met" in r
        assert "app.java" in r


# ── LifecycleEngine ─────────────────────────────────────────────────────

class TestLifecycleEngine:
    def _make_engine(self, **kwargs) -> LifecycleEngine:
        return LifecycleEngine(
            file_paths=["model.java", "repo.java", "service.java"],
            file_deps={
                "model.java": [],
                "repo.java": ["model.java"],
                "service.java": ["repo.java", "model.java"],
            },
            **kwargs,
        )

    def test_initial_state(self):
        engine = self._make_engine()
        assert not engine.all_terminal()
        stats = engine.get_stats()
        assert stats["pending"] == 3

    def test_actionable_respects_deps(self):
        engine = self._make_engine()
        actionable = engine.get_actionable_files()
        paths = [p for p, _ in actionable]

        # model.java has no deps → actionable
        assert "model.java" in paths
        # service.java depends on repo.java + model.java → NOT actionable
        assert "service.java" not in paths

    def test_deps_unblock_after_generating(self):
        engine = self._make_engine()

        # Generate model.java
        engine.process_event("model.java", EventType.DEPS_MET)
        engine.process_event("model.java", EventType.CODE_GENERATED)

        actionable = engine.get_actionable_files()
        paths = [p for p, _ in actionable]

        # model.java is now in REVIEWING → actionable
        assert "model.java" in paths
        # repo.java deps met (model is past GENERATING) → actionable
        assert "repo.java" in paths
        # service.java still blocked (repo.java is PENDING)
        assert "service.java" not in paths

    def test_full_engine_execution(self):
        engine = self._make_engine()

        for f in ["model.java", "repo.java", "service.java"]:
            if engine.has_file(f):
                lc = engine.get_lifecycle(f)
                if lc.phase == FilePhase.PENDING:
                    # Check if actionable
                    actionable_paths = [p for p, _ in engine.get_actionable_files()]
                    if f in actionable_paths:
                        engine.process_event(f, EventType.DEPS_MET)

        # Process all files through happy path
        for f in ["model.java", "repo.java", "service.java"]:
            lc = engine.get_lifecycle(f)
            if lc.phase == FilePhase.PENDING:
                engine.process_event(f, EventType.DEPS_MET)
            engine.process_event(f, EventType.CODE_GENERATED)
            engine.process_event(f, EventType.REVIEW_PASSED)
            engine.process_event(f, EventType.TEST_PASSED)

        assert engine.all_terminal()
        stats = engine.get_stats()
        assert stats["passed"] == 3

    def test_skip_testing_auto_passes(self):
        engine = LifecycleEngine(
            file_paths=["pom.xml", "app.java"],
            file_deps={"pom.xml": [], "app.java": []},
        )
        engine.skip_testing("pom.xml")

        engine.process_event("pom.xml", EventType.DEPS_MET)
        engine.process_event("pom.xml", EventType.CODE_GENERATED)
        engine.process_event("pom.xml", EventType.REVIEW_PASSED)

        # pom.xml should auto-pass (skipped testing)
        lc = engine.get_lifecycle("pom.xml")
        assert lc.phase == FilePhase.PASSED

    def test_terminal_files_not_actionable(self):
        engine = LifecycleEngine(
            file_paths=["a.java"],
            file_deps={"a.java": []},
        )
        engine.process_event("a.java", EventType.DEPS_MET)
        engine.process_event("a.java", EventType.CODE_GENERATED)
        engine.process_event("a.java", EventType.REVIEW_PASSED)
        engine.process_event("a.java", EventType.TEST_PASSED)

        assert engine.get_actionable_files() == []

    def test_results_summary(self):
        engine = LifecycleEngine(
            file_paths=["a.java", "b.java"],
            file_deps={"a.java": [], "b.java": []},
        )
        # a passes
        for evt in [EventType.DEPS_MET, EventType.CODE_GENERATED,
                     EventType.REVIEW_PASSED, EventType.TEST_PASSED]:
            engine.process_event("a.java", evt)
        # b fails
        engine.process_event("b.java", EventType.DEPS_MET)
        engine.process_event("b.java", EventType.RETRIES_EXHAUSTED)

        summary = engine.get_results_summary()
        assert summary["passed"] == 1
        assert summary["failed"] == 1
        assert summary["total_files"] == 2

    def test_full_event_log_is_chronological(self):
        engine = LifecycleEngine(
            file_paths=["a.java", "b.java"],
            file_deps={"a.java": [], "b.java": []},
        )
        engine.process_event("a.java", EventType.DEPS_MET)
        engine.process_event("b.java", EventType.DEPS_MET)
        engine.process_event("a.java", EventType.CODE_GENERATED)

        log = engine.get_full_event_log()
        assert len(log) == 3
        # Timestamps should be non-decreasing
        for i in range(len(log) - 1):
            assert log[i].timestamp <= log[i + 1].timestamp

    def test_external_deps_assumed_available(self):
        """Dependencies not in the engine (external libs) don't block."""
        engine = LifecycleEngine(
            file_paths=["app.java"],
            file_deps={"app.java": ["spring-framework"]},  # external
        )
        actionable = engine.get_actionable_files()
        assert len(actionable) == 1
        assert actionable[0][0] == "app.java"


# ── LifecycleEngine: checkpoint mode ─────────────────────────────────

class TestCheckpointMode:
    """Tests for the checkpoint_mode flag and get_files_in_phase()."""

    def test_checkpoint_mode_defaults_to_false(self):
        engine = LifecycleEngine(
            file_paths=["a.java"],
            file_deps={"a.java": []},
        )
        assert not engine.checkpoint_mode

    def test_checkpoint_mode_can_be_set(self):
        engine = LifecycleEngine(
            file_paths=["a.java"],
            file_deps={"a.java": []},
        )
        engine.checkpoint_mode = True
        assert engine.checkpoint_mode

    def test_checkpoint_mode_via_constructor(self):
        engine = LifecycleEngine(
            file_paths=["a.java"],
            file_deps={"a.java": []},
            checkpoint_mode=True,
        )
        assert engine.checkpoint_mode

    def test_get_files_in_phase(self):
        engine = LifecycleEngine(
            file_paths=["a.java", "b.java", "c.java"],
            file_deps={"a.java": [], "b.java": [], "c.java": []},
        )
        # All start in PENDING
        assert set(engine.get_files_in_phase(FilePhase.PENDING)) == {"a.java", "b.java", "c.java"}
        assert engine.get_files_in_phase(FilePhase.GENERATING) == []

        # Move a.java to GENERATING
        engine.process_event("a.java", EventType.DEPS_MET)
        assert "a.java" in engine.get_files_in_phase(FilePhase.GENERATING)
        assert "a.java" not in engine.get_files_in_phase(FilePhase.PENDING)

    def test_compiled_checkpoint_mode_keeps_building(self):
        """In checkpoint mode for compiled langs, BUILDING files stay in BUILDING."""
        engine = LifecycleEngine(
            file_paths=["a.java"],
            file_deps={"a.java": []},
            compiled=True,
            checkpoint_mode=True,
        )
        engine.process_event("a.java", EventType.DEPS_MET)
        engine.process_event("a.java", EventType.CODE_GENERATED)
        engine.process_event("a.java", EventType.REVIEW_PASSED)

        # In checkpoint mode, the file reaches BUILDING and stays there
        # (the checkpoint will explicitly call BUILD_PASSED later)
        lc = engine.get_lifecycle("a.java")
        assert lc.phase == FilePhase.BUILDING

    def test_non_compiled_still_skips_building(self):
        """For interpreted languages, BUILDING is still auto-skipped."""
        engine = LifecycleEngine(
            file_paths=["a.py"],
            file_deps={"a.py": []},
            compiled=False,
            checkpoint_mode=True,
        )
        engine.process_event("a.py", EventType.DEPS_MET)
        engine.process_event("a.py", EventType.CODE_GENERATED)
        engine.process_event("a.py", EventType.REVIEW_PASSED)

        lc = engine.get_lifecycle("a.py")
        # Should skip BUILDING and go to TESTING
        assert lc.phase == FilePhase.TESTING


# ── FileLifecycle: modify mode & change metadata ─────────────────────

class TestFileLifecycleModifyMode:
    """Tests for generation_task_type and change_metadata on FileLifecycle."""

    def test_default_generation_task_type(self):
        lc = FileLifecycle("app.java")
        assert lc.generation_task_type == "generate_file"

    def test_modify_file_task_type(self):
        lc = FileLifecycle("app.java", generation_task_type="modify_file")
        assert lc.generation_task_type == "modify_file"

    def test_change_metadata_defaults_to_empty(self):
        lc = FileLifecycle("app.java")
        assert lc.change_metadata == {}

    def test_change_metadata_stored(self):
        meta = {"change_type": "modify_function", "target_function": "handle_request"}
        lc = FileLifecycle("app.java", change_metadata=meta)
        assert lc.change_metadata == meta
        assert lc.change_metadata["target_function"] == "handle_request"

    def test_modify_lifecycle_full_happy_path(self):
        """A modify_file lifecycle follows the same phase transitions."""
        lc = FileLifecycle("app.java", generation_task_type="modify_file",
                           change_metadata={"change_type": "add_method"})
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

    def test_change_metadata_none_becomes_empty_dict(self):
        lc = FileLifecycle("app.java", change_metadata=None)
        assert lc.change_metadata == {}


# ── LifecycleEngine: file_overrides ──────────────────────────────────

class TestLifecycleEngineFileOverrides:
    """Tests for LifecycleEngine file_overrides parameter."""

    def test_overrides_set_generation_task_type(self):
        engine = LifecycleEngine(
            file_paths=["existing.java", "new.java"],
            file_deps={"existing.java": [], "new.java": []},
            file_overrides={
                "existing.java": {"generation_task_type": "modify_file"},
            },
        )
        assert engine.get_lifecycle("existing.java").generation_task_type == "modify_file"
        assert engine.get_lifecycle("new.java").generation_task_type == "generate_file"

    def test_overrides_set_change_metadata(self):
        meta = {"change_type": "modify_function", "target_class": "UserService"}
        engine = LifecycleEngine(
            file_paths=["svc.java"],
            file_deps={"svc.java": []},
            file_overrides={
                "svc.java": {"change_metadata": meta},
            },
        )
        lc = engine.get_lifecycle("svc.java")
        assert lc.change_metadata == meta

    def test_overrides_combined(self):
        engine = LifecycleEngine(
            file_paths=["a.java", "b.java"],
            file_deps={"a.java": [], "b.java": ["a.java"]},
            file_overrides={
                "a.java": {
                    "generation_task_type": "modify_file",
                    "change_metadata": {"change_type": "add_method"},
                },
                "b.java": {
                    "generation_task_type": "modify_file",
                    "change_metadata": {"change_type": "rename_function"},
                },
            },
        )
        lc_a = engine.get_lifecycle("a.java")
        lc_b = engine.get_lifecycle("b.java")
        assert lc_a.generation_task_type == "modify_file"
        assert lc_b.generation_task_type == "modify_file"
        assert lc_a.change_metadata["change_type"] == "add_method"
        assert lc_b.change_metadata["change_type"] == "rename_function"

    def test_no_overrides_uses_defaults(self):
        engine = LifecycleEngine(
            file_paths=["app.java"],
            file_deps={"app.java": []},
            file_overrides=None,
        )
        lc = engine.get_lifecycle("app.java")
        assert lc.generation_task_type == "generate_file"
        assert lc.change_metadata == {}
