"""Event-sourced state machine for per-file lifecycle management.

Replaces the static, pre-baked FIX_CODE tasks in the DAG with a dynamic
state machine that routes files through Generate → Review → Fix → Test
cycles driven by actual outcomes (review findings, test results).

Architecture:
  - Each source file gets a ``FileLifecycle`` instance (its own state machine)
  - ``LifecycleEngine`` orchestrates all file lifecycles + inter-file deps
  - Transitions are driven by ``EventType`` values emitted by agents
  - An append-only event log provides full auditability and replay
  - ``max_visits`` guards prevent infinite loops

The networkx DAG is still used for:
  - Inter-file dependency ordering (file A depends on file B)
  - Global-phase ordering (security scan, module review, deploy, docs)
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ── Enums ───────────────────────────────────────────────────────────────

class FilePhase(Enum):
    """Lifecycle phase of a single source file."""

    PENDING = "pending"           # waiting for dependency files to be generated
    GENERATING = "generating"     # CoderAgent producing the file
    REVIEWING = "reviewing"       # ReviewerAgent checking the file
    FIXING = "fixing"             # CoderAgent fixing based on review or test feedback
    BUILDING = "building"         # compiler/type-checker verification
    TESTING = "testing"           # TestAgent generating/running tests
    PASSED = "passed"             # terminal: file completed successfully
    FAILED = "failed"             # terminal: exhausted retries
    DEGRADED = "degraded"         # terminal: completed with quality warnings (fix limits hit)


class EventType(Enum):
    """Events that drive lifecycle transitions."""

    DEPS_MET = "deps_met"
    CODE_GENERATED = "code_generated"
    REVIEW_PASSED = "review_passed"
    REVIEW_FAILED = "review_failed"
    FIX_APPLIED = "fix_applied"
    BUILD_PASSED = "build_passed"
    BUILD_FAILED = "build_failed"
    TEST_PASSED = "test_passed"
    TEST_FAILED = "test_failed"
    RETRIES_EXHAUSTED = "retries_exhausted"


# ── Event (immutable audit record) ──────────────────────────────────────

@dataclass(frozen=True)
class Event:
    """An immutable record of something that happened during file processing."""

    event_type: EventType
    file_path: str
    timestamp: float
    phase_before: FilePhase
    phase_after: FilePhase
    data: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"Event({self.event_type.value}, {self.file_path}, "
            f"{self.phase_before.value}→{self.phase_after.value})"
        )


# ── Transition table ────────────────────────────────────────────────────
# Standard transitions: (from_phase, event_type) → to_phase
# FIX_APPLIED is special — destination depends on fix_trigger (handled in code)

_TRANSITIONS: dict[tuple[FilePhase, EventType], FilePhase] = {
    (FilePhase.PENDING, EventType.DEPS_MET): FilePhase.GENERATING,
    (FilePhase.GENERATING, EventType.CODE_GENERATED): FilePhase.REVIEWING,
    (FilePhase.REVIEWING, EventType.REVIEW_PASSED): FilePhase.BUILDING,
    (FilePhase.REVIEWING, EventType.REVIEW_FAILED): FilePhase.FIXING,
    (FilePhase.BUILDING, EventType.BUILD_PASSED): FilePhase.TESTING,
    (FilePhase.BUILDING, EventType.BUILD_FAILED): FilePhase.FIXING,
    (FilePhase.TESTING, EventType.TEST_PASSED): FilePhase.PASSED,
    (FilePhase.TESTING, EventType.TEST_FAILED): FilePhase.FIXING,
}


# ── Per-file state machine ──────────────────────────────────────────────

class FileLifecycle:
    """State machine tracking one file through Generate → Review → Fix → Test.

    The machine supports two distinct fix cycles:
      1. Review fix cycle:  REVIEWING → FIXING → REVIEWING → ...
      2. Test fix cycle:    TESTING  → FIXING → TESTING  → ...

    Each cycle has its own counter and max-visit guard.  For test fixes,
    the target alternates between "test" (fix the test file) and "source"
    (fix the source file under test) — mirroring the proven strategy from
    the old ``_run_and_fix`` loop.
    """

    def __init__(
        self,
        file_path: str,
        *,
        max_review_fixes: int = 2,
        max_test_fixes: int = 3,
        max_build_fixes: int = 3,
        generation_task_type: str = "generate_file",
        change_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.file_path = file_path
        self.phase = FilePhase.PENDING
        self.max_review_fixes = max_review_fixes
        self.max_test_fixes = max_test_fixes
        self.max_build_fixes = max_build_fixes

        # Task type used in the GENERATING phase: "generate_file" for new
        # files (CoderAgent) or "modify_file" for existing files (PatchAgent).
        self.generation_task_type = generation_task_type

        # Per-file metadata carried through the lifecycle.  Populated by
        # EnhanceLifecyclePlanBuilder with change_type, change_description,
        # target_function, target_class so PatchAgent receives full context.
        self.change_metadata: dict[str, Any] = change_metadata or {}

        # Fix-cycle tracking
        self.review_fix_count = 0
        self.test_fix_count = 0
        self.fix_trigger: str = ""          # "review" | "test"
        self._test_fix_target: str = "test"  # alternates: "test" ↔ "source"

        # Context carried between phases for downstream agents
        self.review_findings: list[str] = []
        self.review_output: str = ""
        self.test_errors: str = ""
        self.build_fix_count: int = 0
        self.build_errors: str = ""
        self.tests_generated: bool = False

        # Append-only event log
        self.event_log: list[Event] = []

    # ── Properties ──────────────────────────────────────────────────

    @property
    def is_terminal(self) -> bool:
        return self.phase in (FilePhase.PASSED, FilePhase.FAILED, FilePhase.DEGRADED)

    @property
    def test_fix_target(self) -> str:
        """Whether the next test-fix should target 'test' or 'source'."""
        return self._test_fix_target

    @property
    def total_fix_count(self) -> int:
        return self.review_fix_count + self.test_fix_count

    # ── Core transition logic ───────────────────────────────────────

    def process_event(
        self,
        event_type: EventType,
        data: dict[str, Any] | None = None,
    ) -> FilePhase:
        """Process an event and transition to the next phase.

        Returns the new phase.
        Raises ``ValueError`` for invalid transitions.
        """
        data = data or {}
        old_phase = self.phase

        # RETRIES_EXHAUSTED — universal terminal transition
        if event_type == EventType.RETRIES_EXHAUSTED:
            self.phase = FilePhase.FAILED
            self._record(event_type, old_phase, self.phase, data)
            return self.phase

        # FIX_APPLIED — destination depends on what triggered the fix
        if event_type == EventType.FIX_APPLIED and self.phase == FilePhase.FIXING:
            if self.fix_trigger == "review":
                self.phase = FilePhase.REVIEWING
            elif self.fix_trigger == "test":
                self.phase = FilePhase.TESTING
                # Alternate fix target for next test-fix cycle
                self._test_fix_target = (
                    "source" if self._test_fix_target == "test" else "test"
                )
            elif self.fix_trigger == "build":
                self.phase = FilePhase.BUILDING
            else:
                raise ValueError(
                    f"FIX_APPLIED with unknown fix_trigger={self.fix_trigger!r} "
                    f"for {self.file_path}"
                )
            self._record(event_type, old_phase, self.phase, data)
            return self.phase

        # Standard transition lookup
        key = (self.phase, event_type)
        new_phase = _TRANSITIONS.get(key)
        if new_phase is None:
            raise ValueError(
                f"No transition: {self.phase.value} --[{event_type.value}]--> ??? "
                f"for {self.file_path}"
            )

        # ── Cycle-limit guards ──────────────────────────────────────
        if event_type == EventType.REVIEW_FAILED:
            self.review_fix_count += 1
            self.fix_trigger = "review"
            self.review_findings = data.get("findings", [])
            self.review_output = data.get("output", "")

            if self.review_fix_count > self.max_review_fixes:
                # Review fixes exhausted — still send the file through BUILD
                # so the compiler can verify it.  Never skip BUILD.
                new_phase = FilePhase.BUILDING
                self.fix_trigger = ""
                logger.warning(
                    "%s: review fix limit (%d) reached — proceeding to build verification",
                    self.file_path, self.max_review_fixes,
                )

        elif event_type == EventType.TEST_FAILED:
            self.test_fix_count += 1
            self.fix_trigger = "test"
            self.test_errors = data.get("errors", "")

            if self.test_fix_count > self.max_test_fixes:
                # Tests never fully passed — mark DEGRADED so the pipeline
                # reports a quality warning rather than a false clean success.
                new_phase = FilePhase.DEGRADED
                self.fix_trigger = ""
                logger.warning(
                    "%s: test fix limit (%d) reached — marking DEGRADED (tests not passing)",
                    self.file_path, self.max_test_fixes,
                )

        elif event_type == EventType.BUILD_FAILED:
            self.build_fix_count += 1
            self.fix_trigger = "build"
            self.build_errors = data.get("errors", "")

            if self.build_fix_count > self.max_build_fixes:
                # Build never fully resolved — mark DEGRADED so the pipeline
                # reports a quality warning rather than a false clean success.
                new_phase = FilePhase.DEGRADED
                self.fix_trigger = ""
                logger.warning(
                    "%s: build fix limit (%d) reached — marking DEGRADED (build not clean)",
                    self.file_path, self.max_build_fixes,
                )

        elif event_type == EventType.REVIEW_PASSED:
            self.review_findings = []
            self.review_output = data.get("output", "")

        elif event_type == EventType.TEST_PASSED:
            self.test_errors = ""

        self.phase = new_phase
        self._record(event_type, old_phase, self.phase, data)
        return self.phase

    def _record(
        self,
        event_type: EventType,
        old_phase: FilePhase,
        new_phase: FilePhase,
        data: dict[str, Any],
    ) -> None:
        event = Event(
            event_type=event_type,
            file_path=self.file_path,
            timestamp=time.monotonic(),
            phase_before=old_phase,
            phase_after=new_phase,
            data=data,
        )
        self.event_log.append(event)
        logger.info(
            "[%s] %s --[%s]--> %s",
            self.file_path, old_phase.value, event_type.value, new_phase.value,
        )


# ── Lifecycle engine (orchestrates all files) ───────────────────────────

class LifecycleEngine:
    """Orchestrates per-file state machines with inter-file dependency awareness.

    Responsibilities:
      - Manages a ``FileLifecycle`` per source file
      - Checks inter-file dependencies before allowing PENDING → GENERATING
      - Provides ``get_actionable_files()`` for the executor to dispatch work
      - Aggregates event logs across all files for observability
      - Tracks which files skip testing (config, deploy layers)
      - Supports checkpoint mode where BUILDING is handled at repo level
    """

    def __init__(
        self,
        file_paths: list[str],
        file_deps: dict[str, list[str]],
        *,
        max_review_fixes: int = 2,
        max_test_fixes: int = 3,
        max_build_fixes: int = 3,
        compiled: bool = False,
        checkpoint_mode: bool = False,
        file_overrides: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self._lifecycles: dict[str, FileLifecycle] = {}
        self._deps = file_deps
        self._skip_testing: set[str] = set()
        self._compiled = compiled
        self._max_build_fixes = max_build_fixes
        # When True, BUILDING is handled by repo-level checkpoints in
        # PipelineExecutor, not per-file.  The REVIEW_PASSED transition
        # still goes to BUILDING, but it's immediately auto-passed unless
        # the checkpoint explicitly manages it.
        self._checkpoint_mode = checkpoint_mode

        # file_overrides allows per-file configuration of generation_task_type
        # and change_metadata (used by the Enhance pipeline).
        overrides = file_overrides or {}

        for path in file_paths:
            fo = overrides.get(path, {})
            self._lifecycles[path] = FileLifecycle(
                path,
                max_review_fixes=max_review_fixes,
                max_test_fixes=max_test_fixes,
                max_build_fixes=max_build_fixes,
                generation_task_type=fo.get("generation_task_type", "generate_file"),
                change_metadata=fo.get("change_metadata"),
            )

    # ── File configuration ──────────────────────────────────────────

    def skip_testing(self, file_path: str) -> None:
        """Mark a file as not needing tests (config, deploy, test layers)."""
        self._skip_testing.add(file_path)

    @property
    def checkpoint_mode(self) -> bool:
        """Whether build verification is handled at repo level."""
        return self._checkpoint_mode

    @checkpoint_mode.setter
    def checkpoint_mode(self, value: bool) -> None:
        self._checkpoint_mode = value

    # ── Lifecycle access ────────────────────────────────────────────

    def get_lifecycle(self, file_path: str) -> FileLifecycle:
        return self._lifecycles[file_path]

    def has_file(self, file_path: str) -> bool:
        return file_path in self._lifecycles

    # ── Event processing ────────────────────────────────────────────

    def process_event(
        self,
        file_path: str,
        event_type: EventType,
        data: dict[str, Any] | None = None,
    ) -> FilePhase:
        """Process an event for a file and handle auto-skip logic."""
        lc = self._lifecycles[file_path]
        new_phase = lc.process_event(event_type, data)

        # Non-compiled languages (Python etc.) skip the BUILDING phase
        if new_phase == FilePhase.BUILDING and not self._compiled:
            new_phase = lc.process_event(EventType.BUILD_PASSED)
            logger.info("[%s] build step skipped (interpreted language)", file_path)

        # In checkpoint mode, BUILDING is handled at repo level by
        # PipelineExecutor.  The file stays in BUILDING until the
        # checkpoint explicitly triggers BUILD_PASSED or BUILD_FAILED.
        # For non-checkpoint mode (legacy), the per-file BuildVerifierAgent
        # handles it as before.

        # Config/deploy files skip testing
        if new_phase == FilePhase.TESTING and file_path in self._skip_testing:
            new_phase = lc.process_event(EventType.TEST_PASSED)
            logger.info("[%s] testing skipped (config/deploy layer)", file_path)

        return new_phase

    def get_files_in_phase(self, phase: FilePhase) -> list[str]:
        """Return all files currently in the given phase."""
        return [
            path for path, lc in self._lifecycles.items()
            if lc.phase == phase
        ]

    # ── Execution queries ───────────────────────────────────────────

    def get_actionable_files(self) -> list[tuple[str, FilePhase]]:
        """Return files ready for the next action.

        A file is actionable if:
          - PENDING and all dependency files are ready:
              * checkpoint mode: dep.phase in {TESTING, PASSED} — dep has
                cleared its tier's build checkpoint and is stable to depend on.
              * standard mode:   dep.phase not in {PENDING, GENERATING} —
                dep has been generated (safe for interpreted languages).
          - In any active phase (GENERATING, REVIEWING, FIXING, TESTING)

        Terminal files (PASSED, FAILED) are never returned.
        """
        actionable: list[tuple[str, FilePhase]] = []

        for path, lc in self._lifecycles.items():
            if lc.is_terminal:
                continue

            if lc.phase == FilePhase.PENDING:
                deps = self._deps.get(path, [])
                if self._checkpoint_mode:
                    # In checkpoint mode a dep is ready only once it has passed
                    # its tier's build checkpoint (phase >= TESTING).  This
                    # prevents a dependent from generating against a file that
                    # is still mid-review/fix and may be substantially rewritten.
                    deps_met = all(
                        dep not in self._lifecycles
                        or self._lifecycles[dep].phase in (
                            FilePhase.TESTING, FilePhase.PASSED, FilePhase.DEGRADED,
                        )
                        for dep in deps
                    )
                else:
                    # Interpreted languages: a dep is ready as soon as it has
                    # left the GENERATING phase (existing behaviour).
                    deps_met = all(
                        dep not in self._lifecycles
                        or self._lifecycles[dep].phase not in (
                            FilePhase.PENDING, FilePhase.GENERATING,
                        )
                        for dep in deps
                    )
                if deps_met:
                    actionable.append((path, lc.phase))
            else:
                actionable.append((path, lc.phase))

        return actionable

    def cascade_failures(self, scope: set[str] | None = None) -> list[str]:
        """Fail any PENDING file whose dependency has FAILED.

        When a file fails (timeout, retries exhausted, etc.), all files
        that depend on it can never proceed.  This method propagates FAILED
        status transitively through the dependency graph so dependents don't
        wait for the staleness timeout.

        Args:
            scope: If provided, only check files within this set (e.g. a
                tier's file set).  If None, check all files.

        Returns:
            List of file paths that were cascade-failed.
        """
        cascaded: list[str] = []
        # Iterate until no more cascades are possible (transitive closure).
        changed = True
        while changed:
            changed = False
            for path, lc in self._lifecycles.items():
                if lc.is_terminal or lc.phase != FilePhase.PENDING:
                    continue
                if scope is not None and path not in scope:
                    continue
                deps = self._deps.get(path, [])
                failed_deps = [
                    d for d in deps
                    if d in self._lifecycles
                    and self._lifecycles[d].phase == FilePhase.FAILED
                ]
                if failed_deps:
                    logger.error(
                        "[%s] Cascade FAILED — dependency %s failed",
                        path, failed_deps[0],
                    )
                    lc.process_event(EventType.RETRIES_EXHAUSTED, {
                        "reason": "dependency_failed",
                        "failed_deps": failed_deps,
                    })
                    cascaded.append(path)
                    changed = True  # re-scan for transitive dependents
        return cascaded

    def all_terminal(self) -> bool:
        """True when every file has reached PASSED or FAILED."""
        return all(lc.is_terminal for lc in self._lifecycles.values())

    # ── Observability ───────────────────────────────────────────────

    def get_stats(self) -> dict[str, int]:
        """Phase counts across all files."""
        return dict(Counter(lc.phase.value for lc in self._lifecycles.values()))

    def get_full_event_log(self) -> list[Event]:
        """Chronologically sorted event log across all files."""
        all_events: list[Event] = []
        for lc in self._lifecycles.values():
            all_events.extend(lc.event_log)
        all_events.sort(key=lambda e: e.timestamp)
        return all_events

    def get_results_summary(self) -> dict[str, Any]:
        """Summary suitable for pipeline reporting."""
        passed = [p for p, lc in self._lifecycles.items() if lc.phase == FilePhase.PASSED]
        failed = [p for p, lc in self._lifecycles.items() if lc.phase == FilePhase.FAILED]
        degraded = [p for p, lc in self._lifecycles.items() if lc.phase == FilePhase.DEGRADED]
        total_fixes = sum(lc.total_fix_count for lc in self._lifecycles.values())
        # Legacy field: files that hit test-fix limits (now DEGRADED phase;
        # kept for backward compatibility with existing dashboard consumers).
        tests_degraded = [
            p for p, lc in self._lifecycles.items()
            if lc.phase == FilePhase.DEGRADED and lc.test_fix_count > 0
        ]
        return {
            "total_files": len(self._lifecycles),
            "passed": len(passed),
            "failed": len(failed),
            "degraded": len(degraded),
            "tests_degraded": len(tests_degraded),
            "total_fix_cycles": total_fixes,
            "passed_files": passed,
            "failed_files": failed,
            "degraded_files": degraded,
            "tests_degraded_files": tests_degraded,
        }

    # ── Persistence (resume support) ────────────────────────────────

    def save_state(self, path: str) -> None:
        """Persist lifecycle state to a JSON file for later resume.

        Saves per-file phase, fix counts, and dependency map so a resumed
        pipeline can skip files that already PASSED.
        """
        import json as _json

        state = {
            "version": 1,
            "checkpoint_mode": self._checkpoint_mode,
            "compiled": self._compiled,
            "deps": {k: list(v) for k, v in self._deps.items()},
            "skip_testing": sorted(self._skip_testing),
            "files": {},
        }
        for path_key, lc in self._lifecycles.items():
            state["files"][path_key] = {
                "phase": lc.phase.value,
                "review_fix_count": lc.review_fix_count,
                "test_fix_count": lc.test_fix_count,
                "build_fix_count": lc.build_fix_count,
                "generation_task_type": lc.generation_task_type,
            }

        with open(path, "w", encoding="utf-8") as f:
            _json.dump(state, f, indent=2)
        logger.info("Saved lifecycle state to %s (%d files)", path, len(state["files"]))

    @classmethod
    def load_state(
        cls,
        path: str,
        *,
        skip_failed: bool = False,
        retry_files: set[str] | None = None,
    ) -> "LifecycleEngine":
        """Load lifecycle state from a JSON file saved by ``save_state()``.

        Files that were PASSED or DEGRADED keep their terminal state so the
        executor skips them.  Files that were FAILED or in intermediate phases
        are reset to PENDING so they can be retried.

        Args:
            path: Path to the state JSON file written by ``save_state()``.
            skip_failed: When True, FAILED files are also kept as terminal
                instead of being reset to PENDING.  Use this to resume a
                pipeline without retrying files that are known to be broken
                (e.g. missing dependencies that need manual intervention).
            retry_files: Explicit set of file paths to force back to PENDING
                regardless of their saved phase (including PASSED/DEGRADED).
                Useful for selectively re-generating specific files after a
                targeted code fix without re-running the full pipeline.

        Returns a new LifecycleEngine with restored state.
        """
        import json as _json

        with open(path, "r", encoding="utf-8") as f:
            state = _json.load(f)

        if state.get("version", 0) != 1:
            raise ValueError(f"Unsupported state file version: {state.get('version')}")

        file_paths = list(state["files"].keys())
        file_deps = {k: list(v) for k, v in state.get("deps", {}).items()}

        engine = cls(
            file_paths=file_paths,
            file_deps=file_deps,
            compiled=state.get("compiled", False),
            checkpoint_mode=state.get("checkpoint_mode", False),
        )

        # Build the set of phases that should be kept as-is (not retried).
        # Always keep PASSED and DEGRADED; optionally also keep FAILED.
        _KEEP_PHASES: set[str] = {FilePhase.PASSED.value, FilePhase.DEGRADED.value}
        if skip_failed:
            _KEEP_PHASES.add(FilePhase.FAILED.value)

        _force_retry = retry_files or set()

        for file_path, file_state in state["files"].items():
            lc = engine.get_lifecycle(file_path)
            saved_phase = file_state["phase"]

            if file_path in _force_retry:
                # Explicit retry requested — reset to PENDING regardless of phase
                lc.phase = FilePhase.PENDING
                logger.info(
                    "[Resume] %s — forced retry (was %s)", file_path, saved_phase
                )
            elif saved_phase in _KEEP_PHASES:
                # Keep terminal state — executor will skip this file
                lc.phase = FilePhase(saved_phase)
                logger.info("[Resume] %s — keeping %s", file_path, saved_phase)
            else:
                # Reset to PENDING for retry (covers FAILED, FIXING, etc.)
                lc.phase = FilePhase.PENDING
                logger.info(
                    "[Resume] %s — reset to PENDING (was %s)", file_path, saved_phase
                )

        for skip_path in state.get("skip_testing", []):
            if engine.has_file(skip_path):
                engine.skip_testing(skip_path)

        return engine
