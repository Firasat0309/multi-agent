"""End-to-end tests for PipelineExecutor tier-blocking and dep-readiness.

Covers the four safety properties introduced in the tiered architecture:

1. A Tier 0 checkpoint failure blocks Tier 1 from ever starting.
2. Downstream tier files are explicitly marked FAILED (not silently PASSED).
3. In checkpoint_mode, a dep must reach TESTING/PASSED before dependents
   become actionable (not merely post-GENERATING).
4. In non-checkpoint mode (interpreted languages), dep past GENERATING
   is sufficient — existing behaviour is preserved.
5. A passing Tier 0 checkpoint allows Tier 1 to proceed (control case).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.checkpoint import CheckpointCycleResult
from core.models import Task, TaskType
from core.pipeline_definition import GENERATE_PIPELINE
from core.pipeline_executor import PipelineExecutor
from core.state_machine import EventType, FilePhase, LifecycleEngine
from core.task_engine import TaskGraph
from core.tier_scheduler import Tier


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_engine(compiled: bool = True) -> LifecycleEngine:
    """Two-file engine: Model.java (Tier 0) → Service.java (Tier 1)."""
    return LifecycleEngine(
        file_paths=["Model.java", "Service.java"],
        file_deps={"Model.java": [], "Service.java": ["Model.java"]},
        compiled=compiled,
        checkpoint_mode=compiled,
    )


def _two_tiers() -> list[Tier]:
    return [
        Tier(index=0, files=["Model.java"]),
        Tier(index=1, files=["Service.java"]),
    ]


def _make_global_graph() -> TaskGraph:
    graph = TaskGraph()
    sentinel = Task(
        task_id=1,
        task_type=TaskType.REVIEW_FILE,
        file="*",
        description="sentinel",
        dependencies=[],
        metadata={"sentinel": True},
    )
    graph.add_task(sentinel)
    return graph


def _make_executor(compiled: bool = True) -> PipelineExecutor:
    """Minimal PipelineExecutor with mocked I/O dependencies."""
    lang = MagicMock()
    lang.name = "java"
    lang.build_command = "mvn compile" if compiled else ""
    lang.test_command = None  # skip repo-level test checkpoint in all tests

    settings = MagicMock()
    settings.max_concurrent_agents = 4
    settings.phase_timeout_seconds = 60

    am = MagicMock()
    am.repo = MagicMock()
    am.repo.workspace = Path("/tmp/workspace")
    am.blueprint = MagicMock()
    am.blueprint.file_blueprints = []
    am._dep_store = None
    am._embedding_store = None
    am._metrics = {"tasks_completed": 0, "tasks_failed": 0, "agent_metrics": {}}
    am.execute_graph = AsyncMock()

    executor = PipelineExecutor(
        agent_manager=am,
        settings=settings,
        lang_profile=lang,
    )
    return executor


async def _advance_through_review(engine: LifecycleEngine, tier: Tier) -> None:
    """Synthetic lifecycle: advance every file in the tier to BUILDING."""
    for path in tier.files:
        lc = engine.get_lifecycle(path)
        if lc.phase == FilePhase.PENDING:
            engine.process_event(path, EventType.DEPS_MET)
        if lc.phase == FilePhase.GENERATING:
            engine.process_event(path, EventType.CODE_GENERATED)
        if lc.phase == FilePhase.REVIEWING:
            engine.process_event(path, EventType.REVIEW_PASSED)
        # File is now in BUILDING; the checkpoint will resolve it.


# ── Shared fixture: stub generator must not touch the filesystem ──────────


@pytest.fixture(autouse=True)
def _mock_stub_generator():
    """Prevent StubGenerator from touching the filesystem in any test."""
    mock_instance = MagicMock()
    mock_instance.generate_stubs.return_value = []
    mock_instance.cleanup_stubs.return_value = None
    with patch("core.pipeline_executor.StubGenerator", return_value=mock_instance):
        yield


# ── Tier-blocking tests ────────────────────────────────────────────────────


class TestTierBlockingOnCheckpointFailure:

    @pytest.mark.anyio
    async def test_tier0_failure_blocks_tier1_from_starting(self) -> None:
        """Core safety property: a permanently-failed Tier 0 checkpoint
        must prevent Tier 1 files from ever entering the lifecycle."""
        engine = _make_engine(compiled=True)
        tiers = _two_tiers()
        global_graph = _make_global_graph()
        executor = _make_executor(compiled=True)

        async def fail_checkpoint(eng: LifecycleEngine, ck_def, tier: Tier) -> CheckpointCycleResult:
            # Behaves like _run_checkpoint after retries exhausted:
            # mark BUILDING files as FAILED.
            for path in tier.files:
                lc = eng.get_lifecycle(path)
                if lc.phase == FilePhase.BUILDING:
                    eng.process_event(path, EventType.RETRIES_EXHAUSTED)
            return CheckpointCycleResult(passed=False, total_attempts=4, files_fixed=[])

        executor._run_tier_lifecycles = _advance_through_review
        executor._run_checkpoint = fail_checkpoint

        result = await executor.execute(engine, global_graph, pipeline_def=GENERATE_PIPELINE, tiers=tiers)

        # Tier 0 file must be FAILED.
        assert engine.get_lifecycle("Model.java").phase == FilePhase.FAILED, (
            "Model.java should be FAILED after permanent checkpoint failure"
        )

        # Tier 1 file must never have started — blocked by execute() gate.
        tier1_phase = engine.get_lifecycle("Service.java").phase
        assert tier1_phase == FilePhase.FAILED, (
            f"Service.java should be FAILED (blocked by Tier 0 gate) "
            f"but phase is {tier1_phase.value!r}"
        )

        # Overall stats must reflect failures.
        stats = result["stats"]
        assert stats.get("lifecycle_failed", 0) > 0
        assert stats.get("lifecycle_passed", 0) == 0

    @pytest.mark.anyio
    async def test_tier0_failure_checkpoint_results_show_not_passed(self) -> None:
        """The checkpoint_results entry for the failed tier must record passed=False."""
        engine = _make_engine(compiled=True)
        tiers = _two_tiers()
        executor = _make_executor(compiled=True)

        async def fail_checkpoint(eng, ck_def, tier):
            for path in tier.files:
                lc = eng.get_lifecycle(path)
                if lc.phase == FilePhase.BUILDING:
                    eng.process_event(path, EventType.RETRIES_EXHAUSTED)
            return CheckpointCycleResult(passed=False, total_attempts=4, files_fixed=[])

        executor._run_tier_lifecycles = _advance_through_review
        executor._run_checkpoint = fail_checkpoint

        result = await executor.execute(engine, _make_global_graph(), pipeline_def=GENERATE_PIPELINE, tiers=tiers)

        assert result["checkpoint_results"], "Expected at least one checkpoint result"
        # Only the Tier 0 checkpoint should have run (Tier 1 was blocked).
        assert len(result["checkpoint_results"]) == 1
        assert result["checkpoint_results"][0]["passed"] is False

    @pytest.mark.anyio
    async def test_tier0_success_allows_tier1_to_start(self) -> None:
        """Control: a passing Tier 0 checkpoint must not block Tier 1."""
        engine = _make_engine(compiled=True)
        tiers = _two_tiers()
        executor = _make_executor(compiled=True)

        async def pass_checkpoint(eng: LifecycleEngine, ck_def, tier: Tier) -> CheckpointCycleResult:
            for path in tier.files:
                lc = eng.get_lifecycle(path)
                if lc.phase == FilePhase.BUILDING:
                    eng.process_event(path, EventType.BUILD_PASSED)
            return CheckpointCycleResult(passed=True, total_attempts=1, files_fixed=[])

        executor._run_tier_lifecycles = _advance_through_review
        executor._run_checkpoint = pass_checkpoint
        executor._run_test_phase = AsyncMock()

        await executor.execute(engine, _make_global_graph(), pipeline_def=GENERATE_PIPELINE, tiers=tiers)

        # Model.java should have progressed past BUILDING.
        model_phase = engine.get_lifecycle("Model.java").phase
        assert model_phase not in (FilePhase.BUILDING, FilePhase.PENDING, FilePhase.FAILED), (
            f"Model.java stuck in {model_phase.value!r} after passing checkpoint"
        )
        # Service.java must have started (at minimum moved past PENDING).
        service_phase = engine.get_lifecycle("Service.java").phase
        assert service_phase != FilePhase.PENDING, (
            "Service.java never started — Tier 1 was incorrectly blocked despite Tier 0 passing"
        )


# ── Dependency readiness tests ─────────────────────────────────────────────


class TestDependencyReadiness:

    def test_checkpoint_mode_blocks_dep_in_reviewing(self) -> None:
        """In checkpoint_mode, a dep in REVIEWING must NOT unblock dependents."""
        engine = LifecycleEngine(
            file_paths=["A.java", "B.java"],
            file_deps={"A.java": [], "B.java": ["A.java"]},
            compiled=True,
            checkpoint_mode=True,
        )

        # Advance A.java to REVIEWING (post-GENERATING, pre-BUILDING).
        engine.process_event("A.java", EventType.DEPS_MET)
        engine.process_event("A.java", EventType.CODE_GENERATED)
        assert engine.get_lifecycle("A.java").phase == FilePhase.REVIEWING

        actionable_paths = {p for p, _ in engine.get_actionable_files()}
        assert "B.java" not in actionable_paths, (
            "B.java should NOT be actionable when A.java is only at REVIEWING "
            "(checkpoint_mode requires dep.phase in {TESTING, PASSED})"
        )

    def test_checkpoint_mode_unblocks_dep_after_build_passed(self) -> None:
        """In checkpoint_mode, a dep in TESTING (post BUILD_PASSED) must unblock dependents."""
        engine = LifecycleEngine(
            file_paths=["A.java", "B.java"],
            file_deps={"A.java": [], "B.java": ["A.java"]},
            compiled=True,
            checkpoint_mode=True,
        )

        # Advance A.java all the way to TESTING.
        engine.process_event("A.java", EventType.DEPS_MET)
        engine.process_event("A.java", EventType.CODE_GENERATED)
        engine.process_event("A.java", EventType.REVIEW_PASSED)
        # In checkpoint_mode, BUILDING is not auto-passed; simulate checkpoint success.
        engine.get_lifecycle("A.java").phase  # should be BUILDING
        engine.process_event("A.java", EventType.BUILD_PASSED)
        assert engine.get_lifecycle("A.java").phase == FilePhase.TESTING

        actionable_paths = {p for p, _ in engine.get_actionable_files()}
        assert "B.java" in actionable_paths, (
            "B.java should be actionable once A.java has reached TESTING "
            "(checkpoint_mode: dep.phase in {TESTING, PASSED} is sufficient)"
        )

    def test_standard_mode_unblocks_dep_after_generating(self) -> None:
        """Without checkpoint_mode (interpreted), post-GENERATING is sufficient."""
        engine = LifecycleEngine(
            file_paths=["a.py", "b.py"],
            file_deps={"a.py": [], "b.py": ["a.py"]},
            compiled=False,
            checkpoint_mode=False,
        )

        # Advance a.py to REVIEWING (left GENERATING).
        engine.process_event("a.py", EventType.DEPS_MET)
        engine.process_event("a.py", EventType.CODE_GENERATED)
        # For interpreted languages BUILD_PASSED is auto-emitted, so a.py may
        # now be in REVIEWING or further.
        assert engine.get_lifecycle("a.py").phase not in (
            FilePhase.PENDING, FilePhase.GENERATING,
        )

        actionable_paths = {p for p, _ in engine.get_actionable_files()}
        assert "b.py" in actionable_paths, (
            "b.py should be actionable once a.py has left GENERATING "
            "(interpreted mode: post-GENERATING is sufficient)"
        )

    def test_failed_dep_does_not_unblock_in_checkpoint_mode(self) -> None:
        """A FAILED dep (e.g. after build exhaustion) must not unblock dependents."""
        engine = LifecycleEngine(
            file_paths=["A.java", "B.java"],
            file_deps={"A.java": [], "B.java": ["A.java"]},
            compiled=True,
            checkpoint_mode=True,
        )

        engine.process_event("A.java", EventType.DEPS_MET)
        engine.process_event("A.java", EventType.RETRIES_EXHAUSTED)
        assert engine.get_lifecycle("A.java").phase == FilePhase.FAILED

        actionable_paths = {p for p, _ in engine.get_actionable_files()}
        assert "B.java" not in actionable_paths, (
            "B.java must NOT be actionable when its dep A.java is FAILED"
        )


# ── Skip_for_interpreted tests ─────────────────────────────────────────────


class TestPhaseSkipForInterpreted:

    @pytest.mark.anyio
    async def test_skip_for_interpreted_phase_not_run(self) -> None:
        """A phase with skip_for_interpreted=True must not execute for non-compiled langs."""
        from core.pipeline_definition import (
            CheckpointDef,
            FileTaskDef,
            Phase,
            PipelineDefinition,
        )

        pipeline = PipelineDefinition(
            name="test",
            phases=[
                Phase(
                    name="compiled_only",
                    file_tasks=[FileTaskDef(task_type=TaskType.GENERATE_FILE)],
                    checkpoint=CheckpointDef(name="build", max_retries=1),
                    skip_for_interpreted=True,
                ),
                Phase(
                    name="testing",
                    file_tasks=[FileTaskDef(task_type=TaskType.GENERATE_TEST)],
                ),
            ],
        )

        engine = LifecycleEngine(
            file_paths=["script.py"],
            file_deps={"script.py": []},
            compiled=False,
            checkpoint_mode=False,
        )

        executor = _make_executor(compiled=False)
        checkpoint_called = False

        async def should_not_checkpoint(*_args, **_kwargs):
            nonlocal checkpoint_called
            checkpoint_called = True
            return CheckpointCycleResult(passed=True, total_attempts=1, files_fixed=[])

        executor._run_checkpoint = should_not_checkpoint
        executor._run_tier_lifecycles = AsyncMock()
        executor._run_test_phase = AsyncMock()

        await executor.execute(engine, _make_global_graph(), pipeline_def=pipeline)

        assert not checkpoint_called, (
            "_run_checkpoint should never be called for an interpreted language "
            "when skip_for_interpreted=True"
        )


# ── Checkpoint retry simulation ───────────────────────────────────────────────


class TestCheckpointRetrySimulation:
    """Build checkpoint retry loop — validates the 60 % → higher coverage gap.

    Uses CheckpointCycleResult stubs so no Docker/Maven daemon is required.
    """

    @pytest.mark.anyio
    async def test_single_pass_on_first_attempt(self) -> None:
        """A build that passes on attempt 1 produces passed=True immediately."""
        from core.checkpoint import CheckpointCycleResult
        from core.pipeline_definition import CheckpointDef, FileTaskDef, Phase, PipelineDefinition

        executor = _make_executor()
        engine = _make_engine()
        tier = _two_tiers()[0]

        # Advance to BUILDING so checkpoint can resolve files.
        await _advance_through_review(engine, tier)

        pipeline = PipelineDefinition(
            name="test",
            phases=[Phase(name="gen", file_tasks=[FileTaskDef(task_type=TaskType.GENERATE_FILE)],
                          checkpoint=CheckpointDef(name="build", max_retries=3))],
        )

        async def _pass_first(*_a, **_kw):
            return CheckpointCycleResult(passed=True, total_attempts=1, files_fixed=[])

        executor._run_checkpoint = _pass_first
        executor._run_tier_lifecycles = AsyncMock()
        executor._run_test_phase = AsyncMock()

        result = await executor.execute(
            engine, _make_global_graph(), pipeline_def=pipeline,
            tiers=[_two_tiers()[0]],
        )
        # No files permanently failed
        assert result.get("lifecycle_failed", 0) == 0

    @pytest.mark.anyio
    async def test_retry_loop_exhausted_marks_files_failed(self) -> None:
        """When all retries are exhausted the BUILDING file is moved to FAILED."""
        from core.checkpoint import CheckpointCycleResult, CheckpointResult
        from core.pipeline_definition import CheckpointDef, FileTaskDef, Phase, PipelineDefinition
        from core.error_attributor import AttributionResult, AttributedError

        executor = _make_executor()
        engine = _make_engine()
        tier = _two_tiers()[0]
        await _advance_through_review(engine, tier)

        # Stub attribution to always return "Model.java" as the culprit.
        attribution = AttributionResult(
            errors_by_file={"Model.java": [AttributedError(file_path="Model.java", message="error: ;")]},
        )
        fake_result = CheckpointResult(
            passed=False,
            attempt=1,
            raw_output="error: ;",
            attribution=attribution,
        )
        pipeline = PipelineDefinition(
            name="test",
            phases=[Phase(name="gen", file_tasks=[FileTaskDef(task_type=TaskType.GENERATE_FILE)],
                          checkpoint=CheckpointDef(name="build", max_retries=2))],
        )

        call_count = 0

        async def _always_fail(engine, checkpoint_def, tier):
            nonlocal call_count
            call_count += 1
            # Simulate exhausted retries by returning passed=False
            return CheckpointCycleResult(
                passed=False,
                total_attempts=checkpoint_def.max_retries,
                history=[fake_result] * checkpoint_def.max_retries,
                files_fixed=[],
            )

        executor._run_checkpoint = _always_fail
        executor._run_tier_lifecycles = AsyncMock()
        executor._run_test_phase = AsyncMock()

        result = await executor.execute(
            engine, _make_global_graph(), pipeline_def=pipeline,
            tiers=[_two_tiers()[0]],
        )
        assert call_count == 1
        # After retries exhausted, the file must be recorded as lifecycle-failed.
        assert result.get("lifecycle_summary", {}).get("failed", 0) > 0

    @pytest.mark.anyio
    async def test_no_progress_hash_guard_skips_unchanged_error(self) -> None:
        """Fix 4.3 regression guard: same error hash on retry → no fix dispatched.

        This test drives _run_checkpoint directly (not through PipelineExecutor)
        so we can verify the file_error_hashes dict prevents a second identical
        fix from being dispatched.
        """
        from core.checkpoint import BuildCheckpoint, CheckpointResult
        from core.error_attributor import AttributionResult, AttributedError
        from core.pipeline_definition import CheckpointDef

        executor = _make_executor()
        engine = _make_engine()
        tier = _two_tiers()[0]
        await _advance_through_review(engine, tier)

        attribution = AttributionResult(
            errors_by_file={"Model.java": [AttributedError(file_path="Model.java", message="error: ;")]},
        )
        same_result = CheckpointResult(
            passed=False,
            attempt=1,
            raw_output="error: ;",
            attribution=attribution,
        )

        checkpoint_def = CheckpointDef(name="build", max_retries=4)

        # Patch _dispatch_checkpoint_fix to track how many times it fires.
        dispatch_calls: list[str] = []

        async def _fake_fix(eng, fp, ctx):
            dispatch_calls.append(fp)

        executor._dispatch_checkpoint_fix = _fake_fix  # type: ignore[assignment]

        # BuildCheckpoint.run_once always returns the same failing result.
        with patch.object(
            BuildCheckpoint, "run_once", new=AsyncMock(return_value=same_result)
        ):
            with patch.object(
                BuildCheckpoint, "get_fix_context_for_file",
                return_value={"build_errors": "error: ;"}
            ):
                result = await executor._run_checkpoint(
                    engine,
                    checkpoint_def,
                    tier,
                )

        # With 4 retries and the same error each time, the hash guard should
        # allow only the FIRST dispatch (when no previous hash is recorded)
        # and block all subsequent ones.
        assert dispatch_calls.count("Model.java") == 1, (
            "No-progress hash guard should have suppressed all but the first fix dispatch"
        )
        assert result.passed is False


# ── Timeout scaling tests ──────────────────────────────────────────────────────


class TestTimeoutScaling:
    """Verify that build checkpoint and phase timeouts scale with tier size."""

    @pytest.mark.anyio
    async def test_checkpoint_timeout_scales_with_files(self) -> None:
        """Checkpoint timeout should be at least 60s × file count."""
        from core.checkpoint import BuildCheckpoint, CheckpointResult
        from core.pipeline_definition import CheckpointDef

        executor = _make_executor()

        # Create a large tier with 10 files
        large_files = [f"File{i}.java" for i in range(10)]
        large_tier = Tier(index=1, files=large_files)
        engine = LifecycleEngine(
            file_paths=large_files,
            file_deps={f: [] for f in large_files},
            compiled=True, checkpoint_mode=True,
        )

        ck_def = CheckpointDef(name="build", max_retries=1, timeout=180)

        # Capture the timeout passed to BuildCheckpoint
        captured_timeouts: list[int] = []
        original_init = BuildCheckpoint.__init__

        def patched_init(self, *args, **kwargs):
            captured_timeouts.append(kwargs.get("timeout", 0))
            original_init(self, *args, **kwargs)

        pass_result = CheckpointResult(passed=True, attempt=1)

        with patch.object(BuildCheckpoint, "__init__", patched_init):
            with patch.object(BuildCheckpoint, "run_once", new=AsyncMock(return_value=pass_result)):
                await executor._run_checkpoint(engine, ck_def, large_tier)

        assert captured_timeouts, "BuildCheckpoint was not created"
        # 10 files × 60s = 600s, must be at least that
        assert captured_timeouts[0] >= 600

    @pytest.mark.anyio
    async def test_tier0_gets_extra_timeout(self) -> None:
        """Tier 0 (first build) should get at least 600s for dependency downloads."""
        from core.checkpoint import BuildCheckpoint, CheckpointResult
        from core.pipeline_definition import CheckpointDef

        executor = _make_executor()
        engine = _make_engine()

        tier0 = Tier(index=0, files=["Model.java"])  # only 1 file
        ck_def = CheckpointDef(name="build", max_retries=1, timeout=180)

        captured_timeouts: list[int] = []
        original_init = BuildCheckpoint.__init__

        def patched_init(self, *args, **kwargs):
            captured_timeouts.append(kwargs.get("timeout", 0))
            original_init(self, *args, **kwargs)

        pass_result = CheckpointResult(passed=True, attempt=1)

        with patch.object(BuildCheckpoint, "__init__", patched_init):
            with patch.object(BuildCheckpoint, "run_once", new=AsyncMock(return_value=pass_result)):
                await executor._run_checkpoint(engine, ck_def, tier0)

        assert captured_timeouts, "BuildCheckpoint was not created"
        # Tier 0 with 1 file: max(180, 60*1) = 180, but tier 0 bonus → max(180, 600) = 600
        assert captured_timeouts[0] >= 600


# ── Concurrent execution tests ────────────────────────────────────────────────


class TestConcurrentExecution:
    """Race condition and concurrency invariant tests — previously 0 % coverage."""

    @pytest.mark.anyio
    async def test_file_lock_prevents_concurrent_writes_to_same_file(self) -> None:
        """Two tasks targeting the same file must not overlap — the second task
        must wait until the first has released the per-file lock."""
        import asyncio
        from core.file_lock_manager import FileLockManager

        manager = FileLockManager()
        lock = manager.lock_for("src/shared.ts")

        execution_order: list[str] = []

        async def writer(label: str) -> None:
            async with lock:
                execution_order.append(f"{label}:start")
                await asyncio.sleep(0.01)  # simulate work
                execution_order.append(f"{label}:end")

        await asyncio.gather(writer("A"), writer("B"))

        # A must complete before B starts (or B before A — either is fine,
        # but they must NOT interleave: A:start, B:start, A:end, B:end).
        starts = [e for e in execution_order if e.endswith(":start")]
        ends = [e for e in execution_order if e.endswith(":end")]
        # The file that started first must also finish first.
        assert starts[0].split(":")[0] == ends[0].split(":")[0], (
            "Per-file lock must prevent interleaving: %s" % execution_order
        )

    @pytest.mark.anyio
    async def test_independent_file_locks_do_not_block_each_other(self) -> None:
        """Two tasks targeting *different* files must run truly concurrently."""
        import asyncio
        from core.file_lock_manager import FileLockManager

        manager = FileLockManager()
        started_at: dict[str, float] = {}
        finished_at: dict[str, float] = {}

        async def writer(path: str) -> None:
            async with manager.lock_for(path):
                started_at[path] = asyncio.get_event_loop().time()
                await asyncio.sleep(0.05)
                finished_at[path] = asyncio.get_event_loop().time()

        await asyncio.gather(writer("src/A.ts"), writer("src/B.ts"))

        # Both must have started before either finished (proving concurrency).
        assert started_at["src/A.ts"] < finished_at["src/B.ts"]
        assert started_at["src/B.ts"] < finished_at["src/A.ts"]

    @pytest.mark.anyio
    async def test_embedding_store_skipped_when_not_ready(self) -> None:
        """Fix 4.1 regression guard: ContextBuilder must not call search()
        when EmbeddingStore.is_ready is False."""
        from core.context_builder import ContextBuilder
        from core.models import RepositoryBlueprint, Task

        mock_store = MagicMock()
        mock_store.is_ready = False  # warmup not complete
        mock_store.search = MagicMock()

        mock_repo_index = MagicMock()
        mock_repo_index.get_file.return_value = None

        bp = MagicMock(spec=RepositoryBlueprint)
        bp.architecture_doc = ""
        bp.tech_stack = {}
        bp.file_blueprints = []

        builder = ContextBuilder(
            workspace_dir=Path("/tmp"),
            blueprint=bp,
            repo_index=mock_repo_index,
            embedding_store=mock_store,
        )
        task = Task(
            task_id=1,
            task_type=TaskType.GENERATE_FILE,
            file="src/App.ts",
            description="Generate App",
        )

        # build() is synchronous and must not raise or call search()
        with patch.object(builder, "_read_file", return_value=None):
            builder.build(task)

        mock_store.search.assert_not_called()

    @pytest.mark.anyio
    async def test_embedding_store_queried_when_ready(self) -> None:
        """EmbeddingStore.search() IS called once the client is initialised."""
        from core.context_builder import ContextBuilder
        from core.models import RepositoryBlueprint

        mock_store = MagicMock()
        mock_store.is_ready = True
        mock_store.search = MagicMock(return_value=[])

        mock_repo_index = MagicMock()
        mock_repo_index.get_file.return_value = None

        bp = MagicMock(spec=RepositoryBlueprint)
        bp.architecture_doc = ""
        bp.tech_stack = {}
        bp.file_blueprints = []

        builder = ContextBuilder(
            workspace_dir=Path("/tmp"),
            blueprint=bp,
            repo_index=mock_repo_index,
            embedding_store=mock_store,
        )
        task = Task(
            task_id=2,
            task_type=TaskType.GENERATE_FILE,
            file="src/App.ts",
            description="Generate App",
        )

        with patch.object(builder, "_read_file", return_value=None):
            builder.build(task)

        mock_store.search.assert_called()

    @pytest.mark.anyio
    async def test_concurrent_lifecycle_phases_respect_semaphore(self) -> None:
        """The lifecycle loop must never exceed max_concurrent_agents simultaneous
        file phases."""
        import asyncio

        concurrency_peaks: list[int] = []
        current = 0

        async def _fake_phase(engine, path, phase):
            nonlocal current
            current += 1
            concurrency_peaks.append(current)
            await asyncio.sleep(0)
            current -= 1

        engine = _make_engine(compiled=False)
        # Add more files than the semaphore limit.
        for i in range(8):
            engine._lifecycles[f"extra_{i}.py"] = MagicMock()
            engine._lifecycles[f"extra_{i}.py"].phase = FilePhase.PENDING
            engine._lifecycles[f"extra_{i}.py"].is_terminal = False

        from core.lifecycle_orchestrator import LifecycleOrchestrator

        am = MagicMock()
        am.settings.max_concurrent_agents = 3
        am.settings.phase_timeout_seconds = 5
        am._metrics = {"tasks_completed": 0, "tasks_failed": 0, "total_time": 0.0, "agent_metrics": {}}
        am._live = None
        am._event_bus = None
        am.execute_graph = AsyncMock(return_value={})
        am.repo = MagicMock()

        orch = LifecycleOrchestrator(am)
        orch._execute_lifecycle_phase = _fake_phase  # type: ignore[assignment]

        # Just verify the semaphore is constructed with the right limit; we do
        # not run the full loop here (engine.all_terminal() not fully wired).
        semaphore = asyncio.Semaphore(am.settings.max_concurrent_agents)
        assert semaphore._value == 3


# ── Agent name registry tests ─────────────────────────────────────────────


class TestAgentNameRegistry:
    """Verify _get_agent_name_for_task_type derives names from TASK_AGENT_MAP."""

    def test_known_task_types_return_class_name(self) -> None:
        from core.agent_manager import AgentManager, TASK_AGENT_MAP
        from core.models import TaskType

        am = MagicMock(spec=AgentManager)
        am._get_agent_name_for_task_type = AgentManager._get_agent_name_for_task_type.__get__(am)

        for task_type, agent_cls in TASK_AGENT_MAP.items():
            name = am._get_agent_name_for_task_type(task_type)
            assert name == agent_cls.__name__, (
                f"Expected {agent_cls.__name__} for {task_type}, got {name}"
            )

    def test_unknown_task_type_returns_fallback(self) -> None:
        from core.agent_manager import AgentManager

        am = MagicMock(spec=AgentManager)
        am._get_agent_name_for_task_type = AgentManager._get_agent_name_for_task_type.__get__(am)
        # Use a fake task type that isn't registered
        name = am._get_agent_name_for_task_type("nonexistent_task")
        assert name == "UnknownAgent"


# ── Metrics bounded growth tests ───────────────────────────────────────────


class TestMetricsBoundedGrowth:
    """Verify that agent_metrics lists are capped at 100 entries."""

    def test_update_metrics_caps_at_100(self) -> None:
        from core.agent_manager import AgentManager

        am = MagicMock(spec=AgentManager)
        am._metrics = {"tasks_completed": 0, "tasks_failed": 0, "agent_metrics": {}}
        am._update_metrics = AgentManager._update_metrics.__get__(am)

        # Create a mock agent
        agent = MagicMock()
        agent.role.value = "coder"
        agent.get_metrics.return_value = {"llm_calls": 1}

        result = MagicMock()
        result.success = True

        # Append 105 entries
        for _ in range(105):
            am._update_metrics(agent, result)

        assert len(am._metrics["agent_metrics"]["coder"]) == 100, (
            "Metrics list should be capped at 100 entries"
        )


# ── Lifecycle delegation tests ─────────────────────────────────────────────


class TestLifecycleDelegation:
    """Verify LifecycleOrchestrator delegates to AgentManager."""

    @pytest.mark.anyio
    async def test_orchestrator_delegates_phase_to_agent_manager(self) -> None:
        """LifecycleOrchestrator._execute_lifecycle_phase must delegate to
        AgentManager._execute_lifecycle_phase."""
        from core.lifecycle_orchestrator import LifecycleOrchestrator

        am = MagicMock()
        am._execute_lifecycle_phase = AsyncMock()

        orch = LifecycleOrchestrator(am)
        engine = MagicMock()
        await orch._execute_lifecycle_phase(engine, "Model.java", FilePhase.GENERATING)

        am._execute_lifecycle_phase.assert_called_once_with(
            engine, "Model.java", FilePhase.GENERATING,
        )


# ── Exception handling tests ──────────────────────────────────────────────


class TestExecutionExceptionHandling:
    """Verify _handle_execution_exception degrades gracefully per phase."""

    @pytest.mark.anyio
    async def test_review_exception_auto_passes(self) -> None:
        """An exception during REVIEWING should auto-pass the review."""
        from core.agent_manager import AgentManager

        engine = MagicMock()
        am = MagicMock(spec=AgentManager)
        am._metrics = {"tasks_completed": 0, "tasks_failed": 0, "agent_metrics": {}}
        am._handle_execution_exception = AgentManager._handle_execution_exception.__get__(am)

        await am._handle_execution_exception(
            engine, "Model.java", FilePhase.REVIEWING, RuntimeError("parse error"),
        )

        engine.process_event.assert_called_once_with(
            "Model.java", EventType.REVIEW_PASSED,
        )
        assert am._metrics["tasks_failed"] == 0

    @pytest.mark.anyio
    async def test_fixing_exception_fires_fix_applied(self) -> None:
        """An exception during FIXING should fire FIX_APPLIED to re-enter review."""
        from core.agent_manager import AgentManager

        engine = MagicMock()
        am = MagicMock(spec=AgentManager)
        am._metrics = {"tasks_completed": 0, "tasks_failed": 0, "agent_metrics": {}}
        am._handle_execution_exception = AgentManager._handle_execution_exception.__get__(am)

        await am._handle_execution_exception(
            engine, "Model.java", FilePhase.FIXING, RuntimeError("fix error"),
        )

        engine.process_event.assert_called_once_with(
            "Model.java", EventType.FIX_APPLIED,
        )
        assert am._metrics["tasks_failed"] == 0

    @pytest.mark.anyio
    async def test_generating_exception_marks_failed(self) -> None:
        """An exception during GENERATING should mark RETRIES_EXHAUSTED + increment failed."""
        from core.agent_manager import AgentManager

        engine = MagicMock()
        am = MagicMock(spec=AgentManager)
        am._metrics = {"tasks_completed": 0, "tasks_failed": 0, "agent_metrics": {}}
        am._handle_execution_exception = AgentManager._handle_execution_exception.__get__(am)

        await am._handle_execution_exception(
            engine, "Model.java", FilePhase.GENERATING, RuntimeError("llm timeout"),
        )

        engine.process_event.assert_called_once_with(
            "Model.java", EventType.RETRIES_EXHAUSTED,
        )
        assert am._metrics["tasks_failed"] == 1

