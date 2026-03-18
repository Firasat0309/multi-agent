"""Unified pipeline executor — tier-based generation with repo-level checkpoints.

Replaces the split between ``execute_with_lifecycle()`` (Generate pipeline)
and ``execute_graph()`` (Enhance pipeline) with a single execution engine
that supports:

  1. **Tier-based file scheduling** — files are grouped by dependency depth
     and processed tier-by-tier so foundational files compile before
     dependent files are generated.

  2. **Per-file review cycles** — each file goes through Generate/Modify →
     Review → Fix loops driven by the existing ``FileLifecycle`` state
     machine.

  3. **Repo-level build checkpoints** — after each tier (for compiled
     languages), the full build command runs.  Errors are attributed to
     specific files and only those files enter fix cycles.

  4. **Test phase** — after all tiers pass the build checkpoint, test
     generation runs for all files.

  5. **Global DAG** — security scan, deploy, docs run last.

This executor is used by both ``RunPipeline`` and ``EnhancePipeline``.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from typing import Any, TYPE_CHECKING

from core.checkpoint import BuildCheckpoint, CheckpointCycleResult, CheckpointResult
from core.error_attributor import CompilerErrorAttributor, extract_error_lines
from core.event_bus import AgentEvent, BusEventType
from core.models import AgentContext, Task, TaskResult, TaskType
from core.state_machine import EventType, FilePhase, LifecycleEngine
from core.stub_generator import StubGenerator
from core.tier_scheduler import Tier, TierScheduler

if TYPE_CHECKING:
    from config.settings import Settings
    from core.agent_manager import AgentManager
    from core.pipeline_definition import (
        IntegrationCheckpointDef,
        PipelineDefinition,
        SecurityCheckpointDef,
    )
    from core.event_bus import EventBus
    from core.language import LanguageProfile
    from core.pipeline_definition import CheckpointDef, PipelineDefinition
    from core.task_engine import TaskGraph

logger = logging.getLogger(__name__)


class PipelineExecutor:
    """Unified tier-aware executor with repo-level build checkpoints.

    Orchestrates file generation/modification through tiers, running build
    checkpoints between tiers for compiled languages, then running tests
    and global tasks.
    """

    def __init__(
        self,
        agent_manager: AgentManager,
        settings: Settings,
        lang_profile: LanguageProfile,
        *,
        event_bus: EventBus | None = None,
    ) -> None:
        self._am = agent_manager
        self._settings = settings
        self._lang = lang_profile
        self._compiled = bool(lang_profile.build_command)
        self._event_bus = event_bus
        self._tier_scheduler = TierScheduler()

        # Files queued for re-verification at the next checkpoint because
        # an upstream dependency was modified after they reached a terminal
        # or post-review phase.
        # NOTE: asyncio is single-threaded — no true concurrent mutation of
        # this set occurs between handlers.  The TOCTOU window (phase check
        # then set.add) is benign: the worst case is a harmless extra
        # re-verification of a file whose phase changed after the check.
        # We document this explicitly and atomically snapshot+clear in
        # _drain_reverify_queue() to prevent duplicate processing.
        self._reverify_queue: set[str] = set()
        self._reverify_lock = asyncio.Lock()

    def _wire_event_bus(self, engine: LifecycleEngine) -> None:
        """Register event bus subscribers for cross-file coordination.

        These handlers enable reactive behaviour that was previously missing:
          - FILE_WRITTEN: when a file is modified during a fix cycle, queue
            its dependents for re-verification at the next checkpoint.
          - BUILD_FAILED: log attributed errors for observability.
        """
        if not self._event_bus:
            return

        dep_store = self._am._dep_store
        if not dep_store:
            return

        async def on_file_changed(event: AgentEvent) -> None:
            """When a file is written, mark its dependents for re-check."""
            if not event.file_path:
                return
            try:
                impact = dep_store.get_impact_analysis(event.file_path)
                dependents = impact.get("direct_dependents", [])
                to_queue: list[str] = []
                for dep_path in dependents:
                    if engine.has_file(dep_path):
                        lc = engine.get_lifecycle(dep_path)
                        # Only queue files that have already moved past review —
                        # files still in early phases will naturally pick up changes.
                        if lc.phase in (FilePhase.BUILDING, FilePhase.TESTING,
                                        FilePhase.PASSED, FilePhase.DEGRADED):
                            to_queue.append(dep_path)
                if to_queue:
                    async with self._reverify_lock:
                        self._reverify_queue.update(to_queue)
                    for dep_path in to_queue:
                        logger.debug(
                            "Queued %s for re-verification (upstream %s changed)",
                            dep_path, event.file_path,
                        )
            except Exception:
                logger.debug(
                    "Could not compute dependents for %s", event.file_path,
                )

        async def on_build_failed(event: AgentEvent) -> None:
            """Log build failure details for observability."""
            affected = event.data.get("affected_files", [])
            tier = event.data.get("tier", "?")
            attempt = event.data.get("attempt", "?")
            if affected:
                logger.info(
                    "[EventBus] Build failed (tier=%s, attempt=%s): %d files affected: %s",
                    tier, attempt, len(affected), affected,
                )

        self._event_bus.subscribe_critical(BusEventType.FILE_WRITTEN, on_file_changed)
        self._event_bus.subscribe(BusEventType.TASK_FAILED, on_build_failed)

    async def execute(
        self,
        engine: LifecycleEngine,
        global_graph: TaskGraph,
        *,
        pipeline_def: PipelineDefinition | None = None,
        tiers: list[Tier] | None = None,
    ) -> dict[str, Any]:
        """Execute the full pipeline — phases are driven by ``pipeline_def``.

        Each ``Phase`` in the definition is dispatched based on the
        ``task_type`` of its first ``FileTaskDef``:

          - ``GENERATE_FILE`` / ``MODIFY_FILE`` → tiered generation with an
            optional per-tier build checkpoint (``phase.checkpoint``).
          - ``GENERATE_TEST`` → flat test generation followed by an optional
            repo-level test checkpoint.

        ``Phase.skip_for_interpreted`` causes a phase to be skipped when the
        language has no build step (Python, Ruby, etc.).

        If ``pipeline_def`` is ``None`` or has no phases, a default two-phase
        structure (generation + test) is synthesised so execution is always
        driven by a uniform phase list.

        Args:
            engine: The lifecycle engine managing per-file state machines.
            global_graph: The global task DAG (security, deploy, docs).
            pipeline_def: Pipeline definition controlling phase order,
                checkpoint config, and skip semantics for each phase.
            tiers: Pre-computed dependency tiers.  If None, all files run as
                   a single tier (backward-compatible behavior).

        Returns:
            Execution result dict compatible with the existing pipeline format.
        """
        from core.pipeline_definition import Phase as _Phase, FileTaskDef as _FileTaskDef

        start_time = time.monotonic()

        # Wire up event bus subscribers for cross-file coordination.
        self._wire_event_bus(engine)

        # If no tiers provided, treat all files as a single tier (backward compat)
        if tiers is None:
            all_files = [p for p, _ in engine.get_actionable_files()]
            all_files += [
                path for path in engine._lifecycles
                if path not in all_files
            ]
            tiers = [Tier(index=0, files=sorted(all_files))]

        total_checkpoint_fixes = 0
        checkpoint_results: list[CheckpointCycleResult] = []

        # Stub generator for forward references (compiled languages only).
        stub_gen = StubGenerator(self._lang.name, self._am.repo.workspace)
        active_stubs: list[str] = []

        # Build blueprint map for richer stubs.
        bp_map: dict[str, object] = {}
        for fb in self._am.blueprint.file_blueprints:
            bp_map[fb.path] = fb

        # ── Resolve phases ──────────────────────────────────────────────────
        # Use the phases from pipeline_def when available; otherwise synthesise
        # a default two-phase structure so the loop below is always uniform.
        phases = pipeline_def.phases if (pipeline_def and pipeline_def.phases) else []
        if not phases:
            first_ck = None
            if pipeline_def:
                first_ck = next(
                    (ph.checkpoint for ph in pipeline_def.phases if ph.checkpoint),
                    None,
                )
            phases = [
                _Phase(
                    name="code_generation",
                    file_tasks=[_FileTaskDef(task_type=TaskType.GENERATE_FILE)],
                    checkpoint=first_ck,
                ),
                _Phase(
                    name="testing",
                    file_tasks=[_FileTaskDef(task_type=TaskType.GENERATE_TEST)],
                ),
            ]

        # Task types that identify a generation phase (processed tier-by-tier).
        _GEN_TYPES = {TaskType.GENERATE_FILE, TaskType.MODIFY_FILE}

        # ── Phase loop ──────────────────────────────────────────────────────
        tier_hard_blocked = False
        skip_agents = self._settings.skip_agents

        for phase in phases:
            if tier_hard_blocked:
                logger.error(
                    "=== Phase '%s' skipped — upstream tier checkpoint failed ===",
                    phase.name,
                )
                continue

            if phase.skip_for_interpreted and not self._compiled:
                logger.info(
                    "=== Phase '%s' skipped (interpreted language) ===", phase.name,
                )
                continue

            first_tt = phase.file_tasks[0].task_type if phase.file_tasks else None

            # --skip-tester: skip testing phases entirely.
            if "tester" in skip_agents and first_tt == TaskType.GENERATE_TEST:
                logger.info(
                    "=== Phase '%s' skipped (--skip-tester) ===", phase.name,
                )
                # Move testable files straight to PASSED so they don't end
                # up stuck in TESTING/BUILDING. Respect the state machine by
                # transitioning BUILDING → TESTING → PASSED.
                for path in list(engine._lifecycles.keys()):
                    lc = engine.get_lifecycle(path)
                    if lc.phase == FilePhase.BUILDING:
                        engine.process_event(path, EventType.BUILD_PASSED)
                        engine.process_event(path, EventType.TEST_PASSED)
                    elif lc.phase == FilePhase.TESTING:
                        engine.process_event(path, EventType.TEST_PASSED)
                continue
            # Checkpoints are only meaningful for compiled languages.
            ck_def = phase.checkpoint if self._compiled else None

            if first_tt in _GEN_TYPES or first_tt is None:
                # ── Tier-based generation + per-tier checkpoints ──────────────
                logger.info("=== Phase: %s ===", phase.name)

                for tier_idx, tier in enumerate(tiers):
                    logger.info(
                        "=== Tier %d: %d files ===", tier.index, len(tier),
                    )

                    # Drop stubs for files about to be generated for real.
                    stubs_to_remove = [s for s in active_stubs if s in set(tier.files)]
                    if stubs_to_remove:
                        stub_gen.cleanup_stubs(stubs_to_remove)
                        active_stubs = [
                            s for s in active_stubs if s not in stubs_to_remove
                        ]

                    # Run per-file lifecycles (Generate/Modify → Review → Fix).
                    await self._run_tier_lifecycles(engine, tier)

                    tier_failed = [
                        f for f in tier.files
                        if engine.get_lifecycle(f).phase == FilePhase.FAILED
                    ]
                    if tier_failed:
                        logger.warning(
                            "Tier %d: %d files failed lifecycle: %s",
                            tier.index, len(tier_failed), tier_failed,
                        )
                        # Cascade failures to later-tier dependents immediately
                        # so they don't hang waiting for a FAILED dependency.
                        all_later = {
                            p for ft in tiers[tier_idx + 1:] for p in ft.files
                        }
                        cascaded = engine.cascade_failures(scope=all_later)
                        if cascaded:
                            logger.warning(
                                "Tier %d failures cascade-failed %d downstream file(s): %s",
                                tier.index, len(cascaded), cascaded,
                            )

                    # Persist lifecycle state after each tier for resume support.
                    self._save_lifecycle_state(engine)

                    if ck_def:
                        # Forward-reference stubs for later-tier files so this
                        # tier's checkpoint can resolve their types.
                        later_files = [
                            p for ft in tiers[tier_idx + 1:] for p in ft.files
                        ]
                        if later_files:
                            new_stubs = stub_gen.generate_stubs(later_files, bp_map)
                            active_stubs.extend(new_stubs)

                        # Flush in-memory stubs to disk just before the build.
                        stub_gen.flush()

                        cycle_result = await self._run_checkpoint(
                            engine, ck_def, tier,
                        )
                        checkpoint_results.append(cycle_result)
                        total_checkpoint_fixes += len(cycle_result.files_fixed)

                        # ── Hard gate: failed checkpoint blocks downstream tiers ─
                        if not cycle_result.passed:
                            # Mark the current tier's BUILDING files as FAILED.
                            # (_run_checkpoint does this internally, but if it
                            # was overridden or returned early, ensure it here.)
                            for path in tier.files:
                                lc = engine.get_lifecycle(path)
                                if lc.phase == FilePhase.BUILDING:
                                    engine.process_event(
                                        path, EventType.RETRIES_EXHAUSTED,
                                    )
                            logger.error(
                                "Tier %d checkpoint failed permanently — "
                                "blocking %d downstream tier(s) from starting",
                                tier.index, len(tiers) - tier_idx - 1,
                            )
                            for blocked_tier in tiers[tier_idx + 1:]:
                                for path in blocked_tier.files:
                                    if not engine.get_lifecycle(path).is_terminal:
                                        engine.process_event(
                                            path, EventType.RETRIES_EXHAUSTED,
                                        )
                            if active_stubs:
                                stub_gen.cleanup_stubs(active_stubs)
                                active_stubs = []
                            tier_hard_blocked = True
                            break

                        # Clean up stubs after a passed checkpoint.
                        if active_stubs:
                            stub_gen.cleanup_stubs(active_stubs)
                            active_stubs = []

            else:
                # ── Test generation phase ─────────────────────────────────────
                logger.info("=== Phase: %s ===", phase.name)
                await self._run_test_phase(engine)

                if ck_def and self._lang.test_command:
                    test_ck = await self._run_test_checkpoint(ck_def.max_retries)
                    if test_ck:
                        checkpoint_results.append(test_ck)
                        total_checkpoint_fixes += len(test_ck.files_fixed)

        # ── Post-generation: Security + Integration (Track A) in parallel
        #    with Advisory tasks (Track B: deploy, docs, arch review) ────────

        security_result: dict[str, Any] = {}
        integration_result: dict[str, Any] = {}

        # Track A: Security checkpoint → Integration checkpoint (sequential)
        async def _track_a() -> None:
            nonlocal security_result, integration_result
            sec_def = pipeline_def.security_checkpoint if pipeline_def else None
            int_def = pipeline_def.integration_checkpoint if pipeline_def else None

            if sec_def and "security" not in skip_agents:
                security_result = await self._run_security_checkpoint(sec_def)
            elif "security" in skip_agents:
                logger.info("=== Security Checkpoint skipped (--skip-security) ===")

            if int_def and "integration" not in skip_agents:
                integration_result = await self._run_integration_checkpoint(int_def)
            elif "integration" in skip_agents:
                logger.info("=== Integration Checkpoint skipped (--skip-integration) ===")


        # Track B: Advisory global tasks (fire-and-forget, parallel)
        async def _track_b() -> None:
            logger.info("=== Advisory Tasks (parallel) ===")
            sentinel = next(
                (t for t in global_graph.tasks.values()
                 if t.metadata.get("sentinel")),
                None,
            )
            if sentinel:
                global_graph.mark_completed(sentinel.task_id)
            await self._am.execute_graph(global_graph)

        # Run both tracks concurrently
        logger.info("=== Post-Generation Phase: Track A (Security → Integration) "
                     "|| Track B (Advisory) ===")
        await asyncio.gather(_track_a(), _track_b())

        # ── Results ─────────────────────────────────────────────────────────
        elapsed = time.monotonic() - start_time

        lifecycle_summary = engine.get_results_summary()
        global_stats = global_graph.get_stats()

        logger.info(
            "Pipeline execution complete in %.1fs. "
            "Files: %d passed, %d degraded, %d failed. Checkpoint fixes: %d. "
            "Global tasks: %s",
            elapsed,
            lifecycle_summary["passed"],
            lifecycle_summary.get("degraded", 0),
            lifecycle_summary["failed"],
            total_checkpoint_fixes,
            global_stats,
        )

        if lifecycle_summary.get("degraded", 0):
            logger.warning(
                "%d file(s) completed with quality warnings (DEGRADED): %s",
                lifecycle_summary["degraded"],
                lifecycle_summary.get("degraded_files", []),
            )

        # Drain and report any critical event bus handler failures so callers
        # know whether re-verification or coordination work was silently lost.
        bus_failures: list[dict[str, str]] = []
        if self._event_bus and self._event_bus.has_failures():
            for failure in self._event_bus.clear_failures():
                bus_failures.append({
                    "handler": failure.handler_name,
                    "event": failure.event_type,
                    "error": failure.error,
                })
            logger.error(
                "%d critical event bus handler failure(s) recorded — "
                "re-verification or coordination may be incomplete: %s",
                len(bus_failures), bus_failures,
            )

        return {
            "stats": {
                **global_stats,
                "lifecycle_passed": lifecycle_summary["passed"],
                "lifecycle_degraded": lifecycle_summary.get("degraded", 0),
                "lifecycle_failed": lifecycle_summary["failed"],
                "lifecycle_total_fixes": lifecycle_summary["total_fix_cycles"],
                "lifecycle_tests_degraded": lifecycle_summary["tests_degraded"],
                "checkpoint_fixes": total_checkpoint_fixes,
            },
            "metrics": self._am._metrics,
            "elapsed_seconds": elapsed,
            "lifecycle_summary": lifecycle_summary,
            "bus_handler_failures": bus_failures,
            "checkpoint_results": [
                {
                    "passed": cr.passed,
                    "attempts": cr.total_attempts,
                    "files_fixed": cr.files_fixed,
                }
                for cr in checkpoint_results
            ],
            "security_checkpoint": security_result,
            "integration_checkpoint": integration_result,
        }

    # ── Tier lifecycle execution ─────────────────────────────────────────

    def _save_lifecycle_state(self, engine: LifecycleEngine) -> None:
        """Persist lifecycle state to disk for resume support.

        Called after each tier completes so a ``--resume`` run can skip
        files that already PASSED.  Non-critical — failures are logged
        but do not interrupt the pipeline.
        """
        try:
            state_path = str(self._am.repo.workspace / ".pipeline_state.json")
            engine.save_state(state_path)
        except Exception:
            logger.debug("Failed to save lifecycle state (non-critical)", exc_info=True)

    async def _drain_reverify_queue(self, tier_files: set[str]) -> set[str]:
        """Atomically drain and return the subset of reverify-queue entries
        that belong to *tier_files*.

        The lock makes the read-then-clear atomic so the ``on_file_changed``
        event handler cannot interleave an insertion between the snapshot
        and the removal, preventing duplicate processing of the same file.
        """
        async with self._reverify_lock:
            drained = self._reverify_queue & tier_files
            self._reverify_queue -= drained
        return drained

    async def _run_tier_lifecycles(
        self,
        engine: LifecycleEngine,
        tier: Tier,
    ) -> None:
        """Run per-file lifecycles for all files in a tier.

        Files go through: PENDING → GENERATING → REVIEWING ↔ FIXING
        The BUILDING phase is skipped at the per-file level — it's handled
        by the repo-level checkpoint after the tier completes.
        """
        semaphore = asyncio.Semaphore(self._settings.max_concurrent_agents)
        # Per-file timeout: each individual file gets a fixed timeout for any
        # single lifecycle phase (generate, review, fix).  This should NOT
        # scale with tier size — a stuck LLM call for one file shouldn't be
        # allowed to hang for 40+ minutes just because the tier is large.
        base_phase_timeout = float(self._settings.phase_timeout_seconds)
        per_file_timeout = min(base_phase_timeout, 600.0)  # cap at 10 min per file
        # Total tier timeout for staleness detection scales with tier size.
        phase_timeout = max(base_phase_timeout, 120.0 * len(tier.files))

        in_flight: set[str] = set()
        running_tasks: set[asyncio.Task[None]] = set()

        # Only process files in this tier
        tier_files = set(tier.files)

        # Staleness detection: if no task completes within this window and
        # non-terminal files remain stuck PENDING (waiting on deps that are
        # themselves stuck in REVIEWING/FIXING and will never reach TESTING),
        # we diagnose and force-fail the stuck files rather than hanging forever.
        # Use 2× phase_timeout as the stale threshold — ample time for any
        # in-flight task to resolve before we declare a deadlock.
        _stale_timeout = phase_timeout * 2
        _last_progress = time.monotonic()

        async def run_file_phase(path: str, phase: FilePhase) -> None:
            async with semaphore:
                try:
                    await asyncio.wait_for(
                        self._am._execute_lifecycle_phase(engine, path, phase),
                        timeout=per_file_timeout,
                    )
                except asyncio.TimeoutError:
                    logger.error(
                        "[%s] %s timed out after %ds — marking FAILED",
                        path, phase.value, int(per_file_timeout),
                    )
                    try:
                        engine.process_event(path, EventType.RETRIES_EXHAUSTED)
                    except Exception as exc:
                        logger.warning(
                            "[%s] Failed to transition to RETRIES_EXHAUSTED: %s "
                            "— forcing FAILED directly",
                            path, exc,
                        )
                        try:
                            engine.get_lifecycle(path).phase = FilePhase.FAILED
                        except Exception:
                            logger.error(
                                "[%s] Could not force FAILED state — file phase is UNKNOWN",
                                path,
                            )
                    self._am._metrics["tasks_failed"] += 1
                finally:
                    in_flight.discard(path)

        # Phases that the tier loop must NOT dispatch — they're handled later:
        #   BUILDING  → handled by _run_checkpoint after the tier completes
        #   TESTING   → handled by _run_test_phase after all tiers + checkpoints
        deferred_phases = {FilePhase.BUILDING, FilePhase.TESTING}

        while True:
            # Cascade failures: if any tier file FAILED (timeout, retries
            # exhausted), immediately fail all PENDING dependents rather
            # than waiting for the staleness timeout.
            cascaded = engine.cascade_failures(scope=tier_files)
            if cascaded:
                _last_progress = time.monotonic()
                logger.warning(
                    "[Tier %d] Cascade-failed %d dependent file(s): %s",
                    tier.index, len(cascaded), cascaded,
                )

            actionable = [
                (path, phase) for path, phase in engine.get_actionable_files()
                if path in tier_files and path not in in_flight
                and phase not in deferred_phases
            ]

            for path, phase in actionable:
                in_flight.add(path)
                _last_progress = time.monotonic()
                t = asyncio.create_task(
                    run_file_phase(path, phase), name=f"tier{tier.index}:{path}",
                )
                running_tasks.add(t)
                t.add_done_callback(running_tasks.discard)

            if not running_tasks:
                # No tasks running and nothing actionable — cascade failures
                # one more time, then check for true deadlocks.
                engine.cascade_failures(scope=tier_files)

                stuck = [
                    path for path in tier_files
                    if not engine.get_lifecycle(path).is_terminal
                    and engine.get_lifecycle(path).phase == FilePhase.PENDING
                    and path not in in_flight
                ]
                if stuck:
                    logger.error(
                        "[Tier %d] Dependency deadlock: %d file(s) are PENDING "
                        "with no actionable work and no in-flight tasks. "
                        "Dependency chain may be waiting on files stuck in "
                        "REVIEWING/FIXING. Failing stuck files: %s",
                        tier.index, len(stuck), stuck,
                    )
                    for path in stuck:
                        try:
                            engine.process_event(path, EventType.RETRIES_EXHAUSTED)
                        except Exception:
                            engine.get_lifecycle(path).phase = FilePhase.FAILED
                break

            # Wait for any task to complete, bounded by the stale timeout.
            # If the timeout fires with no new actionable files, the tier may
            # be deadlocked (e.g. a dep stuck in REVIEWING that will never
            # reach TESTING in checkpoint mode).
            wait_timeout = max(1.0, _stale_timeout - (time.monotonic() - _last_progress))
            try:
                done, _ = await asyncio.wait(
                    running_tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=wait_timeout,
                )
                if done:
                    _last_progress = time.monotonic()
            except Exception:
                pass  # asyncio.wait itself should not raise; be defensive

            # Staleness check: if no progress was made and non-terminal files
            # remain in this tier (PENDING or early phases), diagnose explicitly.
            elapsed_since_progress = time.monotonic() - _last_progress
            if elapsed_since_progress >= _stale_timeout:
                non_terminal = [
                    path for path in tier_files
                    if not engine.get_lifecycle(path).is_terminal
                    and path not in in_flight
                ]
                if non_terminal:
                    for path in non_terminal:
                        lc = engine.get_lifecycle(path)
                        dep_phases = {
                            d: engine.get_lifecycle(d).phase.value
                            for d in engine._deps.get(path, [])
                            if engine.has_file(d)
                        }
                        logger.error(
                            "[Tier %d] Staleness timeout (%.0fs no progress): "
                            "%s is stuck in %s. Dependency phases: %s",
                            tier.index, elapsed_since_progress,
                            path, lc.phase.value, dep_phases,
                        )
                        try:
                            engine.process_event(path, EventType.RETRIES_EXHAUSTED)
                        except Exception:
                            lc.phase = FilePhase.FAILED
                    break

        # Cancel any stragglers
        for t in list(running_tasks):
            t.cancel()
        if running_tasks:
            await asyncio.gather(*list(running_tasks), return_exceptions=True)

    # ── Build checkpoint ─────────────────────────────────────────────────

    async def _run_checkpoint(
        self,
        engine: LifecycleEngine,
        checkpoint_def: CheckpointDef,
        tier: Tier,
    ) -> CheckpointCycleResult:
        """Run a repo-level build checkpoint after a tier completes.

        On failure, dispatches fix tasks to affected files and retries
        the build up to ``max_retries`` times.
        """
        build_command = self._lang.build_command
        if not build_command:
            return CheckpointCycleResult(passed=True, total_attempts=0)

        # Collect all known files for attribution
        known_files = set(engine._lifecycles.keys())

        # Scale timeout with project size.  The base timeout (180s) is fine for
        # incremental rebuilds, but the first `mvn package` on a 50-file project
        # downloads all dependencies + compiles everything — easily 5-10 minutes.
        # If the process is killed early, we get 0 error output → 0 attributed
        # errors → the same death-spiral as the parse-failure bug.
        base_timeout = checkpoint_def.timeout
        scaled_timeout = max(base_timeout, 60 * len(tier.files))
        # First build (tier 0) gets extra time for dependency downloads.
        if tier.index == 0:
            scaled_timeout = max(scaled_timeout, 600)  # at least 10 min
        logger.debug(
            "[Checkpoint] Timeout: base=%ds, scaled=%ds (tier=%d, files=%d)",
            base_timeout, scaled_timeout, tier.index, len(tier.files),
        )

        checkpoint = BuildCheckpoint(
            build_command=build_command,
            terminal=self._am.build_terminal,
            attributor=CompilerErrorAttributor(),
            known_files=known_files,
            max_retries=checkpoint_def.max_retries,
            timeout=scaled_timeout,
            checkpoint_name=f"tier{tier.index}_{checkpoint_def.name}",
        )

        history: list[CheckpointResult] = []
        files_fixed: list[str] = []
        # Track how many fix attempts each file has had across retries.
        # A file that fails repeatedly shouldn't consume unlimited retries.
        fix_attempt_counts: dict[str, int] = {}
        max_fixes_per_file = 2  # cap per-file fix dispatches
        # Track the error fingerprint seen on the previous fix attempt for each
        # file.  If the same errors recur unchanged after a fix, the fix made
        # no progress and further attempts would be futile.
        file_error_hashes: dict[str, str] = {}

        # Atomically drain any files queued for re-verification by event bus
        # handlers.  Using the lock ensures the handler cannot add new entries
        # to the set while we are reading it, preventing the TOCTOU window
        # where a file could be queued after the snapshot but before the clear.
        reverify = await self._drain_reverify_queue(set(tier.files))
        if reverify:
            logger.info(
                "[Checkpoint] Including %d re-verify files from event bus: %s",
                len(reverify), sorted(reverify),
            )

        for attempt in range(1, checkpoint_def.max_retries + 1):
            result = await checkpoint.run_once(attempt=attempt)
            history.append(result)

            if result.passed:
                # Publish build passed event
                if self._event_bus:
                    await self._event_bus.publish(AgentEvent(
                        type=BusEventType.TASK_COMPLETED,
                        task_type=TaskType.VERIFY_BUILD.value,
                        file_path="*",
                        agent_name="build_checkpoint",
                        data={"tier": tier.index, "attempt": attempt},
                    ))

                # Transition files from BUILDING to TESTING (or auto-skip)
                for path in tier.files:
                    lc = engine.get_lifecycle(path)
                    if lc.phase == FilePhase.BUILDING:
                        engine.process_event(path, EventType.BUILD_PASSED)

                return CheckpointCycleResult(
                    passed=True,
                    total_attempts=attempt,
                    history=history,
                    files_fixed=files_fixed,
                )

            # Build failed — fix affected files
            affected = result.affected_files
            unattributed_fallback = False
            if not affected:
                # Errors couldn't be attributed to specific files — pass
                # the raw build output to all tier files so the fix agent
                # has something actionable to work with (instead of empty
                # error context which produces identical-content fixes).
                logger.warning(
                    "[Checkpoint] Build failed but no errors could be attributed "
                    "to specific files. %d unattributed error lines. "
                    "Raw output (first 500 chars): %s",
                    len(result.attribution.unattributed_errors) if result.attribution else 0,
                    result.raw_output[:500],
                )
                # Try to fix all tier files as a fallback
                affected = [
                    f for f in tier.files
                    if not engine.get_lifecycle(f).is_terminal
                ]
                unattributed_fallback = True

            # Dispatch parallel fix tasks for affected files
            fix_tasks: list[asyncio.Task[None]] = []
            semaphore = asyncio.Semaphore(self._settings.max_concurrent_agents)

            for file_path in affected:
                lc = engine.get_lifecycle(file_path)
                if lc.is_terminal:
                    continue

                # Skip files that have already been fixed too many times
                prior_fixes = fix_attempt_counts.get(file_path, 0)
                if prior_fixes >= max_fixes_per_file:
                    logger.warning(
                        "[Checkpoint] Skipping %s — already fixed %d times",
                        file_path, prior_fixes,
                    )
                    continue

                fix_context = checkpoint.get_fix_context_for_file(file_path, result)

                # Skip files that have no attributed errors (and this isn't
                # an unattributed fallback).  Without this check, every tier
                # file gets a fix dispatch with empty build_errors, producing
                # identical-content "fixes" that waste LLM calls.
                if not unattributed_fallback and not fix_context.get("build_errors"):
                    logger.debug(
                        "[Checkpoint] Skipping %s — no errors attributed to this file",
                        file_path,
                    )
                    continue

                # When no errors were attributed to specific files, extract
                # just the error lines from the build output instead of
                # sending the entire log (which can be 50K+ of noise).
                if unattributed_fallback and not fix_context.get("build_errors"):
                    # raw_output is already filtered by extract_error_lines
                    # in checkpoint.run_once(), but add unattributed lines too
                    error_summary = result.raw_output or ""
                    if result.attribution and result.attribution.unattributed_errors:
                        extra = "\n".join(result.attribution.unattributed_errors[:20])
                        if extra not in error_summary:
                            error_summary += "\n" + extra
                    fix_context["build_errors"] = error_summary[:4000]
                    fix_context["fix_trigger"] = "build_unattributed"

                # Skip if this file's errors haven't changed since the last fix
                # attempt — the previous fix made no progress, so another
                # identical fix would almost certainly fail too.
                errors_text = str(fix_context.get("build_errors", ""))
                error_hash = hashlib.md5(errors_text.encode()).hexdigest()[:8]
                if file_error_hashes.get(file_path) == error_hash:
                    logger.warning(
                        "[Checkpoint] Skipping %s — errors unchanged after fix "
                        "(no progress, hash=%s)",
                        file_path, error_hash,
                    )
                    continue
                file_error_hashes[file_path] = error_hash

                fix_context["fix_attempt"] = prior_fixes + 1
                fix_context["max_fix_attempts"] = max_fixes_per_file

                async def fix_file(fp: str, ctx: dict[str, Any]) -> None:
                    async with semaphore:
                        await self._dispatch_checkpoint_fix(engine, fp, ctx)

                fix_tasks.append(
                    asyncio.create_task(fix_file(file_path, fix_context))
                )
                fix_attempt_counts[file_path] = prior_fixes + 1
                if file_path not in files_fixed:
                    files_fixed.append(file_path)

            if fix_tasks:
                await asyncio.gather(*fix_tasks, return_exceptions=True)

            if self._event_bus:
                await self._event_bus.publish(AgentEvent(
                    type=BusEventType.TASK_FAILED,
                    task_type=TaskType.VERIFY_BUILD.value,
                    file_path="*",
                    agent_name="build_checkpoint",
                    data={
                        "tier": tier.index,
                        "attempt": attempt,
                        "affected_files": affected,
                    },
                ))

        # All retries exhausted — mark unresolved BUILDING files as FAILED.
        # Returning passed=False signals execute() to block all downstream tiers.
        logger.error(
            "[Checkpoint] Build checkpoint exhausted %d retries — "
            "marking unresolved BUILDING file(s) as FAILED",
            checkpoint_def.max_retries,
        )

        for path in tier.files:
            lc = engine.get_lifecycle(path)
            if lc.phase == FilePhase.BUILDING:
                engine.process_event(path, EventType.RETRIES_EXHAUSTED)

        return CheckpointCycleResult(
            passed=False,
            total_attempts=checkpoint_def.max_retries,
            history=history,
            files_fixed=files_fixed,
        )

    async def _dispatch_fix(
        self,
        file_path: str,
        fix_context: dict[str, Any],
        *,
        label: str = "Checkpoint",
    ) -> bool:
        """Dispatch a single fix task — shared by build and test checkpoints.

        Builds context, creates a FIX_CODE agent, executes it, and updates
        metrics + embeddings.  Returns True if the fix succeeded.

        Args:
            file_path: The workspace-relative path of the file to fix.
            fix_context: Metadata dict passed as ``task.metadata`` to the agent.
            label: Logging prefix for checkpoint identification.
        """
        from core.context_builder import ContextBuilder
        from core.observability import record_agent_start, record_agent_end

        task = Task(
            task_id=0,
            task_type=TaskType.FIX_CODE,
            file=file_path,
            description=f"Fix errors in {file_path}",
            metadata=fix_context,
        )

        try:
            context_builder = ContextBuilder(
                workspace_dir=self._am.repo.workspace,
                blueprint=self._am.blueprint,
                repo_index=self._am.repo.get_repo_index(),
                dep_store=self._am._dep_store,
                embedding_store=self._am._embedding_store,
            )
            context = await asyncio.to_thread(context_builder.build, task)
        except Exception:
            logger.exception("[%s] Context build failed for %s", label, file_path)
            self._am._metrics["tasks_failed"] += 1
            return False

        try:
            agent = self._am._create_agent(TaskType.FIX_CODE)
            record_agent_start()
            try:
                result = await agent.execute(context)
            finally:
                record_agent_end()

            agent_name = agent.role.value
            if agent_name not in self._am._metrics["agent_metrics"]:
                self._am._metrics["agent_metrics"][agent_name] = []
            metrics_list = self._am._metrics["agent_metrics"][agent_name]
            metrics_list.append(agent.get_metrics())
            if len(metrics_list) > 100:
                metrics_list.pop(0)

            if result.success:
                self._am._metrics["tasks_completed"] += 1
                logger.info("[%s] Fixed errors in %s", label, file_path)
                if self._am._embedding_store and result.files_modified:
                    for fp in result.files_modified:
                        try:
                            content = self._am.repo.read_file(fp)
                            self._am._embedding_store.index_file(fp, content)
                        except Exception as exc:
                            logger.warning(
                                "[%s] Embedding update failed for %s: %s "
                                "— semantic index may be stale",
                                label, fp, exc,
                            )
                return True
            else:
                self._am._metrics["tasks_failed"] += 1
                logger.warning(
                    "[%s] Fix attempt failed for %s: %s",
                    label, file_path, result.errors,
                )
                return False
        except Exception:
            logger.exception("[%s] Error fixing %s", label, file_path)
            self._am._metrics["tasks_failed"] += 1
            return False

    async def _dispatch_checkpoint_fix(
        self,
        engine: LifecycleEngine,
        file_path: str,
        fix_context: dict[str, Any],
    ) -> None:
        """Dispatch a single fix task for a checkpoint build error.

        Thin wrapper around ``_dispatch_fix`` — preserved for backward
        compatibility with callers that pass the ``engine`` argument.
        """
        await self._dispatch_fix(file_path, fix_context, label="Checkpoint")

    # ── Test phase ───────────────────────────────────────────────────────

    async def _run_test_phase(self, engine: LifecycleEngine) -> None:
        """Run test generation for all files that reached TESTING phase."""
        semaphore = asyncio.Semaphore(self._settings.max_concurrent_agents)
        # Scale with the number of files being tested
        testable = engine.get_files_in_phase(FilePhase.TESTING)
        n_test_files = len(testable) if testable else 1
        base_phase_timeout = float(self._settings.phase_timeout_seconds)
        phase_timeout = max(base_phase_timeout, 120.0 * n_test_files)

        in_flight: set[str] = set()
        running_tasks: set[asyncio.Task[None]] = set()

        async def run_test(path: str, phase: FilePhase) -> None:
            async with semaphore:
                try:
                    await asyncio.wait_for(
                        self._am._execute_lifecycle_phase(engine, path, phase),
                        timeout=phase_timeout,
                    )
                except asyncio.TimeoutError:
                    logger.error(
                        "[%s] testing timed out after %ds", path, int(phase_timeout),
                    )
                    try:
                        engine.process_event(path, EventType.RETRIES_EXHAUSTED)
                    except Exception as exc:
                        logger.warning(
                            "[%s] Failed to transition to RETRIES_EXHAUSTED: %s "
                            "— forcing FAILED directly",
                            path, exc,
                        )
                        try:
                            engine.get_lifecycle(path).phase = FilePhase.FAILED
                        except Exception:
                            logger.error(
                                "[%s] Could not force FAILED state — file phase is UNKNOWN",
                                path,
                            )
                finally:
                    in_flight.discard(path)

        while True:
            # Get files in TESTING or FIXING (test fix) phase
            actionable = [
                (path, phase) for path, phase in engine.get_actionable_files()
                if path not in in_flight
                and phase in (FilePhase.TESTING, FilePhase.FIXING)
            ]

            # Files stuck in BUILDING at this point should be marked FAILED,
            # not auto-passed — they never went through build verification.
            for path in list(engine._lifecycles.keys()):
                lc = engine.get_lifecycle(path)
                if lc.phase == FilePhase.BUILDING and path not in in_flight:
                    logger.warning(
                        "[%s] still in BUILDING at test phase — marking as failed "
                        "(missed build checkpoint)",
                        path,
                    )
                    engine.process_event(path, EventType.RETRIES_EXHAUSTED)

            for path, phase in actionable:
                if path in in_flight:
                    continue
                in_flight.add(path)
                t = asyncio.create_task(
                    run_test(path, phase), name=f"test:{path}",
                )
                running_tasks.add(t)
                t.add_done_callback(running_tasks.discard)

            if not running_tasks:
                break

            await asyncio.wait(running_tasks, return_when=asyncio.FIRST_COMPLETED)

        for t in list(running_tasks):
            t.cancel()
        if running_tasks:
            await asyncio.gather(*list(running_tasks), return_exceptions=True)

    # ── Repo-level test checkpoint ───────────────────────────────────────

    async def _run_test_checkpoint(
        self,
        max_retries: int = 2,
    ) -> CheckpointCycleResult | None:
        """Run the language's test command against the entire workspace.

        After per-file test generation, this runs the full test suite
        (e.g., ``mvn test``, ``go test ./...``) to catch integration issues
        that per-file tests miss.

        Returns None if no test command is configured or no terminal available.
        """
        test_command = self._lang.test_command
        if not test_command:
            return None

        if not self._am.test_terminal:
            logger.warning("No test terminal — skipping repo-level test checkpoint")
            return None

        logger.info("=== Repo-Level Test Checkpoint ===")

        known_file_paths = {f.path for f in self._am.repo.get_repo_index().files}

        checkpoint = BuildCheckpoint(
            build_command=test_command,
            terminal=self._am.test_terminal,
            attributor=CompilerErrorAttributor(),
            known_files=known_file_paths,
            max_retries=max_retries,
            timeout=300,  # tests may take longer than builds
            checkpoint_name="test_suite",
        )

        history: list[CheckpointResult] = []
        files_fixed: list[str] = []

        for attempt in range(1, max_retries + 1):
            result = await checkpoint.run_once(attempt=attempt)
            history.append(result)

            if result.passed:
                logger.info(
                    "[TestCheckpoint] Full test suite passed (attempt %d)", attempt,
                )
                if self._event_bus:
                    await self._event_bus.publish(AgentEvent(
                        type=BusEventType.TEST_PASSED,
                        task_type="test_suite",
                        file_path="*",
                        agent_name="test_checkpoint",
                        data={"attempt": attempt},
                    ))
                return CheckpointCycleResult(
                    passed=True,
                    total_attempts=attempt,
                    history=history,
                    files_fixed=files_fixed,
                )

            # Test suite failed — attribute failures and dispatch fixes
            affected = result.affected_files
            if affected:
                logger.warning(
                    "[TestCheckpoint] Test suite failed (attempt %d): %d file errors",
                    attempt, len(affected),
                )
                fix_tasks_list: list[asyncio.Task[None]] = []
                sem = asyncio.Semaphore(self._settings.max_concurrent_agents)

                for file_path in affected:
                    fix_ctx = checkpoint.get_fix_context_for_file(file_path, result)
                    fix_ctx["fix_trigger"] = "test"
                    fix_ctx["test_errors"] = fix_ctx.pop("build_errors", "")

                    async def _fix_test_file(fp: str, ctx: dict[str, Any]) -> None:
                        async with sem:
                            await self._dispatch_fix(fp, ctx, label="TestCheckpoint")

                    fix_tasks_list.append(
                        asyncio.create_task(_fix_test_file(file_path, fix_ctx))
                    )
                    files_fixed.append(file_path)

                if fix_tasks_list:
                    await asyncio.gather(*fix_tasks_list, return_exceptions=True)
            else:
                logger.warning(
                    "[TestCheckpoint] Test suite failed, no file attribution (attempt %d)",
                    attempt,
                )

            if self._event_bus:
                await self._event_bus.publish(AgentEvent(
                    type=BusEventType.TEST_FAILED,
                    task_type="test_suite",
                    file_path="*",
                    agent_name="test_checkpoint",
                    data={"attempt": attempt, "affected_files": affected},
                ))

        logger.warning(
            "[TestCheckpoint] Test suite did not pass after %d retries", max_retries,
        )
        return CheckpointCycleResult(
            passed=False,
            total_attempts=max_retries,
            history=history,
            files_fixed=files_fixed,
        )

    # ── Security checkpoint ──────────────────────────────────────────────

    async def _run_security_checkpoint(
        self,
        sec_def: SecurityCheckpointDef,
    ) -> dict[str, Any]:
        """Run the security hardening checkpoint: scan → fix → rebuild → re-scan.

        Only critical and high severity vulnerabilities trigger fix cycles.
        Medium/low findings are reported but do not block.

        Returns a summary dict with pass/fail, cycle count, and files fixed.
        """
        logger.info("=== Security Hardening Checkpoint ===")

        files_fixed: list[str] = []
        fix_attempt_counts: dict[str, int] = {}
        total_vulns_fixed = 0

        for cycle in range(1, sec_def.max_cycles + 1):
            logger.info("[SecurityCheckpoint] Cycle %d/%d — running scan",
                        cycle, sec_def.max_cycles)

            # Run the security agent
            scan_result = await self._run_security_scan()
            if scan_result is None:
                logger.warning("[SecurityCheckpoint] Security scan returned no result")
                break

            # Extract critical/high vulnerabilities
            vulns = scan_result.metrics.get("vulnerabilities", [])
            critical_high = [
                v for v in vulns
                if v.get("severity") in ("critical", "high")
            ]

            if not critical_high:
                logger.info(
                    "[SecurityCheckpoint] No critical/high vulnerabilities — "
                    "security checkpoint PASSED (cycle %d)", cycle,
                )
                if self._event_bus:
                    await self._event_bus.publish(AgentEvent(
                        type=BusEventType.TASK_COMPLETED,
                        task_type=TaskType.SECURITY_SCAN.value,
                        file_path="*",
                        agent_name="security_checkpoint",
                        data={"cycle": cycle, "total_vulns": len(vulns)},
                    ))
                return {
                    "passed": True,
                    "cycles": cycle,
                    "files_fixed": files_fixed,
                    "remaining_critical_high": 0,
                    "total_vulnerabilities": len(vulns),
                }

            logger.warning(
                "[SecurityCheckpoint] %d critical/high vulnerabilities found — "
                "dispatching fixes",
                len(critical_high),
            )

            # Group vulnerabilities by file for targeted fixes
            vulns_by_file: dict[str, list[dict[str, Any]]] = {}
            for v in critical_high:
                fpath = v.get("file", "")
                if fpath:
                    vulns_by_file.setdefault(fpath, []).append(v)

            # Dispatch SECURITY_FIX tasks for affected files
            fix_tasks: list[asyncio.Task[None]] = []
            semaphore = asyncio.Semaphore(self._settings.max_concurrent_agents)

            for file_path, file_vulns in vulns_by_file.items():
                prior_fixes = fix_attempt_counts.get(file_path, 0)
                if prior_fixes >= sec_def.max_fixes_per_file:
                    logger.warning(
                        "[SecurityCheckpoint] Skipping %s — already fixed %d times",
                        file_path, prior_fixes,
                    )
                    continue

                # Build fix context with vulnerability details and remediation
                vuln_descriptions = "\n".join(
                    f"  [{v['severity'].upper()}] {v.get('type', 'unknown')} "
                    f"(line {v.get('line', '?')}): {v.get('description', '')}\n"
                    f"    Remediation: {v.get('remediation', 'N/A')}"
                    for v in file_vulns
                )
                fix_context: dict[str, Any] = {
                    "fix_trigger": "security",
                    "security_vulnerabilities": vuln_descriptions,
                    "vulnerability_count": len(file_vulns),
                    "fix_attempt": prior_fixes + 1,
                    "max_fix_attempts": sec_def.max_fixes_per_file,
                }

                async def _fix_security(fp: str, ctx: dict[str, Any]) -> None:
                    async with semaphore:
                        await self._dispatch_fix(fp, ctx, label="SecurityCheckpoint")

                fix_tasks.append(
                    asyncio.create_task(_fix_security(file_path, fix_context))
                )
                fix_attempt_counts[file_path] = prior_fixes + 1
                if file_path not in files_fixed:
                    files_fixed.append(file_path)

            if fix_tasks:
                await asyncio.gather(*fix_tasks, return_exceptions=True)
                total_vulns_fixed += len(fix_tasks)

                # Rebuild to verify security fixes compile
                if self._compiled and self._lang.build_command:
                    logger.info(
                        "[SecurityCheckpoint] Rebuilding to verify security fixes compile"
                    )
                    build_ck = BuildCheckpoint(
                        build_command=self._lang.build_command,
                        terminal=self._am.build_terminal,
                        attributor=CompilerErrorAttributor(),
                        known_files={f.path for f in self._am.repo.get_repo_index().files},
                        max_retries=1,
                        timeout=300,
                        checkpoint_name="security_rebuild",
                    )
                    rebuild_result = await build_ck.run_once(attempt=1)
                    if not rebuild_result.passed:
                        logger.warning(
                            "[SecurityCheckpoint] Security fixes broke the build — "
                            "dispatching build fixes"
                        )
                        for fp in rebuild_result.affected_files:
                            build_ctx = build_ck.get_fix_context_for_file(
                                fp, rebuild_result,
                            )
                            build_ctx["fix_trigger"] = "security_rebuild"
                            await self._dispatch_fix(
                                fp, build_ctx, label="SecurityRebuild",
                            )
            else:
                # All files hit their fix limit — no more progress possible
                logger.warning(
                    "[SecurityCheckpoint] No fixable files remaining — "
                    "accepting %d findings as known risk",
                    len(critical_high),
                )
                break

        # Cycles exhausted — report remaining findings
        logger.warning(
            "[SecurityCheckpoint] Security checkpoint exhausted %d cycles — "
            "%d files fixed, remaining findings accepted as known risk",
            sec_def.max_cycles, len(files_fixed),
        )
        return {
            "passed": False,
            "cycles": sec_def.max_cycles,
            "files_fixed": files_fixed,
            "remaining_critical_high": len(critical_high) if 'critical_high' in dir() else 0,
            "total_vulns_fixed": total_vulns_fixed,
        }

    async def _run_security_scan(self) -> TaskResult | None:
        """Execute the SecurityAgent and return its TaskResult."""
        from core.context_builder import ContextBuilder
        from core.observability import record_agent_start, record_agent_end

        task = Task(
            task_id=0,
            task_type=TaskType.SECURITY_SCAN,
            file="*",
            description="Security scan of entire codebase",
        )

        try:
            context_builder = ContextBuilder(
                workspace_dir=self._am.repo.workspace,
                blueprint=self._am.blueprint,
                repo_index=self._am.repo.get_repo_index(),
                dep_store=self._am._dep_store,
                embedding_store=self._am._embedding_store,
            )
            context = await asyncio.to_thread(context_builder.build, task)
        except Exception:
            logger.exception("[SecurityCheckpoint] Context build failed for scan")
            return None

        try:
            agent = self._am._create_agent(TaskType.SECURITY_SCAN)
            record_agent_start()
            try:
                result = await agent.execute(context)
            finally:
                record_agent_end()

            # Track metrics
            agent_name = agent.role.value
            if agent_name not in self._am._metrics["agent_metrics"]:
                self._am._metrics["agent_metrics"][agent_name] = []
            self._am._metrics["agent_metrics"][agent_name].append(agent.get_metrics())
            return result
        except Exception:
            logger.exception("[SecurityCheckpoint] Security scan agent failed")
            return None

    # ── Integration checkpoint ───────────────────────────────────────────

    async def _run_integration_checkpoint(
        self,
        int_def: IntegrationCheckpointDef,
    ) -> dict[str, Any]:
        """Run the integration test verification checkpoint.

        Generates integration tests, runs them, and on failure triages
        whether the test code or source code is at fault.  Dispatches
        targeted fixes accordingly, up to ``max_cycles`` iterations.

        Returns a summary dict with pass/fail, cycle count, and files fixed.
        """
        logger.info("=== Integration Test Checkpoint ===")

        files_fixed: list[str] = []
        test_file: str = ""

        for cycle in range(1, int_def.max_cycles + 1):
            logger.info(
                "[IntegrationCheckpoint] Cycle %d/%d — generating and running tests",
                cycle, int_def.max_cycles,
            )

            # Run integration test agent (generates + runs tests)
            test_result = await self._run_integration_test_agent()
            if test_result is None:
                logger.warning(
                    "[IntegrationCheckpoint] Integration test agent returned no result"
                )
                break

            test_file = test_result.metrics.get("integration_test_file", "")
            tests_passed = test_result.success

            if tests_passed:
                logger.info(
                    "[IntegrationCheckpoint] Integration tests PASSED (cycle %d)",
                    cycle,
                )
                if self._event_bus:
                    await self._event_bus.publish(AgentEvent(
                        type=BusEventType.TEST_PASSED,
                        task_type=TaskType.GENERATE_INTEGRATION_TEST.value,
                        file_path=test_file,
                        agent_name="integration_checkpoint",
                        data={"cycle": cycle},
                    ))
                return {
                    "passed": True,
                    "cycles": cycle,
                    "test_file": test_file,
                    "files_fixed": files_fixed,
                }

            # Tests failed — triage: is it a test bug or source bug?
            test_output = test_result.metrics.get("integration_test_output", "")
            if not test_output and test_result.errors:
                test_output = "\n".join(test_result.errors)

            if not test_output:
                logger.warning(
                    "[IntegrationCheckpoint] No test output to triage — skipping fix"
                )
                break

            triage = await self._triage_integration_failure(
                test_file, test_output,
            )
            fix_target = triage.get("fix_target", "test")  # "test" or "source"
            source_files = triage.get("source_files", [])

            if fix_target == "source" and source_files:
                # Fix source code
                logger.info(
                    "[IntegrationCheckpoint] Triage: source bug in %s — "
                    "dispatching source fix",
                    source_files,
                )
                sem = asyncio.Semaphore(self._settings.max_concurrent_agents)
                fix_tasks: list[asyncio.Task[None]] = []

                for fp in source_files:
                    fix_ctx: dict[str, Any] = {
                        "fix_trigger": "integration_test",
                        "test_errors": test_output[:4000],
                        "test_file": test_file,
                        "fix_attempt": cycle,
                        "max_fix_attempts": int_def.max_cycles,
                    }

                    async def _fix_src(fpath: str, ctx: dict[str, Any]) -> None:
                        async with sem:
                            await self._dispatch_fix(
                                fpath, ctx, label="IntegrationCheckpoint",
                            )

                    fix_tasks.append(asyncio.create_task(_fix_src(fp, fix_ctx)))
                    if fp not in files_fixed:
                        files_fixed.append(fp)

                if fix_tasks:
                    await asyncio.gather(*fix_tasks, return_exceptions=True)

                # Rebuild after source fix
                if self._compiled and self._lang.build_command:
                    rebuild_ck = BuildCheckpoint(
                        build_command=self._lang.build_command,
                        terminal=self._am.build_terminal,
                        attributor=CompilerErrorAttributor(),
                        max_retries=1,
                        timeout=300,
                        checkpoint_name="integration_rebuild",
                    )
                    rebuild = await rebuild_ck.run_once(attempt=1)
                    if not rebuild.passed:
                        logger.warning(
                            "[IntegrationCheckpoint] Source fix broke build — "
                            "dispatching build fixes"
                        )
                        for fp in rebuild.affected_files:
                            build_ctx = rebuild_ck.get_fix_context_for_file(
                                fp, rebuild,
                            )
                            build_ctx["fix_trigger"] = "integration_rebuild"
                            await self._dispatch_fix(
                                fp, build_ctx, label="IntegrationRebuild",
                            )
            else:
                # Fix test code
                logger.info(
                    "[IntegrationCheckpoint] Triage: test bug — rewriting test file"
                )
                fix_ctx_test: dict[str, Any] = {
                    "fix_trigger": "integration_test_code",
                    "test_errors": test_output[:4000],
                    "test_file": test_file,
                    "fix_attempt": cycle,
                    "max_fix_attempts": int_def.max_cycles,
                }
                await self._dispatch_fix(
                    test_file, fix_ctx_test, label="IntegrationCheckpoint",
                )
                if test_file not in files_fixed:
                    files_fixed.append(test_file)

        # Cycles exhausted
        logger.warning(
            "[IntegrationCheckpoint] Integration checkpoint exhausted %d cycles",
            int_def.max_cycles,
        )
        if self._event_bus:
            await self._event_bus.publish(AgentEvent(
                type=BusEventType.TEST_FAILED,
                task_type=TaskType.GENERATE_INTEGRATION_TEST.value,
                file_path=test_file,
                agent_name="integration_checkpoint",
                data={"cycles": int_def.max_cycles},
            ))
        return {
            "passed": False,
            "cycles": int_def.max_cycles,
            "test_file": test_file,
            "files_fixed": files_fixed,
        }

    async def _run_integration_test_agent(self) -> TaskResult | None:
        """Execute the IntegrationTestAgent and return its TaskResult."""
        from core.context_builder import ContextBuilder
        from core.observability import record_agent_start, record_agent_end

        task = Task(
            task_id=0,
            task_type=TaskType.GENERATE_INTEGRATION_TEST,
            file="tests/integration/",
            description="Generate and run integration tests",
        )

        try:
            context_builder = ContextBuilder(
                workspace_dir=self._am.repo.workspace,
                blueprint=self._am.blueprint,
                repo_index=self._am.repo.get_repo_index(),
                dep_store=self._am._dep_store,
                embedding_store=self._am._embedding_store,
            )
            context = await asyncio.to_thread(context_builder.build, task)
        except Exception:
            logger.exception("[IntegrationCheckpoint] Context build failed")
            return None

        try:
            agent = self._am._create_agent(TaskType.GENERATE_INTEGRATION_TEST)
            record_agent_start()
            try:
                result = await agent.execute(context)
            finally:
                record_agent_end()

            agent_name = agent.role.value
            if agent_name not in self._am._metrics["agent_metrics"]:
                self._am._metrics["agent_metrics"][agent_name] = []
            self._am._metrics["agent_metrics"][agent_name].append(agent.get_metrics())
            return result
        except Exception:
            logger.exception("[IntegrationCheckpoint] Integration test agent failed")
            return None

    async def _triage_integration_failure(
        self,
        test_file: str,
        test_output: str,
    ) -> dict[str, Any]:
        """Use the LLM to determine if a test failure is a test bug or source bug.

        Returns:
            {"fix_target": "test" | "source", "source_files": [...]}
        """
        # Read the test file content
        try:
            test_content = self._am.repo.read_file(test_file)
            if test_content is None:
                test_content = "(could not read test file)"
        except Exception:
            test_content = "(could not read test file)"

        # Collect source file summaries
        source_files = [
            fb.path for fb in self._am.blueprint.file_blueprints
            if fb.layer not in ("test", "config", "deploy")
        ]
        source_summary = ", ".join(source_files[:20])

        prompt = (
            "An integration test failed. Determine if the failure is caused by:\n"
            "A) A bug in the TEST CODE (wrong imports, wrong endpoint path, bad "
            "assertions, incorrect test setup)\n"
            "B) A bug in the SOURCE CODE (missing route, wrong response format, "
            "logic error, missing function)\n\n"
            f"Test file: {test_file}\n"
            f"Test code:\n{test_content[:3000]}\n\n"
            f"Test output/errors:\n{test_output[:3000]}\n\n"
            f"Source files in project: {source_summary}\n\n"
            "Respond with JSON (no markdown fences):\n"
            '{"fix_target": "test" or "source", '
            '"source_files": ["file1.py", ...] (only if fix_target is "source"), '
            '"reasoning": "brief explanation"}'
        )

        try:
            data = await self._am.llm.generate_json(prompt)
            fix_target = data.get("fix_target", "test")
            source_files_to_fix = data.get("source_files", [])
            reasoning = data.get("reasoning", "")
            logger.info(
                "[IntegrationCheckpoint] Triage result: %s — %s",
                fix_target, reasoning,
            )
            return {
                "fix_target": fix_target,
                "source_files": source_files_to_fix,
                "reasoning": reasoning,
            }
        except Exception:
            logger.warning(
                "[IntegrationCheckpoint] Triage LLM call failed — "
                "defaulting to test fix"
            )
            return {"fix_target": "test", "source_files": []}
