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
import logging
import time
from typing import Any, TYPE_CHECKING

from core.checkpoint import BuildCheckpoint, CheckpointCycleResult, CheckpointResult
from core.error_attributor import CompilerErrorAttributor
from core.event_bus import AgentEvent, BusEventType
from core.models import Task, TaskType, TaskResult
from core.state_machine import EventType, FilePhase, FileLifecycle, LifecycleEngine
from core.stub_generator import StubGenerator
from core.tier_scheduler import Tier, TierScheduler

if TYPE_CHECKING:
    from config.settings import Settings
    from core.agent_manager import AgentManager
    from core.event_bus import EventBus
    from core.language import LanguageProfile
    from core.pipeline_definition import CheckpointDef, PipelineDefinition
    from core.task_engine import TaskGraph
    from tools.terminal_tools import TerminalTools

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
        self._reverify_queue: set[str] = set()

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
                for dep_path in dependents:
                    if engine.has_file(dep_path):
                        lc = engine.get_lifecycle(dep_path)
                        # Only queue files that have already moved past review —
                        # files still in early phases will naturally pick up changes.
                        if lc.phase in (FilePhase.BUILDING, FilePhase.TESTING,
                                        FilePhase.PASSED):
                            self._reverify_queue.add(dep_path)
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

        self._event_bus.subscribe(BusEventType.FILE_WRITTEN, on_file_changed)
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

                    if ck_def:
                        # Forward-reference stubs for later-tier files so this
                        # tier's checkpoint can resolve their types.
                        later_files = [
                            p for ft in tiers[tier_idx + 1:] for p in ft.files
                        ]
                        if later_files:
                            new_stubs = stub_gen.generate_stubs(later_files, bp_map)
                            active_stubs.extend(new_stubs)

                        cycle_result = await self._run_checkpoint(
                            engine, ck_def, tier,
                        )
                        checkpoint_results.append(cycle_result)
                        total_checkpoint_fixes += len(cycle_result.files_fixed)

                        # ── Hard gate: failed checkpoint blocks downstream tiers ─
                        if not cycle_result.passed:
                            logger.error(
                                "Tier %d checkpoint failed permanently — "
                                "blocking %d downstream tier(s) from starting",
                                tier.index, len(tiers) - tier_idx - 1,
                            )
                            for blocked_tier in tiers[tier_idx + 1:]:
                                for path in blocked_tier.files:
                                    lc = engine.get_lifecycle(path)
                                    if not lc.is_terminal:
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

        # ── Global DAG ──────────────────────────────────────────────────────
        logger.info("=== Global Tasks Phase ===")
        sentinel = next(
            (t for t in global_graph.tasks.values() if t.metadata.get("sentinel")),
            None,
        )
        if sentinel:
            global_graph.mark_completed(sentinel.task_id)
        await self._am.execute_graph(global_graph)

        # ── Results ─────────────────────────────────────────────────────────
        elapsed = time.monotonic() - start_time

        lifecycle_summary = engine.get_results_summary()
        global_stats = global_graph.get_stats()

        logger.info(
            "Pipeline execution complete in %.1fs. "
            "Files: %d passed, %d failed. Checkpoint fixes: %d. "
            "Global tasks: %s",
            elapsed,
            lifecycle_summary["passed"],
            lifecycle_summary["failed"],
            total_checkpoint_fixes,
            global_stats,
        )

        return {
            "stats": {
                **global_stats,
                "lifecycle_passed": lifecycle_summary["passed"],
                "lifecycle_failed": lifecycle_summary["failed"],
                "lifecycle_total_fixes": lifecycle_summary["total_fix_cycles"],
                "lifecycle_tests_degraded": lifecycle_summary["tests_degraded"],
                "checkpoint_fixes": total_checkpoint_fixes,
            },
            "metrics": self._am._metrics,
            "elapsed_seconds": elapsed,
            "lifecycle_summary": lifecycle_summary,
            "checkpoint_results": [
                {
                    "passed": cr.passed,
                    "attempts": cr.total_attempts,
                    "files_fixed": cr.files_fixed,
                }
                for cr in checkpoint_results
            ],
        }

    # ── Tier lifecycle execution ─────────────────────────────────────────

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
        phase_timeout = float(self._settings.phase_timeout_seconds)

        in_flight: set[str] = set()
        running_tasks: set[asyncio.Task[None]] = set()

        # Only process files in this tier
        tier_files = set(tier.files)

        async def run_file_phase(path: str, phase: FilePhase) -> None:
            async with semaphore:
                try:
                    await asyncio.wait_for(
                        self._am._execute_lifecycle_phase(engine, path, phase),
                        timeout=phase_timeout,
                    )
                except asyncio.TimeoutError:
                    logger.error(
                        "[%s] %s timed out after %ds — marking FAILED",
                        path, phase.value, int(phase_timeout),
                    )
                    try:
                        engine.process_event(path, EventType.RETRIES_EXHAUSTED)
                    except Exception:
                        pass
                    self._am._metrics["tasks_failed"] += 1
                finally:
                    in_flight.discard(path)

        # Phases that the tier loop must NOT dispatch — they're handled later:
        #   BUILDING  → handled by _run_checkpoint after the tier completes
        #   TESTING   → handled by _run_test_phase after all tiers + checkpoints
        deferred_phases = {FilePhase.BUILDING, FilePhase.TESTING}

        while True:
            actionable = [
                (path, phase) for path, phase in engine.get_actionable_files()
                if path in tier_files and path not in in_flight
                and phase not in deferred_phases
            ]

            for path, phase in actionable:
                in_flight.add(path)
                t = asyncio.create_task(
                    run_file_phase(path, phase), name=f"tier{tier.index}:{path}",
                )
                running_tasks.add(t)
                t.add_done_callback(running_tasks.discard)

            if not running_tasks:
                break

            await asyncio.wait(running_tasks, return_when=asyncio.FIRST_COMPLETED)

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

        checkpoint = BuildCheckpoint(
            build_command=build_command,
            terminal=self._am.build_terminal,
            attributor=CompilerErrorAttributor(),
            known_files=known_files,
            max_retries=checkpoint_def.max_retries,
            timeout=checkpoint_def.timeout,
            checkpoint_name=f"tier{tier.index}_{checkpoint_def.name}",
        )

        history: list[CheckpointResult] = []
        files_fixed: list[str] = []
        # Track how many fix attempts each file has had across retries.
        # A file that fails repeatedly shouldn't consume unlimited retries.
        fix_attempt_counts: dict[str, int] = {}
        max_fixes_per_file = 2  # cap per-file fix dispatches

        # Drain any files queued for re-verification by event bus handlers.
        # These are dependents of files that were fixed in earlier tiers.
        reverify = self._reverify_queue & set(tier.files)
        if reverify:
            logger.info(
                "[Checkpoint] Including %d re-verify files from event bus: %s",
                len(reverify), sorted(reverify),
            )
            self._reverify_queue -= reverify

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
            if not affected:
                # Errors couldn't be attributed — log and continue
                logger.warning(
                    "[Checkpoint] Build failed but no errors could be attributed. "
                    "Raw output: %s", result.raw_output[:500],
                )
                # Try to fix all tier files as a fallback
                affected = [
                    f for f in tier.files
                    if not engine.get_lifecycle(f).is_terminal
                ]

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

    async def _dispatch_checkpoint_fix(
        self,
        engine: LifecycleEngine,
        file_path: str,
        fix_context: dict[str, Any],
    ) -> None:
        """Dispatch a single fix task for a checkpoint build error."""
        from core.context_builder import ContextBuilder

        lc = engine.get_lifecycle(file_path)

        task = Task(
            task_id=0,
            task_type=TaskType.FIX_CODE,
            file=file_path,
            description=f"Fix build errors in {file_path}",
            metadata=fix_context,
        )

        context_builder = ContextBuilder(
            workspace_dir=self._am.repo.workspace,
            blueprint=self._am.blueprint,
            repo_index=self._am.repo.get_repo_index(),
            dep_store=self._am._dep_store,
            embedding_store=self._am._embedding_store,
        )
        context = context_builder.build(task)

        try:
            agent = self._am._create_agent(TaskType.FIX_CODE)
            from core.observability import record_agent_start, record_agent_end
            record_agent_start()
            try:
                result = await agent.execute(context)
            finally:
                record_agent_end()

            agent_name = agent.role.value
            if agent_name not in self._am._metrics["agent_metrics"]:
                self._am._metrics["agent_metrics"][agent_name] = []
            self._am._metrics["agent_metrics"][agent_name].append(agent.get_metrics())

            if result.success:
                self._am._metrics["tasks_completed"] += 1
                logger.info("[Checkpoint] Fixed build errors in %s", file_path)
                # Update embeddings
                if self._am._embedding_store and result.files_modified:
                    for fp in result.files_modified:
                        try:
                            content = self._am.repo.read_file(fp)
                            self._am._embedding_store.index_file(fp, content)
                        except Exception:
                            pass
            else:
                self._am._metrics["tasks_failed"] += 1
                logger.warning(
                    "[Checkpoint] Fix attempt failed for %s: %s",
                    file_path, result.errors,
                )
        except Exception:
            logger.exception("[Checkpoint] Error fixing %s", file_path)
            self._am._metrics["tasks_failed"] += 1

    # ── Test phase ───────────────────────────────────────────────────────

    async def _run_test_phase(self, engine: LifecycleEngine) -> None:
        """Run test generation for all files that reached TESTING phase."""
        semaphore = asyncio.Semaphore(self._settings.max_concurrent_agents)
        phase_timeout = float(self._settings.phase_timeout_seconds)

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
                    except Exception:
                        pass
                finally:
                    in_flight.discard(path)

        while True:
            # Get files in TESTING or FIXING (test fix) phase
            actionable = [
                (path, phase) for path, phase in engine.get_actionable_files()
                if path not in in_flight
                and phase in (FilePhase.TESTING, FilePhase.FIXING)
            ]

            # Also check for files still in BUILDING that should transition
            for path in list(engine._lifecycles.keys()):
                lc = engine.get_lifecycle(path)
                if lc.phase == FilePhase.BUILDING and path not in in_flight:
                    # Auto-transition to TESTING
                    engine.process_event(path, EventType.BUILD_PASSED)
                    new_phase = lc.phase
                    if new_phase in (FilePhase.TESTING, FilePhase.PASSED):
                        if new_phase == FilePhase.TESTING:
                            actionable.append((path, new_phase))

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
                            task = Task(
                                task_id=0,
                                task_type=TaskType.FIX_CODE,
                                file=fp,
                                description=f"Fix test failures in {fp}",
                                metadata=ctx,
                            )
                            from core.context_builder import ContextBuilder
                            context_builder = ContextBuilder(
                                workspace_dir=self._am.repo.workspace,
                                blueprint=self._am.blueprint,
                                repo_index=self._am.repo.get_repo_index(),
                                dep_store=self._am._dep_store,
                                embedding_store=self._am._embedding_store,
                            )
                            context = context_builder.build(task)
                            try:
                                agent = self._am._create_agent(TaskType.FIX_CODE)
                                fix_result = await agent.execute(context)
                                if fix_result.success:
                                    logger.info("[TestCheckpoint] Fixed %s", fp)
                                else:
                                    logger.warning("[TestCheckpoint] Fix failed: %s", fp)
                            except Exception:
                                logger.exception("[TestCheckpoint] Error fixing %s", fp)

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
