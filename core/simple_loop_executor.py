"""Simple loop executor — tight generate → build → fix per-file loop.

Core execution engine. Minimal per-file loop that converges with few LLM calls:

  PLAN (once)
    ↓
  FOR each tier:
    FOR each file (parallel within tier):
      generate → write → build → fix (loop)
    ↓
  GLOBAL BUILD → DONE

Design decisions:
  - Optional ReviewerAgent pass (feature flag QUICK_REVIEW)
  - No separate review-fix cycle — only build-fix
  - Accumulated error context across fix iterations (memory)
  - Smart retry: escalate prompt when same error repeats
  - Deterministic stopping conditions

LLM calls per file: 1 (generate) + 0-4 (fixes) = 1-5
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from core.checkpoint import BuildCheckpoint, CheckpointResult
from core.context_cache import ContextCache
from core.checkpoint_executor import CheckpointExecutor

__all__ = ["SimpleLoopExecutor"]
from core.error_attributor import (
    CompilerErrorAttributor,
    extract_error_lines,
    extract_referenced_files,
)
from core.event_bus import AgentEvent, BusEventType
from core.fast_validator import FastValidator
from core.fix_loop import FixLoop
from core.keyed_lock import KeyedLock
from core.models import AgentContext, Task, TaskResult, TaskType
from core.feature_flags import feature
from core.graceful_shutdown import shutdown_requested
from core.session_state import SessionState
from core.state_machine import EventType, FilePhase, LifecycleEngine
from core.tier_scheduler import Tier
from memory.fix_memory_store import FixMemoryStore

if TYPE_CHECKING:
    from config.settings import Settings
    from core.agent_manager import AgentManager
    from core.event_bus import EventBus
    from core.hooks import HookRegistry
    from core.language import LanguageProfile
    from core.pipeline_definition import (
        IntegrationCheckpointDef,
        PipelineDefinition,
        SecurityCheckpointDef,
    )
    from core.task_engine import TaskGraph

logger = logging.getLogger(__name__)


@dataclass
class FileLoopResult:
    """Result of processing a single file through the generate-build-fix loop."""
    file_path: str
    success: bool
    attempts: int
    fix_count: int = 0
    errors: list[str] = field(default_factory=list)


class SimpleLoopExecutor:
    """Tight generate → build → fix loop executor.

    For each file: generates code, runs the build, parses errors, fixes
    with accumulated context, and repeats until the build passes or
    max_attempts is reached.

    The executor respects tier ordering so foundational files (models,
    interfaces) are generated before dependent files (services, controllers).
    Within a tier, files are processed concurrently.
    """

    # Max fix attempts per file (including the initial generation).
    MAX_ATTEMPTS = 5

    # Max errors text per fix prompt to avoid token explosion.
    # Raised from 4000 → 8000: truncating too aggressively caused fix agents
    # to miss critical type-mismatch details buried later in compiler output.
    MAX_ERROR_CHARS = 8000

    def __init__(
        self,
        agent_manager: AgentManager,
        settings: Settings,
        lang_profile: LanguageProfile,
        *,
        event_bus: EventBus | None = None,
        hooks: HookRegistry | None = None,
    ) -> None:
        self._am = agent_manager
        self._settings = settings
        self._lang = lang_profile
        self._compiled = bool(lang_profile.build_command)
        self._event_bus = event_bus
        self._hooks: HookRegistry | None = hooks
        self._attributor = CompilerErrorAttributor()
        # Per-module build locks instead of one global lock.
        # The old single asyncio.Lock() serialized ALL builds across all files
        # in all tiers, adding ~15-30s per build wait in a 20-file project.
        # Now each "module" (top-level directory) gets its own lock so builds
        # for independent modules can run concurrently.
        self._build_locks = KeyedLock()
        self._fix_memory = FixMemoryStore(self._am.repo.workspace)
        # Shared context cache — reused within a tier, cleared between tiers
        self._context_cache = ContextCache()
        # Fast syntax validator — checks generated code before expensive builds
        self._fast_validator = FastValidator(self._am.repo.workspace)
        # Session state for resumable runs (initialised lazily in execute())
        self._session: SessionState | None = None
        # Checkpoint executor — extracted build / security / integration logic
        self._checkpoints: CheckpointExecutor | None = None  # Initialised lazily

    def _ensure_checkpoints(self) -> CheckpointExecutor:
        """Lazily initialise the checkpoint executor (needs _fix_file bound)."""
        if self._checkpoints is None:
            self._checkpoints = CheckpointExecutor(
                agent_manager=self._am,
                settings=self._settings,
                lang_profile=self._lang,
                attributor=self._attributor,
                build_locks=self._build_locks,
                dispatch_fix=self._dispatch_fix,
                fix_file=self._fix_file,
                event_bus=self._event_bus,
                hooks=self._hooks,
            )
        return self._checkpoints

    # ── Main entry point ────────────────────────────────────────────────

    async def execute(
        self,
        engine: LifecycleEngine,
        global_graph: TaskGraph,
        *,
        pipeline_def: PipelineDefinition | None = None,
        tiers: list[Tier] | None = None,
    ) -> dict[str, Any]:
        """Execute the full pipeline using the simple loop strategy."""
        start_time = time.monotonic()

        # ── Session resume ──────────────────────────────────────────────
        if feature("SESSION_RESUME"):
            import hashlib as _hl
            _all = sorted(f for t in (tiers or []) for f in t.files)
            _run_id = _hl.md5("|".join(_all).encode()).hexdigest()[:12]
            self._session = SessionState.load_or_create(
                str(self._am.repo.workspace), _run_id,
            )
            logger.info(
                "Session %s loaded: %s", _run_id, self._session.summary,
            )

        # Default: all files in a single tier
        if tiers is None:
            all_files = list(engine._lifecycles.keys())
            tiers = [Tier(index=0, files=sorted(all_files))]

        stats = {
            "total_files": sum(len(t.files) for t in tiers),
            "passed": 0,
            "failed": 0,
            "degraded": 0,
            "total_fixes": 0,
            "total_attempts": 0,
        }

        skip_agents = self._settings.skip_agents

        # ── DAG executor: per-file dependency scheduling ────────────────
        if feature("DAG_EXECUTOR"):
            dag_stats = await self._execute_dag(engine, tiers)
            stats.update(dag_stats)
        else:
            await self._execute_tiered(engine, tiers, stats)

        # ── Global build (one final check) ──────────────────────────────
        global_build_ok = True
        if self._compiled:
            global_build_ok = await self._global_build(engine)

        # ── Runtime smoke test (optional, post-build) ───────────────────
        # After the global build passes, start the application and verify
        # it boots without crashing.  Catches runtime issues (missing beans,
        # circular DI, invalid config) that the compiler won't catch.
        smoke_test_result = None
        if feature("RUNTIME_SMOKE_TEST") and global_build_ok:
            try:
                from core.runtime_smoke_test import RuntimeSmokeTest
                blueprint = self._am.blueprint or getattr(self._am.repo, "blueprint", None)
                if blueprint:
                    smoke = RuntimeSmokeTest(
                        workspace=self._am.repo.workspace,
                        tech_stack=blueprint.tech_stack,
                    )
                    smoke_test_result = await smoke.run()
                    if smoke_test_result.success:
                        logger.info("Runtime smoke test PASSED: %s", smoke_test_result.summary())
                    else:
                        logger.warning("Runtime smoke test FAILED: %s", smoke_test_result.summary())
                        if self._event_bus:
                            self._event_bus.publish(AgentEvent(
                                event_type=BusEventType.LIFECYCLE_TRANSITION,
                                data={
                                    "type": "smoke_test_failed",
                                    "errors": smoke_test_result.errors,
                                    "stdout_tail": smoke_test_result.stdout_tail[-500:],
                                },
                            ))
            except Exception:
                logger.debug("Runtime smoke test failed (non-critical)", exc_info=True)

        # ── Test generation (optional) ──────────────────────────────────
        if "tester" not in skip_agents:
            await self._run_test_phase(engine)

        security_result: dict[str, Any] = {}
        integration_result: dict[str, Any] = {}

        if pipeline_def and (
            pipeline_def.security_checkpoint or pipeline_def.integration_checkpoint
        ):
            logger.info("=== Post-Generation Phase: Track A (Security → Integration) ===")
            if pipeline_def.security_checkpoint and "security" not in skip_agents:
                security_result = await self._run_security_checkpoint(
                    pipeline_def.security_checkpoint,
                )
            elif "security" in skip_agents:
                logger.info("=== Security Checkpoint skipped (--skip-security) ===")

            if pipeline_def.integration_checkpoint and "integration" not in skip_agents:
                integration_result = await self._run_integration_checkpoint(
                    pipeline_def.integration_checkpoint,
                )
            elif "integration" in skip_agents:
                logger.info("=== Integration Checkpoint skipped (--skip-integration) ===")

        # ── Advisory tasks (deploy, docs — fire and forget) ─────────────
        logger.info("=== Advisory Tasks ===")
        sentinel = next(
            (t for t in global_graph.tasks.values()
             if t.metadata.get("sentinel")),
            None,
        )
        if sentinel:
            global_graph.mark_completed(sentinel.task_id)
        await self._am.execute_graph(global_graph)

        # ── Results ─────────────────────────────────────────────────────
        elapsed = time.monotonic() - start_time

        lifecycle_summary = engine.get_results_summary()

        logger.info(
            "SimpleLoop execution complete in %.1fs. "
            "Files: %d passed, %d failed, %d degraded. Total fixes: %d.",
            elapsed,
            stats["passed"],
            stats["failed"],
            lifecycle_summary.get("degraded", 0),
            stats["total_fixes"],
        )

        # ── Flush fix memory to disk ───────────────────────────────────
        await self._fix_memory.flush()

        return {
            "stats": {
                "passed": stats["passed"],
                "failed": stats["failed"],
                "degraded": lifecycle_summary.get("degraded", 0),
                "total_attempts": stats["total_attempts"],
                "lifecycle_passed": lifecycle_summary["passed"],
                "lifecycle_degraded": lifecycle_summary.get("degraded", 0),
                "lifecycle_failed": lifecycle_summary["failed"],
                "lifecycle_total_fixes": stats["total_fixes"],
                "lifecycle_tests_degraded": lifecycle_summary.get("tests_degraded", 0),
                "checkpoint_fixes": 0,
                "final_build_passed": global_build_ok,
            },
            "metrics": self._am._metrics,
            "elapsed_seconds": elapsed,
            "lifecycle_summary": lifecycle_summary,
            "bus_handler_failures": [],
            "checkpoint_results": [],
            "security_checkpoint": security_result,
            "integration_checkpoint": integration_result,
            "smoke_test": {
                "passed": smoke_test_result.success if smoke_test_result else None,
                "startup_time": smoke_test_result.startup_time_seconds if smoke_test_result else None,
                "health_endpoint": smoke_test_result.health_endpoint if smoke_test_result else None,
                "errors": smoke_test_result.errors if smoke_test_result else [],
            } if smoke_test_result else {},
        }

    # ── Core per-file loop ──────────────────────────────────────────────

    async def _process_file(
        self,
        engine: LifecycleEngine,
        file_path: str,
    ) -> FileLoopResult:
        """Process a single file: generate → build → fix loop.

        This is the core loop that replaces the complex multi-phase FSM.
        Each iteration adds error context so the fix agent learns from
        previous failures.
        """
        lc = engine.get_lifecycle(file_path)

        # Transition PENDING → GENERATING
        if lc.phase == FilePhase.PENDING:
            engine.process_event(file_path, EventType.DEPS_MET)

        # If already terminal (e.g., cascaded failure), skip
        if lc.is_terminal:
            return FileLoopResult(
                file_path=file_path,
                success=lc.phase == FilePhase.PASSED,
                attempts=0,
            )

        error_history: list[dict[str, Any]] = []
        fix_count = 0
        fix_loop = FixLoop(file_path, max_attempts=self.MAX_ATTEMPTS, max_error_chars=self.MAX_ERROR_CHARS)
        _file_tokens = 0  # cumulative tokens spent on this file
        _file_token_budget = self._settings.execution.file_token_budget
        # Snapshot of last-known-good file content.  If a fix makes the file
        # worse (introduces MORE errors than before), we can roll back to this
        # snapshot rather than accumulating damage across fix iterations.
        _last_good_content: str | None = None
        _last_error_count: int | None = None

        for attempt in range(self.MAX_ATTEMPTS):
            if shutdown_requested():
                return FileLoopResult(
                    file_path=file_path, success=False, attempts=attempt,
                    fix_count=fix_count, errors=["Shutdown requested"],
                )

            # Check per-file token budget
            if feature("TOKEN_BUDGETS") and _file_token_budget > 0 and _file_tokens >= _file_token_budget:
                logger.warning(
                    "[%s] Per-file token budget exhausted (%d/%d) — "
                    "accepting current state after %d attempts",
                    file_path, _file_tokens, _file_token_budget, attempt,
                )
                if not lc.is_terminal:
                    engine.process_event(file_path, EventType.RETRIES_EXHAUSTED)
                return FileLoopResult(
                    file_path=file_path, success=False, attempts=attempt,
                    fix_count=fix_count,
                    errors=[f"Token budget exhausted ({_file_tokens}/{_file_token_budget})"],
                )

            # Guard: if file became terminal (e.g. DEGRADED), stop
            if lc.is_terminal:
                break

            # ── Step 1: Generate or Fix ─────────────────────────────────
            _tokens_before = (
                self._am.llm.total_input_tokens + self._am.llm.total_output_tokens
            )
            if attempt == 0:
                # Initial generation
                gen_ok = await self._generate_file(engine, file_path)
                if not gen_ok:
                    logger.error("[%s] Generation failed", file_path)
                    engine.process_event(file_path, EventType.RETRIES_EXHAUSTED)
                    return FileLoopResult(
                        file_path=file_path, success=False, attempts=1,
                        errors=["Generation failed"],
                    )
                # After _execute_lifecycle_phase fires CODE_GENERATED, the
                # file is in REVIEWING.  Optionally run a quick review to
                # catch semantic issues (null deref, missing error handling,
                # layer violations) before the build.  When QUICK_REVIEW is
                # off, skip review by firing REVIEW_PASSED.
                # If rewrite was rejected, file stays in GENERATING — retry.
                if lc.phase == FilePhase.REVIEWING:
                    review_findings = await self._quick_review(engine, file_path)
                    if review_findings:
                        # Seed the error history so the fix loop is aware of
                        # semantic issues even before the first build.
                        fix_loop.record_review_findings(review_findings, attempt=0)
                        error_history = fix_loop.error_history_dicts
                    if lc.phase == FilePhase.REVIEWING:
                        # QUICK_REVIEW was off or review already passed
                        engine.process_event(file_path, EventType.REVIEW_PASSED)
                elif lc.phase == FilePhase.GENERATING:
                    # Rewrite rejected — will retry on next attempt
                    logger.warning("[%s] Rewrite rejected, retrying", file_path)
                    continue
            else:
                # ── Snapshot before fix ──────────────────────────────────
                # Save the current file content so we can roll back if the
                # fix makes things worse (more errors than before).
                try:
                    _pre_fix_content = await asyncio.to_thread(
                        self._am.repo.read_file, file_path,
                    )
                except Exception:
                    _pre_fix_content = None

                # Fix with accumulated context
                fix_ok = await self._fix_file(engine, file_path, fix_loop.error_history_dicts)
                fix_count += 1
                fix_loop.fix_count = fix_count
                if not fix_ok:
                    logger.warning("[%s] Fix attempt %d failed", file_path, attempt)
                    # Don't give up — the build might still pass
                engine.process_event(
                    file_path, EventType.FIX_APPLIED,
                )

            # ── Step 2: Build ───────────────────────────────────────────
            # Track tokens consumed by the generate/fix step
            _tokens_after = (
                self._am.llm.total_input_tokens + self._am.llm.total_output_tokens
            )
            _file_tokens += _tokens_after - _tokens_before

            # ── Step 2a: Fast syntax validation (pre-build filter) ──────
            # When FAST_SYNTAX_CHECK is enabled, run a sub-second syntax
            # check before the expensive full-project build. If syntax
            # fails, skip the build entirely and feed errors to fix agent.
            if feature("FAST_SYNTAX_CHECK") and self._compiled:
                _lang = self._lang.name if self._lang else ""
                syntax_ok, syntax_errors = await self._fast_validator.validate_syntax(
                    file_path, _lang,
                )
                if not syntax_ok:
                    logger.info(
                        "[%s] Fast syntax check failed — skipping build (attempt %d)",
                        file_path, attempt + 1,
                    )
                    error_hash = hashlib.md5(syntax_errors.encode()).hexdigest()[:8]
                    fix_loop.record_error(
                        f"SYNTAX ERROR (pre-build):\n{syntax_errors}",
                        attempt=attempt + 1,
                    )
                    error_history = fix_loop.error_history_dicts
                    engine.process_event(
                        file_path, EventType.BUILD_FAILED,
                        data={"errors": syntax_errors},
                    )
                    if lc.is_terminal:
                        break
                    continue  # Skip to fix iteration

            # For interpreted languages (no build command), auto-pass
            if not self._compiled:
                lint_ok = await self._lint_check(file_path)
                if lint_ok:
                    engine.process_event(file_path, EventType.BUILD_PASSED)
                    return FileLoopResult(
                        file_path=file_path, success=True,
                        attempts=attempt + 1, fix_count=fix_count,
                    )
                # Lint failed — treat like build error
                errors_text = f"Linter/type-checker errors in {file_path}"
                fix_loop.record_error(errors_text, attempt=attempt + 1)
                error_history = fix_loop.error_history_dicts
                engine.process_event(
                    file_path, EventType.BUILD_FAILED,
                    data={"errors": errors_text},
                )
                # If build fix limit exceeded, file becomes DEGRADED — stop
                if lc.is_terminal:
                    break
                continue

            # Run incremental build
            build_result = await self._run_incremental_build(file_path)

            if build_result.passed:
                engine.process_event(file_path, EventType.BUILD_PASSED)
                if fix_count > 0 or fix_loop.error_history:
                    last_hash = fix_loop.latest_error_hash
                    await self._fix_memory.record_resolution(
                        file_path,
                        self._build_resolution_summary(error_history),
                        error_hash=last_hash,
                    )
                logger.info(
                    "[%s] Build passed (attempt %d, %d fixes)",
                    file_path, attempt + 1, fix_count,
                )
                return FileLoopResult(
                    file_path=file_path, success=True,
                    attempts=attempt + 1, fix_count=fix_count,
                )

            # ── Step 3: Parse errors and map to this file ───────────────
            raw_output = build_result.raw_output
            attribution = self._attributor.attribute(
                raw_output,
                known_files=set(engine._lifecycles.keys()),
            )

            # Get errors specific to this file
            file_errors_text = attribution.summary_for_file(file_path)
            if not file_errors_text:
                # No errors attributed to this file — could be errors in
                # other files or generic build errors. Use extracted error
                # lines as fallback context.
                file_errors_text = extract_error_lines(
                    raw_output, max_chars=self.MAX_ERROR_CHARS,
                )

            if not file_errors_text.strip():
                # Build failed but no parseable errors — bail to avoid
                # infinite loop with empty fix context
                logger.warning(
                    "[%s] Build failed but no errors could be parsed — stopping",
                    file_path,
                )
                break

            error_hash = hashlib.md5(file_errors_text.encode()).hexdigest()[:8]
            referenced_files = extract_referenced_files(
                raw_output,
                known_files=set(engine._lifecycles.keys()),
            )

            # ── Step 4: Check for stalled progress ──────────────────────
            if fix_loop.error_history and fix_loop.latest_error_hash == error_hash:
                logger.warning(
                    "[%s] Same errors repeated after fix (hash=%s) — "
                    "will escalate on next attempt",
                    file_path, error_hash,
                )

            # ── Step 4b: Rollback if fix made things worse ──────────────
            # Count errors attributed to this file; if the fix introduced
            # MORE errors than the previous iteration, restore the snapshot.
            _current_error_count = len(
                attribution.errors_by_file.get(file_path, [])
            ) if attribution else file_errors_text.count("\n")
            if (
                _last_error_count is not None
                and _current_error_count > _last_error_count
                and _pre_fix_content is not None  # type: ignore[possibly-undefined]
                and attempt > 1
            ):
                logger.warning(
                    "[%s] Fix made things worse (%d→%d errors) — "
                    "rolling back to pre-fix snapshot",
                    file_path, _last_error_count, _current_error_count,
                )
                try:
                    await self._am.repo.write_file(file_path, _pre_fix_content)
                    # Restore the error count to the pre-fix level
                    _current_error_count = _last_error_count
                except Exception:
                    logger.debug(
                        "[%s] Rollback write failed (non-critical)",
                        file_path, exc_info=True,
                    )
            _last_error_count = _current_error_count

            # Record via fix_loop and extract signatures
            sigs: list[str] = []
            if feature("STREAMING_FIX"):
                from core.streaming_fix import extract_error_signature
                sigs = extract_error_signature(file_errors_text) or []

            fix_loop.record_error(
                file_errors_text,
                attempt=attempt + 1,
                referenced_files=referenced_files,
                error_signatures=sigs,
            )
            error_history = fix_loop.error_history_dicts

            if sigs:
                logger.debug(
                    "[%s] Streaming fix signatures: %s",
                    file_path, sigs,
                )

            await self._fix_memory.record_attempt(
                file_path,
                errors_text=file_errors_text,
                error_hash=error_hash,
                referenced_files=referenced_files,
            )

            engine.process_event(
                file_path, EventType.BUILD_FAILED,
                data={"errors": file_errors_text},
            )

            # If build fix limit exceeded, file becomes DEGRADED — stop
            if lc.is_terminal:
                break

        # ── Max attempts exhausted ──────────────────────────────────────
        logger.error(
            "[%s] Failed after %d attempts (%d fixes)",
            file_path, self.MAX_ATTEMPTS, fix_count,
        )
        if not lc.is_terminal:
            engine.process_event(file_path, EventType.RETRIES_EXHAUSTED)

        return FileLoopResult(
            file_path=file_path, success=False,
            attempts=self.MAX_ATTEMPTS, fix_count=fix_count,
            errors=[r.errors_text[:200] for r in fix_loop.error_history[-2:]],
        )

    # ── Execution strategies ──────────────────────────────────────────

    async def _execute_tiered(
        self,
        engine: LifecycleEngine,
        tiers: list[Tier],
        stats: dict[str, Any],
    ) -> None:
        """Tier-sequential execution: process tiers in order, files concurrent within tier."""
        for tier_idx, tier in enumerate(tiers):
            # Skip entire tier if session says it's complete
            if self._session and self._session.is_tier_complete(tier.index):
                skipped = len(tier.files)
                stats["passed"] += skipped
                logger.info(
                    "=== Tier %d: SKIPPED (%d files already passed) ===",
                    tier.index, skipped,
                )
                continue

            tier_names = [p.rsplit("/", 1)[-1] for p in tier.files]
            logger.info(
                "=== Tier %d: %d files === %s",
                tier.index, len(tier), tier_names,
            )

            await self._fire_hook("tier_start", tier_index=tier.index, files=tier.files)

            if shutdown_requested():
                logger.warning("Shutdown requested — aborting at tier %d", tier.index)
                break

            # Filter out files that already passed in a previous session run
            if self._session:
                original_count = len(tier.files)
                tier_files_filtered = [
                    f for f in tier.files if not self._session.is_file_passed(f)
                ]
                if len(tier_files_filtered) < original_count:
                    logger.info(
                        "Tier %d: resuming — %d/%d files already passed, %d remaining",
                        tier.index, original_count - len(tier_files_filtered),
                        original_count, len(tier_files_filtered),
                    )
                    stats["passed"] += original_count - len(tier_files_filtered)
            else:
                tier_files_filtered = tier.files

            # ── Batch generation: group small independent files ─────────
            files_to_process = tier_files_filtered
            if feature("BATCH_GENERATION"):
                files_to_process = await self._try_batch_generate(
                    engine, tier_files_filtered,
                )

            # Process files concurrently within the tier
            semaphore = asyncio.Semaphore(self._settings.max_concurrent_agents)

            async def _process_with_sem(fp: str) -> FileLoopResult:
                async with semaphore:
                    return await self._process_file(engine, fp)

            results = await asyncio.gather(
                *[_process_with_sem(f) for f in files_to_process],
                return_exceptions=True,
            )

            # Collect stats and cascade failures
            tier_failed = []
            for file_path, result in zip(files_to_process, results):
                if isinstance(result, Exception):
                    logger.exception(
                        "Unexpected error processing %s", file_path,
                    )
                    try:
                        engine.process_event(file_path, EventType.RETRIES_EXHAUSTED)
                    except Exception:
                        engine.get_lifecycle(file_path).phase = FilePhase.FAILED
                    stats["failed"] += 1
                    tier_failed.append(file_path)
                    if self._session:
                        self._session.mark_file_failed(file_path, str(result)[:200])
                elif result.success:
                    stats["passed"] += 1
                    stats["total_fixes"] += result.fix_count
                    stats["total_attempts"] += result.attempts
                    if self._session:
                        self._session.mark_file_passed(file_path)
                else:
                    stats["failed"] += 1
                    stats["total_fixes"] += result.fix_count
                    stats["total_attempts"] += result.attempts
                    tier_failed.append(file_path)
                    if self._session:
                        error_msg = "; ".join(result.errors[:2]) if result.errors else "unknown"
                        self._session.mark_file_failed(file_path, error_msg)

            # Cascade failures to downstream tiers
            if tier_failed:
                logger.warning(
                    "Tier %d: %d files failed: %s",
                    tier.index, len(tier_failed), tier_failed,
                )
                all_later = {
                    p for ft in tiers[tier_idx + 1:] for p in ft.files
                }
                cascaded = engine.cascade_failures(scope=all_later)
                if cascaded:
                    logger.warning(
                        "Cascade-failed %d downstream file(s): %s",
                        len(cascaded), cascaded,
                    )
                    stats["failed"] += len(cascaded)

            # ── Cross-file consistency check (post-tier) ────────────────
            # After all files in the tier are generated, validate that
            # cross-file method references are consistent. Issues are
            # logged as warnings and injected into the event bus so
            # downstream tiers can account for mismatches.
            if feature("CROSS_FILE_VALIDATOR"):
                try:
                    from core.cross_file_validator import CrossFileValidator
                    blueprint = self._am.blueprint or getattr(self._am.repo, "blueprint", None)
                    if blueprint:
                        validator = CrossFileValidator(self._am.repo.workspace, blueprint)
                        passed_files = [
                            f for f in tier.files if f not in tier_failed
                        ]
                        if passed_files:
                            report = validator.validate(passed_files)
                            if report.issues:
                                logger.warning(
                                    "Tier %d cross-file validation: %s",
                                    tier.index, report.full_summary(),
                                )
                                if self._event_bus:
                                    self._event_bus.publish(AgentEvent(
                                        event_type=BusEventType.LIFECYCLE_TRANSITION,
                                        data={
                                            "type": "cross_file_validation",
                                            "tier": tier.index,
                                            "issues": [str(i) for i in report.issues],
                                        },
                                    ))
                except Exception:
                    logger.debug(
                        "Cross-file validation failed (non-critical)",
                        exc_info=True,
                    )

            # Mark tier complete and persist session state
            if self._session and not tier_failed:
                self._session.mark_tier_complete(tier.index)
            if self._session:
                try:
                    self._session.save()
                except Exception:
                    logger.debug("Failed to save session state (non-critical)", exc_info=True)

            self._save_lifecycle_state(engine)

            await self._fire_hook(
                "tier_complete",
                tier_index=tier.index,
                passed=len(tier.files) - len(tier_failed),
                failed=len(tier_failed),
            )

            # Clear shared context cache between tiers so modified files
            # are re-read with fresh content in downstream tiers.
            logger.debug("Clearing context cache between tiers (%s)", self._context_cache.stats)
            self._context_cache.clear()

    async def _execute_dag(
        self,
        engine: LifecycleEngine,
        tiers: list[Tier],
    ) -> dict[str, Any]:
        """DAG-driven execution: schedule files by dependency graph, not tiers."""
        from core.dag_executor import DAGExecutor, build_dag_from_tiers
        from core.feature_flags import feature

        all_files = build_dag_from_tiers(engine, tiers)
        logger.info("=== DAG Executor: %d files ===", len(all_files))

        async def _process_fn(fp: str) -> bool:
            result = await self._process_file(engine, fp)
            return result.success

        # --- Speculative execution support ---
        spec_kwargs: dict[str, Any] = {}
        if feature("SPECULATIVE_EXEC"):
            blueprint = self._am.repo.blueprint
            workspace = self._am.repo.workspace

            async def _speculative_generate(fp: str, prompt: str) -> str:
                """Use the LLM to generate speculative file content."""
                resp = await self._am.llm.generate(
                    system_prompt="Generate the requested source file based on the blueprint.",
                    prompt=prompt,
                )
                # Track speculative LLM calls so they appear in metrics
                self._am._metrics.setdefault("speculative_llm_calls", 0)
                self._am._metrics["speculative_llm_calls"] += 1
                return resp.content if hasattr(resp, "content") else str(resp)

            def _write_file(fp: str, content: str) -> None:
                path = workspace / fp
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")

            def _read_file(fp: str) -> str:
                return (workspace / fp).read_text(encoding="utf-8")

            spec_kwargs = dict(
                speculative=True,
                blueprint=blueprint,
                speculative_generate_fn=_speculative_generate,
                write_fn=_write_file,
                read_fn=_read_file,
            )

        dag = DAGExecutor(
            max_concurrency=self._settings.max_concurrent_agents,
            process_fn=_process_fn,
            **spec_kwargs,
        )
        results = await dag.execute(engine, all_files)

        # Aggregate stats
        dag_stats: dict[str, Any] = {
            "passed": sum(1 for v in results.values() if v),
            "failed": sum(1 for v in results.values() if not v),
        }
        self._save_lifecycle_state(engine)
        return dag_stats

    # ── Batch generation ──────────────────────────────────────────────

    async def _try_batch_generate(
        self,
        engine: LifecycleEngine,
        tier_files: list[str],
    ) -> list[str]:
        """Attempt to batch-generate small independent files, returning remaining files.

        Files successfully generated by the batch are transitioned to REVIEWING
        and excluded from the returned list (they still go through build-fix).
        Files that fail or aren't eligible are returned for individual processing.
        """
        from core.batch_generator import plan_batches, build_batch_prompt, parse_batch_response

        blueprint = self._am.repo.blueprint
        if blueprint is None:
            return tier_files

        plan = plan_batches(blueprint, tier_files)
        if not plan.batches:
            return tier_files

        tech_summary = ", ".join(f"{k}: {v}" for k, v in blueprint.tech_stack.items())
        batch_generated: set[str] = set()

        for batch in plan.batches:
            batch_paths = [fb.path for fb in batch.files]
            logger.info(
                "Batch-generating %d files: %s",
                len(batch.files), batch_paths,
            )
            prompt = build_batch_prompt(batch, blueprint, tech_summary)

            try:
                response = await self._am.llm.generate(
                    prompt=prompt,
                    system_prompt=(
                        f"You are an expert developer generating code for the "
                        f"{blueprint.name} project ({blueprint.architecture_style}). "
                        f"Generate complete, production-quality code for each file."
                    ),
                    max_tokens=16384,
                )
                files = parse_batch_response(response.content)
                for path, content in files.items():
                    if path in {fb.path for fb in batch.files}:
                        # Write the file
                        await self._am.repo.write_file(path, content)
                        batch_generated.add(path)
                        # Transition lifecycle
                        lc = engine.get_lifecycle(path)
                        if lc.phase == FilePhase.PENDING:
                            engine.process_event(path, EventType.DEPS_MET)
                        if lc.phase == FilePhase.GENERATING:
                            engine.process_event(path, EventType.CODE_GENERATED)
                        if lc.phase == FilePhase.REVIEWING:
                            engine.process_event(path, EventType.REVIEW_PASSED)

                missed = set(batch_paths) - set(files.keys())
                if missed:
                    logger.warning(
                        "Batch missed %d files, will generate individually: %s",
                        len(missed), missed,
                    )
            except Exception:
                logger.exception(
                    "Batch generation failed for %s — falling back to individual",
                    batch_paths,
                )

        # Return files that still need individual processing (unbatched + missed)
        remaining = [f for f in tier_files if f not in batch_generated]
        if batch_generated:
            logger.info(
                "Batch generated %d files, %d remaining for individual processing",
                len(batch_generated), len(remaining),
            )
        return remaining

    # ── Agent dispatch helpers ──────────────────────────────────────────

    async def _generate_file(
        self,
        engine: LifecycleEngine,
        file_path: str,
    ) -> bool:
        """Run CoderAgent to generate a file. Returns True on success."""
        try:
            await self._am._execute_lifecycle_phase(
                engine, file_path, FilePhase.GENERATING,
            )
            return True
        except Exception:
            logger.exception("[%s] Generation phase failed", file_path)
            return False

    def _is_simple_file(self, file_path: str) -> bool:
        """Check if a file is simple enough to skip review (config, DTO, model).

        Uses the same heuristics as ModelRouter: files in model/config layers
        with ≤2 dependencies and no complex purpose signals are considered
        simple and unlikely to benefit from an LLM review pass.
        """
        blueprint = self._am.blueprint or (
            self._am.repo.blueprint if hasattr(self._am.repo, "blueprint") else None
        )
        if not blueprint:
            return False

        fb = None
        for candidate in blueprint.file_blueprints:
            if candidate.path == file_path:
                fb = candidate
                break
        if fb is None:
            return False

        layer = fb.layer.lower() if fb.layer else ""
        dep_count = len(fb.depends_on)
        purpose = fb.purpose.lower() if fb.purpose else ""

        # Simple layers that rarely have semantic bugs worth reviewing
        _simple_layers = {"model", "config", "dto", "entity", "enum", "constant"}
        _complex_signals = (
            "algorithm", "transaction", "concurren", "stream",
            "websocket", "caching", "pagination", "authentication",
            "authorization", "middleware", "interceptor", "security",
        )
        has_complex_purpose = any(s in purpose for s in _complex_signals)

        return layer in _simple_layers and dep_count <= 2 and not has_complex_purpose

    async def _quick_review(
        self,
        engine: LifecycleEngine,
        file_path: str,
    ) -> list[str]:
        """Run ReviewerAgent for a lightweight pre-build code review.

        Returns a list of critical finding messages (empty if review passed
        or if QUICK_REVIEW feature is disabled).  The findings are injected
        into the fix-loop error context so the coder agent can address
        semantic issues that the compiler wouldn't catch.

        Simple files (config, model/DTO layers with few dependencies) are
        automatically skipped to save an LLM call — the compiler will catch
        any issues in those files.
        """
        if not feature("QUICK_REVIEW"):
            return []

        # Skip review for compiled languages — the compiler catches type
        # errors, missing imports, and undefined symbols more reliably and
        # cheaply than an LLM review pass.  Reserve LLM review for
        # interpreted languages (Python) where there's no compiler gate.
        if self._compiled:
            logger.debug(
                "[%s] Skipping review (compiled language — compiler is the gate)",
                file_path,
            )
            return []

        # Skip review for simple files — config, models, DTOs rarely have
        # semantic bugs that the compiler won't catch anyway.
        if feature("SKIP_SIMPLE_REVIEW") and self._is_simple_file(file_path):
            logger.info(
                "[%s] Skipping review (simple file — model/config/DTO layer)",
                file_path,
            )
            return []

        try:
            await self._am._execute_lifecycle_phase(
                engine, file_path, FilePhase.REVIEWING,
            )
        except Exception:
            logger.warning("[%s] Quick review failed, skipping", file_path)
            # On failure, force through to BUILDING so we don't stall
            if not engine.get_lifecycle(file_path).is_terminal:
                try:
                    engine.process_event(file_path, EventType.REVIEW_PASSED)
                except Exception:
                    pass
            return []

        lc = engine.get_lifecycle(file_path)

        # ReviewerAgent fires REVIEW_PASSED or REVIEW_FAILED via
        # AgentManager._handle_phase_result, which updates the lifecycle.
        # If review failed, the lifecycle transitioned to FIXING with
        # findings stored on the lifecycle object.
        if lc.phase == FilePhase.FIXING:
            findings = list(lc.review_findings or [])
            logger.info(
                "[%s] Quick review found %d critical issue(s) — "
                "will include in fix context",
                file_path, len(findings),
            )
            # Move back to BUILDING so the normal build step runs.
            # The review findings will be injected as pre-seeded error context.
            lc.phase = FilePhase.BUILDING
            lc.fix_trigger = ""
            return findings

        # Review passed — file is now in BUILDING (normal path)
        return []

    async def _fix_file(
        self,
        engine: LifecycleEngine,
        file_path: str,
        error_history: list[dict[str, Any]],
    ) -> bool:
        """Run CoderAgent FIX_CODE with accumulated error context.

        Uses FixLoop for escalation logic and metadata assembly. Delegates
        stall detection and strategy rotation to the FixLoop class.
        """
        from core.context_builder import ContextBuilder
        from core.observability import record_agent_start, record_agent_end

        latest = error_history[-1] if error_history else {}
        errors_text = latest.get("errors_text", "")
        persistent_memory = await self._fix_memory.get_summary(file_path)

        # Cross-file learning: check if other files already resolved this
        # same error class and inject their resolution as a hint.
        error_hash = latest.get("error_hash", "")
        cross_file_hints = ""
        if error_hash:
            cross_file_hints = await self._fix_memory.get_cross_file_hints(error_hash)

        # Enhanced fix memory: semantic category matching for broader hints
        if feature("ENHANCED_FIX_MEMORY") and errors_text and not cross_file_hints:
            semantic_hints = await self._fix_memory.get_semantic_hints(errors_text)
            if semantic_hints:
                cross_file_hints = semantic_hints

        # Build a temporary FixLoop to compute metadata from the error_history dicts.
        # (When called from _process_file, the real FixLoop is maintained there.)
        from core.fix_loop import FixLoop as _FL, ErrorRecord
        temp_loop = _FL(file_path, max_attempts=self.MAX_ATTEMPTS)
        for h in error_history:
            temp_loop.error_history.append(ErrorRecord(
                errors_text=h.get("errors_text", ""),
                error_hash=h.get("error_hash", ""),
                attempt=h.get("attempt", 0),
                referenced_files=h.get("referenced_files", []),
                error_signatures=h.get("error_signatures", []),
            ))

        fix_metadata = temp_loop.build_fix_metadata(
            persistent_memory=persistent_memory,
            cross_file_hints=cross_file_hints,
        )

        # Build task and context
        task = Task(
            task_id=0,
            task_type=TaskType.FIX_CODE,
            file=file_path,
            description=f"Fix build errors in {file_path}",
            metadata=fix_metadata,
        )

        try:
            context_builder = ContextBuilder(
                workspace_dir=self._am.repo.workspace,
                blueprint=self._am.blueprint,
                repo_index=self._am.repo.get_repo_index(),
                dep_store=self._am._dep_store,
                embedding_store=self._am._embedding_store,
                api_contract=self._am._api_contract,
            )
            context = await asyncio.to_thread(context_builder.build, task)
        except Exception:
            logger.exception("[%s] Context build failed for fix", file_path)
            self._am._metrics["tasks_failed"] += 1
            return False

        try:
            agent = self._am._create_agent(TaskType.FIX_CODE)
            record_agent_start()
            try:
                result = await agent.execute(context)
            finally:
                record_agent_end()

            # Track metrics
            agent_name = agent.role.value
            if agent_name not in self._am._metrics["agent_metrics"]:
                self._am._metrics["agent_metrics"][agent_name] = []
            metrics_list = self._am._metrics["agent_metrics"][agent_name]
            metrics_list.append(agent.get_metrics())
            if len(metrics_list) > 100:
                metrics_list.pop(0)

            if result.success:
                self._am._metrics["tasks_completed"] += 1
                # Update embeddings for the modified file
                if self._am._embedding_store and result.files_modified:
                    for fp in result.files_modified:
                        try:
                            content = self._am.repo.read_file(fp)
                            self._am._embedding_store.index_file(fp, content)
                        except Exception:
                            pass
                return True
            else:
                self._am._metrics["tasks_failed"] += 1
                logger.warning(
                    "[%s] Fix failed: %s", file_path, result.errors,
                )
                return False
        except Exception:
            logger.exception("[%s] Error during fix", file_path)
            self._am._metrics["tasks_failed"] += 1
            return False

    async def _dispatch_fix(
        self,
        file_path: str,
        fix_context: dict[str, Any],
        *,
        label: str = "Checkpoint",
    ) -> bool:
        """Dispatch a targeted FIX_CODE task outside the per-file main loop."""
        from core.context_builder import ContextBuilder
        from core.observability import record_agent_end, record_agent_start

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
                api_contract=self._am._api_contract,
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

    # ── Build helpers ───────────────────────────────────────────────────

    async def _get_build_lock(self, file_path: str = "__global__") -> asyncio.Lock:
        """Get or create a per-module build lock.

        Files are grouped by their top-level directory (e.g. 'src/', 'lib/').
        This allows independent modules to build concurrently while still
        serializing builds within the same module to avoid conflicts.
        """
        # Derive module key from first path component
        parts = file_path.replace("\\", "/").split("/")
        module_key = parts[0] if len(parts) > 1 else "__root__"
        return await self._build_locks._get(module_key)

    async def _run_incremental_build(self, file_path: str = "__global__") -> CheckpointResult:
        """Delegate to CheckpointExecutor."""
        return await self._ensure_checkpoints().run_incremental_build(file_path)

    async def _run_full_build(self) -> CheckpointResult:
        """Delegate to CheckpointExecutor."""
        return await self._ensure_checkpoints().run_full_build()

    async def _global_build(self, engine: LifecycleEngine) -> bool:
        """Delegate to CheckpointExecutor."""
        return await self._ensure_checkpoints().run_global_build(engine)

    def _save_lifecycle_state(self, engine: LifecycleEngine) -> None:
        """Persist lifecycle state after each tier for --resume support."""
        try:
            state_path = str(self._am.repo.workspace / ".pipeline_state.json")
            engine.save_state(state_path)
        except Exception:
            logger.debug("Failed to save lifecycle state (non-critical)", exc_info=True)

    async def _fire_hook(self, event: str, **kwargs: Any) -> None:
        """Fire a hook event if a HookRegistry is configured."""
        if self._hooks is None:
            return
        try:
            from core.hooks import HookEvent
            await self._hooks.fire(HookEvent(event), **kwargs)
        except Exception:
            logger.debug("Hook fire failed for %s (non-critical)", event, exc_info=True)

    @staticmethod
    def _build_resolution_summary(error_history: list[dict[str, Any]]) -> str:
        if not error_history:
            return "Build passed without prior recorded errors"
        latest = error_history[-1]
        return (
            "Resolved build failure "
            f"{latest.get('error_hash', 'unknown')} after {len(error_history)} recorded build iteration(s)"
        )

    # ── Light validation (no LLM) ──────────────────────────────────────

    async def _lint_check(self, file_path: str) -> bool:
        """Run linter / type checker for interpreted languages.

        Returns True if no errors detected (or no linter available).
        This replaces the ReviewerAgent for interpreted languages at zero
        LLM cost.
        """
        lint_cmd = self._lang.lint_command if hasattr(self._lang, "lint_command") else None
        if not lint_cmd:
            # No linter configured — auto-pass
            return True

        try:
            result = await self._am.build_terminal.run_command(
                f"{lint_cmd} {file_path}", timeout=60,
            )
            return result.exit_code == 0
        except Exception:
            logger.debug("[%s] Lint check failed (non-critical)", file_path)
            return True  # Don't block on linter failures

    # ── Test phase ──────────────────────────────────────────────────────

    async def _run_test_phase(self, engine: LifecycleEngine) -> None:
        """Run test generation for files that reached TESTING phase.

        Delegates to AgentManager's lifecycle phase execution for test
        generation, keeping test logic consistent with the existing system.
        """
        testable = engine.get_files_in_phase(FilePhase.TESTING)
        if not testable:
            return

        logger.info("=== Test Generation: %d files ===", len(testable))
        semaphore = asyncio.Semaphore(self._settings.max_concurrent_agents)

        async def run_test(path: str) -> None:
            async with semaphore:
                try:
                    await asyncio.wait_for(
                        self._am._execute_lifecycle_phase(
                            engine, path, FilePhase.TESTING,
                        ),
                        timeout=float(self._settings.phase_timeout_seconds),
                    )
                except asyncio.TimeoutError:
                    logger.error("[%s] Test generation timed out", path)
                    try:
                        engine.process_event(path, EventType.RETRIES_EXHAUSTED)
                    except Exception:
                        engine.get_lifecycle(path).phase = FilePhase.FAILED
                except Exception:
                    logger.exception("[%s] Test generation failed", path)

        await asyncio.gather(
            *[run_test(p) for p in testable],
            return_exceptions=True,
        )

    # ── Post-generation checkpoints ───────────────────────────────────

    async def _run_security_checkpoint(
        self,
        sec_def: SecurityCheckpointDef,
    ) -> dict[str, Any]:
        """Delegate to CheckpointExecutor."""
        return await self._ensure_checkpoints().run_security_checkpoint(sec_def)

    async def _run_security_scan(self) -> TaskResult | None:
        """Delegate to CheckpointExecutor."""
        return await self._ensure_checkpoints()._run_security_scan()

    async def _run_integration_checkpoint(
        self,
        int_def: IntegrationCheckpointDef,
    ) -> dict[str, Any]:
        """Delegate to CheckpointExecutor."""
        return await self._ensure_checkpoints().run_integration_checkpoint(int_def)

    async def _run_integration_test_agent(self) -> TaskResult | None:
        """Delegate to CheckpointExecutor."""
        return await self._ensure_checkpoints()._run_integration_test_agent()

    async def _triage_integration_failure(
        self,
        test_file: str,
        test_output: str,
    ) -> dict[str, Any]:
        """Delegate to CheckpointExecutor."""
        return await self._ensure_checkpoints()._triage_integration_failure(test_file, test_output)
