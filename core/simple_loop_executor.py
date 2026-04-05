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
from core.error_attributor import (
    CompilerErrorAttributor,
    extract_error_lines,
    extract_referenced_files,
)
from core.event_bus import AgentEvent, BusEventType
from core.fast_validator import FastValidator
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
    ) -> None:
        self._am = agent_manager
        self._settings = settings
        self._lang = lang_profile
        self._compiled = bool(lang_profile.build_command)
        self._event_bus = event_bus
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
        _file_tokens = 0  # cumulative tokens spent on this file
        _file_token_budget = self._settings.execution.file_token_budget

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
                        review_text = (
                            "🔍 CODE REVIEW found semantic issues "
                            "(these may NOT show up as compiler errors):\n"
                            + "\n".join(f"  - {f}" for f in review_findings)
                        )
                        error_history.append({
                            "errors_text": review_text,
                            "error_hash": hashlib.md5(
                                review_text.encode(),
                            ).hexdigest()[:8],
                            "attempt": 0,
                            "referenced_files": [],
                        })
                    if lc.phase == FilePhase.REVIEWING:
                        # QUICK_REVIEW was off or review already passed
                        engine.process_event(file_path, EventType.REVIEW_PASSED)
                elif lc.phase == FilePhase.GENERATING:
                    # Rewrite rejected — will retry on next attempt
                    logger.warning("[%s] Rewrite rejected, retrying", file_path)
                    continue
            else:
                # Fix with accumulated context
                fix_ok = await self._fix_file(engine, file_path, error_history)
                fix_count += 1
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
                    error_history.append({
                        "errors_text": f"SYNTAX ERROR (pre-build):\n{syntax_errors}",
                        "error_hash": error_hash,
                        "attempt": attempt + 1,
                        "referenced_files": [],
                    })
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
                error_hash = hashlib.md5(errors_text.encode()).hexdigest()[:8]
                error_history.append({
                    "errors_text": errors_text,
                    "error_hash": error_hash,
                    "attempt": attempt + 1,
                })
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
                if fix_count > 0 or error_history:
                    last_hash = error_history[-1]["error_hash"] if error_history else ""
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
            if error_history and error_history[-1]["error_hash"] == error_hash:
                logger.warning(
                    "[%s] Same errors repeated after fix (hash=%s) — "
                    "will escalate on next attempt",
                    file_path, error_hash,
                )

            error_history.append({
                "errors_text": file_errors_text,
                "error_hash": error_hash,
                "attempt": attempt + 1,
                "referenced_files": referenced_files,
            })

            # ── Streaming fix: extract error signatures for early cancellation ──
            if feature("STREAMING_FIX"):
                from core.streaming_fix import extract_error_signature
                sigs = extract_error_signature(file_errors_text)
                if sigs:
                    error_history[-1]["error_signatures"] = sigs
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
            errors=[h["errors_text"][:200] for h in error_history[-2:]],
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

            # Mark tier complete and persist session state
            if self._session and not tier_failed:
                self._session.mark_tier_complete(tier.index)
            if self._session:
                try:
                    self._session.save()
                except Exception:
                    logger.debug("Failed to save session state (non-critical)", exc_info=True)

            self._save_lifecycle_state(engine)

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
        """
        if not feature("QUICK_REVIEW"):
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

        Builds rich fix metadata including:
        - Current build errors
        - Error history from previous attempts (memory)
        - Escalation flag when same error repeats
        """
        from core.context_builder import ContextBuilder
        from core.observability import record_agent_start, record_agent_end

        latest = error_history[-1] if error_history else {}
        errors_text = latest.get("errors_text", "")
        referenced_files = latest.get("referenced_files", [])
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

        # Smart retry: detect stalled progress (exact hash OR similar errors)
        escalate = False
        if len(error_history) >= 2:
            prev_hash = error_history[-2]["error_hash"]
            curr_hash = error_history[-1]["error_hash"]
            if curr_hash == prev_hash:
                # Exact same errors — definitely stalled
                escalate = True
            elif len(error_history) >= 3:
                # Check if we're oscillating between two error states
                # (fix A introduces error B, fix B reintroduces error A)
                recent_hashes = [h["error_hash"] for h in error_history[-3:]]
                if len(set(recent_hashes)) <= 2:
                    escalate = True
                    logger.warning(
                        "[%s] Oscillating between %d error states — escalating",
                        file_path, len(set(recent_hashes)),
                    )
            if not escalate:
                # Fuzzy check: if the error text shares >60% of its lines
                # with the previous attempt, treat as semantically similar
                prev_lines = set(error_history[-2].get("errors_text", "").splitlines())
                curr_lines = set(errors_text.splitlines())
                if prev_lines and curr_lines:
                    overlap = len(prev_lines & curr_lines)
                    total = max(len(prev_lines), len(curr_lines))
                    if total > 0 and overlap / total > 0.6:
                        escalate = True
                        logger.warning(
                            "[%s] Error similarity %.0f%% — escalating",
                            file_path, 100 * overlap / total,
                        )
            if escalate:
                # Strategy rotation: cycle through escalation approaches
                # based on how many consecutive stalled attempts we've seen.
                stall_count = 0
                for i in range(len(error_history) - 1, 0, -1):
                    if error_history[i]["error_hash"] == error_history[i - 1]["error_hash"]:
                        stall_count += 1
                    else:
                        break

                if stall_count >= 3:
                    # Strategy 3: Radical simplification
                    escalation_prefix = (
                        "⚠️ FIX LOOP STALLED ({n} identical attempts). "
                        "RADICAL SIMPLIFICATION REQUIRED:\n"
                        "- Strip the failing section to the SIMPLEST possible "
                        "implementation that compiles\n"
                        "- Use stub/TODO implementations for complex logic\n"
                        "- Remove any clever abstractions — use plain, direct code\n"
                        "- If a dependency is causing issues, remove it and "
                        "hard-code the value\n\n"
                    ).format(n=stall_count + 1)
                elif stall_count >= 2:
                    # Strategy 2: Complete method rewrite
                    escalation_prefix = (
                        "⚠️ FIX LOOP STALLED ({n} identical attempts). "
                        "REWRITE FROM SCRATCH:\n"
                        "- Do NOT patch the existing code — delete the failing "
                        "method/block entirely and rewrite it\n"
                        "- Re-read the dependency signatures in Related Files "
                        "and match them EXACTLY\n"
                        "- If the approach is fundamentally wrong, use an "
                        "alternative algorithm or pattern\n\n"
                    ).format(n=stall_count + 1)
                else:
                    # Strategy 1: Standard escalation (different approach)
                    escalation_prefix = (
                        "⚠️ PREVIOUS FIX ATTEMPT DID NOT RESOLVE THE ERRORS — "
                        "THE SAME BUILD ERRORS PERSIST.\n"
                        "You MUST try a COMPLETELY DIFFERENT approach:\n"
                        "- If you changed a method call, check the actual method "
                        "signature in the Related Files section\n"
                        "- If an import is wrong, check the actual package path\n"
                        "- If a type mismatch, read the full error to understand "
                        "which types are incompatible\n\n"
                    )

                errors_text = escalation_prefix + errors_text

        # Build error history summary for context (memory)
        history_summary = ""
        if len(error_history) >= 2:
            prev_attempts = error_history[:-1]
            history_lines = []
            for h in prev_attempts[-3:]:  # Last 3 attempts max
                history_lines.append(
                    f"  Attempt {h['attempt']}: {h['errors_text'][:300]}..."
                )
            history_summary = (
                "PREVIOUS FIX ATTEMPTS (learn from these — do NOT repeat the same fix):\n"
                + "\n".join(history_lines)
            )

        if persistent_memory:
            history_summary = (
                history_summary + "\n\n" if history_summary else ""
            ) + persistent_memory

        if cross_file_hints:
            history_summary = (
                history_summary + "\n\n" if history_summary else ""
            ) + cross_file_hints

        fix_metadata = {
            "build_errors": errors_text + ("\n\n" + history_summary if history_summary else ""),
            "fix_trigger": "build",
            "fix_attempt": latest.get("attempt", 1),
            "max_fix_attempts": self.MAX_ATTEMPTS,
        }
        if referenced_files:
            fix_metadata["referenced_files"] = referenced_files
        if escalate:
            fix_metadata["escalate_fix"] = True

        # Streaming fix: inject error signatures for early-cancel awareness
        if feature("STREAMING_FIX") and error_history:
            all_sigs = []
            for h in error_history:
                all_sigs.extend(h.get("error_signatures", []))
            if all_sigs:
                # Deduplicate while preserving order
                seen_sigs: set[str] = set()
                unique_sigs = []
                for s in all_sigs:
                    if s not in seen_sigs:
                        seen_sigs.add(s)
                        unique_sigs.append(s)
                fix_metadata["known_bad_patterns"] = unique_sigs[:5]
                fix_metadata["build_errors"] = (
                    fix_metadata["build_errors"]
                    + "\n\n⚠️ KNOWN BAD PATTERNS (do NOT reproduce these in your fix):\n"
                    + "\n".join(f"  • {s}" for s in unique_sigs[:5])
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
        """Run the lightest safe verification command for the current language."""
        build_command = self._lang.type_check_command or self._lang.build_command
        if not build_command:
            return CheckpointResult(passed=True, attempt=0)

        known_files = set()
        for fb in self._am.blueprint.file_blueprints:
            known_files.add(fb.path)

        lock = await self._get_build_lock(file_path)
        async with lock:
            checkpoint = BuildCheckpoint(
                build_command=build_command,
                terminal=self._am.build_terminal,
                attributor=self._attributor,
                known_files=known_files,
                max_retries=1,
                timeout=max(120, 30 * len(known_files)),
                checkpoint_name="simple_loop_incremental",
            )

            return await checkpoint.run_once(attempt=1)

    async def _run_full_build(self) -> CheckpointResult:
        """Run the full repo build command as the final correctness gate."""
        build_command = self._lang.build_command
        if not build_command:
            return CheckpointResult(passed=True, attempt=0)

        known_files = set()
        for fb in self._am.blueprint.file_blueprints:
            known_files.add(fb.path)

        lock = await self._get_build_lock("__global__")

        async with lock:
            checkpoint = BuildCheckpoint(
                build_command=build_command,
                terminal=self._am.build_terminal,
                attributor=self._attributor,
                known_files=known_files,
                max_retries=1,
                timeout=max(180, 60 * len(known_files)),
                checkpoint_name="simple_loop_full",
            )

            return await checkpoint.run_once(attempt=1)

    async def _global_build(self, engine: LifecycleEngine) -> bool:
        """Run one global build as a final sanity check.

        If it fails, attempt to fix cross-file errors once.
        """
        logger.info("=== Global Build (final check) ===")

        build_result = await self._run_full_build()

        if build_result.passed:
            logger.info("Global build passed ✓")
            # Advance any BUILDING files to TESTING
            for path in list(engine._lifecycles.keys()):
                lc = engine.get_lifecycle(path)
                if lc.phase == FilePhase.BUILDING:
                    engine.process_event(path, EventType.BUILD_PASSED)
            return True

        # Global build failed — attempt one round of cross-file fixes
        logger.warning("Global build failed — attempting cross-file fixes")

        attribution = self._attributor.attribute(
            build_result.raw_output,
            known_files=set(engine._lifecycles.keys()),
        )

        if not attribution.affected_files:
            logger.error(
                "Global build failed but no errors attributed to files. "
                "Raw output: %s",
                build_result.raw_output[:500],
            )
            return False

        # Fix affected files in parallel
        semaphore = asyncio.Semaphore(self._settings.max_concurrent_agents)
        fix_tasks = []

        for file_path in attribution.affected_files:
            lc = engine.get_lifecycle(file_path)
            if lc.is_terminal and lc.phase != FilePhase.PASSED:
                continue

            errors_text = attribution.summary_for_file(file_path)
            if not errors_text:
                continue

            error_history = [{
                "errors_text": errors_text,
                "error_hash": hashlib.md5(errors_text.encode()).hexdigest()[:8],
                "attempt": 1,
            }]

            async def _fix(fp: str, hist: list[dict[str, Any]]) -> bool:
                async with semaphore:
                    if engine.get_lifecycle(fp).phase == FilePhase.BUILDING:
                        engine.process_event(fp, EventType.BUILD_FAILED,
                                             data={"errors": hist[-1]["errors_text"]})
                    return await self._fix_file(engine, fp, hist)

            fix_tasks.append(_fix(file_path, error_history))

        if fix_tasks:
            await asyncio.gather(*fix_tasks, return_exceptions=True)

            # Retry the global build
            retry_result = await self._run_full_build()
            if retry_result.passed:
                logger.info("Global build passed after cross-file fixes ✓")
                for path in list(engine._lifecycles.keys()):
                    lc = engine.get_lifecycle(path)
                    if lc.phase == FilePhase.BUILDING:
                        engine.process_event(path, EventType.BUILD_PASSED)
                return True

        logger.error("Global build still failing after cross-file fix attempt")
        return False

    def _save_lifecycle_state(self, engine: LifecycleEngine) -> None:
        """Persist lifecycle state after each tier for --resume support."""
        try:
            state_path = str(self._am.repo.workspace / ".pipeline_state.json")
            engine.save_state(state_path)
        except Exception:
            logger.debug("Failed to save lifecycle state (non-critical)", exc_info=True)

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
        """Run the security hardening checkpoint against the final workspace."""
        logger.info("=== Security Hardening Checkpoint ===")

        files_fixed: list[str] = []
        fix_attempt_counts: dict[str, int] = {}
        total_vulns_fixed = 0
        remaining_critical_high = 0

        for cycle in range(1, sec_def.max_cycles + 1):
            logger.info(
                "[SecurityCheckpoint] Cycle %d/%d — running scan",
                cycle,
                sec_def.max_cycles,
            )

            scan_result = await self._run_security_scan()
            if scan_result is None:
                logger.warning("[SecurityCheckpoint] Security scan returned no result")
                break

            vulns = scan_result.metrics.get("vulnerabilities", [])
            critical_high = [
                vuln for vuln in vulns
                if vuln.get("severity") in ("critical", "high")
            ]
            remaining_critical_high = len(critical_high)

            if not critical_high:
                logger.info(
                    "[SecurityCheckpoint] No critical/high vulnerabilities — "
                    "security checkpoint PASSED (cycle %d)",
                    cycle,
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

            vulns_by_file: dict[str, list[dict[str, Any]]] = {}
            for vuln in critical_high:
                file_path = vuln.get("file", "")
                if file_path:
                    vulns_by_file.setdefault(file_path, []).append(vuln)

            fix_tasks: list[asyncio.Task[None]] = []
            semaphore = asyncio.Semaphore(self._settings.max_concurrent_agents)

            for file_path, file_vulns in vulns_by_file.items():
                prior_fixes = fix_attempt_counts.get(file_path, 0)
                if prior_fixes >= sec_def.max_fixes_per_file:
                    logger.warning(
                        "[SecurityCheckpoint] Skipping %s — already fixed %d times",
                        file_path,
                        prior_fixes,
                    )
                    continue

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

                fix_tasks.append(asyncio.create_task(_fix_security(file_path, fix_context)))
                fix_attempt_counts[file_path] = prior_fixes + 1
                if file_path not in files_fixed:
                    files_fixed.append(file_path)

            if fix_tasks:
                await asyncio.gather(*fix_tasks, return_exceptions=True)
                total_vulns_fixed += len(fix_tasks)

                if self._compiled and self._lang.build_command:
                    logger.info(
                        "[SecurityCheckpoint] Rebuilding to verify security fixes compile",
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
                            "dispatching build fixes",
                        )
                        for file_path in rebuild_result.affected_files:
                            build_ctx = build_ck.get_fix_context_for_file(
                                file_path,
                                rebuild_result,
                            )
                            build_ctx["fix_trigger"] = "security_rebuild"
                            await self._dispatch_fix(
                                file_path,
                                build_ctx,
                                label="SecurityRebuild",
                            )
            else:
                logger.warning(
                    "[SecurityCheckpoint] No fixable files remaining — "
                    "accepting %d findings as known risk",
                    len(critical_high),
                )
                break

        logger.warning(
            "[SecurityCheckpoint] Security checkpoint exhausted %d cycles — "
            "%d files fixed, remaining findings accepted as known risk",
            sec_def.max_cycles,
            len(files_fixed),
        )
        return {
            "passed": False,
            "cycles": sec_def.max_cycles,
            "files_fixed": files_fixed,
            "remaining_critical_high": remaining_critical_high,
            "total_vulns_fixed": total_vulns_fixed,
        }

    async def _run_security_scan(self) -> TaskResult | None:
        """Execute the SecurityAgent and return its TaskResult."""
        from core.context_builder import ContextBuilder
        from core.observability import record_agent_end, record_agent_start

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
                api_contract=self._am._api_contract,
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

            agent_name = agent.role.value
            if agent_name not in self._am._metrics["agent_metrics"]:
                self._am._metrics["agent_metrics"][agent_name] = []
            self._am._metrics["agent_metrics"][agent_name].append(agent.get_metrics())
            return result
        except Exception:
            logger.exception("[SecurityCheckpoint] Security scan agent failed")
            return None

    async def _run_integration_checkpoint(
        self,
        int_def: IntegrationCheckpointDef,
    ) -> dict[str, Any]:
        """Run the integration test checkpoint against the final workspace."""
        logger.info("=== Integration Test Checkpoint ===")

        files_fixed: list[str] = []
        test_file = ""

        for cycle in range(1, int_def.max_cycles + 1):
            logger.info(
                "[IntegrationCheckpoint] Cycle %d/%d — generating and running tests",
                cycle,
                int_def.max_cycles,
            )

            test_result = await self._run_integration_test_agent()
            if test_result is None:
                logger.warning(
                    "[IntegrationCheckpoint] Integration test agent returned no result",
                )
                break

            test_file = test_result.metrics.get("integration_test_file", "")
            if test_result.success:
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

            test_output = test_result.metrics.get("integration_test_output", "")
            if not test_output and test_result.errors:
                test_output = "\n".join(test_result.errors)

            if not test_output:
                logger.warning(
                    "[IntegrationCheckpoint] No test output to triage — skipping fix",
                )
                break

            triage = await self._triage_integration_failure(test_file, test_output)
            fix_target = triage.get("fix_target", "test")
            source_files = triage.get("source_files", [])

            if fix_target == "source" and source_files:
                logger.info(
                    "[IntegrationCheckpoint] Triage: source bug in %s — dispatching source fix",
                    source_files,
                )
                semaphore = asyncio.Semaphore(self._settings.max_concurrent_agents)
                fix_tasks: list[asyncio.Task[None]] = []

                for file_path in source_files:
                    fix_ctx: dict[str, Any] = {
                        "fix_trigger": "integration_test",
                        "test_errors": test_output[:4000],
                        "test_file": test_file,
                        "fix_attempt": cycle,
                        "max_fix_attempts": int_def.max_cycles,
                    }

                    async def _fix_source(fp: str, ctx: dict[str, Any]) -> None:
                        async with semaphore:
                            await self._dispatch_fix(
                                fp,
                                ctx,
                                label="IntegrationCheckpoint",
                            )

                    fix_tasks.append(asyncio.create_task(_fix_source(file_path, fix_ctx)))
                    if file_path not in files_fixed:
                        files_fixed.append(file_path)

                if fix_tasks:
                    await asyncio.gather(*fix_tasks, return_exceptions=True)

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
                            "dispatching build fixes",
                        )
                        for file_path in rebuild.affected_files:
                            build_ctx = rebuild_ck.get_fix_context_for_file(
                                file_path,
                                rebuild,
                            )
                            build_ctx["fix_trigger"] = "integration_rebuild"
                            await self._dispatch_fix(
                                file_path,
                                build_ctx,
                                label="IntegrationRebuild",
                            )
            else:
                logger.info(
                    "[IntegrationCheckpoint] Triage: test bug — rewriting test file",
                )
                fix_ctx_test: dict[str, Any] = {
                    "fix_trigger": "integration_test_code",
                    "test_errors": test_output[:4000],
                    "test_file": test_file,
                    "fix_attempt": cycle,
                    "max_fix_attempts": int_def.max_cycles,
                }
                await self._dispatch_fix(
                    test_file,
                    fix_ctx_test,
                    label="IntegrationCheckpoint",
                )
                if test_file not in files_fixed:
                    files_fixed.append(test_file)

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
        from core.observability import record_agent_end, record_agent_start

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
                api_contract=self._am._api_contract,
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
        """Use the LLM to classify integration failures as test or source bugs."""
        try:
            test_content = self._am.repo.read_file(test_file)
            if test_content is None:
                test_content = "(could not read test file)"
        except Exception:
            logger.debug("Could not read test file %s", test_file)
            test_content = "(could not read test file)"

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
                fix_target,
                reasoning,
            )
            return {
                "fix_target": fix_target,
                "source_files": source_files_to_fix,
                "reasoning": reasoning,
            }
        except Exception:
            logger.warning(
                "[IntegrationCheckpoint] Triage LLM call failed — defaulting to test fix",
            )
            return {"fix_target": "test", "source_files": []}
