"""DAG-driven executor — per-file dependency scheduling for max parallelism.

Replaces the tier-sequential execution model with a DAG scheduler that
starts each file as soon as all its dependencies have completed.  This
eliminates artificial waiting within tiers and between tiers.

Example: in a 20-file project with tiers [models, repos, services, controllers]:
  - Tier executor: 4 sequential rounds even if tier 2 has only 1 file
  - DAG executor: a service starts as soon as its 2 model deps pass

Gated behind the ``DAG_EXECUTOR`` feature flag.

The DAG executor delegates per-file processing to the existing
``SimpleLoopExecutor._process_file`` method, preserving all existing
build-fix loop logic.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from typing import Any, Callable, Awaitable, TYPE_CHECKING

if TYPE_CHECKING:
    from core.state_machine import LifecycleEngine
    from core.models import RepositoryBlueprint

logger = logging.getLogger(__name__)


class DAGExecutor:
    """Schedules file processing based on dependency DAG for maximum parallelism.

    Each file is an asyncio task that waits on an ``asyncio.Event`` per
    dependency.  When a file completes (pass or fail), it signals its event
    so dependents can proceed.

    When ``speculative=True``, files with incomplete (but non-failed) deps
    can start generation speculatively using blueprint info.  If deps later
    fail or actual exports diverge from assumptions, speculative output is
    discarded and the file is regenerated normally.

    Args:
        max_concurrency: Maximum number of files processing simultaneously.
        process_fn: Async callable ``(file_path: str) -> bool`` that processes
            a file and returns True on success.
        speculative: Enable speculative execution (requires blueprint).
        blueprint: Repository blueprint for speculative prompt construction.
        speculative_generate_fn: Async callable ``(file_path, prompt) -> str``
            that generates speculative content from a blueprint-based prompt.
        write_fn: Callable ``(file_path, content) -> None`` that writes a file.
        read_fn: Callable ``(file_path) -> str`` that reads generated file content.
    """

    def __init__(
        self,
        max_concurrency: int,
        process_fn: Callable[[str], Awaitable[bool]],
        *,
        speculative: bool = False,
        blueprint: "RepositoryBlueprint | None" = None,
        speculative_generate_fn: Callable[[str, str], Awaitable[str]] | None = None,
        write_fn: Callable[[str, str], None] | None = None,
        read_fn: Callable[[str], str] | None = None,
    ) -> None:
        self._max_concurrency = max_concurrency
        self._process_fn = process_fn
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._speculative = speculative and blueprint is not None
        self._blueprint = blueprint
        self._speculative_generate_fn = speculative_generate_fn
        self._write_fn = write_fn
        self._read_fn = read_fn

    async def execute(
        self,
        engine: "LifecycleEngine",
        all_files: list[str],
    ) -> dict[str, bool]:
        """Execute all files respecting dependency ordering.

        Returns a dict of ``{file_path: success}`` for each file.
        """
        start_time = time.monotonic()

        # Build dependency graph from the lifecycle engine
        dep_graph: dict[str, list[str]] = {}
        for fp in all_files:
            lc = engine.get_lifecycle(fp)
            deps = [d for d in lc.depends_on if d in set(all_files)]
            dep_graph[fp] = deps

        # Create completion events for each file
        events: dict[str, asyncio.Event] = {fp: asyncio.Event() for fp in all_files}
        results: dict[str, bool] = {}
        failed: set[str] = set()
        # Track speculative results keyed by file path
        speculative_cache: dict[str, Any] = {}

        async def _try_speculative(fp: str) -> bool:
            """Attempt speculative generation using blueprint info.

            Returns True if speculative content was generated and cached.
            Does NOT write the file — validation happens after deps finish.
            """
            if not self._speculative or not self._speculative_generate_fn:
                return False

            from core.speculative_exec import build_speculative_prompt, SpeculativeResult

            fb_map = {fb.path: fb for fb in self._blueprint.file_blueprints}
            file_bp = fb_map.get(fp)
            if file_bp is None or not file_bp.depends_on:
                return False

            try:
                prompt, assumed_exports = build_speculative_prompt(
                    file_bp, self._blueprint,
                )
                content = await self._speculative_generate_fn(fp, prompt)
                speculative_cache[fp] = SpeculativeResult(
                    file_path=fp,
                    content=content,
                    assumed_exports=assumed_exports,
                )
                logger.info("[DAG/Speculative] Generated speculative content for %s", fp)
                return True
            except Exception:
                logger.debug("[DAG/Speculative] Failed to generate speculative for %s", fp, exc_info=True)
                return False

        async def _validate_speculative(fp: str) -> bool:
            """Validate speculative output against actual dep contents.

            If valid, write the file and run build-fix. If invalid, discard.
            Returns True if speculative output was used successfully.
            """
            spec_result = speculative_cache.get(fp)
            if spec_result is None:
                return False

            # Read actual dependency contents
            actual_contents: dict[str, str] = {}
            if self._read_fn:
                for dep_path in spec_result.assumed_exports:
                    try:
                        actual_contents[dep_path] = self._read_fn(dep_path)
                    except Exception:
                        pass

            if spec_result.validate_against_actuals(actual_contents, self._blueprint):
                # Speculative output is valid — write it and process (build/fix only)
                if self._write_fn:
                    self._write_fn(fp, spec_result.content)
                logger.info("[DAG/Speculative] %s: speculative output VALID, proceeding to build", fp)
                return True
            else:
                logger.info("[DAG/Speculative] %s: speculative output INVALID, regenerating", fp)
                del speculative_cache[fp]
                return False

        async def _process_with_deps(fp: str) -> None:
            """Wait for dependencies, then process the file."""
            deps = dep_graph.get(fp, [])

            # --- Speculative path: start generation while deps are in-progress ---
            speculative_task = None
            if deps and self._speculative:
                # Check if any deps are still pending
                pending_deps = [d for d in deps if d in events and not events[d].is_set()]
                if pending_deps:
                    speculative_task = asyncio.create_task(_try_speculative(fp))

            # Wait for all dependencies to complete
            if deps:
                await asyncio.gather(*[events[d].wait() for d in deps if d in events])

                # Check if any dependency failed
                failed_deps = [d for d in deps if d in failed]
                if failed_deps:
                    logger.warning(
                        "[DAG] %s skipped — dependencies failed: %s",
                        fp, failed_deps,
                    )
                    results[fp] = False
                    failed.add(fp)
                    events[fp].set()
                    # Cancel speculative task if running
                    if speculative_task and not speculative_task.done():
                        speculative_task.cancel()
                    return

            # If we had a speculative task, await it and try to validate
            if speculative_task:
                try:
                    await speculative_task
                except asyncio.CancelledError:
                    pass

            # Process with concurrency limit
            async with self._semaphore:
                try:
                    # Try using speculative output first
                    if fp in speculative_cache:
                        used = await _validate_speculative(fp)
                        if used:
                            # Still need to build/fix — _process_fn handles that
                            success = await self._process_fn(fp)
                            results[fp] = success
                            if not success:
                                failed.add(fp)
                            return

                    # Normal path: generate from scratch
                    success = await self._process_fn(fp)
                    results[fp] = success
                    if not success:
                        failed.add(fp)
                except Exception:
                    logger.exception("[DAG] Unexpected error processing %s", fp)
                    results[fp] = False
                    failed.add(fp)
                finally:
                    events[fp].set()

        # Launch all files as concurrent tasks
        tasks = [asyncio.create_task(_process_with_deps(fp)) for fp in all_files]

        # Wait for all to complete
        await asyncio.gather(*tasks, return_exceptions=True)

        elapsed = time.monotonic() - start_time
        passed = sum(1 for v in results.values() if v)
        spec_used = sum(1 for s in speculative_cache.values() if s.valid)
        logger.info(
            "[DAG] Execution complete in %.1fs: %d passed, %d failed out of %d files"
            " (speculative: %d attempted, %d used)",
            elapsed, passed, len(results) - passed, len(results),
            len(speculative_cache), spec_used,
        )

        return results


def build_dag_from_tiers(
    engine: "LifecycleEngine",
    tiers: list[Any],
) -> list[str]:
    """Flatten tiers into a single file list ordered by dependencies.

    The DAG executor doesn't use tiers — it schedules based on the actual
    dependency graph.  This function extracts all files in tier order for
    the DAG executor to process.
    """
    all_files: list[str] = []
    seen: set[str] = set()
    for tier in tiers:
        for fp in tier.files:
            if fp not in seen:
                all_files.append(fp)
                seen.add(fp)
    return all_files
