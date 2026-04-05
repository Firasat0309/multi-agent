"""Checkpoint executor — orchestrates build, security, and integration verification.

Extracted from SimpleLoopExecutor to isolate checkpoint orchestration
logic from the core generate → build → fix loop.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any, Callable, Awaitable, TYPE_CHECKING

from core.checkpoint import BuildCheckpoint, CheckpointResult
from core.error_attributor import CompilerErrorAttributor
from core.event_bus import AgentEvent, BusEventType
from core.keyed_lock import KeyedLock
from core.models import AgentContext, Task, TaskResult, TaskType
from core.state_machine import EventType, FilePhase, LifecycleEngine

if TYPE_CHECKING:
    from config.settings import Settings
    from core.agent_manager import AgentManager
    from core.event_bus import EventBus
    from core.hooks import HookRegistry
    from core.language import LanguageProfile
    from core.pipeline_definition import (
        IntegrationCheckpointDef,
        SecurityCheckpointDef,
    )

logger = logging.getLogger(__name__)

__all__ = ["CheckpointExecutor"]

# Type alias for the fix dispatch callback injected from the executor.
FixDispatcher = Callable[[str, dict[str, Any], str], Awaitable[None]]


class CheckpointExecutor:
    """Orchestrates build checkpoints and post-generation verification phases.

    Responsibilities:
      - Incremental (type-check) and full repo builds
      - Global build with cross-file fix attribution
      - Security hardening checkpoint (scan → fix → rebuild cycles)
      - Integration test checkpoint (generate → triage → fix cycles)

    This class does NOT own agents or the generate/fix loop — it receives
    a ``dispatch_fix`` callback for delegating code repairs back to the
    parent executor.
    """

    def __init__(
        self,
        agent_manager: AgentManager,
        settings: Settings,
        lang_profile: LanguageProfile,
        attributor: CompilerErrorAttributor,
        build_locks: KeyedLock,
        dispatch_fix: FixDispatcher,
        fix_file: Callable[..., Awaitable[bool]],
        *,
        event_bus: EventBus | None = None,
        hooks: HookRegistry | None = None,
    ) -> None:
        self._am = agent_manager
        self._settings = settings
        self._lang = lang_profile
        self._attributor = attributor
        self._build_locks = build_locks
        self._dispatch_fix = dispatch_fix
        self._fix_file = fix_file
        self._event_bus = event_bus
        self._hooks = hooks
        self._compiled: bool = bool(lang_profile.build_command)

    # ── Hook helper ──────────────────────────────────────────────────────────

    async def _fire_hook(self, event_name: str, **kwargs: Any) -> None:
        if self._hooks is None:
            return
        try:
            from core.hooks import HookEvent
            event = HookEvent(event_name)
            await self._hooks.fire(event, **kwargs)
        except (ValueError, Exception):
            pass

    async def _publish_event(
        self, event_type: BusEventType, *, file_path: str = "", **data: Any,
    ) -> None:
        """Publish an event on the event bus (if wired)."""
        if self._event_bus is None:
            return
        await self._event_bus.publish(
            AgentEvent(type=event_type, file_path=file_path, data=data),
        )

    async def _get_build_lock(self, file_path: str = "__global__") -> asyncio.Lock:
        parts = file_path.replace("\\", "/").split("/")
        module_key = parts[0] if len(parts) > 1 else "__root__"
        return await self._build_locks._get(module_key)

    # ── Build checkpoints ────────────────────────────────────────────────────

    async def run_incremental_build(self, file_path: str = "__global__") -> CheckpointResult:
        """Run the lightest safe verification command for the current language."""
        build_command = self._lang.type_check_command or self._lang.build_command
        if not build_command:
            return CheckpointResult(passed=True, attempt=0)

        known_files = {fb.path for fb in self._am.blueprint.file_blueprints}

        await self._fire_hook("pre_build_checkpoint", file_path=file_path, build_type="incremental")

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
            result = await checkpoint.run_once(attempt=1)

        await self._fire_hook(
            "post_build_checkpoint",
            file_path=file_path,
            build_type="incremental",
            passed=result.passed,
        )
        if result.passed:
            await self._publish_event(
                BusEventType.BUILD_PASSED,
                file_path=file_path,
                build_type="incremental",
            )
        else:
            await self._publish_event(
                BusEventType.BUILD_FAILED,
                file_path=file_path,
                build_type="incremental",
                output=result.raw_output[:500] if result.raw_output else "",
            )
        return result

    async def run_full_build(self) -> CheckpointResult:
        """Run the full repo build command as the final correctness gate."""
        build_command = self._lang.build_command
        if not build_command:
            return CheckpointResult(passed=True, attempt=0)

        known_files = {fb.path for fb in self._am.blueprint.file_blueprints}

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

    async def run_global_build(self, engine: LifecycleEngine) -> bool:
        """Run one global build as a final sanity check.

        If it fails, attempt to fix cross-file errors once.
        """
        logger.info("=== Global Build (final check) ===")

        build_result = await self.run_full_build()

        if build_result.passed:
            logger.info("Global build passed ✓")
            await self._publish_event(BusEventType.BUILD_PASSED, build_type="global")
            for path in list(engine._lifecycles.keys()):
                lc = engine.get_lifecycle(path)
                if lc.phase == FilePhase.BUILDING:
                    engine.process_event(path, EventType.BUILD_PASSED)
            return True

        logger.warning("Global build failed — attempting cross-file fixes")
        await self._publish_event(
            BusEventType.BUILD_FAILED,
            build_type="global",
            output=build_result.raw_output[:500] if build_result.raw_output else "",
        )

        attribution = self._attributor.attribute(
            build_result.raw_output,
            known_files=set(engine._lifecycles.keys()),
        )

        if not attribution.affected_files:
            logger.error(
                "Global build failed but no errors attributed to files. Raw output: %s",
                build_result.raw_output[:500],
            )
            return False

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

            retry_result = await self.run_full_build()
            if retry_result.passed:
                logger.info("Global build passed after cross-file fixes ✓")
                await self._publish_event(BusEventType.BUILD_PASSED, build_type="global_retry")
                for path in list(engine._lifecycles.keys()):
                    lc = engine.get_lifecycle(path)
                    if lc.phase == FilePhase.BUILDING:
                        engine.process_event(path, EventType.BUILD_PASSED)
                return True

        logger.error("Global build still failing after cross-file fix attempt")
        return False

    # ── Security checkpoint ──────────────────────────────────────────────────

    async def run_security_checkpoint(
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
            logger.info("[SecurityCheckpoint] Cycle %d/%d — running scan", cycle, sec_def.max_cycles)

            scan_result = await self._run_security_scan()
            if scan_result is None:
                logger.warning("[SecurityCheckpoint] Security scan returned no result")
                break

            vulns = scan_result.metrics.get("vulnerabilities", [])
            critical_high = [v for v in vulns if v.get("severity") in ("critical", "high")]
            remaining_critical_high = len(critical_high)

            if not critical_high:
                logger.info("[SecurityCheckpoint] No critical/high vulnerabilities — PASSED (cycle %d)", cycle)
                if self._event_bus:
                    await self._event_bus.publish(AgentEvent(
                        type=BusEventType.TASK_COMPLETED,
                        task_type=TaskType.SECURITY_SCAN.value,
                        file_path="*",
                        agent_name="security_checkpoint",
                        data={"cycle": cycle, "total_vulns": len(vulns)},
                    ))
                return {
                    "passed": True, "cycles": cycle, "files_fixed": files_fixed,
                    "remaining_critical_high": 0, "total_vulnerabilities": len(vulns),
                }

            logger.warning("[SecurityCheckpoint] %d critical/high vulnerabilities found", len(critical_high))

            vulns_by_file: dict[str, list[dict[str, Any]]] = {}
            for vuln in critical_high:
                fp = vuln.get("file", "")
                if fp:
                    vulns_by_file.setdefault(fp, []).append(vuln)

            fix_tasks: list[asyncio.Task[None]] = []
            semaphore = asyncio.Semaphore(self._settings.max_concurrent_agents)

            for file_path, file_vulns in vulns_by_file.items():
                prior_fixes = fix_attempt_counts.get(file_path, 0)
                if prior_fixes >= sec_def.max_fixes_per_file:
                    logger.warning("[SecurityCheckpoint] Skipping %s — already fixed %d times", file_path, prior_fixes)
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
                        await self._dispatch_fix(fp, ctx, "SecurityCheckpoint")

                fix_tasks.append(asyncio.create_task(_fix_security(file_path, fix_context)))
                fix_attempt_counts[file_path] = prior_fixes + 1
                if file_path not in files_fixed:
                    files_fixed.append(file_path)

            if fix_tasks:
                await asyncio.gather(*fix_tasks, return_exceptions=True)
                total_vulns_fixed += len(fix_tasks)

                if self._compiled and self._lang.build_command:
                    logger.info("[SecurityCheckpoint] Rebuilding to verify security fixes compile")
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
                        logger.warning("[SecurityCheckpoint] Security fixes broke the build")
                        for file_path in rebuild_result.affected_files:
                            build_ctx = build_ck.get_fix_context_for_file(file_path, rebuild_result)
                            build_ctx["fix_trigger"] = "security_rebuild"
                            await self._dispatch_fix(file_path, build_ctx, "SecurityRebuild")
            else:
                logger.warning("[SecurityCheckpoint] No fixable files remaining — accepting findings as known risk")
                break

        logger.warning(
            "[SecurityCheckpoint] Exhausted %d cycles — %d files fixed",
            sec_def.max_cycles, len(files_fixed),
        )
        return {
            "passed": False, "cycles": sec_def.max_cycles, "files_fixed": files_fixed,
            "remaining_critical_high": remaining_critical_high, "total_vulns_fixed": total_vulns_fixed,
        }

    async def _run_security_scan(self) -> TaskResult | None:
        """Execute the SecurityAgent and return its TaskResult."""
        from core.context_builder import ContextBuilder
        from core.observability import record_agent_end, record_agent_start

        task = Task(
            task_id=0, task_type=TaskType.SECURITY_SCAN, file="*",
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

    # ── Integration test checkpoint ──────────────────────────────────────────

    async def run_integration_checkpoint(
        self,
        int_def: IntegrationCheckpointDef,
    ) -> dict[str, Any]:
        """Run the integration test checkpoint against the final workspace."""
        logger.info("=== Integration Test Checkpoint ===")

        files_fixed: list[str] = []
        test_file = ""

        for cycle in range(1, int_def.max_cycles + 1):
            logger.info("[IntegrationCheckpoint] Cycle %d/%d", cycle, int_def.max_cycles)

            test_result = await self._run_integration_test_agent()
            if test_result is None:
                logger.warning("[IntegrationCheckpoint] Integration test agent returned no result")
                break

            test_file = test_result.metrics.get("integration_test_file", "")
            if test_result.success:
                logger.info("[IntegrationCheckpoint] Integration tests PASSED (cycle %d)", cycle)
                if self._event_bus:
                    await self._event_bus.publish(AgentEvent(
                        type=BusEventType.TEST_PASSED,
                        task_type=TaskType.GENERATE_INTEGRATION_TEST.value,
                        file_path=test_file,
                        agent_name="integration_checkpoint",
                        data={"cycle": cycle},
                    ))
                return {"passed": True, "cycles": cycle, "test_file": test_file, "files_fixed": files_fixed}

            test_output = test_result.metrics.get("integration_test_output", "")
            if not test_output and test_result.errors:
                test_output = "\n".join(test_result.errors)
            if not test_output:
                logger.warning("[IntegrationCheckpoint] No test output to triage")
                break

            triage = await self._triage_integration_failure(test_file, test_output)
            fix_target = triage.get("fix_target", "test")
            source_files = triage.get("source_files", [])

            if fix_target == "source" and source_files:
                logger.info("[IntegrationCheckpoint] Triage: source bug in %s", source_files)
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
                            await self._dispatch_fix(fp, ctx, "IntegrationCheckpoint")

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
                        max_retries=1, timeout=300,
                        checkpoint_name="integration_rebuild",
                    )
                    rebuild = await rebuild_ck.run_once(attempt=1)
                    if not rebuild.passed:
                        logger.warning("[IntegrationCheckpoint] Source fix broke build")
                        for file_path in rebuild.affected_files:
                            build_ctx = rebuild_ck.get_fix_context_for_file(file_path, rebuild)
                            build_ctx["fix_trigger"] = "integration_rebuild"
                            await self._dispatch_fix(file_path, build_ctx, "IntegrationRebuild")
            else:
                logger.info("[IntegrationCheckpoint] Triage: test bug — rewriting test file")
                fix_ctx_test: dict[str, Any] = {
                    "fix_trigger": "integration_test_code",
                    "test_errors": test_output[:4000],
                    "test_file": test_file,
                    "fix_attempt": cycle,
                    "max_fix_attempts": int_def.max_cycles,
                }
                await self._dispatch_fix(test_file, fix_ctx_test, "IntegrationCheckpoint")
                if test_file not in files_fixed:
                    files_fixed.append(test_file)

        logger.warning("[IntegrationCheckpoint] Exhausted %d cycles", int_def.max_cycles)
        if self._event_bus:
            await self._event_bus.publish(AgentEvent(
                type=BusEventType.TEST_FAILED,
                task_type=TaskType.GENERATE_INTEGRATION_TEST.value,
                file_path=test_file,
                agent_name="integration_checkpoint",
                data={"cycles": int_def.max_cycles},
            ))
        return {"passed": False, "cycles": int_def.max_cycles, "test_file": test_file, "files_fixed": files_fixed}

    async def _run_integration_test_agent(self) -> TaskResult | None:
        """Execute the IntegrationTestAgent and return its TaskResult."""
        from core.context_builder import ContextBuilder
        from core.observability import record_agent_end, record_agent_start

        task = Task(
            task_id=0, task_type=TaskType.GENERATE_INTEGRATION_TEST,
            file="tests/integration/", description="Generate and run integration tests",
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
            logger.info("[IntegrationCheckpoint] Triage result: %s — %s", fix_target, reasoning)
            return {"fix_target": fix_target, "source_files": source_files_to_fix, "reasoning": reasoning}
        except Exception:
            logger.warning("[IntegrationCheckpoint] Triage LLM call failed — defaulting to test fix")
            return {"fix_target": "test", "source_files": []}
