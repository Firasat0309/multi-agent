"""Agent manager — thin coordination hub that wires agents, tools, and state.

Responsibilities kept here:
  - :meth:`__init__` — initialises all shared resources (LLM, terminals, metrics).
  - :meth:`_create_agent` — factory: maps a ``TaskType`` to the right agent class.
  - Public delegation shims — ``execute_graph``, ``execute_with_lifecycle``,
    ``execute_with_checkpoints`` — that preserve the historic public API while
    delegating implementation to the focused sub-modules:

      * :class:`~core.task_dispatcher.TaskDispatcher` — task-graph execution
      * :class:`~core.lifecycle_orchestrator.LifecycleOrchestrator` — per-file
        lifecycle FSM + global DAG hand-off

The static helpers ``_build_lifecycle_metadata`` and ``_extract_event_data`` are
kept as forwarding aliases so any code that calls ``AgentManager._build…``
continues to work without modification.
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

import asyncio

from core.state_machine import FilePhase, EventType
from core.models import Task
from core.context_builder import ContextBuilder
from core.event_bus import AgentEvent, BusEventType
from core.observability import record_agent_start, record_agent_end

from agents.architect_agent import ArchitectAgent
from agents.base_agent import BaseAgent
from agents.build_verifier_agent import BuildVerifierAgent
from agents.coder_agent import CoderAgent
from agents.deploy_agent import DeployAgent
from agents.integration_test_agent import IntegrationTestAgent
from agents.patch_agent import PatchAgent
from agents.reviewer_agent import ReviewerAgent
from agents.security_agent import SecurityAgent
from agents.test_agent import TestAgent
from agents.planner_agent import PlannerAgent
from agents.writer_agent import WriterAgent
# ── Fullstack / Frontend agents ───────────────────────────────────────────────
from agents.product_planner_agent import ProductPlannerAgent
from agents.api_contract_agent import APIContractAgent
from agents.design_parser_agent import DesignParserAgent
from agents.component_planner_agent import ComponentPlannerAgent
from agents.component_dag_agent import ComponentDAGAgent
from agents.component_generator_agent import ComponentGeneratorAgent
from agents.api_integration_agent import APIIntegrationAgent
from agents.state_management_agent import StateManagementAgent
from config.settings import Settings
from core.event_bus import EventBus
from core.file_lock_manager import FileLockManager
from core.language import detect_language_from_blueprint
from core.lifecycle_orchestrator import LifecycleOrchestrator
from core.llm_client import LLMClient
from core.models import (
    RepositoryBlueprint,
    TaskType,
)
from core.repository_manager import RepositoryManager
from core.state_machine import LifecycleEngine
from core.task_dispatcher import TaskDispatcher
from core.task_engine import TaskGraph
from core.tier_scheduler import Tier
from tools.terminal_tools import TerminalTools
from core.mcp_client import MCPClient

if TYPE_CHECKING:
    from core.live_console import LiveConsole
    from memory.dependency_graph import DependencyGraphStore
    from memory.embedding_store import EmbeddingStore
    from sandbox.sandbox_runner import SandboxManager

logger = logging.getLogger(__name__)

# Mapping from task type to agent class.  Consulted by _create_agent() and
# available to tests / callers that need to inspect the registry directly.
TASK_AGENT_MAP: dict[TaskType, type[BaseAgent]] = {
    TaskType.GENERATE_FILE: CoderAgent,
    TaskType.REVIEW_FILE: ReviewerAgent,
    TaskType.REVIEW_MODULE: ReviewerAgent,
    TaskType.REVIEW_ARCHITECTURE: ReviewerAgent,
    TaskType.GENERATE_TEST: TestAgent,
    TaskType.SECURITY_SCAN: SecurityAgent,
    TaskType.GENERATE_DEPLOY: DeployAgent,
    TaskType.GENERATE_DOCS: WriterAgent,
    TaskType.FIX_CODE: CoderAgent,
    TaskType.MODIFY_FILE: PatchAgent,
    TaskType.GENERATE_INTEGRATION_TEST: IntegrationTestAgent,
    TaskType.DESIGN_ARCHITECTURE: ArchitectAgent,
    TaskType.CREATE_PLAN: PlannerAgent,
    TaskType.VERIFY_BUILD: BuildVerifierAgent,
    # ── Fullstack / Frontend task types ──────────────────────────────────────
    TaskType.PLAN_PRODUCT: ProductPlannerAgent,
    TaskType.GENERATE_API_CONTRACT: APIContractAgent,
    TaskType.PARSE_DESIGN: DesignParserAgent,
    TaskType.PLAN_COMPONENTS: ComponentPlannerAgent,
    TaskType.BUILD_COMPONENT_DAG: ComponentDAGAgent,
    TaskType.GENERATE_COMPONENT: ComponentGeneratorAgent,
    TaskType.INTEGRATE_API: APIIntegrationAgent,
    TaskType.MANAGE_STATE: StateManagementAgent,
}


class AgentManager:
    """Thin coordination hub: initialises shared state and delegates execution.

    All long-running execution logic lives in :class:`~core.task_dispatcher.TaskDispatcher`
    and :class:`~core.lifecycle_orchestrator.LifecycleOrchestrator`.  The public
    method signatures below are preserved verbatim for backward compatibility.
    """

    def __init__(
        self,
        settings: Settings,
        llm_client: LLMClient,
        repo_manager: RepositoryManager,
        blueprint: RepositoryBlueprint,
        live_console: LiveConsole | None = None,
        sandbox_manager: SandboxManager | None = None,
        build_sandbox_id: str | None = None,
        test_sandbox_id: str | None = None,
        dep_store: DependencyGraphStore | None = None,
        embedding_store: EmbeddingStore | None = None,
        event_bus: EventBus | None = None,
        mcp_client: MCPClient | None = None,
    ) -> None:
        self.settings = settings
        self.llm = llm_client
        self.repo = repo_manager
        self.blueprint = blueprint
        self._live = live_console
        self._lang = detect_language_from_blueprint(blueprint.tech_stack)
        self._dep_store = dep_store
        self._embedding_store = embedding_store
        self._event_bus = event_bus
        self.mcp_client = mcp_client

        # Two-tier terminal tools: build (network-capable) and test (isolated).
        # TestAgent gets the test terminal; all other agents that need a
        # terminal (SecurityAgent, etc.) get the build terminal.
        self.build_terminal = TerminalTools(
            repo_manager.workspace,
            language=self._lang,
            sandbox_manager=sandbox_manager,
            sandbox_id=build_sandbox_id,
        )
        self.test_terminal = TerminalTools(
            repo_manager.workspace,
            language=self._lang,
            sandbox_manager=sandbox_manager,
            sandbox_id=test_sandbox_id,
        )
        if sandbox_manager and build_sandbox_id:
            logger.info(
                "Build commands routed to sandbox %s; test execution to sandbox %s",
                build_sandbox_id, test_sandbox_id,
            )
        self._metrics: dict[str, Any] = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_time": 0.0,
            "agent_metrics": {},
        }
        self._file_locks = FileLockManager()

        # Execution collaborators — owned here so each call doesn't re-allocate.
        self._dispatcher = TaskDispatcher(self)
        self._orchestrator = LifecycleOrchestrator(self)

    def _create_agent(self, task_type: TaskType) -> BaseAgent:
        agent_cls = TASK_AGENT_MAP.get(task_type)
        if agent_cls is None:
            raise ValueError(f"No agent registered for task type: {task_type}")

        # TestAgent / IntegrationTestAgent → test terminal (no-network sandbox).
        # SecurityAgent → build terminal (bandit needs the full env).
        if agent_cls in (TestAgent, IntegrationTestAgent):
            return agent_cls(
                llm_client=self.llm,
                repo_manager=self.repo,
                terminal=self.test_terminal,
                mcp_client=self.mcp_client,
            )
        if agent_cls is SecurityAgent:
            return agent_cls(
                llm_client=self.llm,
                repo_manager=self.repo,
                terminal=self.build_terminal,
                mcp_client=self.mcp_client,
            )
        if agent_cls is BuildVerifierAgent:
            return agent_cls(
                llm_client=self.llm,
                repo_manager=self.repo,
                terminal=self.build_terminal,
                mcp_client=self.mcp_client,
            )
        return agent_cls(llm_client=self.llm, repo_manager=self.repo, mcp_client=self.mcp_client)

    # ── Public execution API — all implementation lives in the collaborators ──

    async def execute_graph(self, task_graph: TaskGraph) -> dict[str, Any]:
        """Execute all tasks in *task_graph* respecting dependency order.

        Delegates to :class:`~core.task_dispatcher.TaskDispatcher`.
        """
        return await self._dispatcher.execute_graph(task_graph)

    async def _execute_task(self, task: Any, task_graph: TaskGraph) -> None:
        """Execute a single task — kept for backward compatibility with tests.

        Delegates to :class:`~core.task_dispatcher.TaskDispatcher`.
        """
        return await self._dispatcher._execute_task(task, task_graph)

    async def execute_with_lifecycle(
        self,
        engine: LifecycleEngine,
        global_graph: TaskGraph,
    ) -> dict[str, Any]:
        """Drive per-file lifecycle FSM then run the global DAG.

        Delegates to :class:`~core.lifecycle_orchestrator.LifecycleOrchestrator`.
        """
        return await self._orchestrator.execute_with_lifecycle(engine, global_graph)

    async def _execute_lifecycle_phase(
        self,
        engine: LifecycleEngine,
        file_path: str,
        phase: Any,
    ) -> None:
        """Execute one lifecycle phase for a single file.

        Kept as a dedicated hook for backward compatibility with tests.
        """
        lc = engine.get_lifecycle(file_path)

        # ── PENDING: fire DEPS_MET to move to GENERATING ──────────
        if phase == FilePhase.PENDING:
            engine.process_event(file_path, EventType.DEPS_MET)
            phase = lc.phase  # now GENERATING

        # --skip-reviewer: auto-pass review phases
        if phase == FilePhase.REVIEWING and "reviewer" in self.settings.skip_agents:
            logger.info("[%s] Skipping review (--skip-reviewer)", file_path)
            engine.process_event(file_path, EventType.REVIEW_PASSED)
            return

        # Map phase → task type + event on success/failure.
        # For GENERATING: use the lifecycle's generation_task_type which is
        # MODIFY_FILE for the Enhance pipeline, GENERATE_FILE for Generate.
        gen_task_type = TaskType(lc.generation_task_type)
        gen_verb = "Modify" if gen_task_type == TaskType.MODIFY_FILE else "Generate"

        phase_config: dict[FilePhase, dict[str, Any]] = {
            FilePhase.GENERATING: {
                "task_type": gen_task_type,
                "success_event": EventType.CODE_GENERATED,
                "failure_event": EventType.RETRIES_EXHAUSTED,
                "description": f"{gen_verb} {file_path}",
            },
            FilePhase.REVIEWING: {
                "task_type": TaskType.REVIEW_FILE,
                "success_event": EventType.REVIEW_PASSED,
                "failure_event": EventType.REVIEW_FAILED,
                "description": f"Review {file_path}",
            },
            FilePhase.FIXING: {
                "task_type": TaskType.FIX_CODE,
                "success_event": EventType.FIX_APPLIED,
                # If the fix agent fails, fire RETRIES_EXHAUSTED to stop the
                # cycle — the old FIX_APPLIED caused infinite loops where
                # broken code cycled until limits were hit and then was
                # incorrectly marked as PASSED.
                "failure_event": EventType.RETRIES_EXHAUSTED,
                "description": f"Fix {file_path} ({lc.fix_trigger} issues)",
            },
            FilePhase.BUILDING: {
                "task_type": TaskType.VERIFY_BUILD,
                "success_event": EventType.BUILD_PASSED,
                "failure_event": EventType.BUILD_FAILED,
                "description": f"Verify build for {file_path}",
            },
            FilePhase.TESTING: {
                "task_type": TaskType.GENERATE_TEST,
                "success_event": EventType.TEST_PASSED,
                "failure_event": EventType.TEST_FAILED,
                "description": f"Test {file_path}",
            },
        }

        config = phase_config.get(phase)
        if config is None:
            logger.warning("No action for phase %s on %s", phase.value, file_path)
            return

        # Build a synthetic Task for the context builder.
        # Merge lifecycle change_metadata (change_type, target_function etc.)
        # so PatchAgent/CoderAgent receives the full context.
        task_meta = self._build_lifecycle_metadata(lc)
        task_meta.update(lc.change_metadata)

        task = Task(
            task_id=0,
            task_type=config["task_type"],
            file=file_path,
            description=config["description"],
            metadata=task_meta,
        )

        if self._live:
            self._live.log(f"[cyan]Lifecycle:[/cyan] {config['description']}")

        # Build context and execute
        context_builder = ContextBuilder(
            workspace_dir=self.repo.workspace,
            blueprint=self.blueprint,
            repo_index=self.repo.get_repo_index(),
            dep_store=self._dep_store,
            embedding_store=self._embedding_store,
        )
        # Run context build in a thread — embedding_store.search() loads the
        # SentenceTransformer model synchronously on first call, which makes
        # HuggingFace network requests that would otherwise block the event loop.
        context = await asyncio.to_thread(context_builder.build, task)

        try:
            agent = self._create_agent(config["task_type"])
            record_agent_start()
            try:
                result = await agent.execute(context)
            finally:
                record_agent_end()

            # Track metrics
            agent_name = agent.role.value
            if agent_name not in self._metrics["agent_metrics"]:
                self._metrics["agent_metrics"][agent_name] = []
            self._metrics["agent_metrics"][agent_name].append(agent.get_metrics())

            # Determine success/failure for lifecycle events.
            # Review is special: ReviewerAgent always returns success=True (task
            # completed), but uses result.metrics["passed"] to indicate the
            # actual review verdict.
            if config["task_type"] == TaskType.REVIEW_FILE:
                # Treat a task-level failure (e.g. JSON parse error) as a
                # review failure rather than silently passing the file through.
                # Also change the default to False so an empty metrics dict
                # (result of a ValidationError) doesn't produce a false pass.
                review_passed = result.success and result.metrics.get("passed", False)
                if review_passed:
                    event_data = {"output": result.output}
                    engine.process_event(file_path, EventType.REVIEW_PASSED, event_data)
                    if self._event_bus:
                        await self._event_bus.publish(AgentEvent(
                            type=BusEventType.REVIEW_PASSED,
                            task_type=config["task_type"].value,
                            file_path=file_path,
                            agent_name=agent_name,
                        ))
                else:
                    event_data = {"findings": result.errors, "output": result.output}
                    engine.process_event(file_path, EventType.REVIEW_FAILED, event_data)
                    if self._event_bus:
                        await self._event_bus.publish(AgentEvent(
                            type=BusEventType.REVIEW_FAILED,
                            task_type=config["task_type"].value,
                            file_path=file_path,
                            agent_name=agent_name,
                            data={"findings": result.errors},
                        ))
                self._metrics["tasks_completed"] += 1
            elif result.success:
                self._metrics["tasks_completed"] += 1
                event_data = self._extract_event_data(result, config["task_type"])
                engine.process_event(file_path, config["success_event"], event_data)
                logger.info("[%s] %s succeeded", file_path, phase.value)
                # Incremental embedding update for lifecycle path
                if self._embedding_store and result.files_modified:
                    for fp in result.files_modified:
                        try:
                            content = self.repo.read_file(fp)
                            self._embedding_store.index_file(fp, content)
                        except Exception:
                            logger.debug("Embedding update skipped for %s", fp)
                        if self._event_bus:
                            await self._event_bus.publish(AgentEvent(
                                type=BusEventType.FILE_WRITTEN,
                                task_type=config["task_type"].value,
                                file_path=fp,
                                agent_name=agent_name,
                            ))
                if self._event_bus:
                    bus_type = (
                        BusEventType.TEST_PASSED
                        if config["task_type"] == TaskType.GENERATE_TEST
                        else BusEventType.TASK_COMPLETED
                    )
                    await self._event_bus.publish(AgentEvent(
                        type=bus_type,
                        task_type=config["task_type"].value,
                        file_path=file_path,
                        agent_name=agent_name,
                    ))
                if self._live:
                    self._live.log(f"[green]Done:[/green] {config['description']}")
            else:
                event_data = self._extract_event_data(result, config["task_type"])
                engine.process_event(file_path, config["failure_event"], event_data)
                logger.warning("[%s] %s failed: %s", file_path, phase.value, result.errors)
                if self._event_bus:
                    bus_type = (
                        BusEventType.TEST_FAILED
                        if config["task_type"] == TaskType.GENERATE_TEST
                        else BusEventType.TASK_FAILED
                    )
                    await self._event_bus.publish(AgentEvent(
                        type=bus_type,
                        task_type=config["task_type"].value,
                        file_path=file_path,
                        agent_name=agent_name,
                        data={"errors": result.errors},
                    ))
                if self._live:
                    self._live.log(
                        f"[yellow]Issue:[/yellow] {config['description']} — "
                        f"transitioning via {config['failure_event'].value}"
                    )

        except Exception as e:
            logger.exception("[%s] %s error", file_path, phase.value)
            # For review phase, treat crash as REVIEW_FAILED so the file still
            # goes through build verification rather than silently passing.
            if phase == FilePhase.REVIEWING:
                engine.process_event(
                    file_path, EventType.REVIEW_FAILED,
                    {"findings": [f"Review crashed: {e}"], "output": ""},
                )
            elif phase == FilePhase.FIXING:
                # Fix crashed — fire RETRIES_EXHAUSTED so the file stops
                # cycling and is marked FAILED instead of looping forever.
                engine.process_event(file_path, EventType.RETRIES_EXHAUSTED)
                self._metrics["tasks_failed"] += 1
            else:
                engine.process_event(file_path, EventType.RETRIES_EXHAUSTED)
                self._metrics["tasks_failed"] += 1

    async def execute_with_checkpoints(
        self,
        engine: LifecycleEngine,
        global_graph: TaskGraph,
        *,
        tiers: list[Tier] | None = None,
        pipeline_def: Any = None,
    ) -> dict[str, Any]:
        """Tier-scheduled execution with repo-level build checkpoints.

        Delegates to :class:`~core.lifecycle_orchestrator.LifecycleOrchestrator`.
        """
        return await self._orchestrator.execute_with_checkpoints(
            engine,
            global_graph,
            tiers=tiers,
            pipeline_def=pipeline_def,
        )

    # ── Static helpers — forwarding aliases for backward compatibility ────────

    @staticmethod
    def _build_lifecycle_metadata(lc: Any) -> dict[str, Any]:
        """Forwarding alias — implementation lives in LifecycleOrchestrator."""
        return LifecycleOrchestrator._build_lifecycle_metadata(lc)

    @staticmethod
    def _extract_event_data(result: Any, task_type: Any) -> dict[str, Any]:
        """Forwarding alias — implementation lives in LifecycleOrchestrator."""
        return LifecycleOrchestrator._extract_event_data(result, task_type)

