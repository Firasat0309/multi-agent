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
            )
        if agent_cls is SecurityAgent:
            return agent_cls(
                llm_client=self.llm,
                repo_manager=self.repo,
                terminal=self.build_terminal,
            )
        if agent_cls is BuildVerifierAgent:
            return agent_cls(
                llm_client=self.llm,
                repo_manager=self.repo,
                terminal=self.build_terminal,
            )
        return agent_cls(llm_client=self.llm, repo_manager=self.repo)

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
        """Execute one lifecycle phase — kept for backward compatibility with tests.

        Delegates to :class:`~core.lifecycle_orchestrator.LifecycleOrchestrator`.
        """
        return await self._orchestrator._execute_lifecycle_phase(engine, file_path, phase)

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

