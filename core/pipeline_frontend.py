"""Frontend pipeline — generates a complete UI codebase from design spec + API contract."""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import json as _json
import logging
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from agents.design_parser_agent import DesignParserAgent
from agents.component_planner_agent import ComponentPlannerAgent
from agents.component_dag_agent import ComponentDAGAgent
from config.settings import Settings
from core.agent_manager import AgentManager
from core.event_bus import EventBus, AgentEvent, BusEventType
from core.import_validator import ImportValidator
from core.language import detect_language_from_blueprint
from core.llm_client import LLMClient, calculate_cost
from core.models import (
    APIContract,
    ComponentPlan,
    FileBlueprint,
    ProductRequirements,
    RepositoryBlueprint,
    Task,
    TaskType,
    TokenCost,
    UIDesignSpec,
    AgentContext,
)
from core.ast_extractor import ASTExtractor
from core.pipeline_definition import FRONTEND_PIPELINE
from core.repository_manager import RepositoryManager
from core.sandbox_orchestrator import SandboxOrchestrator, SandboxUnavailableError
from core.tsx_compiler import TSXCompiler
from core.workspace_indexer import index_workspace
from memory.dependency_graph import DependencyGraphStore
from memory.embedding_store import EmbeddingStore
from core.mcp_client import MCPClient

if TYPE_CHECKING:
    from core.live_console import LiveConsole
    from core.pipeline import PipelineResult

logger = logging.getLogger(__name__)


class FrontendPipeline:
    """Runs the full frontend generation workflow.

    Phases:
      1. Design Parsing      — parse Figma URL / prose into UIDesignSpec
      2. Component Planning  — decompose design into a ComponentPlan
      3. DAG Building        — topological sort of component dependencies
      4. Component Generation— generate source code per component (tiers)
      4.5 TSX Compilation    — type-check with fix-retry loop
      5. API Integration     — typed API client + SWR/React-Query hooks
      6. State Management    — Zustand/Redux/Pinia store layer
    """

    def __init__(
        self,
        settings: Settings,
        llm: LLMClient,
        live: LiveConsole | None = None,
        root_write_lock: asyncio.Lock | None = None,
        interactive: bool = True,
    ) -> None:
        self._settings = settings
        self._llm = llm
        self._live = live
        self._root_write_lock = root_write_lock
        self._interactive = interactive

    # ── Public entry point ────────────────────────────────────────────────────

    async def execute(
        self,
        requirements: ProductRequirements,
        api_contract: APIContract | None,
        start_time: float,
        figma_url: str = "",
        frontend_workspace: Path | None = None,
        backend_blueprint: RepositoryBlueprint | None = None,
    ) -> PipelineResult:
        """Run all frontend generation phases and return a PipelineResult."""
        from core.pipeline import PipelineResult

        workspace = frontend_workspace or (self._settings.workspace_dir / "frontend")
        workspace.mkdir(parents=True, exist_ok=True)
        repo_manager = RepositoryManager(workspace)
        if self._root_write_lock is not None:
            repo_manager._root_write_lock = self._root_write_lock

        # Wire EmbeddingStore so every write_file call indexes the component
        # for semantic context queries used by later-tier ComponentGeneratorAgent tasks.
        _fe_embedding_store = EmbeddingStore(
            persist_dir=str(workspace / ".chroma"),
            embedding_model=self._settings.memory.embedding_model,
        )
        asyncio.ensure_future(asyncio.to_thread(_fe_embedding_store._ensure_client))
        repo_manager._embedding_store = _fe_embedding_store

        errors: list[str] = []
        metrics: dict = {}
        component_plan: ComponentPlan | None = None

        # Initialize embedded MCP client if configured
        mcp_client: MCPClient | None = None
        if self._settings.mcp_server_command:
            try:
                mcp_client = MCPClient(self._settings.mcp_server_command)
                await mcp_client.initialize()
            except Exception as e:
                logger.error("Failed to initialize MCP client in frontend pipeline: %e", e)
                mcp_client = None

        # ── Sandbox setup ─────────────────────────────────────────────────────
        sandbox: SandboxOrchestrator | None = None

        try:
            # ── Phase 1: Design Parsing ───────────────────────────────────────────
            self._phase("FE: Design Parsing", "running")
            logger.info("[FE Phase 1] Parsing design spec...")
            design_spec: UIDesignSpec | None = None
            parser = DesignParserAgent(llm_client=self._llm, repo_manager=repo_manager, mcp_client=mcp_client)
            design_spec = await parser.parse_design(requirements, figma_url)
            logger.info(
                "UIDesignSpec: %d pages, framework=%s",
                len(design_spec.pages), design_spec.framework,
            )
            self._complete_phase("FE: Design Parsing")
        except Exception as exc:
            logger.exception("Design parsing failed")
            errors.append(f"Design parsing failed: {exc}")
            self._fail_phase("FE: Design Parsing", str(exc))
            # Non-fatal — continue with an empty spec
            design_spec = UIDesignSpec(framework=self._infer_framework(requirements))

        # ── Phase 2: Component Planning ───────────────────────────────────────
        self._phase("FE: Component Planning", "running")
        logger.info("[FE Phase 2] Planning components...")
        try:
            planner = ComponentPlannerAgent(llm_client=self._llm, repo_manager=repo_manager, mcp_client=mcp_client)
            component_plan = await planner.plan_components(design_spec, api_contract, requirements)
            logger.info(
                "ComponentPlan: %d components, state=%s",
                len(component_plan.components), component_plan.state_solution,
            )
            self._complete_phase("FE: Component Planning")
        except Exception as exc:
            logger.exception("Component planning failed")
            errors.append(f"Component planning failed: {exc}")
            self._fail_phase("FE: Component Planning", str(exc))
            return PipelineResult(
                success=False,
                workspace_path=workspace,
                errors=errors,
                elapsed_seconds=time.monotonic() - start_time,
            )

        # ── Approval Gate 3: Frontend Architecture ─────────────────────────────
        if self._settings.require_architecture_approval:
            from core.architecture_approver import (
                ArchitectureApprover,
                ArchitecturePendingApprovalError,
            )
            approver = ArchitectureApprover(
                interactive=self._interactive,
                workspace=workspace,
                live=self._live,
            )
            while True:
                try:
                    result = approver.approve_frontend_architecture(component_plan)
                except ArchitecturePendingApprovalError as e:
                    logger.info("Frontend architecture pending human approval: %s", e)
                    return PipelineResult(
                        success=False,
                        workspace_path=workspace,
                        errors=[str(e)],
                        elapsed_seconds=time.monotonic() - start_time,
                    )
                if result is True:
                    logger.info("Frontend architecture approved by user")
                    break
                if result is False:
                    logger.info("Frontend architecture rejected by user")
                    return PipelineResult(
                        success=False,
                        workspace_path=workspace,
                        errors=["Frontend architecture rejected by user"],
                        elapsed_seconds=time.monotonic() - start_time,
                    )
                # result is a str — user feedback for revision
                logger.info("User requested frontend architecture revision: %s", result)
                try:
                    component_plan = await planner.revise_components(
                        component_plan, result,
                    )
                    logger.info(
                        "ComponentPlan revised: %d components",
                        len(component_plan.components),
                    )
                except Exception as exc:
                    logger.exception("Frontend architecture revision failed")
                    errors.append(f"Frontend architecture revision failed: {exc}")
                    return PipelineResult(
                        success=False,
                        workspace_path=workspace,
                        errors=errors,
                        elapsed_seconds=time.monotonic() - start_time,
                    )

        # Initialize RepositoryManager with a synthetic frontend blueprint
        frontend_blueprint = self._make_frontend_blueprint(requirements, component_plan)
        repo_manager.initialize(frontend_blueprint)
        metrics["components_planned"] = len(component_plan.components)

        # ── Write package.json and .env.local ─────────────────────────────────
        self._write_config_files(workspace, component_plan, requirements)

        # ── Sandbox + AgentManager setup ──────────────────────────────────────
        lang_profile = detect_language_from_blueprint(frontend_blueprint.tech_stack)

        sandbox = SandboxOrchestrator(self._settings)
        try:
            sb = await sandbox.setup(lang_profile)
        except SandboxUnavailableError as e:
            self._fail_phase("FE: Sandbox Setup", str(e))
            return PipelineResult(
                success=False,
                workspace_path=workspace,
                blueprint=frontend_blueprint,
                errors=[str(e)],
                elapsed_seconds=time.monotonic() - start_time,
            )

        event_bus = EventBus()
        dep_store = DependencyGraphStore(workspace)

        agent_manager = AgentManager(
            settings=self._settings,
            llm_client=self._llm,
            repo_manager=repo_manager,
            blueprint=frontend_blueprint,
            live_console=self._live,
            sandbox_manager=sb.manager,
            build_sandbox_id=sb.build_id,
            test_sandbox_id=sb.test_id,
            dep_store=dep_store,
            embedding_store=_fe_embedding_store,
            event_bus=event_bus,
            mcp_client=mcp_client,
        )

        try:
            # ── Phase 3: Component DAG ────────────────────────────────────────────
            self._phase("FE: Component DAG", "running")
            logger.info("[FE Phase 3] Building component dependency graph...")
            ordered_components = component_plan.components  # default linear order
            tier_map: dict[str, int] = {}
            try:
                dag_agent = ComponentDAGAgent(llm_client=self._llm, repo_manager=repo_manager)
                ordered_components, tier_map = dag_agent.build_dag(component_plan)
                logger.info(
                    "DAG: %d components across %d tiers",
                    len(ordered_components),
                    max(tier_map.values()) + 1 if tier_map else 1,
                )
                self._complete_phase("FE: Component DAG")
            except Exception as exc:
                logger.exception("Component DAG construction failed")
                errors.append(f"Component DAG failed: {exc}")
                self._fail_phase("FE: Component DAG", str(exc))
                # Non-fatal: fall back to planner order

            # ── Phase 4: Component Generation (tier-by-tier) ──────────────────────
            self._phase("FE: Component Generation", "running")

            # Detect ghost dependencies — components referenced in depends_on but
            # not present in the plan — and synthesize minimal stubs so they are
            # generated rather than left as empty nodes in the dependency graph.
            planned_names = {c.name for c in ordered_components}
            ghost_components = self._find_ghost_dependencies(
                ordered_components, planned_names, framework=component_plan.framework,
            )
            if ghost_components:
                logger.info(
                    "[FE Phase 4] Adding %d unplanned dependency stubs: %s",
                    len(ghost_components),
                    [c.name for c in ghost_components],
                )
                # Ghosts go in tier 0 (no dependencies of their own)
                for gc in ghost_components:
                    tier_map[gc.name] = 0
                ordered_components = list(ghost_components) + list(ordered_components)
                # Rebuild the blueprint to include ghost paths
                frontend_blueprint = self._make_frontend_blueprint(
                    requirements, component_plan, extra_components=ghost_components,
                )
            max_tier = max(tier_map.values()) + 1 if tier_map else 1

            logger.info("[FE Phase 4] Generating %d components...", len(ordered_components))
            generator = agent_manager._create_agent(TaskType.GENERATE_COMPONENT)
            gen_errors: list[str] = []

            for tier_idx in range(max_tier):
                tier_components = [
                    c for c in ordered_components
                    if tier_map.get(c.name, 0) == tier_idx
                ] if tier_map else (
                    ordered_components if tier_idx == 0 else []
                )

                # Run all components within the same tier concurrently
                tasks_coros = [
                    generator.execute(
                        AgentContext(
                            task=Task(
                                task_id=i,
                                task_type=TaskType.GENERATE_COMPONENT,
                                file=comp.file_path,
                                description=f"Generate {comp.name} component",
                                metadata={
                                    "component": comp,
                                    "component_plan": component_plan,
                                    "api_contract": api_contract,
                                    "design_spec": design_spec,
                                    "requirements": requirements,
                                },
                            ),
                            blueprint=frontend_blueprint,
                        )
                    )
                    for i, comp in enumerate(tier_components)
                ]
                results = await asyncio.gather(*tasks_coros, return_exceptions=True)
                for comp, res in zip(tier_components, results):
                    if isinstance(res, Exception):
                        gen_errors.append(f"{comp.name}: {res}")
                        logger.error("Component generation failed for %s: %s", comp.name, res)
                        await event_bus.publish(AgentEvent(
                            type=BusEventType.TASK_FAILED,
                            task_type=TaskType.GENERATE_COMPONENT.value,
                            file_path=comp.file_path,
                            agent_name="ComponentGeneratorAgent",
                            data={"error": str(res)},
                        ))
                    elif not res.success:
                        gen_errors.append(f"{comp.name}: {'; '.join(res.errors)}")
                        logger.warning("Component generation incomplete for %s", comp.name)
                        await event_bus.publish(AgentEvent(
                            type=BusEventType.TASK_FAILED,
                            task_type=TaskType.GENERATE_COMPONENT.value,
                            file_path=comp.file_path,
                            agent_name="ComponentGeneratorAgent",
                            data={"errors": res.errors},
                        ))
                    else:
                        logger.debug("Generated component: %s", comp.file_path)
                        await event_bus.publish(AgentEvent(
                            type=BusEventType.TASK_COMPLETED,
                            task_type=TaskType.GENERATE_COMPONENT.value,
                            file_path=comp.file_path,
                            agent_name="ComponentGeneratorAgent",
                        ))
                        await event_bus.publish(AgentEvent(
                            type=BusEventType.FILE_WRITTEN,
                            file_path=comp.file_path,
                            agent_name="ComponentGeneratorAgent",
                        ))

            if gen_errors:
                errors.extend(gen_errors)
                logger.warning("%d component generation errors", len(gen_errors))
            metrics["components_generated"] = len(ordered_components) - len(gen_errors)
            self._complete_phase("FE: Component Generation")

            # ── Phase 4.5: TSX Compilation Check with Fix-Retry ───────────────────
            self._phase("FE: TypeScript Compilation", "running")
            logger.info("[FE Phase 4.5] Running TypeScript compilation check...")
            tsx_compiler = TSXCompiler()
            checkpoint_def = FRONTEND_PIPELINE.phases[0].checkpoint
            max_fix_retries = checkpoint_def.max_retries if checkpoint_def else 2

            compile_result = await tsx_compiler.check(workspace)
            tsx_build_passed = True

            if not compile_result.tsc_available:
                logger.info(
                    "[FE Phase 4.5] tsc not on PATH — skipping TypeScript compilation check"
                )
            elif compile_result.errors:
                # Fix-retry loop: group errors by file, dispatch FIX_CODE tasks
                for retry in range(max_fix_retries):
                    logger.warning(
                        "[FE Phase 4.5] %d TSX error(s) — fix attempt %d/%d",
                        len(compile_result.errors), retry + 1, max_fix_retries,
                    )
                    # Group errors by file
                    errors_by_file: dict[str, list[str]] = defaultdict(list)
                    for err in compile_result.errors:
                        errors_by_file[err.file].append(
                            f"{err.file}:{err.line}: [{err.code}] {err.message}"
                        )

                    # Dispatch fix tasks for each affected file
                    fix_coros = []
                    for file_path, file_errors in errors_by_file.items():
                        fix_agent = agent_manager._create_agent(TaskType.FIX_CODE)
                        fix_ctx = AgentContext(
                            task=Task(
                                task_id=10000 + retry * 100 + len(fix_coros),
                                task_type=TaskType.FIX_CODE,
                                file=file_path,
                                description=f"Fix TSX compilation errors in {file_path}",
                                metadata={
                                    "fix_trigger": "build",
                                    "build_errors": "\n".join(file_errors),
                                },
                            ),
                            blueprint=frontend_blueprint,
                        )
                        fix_coros.append(fix_agent.execute(fix_ctx))

                    fix_results = await asyncio.gather(*fix_coros, return_exceptions=True)
                    for fix_res in fix_results:
                        if isinstance(fix_res, Exception):
                            logger.error("Fix task failed: %s", fix_res)

                    # Re-check compilation
                    compile_result = await tsx_compiler.check(workspace)
                    if not compile_result.errors:
                        logger.info("[FE Phase 4.5] TSX compilation passed after %d fix(es)", retry + 1)
                        break
                else:
                    # Exhausted retries
                    tsx_build_passed = False

                if compile_result.errors:
                    metrics["tsx_compile_errors"] = len(compile_result.errors)
                    for err in compile_result.errors:
                        errors.append(
                            f"TSX compile error {err.file}:{err.line}: [{err.code}] {err.message}"
                        )
                    logger.warning(
                        "[FE Phase 4.5] %d TSX error(s) remain after %d retries",
                        len(compile_result.errors), max_fix_retries,
                    )
            else:
                logger.info("[FE Phase 4.5] TypeScript compilation passed")

            # Publish build event
            if tsx_build_passed:
                await event_bus.publish(AgentEvent(
                    type=BusEventType.BUILD_PASSED,
                    data={"checkpoint": "tsx_compilation"},
                ))
            else:
                await event_bus.publish(AgentEvent(
                    type=BusEventType.BUILD_FAILED,
                    data={
                        "checkpoint": "tsx_compilation",
                        "error_count": len(compile_result.errors) if compile_result.errors else 0,
                    },
                ))
            self._complete_phase("FE: TypeScript Compilation")

            # ── Phase 4.6: Cross-component import validation + auto-fix ────────────
            self._phase("FE: Import Validation", "running")
            logger.info("[FE Phase 4.6] Validating cross-component imports...")
            import_errors = self._validate_component_imports(workspace, ordered_components)
            if import_errors:
                # Attempt auto-fix: rewrite broken relative imports to correct paths
                fixed_count = self._auto_fix_imports(workspace, ordered_components)
                if fixed_count > 0:
                    logger.info("[FE Phase 4.6] Auto-fixed %d import(s), re-validating...", fixed_count)
                    import_errors = self._validate_component_imports(workspace, ordered_components)

            if import_errors:
                metrics["import_errors"] = len(import_errors)
                errors.extend(import_errors)
                logger.warning("[FE Phase 4.6] %d unresolved import(s) after auto-fix", len(import_errors))
            else:
                logger.info("[FE Phase 4.6] All component imports resolved")
            self._complete_phase("FE: Import Validation")

            # ── Phases 5-8: API → State (sequential), Docs + Deploy (parallel) ──
            # State management depends on api.ts exports, so Phase 6 runs after
            # Phase 5.  Docs and Deploy have no dependencies — they run in
            # parallel alongside the API→State sequence.
            logger.info(
                "[FE Phases 5-8] Running API integration → state management "
                "(sequential), documentation + deployment (parallel)..."
            )

            async def _run_api_integration() -> None:
                """Phase 5: API Integration."""
                self._phase("FE: API Integration", "running")
                logger.info("[FE Phase 5] Generating API client layer...")
                try:
                    integrator = agent_manager._create_agent(TaskType.INTEGRATE_API)
                    # Scan backend models for type reference
                    be_models = self._scan_backend_models(
                        workspace.parent / "backend",
                        backend_blueprint=backend_blueprint,
                    )
                    api_ctx = AgentContext(
                        task=Task(
                            task_id=9000,
                            task_type=TaskType.INTEGRATE_API,
                            file="src/lib/api.ts",
                            description="Generate typed API client and hooks",
                            metadata={
                                "api_contract": api_contract,
                                "component_plan": component_plan,
                                "requirements": requirements,
                                "backend_models": be_models,
                            },
                        ),
                        blueprint=frontend_blueprint,
                        file_blueprint=FileBlueprint(
                            path="src/lib/api.ts",
                            purpose="Base Axios/Fetch API client with auth interceptors",
                            language="typescript",
                            layer="lib",
                        ),
                    )
                    api_result = await integrator.execute(api_ctx)
                    if api_result.success:
                        await event_bus.publish(AgentEvent(
                            type=BusEventType.TASK_COMPLETED,
                            task_id=9000,
                            task_type=TaskType.INTEGRATE_API.value,
                            file_path="src/lib/api.ts",
                            agent_name="APIIntegrationAgent",
                        ))
                    else:
                        errors.extend(api_result.errors)
                        await event_bus.publish(AgentEvent(
                            type=BusEventType.TASK_FAILED,
                            task_id=9000,
                            task_type=TaskType.INTEGRATE_API.value,
                            file_path="src/lib/api.ts",
                            agent_name="APIIntegrationAgent",
                            data={"errors": api_result.errors},
                        ))
                    self._complete_phase("FE: API Integration")
                except Exception as exc:
                    logger.exception("API integration failed")
                    errors.append(f"API integration failed: {exc}")
                    self._fail_phase("FE: API Integration", str(exc))

            async def _run_state_management() -> None:
                """Phase 6: State Management."""
                self._phase("FE: State Management", "running")
                logger.info("[FE Phase 6] Generating state management layer (%s)...",
                            component_plan.state_solution)
                try:
                    state_agent = agent_manager._create_agent(TaskType.MANAGE_STATE)
                    state_ctx = AgentContext(
                        task=Task(
                            task_id=9001,
                            task_type=TaskType.MANAGE_STATE,
                            file="src/store/index.ts",
                            description=f"Generate {component_plan.state_solution} store layer",
                            metadata={
                                "component_plan": component_plan,
                                "api_contract": api_contract,
                                "requirements": requirements,
                            },
                        ),
                        blueprint=frontend_blueprint,
                        file_blueprint=FileBlueprint(
                            path="src/store/index.ts",
                            purpose="State management barrel / root store",
                            language="typescript",
                            layer="store",
                        ),
                    )
                    state_result = await state_agent.execute(state_ctx)
                    if state_result.success:
                        await event_bus.publish(AgentEvent(
                            type=BusEventType.TASK_COMPLETED,
                            task_id=9001,
                            task_type=TaskType.MANAGE_STATE.value,
                            file_path="src/store/index.ts",
                            agent_name="StateManagementAgent",
                        ))
                    else:
                        errors.extend(state_result.errors)
                        await event_bus.publish(AgentEvent(
                            type=BusEventType.TASK_FAILED,
                            task_id=9001,
                            task_type=TaskType.MANAGE_STATE.value,
                            file_path="src/store/index.ts",
                            agent_name="StateManagementAgent",
                            data={"errors": state_result.errors},
                        ))
                    self._complete_phase("FE: State Management")
                except Exception as exc:
                    logger.exception("State management generation failed")
                    errors.append(f"State management failed: {exc}")
                    self._fail_phase("FE: State Management", str(exc))

            async def _run_documentation() -> None:
                """Phase 7: Documentation (WriterAgent)."""
                self._phase("FE: Documentation", "running")
                logger.info("[FE Phase 7] Generating frontend documentation...")
                try:
                    writer_agent = agent_manager._create_agent(TaskType.GENERATE_DOCS)
                    # Populate related_files with generated page/route components
                    # so the WriterAgent can reference real code instead of hallucinating.
                    related: dict[str, str] = {}
                    for comp in ordered_components:
                        abs_path = workspace / comp.file_path
                        if abs_path.exists():
                            try:
                                content = abs_path.read_text(encoding="utf-8", errors="replace")
                                related[comp.file_path] = content[:3000]
                            except OSError:
                                pass
                    # Also include api.ts and store if they exist
                    for extra in ("src/lib/api.ts", "src/store/index.ts"):
                        ep = workspace / extra
                        if ep.exists():
                            try:
                                related[extra] = ep.read_text(encoding="utf-8", errors="replace")[:3000]
                            except OSError:
                                pass

                    writer_ctx = AgentContext(
                        task=Task(
                            task_id=9002,
                            task_type=TaskType.GENERATE_DOCS,
                            file="docs/README.md",
                            description="Generate frontend documentation (README, API docs, Changelog)",
                            metadata={
                                "component_plan": component_plan,
                                "api_contract": api_contract,
                                "requirements": requirements,
                            },
                        ),
                        blueprint=frontend_blueprint,
                        related_files=related,
                    )
                    writer_result = await writer_agent.execute(writer_ctx)
                    if writer_result.success:
                        logger.info("[FE Phase 7] Documentation generated: %s", writer_result.output)
                        await event_bus.publish(AgentEvent(
                            type=BusEventType.TASK_COMPLETED,
                            task_id=9002,
                            task_type=TaskType.GENERATE_DOCS.value,
                            file_path="docs/README.md",
                            agent_name="WriterAgent",
                        ))
                    else:
                        errors.extend(writer_result.errors)
                        logger.warning("[FE Phase 7] Documentation generation had errors: %s", writer_result.errors)
                        await event_bus.publish(AgentEvent(
                            type=BusEventType.TASK_FAILED,
                            task_id=9002,
                            task_type=TaskType.GENERATE_DOCS.value,
                            file_path="docs/README.md",
                            agent_name="WriterAgent",
                            data={"errors": writer_result.errors},
                        ))
                    self._complete_phase("FE: Documentation")
                except Exception as exc:
                    logger.exception("Frontend documentation generation failed")
                    errors.append(f"Documentation generation failed: {exc}")
                    self._fail_phase("FE: Documentation", str(exc))

            async def _run_deployment() -> None:
                """Phase 8: Deployment Artifacts (DeployAgent)."""
                self._phase("FE: Deployment", "running")
                logger.info("[FE Phase 8] Generating deployment artifacts (Dockerfile, K8s)...")
                try:
                    deploy_agent = agent_manager._create_agent(TaskType.GENERATE_DEPLOY)
                    deploy_ctx = AgentContext(
                        task=Task(
                            task_id=9003,
                            task_type=TaskType.GENERATE_DEPLOY,
                            file="deploy/Dockerfile",
                            description="Generate frontend deployment artifacts (Dockerfile, docker-compose, K8s manifests)",
                            metadata={
                                "component_plan": component_plan,
                                "api_contract": api_contract,
                                "requirements": requirements,
                            },
                        ),
                        blueprint=frontend_blueprint,
                    )
                    deploy_result = await deploy_agent.execute(deploy_ctx)
                    if deploy_result.success:
                        logger.info("[FE Phase 8] Deployment artifacts generated: %s", deploy_result.output)
                        await event_bus.publish(AgentEvent(
                            type=BusEventType.TASK_COMPLETED,
                            task_id=9003,
                            task_type=TaskType.GENERATE_DEPLOY.value,
                            file_path="deploy/Dockerfile",
                            agent_name="DeployAgent",
                        ))
                    else:
                        errors.extend(deploy_result.errors)
                        logger.warning("[FE Phase 8] Deployment generation had errors: %s", deploy_result.errors)
                        await event_bus.publish(AgentEvent(
                            type=BusEventType.TASK_FAILED,
                            task_id=9003,
                            task_type=TaskType.GENERATE_DEPLOY.value,
                            file_path="deploy/Dockerfile",
                            agent_name="DeployAgent",
                            data={"errors": deploy_result.errors},
                        ))
                    self._complete_phase("FE: Deployment")
                except Exception as exc:
                    logger.exception("Frontend deployment artifact generation failed")
                    errors.append(f"Deployment generation failed: {exc}")
                    self._fail_phase("FE: Deployment", str(exc))

            async def _run_api_then_state_then_docs() -> None:
                """Phase 5 → 6 → 7: API → State → Documentation (sequential).

                State stores import from api.ts, so api.ts must exist first.
                Documentation needs generated component + api.ts + store files
                for accurate related_files context.
                """
                await _run_api_integration()
                await _run_state_management()
                await _run_documentation()

            _parallel_start = time.monotonic()
            await asyncio.gather(
                _run_api_then_state_then_docs(),
                _run_deployment(),
            )
            logger.info(
                "[FE Phases 5-8] Phases complete in %.1fs",
                time.monotonic() - _parallel_start,
            )

            # ── Phase 9: Final Build Check ─────────────────────────────────────
            # Run npm install + tsc/next build AFTER all files are generated.
            # Phase 4.5 only catches errors in components; this catches errors
            # introduced by API integration, state management, and cross-file
            # type mismatches.
            self._phase("FE: Final Build Check", "running")
            logger.info("[FE Phase 9] Running final build check (npm install + type check)...")
            try:
                final_build_ok = await self._run_final_build_check(
                    workspace, frontend_blueprint, agent_manager, event_bus,
                    component_plan, errors, metrics,
                    api_contract=api_contract,
                    requirements=requirements,
                    ordered_components=ordered_components,
                )
                if final_build_ok:
                    self._complete_phase("FE: Final Build Check")
                else:
                    self._fail_phase("FE: Final Build Check", "Build errors remain")
            except Exception as exc:
                logger.exception("Final build check failed")
                errors.append(f"Final build check failed: {exc}")
                self._fail_phase("FE: Final Build Check", str(exc))

            # ── Finalise: persist dep graph, repo index, and embeddings ──────────
            try:
                fe_settings = copy.copy(self._settings)
                fe_settings.workspace_dir = workspace
                fe_settings.memory = dataclasses.replace(
                    self._settings.memory,
                    chroma_persist_dir=str(workspace / ".chroma"),
                )
                index_workspace(repo_manager, fe_settings)
            except Exception:
                logger.warning("Frontend workspace indexing failed", exc_info=True)

            elapsed = time.monotonic() - start_time
            # Pipeline fails if component planning failed OR final build has errors
            has_critical_error = any(
                e for e in errors
                if "Component planning failed" in e
            )
            has_build_errors = metrics.get("final_build_errors", 0) > 0
            success = not has_critical_error and not has_build_errors

            token_cost = self._build_token_cost()

            # Write frontend report
            self._write_report(
                workspace=workspace,
                success=success,
                elapsed=elapsed,
                requirements=requirements,
                component_plan=component_plan,
                metrics=metrics,
                errors=errors,
                token_cost=token_cost,
            )

            return PipelineResult(
                success=success,
                workspace_path=workspace,
                blueprint=frontend_blueprint,
                metrics=metrics,
                errors=errors,
                elapsed_seconds=elapsed,
                token_cost=token_cost,
            )
        finally:
            # Ensure sandbox and MCP client are cleaned up even on failure.
            try:
                if sandbox is not None:
                    await sandbox.teardown()
            except Exception:
                logger.warning("Error during sandbox teardown", exc_info=True)
            if mcp_client:
                await mcp_client.close()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_token_cost(self) -> TokenCost:
        cost = calculate_cost(
            self._llm.config.model,
            self._llm.total_input_tokens,
            self._llm.total_output_tokens,
        )
        return TokenCost(
            input_tokens=self._llm.total_input_tokens,
            output_tokens=self._llm.total_output_tokens,
            model=self._llm.config.model,
            cost_usd=cost,
        )

    @staticmethod
    def _write_report(
        *,
        workspace: Path,
        success: bool,
        elapsed: float,
        requirements: ProductRequirements,
        component_plan: ComponentPlan,
        metrics: dict,
        errors: list[str],
        token_cost: TokenCost,
    ) -> None:
        """Write frontend_report.json to the workspace."""
        report = {
            "mode": "frontend",
            "success": success,
            "elapsed_seconds": round(elapsed, 2),
            "project": requirements.title,
            "framework": component_plan.framework,
            "state_solution": component_plan.state_solution,
            "metrics": metrics,
            "errors": errors,
            "components_planned": len(component_plan.components),
            "components_generated": metrics.get("components_generated", 0),
            "token_cost": {
                "input_tokens": token_cost.input_tokens,
                "output_tokens": token_cost.output_tokens,
                "model": token_cost.model,
                "cost_usd": round(token_cost.cost_usd, 4),
            },
        }
        report_path = workspace / "frontend_report.json"
        try:
            report_path.write_text(_json.dumps(report, indent=2), encoding="utf-8")
            logger.info("Wrote frontend_report.json")
        except Exception:
            logger.warning("Failed to write frontend_report.json", exc_info=True)

    def _make_frontend_blueprint(
        self,
        requirements: ProductRequirements,
        plan: ComponentPlan,
        extra_components: list | None = None,
    ) -> RepositoryBlueprint:
        """Build a RepositoryBlueprint from the ComponentPlan for RepositoryManager.

        *extra_components* are additional UIComponent objects (e.g. ghost
        dependency stubs) that should appear in the blueprint but are not
        part of ``plan.components``.
        """
        all_components = list(plan.components)
        if extra_components:
            # Avoid duplicates — ghosts may share names with planned components
            planned_paths = {c.file_path for c in all_components}
            for ec in extra_components:
                if ec.file_path not in planned_paths:
                    all_components.append(ec)

        file_blueprints = [
            FileBlueprint(
                path=comp.file_path,
                purpose=comp.description,
                depends_on=comp.depends_on,
                exports=[comp.name],
                language="typescript",
                layer=comp.layer or comp.component_type,
            )
            for comp in all_components
        ]
        fw = plan.framework
        tech_stack: dict[str, str] = {
            "language": "typescript",
            "framework": fw,
            "build_tool": "npm",
            "state": plan.state_solution,
            "styling": requirements.tech_preferences.get("styling", "tailwind"),
        }
        return RepositoryBlueprint(
            name=f"{requirements.title}-frontend",
            description=f"Frontend for {requirements.title}",
            architecture_style="SPA",
            tech_stack=tech_stack,
            folder_structure=["src/components", "src/pages", "src/store",
                              "src/lib", "src/hooks"],
            file_blueprints=file_blueprints,
        )

    @staticmethod
    def _validate_component_imports(
        workspace: Path,
        components: list,
    ) -> list[str]:
        """Check that every relative import in each generated TSX component file
        resolves to an existing file in the workspace.

        Returns a list of human-readable error strings, one per broken import.
        """
        validator = ImportValidator()
        # Build the set of all .ts/.tsx files that exist in the workspace
        known_files: set[str] = {
            p.relative_to(workspace).as_posix()
            for p in workspace.rglob("*.ts")
        } | {
            p.relative_to(workspace).as_posix()
            for p in workspace.rglob("*.tsx")
        }
        errors: list[str] = []
        for comp in components:
            file_path: str = comp.file_path
            abs_path = workspace / file_path
            if not abs_path.exists():
                continue
            try:
                content = abs_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            broken = validator._validate_typescript(content, file_path, known_files)
            for imp in broken:
                errors.append(f"{file_path}: unresolved import '{imp}'")
        return errors

    @staticmethod
    def _auto_fix_imports(
        workspace: Path,
        components: list,
    ) -> int:
        """Attempt to fix broken relative imports by finding the correct path.

        Scans each component file for relative imports that don't resolve,
        then searches the workspace for a file whose basename matches the
        import target and rewrites the import path accordingly.

        Returns the number of imports fixed.
        """
        import posixpath

        # Build lookup: basename (no extension) → list of workspace-relative paths
        ts_files: dict[str, list[str]] = {}
        ts_files_lower: dict[str, list[str]] = {}
        for ext in ("*.ts", "*.tsx"):
            for p in workspace.rglob(ext):
                rel = p.relative_to(workspace).as_posix()
                stem = p.stem  # e.g. "authStore" from "authStore.ts"
                ts_files.setdefault(stem, []).append(rel)
                ts_files_lower.setdefault(stem.lower(), []).append(rel)

        def _find_target(name: str) -> list[str]:
            """Find target file by basename with fallback strategies."""
            # Exact match
            matches = ts_files.get(name, [])
            if matches:
                return matches
            # Strip 'use' prefix: useAuthStore → AuthStore → authStore
            if name.startswith("use") and len(name) > 3:
                stripped = name[3:]
                matches = ts_files.get(stripped, [])
                if matches:
                    return matches
                lower_first = stripped[0].lower() + stripped[1:]
                matches = ts_files.get(lower_first, [])
                if matches:
                    return matches
            # Case-insensitive fallback
            matches = ts_files_lower.get(name.lower(), [])
            if matches:
                return matches
            return []

        imp_re = re.compile(
            r"""(import\s+(?:type\s+)?(?:[^'"\n;]+?\s+from\s+)?['\"])([^'\"]+)(['\"])"""
        )

        # Pre-compute the set of all TS/TSX files once (avoid re-globbing per import)
        existing: set[str] = set()
        for ext in ("*.ts", "*.tsx"):
            for p in workspace.rglob(ext):
                existing.add(p.relative_to(workspace).as_posix())

        fixed_total = 0
        for comp in components:
            file_path: str = comp.file_path
            abs_path = workspace / file_path
            if not abs_path.exists():
                continue
            try:
                content = abs_path.read_text(encoding="utf-8")
            except OSError:
                continue

            file_dir = posixpath.dirname(file_path)
            new_content = content
            changed = False

            for m in imp_re.finditer(content):
                imp = m.group(2)

                # Handle @/ alias imports (e.g. @/store/useAuthStore)
                if imp.startswith("@/"):
                    # Resolve @/ → src/
                    resolved = "src/" + imp[2:]
                    candidates = [resolved + e for e in (".ts", ".tsx", ".js", ".jsx",
                                                          "/index.ts", "/index.tsx")]
                    candidates.append(resolved)
                    if any(c in existing for c in candidates):
                        continue  # @/ import resolves fine

                    # Broken @/ import — try to find target by basename
                    target_name = posixpath.basename(imp)
                    matches = _find_target(target_name)
                    if len(matches) == 1:
                        target_rel = matches[0]
                        target_no_ext = re.sub(r"\.(tsx?|jsx?)$", "", target_rel)
                        new_imp = posixpath.relpath(target_no_ext, file_dir)
                        if not new_imp.startswith("."):
                            new_imp = "./" + new_imp
                    elif len(matches) == 0:
                        # No match at all — rewrite @/ to relative ../
                        # so it at least points at the right directory
                        target_no_ext = "src/" + imp[2:]
                        new_imp = posixpath.relpath(target_no_ext, file_dir)
                        if not new_imp.startswith("."):
                            new_imp = "./" + new_imp
                    else:
                        continue  # ambiguous, skip

                    if new_imp != imp:
                        old_full = m.group(0)
                        new_full = m.group(1) + new_imp + m.group(3)
                        new_content = new_content.replace(old_full, new_full, 1)
                        changed = True
                        fixed_total += 1
                        logger.info(
                            "[FE Import Fix] %s: '%s' → '%s'",
                            file_path, imp, new_imp,
                        )
                    continue

                if not imp.startswith("."):
                    continue

                # Check if this relative import already resolves
                raw = posixpath.normpath(posixpath.join(file_dir, imp)).lstrip("/")
                candidates = [raw + e for e in (".ts", ".tsx", ".js", ".jsx",
                                                 "/index.ts", "/index.tsx")]
                candidates.append(raw)
                if any(c in existing for c in candidates):
                    continue

                # Import is broken — try to find the target file by basename
                target_name = posixpath.basename(imp)
                matches = ts_files.get(target_name, [])
                if len(matches) == 1:
                    # Compute correct relative path
                    target_rel = matches[0]
                    # Remove extension for the import
                    target_no_ext = re.sub(r"\.(tsx?|jsx?)$", "", target_rel)
                    new_imp = posixpath.relpath(target_no_ext, file_dir)
                    if not new_imp.startswith("."):
                        new_imp = "./" + new_imp
                    if new_imp != imp:
                        old_full = m.group(0)
                        new_full = m.group(1) + new_imp + m.group(3)
                        new_content = new_content.replace(old_full, new_full, 1)
                        changed = True
                        fixed_total += 1
                        logger.info(
                            "[FE Import Fix] %s: '%s' → '%s'",
                            file_path, imp, new_imp,
                        )

            if changed:
                abs_path.write_text(new_content, encoding="utf-8")

        return fixed_total

    async def _run_final_build_check(
        self,
        workspace: Path,
        blueprint: RepositoryBlueprint,
        agent_manager: AgentManager,
        event_bus: EventBus,
        component_plan: ComponentPlan,
        errors: list[str],
        metrics: dict,
        api_contract: "APIContract | None" = None,
        requirements: "ProductRequirements | None" = None,
        ordered_components: "list | None" = None,
    ) -> bool:
        """Run npm install + tsc after ALL generation phases are done.

        This catches type errors introduced by API integration, state management,
        and cross-file mismatches that Phase 4.5 (which runs before Phases 5-8)
        cannot detect.

        Returns True if the build passes (or tools are unavailable).
        """
        # Step 1: npm install (required for tsc to resolve node_modules types)
        try:
            proc = await asyncio.create_subprocess_exec(
                "npm", "install", "--ignore-scripts",
                cwd=str(workspace),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            stdout_bytes, _ = await asyncio.wait_for(proc.communicate(), timeout=120.0)
            if proc.returncode != 0:
                logger.warning(
                    "[FE Phase 9] npm install failed (rc=%d): %s",
                    proc.returncode, stdout_bytes.decode(errors="replace")[:500],
                )
                # Non-fatal — tsc may still work with partial deps
            else:
                logger.info("[FE Phase 9] npm install completed")
        except FileNotFoundError:
            logger.warning("[FE Phase 9] npm not found on PATH — skipping final build check")
            return True
        except asyncio.TimeoutError:
            logger.warning("[FE Phase 9] npm install timed out after 120s")

        # Step 2: tsc --noEmit (full type check)
        tsx_compiler = TSXCompiler()
        max_retries = self._settings.build_checkpoint_retries

        compile_result = await tsx_compiler.check(workspace)
        if not compile_result.tsc_available:
            logger.warning("[FE Phase 9] tsc not available — skipping final type check")
            return True

        if not compile_result.errors:
            logger.info("[FE Phase 9] Final build check passed — 0 errors")
            await event_bus.publish(AgentEvent(
                type=BusEventType.BUILD_PASSED,
                data={"checkpoint": "final_build"},
            ))
            return True

        # Step 3: Fix-retry loop
        for retry in range(max_retries):
            errors_by_file = compile_result.errors_by_file()
            logger.warning(
                "[FE Phase 9] %d error(s) in %d file(s) — fix attempt %d/%d",
                len(compile_result.errors), len(errors_by_file),
                retry + 1, max_retries,
            )

            fix_coros = []
            for file_path, file_errors in errors_by_file.items():
                fix_agent = agent_manager._create_agent(TaskType.FIX_CODE)
                error_text = "\n".join(
                    f"{e.file}({e.line},{e.col}): error {e.code}: {e.message}"
                    for e in file_errors
                )
                # Find the component metadata for this file
                fix_component = None
                if ordered_components:
                    for c in ordered_components:
                        if c.file_path == file_path:
                            fix_component = c
                            break

                fix_metadata: dict = {
                    "fix_trigger": "build",
                    "build_errors": error_text,
                }
                if fix_component:
                    fix_metadata["component"] = fix_component
                if component_plan:
                    fix_metadata["component_plan"] = component_plan
                if api_contract:
                    fix_metadata["api_contract"] = api_contract
                if requirements:
                    fix_metadata["requirements"] = requirements

                # Extract AST stubs of imported files (compact signatures,
                # not full source) so the fixer knows the types/exports
                # without blowing up the context window.
                related: dict[str, str] = {}
                broken_abs = workspace / file_path
                if broken_abs.exists():
                    try:
                        content = broken_abs.read_text(encoding="utf-8", errors="replace")
                        _ast = ASTExtractor()
                        import posixpath as _posixpath
                        for line in content.splitlines():
                            if "import " in line and "from " in line:
                                parts = line.split("from ")
                                if len(parts) > 1:
                                    imp_path = parts[-1].strip().strip("'\"").strip(";")
                                    if imp_path.startswith("."):
                                        file_dir = _posixpath.dirname(file_path)
                                        for ext in (".ts", ".tsx", "/index.ts", "/index.tsx"):
                                            # Normalise the relative path (collapse ..)
                                            # using posixpath to stay platform-independent
                                            candidate = _posixpath.normpath(
                                                _posixpath.join(file_dir, imp_path + ext)
                                            )
                                            abs_candidate = workspace / candidate
                                            if abs_candidate.exists():
                                                try:
                                                    raw = abs_candidate.read_text(
                                                        encoding="utf-8", errors="replace"
                                                    )
                                                    stub = _ast.extract_stub(
                                                        candidate, raw, "typescript",
                                                    )
                                                    related[candidate] = stub or raw[:2000]
                                                except OSError:
                                                    pass
                                                break
                    except OSError:
                        pass

                # Build file blueprint for the broken file
                fix_fb = None
                if fix_component:
                    fix_fb = FileBlueprint(
                        path=file_path,
                        purpose=fix_component.description or fix_component.name,
                        depends_on=fix_component.depends_on or [],
                        language="typescript",
                    )

                fix_ctx = AgentContext(
                    task=Task(
                        task_id=11000 + retry * 100 + len(fix_coros),
                        task_type=TaskType.FIX_CODE,
                        file=file_path,
                        description=f"Fix build errors in {file_path}",
                        metadata=fix_metadata,
                    ),
                    blueprint=blueprint,
                    file_blueprint=fix_fb,
                    related_files=related if related else None,
                )
                fix_coros.append(fix_agent.execute(fix_ctx))

            fix_results = await asyncio.gather(*fix_coros, return_exceptions=True)
            for fix_res in fix_results:
                if isinstance(fix_res, Exception):
                    logger.error("Fix task failed: %s", fix_res)

            compile_result = await tsx_compiler.check(workspace)
            if not compile_result.errors:
                logger.info(
                    "[FE Phase 9] Final build passed after %d fix attempt(s)", retry + 1
                )
                await event_bus.publish(AgentEvent(
                    type=BusEventType.BUILD_PASSED,
                    data={"checkpoint": "final_build"},
                ))
                return True

        # Exhausted retries
        metrics["final_build_errors"] = len(compile_result.errors)
        for err in compile_result.errors:
            errors.append(
                f"Final build error {err.file}:{err.line}: [{err.code}] {err.message}"
            )
        logger.warning(
            "[FE Phase 9] %d error(s) remain after %d fix attempts",
            len(compile_result.errors), max_retries,
        )
        await event_bus.publish(AgentEvent(
            type=BusEventType.BUILD_FAILED,
            data={
                "checkpoint": "final_build",
                "error_count": len(compile_result.errors),
            },
        ))
        return False

    @staticmethod
    def _scan_backend_models(
        backend_workspace: Path,
        backend_blueprint: "RepositoryBlueprint | None" = None,
    ) -> dict[str, str]:
        """Extract backend model/entity/DTO signatures for FE type generation.

        Two strategies (in order of preference):

        **Strategy A — Blueprint-driven** (no filesystem scan):
          Uses ``backend_blueprint.file_blueprints`` to find model files by
          ``layer`` (model, entity, dto).  If the file exists on disk (BE may
          have finished first), extracts an AST stub.  Otherwise returns the
          blueprint metadata (path, purpose, exports) — still useful for the
          LLM to generate matching TypeScript interfaces.

        **Strategy B — Filesystem discovery** (fallback):
          Uses ``find`` shell command to locate model directories, then reads
          only files from those directories.  This avoids walking the entire
          tree with ``rglob`` which is slow on deep Java/Go projects.
        """
        _EXT_TO_LANG = {
            ".java": "java", ".py": "python", ".go": "go",
            ".ts": "typescript", ".rs": "rust", ".cs": "csharp",
            ".kt": "java",
        }
        _ast = ASTExtractor()
        models: dict[str, str] = {}

        # ── Strategy A: use the blueprint we already have ──────────────────
        if backend_blueprint and backend_blueprint.file_blueprints:
            model_layers = {"model", "entity", "dto", "domain", "schema"}
            for fb in backend_blueprint.file_blueprints:
                layer = (fb.layer or "").lower()
                if layer not in model_layers:
                    continue

                # Try to read the actual generated file for AST extraction
                abs_path = backend_workspace / fb.path
                if abs_path.exists():
                    try:
                        raw = abs_path.read_text(encoding="utf-8", errors="replace")
                        lang = _EXT_TO_LANG.get(abs_path.suffix, "")
                        stub = _ast.extract_stub(fb.path, raw, lang) if lang else None
                        models[fb.path] = stub if stub else raw[:1500]
                    except OSError:
                        # File exists but unreadable — use blueprint metadata
                        models[fb.path] = (
                            f"// {fb.path} (from blueprint)\n"
                            f"// Purpose: {fb.purpose}\n"
                            f"// Exports: {', '.join(fb.exports or [])}\n"
                        )
                else:
                    # BE hasn't generated this file yet (parallel execution)
                    # — still provide the blueprint metadata so the FE LLM
                    # knows what entities exist and what they export
                    models[fb.path] = (
                        f"// {fb.path} (not yet generated — from blueprint)\n"
                        f"// Purpose: {fb.purpose}\n"
                        f"// Exports: {', '.join(fb.exports or [])}\n"
                        f"// Layer: {fb.layer}\n"
                    )

                if len(models) >= 15:
                    break

            if models:
                return models

        # ── Strategy B: filesystem discovery via find ──────────────────────
        if not backend_workspace.exists():
            return models

        # Use find to locate model directories — much faster than rglob
        # on deep Java trees (src/main/java/com/example/...)
        import subprocess
        model_dir_names = ("model", "models", "entity", "entities",
                           "dto", "dtos", "schema", "schemas", "domain")
        # Build find command: search for directories matching model names
        # -maxdepth 8 prevents runaway on deeply nested node_modules etc.
        name_args: list[str] = []
        for i, d in enumerate(model_dir_names):
            if i > 0:
                name_args.extend(["-o", "-iname", d])
            else:
                name_args.extend(["-iname", d])

        find_ok = False
        model_dirs: list[Path] = []
        try:
            result = subprocess.run(
                ["find", str(backend_workspace), "-maxdepth", "8",
                 "-type", "d", "("] + name_args + [")"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                model_dirs = [
                    Path(d.strip()) for d in result.stdout.strip().splitlines()
                    if d.strip()
                ]
                find_ok = True
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass

        if not find_ok:
            # find not available or failed (e.g. Windows find.exe) — fall back
            # to checking known relative paths
            model_dirs = [
                backend_workspace / d
                for d in model_dir_names
                if (backend_workspace / d).is_dir()
            ]
            # Also check src/main/java/*/model etc.
            for p in backend_workspace.glob("src/**/"):
                if p.name.lower() in set(model_dir_names) and p.is_dir():
                    model_dirs.append(p)

        source_exts = set(_EXT_TO_LANG.keys())
        for model_dir in model_dirs:
            if not model_dir.is_dir():
                continue
            # Read only files directly in the model directory (not recursive)
            for p in sorted(model_dir.iterdir()):
                if not p.is_file() or p.suffix not in source_exts:
                    continue
                try:
                    rel = p.relative_to(backend_workspace).as_posix()
                    raw = p.read_text(encoding="utf-8", errors="replace")
                    lang = _EXT_TO_LANG.get(p.suffix, "")
                    stub = _ast.extract_stub(rel, raw, lang) if lang else None
                    models[rel] = stub if stub else raw[:1500]
                except (OSError, ValueError):
                    pass
                if len(models) >= 15:
                    return models

        return models

    @staticmethod
    def _infer_framework(requirements: ProductRequirements) -> str:
        pref = requirements.tech_preferences.get("frontend", "").lower()
        if "vue" in pref:
            return "vue"
        if "angular" in pref:
            return "angular"
        if "react" in pref and "next" not in pref:
            return "react"
        return "nextjs"

    def _phase(self, name: str, status: str) -> None:
        if self._live:
            if status in ("completed", "done"):
                self._live.complete_phase(name)
            elif status == "failed":
                self._live.fail_phase(name)
            else:
                self._live.set_phase(name, status)
        logger.debug("Phase %s: %s", name, status)

    def _complete_phase(self, name: str) -> None:
        if self._live:
            self._live.complete_phase(name)
        logger.debug("Phase %s: done", name)

    def _fail_phase(self, name: str, reason: str) -> None:
        logger.error("Phase %s failed: %s", name, reason)
        if self._live:
            self._live.fail_phase(name, reason)

    # ── Config file generation ─────────────────────────────────────────────────

    @staticmethod
    def _write_config_files(
        workspace: Path,
        plan: ComponentPlan,
        requirements: "ProductRequirements",
    ) -> None:
        """Write package.json and .env.local to the workspace.

        package.json is sourced from the LLM-generated ``plan.package_json``.
        If the LLM returned an empty dict, a sensible framework-specific
        default is used so the workspace is always bootable.

        .env.local contains placeholder environment variables that the
        developer needs to fill in before running the app.
        """
        # ── package.json ─────────────────────────────────────────────────────
        pkg = dict(plan.package_json) if plan.package_json else {}
        if not pkg.get("name"):
            pkg["name"] = requirements.title.lower().replace(" ", "-")
        if not pkg.get("version"):
            pkg["version"] = "0.1.0"
        if not pkg.get("private"):
            pkg["private"] = True

        # Ensure minimal framework scripts are present
        fw = plan.framework.lower()
        if "next" in fw:
            pkg.setdefault("scripts", {
                "dev": "next dev",
                "build": "next build",
                "start": "next start",
                "lint": "next lint",
                "type-check": "tsc --noEmit",
            })
            deps = pkg.setdefault("dependencies", {})
            deps.setdefault("next", "^14.0.0")
            deps.setdefault("react", "^18.0.0")
            deps.setdefault("react-dom", "^18.0.0")
            deps.setdefault("axios", "^1.7.0")
            deps.setdefault("swr", "^2.2.0")
            dev_deps = pkg.setdefault("devDependencies", {})
            dev_deps.setdefault("@types/node", "^20.0.0")
            dev_deps.setdefault("@types/react", "^18.0.0")
            dev_deps.setdefault("typescript", "^5.0.0")
            if requirements.tech_preferences.get("styling", "").lower() == "tailwind":
                dev_deps.setdefault("tailwindcss", "^3.4.0")
                dev_deps.setdefault("autoprefixer", "^10.0.0")
                dev_deps.setdefault("postcss", "^8.0.0")
        elif "vue" in fw:
            pkg.setdefault("scripts", {
                "dev": "vite",
                "build": "vue-tsc && vite build",
                "preview": "vite preview",
                "type-check": "vue-tsc --noEmit",
            })
            deps = pkg.setdefault("dependencies", {})
            deps.setdefault("vue", "^3.4.0")
            deps.setdefault("vue-router", "^4.3.0")
            deps.setdefault("pinia", "^2.1.0")
            deps.setdefault("axios", "^1.7.0")
            dev_deps = pkg.setdefault("devDependencies", {})
            dev_deps.setdefault("vite", "^5.0.0")
            dev_deps.setdefault("@vitejs/plugin-vue", "^5.0.0")
            dev_deps.setdefault("vue-tsc", "^2.0.0")
            dev_deps.setdefault("typescript", "^5.0.0")
            if requirements.tech_preferences.get("styling", "").lower() == "tailwind":
                dev_deps.setdefault("tailwindcss", "^3.4.0")
                dev_deps.setdefault("autoprefixer", "^10.0.0")
                dev_deps.setdefault("postcss", "^8.0.0")
        elif "angular" in fw:
            pkg.setdefault("scripts", {"start": "ng serve", "build": "ng build"})
            pkg.setdefault("dependencies", {"@angular/core": "^17.0.0"})
        else:
            pkg.setdefault("scripts", {"start": "react-scripts start"})
            pkg.setdefault("dependencies", {"react": "^18.0.0", "react-dom": "^18.0.0"})

        # State management dependency
        state = plan.state_solution.lower()
        if "zustand" in state:
            pkg.setdefault("dependencies", {})["zustand"] = pkg.get(
                "dependencies", {}
            ).get("zustand", "^4.5.0")
            pkg.setdefault("dependencies", {}).setdefault("immer", "^10.0.0")
        elif "redux" in state:
            pkg.setdefault("dependencies", {}).setdefault("@reduxjs/toolkit", "^2.0.0")
            pkg.setdefault("dependencies", {}).setdefault("react-redux", "^9.0.0")

        pkg_path = workspace / "package.json"
        pkg_path.write_text(_json.dumps(pkg, indent=2), encoding="utf-8")
        logger.info("Wrote package.json (%d deps)", len(pkg.get("dependencies", {})))

        # ── .env.local ────────────────────────────────────────────────────────
        api_base = plan.api_base_url or "/api/v1"
        if "next" in fw:
            env_prefix = "NEXT_PUBLIC_"
        elif "vue" in fw:
            env_prefix = "VITE_"
        elif "angular" in fw:
            env_prefix = ""  # Angular uses environments/*.ts, not env vars
        else:
            env_prefix = "REACT_APP_"
        env_content = (
            "# Auto-generated by codegen frontend pipeline\n"
            "# Fill in the values before running the application.\n\n"
            f"{env_prefix}API_BASE_URL={api_base}\n"
            f"{env_prefix}APP_NAME={requirements.title}\n"
        )
        env_path = workspace / ".env.local"
        env_path.write_text(env_content, encoding="utf-8")
        logger.info("Wrote .env.local")

        # ── globals.css ───────────────────────────────────────────────────
        styling = requirements.tech_preferences.get("styling", "tailwind").lower()
        tailwind_directives = (
            "@tailwind base;\n"
            "@tailwind components;\n"
            "@tailwind utilities;\n\n"
        )
        base_css = (
            ":root {\n"
            "  --foreground-rgb: 0, 0, 0;\n"
            "  --background-rgb: 255, 255, 255;\n"
            "}\n\n"
            "body {\n"
            "  color: rgb(var(--foreground-rgb));\n"
            "  background: rgb(var(--background-rgb));\n"
            "  font-family: system-ui, -apple-system, sans-serif;\n"
            "}\n"
        )
        if "tailwind" in styling:
            globals_css = tailwind_directives + base_css
        else:
            globals_css = (
                "*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }\n"
                "body {\n"
                "  font-family: system-ui, -apple-system, sans-serif;\n"
                "  line-height: 1.5;\n"
                "}\n"
            )

        # Framework-specific CSS location
        if "vue" in fw:
            css_dir = workspace / "src" / "assets"
            css_file = "main.css"
        elif "next" in fw:
            css_dir = workspace / "src" / "app"
            css_file = "globals.css"
        else:
            css_dir = workspace / "src"
            css_file = "index.css"
        css_dir.mkdir(parents=True, exist_ok=True)
        (css_dir / css_file).write_text(globals_css, encoding="utf-8")
        logger.info("Wrote %s/%s", css_dir.relative_to(workspace), css_file)

        # ── tsconfig.json (framework-aware) ──────────────────────────────
        if "vue" in fw:
            tsconfig = {
                "compilerOptions": {
                    "target": "ES2020",
                    "useDefineForClassFields": True,
                    "module": "ESNext",
                    "lib": ["ES2020", "DOM", "DOM.Iterable"],
                    "skipLibCheck": True,
                    "moduleResolution": "bundler",
                    "allowImportingTsExtensions": True,
                    "resolveJsonModule": True,
                    "isolatedModules": True,
                    "noEmit": True,
                    "jsx": "preserve",
                    "strict": True,
                    "noUnusedLocals": False,
                    "noUnusedParameters": False,
                    "noFallthroughCasesInSwitch": True,
                    "paths": {"@/*": ["./src/*"]},
                },
                "include": ["src/**/*.ts", "src/**/*.tsx", "src/**/*.vue"],
                "references": [{"path": "./tsconfig.node.json"}],
            }
            # Vue also needs tsconfig.node.json for vite config
            tsconfig_node = {
                "compilerOptions": {
                    "composite": True,
                    "skipLibCheck": True,
                    "module": "ESNext",
                    "moduleResolution": "bundler",
                    "allowSyntheticDefaultImports": True,
                },
                "include": ["vite.config.ts"],
            }
            (workspace / "tsconfig.node.json").write_text(
                _json.dumps(tsconfig_node, indent=2), encoding="utf-8"
            )
        elif "angular" in fw:
            tsconfig = {
                "compilerOptions": {
                    "target": "ES2022",
                    "module": "ES2022",
                    "lib": ["ES2022", "dom"],
                    "strict": True,
                    "esModuleInterop": True,
                    "moduleResolution": "node",
                    "paths": {"@/*": ["./src/*"]},
                },
                "include": ["src/**/*.ts"],
            }
        else:
            # Next.js / React
            tsconfig = {
                "compilerOptions": {
                    "target": "es5",
                    "lib": ["dom", "dom.iterable", "esnext"],
                    "allowJs": True,
                    "skipLibCheck": True,
                    "strict": True,
                    "noEmit": True,
                    "esModuleInterop": True,
                    "module": "esnext",
                    "moduleResolution": "bundler",
                    "resolveJsonModule": True,
                    "isolatedModules": True,
                    "jsx": "preserve",
                    "incremental": True,
                    "paths": {"@/*": ["./src/*"]},
                },
                "include": ["**/*.ts", "**/*.tsx"],
                "exclude": ["node_modules"],
            }
            if "next" in fw:
                tsconfig["compilerOptions"]["plugins"] = [{"name": "next"}]
                tsconfig["include"] = [
                    "next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts",
                ]

        tsconfig_path = workspace / "tsconfig.json"
        tsconfig_path.write_text(_json.dumps(tsconfig, indent=2), encoding="utf-8")
        logger.info("Wrote tsconfig.json")

        # ── Framework-specific config ─────────────────────────────────────
        if "next" in fw:
            next_config = (
                "/** @type {import('next').NextConfig} */\n"
                "const nextConfig = {};\n\n"
                "module.exports = nextConfig;\n"
            )
            (workspace / "next.config.js").write_text(next_config, encoding="utf-8")
            logger.info("Wrote next.config.js")

            # Next.js App Router requires a root layout — without it, every
            # page.tsx triggers "doesn't have a root layout" and the build fails.
            app_dir = workspace / "src" / "app"
            app_dir.mkdir(parents=True, exist_ok=True)
            layout_path = app_dir / "layout.tsx"
            if not layout_path.exists():
                app_name = requirements.title or "App"
                layout_tsx = (
                    "import type { Metadata } from 'next';\n"
                    "import './globals.css';\n\n"
                    "export const metadata: Metadata = {\n"
                    f"  title: '{app_name}',\n"
                    f"  description: '{app_name}',\n"
                    "};\n\n"
                    "export default function RootLayout({\n"
                    "  children,\n"
                    "}: {\n"
                    "  children: React.ReactNode;\n"
                    "}) {\n"
                    "  return (\n"
                    '    <html lang="en">\n'
                    "      <body>{children}</body>\n"
                    "    </html>\n"
                    "  );\n"
                    "}\n"
                )
                layout_path.write_text(layout_tsx, encoding="utf-8")
                logger.info("Wrote src/app/layout.tsx")
        elif "vue" in fw:
            vite_config = (
                "import { defineConfig } from 'vite';\n"
                "import vue from '@vitejs/plugin-vue';\n"
                "import { fileURLToPath, URL } from 'node:url';\n\n"
                "export default defineConfig({\n"
                "  plugins: [vue()],\n"
                "  resolve: {\n"
                "    alias: {\n"
                "      '@': fileURLToPath(new URL('./src', import.meta.url)),\n"
                "    },\n"
                "  },\n"
                "});\n"
            )
            (workspace / "vite.config.ts").write_text(vite_config, encoding="utf-8")
            logger.info("Wrote vite.config.ts")

            # Vue entry point: index.html
            index_html = (
                '<!DOCTYPE html>\n<html lang="en">\n<head>\n'
                '  <meta charset="UTF-8" />\n'
                '  <meta name="viewport" content="width=device-width, initial-scale=1.0" />\n'
                f'  <title>{requirements.title}</title>\n'
                '</head>\n<body>\n'
                '  <div id="app"></div>\n'
                '  <script type="module" src="/src/main.ts"></script>\n'
                '</body>\n</html>\n'
            )
            (workspace / "index.html").write_text(index_html, encoding="utf-8")
            logger.info("Wrote index.html")

            # Vue main.ts entry point
            src_dir = workspace / "src"
            src_dir.mkdir(parents=True, exist_ok=True)
            main_ts = (
                "import { createApp } from 'vue';\n"
                "import { createPinia } from 'pinia';\n"
                "import App from './App.vue';\n"
                "import router from './router';\n"
                "import './assets/main.css';\n\n"
                "const app = createApp(App);\n"
                "app.use(createPinia());\n"
                "app.use(router);\n"
                "app.mount('#app');\n"
            )
            (src_dir / "main.ts").write_text(main_ts, encoding="utf-8")
            logger.info("Wrote src/main.ts")

        # ── tailwind.config.js + postcss.config.js ────────────────────────
        if "tailwind" in styling:
            if "vue" in fw:
                content_globs = (
                    "    './index.html',\n"
                    "    './src/**/*.{vue,js,ts,jsx,tsx}',\n"
                )
            else:
                content_globs = (
                    "    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',\n"
                    "    './src/components/**/*.{js,ts,jsx,tsx,mdx}',\n"
                    "    './src/app/**/*.{js,ts,jsx,tsx,mdx}',\n"
                )
            tailwind_config = (
                "/** @type {import('tailwindcss').Config} */\n"
                "module.exports = {\n"
                "  content: [\n"
                f"{content_globs}"
                "  ],\n"
                "  theme: { extend: {} },\n"
                "  plugins: [],\n"
                "};\n"
            )
            (workspace / "tailwind.config.js").write_text(tailwind_config, encoding="utf-8")
            logger.info("Wrote tailwind.config.js")

            postcss_config = (
                "module.exports = {\n"
                "  plugins: {\n"
                "    tailwindcss: {},\n"
                "    autoprefixer: {},\n"
                "  },\n"
                "};\n"
            )
            (workspace / "postcss.config.js").write_text(postcss_config, encoding="utf-8")
            logger.info("Wrote postcss.config.js")

    # ── Ghost dependency detection ─────────────────────────────────────────────

    @staticmethod
    def _find_ghost_dependencies(
        planned_components: "list",
        planned_names: "set[str]",
        *,
        framework: str = "nextjs",
    ) -> "list":
        """Return UIComponent stubs for every component name referenced in
        ``depends_on`` that is not present in ``planned_names``.

        Ghost components appear in the dependency graph but were never added
        to the ComponentPlan (e.g. UserDropdown, Alert, ProgressBar).
        Without stubs, they become dead nodes and the importing file ends up
        with a broken import at runtime.

        File-path and type are inferred from naming conventions:
          - Dropdown / Modal / Menu / Popover → components/ui/
          - Alert / Badge / Tag / Chip / Toast → components/ui/
          - Layout / Header / Footer / Sidebar / Nav → components/layout/
          - Everything else → components/shared/
        """
        from core.models import UIComponent  # local import to avoid circular

        _UI_KEYWORDS = {"dropdown", "modal", "menu", "popover", "alert",
                        "badge", "tag", "chip", "toast", "button", "input",
                        "card", "avatar", "icon", "spinner", "skeleton",
                        "progressbar", "progress", "list", "table", "tabs"}
        _LAYOUT_KEYWORDS = {"layout", "header", "footer", "sidebar", "nav",
                             "navbar", "drawer", "panel"}

        referenced: dict[str, str] = {}  # name -> implied_by (first referrer)
        for comp in planned_components:
            for dep in comp.depends_on:
                if dep not in planned_names and dep not in referenced:
                    referenced[dep] = comp.name

        ext = ".vue" if "vue" in framework.lower() else ".tsx"
        ghosts: list = []
        for name, implied_by in referenced.items():
            lower = name.lower()
            if any(kw in lower for kw in _LAYOUT_KEYWORDS):
                layer = "layout"
                folder = "components/layout"
            elif any(kw in lower for kw in _UI_KEYWORDS):
                layer = "ui"
                folder = "components/ui"
            else:
                layer = "shared"
                folder = "components/shared"

            file_path = f"src/{folder}/{name}{ext}"
            ghosts.append(
                UIComponent(
                    name=name,
                    file_path=file_path,
                    component_type=layer,
                    description=f"Auto-stub for missing dependency '{name}' (required by {implied_by})",
                    layer=folder,
                )
            )
        return ghosts
