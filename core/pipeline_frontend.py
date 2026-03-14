"""Frontend pipeline — generates a complete UI codebase from design spec + API contract."""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from agents.design_parser_agent import DesignParserAgent
from agents.component_planner_agent import ComponentPlannerAgent
from agents.component_dag_agent import ComponentDAGAgent
from agents.component_generator_agent import ComponentGeneratorAgent
from agents.api_integration_agent import APIIntegrationAgent
from agents.state_management_agent import StateManagementAgent
from config.settings import Settings
from core.import_validator import ImportValidator
from core.llm_client import LLMClient
from core.tsx_compiler import TSXCompiler
from core.models import (
    APIContract,
    ComponentPlan,
    FileBlueprint,
    ProductRequirements,
    RepositoryBlueprint,
    Task,
    TaskType,
    UIDesignSpec,
    AgentContext,
)
from core.repository_manager import RepositoryManager
from core.workspace_indexer import index_workspace
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
      5. API Integration     — typed API client + SWR/React-Query hooks
      6. State Management    — Zustand/Redux/Pinia store layer

    The pipeline is intentionally self-contained: it does NOT depend on
    the backend RunPipeline and can be run standalone or in parallel.
    """

    def __init__(
        self,
        settings: Settings,
        llm: LLMClient,
        live: LiveConsole | None = None,
        root_write_lock: asyncio.Lock | None = None,
    ) -> None:
        self._settings = settings
        self._llm = llm
        self._live = live
        self._root_write_lock = root_write_lock

    # ── Public entry point ────────────────────────────────────────────────────

    async def execute(
        self,
        requirements: ProductRequirements,
        api_contract: APIContract | None,
        start_time: float,
        figma_url: str = "",
        frontend_workspace: Path | None = None,
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

        # Initialize RepositoryManager with a synthetic frontend blueprint
        frontend_blueprint = self._make_frontend_blueprint(requirements, component_plan)
        repo_manager.initialize(frontend_blueprint)
        metrics["components_planned"] = len(component_plan.components)

        # ── Write package.json and .env.local ─────────────────────────────────
        self._write_config_files(workspace, component_plan, requirements)

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
        ghost_components = self._find_ghost_dependencies(ordered_components, planned_names)
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
            frontend_blueprint = self._make_frontend_blueprint(requirements, component_plan)
        max_tier = max(tier_map.values()) + 1 if tier_map else 1

        logger.info("[FE Phase 4] Generating %d components...", len(ordered_components))
        generator = ComponentGeneratorAgent(llm_client=self._llm, repo_manager=repo_manager, mcp_client=mcp_client)
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
                elif not res.success:
                    gen_errors.append(f"{comp.name}: {'; '.join(res.errors)}")
                    logger.warning("Component generation incomplete for %s", comp.name)
                else:
                    logger.debug("Generated component: %s", comp.file_path)

        if gen_errors:
            errors.extend(gen_errors)
            logger.warning("%d component generation errors", len(gen_errors))
        metrics["components_generated"] = len(ordered_components) - len(gen_errors)
        self._complete_phase("FE: Component Generation")

        # ── Phase 4.5: TypeScript Compilation Check ───────────────────────────
        self._phase("FE: TypeScript Compilation", "running")
        logger.info("[FE Phase 4.5] Running TypeScript compilation check...")
        tsx_compiler = TSXCompiler()
        compile_result = await tsx_compiler.check(workspace)
        if not compile_result.tsc_available:
            logger.info(
                "[FE Phase 4.5] tsc not on PATH — skipping TypeScript compilation check"
            )
        elif compile_result.errors:
            metrics["tsx_compile_errors"] = len(compile_result.errors)
            for err in compile_result.errors:
                errors.append(
                    f"TSX compile error {err.file}:{err.line}: [{err.code}] {err.message}"
                )
            logger.warning(
                "[FE Phase 4.5] %d TypeScript compile error(s)", len(compile_result.errors)
            )
        else:
            logger.info("[FE Phase 4.5] TypeScript compilation passed")
        self._complete_phase("FE: TypeScript Compilation")

        # ── Phase 4.6: Cross-component import validation ───────────────────────
        self._phase("FE: Import Validation", "running")
        logger.info("[FE Phase 4.6] Validating cross-component imports...")
        import_errors = self._validate_component_imports(workspace, ordered_components)
        if import_errors:
            metrics["import_errors"] = len(import_errors)
            errors.extend(import_errors)
            logger.warning("[FE Phase 4.6] %d unresolved import(s)", len(import_errors))
        else:
            logger.info("[FE Phase 4.6] All component imports resolved")
        self._complete_phase("FE: Import Validation")

        # ── Phase 5: API Integration ──────────────────────────────────────────
        self._phase("FE: API Integration", "running")
        logger.info("[FE Phase 5] Generating API client layer...")
        try:
            integrator = APIIntegrationAgent(llm_client=self._llm, repo_manager=repo_manager)
            api_ctx = AgentContext(
                task=Task(
                    task_id=9000,
                    task_type=TaskType.INTEGRATE_API,
                    file="src/lib/api.ts",
                    description="Generate typed API client and hooks",
                    metadata={
                        "api_contract": api_contract,
                        "component_plan": component_plan,
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
            if not api_result.success:
                errors.extend(api_result.errors)
            self._complete_phase("FE: API Integration")
        except Exception as exc:
            logger.exception("API integration failed")
            errors.append(f"API integration failed: {exc}")
            self._fail_phase("FE: API Integration", str(exc))

        # ── Phase 6: State Management ─────────────────────────────────────────
        self._phase("FE: State Management", "running")
        logger.info("[FE Phase 6] Generating state management layer (%s)...",
                    component_plan.state_solution)
        try:
            state_agent = StateManagementAgent(llm_client=self._llm, repo_manager=repo_manager)
            state_ctx = AgentContext(
                task=Task(
                    task_id=9001,
                    task_type=TaskType.MANAGE_STATE,
                    file="src/store/index.ts",
                    description=f"Generate {component_plan.state_solution} store layer",
                    metadata={"component_plan": component_plan},
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
            if not state_result.success:
                errors.extend(state_result.errors)
            self._complete_phase("FE: State Management")
        except Exception as exc:
            logger.exception("State management generation failed")
            errors.append(f"State management failed: {exc}")
            self._fail_phase("FE: State Management", str(exc))

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
        finally:
            if mcp_client:
                await mcp_client.close()

        success = not any(
            e for e in errors
            if "Component planning failed" in e
        )
        return PipelineResult(
            success=success,
            workspace_path=workspace,
            blueprint=frontend_blueprint,
            metrics=metrics,
            errors=errors,
            elapsed_seconds=time.monotonic() - start_time,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _make_frontend_blueprint(
        self,
        requirements: ProductRequirements,
        plan: ComponentPlan,
    ) -> RepositoryBlueprint:
        """Build a RepositoryBlueprint from the ComponentPlan for RepositoryManager."""
        file_blueprints = [
            FileBlueprint(
                path=comp.file_path,
                purpose=comp.description,
                depends_on=comp.depends_on,
                exports=[comp.name],
                language="typescript",
                layer=comp.layer or comp.component_type,
            )
            for comp in plan.components
        ]
        fw = plan.framework
        tech_stack: dict[str, str] = {
            "language": "typescript",
            "framework": fw,
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
        import json as _json

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
            dev_deps = pkg.setdefault("devDependencies", {})
            dev_deps.setdefault("@types/node", "^20.0.0")
            dev_deps.setdefault("@types/react", "^18.0.0")
            dev_deps.setdefault("typescript", "^5.0.0")
            if requirements.tech_preferences.get("styling", "").lower() == "tailwind":
                dev_deps.setdefault("tailwindcss", "^3.4.0")
                dev_deps.setdefault("autoprefixer", "^10.0.0")
                dev_deps.setdefault("postcss", "^8.0.0")
        elif "vue" in fw:
            pkg.setdefault("scripts", {"dev": "vite", "build": "vite build"})
            pkg.setdefault("dependencies", {"vue": "^3.4.0"})
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
        elif "redux" in state:
            pkg.setdefault("dependencies", {}).setdefault("@reduxjs/toolkit", "^2.0.0")
            pkg.setdefault("dependencies", {}).setdefault("react-redux", "^9.0.0")

        pkg_path = workspace / "package.json"
        pkg_path.write_text(_json.dumps(pkg, indent=2), encoding="utf-8")
        logger.info("Wrote package.json (%d deps)", len(pkg.get("dependencies", {})))

        # ── .env.local ────────────────────────────────────────────────────────
        api_base = plan.api_base_url or "/api/v1"
        env_content = (
            "# Auto-generated by codegen frontend pipeline\n"
            "# Fill in the values before running the application.\n\n"
            f"NEXT_PUBLIC_API_BASE_URL={api_base}\n"
            "NEXT_PUBLIC_APP_NAME=" + requirements.title + "\n"
            "# NEXT_PUBLIC_AUTH_URL=https://your-auth-service.example.com\n"
            "# DATABASE_URL=postgresql://user:password@localhost:5432/dbname\n"
            "# JWT_SECRET=change-me-in-production\n"
        )
        env_path = workspace / ".env.local"
        env_path.write_text(env_content, encoding="utf-8")
        logger.info("Wrote .env.local")

    # ── Ghost dependency detection ─────────────────────────────────────────────

    @staticmethod
    def _find_ghost_dependencies(
        planned_components: "list",
        planned_names: "set[str]",
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

            ext = ".tsx"
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
