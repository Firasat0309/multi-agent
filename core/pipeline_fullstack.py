"""Fullstack pipeline — orchestrates Product Planner → Architect → parallel BE + FE."""

from __future__ import annotations

import asyncio
import json as _json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from agents.architect_agent import ArchitectAgent
from agents.api_contract_agent import APIContractAgent
from agents.product_planner_agent import ProductPlannerAgent
from config.settings import Settings
from core.llm_client import LLMClient
from core.models import (
    APIContract,
    FullstackBlueprint,
    ProductRequirements,
    RepositoryBlueprint,
)
from core.repository_manager import RepositoryManager

if TYPE_CHECKING:
    from core.live_console import LiveConsole
    from core.pipeline import PipelineResult

logger = logging.getLogger(__name__)


class FullstackPipeline:
    """Runs the complete fullstack generation workflow.

    Architecture:

        User Prompt
            │
            ▼
        ProductPlannerAgent     (requirements extraction)
            │
            ▼
        ArchitectAgent          (backend RepositoryBlueprint)
            │
            ▼
        APIContractAgent        (OpenAPI contract)
            │
            ├─────────────────────────────────────┐
            ▼                                     ▼
        RunPipeline (backend)              FrontendPipeline (frontend)
            │                                     │
            └──────────────── merge ──────────────┘
                                │
                                ▼
                        FullstackPipelineResult

    The backend and frontend pipelines run concurrently via asyncio.gather
    once the API contract is available.
    """

    def __init__(
        self,
        settings: Settings,
        llm: LLMClient,
        live: LiveConsole | None = None,
        interactive: bool = True,
    ) -> None:
        self._settings = settings
        self._llm = llm
        self._live = live
        self._interactive = interactive
        # Shared lock serialises concurrent writes to root-level workspace files
        # (e.g. docker-compose.yml, .gitignore) from the parallel BE+FE pipelines.
        self._root_write_lock: asyncio.Lock = asyncio.Lock()

    # ── Public entry point ────────────────────────────────────────────────────

    async def execute(
        self,
        user_prompt: str,
        start_time: float,
        figma_url: str = "",
    ) -> PipelineResult:
        """Run the complete fullstack pipeline."""
        from core.pipeline import PipelineResult
        from core.pipeline_run import RunPipeline
        from core.pipeline_frontend import FrontendPipeline

        errors: list[str] = []
        fullstack_blueprint = FullstackBlueprint()

        # ── Phase 1: Product Planning ─────────────────────────────────────────
        self._phase("Product Planning", "running")
        logger.info("[FS Phase 1] Planning product requirements...")

        root_workspace = self._settings.workspace_dir
        shared_repo_manager = RepositoryManager(root_workspace)

        requirements: ProductRequirements | None = None
        try:
            planner = ProductPlannerAgent(
                llm_client=self._llm, repo_manager=shared_repo_manager
            )
            requirements = await planner.plan_product(user_prompt)
            fullstack_blueprint.product_requirements = requirements
            logger.info(
                "Product planned: %s (FE=%s, BE=%s)",
                requirements.title, requirements.has_frontend, requirements.has_backend,
            )
            self._complete_phase("Product Planning")
        except Exception as exc:
            logger.exception("Product planning failed")
            errors.append(f"Product planning failed: {exc}")
            self._fail_phase("Product Planning", str(exc))
            return PipelineResult(
                success=False,
                workspace_path=root_workspace,
                errors=errors,
                elapsed_seconds=time.monotonic() - start_time,
            )

        # ── Approval Gate 1: Product Plan ────────────────────────────────────
        if self._settings.require_architecture_approval:
            from core.architecture_approver import (
                ArchitectureApprover,
                ArchitecturePendingApprovalError,
            )
            approver = ArchitectureApprover(
                interactive=self._interactive,
                workspace=root_workspace,
                live=self._live,
            )
            try:
                approved = approver.approve_product_plan(requirements)
            except ArchitecturePendingApprovalError as e:
                logger.info("Product plan pending human approval: %s", e)
                return PipelineResult(
                    success=False,
                    workspace_path=root_workspace,
                    errors=[str(e)],
                    elapsed_seconds=time.monotonic() - start_time,
                )
            if not approved:
                logger.info("Product plan rejected by user")
                return PipelineResult(
                    success=False,
                    workspace_path=root_workspace,
                    errors=["Product plan rejected by user"],
                    elapsed_seconds=time.monotonic() - start_time,
                )
            logger.info("Product plan approved by user")

        # ── Phase 2: Backend Architecture Design ──────────────────────────────
        self._phase("Backend Architecture", "running")
        logger.info("[FS Phase 2] Designing backend architecture...")

        backend_blueprint: RepositoryBlueprint | None = None
        try:
            # Enrich the user prompt with the structured requirements
            enriched_prompt = self._enrich_prompt(user_prompt, requirements)
            architect = ArchitectAgent(
                llm_client=self._llm, repo_manager=shared_repo_manager
            )
            backend_blueprint = await architect.design_architecture(enriched_prompt)
            fullstack_blueprint.backend_blueprint = backend_blueprint
            logger.info(
                "Backend blueprint: %s (%d files)",
                backend_blueprint.name, len(backend_blueprint.file_blueprints),
            )
            self._complete_phase("Backend Architecture")
        except Exception as exc:
            logger.exception("Backend architecture design failed")
            errors.append(f"Backend architecture failed: {exc}")
            self._fail_phase("Backend Architecture", str(exc))
            return PipelineResult(
                success=False,
                workspace_path=root_workspace,
                errors=errors,
                elapsed_seconds=time.monotonic() - start_time,
            )

        # ── Approval Gate 2: Backend Architecture ────────────────────────────
        if self._settings.require_architecture_approval:
            from core.architecture_approver import (
                ArchitectureApprover,
                ArchitecturePendingApprovalError,
            )
            approver = ArchitectureApprover(
                interactive=self._interactive,
                workspace=root_workspace,
                live=self._live,
            )
            try:
                approved = approver.approve_backend_architecture(backend_blueprint)
            except ArchitecturePendingApprovalError as e:
                logger.info("Backend architecture pending human approval: %s", e)
                return PipelineResult(
                    success=False,
                    workspace_path=root_workspace,
                    errors=[str(e)],
                    elapsed_seconds=time.monotonic() - start_time,
                )
            if not approved:
                logger.info("Backend architecture rejected by user")
                return PipelineResult(
                    success=False,
                    workspace_path=root_workspace,
                    errors=["Backend architecture rejected by user"],
                    elapsed_seconds=time.monotonic() - start_time,
                )
            logger.info("Backend architecture approved by user")

        # ── Phase 3: API Contract Generation ─────────────────────────────────
        self._phase("API Contract Generation", "running")
        logger.info("[FS Phase 3] Generating API contract...")

        api_contract: APIContract | None = None
        try:
            contract_agent = APIContractAgent(
                llm_client=self._llm, repo_manager=shared_repo_manager
            )
            api_contract = await contract_agent.generate_contract(
                requirements, backend_blueprint
            )
            if not api_contract.endpoints:
                raise ValueError(
                    "API contract generation returned 0 endpoints "
                    "(likely truncated/unparseable LLM response)"
                )
            fullstack_blueprint.api_contract = api_contract
            logger.info(
                "API contract: %s (%d endpoints)",
                api_contract.title, len(api_contract.endpoints),
            )
            # Persist the contract to disk for inspection and downstream use
            contract_path = self._settings.workspace_dir / "api_contract.json"
            try:
                contract_data = {
                    "title": api_contract.title,
                    "version": api_contract.version,
                    "base_url": api_contract.base_url,
                    "contract_format": api_contract.contract_format,
                    "endpoints": [
                        {
                            "path": ep.path,
                            "method": ep.method,
                            "description": ep.description,
                            "auth_required": ep.auth_required,
                            "tags": ep.tags,
                        }
                        for ep in api_contract.endpoints
                    ],
                    "schemas": api_contract.schemas,
                }
                contract_path.write_text(
                    _json.dumps(contract_data, indent=2), encoding="utf-8"
                )
                logger.info("Wrote api_contract.json (%d endpoints)", len(api_contract.endpoints))
            except Exception:
                logger.warning("Failed to write api_contract.json", exc_info=True)
            self._complete_phase("API Contract Generation")
        except Exception as exc:
            logger.exception("API contract generation failed")
            errors.append(f"API contract generation failed: {exc}")
            self._fail_phase("API Contract Generation", str(exc))
            return PipelineResult(
                success=False,
                workspace_path=self._settings.workspace_dir,
                errors=errors,
                elapsed_seconds=time.monotonic() - start_time,
            )

        # ── Phase 4: Parallel Backend + Frontend ──────────────────────────────
        self._phase("Parallel BE+FE Generation", "running")
        logger.info("[FS Phase 4] Running backend and frontend pipelines in parallel...")

        # Backend workspace: workspace/backend
        backend_settings = self._settings.model_copy(
            update={"workspace_dir": root_workspace / "backend"}
        ) if hasattr(self._settings, "model_copy") else _override_workspace(
            self._settings, root_workspace / "backend"
        )

        # Frontend workspace: workspace/frontend
        frontend_workspace = root_workspace / "frontend"

        backend_pipeline = RunPipeline(
            backend_settings, self._llm, self._live,
            root_write_lock=self._root_write_lock,
        )
        frontend_pipeline = FrontendPipeline(
            self._settings, self._llm, self._live,
            root_write_lock=self._root_write_lock,
            interactive=self._interactive,
        )

        # Build the enriched prompt that carries all context for the backend
        backend_prompt = (
            f"{enriched_prompt}\n\n"
            f"API contract base URL: {api_contract.base_url if api_contract else '/api/v1'}\n"
            f"Endpoints to implement: "
            + (
                ", ".join(f"{ep.method} {ep.path}" for ep in api_contract.endpoints)
                if api_contract else "per architecture"
            )
        )

        be_task: asyncio.Task | None = None
        fe_task: asyncio.Task | None = None

        coroutines = []
        labels = []

        if requirements.has_backend:
            coroutines.append(
                backend_pipeline.execute(backend_prompt, start_time)
            )
            labels.append("backend")

        if requirements.has_frontend:
            coroutines.append(
                frontend_pipeline.execute(
                    requirements,
                    api_contract,
                    start_time,
                    figma_url=figma_url,
                    frontend_workspace=frontend_workspace,
                    backend_blueprint=backend_blueprint,
                )
            )
            labels.append("frontend")

        results = await asyncio.gather(*coroutines, return_exceptions=True)

        be_result = None
        fe_result = None
        for label, res in zip(labels, results):
            if isinstance(res, Exception):
                errors.append(f"{label} pipeline failed: {res}")
                logger.error("%s pipeline raised exception: %s", label, res)
            elif label == "backend":
                be_result = res
                if not res.success:
                    errors.extend([f"BE: {e}" for e in res.errors])
            elif label == "frontend":
                fe_result = res
                if not res.success:
                    errors.extend([f"FE: {e}" for e in res.errors])

        self._complete_phase("Parallel BE+FE Generation")

        # ── Aggregate results ─────────────────────────────────────────────────
        be_ok = (not requirements.has_backend) or (be_result is not None and be_result.success)
        fe_ok = (not requirements.has_frontend) or (fe_result is not None and fe_result.success)
        success = be_ok and fe_ok

        task_stats: dict = {}
        if be_result:
            task_stats.update({f"be_{k}": v for k, v in be_result.task_stats.items()})
        if fe_result:
            task_stats.update({f"fe_{k}": v for k, v in fe_result.metrics.items()})

        elapsed = time.monotonic() - start_time
        logger.info(
            "Fullstack pipeline complete in %.1fs — success=%s",
            elapsed, success,
        )

        return PipelineResult(
            success=success,
            workspace_path=root_workspace,
            blueprint=backend_blueprint,
            task_stats=task_stats,
            metrics={
                "fullstack": True,
                "has_frontend": requirements.has_frontend,
                "has_backend": requirements.has_backend,
                "api_endpoints": len(api_contract.endpoints) if api_contract else 0,
            },
            errors=errors,
            elapsed_seconds=elapsed,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _enrich_prompt(user_prompt: str, req: ProductRequirements) -> str:
        """Append structured requirements context to the user prompt."""
        lines = [user_prompt, "", "=== Product Requirements ==="]
        lines.append(f"Title: {req.title}")
        lines.append(f"Description: {req.description}")
        if req.features:
            lines.append("Features: " + ", ".join(req.features))
        if req.tech_preferences:
            # Filter out 'none' values so downstream agents use their own defaults
            effective_prefs = {k: v for k, v in req.tech_preferences.items()
                              if v.lower() != "none"}
            if effective_prefs:
                prefs = "; ".join(f"{k}={v}" for k, v in effective_prefs.items())
                lines.append(f"Tech preferences: {prefs}")
        return "\n".join(lines)

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


# ── Settings workspace override helper ───────────────────────────────────────

def _override_workspace(settings: Settings, new_workspace: Path) -> Settings:
    """Return a shallow copy of settings with workspace_dir replaced.

    Works regardless of whether Settings is a Pydantic model or a dataclass.
    """
    import copy
    cloned = copy.copy(settings)
    object.__setattr__(cloned, "workspace_dir", new_workspace)
    return cloned
