"""API Integration Agent — wires frontend components to backend API endpoints."""

from __future__ import annotations

import logging

from agents.base_agent import BaseAgent
from core.agent_tools import ToolDefinition
from core.models import (
    AgentContext,
    AgentRole,
    APIContract,
    ComponentPlan,
    TaskResult,
)

logger = logging.getLogger(__name__)

_WRITE_TOOL = ToolDefinition(
    name="write_file",
    description="Write API client / service files to disk.",
    input_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    },
)

_READ_TOOL = ToolDefinition(
    name="read_file",
    description="Read an existing component file for context.",
    input_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "start_line": {"type": "integer"},
            "end_line": {"type": "integer"},
        },
        "required": ["path"],
    },
)


class APIIntegrationAgent(BaseAgent):
    """Generates a typed API client layer for the frontend.

    Produces:
    - ``src/lib/api.ts`` — Axios/Fetch client with interceptors and error handling
    - ``src/hooks/use<Resource>.ts`` — SWR/React Query hooks for each resource group
    - Type definitions derived from the API contract schemas

    Uses the agentic loop so it can read generated component files and
    produce integration code that matches the actual usage patterns.
    """

    role = AgentRole.API_INTEGRATOR

    @property
    def tools(self) -> list[ToolDefinition]:
        return [_READ_TOOL, _WRITE_TOOL]

    @property
    def system_prompt(self) -> str:
        return (
            "You are a senior frontend API integration engineer.\n"
            "Your task is to generate a typed API client layer for a React/Next.js/Vue\n"
            "frontend that connects to a REST backend.\n\n"
            "Files to produce (write each using write_file):\n"
            "  1. src/lib/api.ts              — Base Axios/Fetch client with auth interceptors\n"
            "  2. src/lib/types.ts            — TypeScript interfaces from API contract schemas\n"
            "  For React/Next.js:\n"
            "  3. src/hooks/use<Resource>.ts  — One SWR/React Query hook per resource group\n"
            "  For Vue 3:\n"
            "  3. src/composables/use<Resource>.ts — One Vue composable per resource group\n"
            "     (use ref/reactive + onMounted for data fetching, NOT SWR)\n\n"
            "Rules:\n"
            "- Use Axios for HTTP client; add request interceptor for JWT Bearer token.\n"
            "- Derive TypeScript interfaces from the API schema definitions — exact field names.\n"
            "- React/Next.js: use SWR or React Query for data fetching hooks.\n"
            "- Vue 3: use Vue composables with ref/reactive/onMounted — do NOT use SWR.\n"
            "- Handle loading, error, and success states in every hook/composable.\n"
            "- Export all types from src/lib/types.ts; hooks/composables from their own files.\n"
            "- Do NOT inline API URLs — always import from a central config/env variable.\n"
            "- NEXT_PUBLIC_API_URL for Next.js; VITE_API_URL for Vite/Vue projects.\n\n"
            "CROSS-FILE CONSISTENCY:\n"
            "- Component files that make API calls are shown in 'Related Files' below.\n"
            "- READ them to see what API functions/hooks they already import.\n"
            "- Your generated api.ts and hooks MUST export the EXACT function names\n"
            "  that components already import (e.g. if a component imports { login } from\n"
            "  '../lib/api', you MUST export a function named 'login').\n"
            "- Match the exact parameter and return types components expect."
        )

    def _build_prompt(self, context: AgentContext) -> str:
        contract: APIContract | None = context.task.metadata.get("api_contract")
        plan: ComponentPlan | None = context.task.metadata.get("component_plan")
        backend_models: dict[str, str] | None = context.task.metadata.get("backend_models")

        contract_text = ""
        if contract:
            endpoint_lines = "\n".join(
                f"  {ep.method} {ep.path} — {ep.description}"
                f"{' [auth]' if ep.auth_required else ''}"
                for ep in contract.endpoints
            )
            contract_text = (
                f"API base URL: {contract.base_url}\n"
                f"Endpoints:\n{endpoint_lines}\n"
            )

            # Include full schema definitions so the LLM generates accurate types
            if contract.schemas:
                import json as _json
                schema_text = _json.dumps(contract.schemas, indent=2)
                if len(schema_text) > 4000:
                    schema_text = schema_text[:4000] + "\n... (truncated)"
                contract_text += f"\nAPI Schema definitions (use these for TypeScript interfaces):\n{schema_text}\n"
            else:
                contract_text += f"Schemas: (none defined — infer from endpoint names)\n"

        plan_text = ""
        if plan:
            plan_text = (
                f"Framework: {plan.framework}\n"
                f"State solution: {plan.state_solution}\n"
            )

        # Backend model source code for accurate type matching
        be_model_text = ""
        if backend_models:
            be_model_text = "\nBackend entity models (match your TypeScript interfaces to these):\n"
            for path, content in list(backend_models.items())[:10]:
                be_model_text += f"\n--- {path} ---\n{content}\n"

        return (
            f"{contract_text}\n{plan_text}\n{be_model_text}\n"
            "Generate ALL API integration files and write each to disk using write_file.\n"
            "IMPORTANT: Your TypeScript interfaces in src/lib/types.ts MUST match the\n"
            "schema definitions and backend models above. Use the exact field names and types."
        )

    async def execute(self, context: AgentContext) -> TaskResult:
        try:
            result = await self.execute_agentic(context)
            if result.success:
                logger.info(
                    "APIIntegrationAgent: generated %d integration files",
                    len(result.files_modified),
                )
            return result
        except Exception as exc:
            logger.exception("APIIntegrationAgent.execute failed")
            return TaskResult(success=False, errors=[str(exc)])
