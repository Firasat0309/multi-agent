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
            "Your task is to generate a typed API client layer for a frontend\n"
            "application that connects to a REST backend.\n\n"
            "FOR REACT / NEXT.JS:\n"
            "  Files to produce (write each using write_file):\n"
            "  1. src/lib/api.ts          — Base Axios/Fetch client with auth interceptors\n"
            "  2. src/lib/types.ts        — TypeScript interfaces from API contract schemas\n"
            "  3. src/hooks/use<X>.ts     — One custom hook per resource group using SWR or React Query\n\n"

            "FOR VUE 3:\n"
            "  Files to produce (write each using write_file):\n"
            "  1. src/lib/api.ts          — Base Axios client with auth interceptors\n"
            "  2. src/lib/types.ts        — TypeScript interfaces from API contract schemas\n"
            "  3. src/composables/use<X>.ts — One composable per resource group using ref/reactive\n"
            "     Each composable should return { data, error, loading, fetch, mutate } as refs.\n"
            "     Use the Composition API pattern (NOT Pinia — Pinia is for client state, not data fetching).\n"
            "     Example:\n"
            "       export function useUsers() {\n"
            "         const data = ref<User[]>([]);\n"
            "         const loading = ref(false);\n"
            "         const error = ref<string | null>(null);\n"
            "         async function fetchUsers() { ... }\n"
            "         return { data, loading, error, fetchUsers };\n"
            "       }\n\n"

            "Rules:\n"
            "- Use Axios for HTTP client; add request interceptor for JWT Bearer token.\n"
            "- Derive TypeScript interfaces from the API schema definitions.\n"
            "- Handle loading, error, and success states in every hook/composable.\n"
            "- Export all types and hooks from index barrel files.\n"
            "- Do NOT inline API URLs — always import from a central config/env variable.\n"
            "- NEXT_PUBLIC_API_URL for Next.js; VITE_API_URL for Vite/Vue; REACT_APP_API_URL for CRA.\n"
            "- Use RELATIVE imports for api.ts (e.g. '../lib/api'), not @/ aliases."
        )

    def _build_prompt(self, context: AgentContext) -> str:
        contract: APIContract | None = context.task.metadata.get("api_contract")
        plan: ComponentPlan | None = context.task.metadata.get("component_plan")

        contract_text = ""
        if contract:
            endpoint_lines = "\n".join(
                f"  {ep.method} {ep.path} — {ep.description}"
                f"{' [auth]' if ep.auth_required else ''}"
                for ep in contract.endpoints
            )
            schema_names = list(contract.schemas.keys())
            contract_text = (
                f"API base URL: {contract.base_url}\n"
                f"Endpoints:\n{endpoint_lines}\n"
                f"Schemas: {schema_names}\n"
            )

        plan_text = ""
        if plan:
            plan_text = (
                f"Framework: {plan.framework}\n"
                f"State solution: {plan.state_solution}\n"
            )

        return (
            f"{contract_text}\n{plan_text}\n"
            "Generate ALL API integration files and write each to disk using write_file."
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
