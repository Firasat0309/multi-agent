"""State Management Agent — generates the frontend state management layer."""

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
    description="Write state store files to disk.",
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
    description="Read existing files for context.",
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


class StateManagementAgent(BaseAgent):
    """Generates the client-side state management layer.

    Supports:
    - **Zustand** (React/Next.js default): one store file per state slice
    - **Redux Toolkit** (React/Next.js): slices + store root + typed hooks
    - **Pinia** (Vue 3): one store file per domain entity
    - **React Context** (lightweight React): context + provider + typed hook

    Also generates an ``src/store/index.ts`` barrel that exports everything.
    """

    role = AgentRole.STATE_MANAGER
    max_iterations: int = 20  # multi-file agent: reads context + writes several store files

    @property
    def tools(self) -> list[ToolDefinition]:
        return [_READ_TOOL, _WRITE_TOOL]

    @property
    def system_prompt(self) -> str:
        return (
            "You are a senior frontend state-management engineer.\n"
            "Your task is to generate the complete client-side state layer for a modern\n"
            "web application.\n\n"
            "When state_solution is 'zustand' (React/Next.js):\n"
            "  - Create src/store/<slice>.ts for each domain entity.\n"
            "  - Use immer middleware for nested state updates.\n"
            "  - Export typed selector hooks from each store file.\n\n"
            "When state_solution is 'redux' (React/Next.js):\n"
            "  - Create src/store/slices/<slice>Slice.ts (createSlice + async thunks).\n"
            "  - Create src/store/index.ts (configureStore + RootState + AppDispatch).\n"
            "  - Create src/store/hooks.ts (typed useAppSelector / useAppDispatch).\n\n"
            "When state_solution is 'pinia' (Vue 3):\n"
            "  - Create src/stores/<entity>.ts for each domain store (plural: stores/).\n"
            "  - Use defineStore with setup() syntax and fully typed state:\n"
            "      export const useAuthStore = defineStore('auth', () => {\n"
            "        const user = ref<User | null>(null)\n"
            "        async function login(creds) { ... }\n"
            "        return { user, login }\n"
            "      })\n"
            "  - Consumers must use storeToRefs() to destructure reactive state:\n"
            "      const { user } = storeToRefs(useAuthStore())\n"
            "  - Write barrel: src/stores/index.ts re-exporting all stores.\n\n"
            "When state_solution is 'context' (React lightweight):\n"
            "  - Create src/context/<Entity>Context.tsx with provider + hook.\n\n"
            "Universal rules:\n"
            "- Use TypeScript throughout.\n"
            "- Keep async operations in the store (thunks, Zustand actions, Pinia actions).\n"
            "- File locations:\n"
            "    Vue/Pinia:       src/stores/<entity>.ts  (plural: stores/)\n"
            "    React/Zustand:   src/store/<slice>.ts    (singular: store/)\n"
            "    Redux:           src/store/slices/<slice>Slice.ts\n"
            "- Each file exports a named hook (e.g. useAuthStore, useTaskStore).\n"
            "- Write a barrel index that re-exports ALL store hooks:\n"
            "    Vue/Pinia: src/stores/index.ts\n"
            "    React:     src/store/index.ts\n"
            "- Write each file to disk using the write_file tool.\n"
            "- Do NOT duplicate API call logic — import from src/lib/api.ts.\n"
            "  Read the api.ts file first using read_file to see what functions are exported,\n"
            "  then import those exact function names in the store.\n"
            "- Use RELATIVE imports for api.ts (e.g. '../lib/api'), not @/ aliases.\n\n"
            "CRITICAL — CROSS-FILE CONSISTENCY:\n"
            "- Component files are shown in 'Related Files' section below.\n"
            "- READ them carefully to see what store hooks and methods they already import.\n"
            "- You MUST use the EXACT same method/hook names that components expect.\n"
            "  For example, if a component does `const { fetchUser } = useAuthStore()`,\n"
            "  your store MUST export a `fetchUser` method, NOT `fetchMe` or `getUser`.\n"
            "- Match the exact state shape (property names) that components destructure."
        )

    def _build_prompt(self, context: AgentContext) -> str:
        plan: ComponentPlan | None = context.task.metadata.get("component_plan")
        contract: APIContract | None = context.task.metadata.get("api_contract")

        if plan is None:
            logger.warning("StateManagementAgent: no component_plan in metadata — "
                           "generating generic stores (auth, ui)")
            return (
                "No component plan was provided.  Generate a minimal state management "
                "layer with an auth store (login/logout/token) and a UI store (sidebar "
                "toggle, theme).  Use zustand with TypeScript.  Write each file to disk "
                "using write_file."
            )

        # Collect unique state slice names needed by components
        all_slices: set[str] = set()
        for comp in plan.components:
            all_slices.update(comp.state_needs)

        slices_text = ", ".join(sorted(all_slices)) if all_slices else "auth, ui"

        # Include API endpoints so store actions know what calls to make
        api_text = ""
        if contract and contract.endpoints:
            endpoint_lines = "\n".join(
                f"  {ep.method} {ep.path} — {ep.description}"
                for ep in contract.endpoints
            )
            api_text = f"Available API endpoints:\n{endpoint_lines}\n\n"

        # Include API contract schemas so store TypeScript types match the API exactly
        schema_text = ""
        if contract and contract.schemas:
            import json as _json
            schema_text = (
                "API CONTRACT SCHEMAS — define your TypeScript types to match these EXACTLY.\n"
                "Do NOT invent field names; use only the fields listed here:\n"
            )
            for name, definition in contract.schemas.items():
                try:
                    schema_text += f"  {name}: {_json.dumps(definition)}\n"
                except Exception:
                    schema_text += f"  {name}: {definition}\n"
            schema_text += "\n"

        # Check if api.ts is pre-loaded in related_files
        api_preloaded = any(
            "api.ts" in p for p in (context.related_files or {})
        )
        if api_preloaded:
            api_read_instruction = (
                "IMPORTANT: src/lib/api.ts is already provided in the Related Files section below.\n"
                "Do NOT call read_file for it — use the content shown to see exact function exports.\n"
                "Import those exact exports in your stores.\n\n"
            )
        else:
            api_read_instruction = (
                "IMPORTANT: Use read_file to read src/lib/api.ts to see what functions/client\n"
                "it exports. If the file does not exist yet, define your own axios instance\n"
                "in the store using: import axios from 'axios' and set baseURL from env.\n"
                "Import those exact exports in your stores.\n\n"
            )

        # Vue/Pinia-specific store path guidance
        store_path_hint = ""
        fw = (plan.framework or "").lower()
        if "vue" in fw or plan.state_solution.lower() == "pinia":
            store_path_hint = (
                "\nVUE / PINIA STORE PATHS:\n"
                "- Write stores to src/stores/<entity>.ts (PLURAL: stores/, not store/)\n"
                "- Use defineStore with setup() syntax for full TypeScript support.\n"
                "- Write barrel: src/stores/index.ts that re-exports ALL store hooks.\n"
                "- Use import.meta.env.VITE_API_BASE_URL for API base URL (NOT process.env).\n\n"
            )

        return (
            f"Framework: {plan.framework}\n"
            f"State solution: {plan.state_solution}\n"
            f"API base URL: {plan.api_base_url}\n"
            f"Required state slices: {slices_text}\n\n"
            f"{api_text}"
            f"{schema_text}"
            f"{store_path_hint}"
            f"{api_read_instruction}"
            "Generate ALL state management files and write each to disk using write_file."
        )

    async def execute(self, context: AgentContext) -> TaskResult:
        plan: ComponentPlan | None = context.task.metadata.get("component_plan")
        solution = plan.state_solution if plan else "zustand"
        try:
            result = await self.execute_agentic(context)
            if result.success:
                logger.info(
                    "StateManagementAgent: generated %s state layer (%d files)",
                    solution,
                    len(result.files_modified),
                )
            return result
        except Exception as exc:
            logger.exception("StateManagementAgent.execute failed")
            return TaskResult(success=False, errors=[str(exc)])
