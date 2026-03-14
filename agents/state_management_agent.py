"""State Management Agent — generates the frontend state management layer."""

from __future__ import annotations

import logging

from agents.base_agent import BaseAgent
from core.agent_tools import ToolDefinition
from core.models import (
    AgentContext,
    AgentRole,
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
            "  - Create src/stores/<entity>.ts for each domain store.\n"
            "  - Use defineStore with setup syntax and typed state.\n\n"
            "When state_solution is 'context' (React lightweight):\n"
            "  - Create src/context/<Entity>Context.tsx with provider + hook.\n\n"
            "Universal rules:\n"
            "- Use TypeScript throughout.\n"
            "- Keep async operations in the store (thunks, Zustand actions, etc.).\n"
            "- Write a barrel src/store/index.ts exporting all stores/slices.\n"
            "- Write each file to disk using the write_file tool.\n"
            "- Do NOT duplicate API call logic — import from src/lib/api.ts."
        )

    def _build_prompt(self, context: AgentContext) -> str:
        plan: ComponentPlan | None = context.task.metadata.get("component_plan")

        if plan is None:
            return "Generate state management files for the frontend application."

        # Collect unique state slice names needed by components
        all_slices: set[str] = set()
        for comp in plan.components:
            all_slices.update(comp.state_needs)

        slices_text = ", ".join(sorted(all_slices)) if all_slices else "auth, ui"
        return (
            f"Framework: {plan.framework}\n"
            f"State solution: {plan.state_solution}\n"
            f"API base URL: {plan.api_base_url}\n"
            f"Required state slices: {slices_text}\n\n"
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
