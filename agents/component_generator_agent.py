"""Component Generator Agent — generates source code for a single UI component."""

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
    UIComponent,
)

logger = logging.getLogger(__name__)

_WRITE_TOOL = ToolDefinition(
    name="write_file",
    description="Write the generated component source code to disk.",
    input_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path relative to workspace"},
            "content": {"type": "string", "description": "Complete file content"},
        },
        "required": ["path", "content"],
    },
)

_READ_TOOL = ToolDefinition(
    name="read_file",
    description="Read an existing file for context.",
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


class ComponentGeneratorAgent(BaseAgent):
    """Generates production-quality source code for a single UI component.

    Uses the agentic tool-use loop (read → write) so the LLM can inspect
    adjacent already-generated files before writing its output.
    """

    role = AgentRole.COMPONENT_GENERATOR

    @property
    def tools(self) -> list[ToolDefinition]:
        return [_READ_TOOL, _WRITE_TOOL]

    @property
    def system_prompt(self) -> str:
        return (
            "You are a senior frontend engineer agent specialising in React/Next.js,\n"
            "Vue 3 (Composition API), and TypeScript.\n\n"
            "Your task is to generate a single, production-quality UI component file.\n\n"
            "Rules:\n"
            "- Use TypeScript throughout (.tsx for React, <script setup lang='ts'> for Vue).\n"
            "- If the component has a `figma_node_id`, use your MCP tools to fetch its code skeleton FIRST.\n"
            "- Keep components focused on a single responsibility.\n"
            "- Use named exports for components and types.\n"
            "- Import child components by relative path from the same src/ tree.\n"
            "- Use the state store (Zustand/Redux/Pinia) ONLY for cross-cutting state;\n"
            "  use local useState/ref for component-local state.\n"
            "- Wrap async data fetching in a custom hook or SWR/React Query.\n"
            "- Include JSDoc comment block at the top of the component.\n"
            "- Write the full file to disk using the write_file tool.\n"
            "- Do NOT include test code in the component file."
        )

    def _build_prompt(self, context: AgentContext) -> str:
        component: UIComponent | None = context.task.metadata.get("component")
        plan: ComponentPlan | None = context.task.metadata.get("component_plan")
        contract: APIContract | None = context.task.metadata.get("api_contract")

        comp_text = ""
        if component:
            comp_text = (
                f"Component: {component.name}\n"
                f"File path: {component.file_path}\n"
                f"Type: {component.component_type}\n"
                f"Description: {component.description}\n"
                f"Figma Node ID: {getattr(component, 'figma_node_id', 'None')}\n"
                f"Props: {component.props}\n"
                f"State needs: {component.state_needs}\n"
                f"API calls: {component.api_calls}\n"
                f"Depends on: {component.depends_on}\n"
                f"Children: {component.children}\n"
            )

        plan_text = ""
        if plan:
            plan_text = (
                f"Framework: {plan.framework}\n"
                f"State solution: {plan.state_solution}\n"
                f"API base URL: {plan.api_base_url}\n"
                f"Routing: {plan.routing_solution}\n"
            )

        contract_text = ""
        if contract and component and component.api_calls:
            relevant = [
                ep for ep in contract.endpoints
                if any(call in ep.path for call in component.api_calls)
            ]
            if relevant:
                endpoint_lines = "\n".join(
                    f"  {ep.method} {ep.path}: {ep.description}"
                    for ep in relevant
                )
                contract_text = f"Relevant API endpoints:\n{endpoint_lines}\n"

        return (
            f"{comp_text}\n{plan_text}\n{contract_text}\n"
            "If a Figma Node ID is provided, use your tools to fetch the structural code skeleton FIRST. "
            "Hydrate the skeleton with the described API and state handlers, then generate "
            "the complete component source code and write it to disk using write_file."
        )

    async def execute(self, context: AgentContext) -> TaskResult:
        component: UIComponent | None = context.task.metadata.get("component")
        if component is None:
            return TaskResult(success=False, errors=["No component in task metadata"])
        try:
            result = await self.execute_agentic(context)
            if result.success:
                logger.info("ComponentGeneratorAgent: generated %s", component.file_path)
            return result
        except Exception as exc:
            logger.exception("ComponentGeneratorAgent.execute failed for %s", component.name)
            return TaskResult(success=False, errors=[str(exc)])
