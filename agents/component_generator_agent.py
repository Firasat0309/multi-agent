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
    ProductRequirements,
    TaskResult,
    UIComponent,
    UIDesignSpec,
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

_LIST_TOOL = ToolDefinition(
    name="list_files",
    description="List files in a directory to discover store/lib file names.",
    input_schema={
        "type": "object",
        "properties": {
            "directory": {"type": "string", "description": "Directory path relative to workspace"},
            "pattern": {"type": "string", "description": "Glob pattern (default: **)"},
        },
        "required": [],
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
        return [_READ_TOOL, _WRITE_TOOL, _LIST_TOOL]

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
            "- Use named exports for reusable components and types.\n"
            "- Import child components by relative path from the same src/ tree.\n"
            "- Use the state store (Zustand/Redux/Pinia) ONLY for cross-cutting state;\n"
            "  use local useState/ref for component-local state.\n"
            "- Wrap async data fetching in a custom hook or SWR/React Query.\n"
            "- Include JSDoc comment block at the top of the component.\n"
            "- Write the full file to disk using the write_file tool.\n"
            "- Do NOT include test code in the component file.\n\n"

            "NEXT.JS APP ROUTER — CRITICAL RULES:\n"
            "- Files in src/app/**/page.tsx MUST use 'export default function'. Next.js\n"
            "  pages REQUIRE a default export. Named exports like 'export function DashboardPage'\n"
            "  will cause a build error. Use: export default function DashboardPage() { ... }\n"
            "- Similarly, src/app/**/layout.tsx MUST use 'export default function'.\n"
            "- Any component that uses React hooks (useState, useEffect, useContext, useRef,\n"
            "  useCallback, useMemo, useReducer), event handlers (onClick, onChange, onSubmit),\n"
            "  useRouter, usePathname, useSearchParams, or any browser-only API MUST have\n"
            '  "use client"; as the VERY FIRST LINE of the file (before all imports).\n'
            "- Page components in src/app/ that only render other components and do NOT use\n"
            "  hooks or event handlers directly should be Server Components (no 'use client').\n"
            "- If in doubt, add 'use client'; — a client component that could be a server\n"
            "  component is acceptable; a server component that uses hooks is a build error.\n\n"

            "IMPORT RULES:\n"
            "- Use RELATIVE imports (e.g. '../ui/Button') for importing from other components\n"
            "  within the same src/ tree. Do NOT use @/ path aliases.\n"
            "- For store imports: FIRST use list_files with directory='src/store' to see\n"
            "  which store files actually exist, then import from the EXACT file name found.\n"
            "  If no store files exist yet, use the naming pattern from state_needs\n"
            "  (e.g. state_needs=['auth'] → '../../store/authStore').\n"
            "- For API client imports, use relative paths (e.g. '../../lib/api').\n"
            "- This ensures all imports resolve correctly without tsconfig path aliases."
        )

    def _build_prompt(self, context: AgentContext) -> str:
        component: UIComponent | None = context.task.metadata.get("component")
        plan: ComponentPlan | None = context.task.metadata.get("component_plan")
        contract: APIContract | None = context.task.metadata.get("api_contract")
        design_spec: UIDesignSpec | None = context.task.metadata.get("design_spec")
        requirements: ProductRequirements | None = context.task.metadata.get("requirements")

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

        # Design tokens for consistent styling
        design_text = ""
        if design_spec:
            if design_spec.global_styles:
                styles = ", ".join(f"{k}: {v}" for k, v in design_spec.global_styles.items())
                design_text += f"Global styles: {styles}\n"
            if design_spec.design_tokens:
                import json as _json
                tokens_str = _json.dumps(design_spec.design_tokens, indent=2)
                if len(tokens_str) > 1000:
                    tokens_str = tokens_str[:1000] + "..."
                design_text += f"Design tokens:\n{tokens_str}\n"

        # Business context from product requirements
        req_text = ""
        if requirements:
            req_text = f"Product: {requirements.title}\n"
            if requirements.features:
                req_text += f"Features: {', '.join(requirements.features[:5])}\n"

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

        # Build dependency reading instructions
        dep_instructions = ""
        if component and component.depends_on:
            dep_instructions = (
                "\nBEFORE generating code:\n"
                f"1. Use read_file to read each dependency: {', '.join(component.depends_on)}\n"
                "2. Check what each dependency ACTUALLY exports (exact names)\n"
                "3. Only import exports that actually exist in the file\n"
                "4. Do NOT assume a component exports sub-components (e.g. Card does NOT\n"
                "   export CardHeader/CardTitle/CardContent unless you verify it does)\n\n"
            )

        return (
            f"{req_text}{comp_text}\n{plan_text}\n{design_text}\n{contract_text}\n"
            f"{dep_instructions}"
            "If a Figma Node ID is provided, use your tools to fetch the structural code skeleton FIRST. "
            "Hydrate the skeleton with the described API and state handlers, then generate "
            "the complete component source code and write it to disk using write_file."
        )

    async def execute(self, context: AgentContext) -> TaskResult:
        component: UIComponent | None = context.task.metadata.get("component")
        if component is None:
            return TaskResult(success=False, errors=["No component in task metadata"])

        # Inject the target file path into file_blueprint so the base agent's
        # agentic loop can detect when the file has been written and exit early
        # instead of looping until max_iterations rewriting the same file.
        if context.file_blueprint is None and component.file_path:
            from core.models import FileBlueprint
            context = AgentContext(
                task=context.task,
                blueprint=context.blueprint,
                file_blueprint=FileBlueprint(
                    path=component.file_path,
                    purpose=component.description or component.name,
                    depends_on=component.depends_on or [],
                ),
                related_files=context.related_files,
                architecture_summary=context.architecture_summary,
                dependency_info=context.dependency_info,
            )

        try:
            result = await self.execute_agentic(context)
            if result.success:
                logger.info("ComponentGeneratorAgent: generated %s", component.file_path)
            return result
        except Exception as exc:
            logger.exception("ComponentGeneratorAgent.execute failed for %s", component.name)
            return TaskResult(success=False, errors=[str(exc)])
