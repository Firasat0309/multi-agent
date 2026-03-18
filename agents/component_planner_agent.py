"""Component Planner Agent — splits a UI design spec into a ComponentPlan."""

from __future__ import annotations

import json
import logging
from typing import Any

from agents.base_agent import BaseAgent
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


class ComponentPlannerAgent(BaseAgent):
    """Decomposes a UIDesignSpec into a flat ComponentPlan.

    Each component gets a name, file path, type classification, and metadata
    describing what props it accepts, what state slices it needs, and which
    API endpoints it must call.  The actual dependency ordering is handled
    by the subsequent ComponentDAGAgent.
    """

    role = AgentRole.COMPONENT_PLANNER

    @property
    def system_prompt(self) -> str:
        return (
            "You are a senior frontend architect agent. Your job is to decompose a UI\n"
            "design spec into an exhaustive list of components following Atomic Design\n"
            "principles.\n\n"
            "Respond with a JSON object matching this schema:\n"
            "{\n"
            '  "framework": "react|nextjs|vue|angular",\n'
            '  "state_solution": "zustand|redux|context|pinia",\n'
            '  "api_base_url": "/api/v1",\n'
            '  "routing_solution": "react-router|nextjs|vue-router",\n'
            '  "package_json": {"dependencies": {}, "devDependencies": {}},\n'
            '  "components": [\n'
            "    {\n"
            '      "name": "UserList",\n'
            '      "file_path": "src/components/feature/UserList.tsx",\n'
            '      "component_type": "feature",\n'
            '      "description": "Displays paginated list of users",\n'
            '      "figma_node_id": "1:23",\n'
            '      "props": ["users: User[]", "onSelect: (id: number) => void"],\n'
            '      "state_needs": ["userStore"],\n'
            '      "api_calls": ["/api/v1/users"],\n'
            '      "depends_on": ["UserCard", "Pagination"],\n'
            '      "children": ["UserCard"],\n'
            '      "layer": "components/feature"\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Component type classifications:\n"
            "  pages     → Top-level route components (pages/ dir for Next.js)\n"
            "  layout    → Structural wrappers (Header, Footer, Sidebar, Layout)\n"
            "  feature   → Complex domain-specific components (UserList, Dashboard)\n"
            "  ui        → Pure presentational atoms (Button, Input, Card, Modal)\n"
            "  shared    → Cross-cutting utilities (ErrorBoundary, SEO, ThemeProvider)\n\n"
            "Rules:\n"
            "- Every page from the design spec must map to at least one 'pages' component.\n"
            "- Preserve exact `figma_node_id` strings if the spec lists them, otherwise null.\n"
            "- All components must have explicit file_path with correct extension (.tsx/.vue).\n"
            "- Use TypeScript for React/Next.js/Angular; use <script setup lang='ts'> for Vue.\n"
            "- api_calls must be populated for any component that fetches data.\n"
            "- state_needs should reference the Zustand/Redux store slice names.\n"
            "- Output ONLY the JSON object — no markdown code fences."
        )

    async def execute(self, context: AgentContext) -> TaskResult:
        design_spec: UIDesignSpec | None = context.task.metadata.get("design_spec")
        contract: APIContract | None = context.task.metadata.get("api_contract")
        requirements: ProductRequirements | None = context.task.metadata.get("requirements")
        try:
            plan = await self.plan_components(design_spec, contract, requirements)
            return TaskResult(
                success=True,
                output=f"Component plan created: {len(plan.components)} components, "
                       f"framework={plan.framework}, state={plan.state_solution}",
                metrics={
                    **self.get_metrics(),
                    "components": len(plan.components),
                    "framework": plan.framework,
                },
            )
        except Exception as exc:
            logger.exception("ComponentPlannerAgent.execute failed")
            return TaskResult(success=False, errors=[str(exc)])

    async def plan_components(
        self,
        design_spec: UIDesignSpec | None,
        contract: APIContract | None,
        requirements: ProductRequirements | None,
    ) -> ComponentPlan:
        """Build a ComponentPlan from the design spec and API contract."""
        logger.info("ComponentPlannerAgent: planning components")

        spec_text = ""
        if design_spec:
            spec_text = (
                f"Framework: {design_spec.framework}\n"
                f"Pages: {', '.join(design_spec.pages)}\n"
                f"Description: {design_spec.design_description}\n"
            )

        api_text = ""
        if contract:
            endpoints = [f"{ep.method} {ep.path}" for ep in contract.endpoints]
            api_text = f"API endpoints: {', '.join(endpoints)}\n"

        req_text = ""
        if requirements:
            req_text = (
                f"Product: {requirements.title}\n"
                f"Features: {', '.join(requirements.features)}\n"
                f"State: {requirements.tech_preferences.get('state', 'zustand')}\n"
                f"Styling: {requirements.tech_preferences.get('styling', 'tailwind')}\n"
            )

        raw = await self._call_llm_json(
            f"{req_text}{spec_text}{api_text}\n"
            "Produce the complete ComponentPlan JSON object now."
        )
        self._metrics["llm_calls"] += 1
        return self._parse_plan(raw)

    async def revise_components(
        self,
        current: ComponentPlan,
        feedback: str,
    ) -> ComponentPlan:
        """Revise a component plan based on user feedback."""
        logger.info("ComponentPlannerAgent: revising component plan with user feedback")
        current_json = json.dumps({
            "framework": current.framework,
            "state_solution": current.state_solution,
            "routing_solution": current.routing_solution,
            "api_base_url": current.api_base_url,
            "components": [
                {
                    "name": c.name,
                    "component_type": c.component_type,
                    "file_path": c.file_path,
                    "description": c.description,
                }
                for c in current.components
            ],
        }, indent=2)
        raw = await self._call_llm_json(
            f"Current component plan:\n{current_json}\n\n"
            f"User feedback — apply these changes:\n{feedback}\n\n"
            "Produce the REVISED ComponentPlan JSON object now. "
            "Keep everything the user did not mention, only change what they asked for."
        )
        self._metrics["llm_calls"] += 1
        return self._parse_plan(raw)

    # ─────────────────────────────────────────────────────────────────────────

    def _parse_plan(self, raw: dict[str, Any]) -> ComponentPlan:
        components = [
            UIComponent(
                name=str(c.get("name", "Component")),
                file_path=str(c.get("file_path", "")),
                component_type=str(c.get("component_type", "ui")),
                description=str(c.get("description", "")),
                figma_node_id=str(c.get("figma_node_id", "")) if c.get("figma_node_id") else None,
                props=[str(p) for p in c.get("props", [])],
                state_needs=[str(s) for s in c.get("state_needs", [])],
                api_calls=[str(a) for a in c.get("api_calls", [])],
                depends_on=[str(d) for d in c.get("depends_on", [])],
                children=[str(ch) for ch in c.get("children", [])],
                layer=str(c.get("layer", "")),
            )
            for c in raw.get("components", [])
            if isinstance(c, dict)
        ]
        return ComponentPlan(
            components=components,
            framework=str(raw.get("framework", "nextjs")),
            state_solution=str(raw.get("state_solution", "zustand")),
            api_base_url=str(raw.get("api_base_url", "/api/v1")),
            routing_solution=str(raw.get("routing_solution", "")),
            package_json=raw.get("package_json", {}),
        )
