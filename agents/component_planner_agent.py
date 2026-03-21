"""Component Planner Agent — splits a UI design spec into a ComponentPlan."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

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
            "- api_calls must ONLY use endpoint paths from the provided API contract.\n"
            "  Do NOT invent or guess endpoint paths. Copy them exactly as given.\n"
            "- Output ONLY the JSON object — no markdown code fences.\n\n"

            "FRAMEWORK-SPECIFIC RULES — these are MANDATORY based on the Framework field:\n"
            "- If framework=vue:\n"
            "    file_path extension MUST be .vue for components\n"
            "    state_solution MUST be 'pinia'\n"
            "    routing_solution MUST be 'vue-router'\n"
            "    state_needs should reference Pinia store names (e.g. 'useAuthStore')\n"
            "- If framework=nextjs:\n"
            "    file_path extension MUST be .tsx\n"
            "    state_solution should be 'zustand' (preferred) or 'redux'\n"
            "    routing_solution MUST be 'nextjs'\n"
            "    pages use src/app/ directory with page.tsx/layout.tsx conventions\n"
            "- If framework=react:\n"
            "    file_path extension MUST be .tsx\n"
            "    state_solution should be 'zustand' or 'redux'\n"
            "    routing_solution MUST be 'react-router'\n"
            "- If framework=angular:\n"
            "    file_path extension MUST be .ts (e.g. user-list.component.ts)\n"
            "    routing_solution MUST be 'angular'\n"
            "- Store/lib/composable files always use .ts extension (never .vue)."
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
        if contract and contract.endpoints:
            endpoint_lines = [f"  {ep.method} {ep.path} — {ep.description}" for ep in contract.endpoints]
            api_text = (
                "=== API Contract (use ONLY these endpoints in api_calls) ===\n"
                f"Base URL: {contract.base_url}\n"
                + "\n".join(endpoint_lines) + "\n"
                "Do NOT invent endpoint paths. Use the exact paths listed above.\n"
            )

        req_text = ""
        if requirements:
            req_text = (
                f"Product: {requirements.title}\n"
                f"Features: {', '.join(requirements.features)}\n"
                f"State: {requirements.tech_preferences.get('state', 'zustand')}\n"
                f"Styling: {requirements.tech_preferences.get('styling', 'tailwind')}\n"
            )
            if requirements.user_stories:
                req_text += "User stories (design components to fulfill these):\n"
                for story in requirements.user_stories:
                    req_text += f"  - {story}\n"

        raw = await self._call_llm_json(
            f"{req_text}{spec_text}{api_text}\n"
            "Produce the complete ComponentPlan JSON object now."
        )
        self._metrics["llm_calls"] += 1
        plan = self._parse_plan(raw)
        if contract:
            self._validate_api_calls(plan, contract)
        return plan

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

    @staticmethod
    def _validate_api_calls(plan: "ComponentPlan", contract: "APIContract") -> None:
        """Warn when a component's api_calls reference endpoints not in the contract.

        Generates a warning (not an error) per offending component so that the
        pipeline continues — the architect may have intentionally planned for
        future endpoints.  The warnings appear in the log for human review.

        Also rewrites each offending call to the closest matching contract
        endpoint (by common path prefix length) so that generated TypeScript
        uses a valid URL rather than a hallucinated one.  The substitution is
        best-effort and logged so it can be reviewed.
        """
        if not contract or not contract.endpoints:
            return

        # Build a set of valid path+method strings and a set of valid paths
        valid_paths: set[str] = {ep.path for ep in contract.endpoints}

        def _closest(call_path: str) -> str | None:
            """Return the contract endpoint path with the best match.

            Scores by common prefix segments first, then falls back to
            matching the last meaningful path segment (e.g. 'login' in
            '/api/v1/auth/login' matches '/login' or '/auth/login').
            """
            call_parts = [p for p in call_path.strip("/").split("/") if p]
            # Strip common prefixes like "api", "v1" for tail matching
            call_tail = [p for p in call_parts
                         if p not in ("api", "v1", "v2") and not (p.startswith("v") and p[1:].isdigit())]

            best: str | None = None
            best_score = 0
            for vp in valid_paths:
                vp_parts = [p for p in vp.strip("/").split("/") if p]
                # Primary: common prefix segments
                common = sum(1 for a, b in zip(call_parts, vp_parts) if a == b)
                # Secondary: tail segment match (e.g. "login" == "login")
                vp_tail = [p for p in vp_parts
                           if p not in ("api", "v1", "v2") and not (p.startswith("v") and p[1:].isdigit())]
                if vp_tail and call_tail and vp_tail[-1] == call_tail[-1]:
                    common = max(common, 1)
                if common > best_score:
                    best_score = common
                    best = vp
            return best if best_score > 0 else None

        for comp in plan.components:
            if not comp.api_calls:
                continue
            corrected: list[str] = []
            for call in comp.api_calls:
                # Strip query parameters for matching
                call_path = call.split("?")[0]
                if call_path in valid_paths:
                    corrected.append(call)
                    continue
                # Not found — try closest match
                nearest = _closest(call_path)
                if nearest:
                    logger.warning(
                        "[ComponentPlanner] Component '%s' references unknown "
                        "endpoint '%s' — replaced with closest contract match '%s'",
                        comp.name, call_path, nearest,
                    )
                    corrected.append(nearest)
                else:
                    logger.warning(
                        "[ComponentPlanner] Component '%s' references unknown "
                        "endpoint '%s' with no close match in the API contract "
                        "(%d endpoints).  Keeping as-is — verify manually.",
                        comp.name, call_path, len(contract.endpoints),
                    )
                    corrected.append(call)
            comp.api_calls[:] = corrected

    def _parse_plan(self, raw: dict[str, Any]) -> ComponentPlan:
        framework = str(raw.get("framework", "nextjs")).lower()

        # Derive correct defaults based on framework
        if "vue" in framework:
            default_state = "pinia"
            default_routing = "vue-router"
        elif "angular" in framework:
            default_state = str(raw.get("state_solution", "ngrx"))
            default_routing = "angular"
        elif "next" in framework:
            default_state = "zustand"
            default_routing = "nextjs"
        else:
            default_state = "zustand"
            default_routing = "react-router"

        components = [
            UIComponent(
                name=str(c.get("name", "Component")),
                file_path=self._fix_extension(str(c.get("file_path", "")), framework),
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
            framework=framework,
            state_solution=str(raw.get("state_solution", default_state)),
            api_base_url=str(raw.get("api_base_url", "/api/v1")),
            routing_solution=str(raw.get("routing_solution", default_routing)),
            package_json=raw.get("package_json", {}),
        )

    @staticmethod
    def _fix_extension(file_path: str, framework: str) -> str:
        """Ensure file extension matches framework conventions.

        Vue components must use .vue; store/lib/hooks stay as .ts.
        React/Next.js/Angular components must use .tsx/.ts (not .vue).
        """
        if not file_path:
            return file_path
        # Store, lib, hooks, composables are always .ts regardless of framework
        if any(seg in file_path for seg in ("/store/", "/stores/", "/lib/", "/hooks/", "/composables/")):
            return file_path
        if "vue" in framework:
            # Component files should be .vue; fix .tsx → .vue
            for wrong_ext in (".tsx", ".jsx"):
                if file_path.endswith(wrong_ext):
                    file_path = file_path[: -len(wrong_ext)] + ".vue"
                    break
        else:
            # React/Next/Angular: fix .vue → .tsx
            if file_path.endswith(".vue"):
                file_path = file_path[:-4] + ".tsx"
        return file_path
