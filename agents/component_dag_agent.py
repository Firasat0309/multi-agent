"""Component DAG Agent — orders components by their dependency graph."""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Any

from agents.base_agent import BaseAgent
from core.models import (
    AgentContext,
    AgentRole,
    ComponentPlan,
    TaskResult,
    UIComponent,
)

logger = logging.getLogger(__name__)


class ComponentDAGAgent(BaseAgent):
    """Topologically sorts the ComponentPlan into an ordered build sequence.

    It resolves ``depends_on`` relationships, detects cycles, and annotates
    each component with a ``generation`` tier number — matching the tier-based
    execution model used by the existing backend pipeline.

    This agent does NOT call the LLM; it performs deterministic graph analysis.
    """

    role = AgentRole.COMPONENT_DAG_BUILDER

    async def execute(self, context: AgentContext) -> TaskResult:
        plan: ComponentPlan | None = context.task.metadata.get("component_plan")
        if plan is None:
            return TaskResult(success=False, errors=["No component_plan in task metadata"])
        try:
            ordered, tiers = self.build_dag(plan)
            return TaskResult(
                success=True,
                output=f"Component DAG built: {len(ordered)} components across {max(tiers.values()) + 1} tiers",
                metrics={
                    **self.get_metrics(),
                    "components": len(ordered),
                    "tiers": max(tiers.values()) + 1 if tiers else 1,
                    "execution_order": [c.name for c in ordered],
                    "tier_map": tiers,
                },
            )
        except Exception as exc:
            logger.exception("ComponentDAGAgent.execute failed")
            return TaskResult(success=False, errors=[str(exc)])

    def build_dag(
        self, plan: ComponentPlan
    ) -> tuple[list[UIComponent], dict[str, int]]:
        """Return (topological_order, tier_map).

        tier_map maps component name → integer generation (0 = no dependencies).
        Raises ValueError on cycle detection.
        """
        components = {c.name: c for c in plan.components}
        # Build adjacency list: name → set of names it depends on
        deps: dict[str, set[str]] = {
            c.name: set(c.depends_on) & components.keys()
            for c in plan.components
        }
        # In-degree for Kahn's algorithm
        in_degree: dict[str, int] = {name: 0 for name in components}
        for name, dep_set in deps.items():
            for dep in dep_set:
                in_degree[name] += 1

        queue: deque[str] = deque(
            name for name, degree in in_degree.items() if degree == 0
        )
        tier_map: dict[str, int] = {name: 0 for name in components}
        order: list[str] = []

        while queue:
            name = queue.popleft()
            order.append(name)
            # All components that depend on this one
            for dependent_name, dep_set in deps.items():
                if name in dep_set:
                    tier_map[dependent_name] = max(
                        tier_map[dependent_name], tier_map[name] + 1
                    )
                    in_degree[dependent_name] -= 1
                    if in_degree[dependent_name] == 0:
                        queue.append(dependent_name)

        if len(order) != len(components):
            cyclic = set(components) - set(order)
            raise ValueError(f"Cycle detected among components: {cyclic}")

        ordered_components = [components[name] for name in order]
        logger.info(
            "Component DAG: %d components, %d tiers",
            len(ordered_components),
            max(tier_map.values()) + 1 if tier_map else 1,
        )
        return ordered_components, tier_map
