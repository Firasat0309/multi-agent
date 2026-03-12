"""Product Planner Agent — converts a user prompt into structured product requirements."""

from __future__ import annotations

import json
import logging
from typing import Any

from agents.base_agent import BaseAgent
from core.models import (
    AgentContext,
    AgentRole,
    ProductRequirements,
    TaskResult,
)

logger = logging.getLogger(__name__)


class ProductPlannerAgent(BaseAgent):
    """Converts a free-form user prompt into structured ProductRequirements.

    This is the first stage of the fullstack pipeline: it decides whether the
    project needs a frontend, a backend, or both, extracts technology
    preferences, and enumerates high-level user stories and features.
    """

    role = AgentRole.PRODUCT_PLANNER

    @property
    def system_prompt(self) -> str:
        return (
            "You are a senior product planner agent in a fullstack code-generation system.\n"
            "Your job is to translate a user's natural-language prompt into a structured\n"
            "product requirements document.\n\n"
            "Respond with a JSON object matching this schema:\n"
            "{\n"
            '  "title": "Short project title",\n'
            '  "description": "2-3 sentence description of what gets built",\n'
            '  "user_stories": ["As a user, I want ...", ...],\n'
            '  "features": ["Feature 1", "Feature 2", ...],\n'
            '  "tech_preferences": {\n'
            '    "frontend": "React/Next.js|Vue|Angular|none",\n'
            '    "backend": "FastAPI|Express|Spring Boot|Django|none",\n'
            '    "db": "PostgreSQL|MySQL|SQLite|MongoDB|h2|none",\n'
            '    "state": "zustand|redux|context|pinia|none",\n'
            '    "styling": "tailwind|css-modules|styled-components|mui|none"\n'
            "  },\n"
            '  "has_frontend": true,\n'
            '  "has_backend": true\n'
            "}\n\n"
            "Rules:\n"
            "- Set has_frontend=false only when the user explicitly asks for a pure API/backend.\n"
            "- Set has_backend=false only when the user explicitly asks for a static site or\n"
            "  frontend-only application.\n"
            "- Default frontend framework is React/Next.js; default backend is FastAPI.\n"
            "- Keep user_stories concise (≤ 8) and focused on end-user value.\n"
            "- Keep features concise (≤ 10) and implementation-agnostic.\n"
            "- Output ONLY the JSON object — no markdown code fences."
        )

    async def execute(self, context: AgentContext) -> TaskResult:
        prompt = context.task.metadata.get("user_prompt", context.task.description)
        try:
            requirements = await self.plan_product(prompt)
            return TaskResult(
                success=True,
                output=f"Product planned: {requirements.title} "
                       f"(FE={requirements.has_frontend}, BE={requirements.has_backend})",
                metrics={
                    **self.get_metrics(),
                    "has_frontend": requirements.has_frontend,
                    "has_backend": requirements.has_backend,
                    "features": len(requirements.features),
                },
            )
        except Exception as exc:
            logger.exception("ProductPlannerAgent.execute failed")
            return TaskResult(success=False, errors=[str(exc)])

    async def plan_product(self, user_prompt: str) -> ProductRequirements:
        """Parse the user prompt into structured ProductRequirements."""
        logger.info("ProductPlannerAgent: planning product from prompt")
        raw = await self._call_llm_json(
            f"User request:\n{user_prompt}\n\n"
            "Produce the product requirements JSON document now."
        )
        self._metrics["llm_calls"] += 1
        return self._parse_requirements(raw)

    # ─────────────────────────────────────────────────────────────────────────

    def _parse_requirements(self, raw: dict[str, Any]) -> ProductRequirements:
        tech = raw.get("tech_preferences", {})
        if not isinstance(tech, dict):
            tech = {}
        return ProductRequirements(
            title=str(raw.get("title", "Untitled Project")),
            description=str(raw.get("description", "")),
            user_stories=[str(s) for s in raw.get("user_stories", [])],
            features=[str(f) for f in raw.get("features", [])],
            tech_preferences={k: str(v) for k, v in tech.items()},
            has_frontend=bool(raw.get("has_frontend", True)),
            has_backend=bool(raw.get("has_backend", True)),
        )
