"""Design Parser Agent — parses a Figma URL or textual UI description into a UIDesignSpec."""

from __future__ import annotations

import logging
from typing import Any

from agents.base_agent import BaseAgent
from core.models import (
    AgentContext,
    AgentRole,
    ProductRequirements,
    TaskResult,
    UIDesignSpec,
)

logger = logging.getLogger(__name__)


class DesignParserAgent(BaseAgent):
    """Converts a Figma design URL or a prose UI description into a structured
    UIDesignSpec that subsequent frontend agents can consume.

    When a Figma URL is supplied the agent produces an approximate layout tree
    from the URL structure and any description context.  For prose descriptions
    it relies entirely on LLM interpretation.
    """

    role = AgentRole.DESIGN_PARSER

    @property
    def system_prompt(self) -> str:
        return (
            "You are a senior UI/UX design parser agent.  Your job is to read a design\n"
            "description (or Figma URL context) and produce a structured UIDesignSpec.\n\n"
            "Respond with a JSON object matching this schema:\n"
            "{\n"
            '  "framework": "react|nextjs|vue|angular",\n'
            '  "design_description": "1-2 sentence summary of the overall UI",\n'
            '  "figma_url": "<url or empty>",\n'
            '  "pages": ["Home", "Dashboard", "Profile", ...],\n'
            '  "global_styles": {\n'
            '    "primary_color": "#3B82F6",\n'
            '    "font_family": "Inter, sans-serif",\n'
            '    "border_radius": "8px"\n'
            "  },\n"
            '  "design_tokens": {\n'
            '    "colors": {"brand": "#3B82F6", "bg": "#FFFFFF"},\n'
            '    "spacing": {"sm": "8px", "md": "16px", "lg": "24px"},\n'
            '    "typography": {"base": "16px", "lg": "20px"}\n'
            "  }\n"
            "}\n\n"
            "Rules:\n"
            "- Infer the framework from tech_preferences; default to 'nextjs'.\n"
            "- List every distinct page/screen the user mentioned or implied.\n"
            "- Derive reasonable design tokens even when the user hasn't specified them.\n"
            "- Output ONLY the JSON object — no markdown code fences."
        )

    async def execute(self, context: AgentContext) -> TaskResult:
        requirements: ProductRequirements | None = context.task.metadata.get("requirements")
        figma_url: str = context.task.metadata.get("figma_url", "")
        try:
            spec = await self.parse_design(requirements, figma_url)
            return TaskResult(
                success=True,
                output=f"Design parsed: {len(spec.pages)} pages, framework={spec.framework}",
                metrics={**self.get_metrics(), "pages": len(spec.pages)},
            )
        except Exception as exc:
            logger.exception("DesignParserAgent.execute failed")
            return TaskResult(success=False, errors=[str(exc)])

    async def parse_design(
        self,
        requirements: ProductRequirements | None,
        figma_url: str = "",
    ) -> UIDesignSpec:
        """Parse design information into a UIDesignSpec."""
        logger.info("DesignParserAgent: parsing design")

        req_text = ""
        if requirements:
            fw = requirements.tech_preferences.get("frontend", "nextjs").lower()
            req_text = (
                f"Product: {requirements.title}\n"
                f"Description: {requirements.description}\n"
                f"Features: {', '.join(requirements.features)}\n"
                f"Preferred framework: {fw}\n"
                f"Preferred styling: {requirements.tech_preferences.get('styling', 'tailwind')}\n"
            )

        figma_context = f"Figma URL: {figma_url}\n" if figma_url else ""

        raw = await self._call_llm_json(
            f"{req_text}{figma_context}\n"
            "Parse the design and return the UIDesignSpec JSON object now."
        )
        self._metrics["llm_calls"] += 1
        return self._parse_spec(raw, requirements, figma_url)

    # ─────────────────────────────────────────────────────────────────────────

    def _parse_spec(
        self,
        raw: dict[str, Any],
        requirements: ProductRequirements | None,
        figma_url: str,
    ) -> UIDesignSpec:
        # Derive framework from requirements if available and not in raw
        framework = str(raw.get("framework", "nextjs")).lower()
        if requirements:
            pref = requirements.tech_preferences.get("frontend", "").lower()
            if pref and framework == "nextjs":
                if "vue" in pref:
                    framework = "vue"
                elif "angular" in pref:
                    framework = "angular"
                elif "react" in pref and "next" not in pref:
                    framework = "react"

        return UIDesignSpec(
            framework=framework,
            design_description=str(raw.get("design_description", "")),
            figma_url=figma_url or str(raw.get("figma_url", "")),
            pages=[str(p) for p in raw.get("pages", ["Home"])],
            global_styles={k: str(v) for k, v in raw.get("global_styles", {}).items()},
            design_tokens=raw.get("design_tokens", {}),
        )
