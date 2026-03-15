"""Design Parser Agent — parses a Figma URL or textual UI description into a UIDesignSpec."""

from __future__ import annotations

import logging
import re
from typing import Any


_FIGMA_FILE_RE = re.compile(r"figma\.com/(?:file|design)/([A-Za-z0-9_-]+)")
_FIGMA_API_BASE = "https://api.figma.com/v1"

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

    When a Figma URL is supplied the agent uses its MCP tools to explore
    the layout tree from the URL. For prose descriptions it relies entirely 
    on LLM interpretation.
    """

    role = AgentRole.DESIGN_PARSER

    @property
    def system_prompt(self) -> str:
        return (
            "You are a senior UI/UX design parser agent.  Your job is to read a design\n"
            "description (or Figma URL context) and produce a structured UIDesignSpec.\n\n"
            "If a Figma URL is provided, YOU MUST USE YOUR TOOLS to query the Figma design "
            "to extract the necessary metadata (pages, core styles, overall structure) "
            "BEFORE generating your final response.\n\n"
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

    def _build_prompt(self, context: AgentContext) -> str:
        requirements: ProductRequirements | None = context.task.metadata.get("requirements")
        figma_url: str = context.task.metadata.get("figma_url", "")
        
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

        figma_context = f"Figma URL provided: {figma_url}\n" if figma_url else "No Figma URL provided.\n"
        if figma_url:
            file_key = self._extract_figma_key(figma_url)
            if file_key:
                figma_context += f"The Figma File Key is: {file_key}. Use your MCP tools to get a summary of this file.\n"

        return (
            f"{req_text}{figma_context}\n"
            "Use your tools to explore the design if necessary, then output the UIDesignSpec JSON object."
        )

    async def parse_design(
        self,
        requirements: ProductRequirements,
        figma_url: str = "",
    ) -> UIDesignSpec:
        """Convenience facade called by FrontendPipeline.

        Builds an AgentContext, runs the agentic loop via execute(), then
        returns the parsed UIDesignSpec from task metadata.

        Raises on failure so the caller can handle the exception and fall
        back to an empty spec (matching the try/except in pipeline_frontend).
        """
        from core.models import (
            RepositoryBlueprint,
            Task,
            TaskType,
            AgentContext as _AgentContext,
        )

        # DesignParser doesn't use blueprint file metadata — provide a minimal stub.
        stub_blueprint = RepositoryBlueprint(
            name="design-parser-stub",
            description="",
            architecture_style="",
        )
        context = _AgentContext(
            task=Task(
                task_id=0,
                task_type=TaskType.PARSE_DESIGN,
                file="",
                description="Parse design spec",
                metadata={"requirements": requirements, "figma_url": figma_url},
            ),
            blueprint=stub_blueprint,
        )
        result = await self.execute(context)
        if not result.success:
            raise RuntimeError(
                f"DesignParserAgent failed: {'; '.join(result.errors)}"
            )
        spec: UIDesignSpec | None = context.task.metadata.get("design_spec")
        if spec is None:
            raise RuntimeError(
                "DesignParserAgent returned success but no design_spec in metadata"
            )
        return spec

    @staticmethod
    def _repair_json(text: str) -> str:
        """Best-effort repair of common LLM JSON mistakes.

        Handles: trailing commas before } or ], single-quoted strings,
        JS-style comments (// and /* */), and unquoted property names.
        """
        # Strip JS-style comments
        text = re.sub(r"//[^\n]*", "", text)
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
        # Replace single-quoted strings with double-quoted
        # (only when the single quote is used as a string delimiter)
        text = re.sub(r"(?<=[:,\[\{])\s*'([^']*)'", r' "\1"', text)
        text = re.sub(r"'([^']*)'(?=\s*[:,\]\}])", r'"\1"', text)
        # Remove trailing commas before } or ]
        text = re.sub(r",\s*([}\]])", r"\1", text)
        # Quote unquoted property names: { key: "val" } → { "key": "val" }
        text = re.sub(r'(?<=[{,])\s*([a-zA-Z_]\w*)\s*:', r' "\1":', text)
        return text

    async def execute(self, context: AgentContext) -> TaskResult:
        figma_url: str = context.task.metadata.get("figma_url", "")
        try:
            # Drop into the standard agentic tool loop instead of calling LLM once
            result = await self.execute_agentic(context)
            if not result.success:
                return result

            # Parse the final text output into the UIDesignSpec
            import json
            try:
                # Basic string cleanup to extract JSON if the LLM wrapped it
                content = result.output.strip()
                if content.startswith("```"):
                    content = content.split("\n", 1)[1]
                    if content.endswith("```"):
                        content = content[:-3].rsplit("```", 1)[0].rsplit("\n", 1)[0]

                # Try strict parse first, then repair on failure
                try:
                    raw = json.loads(content)
                except json.JSONDecodeError:
                    repaired = self._repair_json(content)
                    # Try to extract a JSON object if there's surrounding text
                    start = repaired.find("{")
                    end = repaired.rfind("}") + 1
                    if start != -1 and end > start:
                        repaired = repaired[start:end]
                    raw = json.loads(repaired)
                    logger.info("DesignParserAgent: JSON repair succeeded")

                requirements: ProductRequirements | None = context.task.metadata.get("requirements")
                spec = self._parse_spec(raw, requirements, figma_url)

                # Overwrite metadata so next agent in DAG has it
                context.task.metadata["design_spec"] = spec

                return TaskResult(
                    success=True,
                    output=f"Design parsed: {len(spec.pages)} pages, framework={spec.framework}",
                    metrics={**self.get_metrics(), "pages": len(spec.pages)},
                )
            except json.JSONDecodeError as exc:
                logger.error(f"DesignParserAgent failed to parse JSON from output: {result.output[:100]}...")
                return TaskResult(success=False, errors=[f"Invalid JSON returned: {exc}"])

        except Exception as exc:
            logger.exception("DesignParserAgent.execute failed")
            return TaskResult(success=False, errors=[str(exc)])
            
    # ── Figma REST API helpers ─────────────────────────────────────────────────

    @staticmethod
    def _extract_figma_key(url: str) -> str | None:
        """Return the Figma file key from a figma.com URL, or None."""
        m = _FIGMA_FILE_RE.search(url)
        return m.group(1) if m else None

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
