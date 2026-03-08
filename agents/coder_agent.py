"""Coder agent - generates code for a specific file following the blueprint."""

from __future__ import annotations

import logging
from typing import Any

from agents.base_agent import BaseAgent
from core.language import get_language_profile, LanguageProfile
from core.models import AgentContext, AgentRole, TaskResult

logger = logging.getLogger(__name__)


class CoderAgent(BaseAgent):
    role = AgentRole.CODER

    def _get_system_prompt(self, language: str) -> str:
        profile = get_language_profile(language)
        return (
            f"You are an expert {profile.display_name} developer agent. "
            f"You generate production-quality {profile.display_name} code.\n\n"
            "Rules:\n"
            f"- Generate ONLY the {profile.display_name} file content, no markdown fences or explanations\n"
            "- Follow the file blueprint exactly (purpose, exports, dependencies)\n"
            "- Import from the correct modules as specified in the dependency graph\n"
            f"- Follow idiomatic {profile.display_name} conventions and best practices\n"
            "- Use proper type annotations/generics\n"
            "- Include proper error handling\n"
            "- Use dependency injection patterns\n"
            "- Do NOT add placeholder or TODO comments - write complete implementations\n"
            "- Do NOT import modules that aren't in the dependency graph"
        )

    @property
    def system_prompt(self) -> str:
        return self._get_system_prompt("python")

    async def execute(self, context: AgentContext) -> TaskResult:
        """Generate code for the assigned file."""
        if not context.file_blueprint:
            return TaskResult(
                success=False,
                errors=[f"No blueprint found for {context.task.file}"],
            )

        fb = context.file_blueprint
        lang = fb.language or "python"
        profile = get_language_profile(lang)
        logger.info(f"Generating {profile.display_name} code for {fb.path}")

        formatted_context = self._format_context(context)

        prompt = (
            f"{formatted_context}\n\n"
            f"Generate the complete {profile.display_name} file for: {fb.path}\n"
            f"Purpose: {fb.purpose}\n"
            f"This file must export: {', '.join(fb.exports) if fb.exports else 'appropriate classes/functions'}\n"
            f"Layer: {fb.layer}\n\n"
            f"Generate the complete, working {profile.display_name} code. Output only the code, nothing else."
        )

        code = await self._call_llm(prompt, system_override=self._get_system_prompt(lang))
        code = self._clean_code(code, profile)

        self.repo.write_file(fb.path, code)

        return TaskResult(
            success=True,
            output=f"Generated {fb.path} ({len(code)} bytes, {profile.display_name})",
            files_modified=[fb.path],
            metrics=self.get_metrics(),
        )

    def _clean_code(self, code: str, profile: LanguageProfile) -> str:
        """Strip markdown fences and leading/trailing whitespace."""
        code = code.strip()
        fence = f"```{profile.code_fence_name}"
        if code.startswith(fence):
            code = code[len(fence):]
        elif code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
        return code.strip() + "\n"
