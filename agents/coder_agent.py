"""Coder agent - generates code for a specific file following the blueprint."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from agents.base_agent import BaseAgent
from core.language import get_language_profile, LanguageProfile
from core.models import AgentContext, AgentRole, TaskResult

logger = logging.getLogger(__name__)

# File extensions that are configuration/resource files, not source code.
# These need format-specific generation, not a language-code prompt.
_CONFIG_FILE_FORMATS: dict[str, str] = {
    ".properties": "Java Spring Boot application.properties format (key=value pairs, no code)",
    ".yaml": "YAML format",
    ".yml":  "YAML format",
    ".xml":  "XML format",
    ".json": "JSON format",
    ".toml": "TOML format",
    ".ini":  "INI format",
    ".env":  "dotenv format (KEY=value lines)",
    ".sql":  "SQL",
    ".sh":   "Bash shell script",
    ".dockerfile": "Dockerfile",
}
_DOCKERFILE_NAMES = {"dockerfile", "dockerfile.dev", "dockerfile.prod"}


def _config_format(path: str) -> str | None:
    """Return the expected content format for non-source files, or None for source files."""
    name = Path(path).name.lower()
    if name in _DOCKERFILE_NAMES:
        return "Dockerfile"
    suffix = Path(path).suffix.lower()
    return _CONFIG_FILE_FORMATS.get(suffix)


class CoderAgent(BaseAgent):
    role = AgentRole.CODER

    def _get_source_system_prompt(self, language: str) -> str:
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

    def _get_config_system_prompt(self, fmt: str) -> str:
        return (
            f"You are a configuration file generator. You produce correct {fmt} files.\n\n"
            "Rules:\n"
            f"- Generate ONLY valid {fmt} content — no code, no markdown fences, no explanations\n"
            "- Use realistic values appropriate for the project and tech stack\n"
            "- Include comments where the format supports them\n"
            "- Do NOT wrap the output in any programming language"
        )

    @property
    def system_prompt(self) -> str:
        return self._get_source_system_prompt("python")

    async def execute(self, context: AgentContext) -> TaskResult:
        """Generate code or config content for the assigned file."""
        if not context.file_blueprint:
            return TaskResult(
                success=False,
                errors=[f"No blueprint found for {context.task.file}"],
            )

        fb = context.file_blueprint
        fmt = _config_format(fb.path)

        if fmt:
            return await self._generate_config(context, fmt)
        else:
            return await self._generate_source(context)

    async def _generate_source(self, context: AgentContext) -> TaskResult:
        """Generate source code for a programming language file."""
        fb = context.file_blueprint
        lang = fb.language or context.blueprint.tech_stack.get("language", "python")
        profile = get_language_profile(lang)
        logger.info(f"Generating {profile.display_name} source: {fb.path}")

        prompt = (
            f"{self._format_context(context)}\n\n"
            f"Generate the complete {profile.display_name} file for: {fb.path}\n"
            f"Purpose: {fb.purpose}\n"
            f"This file must export: {', '.join(fb.exports) if fb.exports else 'appropriate classes/functions'}\n"
            f"Layer: {fb.layer}\n\n"
            f"Generate the complete, working {profile.display_name} code. Output only the code, nothing else."
        )

        code = await self._call_llm(prompt, system_override=self._get_source_system_prompt(lang))
        code = self._clean_fences(code, profile.code_fence_name)
        self.repo.write_file(fb.path, code)

        return TaskResult(
            success=True,
            output=f"Generated {fb.path} ({len(code)} bytes, {profile.display_name})",
            files_modified=[fb.path],
            metrics=self.get_metrics(),
        )

    async def _generate_config(self, context: AgentContext, fmt: str) -> TaskResult:
        """Generate a configuration/resource file (properties, yaml, xml, etc.)."""
        fb = context.file_blueprint
        tech = context.blueprint.tech_stack
        logger.info(f"Generating config file ({fmt}): {fb.path}")

        prompt = (
            f"Project: {context.blueprint.name}\n"
            f"Tech stack: {tech}\n"
            f"Architecture: {context.blueprint.architecture_style}\n\n"
            f"Generate the file: {fb.path}\n"
            f"Purpose: {fb.purpose}\n"
            f"Format: {fmt}\n\n"
            f"Output only the {fmt} content. No code, no markdown, no explanations."
        )

        content = await self._call_llm(prompt, system_override=self._get_config_system_prompt(fmt))
        content = self._clean_fences(content, "")  # strip any accidental fences
        self.repo.write_file(fb.path, content)

        return TaskResult(
            success=True,
            output=f"Generated config {fb.path} ({len(content)} bytes, {fmt})",
            files_modified=[fb.path],
            metrics=self.get_metrics(),
        )

    def _clean_fences(self, content: str, lang_fence: str) -> str:
        """Strip any markdown code fences the LLM may have added."""
        content = content.strip()
        for fence in (f"```{lang_fence}", "```"):
            if content.startswith(fence):
                content = content[len(fence):]
                break
        if content.endswith("```"):
            content = content[:-3]
        return content.strip() + "\n"
