"""Coder agent - generates code for a specific file following the blueprint."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from agents.base_agent import BaseAgent
from core.language import get_language_profile, LanguageProfile
from core.models import AgentContext, AgentRole, TaskResult, TaskType

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

# Build/project config files identified by exact filename (case-insensitive)
_BUILD_CONFIG_NAMES: dict[str, str] = {
    "pom.xml": "Maven POM XML (project build configuration with dependencies, plugins, and properties)",
    "build.gradle": "Gradle build script (Groovy DSL)",
    "build.gradle.kts": "Gradle build script (Kotlin DSL)",
    "settings.gradle": "Gradle settings",
    "settings.gradle.kts": "Gradle settings (Kotlin DSL)",
    "go.mod": "Go module file",
    "go.sum": "Go module checksums",
    "cargo.toml": "Rust Cargo manifest",
    "package.json": "Node.js/TypeScript package.json",
    "tsconfig.json": "TypeScript compiler configuration",
    "requirements.txt": "Python pip requirements (one package per line)",
    "pyproject.toml": "Python project configuration (TOML format)",
    ".csproj": "C# project file (MSBuild XML)",
}


def _config_format(path: str) -> str | None:
    """Return the expected content format for non-source files, or None for source files."""
    name = Path(path).name.lower()
    if name in _DOCKERFILE_NAMES:
        return "Dockerfile"
    # Check build config files by exact name
    if name in _BUILD_CONFIG_NAMES:
        return _BUILD_CONFIG_NAMES[name]
    # Check by file extension for .csproj etc.
    suffix = Path(path).suffix.lower()
    if suffix == ".csproj":
        return _BUILD_CONFIG_NAMES[".csproj"]
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
        """Generate or fix code for the assigned file."""
        if context.task.task_type == TaskType.FIX_CODE:
            return await self._fix_code(context)

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
        await self.repo.async_write_file(fb.path, code)

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

        # For pom.xml, enforce correct Java 17 configuration with up-to-date Maven fields
        pom_hint = ""
        if Path(fb.path).name.lower() == "pom.xml":
            pom_hint = (
                "\nIMPORTANT Java 17 requirements for pom.xml:\n"
                "- Use Spring Boot parent 3.x (e.g. 3.2.x) which targets Java 17 by default\n"
                "- Set <java.version>17</java.version> in <properties>\n"
                "- Set <maven.compiler.source>17</maven.compiler.source> and "
                "<maven.compiler.target>17</maven.compiler.target> in <properties>\n"
                "- In the maven-compiler-plugin configuration use <release>17</release> "
                "(NOT <source>/<target> inside the plugin — use the properties instead)\n"
                "- Use <maven.compiler.release>17</maven.compiler.release> as the canonical property\n"
                "- Include all required Spring Boot starter dependencies (web, data-jpa, validation, test)\n"
                "- The pom.xml must be complete and valid — do not truncate it"
            )

        prompt = (
            f"Project: {context.blueprint.name}\n"
            f"Tech stack: {tech}\n"
            f"Architecture: {context.blueprint.architecture_style}\n\n"
            f"Generate the file: {fb.path}\n"
            f"Purpose: {fb.purpose}\n"
            f"Format: {fmt}\n"
            f"{pom_hint}\n"
            f"Output only the {fmt} content. No code, no markdown, no explanations."
        )

        content = await self._call_llm(prompt, system_override=self._get_config_system_prompt(fmt))
        content = self._clean_fences(content, "")  # strip any accidental fences
        await self.repo.async_write_file(fb.path, content)

        return TaskResult(
            success=True,
            output=f"Generated config {fb.path} ({len(content)} bytes, {fmt})",
            files_modified=[fb.path],
            metrics=self.get_metrics(),
        )

    async def _fix_code(self, context: AgentContext) -> TaskResult:
        """Fix a file based on review findings."""
        file_path = context.task.file
        review_errors: list[str] = context.task.metadata.get("review_errors", [])
        review_output: str = context.task.metadata.get("review_output", "")

        # Read current file content from the repo (non-blocking)
        current_content = await self.repo.async_read_file(file_path) or ""

        # If review passed (no errors) — nothing to fix, skip LLM call
        if not review_errors and "PASSED" in review_output:
            logger.info(f"Skipping fix for {file_path} — review passed")
            return TaskResult(
                success=True,
                output=f"No fixes needed for {file_path} (review passed)",
                files_modified=[],
                metrics=self.get_metrics(),
            )

        fb = context.file_blueprint
        lang = (fb.language if fb else None) or context.blueprint.tech_stack.get("language", "python")
        profile = get_language_profile(lang)
        logger.info(f"Fixing {file_path} based on {len(review_errors)} review issue(s)")

        issues_text = "\n".join(f"- {e}" for e in review_errors) if review_errors else review_output

        prompt = (
            f"{self._format_context(context)}\n\n"
            f"The following file has review issues that must be fixed:\n\n"
            f"File: {file_path}\n\n"
            f"Current content:\n```{profile.code_fence_name}\n{current_content}\n```\n\n"
            f"Review findings to fix:\n{issues_text}\n\n"
            f"Output the complete corrected file. Output only the code, nothing else."
        )

        fixed_code = await self._call_llm(prompt, system_override=self._get_source_system_prompt(lang))
        fixed_code = self._clean_fences(fixed_code, profile.code_fence_name)
        await self.repo.async_write_file(file_path, fixed_code)

        return TaskResult(
            success=True,
            output=f"Fixed {file_path} ({len(review_errors)} issue(s) addressed)",
            files_modified=[file_path],
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
