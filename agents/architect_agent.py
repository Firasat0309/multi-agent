"""Architect agent - creates the repository blueprint from a user prompt."""

from __future__ import annotations

import json
import logging
from typing import Any

from agents.base_agent import BaseAgent
from core.language import detect_language_from_blueprint
from core.llm_client import LLMClient
from core.models import (
    AgentContext,
    AgentRole,
    FileBlueprint,
    RepositoryBlueprint,
    Task,
    TaskResult,
)
from core.repository_manager import RepositoryManager

logger = logging.getLogger(__name__)


class ArchitectAgent(BaseAgent):
    role = AgentRole.ARCHITECT

    @property
    def system_prompt(self) -> str:
        return (
            "You are a senior software architect agent. Your job is to design a complete "
            "backend system architecture from a user's requirements.\n\n"
            "IMPORTANT: Detect the programming language from the user's prompt. If they say "
            "'Java', 'Spring Boot', 'Go', 'Gin', 'TypeScript', 'Express', 'Rust', 'C#', "
            "'.NET' etc., use THAT language. If unspecified, use Python.\n\n"
            "You must produce a JSON response with this exact structure:\n"
            "{\n"
            '  "name": "project-name",\n'
            '  "description": "what this project does",\n'
            '  "architecture_style": "REST|GraphQL|gRPC",\n'
            '  "tech_stack": {"language": "<detected-language>", "framework": "<framework>", "db": "postgresql", ...},\n'
            '  "folder_structure": ["controllers", "services", "repositories", "models", "config"],\n'
            '  "file_blueprints": [\n'
            "    {\n"
            '      "path": "models/User.java",\n'
            '      "purpose": "User database model",\n'
            '      "depends_on": [],\n'
            '      "exports": ["User", "UserCreate", "UserUpdate"],\n'
            '      "language": "<detected-language>",\n'
            '      "layer": "model"\n'
            "    }\n"
            "  ],\n"
            '  "architecture_doc": "# Architecture\\n..."\n'
            "}\n\n"
            "Rules:\n"
            "- Use the correct file extensions for the chosen language (.py, .java, .go, .ts, .rs, .cs)\n"
            "- Use the idiomatic framework and folder conventions for that language\n"
            "- Design clean layered architecture (models -> repositories -> services -> controllers)\n"
            "- Include config files (database config, app config, main entrypoint)\n"
            "- ALWAYS include the build/project configuration file:\n"
            "  * Java/Maven: pom.xml (with Spring Boot parent, dependencies, plugins)\n"
            "  * Java/Gradle: build.gradle\n"
            "  * Go: go.mod\n"
            "  * TypeScript/Node: package.json (with scripts, dependencies)\n"
            "  * Rust: Cargo.toml\n"
            "  * C#/.NET: *.csproj\n"
            "  * Python: requirements.txt or pyproject.toml\n"
            "- Every file must have a clear purpose and explicit dependencies\n"
            "- Use dependency injection patterns idiomatic to the language\n"
            "- Follow SOLID principles\n"
            "- Include proper error handling and validation layers\n"
            "- The architecture_doc should be a comprehensive markdown document"
        )

    async def execute(self, context: AgentContext) -> TaskResult:
        """Not used directly - use design_architecture instead."""
        return TaskResult(success=False, errors=["Use design_architecture() method"])

    async def design_architecture(self, user_prompt: str) -> RepositoryBlueprint:
        """Design a complete repository blueprint from user requirements."""
        logger.info("Designing architecture from user prompt")

        result = await self._call_llm_json(user_prompt)
        return self._parse_blueprint(result)

    def _parse_blueprint(self, data: dict[str, Any]) -> RepositoryBlueprint:
        from core.language import detect_language_from_blueprint, get_language_profile
        tech_stack = data.get("tech_stack", {})
        # Derive the project-level language once so it can fill in any missing per-file fields
        project_lang = detect_language_from_blueprint(tech_stack).name

        file_blueprints = [
            FileBlueprint(
                path=fb["path"],
                purpose=fb["purpose"],
                depends_on=fb.get("depends_on", []),
                exports=fb.get("exports", []),
                language=self._resolve_file_language(fb.get("language", ""), fb["path"], project_lang),
                layer=fb.get("layer", ""),
            )
            for fb in data.get("file_blueprints", [])
        ]

        return RepositoryBlueprint(
            name=data.get("name", "generated-project"),
            description=data.get("description", ""),
            architecture_style=data.get("architecture_style", "REST"),
            tech_stack=data.get("tech_stack", {}),
            folder_structure=data.get("folder_structure", []),
            file_blueprints=file_blueprints,
            architecture_doc=data.get("architecture_doc", ""),
        )

    def _resolve_file_language(self, llm_lang: str, path: str, project_lang: str) -> str:
        """Return the correct language tag for a file.

        Priority:
        1. Extension-based override — .java/.go/.ts/.rs/.cs/.py files are unambiguous.
        2. LLM-provided language tag (if non-empty and not the wrong default).
        3. Project-level language derived from tech_stack.
        """
        ext_map = {
            ".py": "python", ".java": "java", ".go": "go",
            ".ts": "typescript", ".rs": "rust", ".cs": "csharp",
            ".kt": "java", ".scala": "java",
        }
        for ext, lang in ext_map.items():
            if path.endswith(ext):
                return lang

        # Non-source files (.properties, .yaml, .xml, .json, pom.xml, Dockerfile …)
        # Tag them with the project language so the coder knows the ecosystem,
        # but the coder agent will detect them as config files from the extension.
        if llm_lang and llm_lang != "python":
            return llm_lang

        return project_lang


