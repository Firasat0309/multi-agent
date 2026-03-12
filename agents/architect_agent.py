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

            "STEP 1 — DETECT LANGUAGE:\n"
            "Detect the programming language from the user's prompt. If they mention "
            "'Java', 'Spring Boot', 'Go', 'Gin', 'TypeScript', 'Express', 'Rust', "
            "'C#', '.NET', etc., use THAT language. If unspecified, default to Python.\n\n"

            "STEP 2 — DETECT DATABASE:\n"
            "If the user explicitly names a database (PostgreSQL, MySQL, MongoDB, Redis, "
            "SQLite, etc.), use it. If they do NOT specify any database:\n"
            "  - Java/Spring Boot → use H2 in-memory (jdbc:h2:mem:testdb)\n"
            "  - Python → use SQLite in-memory\n"
            "  - Go → use SQLite in-memory\n"
            "  - TypeScript/Node → use SQLite in-memory\n"
            "  - Rust → use SQLite in-memory\n"
            "  - C#/.NET → use SQLite in-memory (or InMemory EF provider)\n\n"

            "STEP 3 — DESIGN ARCHITECTURE:\n"
            "Design a clean layered architecture with these layers (bottom to top):\n"
            "  model → repository → service → controller\n"
            "Plus cross-cutting: config, middleware, util\n\n"

            "STEP 4 — PRODUCE JSON:\n"
            "You MUST respond with a single JSON object (no markdown fences, no prose):\n"
            "{\n"
            '  "name": "project-name-in-kebab-case",\n'
            '  "description": "One-sentence description of what this project does",\n'
            '  "architecture_style": "REST" | "GraphQL" | "gRPC",\n'
            '  "tech_stack": {\n'
            '    "language": "<detected-language-lowercase>",\n'
            '    "framework": "<primary-framework>",\n'
            '    "db": "<database>",\n'
            '    "build_tool": "<maven|gradle|go|npm|cargo|dotnet>"\n'
            "  },\n"
            '  "folder_structure": ["src/main/java/com/example/models", ...],\n'
            '  "file_blueprints": [\n'
            "    {\n"
            '      "path": "<full-relative-path-with-extension>",\n'
            '      "purpose": "<one-sentence: what this file does>",\n'
            '      "depends_on": ["<path-of-file-this-imports-from>"],\n'
            '      "exports": ["<ClassName>", "<functionName>"],\n'
            '      "language": "<language-tag>",\n'
            '      "layer": "model|repository|service|controller|config|middleware|util"\n'
            "    }\n"
            "  ],\n"
            '  "architecture_doc": "<markdown-string>"\n'
            "}\n\n"

            "RULES FOR file_blueprints:\n"
            "- Use correct file extensions for the chosen language (.py, .java, .go, .ts, .rs, .cs)\n"
            "- Use idiomatic folder conventions for the language and framework\n"
            "- For Java/Maven: use FULL paths including src/main/java/<package>/... prefix\n"
            "- For Go: use module-relative paths\n"
            "- For Python/TypeScript: use project-root-relative paths\n"
            "- Every file in depends_on MUST reference another file_blueprint path (no dangling references)\n"
            "- depends_on MUST NOT create circular dependencies\n"
            "- Models depend on nothing; repositories depend on models; services depend on "
            "repositories; controllers depend on services\n"
            "- Every file MUST have at least one item in exports\n"
            "- Include 10-20 files for a typical CRUD application (minimum 8, maximum 30)\n"
            "- ALWAYS include the build/project configuration file as the FIRST blueprint:\n"
            "  * Java/Maven: pom.xml  * Java/Gradle: build.gradle\n"
            "  * Go: go.mod  * TypeScript/Node: package.json\n"
            "  * Rust: Cargo.toml  * C#/.NET: *.csproj\n"
            "  * Python: requirements.txt or pyproject.toml\n"
            "- ALWAYS include the application entry point / main configuration file\n"
            "- ALWAYS include the database configuration file (application.properties, .env, etc.)\n"
            "- Use consistent naming conventions throughout: snake_case for Python, "
            "PascalCase for Java/C# classes, camelCase for TypeScript\n\n"

            "RULES FOR architecture_doc:\n"
            "The architecture_doc MUST be a markdown string with these exact sections:\n"
            "# Architecture\\n"
            "## Overview (2-3 sentences)\\n"
            "## Tech Stack (bullet list of language, framework, db, build tool)\\n"
            "## Layer Diagram (ASCII showing model→repo→service→controller)\\n"
            "## API Endpoints (table: Method | Path | Description | Controller)\\n"
            "## Database Schema (table: Entity | Fields | Relationships)\\n"
            "## Configuration (list of config files and their purpose)\\n"
        )

    async def execute(self, context: AgentContext) -> TaskResult:
        """Design a repository architecture from context.task.description."""
        prompt = context.task.metadata.get("user_prompt", context.task.description)
        try:
            blueprint = await self.design_architecture(prompt)
            return TaskResult(
                success=True,
                output=f"Architecture designed: {blueprint.name} ({len(blueprint.file_blueprints)} files)",
                metrics={**self.get_metrics(), "blueprint_name": blueprint.name},
            )
        except Exception as exc:
            logger.exception("ArchitectAgent.execute failed")
            return TaskResult(success=False, errors=[str(exc)])

    # Known database keywords — used to detect an explicit DB choice in user prompts.
    _DB_KEYWORDS: frozenset[str] = frozenset({
        "postgresql", "postgres", "mysql", "mariadb", "oracle", "sqlite",
        "mongodb", "redis", "cassandra", "dynamodb", "mssql", "sql server",
        "cockroachdb", "firestore", "firebase", "supabase", "neo4j", "h2",
        "elasticsearch", "clickhouse", "timescaledb", "planetscale",
    })

    @classmethod
    def _user_specified_db(cls, prompt: str) -> bool:
        """Return True if the prompt explicitly mentions a database technology."""
        lower = prompt.lower()
        return any(kw in lower for kw in cls._DB_KEYWORDS)

    async def design_architecture(self, user_prompt: str) -> RepositoryBlueprint:
        """Design a complete repository blueprint from user requirements.

        Uses a higher max_tokens limit because the architecture JSON (with
        15+ file blueprints and a full architecture_doc) routinely exceeds
        the default 8192 tokens.  If the response is truncated, we attempt
        one continuation to complete the JSON.
        """
        logger.info("Designing architecture from user prompt")

        # Belt-and-suspenders: if the user never mentioned a DB, append an
        # explicit instruction so the LLM cannot silently default to Postgres.
        effective_prompt = user_prompt
        if not self._user_specified_db(user_prompt):
            logger.info("No database specified in prompt — defaulting to H2 in-memory")
            effective_prompt = (
                user_prompt
                + "\n\n[SYSTEM NOTE: The user did not specify a database. "
                "Use H2 in-memory database (db: 'h2'). "
                "Configure it as an embedded in-memory datasource.]"
            )

        # Architecture responses are large — use 16k tokens to avoid truncation.
        response = await self.llm.generate(
            system_prompt=self.system_prompt + "\n\nRespond with valid JSON only. No markdown fences.",
            user_prompt=effective_prompt,
            max_tokens=16384,
        )
        self._metrics["llm_calls"] += 1
        self._metrics["tokens_used"] += sum(response.usage.values())

        text = response.content

        # If still truncated, request one continuation
        if response.stop_reason in ("max_tokens", "length", "MAX_TOKENS"):
            logger.warning("Architecture response truncated — requesting continuation")
            continuation = await self.llm.generate(
                system_prompt=self.system_prompt + "\n\nRespond with valid JSON only. No markdown fences.",
                user_prompt=(
                    f"{effective_prompt}\n\n"
                    f"Your previous JSON response was cut off. Here is the end:\n"
                    f"```\n{text[-500:]}\n```\n"
                    f"Continue EXACTLY from the cut-off point to complete the JSON. "
                    f"Output only the continuation — no preamble."
                ),
                max_tokens=8192,
            )
            self._metrics["llm_calls"] += 1
            self._metrics["tokens_used"] += sum(continuation.usage.values())
            text += continuation.content

        result = self._parse_json_response(text)
        return self._parse_blueprint(result)

    @staticmethod
    def _parse_json_response(text: str) -> dict[str, Any]:
        """Parse JSON from LLM output, handling fences and partial responses."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to extract the outermost JSON object
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
            logger.error("Could not parse architecture JSON: %s...", text[:200])
            return {}

    def _parse_blueprint(self, data: dict[str, Any]) -> RepositoryBlueprint:
        from core.language import detect_language_from_blueprint, get_language_profile
        from core.llm_schema import validate_architecture_response

        # Raises pydantic.ValidationError on structurally invalid data.
        validated = validate_architecture_response(data)

        tech_stack = validated.tech_stack
        project_lang = detect_language_from_blueprint(tech_stack).name

        file_blueprints = [
            FileBlueprint(
                path=fb.path,
                purpose=fb.purpose,
                depends_on=fb.depends_on,
                exports=fb.exports,
                language=self._resolve_file_language(fb.language, fb.path, project_lang),
                layer=fb.layer,
            )
            for fb in validated.file_blueprints
        ]

        return RepositoryBlueprint(
            name=validated.name,
            description=validated.description,
            architecture_style=validated.architecture_style,
            tech_stack=tech_stack,
            folder_structure=validated.folder_structure,
            file_blueprints=file_blueprints,
            architecture_doc=validated.architecture_doc,
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


