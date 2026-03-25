"""Architect agent - creates the repository blueprint from a user prompt."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from agents.base_agent import BaseAgent
from core.language import LANGUAGE_PROFILES, detect_language_from_blueprint
from core.models import (
    AgentContext,
    AgentRole,
    FileBlueprint,
    RepositoryBlueprint,
    TaskResult,
)

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
            "You MUST respond with a single JSON object (no markdown fences, no prose).\n"
            "Do NOT include an 'architecture_doc' field — it will be generated separately.\n"
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
            "  ]\n"
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
            "PascalCase for Java/C# classes, camelCase for TypeScript\n"
            "- BACKEND ONLY: Do NOT include any frontend files. No React, Vue, Angular, "
            "Next.js, Svelte, or other frontend framework files. No .tsx, .jsx, or "
            "frontend/ directories. The frontend is generated by a separate pipeline.\n"
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

    # Language → default in-memory DB when the user doesn't specify one.
    _LANG_DEFAULT_DB: dict[str, str] = {
        "java": "Use H2 in-memory database (db: 'h2').",
        "python": "Use SQLite in-memory database (db: 'sqlite').",
        "go": "Use SQLite in-memory database (db: 'sqlite').",
        "typescript": "Use SQLite in-memory database (db: 'sqlite').",
        "rust": "Use SQLite in-memory database (db: 'sqlite').",
        "csharp": "Use SQLite in-memory database (db: 'sqlite').",
    }

    @classmethod
    def _default_db_note(cls, prompt: str) -> str:
        """Return a language-appropriate default DB instruction.

        Detects the language from the prompt text using common framework
        keywords, then returns the matching in-memory DB recommendation.
        """
        lower = prompt.lower()
        # Check for language/framework keywords in the prompt
        if any(kw in lower for kw in ("java", "spring boot", "spring", "maven", "gradle")):
            return cls._LANG_DEFAULT_DB["java"]
        if any(kw in lower for kw in ("golang", "gin", "echo", "fiber")) or re.search(r'\bgo\b', lower):
            return cls._LANG_DEFAULT_DB["go"]
        if any(kw in lower for kw in ("typescript", "express", "nestjs", "node")):
            return cls._LANG_DEFAULT_DB["typescript"]
        if any(kw in lower for kw in ("rust", "actix", "axum", "rocket")):
            return cls._LANG_DEFAULT_DB["rust"]
        if any(kw in lower for kw in ("c#", "csharp", ".net", "asp.net", "dotnet")):
            return cls._LANG_DEFAULT_DB["csharp"]
        # Default: Python (the pipeline default language)
        return cls._LANG_DEFAULT_DB["python"]

    async def _llm_with_heartbeat(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        label: str = "LLM call",
    ):
        """Call ``self.llm.generate`` with periodic heartbeat logs.

        Long LLM calls (especially on Gemini) can take 30-90s.  Without
        visible output the user thinks the pipeline is stuck.
        """
        _HEARTBEAT = 15  # seconds between "still waiting" messages
        done = asyncio.Event()

        async def _heartbeat() -> None:
            elapsed = 0
            while not done.is_set():
                await asyncio.sleep(_HEARTBEAT)
                if done.is_set():
                    break
                elapsed += _HEARTBEAT
                logger.info(
                    "⏳ Waiting for %s response… (%ds elapsed)",
                    label, elapsed,
                )

        hb = asyncio.create_task(_heartbeat())
        try:
            return await self.llm.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
            )
        finally:
            done.set()
            hb.cancel()

    async def design_architecture(self, user_prompt: str) -> RepositoryBlueprint:
        """Design a complete repository blueprint from user requirements.

        Split into two LLM calls:
        1. Main call — file blueprints + tech_stack (compact JSON, fits in one response)
        2. Doc call  — architecture_doc markdown (fetched separately to avoid truncation)
        """
        logger.info("Designing architecture from user prompt")

        # Belt-and-suspenders: if the user never mentioned a DB, append an
        # explicit instruction so the LLM cannot silently default to Postgres.
        effective_prompt = user_prompt
        if not self._user_specified_db(user_prompt):
            db_note = self._default_db_note(user_prompt)
            logger.info("No database specified in prompt — injecting: %s", db_note)
            effective_prompt = (
                user_prompt
                + f"\n\n[SYSTEM NOTE: The user did not specify a database. {db_note} "
                "Configure it as an embedded in-memory datasource.]"
            )

        # ── Call 1: file blueprints + tech_stack (no architecture_doc) ────────
        logger.info("Sending architecture request to LLM (max_tokens=12288) — this may take 30-90s …")
        response = await self._llm_with_heartbeat(
            system_prompt=self.system_prompt + "\n\nRespond with valid JSON only. No markdown fences.",
            user_prompt=effective_prompt,
            max_tokens=12288,
            label="architecture design",
        )
        self._metrics["llm_calls"] += 1
        self._metrics["tokens_used"] += sum(response.usage.values())

        text = response.content

        if response.stop_reason in ("max_tokens", "length", "MAX_TOKENS"):
            logger.warning(
                "Architecture response truncated at %d chars — JSON is likely incomplete",
                len(text),
            )

        try:
            result = self._parse_json_response(text)
        except ValueError as parse_err:
            # Retry: ask the LLM to fix its own malformed JSON
            logger.warning(
                "Architecture JSON parse failed (%s) — retrying with corrective prompt",
                parse_err,
            )
            result = await self._retry_json_parse(text, parse_err)

        blueprint = self._parse_blueprint(result)

        # ── Call 2: architecture_doc (separate, non-critical) ─────────────────
        blueprint = await self._fetch_architecture_doc(blueprint, effective_prompt)

        return blueprint

    async def revise_architecture(
        self,
        user_prompt: str,
        current: RepositoryBlueprint,
        feedback: str,
    ) -> RepositoryBlueprint:
        """Revise an existing architecture blueprint based on user feedback."""
        logger.info("Revising architecture with user feedback")
        file_list = "\n".join(
            f"  - {fb.path} [{fb.layer}]: {fb.purpose}"
            for fb in current.file_blueprints
        )
        revision_prompt = (
            f"Original requirements:\n{user_prompt}\n\n"
            f"Current architecture:\n"
            f"  Name: {current.name}\n"
            f"  Style: {current.architecture_style}\n"
            f"  Tech stack: {json.dumps(current.tech_stack)}\n"
            f"  Files:\n{file_list}\n\n"
            f"User feedback — apply these changes:\n{feedback}\n\n"
            "Produce the REVISED architecture JSON now. "
            "Keep everything the user did not mention, only change what they asked for."
        )

        effective_prompt = revision_prompt
        if not self._user_specified_db(user_prompt) and not self._user_specified_db(feedback):
            db_note = self._default_db_note(user_prompt)
            effective_prompt += (
                f"\n\n[SYSTEM NOTE: The user did not specify a database. {db_note}]"
            )

        response = await self._llm_with_heartbeat(
            system_prompt=self.system_prompt + "\n\nRespond with valid JSON only. No markdown fences.",
            user_prompt=effective_prompt,
            max_tokens=12288,
            label="architecture revision",
        )
        self._metrics["llm_calls"] += 1
        self._metrics["tokens_used"] += sum(response.usage.values())

        try:
            result = self._parse_json_response(response.content)
        except ValueError as parse_err:
            result = await self._retry_json_parse(response.content, parse_err)

        blueprint = self._parse_blueprint(result)
        blueprint = await self._fetch_architecture_doc(blueprint, effective_prompt)
        return blueprint

    async def _fetch_architecture_doc(
        self, blueprint: RepositoryBlueprint, user_prompt: str
    ) -> RepositoryBlueprint:
        """Fetch architecture documentation in a separate LLM call.

        This is intentionally non-fatal — if it fails, the blueprint is still
        usable with an empty architecture_doc.
        """
        file_list = "\n".join(
            f"  - {fb.path} [{fb.layer}]: {fb.purpose}"
            for fb in blueprint.file_blueprints
        )
        doc_prompt = (
            f"Generate architecture documentation for this project.\n\n"
            f"Project: {blueprint.name}\n"
            f"Description: {blueprint.description}\n"
            f"Style: {blueprint.architecture_style}\n"
            f"Tech stack: {blueprint.tech_stack}\n"
            f"Files:\n{file_list}\n\n"
            f"Original requirements: {user_prompt[:2000]}\n\n"
            "Write a markdown document with these exact sections:\n"
            "# Architecture\n"
            "## Overview (2-3 sentences)\n"
            "## Tech Stack (bullet list of language, framework, db, build tool)\n"
            "## Layer Diagram (ASCII showing model→repo→service→controller)\n"
            "## API Endpoints (table: Method | Path | Description | Controller)\n"
            "## Database Schema (table: Entity | Fields | Relationships)\n"
            "## Configuration (list of config files and their purpose)\n\n"
            "Output ONLY the markdown — no JSON wrapper, no fences."
        )

        try:
            logger.info("Fetching architecture documentation…")
            doc_response = await self._llm_with_heartbeat(
                system_prompt="You are a technical writer. Output clean markdown only.",
                user_prompt=doc_prompt,
                max_tokens=4096,
                label="architecture doc",
            )
            self._metrics["llm_calls"] += 1
            self._metrics["tokens_used"] += sum(doc_response.usage.values())
            blueprint.architecture_doc = doc_response.content.strip()
        except Exception:
            logger.warning("Failed to fetch architecture doc — continuing without it")

        return blueprint

    @staticmethod
    def _repair_json(text: str) -> str:
        """Best-effort repair of common LLM JSON mistakes.

        Delegates to the centralized repair in LLMClient which handles
        trailing commas, single-quoted strings, JS comments, unquoted
        property names, AND truncated output (unclosed braces/brackets).
        """
        from core.llm_client import LLMClient
        return LLMClient._repair_json_text(text)

    @classmethod
    def _parse_json_response(cls, text: str) -> dict[str, Any]:
        """Parse JSON from LLM output, handling fences, partial responses, and repair."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        # 1. Try strict parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 2. Try extracting the outermost JSON object
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass

        # 3. Try repair (handles trailing commas, truncated JSON, etc.)
        try:
            extracted = text[start:] if start != -1 else text
            repaired = cls._repair_json(extracted)
            result = json.loads(repaired)
            logger.info("Architecture JSON repair succeeded")
            return result
        except (json.JSONDecodeError, Exception):
            pass

        logger.error("Could not parse architecture JSON: %s...", text[:200])
        raise ValueError(
            "Failed to parse architecture JSON from LLM response. "
            "The model returned malformed or truncated JSON. "
        )

    async def _retry_json_parse(
        self, malformed_text: str, parse_error: Exception
    ) -> dict[str, Any]:
        """Ask the LLM to fix its own malformed architecture JSON.

        Returns the parsed dict on success, raises ValueError on failure.
        """
        corrective_prompt = (
            "Your previous response was not valid JSON. The parser returned:\n"
            f"  {parse_error}\n\n"
            "Here is the broken output (first 3000 chars):\n"
            f"{malformed_text[:3000]}\n\n"
            "Please return ONLY the corrected and COMPLETE JSON object — no explanation, "
            "no markdown fences. Fix the syntax error and output valid JSON.\n"
            "Make sure all arrays and objects are properly closed."
        )
        retry_response = await self._llm_with_heartbeat(
            system_prompt=self.system_prompt + "\n\nRespond with valid JSON only. No markdown fences.",
            user_prompt=corrective_prompt,
            max_tokens=12288,
            label="architecture JSON retry",
        )
        self._metrics["llm_calls"] += 1
        self._metrics["tokens_used"] += sum(retry_response.usage.values())

        try:
            result = self._parse_json_response(retry_response.content)
            logger.info("Architecture JSON retry succeeded")
            return result
        except ValueError:
            raise ValueError(
                "Failed to parse architecture JSON after retry. "
                "The model returned malformed or truncated JSON on both attempts. "
                "Please try again — the LLM may need a simpler prompt or a different provider."
            )

    def _parse_blueprint(self, data: dict[str, Any]) -> RepositoryBlueprint:
        from core.llm_schema import validate_architecture_response

        # Raises pydantic.ValidationError on structurally invalid data.
        validated = validate_architecture_response(data)

        # Fail fast if the LLM returned valid JSON but with no useful content
        if not validated.file_blueprints:
            raise ValueError(
                "Architecture design returned 0 file blueprints. "
                "The LLM response was valid JSON but contained no files to generate. "
                "This usually means the response was truncated or malformed. "
                "Please try again."
            )
        if not validated.tech_stack.get("language"):
            raise ValueError(
                "Architecture design is missing 'language' in tech_stack. "
                "The LLM did not specify a programming language. "
                "Please try again with an explicit language in your prompt "
                "(e.g. 'Java Spring Boot', 'Python FastAPI')."
            )

        # Validate that the language is recognized — don't silently default
        raw_lang = validated.tech_stack["language"].lower().strip()
        if raw_lang not in LANGUAGE_PROFILES:
            supported = ", ".join(sorted(LANGUAGE_PROFILES.keys()))
            raise ValueError(
                f"Architecture design specified unrecognized language '{raw_lang}'. "
                f"Supported languages: {supported}. "
                f"Please try again with a supported language in your prompt."
            )

        # Detect duplicate file paths — last-one-wins silently loses work
        seen_paths: dict[str, int] = {}
        for i, fb in enumerate(validated.file_blueprints):
            if fb.path in seen_paths:
                logger.warning(
                    "Duplicate file path '%s' in blueprint (indices %d and %d) "
                    "— keeping first occurrence, dropping duplicate",
                    fb.path, seen_paths[fb.path], i,
                )
            else:
                seen_paths[fb.path] = i
        if len(seen_paths) < len(validated.file_blueprints):
            # Deduplicate: keep first occurrence of each path
            deduped: list[Any] = []
            seen: set[str] = set()
            for fb in validated.file_blueprints:
                if fb.path not in seen:
                    deduped.append(fb)
                    seen.add(fb.path)
            validated.file_blueprints = deduped

        # Safety net: strip any frontend files the LLM included despite instructions.
        # The frontend is generated by a separate pipeline — having them here
        # causes duplicate/conflicting output.
        _FE_MARKERS = ("frontend/", "src/app/", "src/pages/", "src/components/", "src/store/")
        _FE_EXTENSIONS = (".tsx", ".jsx")
        before_count = len(validated.file_blueprints)
        validated.file_blueprints = [
            fb for fb in validated.file_blueprints
            if not any(fb.path.startswith(m) for m in _FE_MARKERS)
            and not any(fb.path.endswith(ext) for ext in _FE_EXTENSIONS)
        ]
        dropped = before_count - len(validated.file_blueprints)
        if dropped:
            logger.warning(
                "Dropped %d frontend file(s) from backend blueprint — "
                "frontend is handled by a separate pipeline",
                dropped,
            )
            # Clean dangling depends_on references to dropped files
            remaining_paths = {fb.path for fb in validated.file_blueprints}
            for fb in validated.file_blueprints:
                fb.depends_on = [d for d in fb.depends_on if d in remaining_paths]

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

        # --- Mandatory Spring Boot files injection ---
        # LLMs frequently omit Application.java and application.properties.
        # Inject them when the project is Java-based and they're missing.
        if project_lang == "java":
            existing_paths = {fb.path for fb in file_blueprints}

            # Detect common Java package prefix from existing source paths
            java_dirs = [
                fp for fp in existing_paths
                if fp.endswith(".java") and "src/main/java/" in fp
            ]
            pkg_prefix = ""
            # Known layer directory names — if the parent dir of a file is one
            # of these, it's a sub-package, not the project root package.
            _LAYER_DIRS = frozenset({
                "controller", "controllers", "service", "services",
                "repository", "repositories", "model", "models",
                "entity", "entities", "config", "configuration",
                "util", "utils", "helper", "helpers", "dto", "dtos",
                "exception", "exceptions", "filter", "filters",
                "security", "middleware", "interceptor", "mapper",
            })
            if java_dirs:
                # Compute parent directories of all Java source files
                java_parent_dirs = sorted(set(
                    "/".join(fp.split("/")[:-1]) for fp in java_dirs
                ))
                # Compute common path prefix across all parent dirs
                split_dirs = [d.split("/") for d in java_parent_dirs]
                min_len = min(len(d) for d in split_dirs)
                common_parts: list[str] = []
                for i in range(min_len):
                    vals = {d[i] for d in split_dirs}
                    if len(vals) == 1:
                        common_parts.append(vals.pop())
                    else:
                        break
                # If all files are in the same layer dir, the common prefix
                # ends with that layer — strip it to get the root package.
                if common_parts and common_parts[-1].lower() in _LAYER_DIRS:
                    common_parts = common_parts[:-1]
                pkg_prefix = "/".join(common_parts) if common_parts else "src/main/java"
            else:
                pkg_prefix = "src/main/java"

            # Inject Application.java if no entry-point class exists
            has_entry = any(
                fp.endswith(("Application.java", "App.java", "Main.java"))
                for fp in existing_paths
            )
            if not has_entry:
                # Derive class name from project name
                raw_name = validated.name.replace("-", " ").replace("_", " ")
                class_name = "".join(w.capitalize() for w in raw_name.split()) + "Application"
                # Java class names must start with a letter
                if class_name and not class_name[0].isalpha():
                    class_name = "App" + class_name
                app_path = f"{pkg_prefix}/{class_name}.java"
                file_blueprints.append(FileBlueprint(
                    path=app_path,
                    purpose=f"@SpringBootApplication entry point — {class_name}",
                    depends_on=[],
                    exports=[class_name],
                    language="java",
                    layer="infrastructure",
                ))
                logger.warning(
                    "Injected mandatory %s — LLM omitted the Spring Boot entry point",
                    app_path,
                )

            # Inject application.properties if no config file exists
            has_config = any(
                fp.endswith(("application.properties", "application.yml", "application.yaml"))
                for fp in existing_paths
            )
            if not has_config:
                props_path = "src/main/resources/application.properties"
                # Detect whether the project includes JWT / Spring Security so the
                # generated properties file includes the required keys.
                _lower_paths = " ".join(existing_paths).lower()
                # Use word-boundary for 'auth' to avoid false positives (e.g. 'AuthorController')
                _has_jwt = (
                    any(kw in _lower_paths for kw in ("jwt", "security", "token"))
                    or bool(re.search(r'\bauth\b', _lower_paths))
                )
                _props_purpose = (
                    "Spring Boot application configuration — server port, datasource, JPA settings"
                )
                if _has_jwt:
                    _props_purpose += (
                        "; JWT security settings (jwt.secret, jwt.expiration=86400000)"
                        "; Spring Security CORS and session management"
                    )
                file_blueprints.append(FileBlueprint(
                    path=props_path,
                    purpose=_props_purpose,
                    depends_on=[],
                    exports=[],
                    language="java",
                    layer="infrastructure",
                ))
                logger.warning(
                    "Injected mandatory %s — LLM omitted Spring Boot configuration",
                    props_path,
                )

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
        2. Project-level language for everything else (config, build, resource files).

        Non-source files (.properties, .yaml, .xml, pom.xml, Dockerfile, etc.)
        always get the project language so downstream agents know the ecosystem
        and use the correct build tools, conventions, and prompts.
        """
        ext_map = {
            ".py": "python", ".java": "java", ".go": "go",
            ".ts": "typescript", ".rs": "rust", ".cs": "csharp",
            ".kt": "java", ".scala": "java",
        }
        for ext, lang in ext_map.items():
            if path.endswith(ext):
                return lang

        return project_lang


