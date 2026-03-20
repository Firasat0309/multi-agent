"""Coder agent - generates code for a specific file following the blueprint."""

from __future__ import annotations

import logging
import re as _re
from pathlib import Path
from typing import Any

from agents.base_agent import BaseAgent
from core.agent_tools import CODER_TOOLS, ToolDefinition
from core.error_attributor import extract_error_lines
from core.language import get_language_profile, LanguageProfile
from core.models import AgentContext, AgentRole, TaskResult, TaskType

# Files with more lines than this threshold are modified via unified diff instead
# of a full-file rewrite.  Diffs are ~5–20× cheaper in tokens for large files.
_LARGE_FILE_THRESHOLD = 200

# Max chars of build/compiler output to include in fix prompts.
# Maven can dump 50K+ chars; the relevant error lines are always near the end.
_MAX_ERROR_CHARS = 3_000
# Max chars of a single related file included in fix context.
_MAX_RELATED_FILE_CHARS = 2_000

# Maximum allowed content growth factor for fix/modify rewrites.
# If new content exceeds original_size * this factor, the rewrite is rejected
# as it likely contains duplicated code from LLM hallucination.
_MAX_CONTENT_GROWTH = 1.35

logger = logging.getLogger(__name__)


def _has_duplicate_definitions(content: str, language: str) -> bool:
    """Detect duplicate class/function/method definitions in source code.

    Returns True if the same top-level definition name appears more than once,
    which is a strong signal of LLM output duplication.
    """
    # Language-specific patterns for top-level definitions
    patterns: dict[str, _re.Pattern[str]] = {
        "python": _re.compile(r"^(?:class|def|async\s+def)\s+(\w+)", _re.MULTILINE),
        "java": _re.compile(r"^\s*(?:public|private|protected)?\s*(?:static\s+)?(?:class|interface|enum|record)\s+(\w+)", _re.MULTILINE),
        "typescript": _re.compile(r"^(?:export\s+)?(?:class|interface|enum|function|const|type)\s+(\w+)", _re.MULTILINE),
        "go": _re.compile(r"^(?:func|type)\s+(\w+)", _re.MULTILINE),
        "csharp": _re.compile(r"^\s*(?:public|private|protected|internal)?\s*(?:static\s+)?(?:class|interface|enum|struct|record)\s+(\w+)", _re.MULTILINE),
        "rust": _re.compile(r"^(?:pub\s+)?(?:fn|struct|enum|trait|type)\s+(\w+)", _re.MULTILINE),
    }
    pattern = patterns.get(language)
    if not pattern:
        return False
    names = pattern.findall(content)
    if language == "go":
        # Go permits multiple init functions per file; they should not be treated as duplicates.
        names = [name for name in names if name != "init"]
    return len(names) != len(set(names))


def _validate_rewrite(
    new_content: str,
    original_content: str,
    profile: LanguageProfile,
    file_path: str,
    operation: str,
) -> TaskResult | None:
    """Validate a code rewrite and return TaskResult if validation fails, None if valid.

    Returns ``success=True`` with ``files_modified=[]`` when the rewrite is
    rejected.  This is deliberate: in the lifecycle flow,
    ``success=False`` fires ``RETRIES_EXHAUSTED`` which **immediately kills
    the file** — a single bad LLM output (probabilistic!) would permanently
    block the file and cascade failures to all downstream dependents.
    ``success=True`` fires ``FIX_APPLIED`` instead, which sends the file back
    through the review→fix cycle for another attempt.  It burns one review
    round on unchanged code, but the file stays alive and the fix-count
    budget eventually terminates the loop gracefully via DEGRADED.

    For the checkpoint path (``_dispatch_fix``), the return value is
    discarded anyway — the build simply re-runs regardless.
    """
    original_size = len(original_content)

    # ── Guard: reject rewrites that inflate the file (likely LLM duplication) ──
    if original_size > 0 and len(new_content) > original_size * _MAX_CONTENT_GROWTH:
        logger.warning(
            "%s rewrite rejected for %s: new size (%d) exceeds %.0f%% of original (%d). "
            "Keeping original to prevent content duplication.",
            operation, file_path, len(new_content), _MAX_CONTENT_GROWTH * 100, original_size,
        )
        return TaskResult(
            success=True,
            output=f"{operation} for {file_path} skipped — rewrite was too large (likely duplicated content)",
            files_modified=[],
            metrics={"rewrite_rejected": True},
        )

    # ── Guard: detect duplicate class/function definitions ────────────────
    if _has_duplicate_definitions(new_content, profile.name):
        logger.warning(
            "%s rewrite rejected for %s: duplicate definitions detected. "
            "Keeping original to prevent content duplication.",
            operation, file_path,
        )
        return TaskResult(
            success=True,
            output=f"{operation} for {file_path} skipped — duplicate definitions detected in LLM output",
            files_modified=[],
            metrics={"rewrite_rejected": True},
        )

    # ── Guard: skip write if content is identical ─────────────────────────
    if new_content.rstrip() == original_content.rstrip():
        logger.info("%s for %s produced identical content — skipping write", operation, file_path)
        return TaskResult(
            success=True,
            output=f"No changes needed for {file_path} (content unchanged after {operation.lower()})",
            files_modified=[],
            metrics={},
        )

    return None  # Validation passed


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

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Set before execute_agentic() so system_prompt uses the right language.
        self._current_language: str = "python"

    @property
    def tools(self) -> list[ToolDefinition]:
        return CODER_TOOLS

    # Per-language syntax rules injected into the system prompt so the LLM
    # doesn't produce the most common syntax errors for that language.
    _LANGUAGE_SYNTAX_RULES: dict[str, str] = {
        "java": (
            "\n\nCRITICAL Java syntax rules — every violation makes the file uncompilable:\n"
            "- Every import statement MUST end with a semicolon: `import java.util.List;`\n"
            "- Every field/variable declaration MUST end with exactly ONE semicolon — NEVER `;;`\n"
            "- serialVersionUID must be initialised: `private static final long serialVersionUID = 1L;`\n"
            "- Every statement (assignment, return, method call, throw) MUST end with a semicolon\n"
            "- Method/constructor signatures do NOT end with a semicolon — only close with `{}`\n"
            "- Class/interface/enum declarations open with `{` and close with `}` — no trailing semicolon\n"
            "- Annotations (@Override, @Autowired, @Entity) go on the line BEFORE the annotated element\n"
            "- Generic type parameters use angle brackets: `List<String>`, not raw `List`\n"
            "- Before writing the file, mentally scan every line for missing or duplicate semicolons"
        ),
        "typescript": (
            "\n\nCRITICAL TypeScript syntax rules:\n"
            "- Every import statement MUST end with a semicolon\n"
            "- Every statement MUST end with a semicolon\n"
            "- Use explicit return types on public functions and methods\n"
            "- Interface and type declarations do NOT use `=` unless it is a type alias\n"
            "- Decorators (@Injectable, @Component) go on the line BEFORE the class/method"
        ),
        "csharp": (
            "\n\nCRITICAL C# syntax rules:\n"
            "- Every using statement MUST end with a semicolon: `using System.Linq;`\n"
            "- Every field/property/statement MUST end with a semicolon\n"
            "- Attributes ([HttpGet], [Authorize]) go on the line BEFORE the annotated element\n"
            "- Auto-properties use `{ get; set; }` syntax"
        ),
    }

    def _get_source_system_prompt(self, language: str) -> str:
        profile = get_language_profile(language)
        lang_rules = self._LANGUAGE_SYNTAX_RULES.get(profile.name, "")
        return (
            f"You are an expert {profile.display_name} developer agent working inside "
            f"an automated code generation pipeline.\n\n"

            "YOUR TASK: Generate a single, complete, production-quality source file.\n\n"

            "CRITICAL REQUIREMENTS:\n"
            f"1. Output ONLY raw {profile.display_name} code — no markdown fences, no ```, "
            "no explanations, no comments like '// rest of code here'\n"
            "2. The file MUST be syntactically valid and compilable on its own\n"
            "3. EVERY class, method, and function must be FULLY implemented — no stubs, "
            "no TODOs, no placeholder comments, no '// implement later'\n"
            "4. EVERY import/dependency must match exactly what the dependency files export "
            "(check class names, method signatures, package paths)\n"
            "5. Follow the file blueprint EXACTLY: implement all listed exports\n\n"

            "DEPENDENCY RULES (CRITICAL — violations cause build failures):\n"
            "- Only import from files listed in 'depends_on' — never invent dependencies\n"
            "- Match the exact class/function names from the dependency's 'exports' list\n"
            "- Use the correct import path based on the dependency's file path\n"
            "- Related Files contains AST stubs showing the REAL API surface of "
            "dependency files — these are extracted from actual generated source code\n"
            "- Use EXACT method names from the AST stub — do NOT rename, abbreviate, "
            "or 'improve' method names (e.g., stub shows 'registerUser' → call "
            "'registerUser', NOT 'register')\n\n"

            "CODE QUALITY RULES:\n"
            "- Use dependency injection (constructor injection preferred)\n"
            "- Include proper error handling: validate inputs, catch exceptions, "
            "return meaningful error responses\n"
            "- Use the framework's idiomatic patterns (annotations, decorators, etc.)\n"
            f"- Follow idiomatic {profile.display_name} naming conventions\n"
            "- Use proper type annotations/generics throughout\n"
            "- Do NOT import modules that aren't in the dependency graph"
            f"{lang_rules}"
        )

    def _get_config_system_prompt(self, fmt: str) -> str:
        return (
            f"You are a configuration file generator. You produce correct, complete {fmt} files.\n\n"
            "CRITICAL REQUIREMENTS:\n"
            f"1. Output ONLY valid {fmt} content — no code, no markdown fences, no explanations\n"
            "2. The file MUST be syntactically valid and parseable\n"
            "3. Use realistic, appropriate values for the project and tech stack\n"
            "4. Include ALL required dependencies, plugins, and configurations\n"
            "5. Do NOT truncate the file — output the COMPLETE content\n"
            "6. Do NOT wrap the output in any programming language or markdown\n"
            "7. Include comments where the format supports them to explain non-obvious settings"
        )

    @property
    def system_prompt(self) -> str:
        return self._get_source_system_prompt(self._current_language)

    def _build_prompt(self, context: AgentContext) -> str:
        """Initial user message for the agentic GENERATE_FILE loop."""
        fb = context.file_blueprint
        if fb is None:
            return self._format_context(context)
        lang = fb.language or context.blueprint.tech_stack.get("language", "python")
        profile = get_language_profile(lang)

        # For compiled languages, embed a pre-write syntax reminder directly in
        # the user turn so the model sees it immediately before producing code.
        syntax_reminder = ""
        lang_rules = self._LANGUAGE_SYNTAX_RULES.get(profile.name, "")
        if lang_rules:
            syntax_reminder = (
                f"\n\nBefore calling write_file, verify every line of your code against "
                f"these {profile.display_name} syntax rules:{lang_rules}"
            )

        # ── Extract dependency method signatures for inline embedding ────
        # LLMs (especially smaller ones) may skim Related Files and guess
        # method names / return types.  Extracting the key signatures and
        # placing them directly in the instruction block makes mismatches
        # much less likely.
        dep_signatures = self._extract_dep_signatures(context)

        # ── API contract section for controller / handler / route files ──
        # When an API contract is available and this file is a controller-layer
        # file, embed the relevant endpoint details directly in the prompt so
        # the model generates the exact request/response types required.
        api_contract_section = self._extract_api_contract_section(context)

        return (
            f"{self._format_context(context)}\n\n"
            f"Generate the complete {profile.display_name} file for: {fb.path}\n"
            f"Purpose: {fb.purpose}\n"
            f"Layer: {fb.layer}\n"
            f"Must export: {', '.join(fb.exports) if fb.exports else 'appropriate classes/functions'}\n\n"
            f"{api_contract_section}"
            f"{dep_signatures}"
            f"INSTRUCTIONS:\n"
            f"1. Check the Related Files section below — dependency signatures are shown as "
            f"AST stubs with EXACT method names, parameter types, and return types\n"
            f"2. If a dependency IS shown in Related Files, use its EXACT method names — "
            f"the stub shows the real generated code's API surface\n"
            f"3. If a dependency is NOT shown (it may not exist yet), use read_file to "
            f"check if it exists on disk before writing code\n"
            f"4. Write the COMPLETE, WORKING {profile.display_name} code via write_file\n\n"
            f"CRITICAL — CROSS-FILE CONSISTENCY:\n"
            f"- AST stubs in Related Files show the REAL method signatures from generated "
            f"dependency code. These are NOT placeholders — they are the actual API.\n"
            f"- When calling methods on injected services/dependencies, copy the EXACT method "
            f"names from the stub (e.g., if stub shows 'registerUser(RegisterRequest)', "
            f"call 'registerUser(request)', NOT 'register(request)')\n"
            f"- Match parameter types and return types exactly as shown in the stub\n\n"
            f"COMPLETENESS CHECKLIST (verify before calling write_file):\n"
            f"- Every export listed in the blueprint is defined\n"
            f"- Every import matches a real class/function from a dependency file\n"
            f"- Every method call matches the EXACT signature from the dependency\n"
            f"- Every method has a full implementation (no empty bodies, no TODOs)\n"
            f"- All error cases are handled (try/catch, null checks, validation)\n"
            f"- The file is syntactically valid {profile.display_name}\n\n"
            f"Call write_file with path='{fb.path}' and the complete code. "
            f"Do NOT output code as plain text — always use the write_file tool."
            f"{syntax_reminder}"
        )

    @staticmethod
    def _extract_api_contract_section(context: AgentContext) -> str:
        """Return an API contract block for any code file (denylist approach).

        Injects contract details to ALL files except non-code layers where
        the contract is irrelevant (config, infrastructure, build, deploy,
        docs).  This is language-agnostic — no hardcoded allowlist of layer
        names that would silently miss architect-chosen naming conventions.

        The injection format adapts based on what the file likely needs:

        - **Endpoint-facing layers** (file stem contains controller/handler/
          router/route/service keywords, or layer is explicitly one of those):
          Full endpoint details with request/response schemas.

        - **All other code layers** (models, DTOs, repositories, mappers,
          adapters, utilities, etc.):
          Shared schema definitions so field names and types align across the
          entire codebase.

        Endpoint filtering uses a two-pass strategy so files like
        ``UserAuthController.java`` still match ``/users`` endpoints:
          Pass 1 — check if the file-stem word appears in the endpoint path or tags.
          Pass 2 — fall back to all endpoints when no matches found.
        """
        fb = context.file_blueprint
        if not context.api_contract or not fb:
            return ""

        ac = context.api_contract
        layer = fb.layer.lower() if fb.layer else ""

        # Skip layers where the API contract adds no value.
        _SKIP_LAYERS = {
            "config", "configuration", "infrastructure", "build", "deploy",
            "docs", "documentation", "test", "testing", "migration",
        }
        if layer in _SKIP_LAYERS:
            return ""

        # Also skip common config file patterns by path when layer is empty
        # or non-standard (architect didn't set a recognised layer name).
        # Uses filename/segment matching (not substring) to avoid false
        # positives like "reconfiguration" matching "config".
        path_lower = fb.path.lower()
        _path_parts = {p.lower() for p in Path(fb.path).parts}
        _filename = Path(fb.path).name.lower()
        _SKIP_PATH_SEGMENTS = {
            "config", "configuration", "migration", "migrations",
        }
        _SKIP_FILENAMES = {
            "dockerfile", "docker-compose.yml", "docker-compose.yaml",
            "webpack.config.js", "webpack.config.ts",
            "vite.config.js", "vite.config.ts", "vite.config.mjs",
            "tsconfig.json", "tsconfig.base.json",
            ".eslintrc.js", ".eslintrc.json", "eslint.config.js",
            ".prettierrc", ".prettierrc.json",
            "build.gradle", "build.gradle.kts", "settings.gradle",
            "pom.xml",
            "settings.py", "manage.py",
            "application.properties", "application.yml", "application.yaml",
        }
        if _path_parts & _SKIP_PATH_SEGMENTS or _filename in _SKIP_FILENAMES:
            return ""

        if not ac.endpoints and not ac.schemas:
            return ""

        import re as _re
        import json as _json

        # ── Derive entity keywords from multiple sources ─────────────────
        # Language-agnostic: works for CamelCase (Java/C#/Kotlin),
        # snake_case (Python/Rust), dot-separated (TS), and directory-
        # based naming (Go: internal/user/handler.go).
        _NOISE_WORDS = {
            # Layer/role suffixes (all languages)
            "controller", "handler", "router", "route", "resource", "api",
            "service", "usecase", "repository", "repo", "impl", "model",
            "entity", "dto", "request", "response", "schema", "store",
            "crud", "views", "view", "concerns", "middleware", "guard",
            "interceptor", "filter", "pipe", "module", "provider",
            "factory", "adapter", "mapper", "gateway", "facade",
            "transport", "delivery", "presentation", "domain", "port",
            # Generic filenames
            "index", "mod", "lib", "main", "base", "abstract", "interface",
            "types", "utils", "helpers", "constants", "common", "shared",
            "src", "app", "internal", "pkg", "cmd",
        }

        def _tokenize(text: str) -> set[str]:
            """Split text on CamelCase, snake_case, dots, hyphens."""
            parts = _re.split(r"(?<=[a-z])(?=[A-Z])|[^a-zA-Z0-9]+", text)
            return {w.lower() for w in parts if len(w) > 1}

        def _singularize(word: str) -> str:
            """Simple English singular: tasks→task, entities→entity.

            Conservative: only strips suffixes that are almost always
            inflectional. Leaves ambiguous words (status, address, bus)
            untouched to avoid corrupting keywords.
            """
            _NO_STRIP = {
                "status", "address", "bus", "class", "process",
                "access", "success", "progress", "canvas", "alias",
                "bonus", "campus", "corpus", "focus", "genus",
                "nexus", "radius", "stimulus", "virus", "consensus",
            }
            if word in _NO_STRIP:
                return word
            if word.endswith("ies") and len(word) > 4:
                return word[:-3] + "y"
            if word.endswith("ses") or word.endswith("xes") or word.endswith("zes"):
                return word[:-2]
            if word.endswith("es") and len(word) > 4:
                return word[:-2]
            if word.endswith("s") and len(word) > 3 and not word.endswith("ss"):
                return word[:-1]
            return word

        raw_words: set[str] = set()

        # Source 1: file stem (TaskController.java → task)
        raw_words |= _tokenize(Path(fb.path).stem)

        # Source 2: parent directories (internal/user/handler.go → user)
        for part in Path(fb.path).parent.parts:
            raw_words |= _tokenize(part)

        # Source 3: blueprint purpose ("Task business logic" → task)
        if fb.purpose:
            raw_words |= _tokenize(fb.purpose)

        # Source 4: blueprint exports (["TaskService"] → task)
        for export in (fb.exports or []):
            raw_words |= _tokenize(export)

        # Filter noise and add singular forms
        words: set[str] = set()
        for w in raw_words:
            if w not in _NOISE_WORDS:
                words.add(w)
                singular = _singularize(w)
                if singular != w:
                    words.add(singular)

        # ── Detect whether this file is endpoint-facing ──────────────────
        # Check layer, path tokens, AND purpose tokens — covers all language
        # conventions. Uses token-level matching (not substring) to avoid
        # false positives like "forest" matching "rest".
        _ENDPOINT_HINTS = {
            "controller", "handler", "router", "route", "api", "resource",
            "service", "usecase", "endpoint", "transport", "delivery",
            "rest", "web", "grpc", "graphql", "crud", "views",
        }
        _check_tokens = {layer} | _tokenize(path_lower) | _tokenize((fb.purpose or "").lower())
        is_endpoint_facing = bool(_check_tokens & _ENDPOINT_HINTS)

        # ── Endpoint-facing files: full endpoint details + schemas ───────
        if is_endpoint_facing and ac.endpoints:
            def _matches(ep: "APIEndpoint") -> bool:
                ep_path_lower = ep.path.lower()
                tags_lower = " ".join(ep.tags).lower()
                return any(w in ep_path_lower or w in tags_lower for w in words)

            relevant = [ep for ep in ac.endpoints if _matches(ep)] or list(ac.endpoints)

            lines = [
                "API CONTRACT — implement these endpoints EXACTLY as specified:",
                f"  Base URL: {ac.base_url}",
            ]
            for ep in relevant:
                auth = " [requires authentication]" if ep.auth_required else ""
                lines.append(f"  {ep.method} {ep.path}{auth}")
                lines.append(f"    Description: {ep.description}")
                if ep.request_schema:
                    try:
                        lines.append(f"    Request body: {_json.dumps(ep.request_schema)}")
                    except Exception:
                        pass
                if ep.response_schema:
                    try:
                        lines.append(f"    Response body: {_json.dumps(ep.response_schema)}")
                    except Exception:
                        pass

            # Append relevant shared schemas
            if ac.schemas:
                schema_lines: list[str] = []
                ep_text = " ".join(
                    _json.dumps(ep.request_schema or {}) + _json.dumps(ep.response_schema or {})
                    for ep in relevant
                )
                for schema_name, schema_def in ac.schemas.items():
                    if schema_name.lower() in ep_text.lower() or any(
                        w in schema_name.lower() for w in words
                    ):
                        try:
                            schema_lines.append(
                                f"  {schema_name}: {_json.dumps(schema_def)}"
                            )
                        except Exception:
                            pass
                if schema_lines:
                    lines.append("")
                    lines.append("Shared DTO/Model schemas (use these EXACT field names and types):")
                    lines.extend(schema_lines)

            lines.append("")
            return "\n".join(lines) + "\n"

        # ── All other code files: inject schema definitions ──────────────
        if not ac.schemas:
            return ""

        # Filter schemas by keyword match.  Only fall back to all schemas
        # when we have entity keywords but none matched — if `words` is
        # empty (all tokens were noise), we have no signal and should not
        # dump the entire schema catalog into the prompt.
        def _schema_matches(name: str) -> bool:
            name_lower = name.lower()
            return any(w in name_lower for w in words)

        if not words:
            return ""

        relevant_schemas = {
            k: v for k, v in ac.schemas.items() if _schema_matches(k)
        }
        if not relevant_schemas:
            relevant_schemas = ac.schemas

        lines = [
            "API CONTRACT SCHEMAS — your model/DTO MUST match these definitions exactly.",
            "Field names, types, and required/optional status must align so controllers",
            "and services that reference this model compile without type mismatches:",
            "",
        ]
        for schema_name, schema_def in relevant_schemas.items():
            lines.append(f"  Schema: {schema_name}")
            try:
                lines.append(f"    {_json.dumps(schema_def, indent=4)}")
            except Exception:
                lines.append(f"    {schema_def}")
        lines.append("")

        # Show which endpoints use these schemas for usage context.
        if ac.endpoints:
            usage_lines: list[str] = []
            for ep in ac.endpoints:
                ep_schemas = _json.dumps(ep.request_schema or {}) + _json.dumps(ep.response_schema or {})
                for sname in relevant_schemas:
                    if sname.lower() in ep_schemas.lower():
                        usage_lines.append(
                            f"  {ep.method} {ep.path} uses {sname}"
                        )
                        break
            if usage_lines:
                lines.append("Endpoints referencing these schemas:")
                lines.extend(usage_lines)
                lines.append("")

        return "\n".join(lines) + "\n"

    @staticmethod
    def _extract_dep_signatures(context: AgentContext) -> str:
        """Pull method signatures AND type definitions from dependency AST stubs.

        Instead of relying on the LLM to find and read signatures buried deep
        in Related Files, this extracts method/function lines AND class/type
        definitions and places them prominently in the prompt.  This prevents
        type-mismatch errors (e.g. User vs String) because the LLM can see
        both the method return types AND the type structures.
        """
        if not context.related_files or not context.file_blueprint:
            return ""

        dep_paths = set(context.file_blueprint.depends_on)
        if not dep_paths:
            return ""

        import re as _re

        # Patterns that look like method/function signatures across languages
        _SIG_PATTERNS = _re.compile(
            r"^\s*(?:"
            # Java/Kotlin/C#: public ReturnType methodName(...)
            r"(?:public|protected|private|static|abstract|override|async|suspend|fun)\s+.*\(.*\)"
            r"|"
            # TypeScript/JS: export function/const name(...) / async function name(...)
            r"(?:export\s+)?(?:async\s+)?(?:function|const|let)\s+\w+.*[\(=]"
            r"|"
            # Go: func (r *Receiver) MethodName(...)  or  func FuncName(...)
            r"func\s+(?:\([^)]*\)\s+)?\w+\("
            r"|"
            # Python: def method_name(self, ...)
            r"def\s+\w+\("
            r"|"
            # Rust: pub fn method_name(...)
            r"pub(?:\s+async)?\s+fn\s+\w+\("
            r"|"
            # Interface/type members: methodName(params): ReturnType
            r"\w+\(.*\)\s*:\s*\S+"
            r")"
        )

        # Patterns for type/class/record/interface/enum definitions and fields
        _TYPE_DEF_PATTERNS = _re.compile(
            r"^\s*(?:"
            # Java/C#: public class/interface/enum/record Name ...
            r"(?:public|private|protected)?\s*(?:static\s+)?(?:abstract\s+)?(?:class|interface|enum|record)\s+\w+"
            r"|"
            # TypeScript: export interface/type/class/enum Name
            r"(?:export\s+)?(?:interface|type|class|enum)\s+\w+"
            r"|"
            # Go: type Name struct/interface
            r"type\s+\w+\s+(?:struct|interface)"
            r"|"
            # Python: class Name / @dataclass
            r"class\s+\w+"
            r"|"
            # Rust: pub struct/enum/trait Name
            r"(?:pub\s+)?(?:struct|enum|trait)\s+\w+"
            r")"
        )

        # Field patterns — capture type info for model/DTO/entity classes
        _FIELD_PATTERNS = _re.compile(
            r"^\s*(?:"
            # Java: private String name; / private Long id;
            r"(?:private|protected|public)?\s*(?:static\s+)?(?:final\s+)?\w[\w<>,\s?]*\s+\w+\s*[;=]"
            r"|"
            # TypeScript: name: string; / readonly id: number;
            r"(?:readonly\s+)?\w+\??\s*:\s*\S+"
            r"|"
            # Go struct field: Name string `json:\"name\"`
            r"\w+\s+\S+\s*`"
            r")"
        )

        sections: list[str] = []
        for dep_path in dep_paths:
            content = context.related_files.get(dep_path, "")
            if not content:
                continue
            sigs: list[str] = []
            types: list[str] = []
            for line in content.splitlines():
                stripped = line.strip()
                if not stripped or stripped in ("{", "}", "};"):
                    continue
                if stripped.startswith("//") or stripped.startswith("#"):
                    continue
                if _TYPE_DEF_PATTERNS.match(stripped):
                    typedef = stripped.rstrip("{").rstrip().strip()
                    types.append(f"  {typedef}")
                elif _SIG_PATTERNS.match(stripped):
                    sig = stripped.rstrip("{").rstrip().rstrip(";").strip()
                    sigs.append(f"    {sig}")
                elif _FIELD_PATTERNS.match(stripped):
                    field = stripped.rstrip(";").strip()
                    types.append(f"    {field}")

            parts: list[str] = []
            if types or sigs:
                parts.append(f"  {dep_path}:")
            if types:
                parts.extend(types)
            if sigs:
                if types:
                    parts.append("    Methods:")
                parts.extend(sigs)

            if parts:
                sections.append("\n".join(parts))

        if not sections:
            return ""

        return (
            "DEPENDENCY SIGNATURES (use these EXACT names, types, and return types):\n"
            + "\n".join(sections)
            + "\n\n"
        )

    def _parse_agentic_result(
        self,
        context: AgentContext,
        final_text: str,
        files_written: list[str],
    ) -> TaskResult:
        fb = context.file_blueprint
        if fb and fb.path not in files_written:
            logger.warning(
                "CoderAgent agentic loop finished but %s was not written via write_file tool; "
                "checking %d written files: %s",
                fb.path, len(files_written), files_written,
            )
            # A generation task that produced no file is a hard failure — there
            # is nothing to review, build, or test.  Return failure so the
            # lifecycle engine handles it rather than passing a ghost file into
            # downstream phases.
            return TaskResult(
                success=False,
                output=f"CoderAgent did not write {fb.path}",
                errors=[f"{fb.path} was not written by the write_file tool"],
                metrics=self.get_metrics(),
            )
        out = (
            f"Generated {fb.path} via agentic loop ({len(files_written)} file(s) written)"
            if fb else "Agentic generation completed"
        )
        return TaskResult(
            success=True,
            output=out,
            files_modified=files_written,
            metrics=self.get_metrics(),
        )

    async def execute(self, context: AgentContext) -> TaskResult:
        """Generate, fix, or modify code for the assigned file."""
        if context.task.task_type == TaskType.FIX_CODE:
            return await self._fix_code(context)

        if context.task.task_type == TaskType.MODIFY_FILE:
            return await self._modify_file(context)

        if not context.file_blueprint:
            return TaskResult(
                success=False,
                errors=[f"No blueprint found for {context.task.file}"],
            )

        fb = context.file_blueprint
        fmt = _config_format(fb.path)

        if fmt:
            return await self._generate_config(context, fmt)

        # If extension-based detection missed it but the layer says "config",
        # treat it as a generic config file so we don't generate source code
        # for non-source files (e.g., custom .conf, .cfg, resource files).
        if not fmt and fb.layer == "config":
            suffix = Path(fb.path).suffix.lower()
            fallback_fmt = f"{suffix.lstrip('.').upper() or 'plain text'} configuration file"
            logger.info(
                "File %s has layer=config but unrecognized extension '%s' — "
                "treating as config file (%s)",
                fb.path, suffix, fallback_fmt,
            )
            return await self._generate_config(context, fallback_fmt)

        # Source files: use the agentic tool-use loop so the agent can
        # read dependency interfaces and write the file directly.
        lang = fb.language or context.blueprint.tech_stack.get("language", "python")
        self._current_language = lang  # used by system_prompt property during the loop
        return await self.execute_agentic(context)

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
            db_hint = ""
            if tech.get("db", "").lower() == "h2":
                db_hint = (
                    "\n- Include the H2 in-memory database dependency with runtime scope:\n"
                    "  <dependency><groupId>com.h2database</groupId>"
                    "<artifactId>h2</artifactId><scope>runtime</scope></dependency>"
                )
            pom_hint = (
                "\nIMPORTANT Java 17 requirements for pom.xml:\n"
                "- Use Spring Boot parent 3.x (e.g. 3.2.x) which targets Java 17 by default\n"
                "- Set <java.version>17</java.version> in <properties>\n"
                "- Set <maven.compiler.source>17</maven.compiler.source> and "
                "<maven.compiler.target>17</maven.compiler.target> in <properties>\n"
                "- In the maven-compiler-plugin configuration use <release>17</release> "
                "(NOT <source>/<target> inside the plugin — use the properties instead)\n"
                "- Use <maven.compiler.release>17</maven.compiler.release> as the canonical property\n"
                "- Include all required Spring Boot starter dependencies (web, data-jpa, validation, test)"
                + db_hint + "\n"
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
        build_errors: str = context.task.metadata.get("build_errors", "")
        test_errors: str = context.task.metadata.get("test_errors", "")
        security_errors: str = context.task.metadata.get("security_vulnerabilities", "")
        fix_trigger: str = context.task.metadata.get("fix_trigger", "review")

        # Read current file content from the repo (non-blocking)
        current_content = await self.repo.async_read_file(file_path) or ""

        # If review passed (no errors) — nothing to fix, skip LLM call
        if not review_errors and "PASSED" in review_output and fix_trigger == "review":
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

        # ── Cap error text to avoid token explosion ──────────────────────────
        # Build output (Maven, javac, tsc) can be 50K+ chars.  Use
        # extract_error_lines() to pull only error-relevant lines instead of
        # blind tail truncation.  Tail truncation loses the root-cause error
        # (first file that broke) in favour of downstream cascading failures,
        # causing the fix agent to treat symptoms rather than the cause.
        if fix_trigger in ("build", "build_unattributed") and build_errors:
            filtered = extract_error_lines(build_errors, max_chars=_MAX_ERROR_CHARS)
            issues_text = (
                "Compilation/build errors to fix:\n" + filtered + "\n\n"
                "If the error is 'cannot find symbol' for a method call on a dependency, "
                "check the Related Files section above for the dependency's actual method "
                "signatures and use the EXACT method name shown there."
            )
        elif fix_trigger == "security" and security_errors:
            # Security vulnerabilities — already formatted with severity,
            # type, description, and remediation guidance.
            capped = security_errors[:_MAX_ERROR_CHARS]
            issues_text = (
                "Security vulnerabilities to fix:\n" + capped + "\n\n"
                "IMPORTANT: Apply the remediation for each vulnerability. "
                "Do NOT remove or break existing functionality — only fix "
                "the security issue (e.g., parameterize queries, sanitize "
                "inputs, remove hardcoded secrets, add auth checks)."
            )
        elif fix_trigger == "security_rebuild" and build_errors:
            filtered = extract_error_lines(build_errors, max_chars=_MAX_ERROR_CHARS)
            issues_text = (
                "Security fixes broke the build. Fix compilation errors:\n"
                + filtered
            )
        elif fix_trigger in ("test", "integration_test") and test_errors:
            filtered = extract_error_lines(test_errors, max_chars=_MAX_ERROR_CHARS)
            issues_text = "Test failures to fix:\n" + filtered
        elif fix_trigger == "integration_test_code" and test_errors:
            # Fixing the test code itself, not the source
            filtered = extract_error_lines(test_errors, max_chars=_MAX_ERROR_CHARS)
            issues_text = (
                "Integration test code has errors — fix the TEST code:\n"
                + filtered + "\n\n"
                "IMPORTANT: Fix the test imports, assertions, endpoint paths, "
                "or test setup — do NOT change the source code being tested."
            )
        elif fix_trigger == "integration_rebuild" and build_errors:
            filtered = extract_error_lines(build_errors, max_chars=_MAX_ERROR_CHARS)
            issues_text = (
                "Source fix for integration tests broke the build. "
                "Fix compilation errors:\n" + filtered
            )
        elif review_errors:
            issues_text = "\n".join(f"- {e}" for e in review_errors)
        else:
            raw = review_output
            issues_text = extract_error_lines(raw, max_chars=_MAX_ERROR_CHARS) if len(raw) > _MAX_ERROR_CHARS else raw

        # For compiled languages add an explicit post-fix syntax checklist so the
        # model validates its own output before returning.
        syntax_checklist = self._LANGUAGE_SYNTAX_RULES.get(profile.name, "")
        if syntax_checklist:
            syntax_note = (
                f"\n\nAfter writing the fixed code, verify:\n"
                f"{syntax_checklist.strip()}"
            )
        else:
            syntax_note = ""

        # ── Slim context: fix tasks don't need the full dependency tree ───────
        fix_context_header = self._format_fix_context(context)
        dep_sigs = self._extract_dep_signatures(context)

        line_count = len(current_content.splitlines())

        # ── Large files: use diff output to avoid token-doubling ──────────────
        if line_count > _LARGE_FILE_THRESHOLD:
            result = await self._try_diff_fix(
                context, file_path, current_content, lang, profile,
                fix_context_header, issues_text, fix_trigger,
            )
            if result is not None:
                return result
            logger.warning(
                "Diff fix failed for %s (%d lines) — falling back to full rewrite",
                file_path, line_count,
            )

        # ── Small files or diff fallback: full-file rewrite ──────────────────
        prompt = (
            f"{fix_context_header}\n\n"
            f"{dep_sigs}"
            f"FIX TASK for: {file_path}\n\n"
            f"Current content:\n```{profile.code_fence_name}\n{current_content}\n```\n\n"
            f"Issues to fix ({fix_trigger}):\n{issues_text}\n\n"
            "INSTRUCTIONS:\n"
            "1. Fix ONLY the specific issues listed above\n"
            "2. Do NOT remove, rename, or reorganize any existing methods or classes\n"
            "3. Do NOT change method signatures unless the issue specifically requires it\n"
            "4. Do NOT add new functionality beyond what is needed to fix the issues\n"
            "5. Preserve all existing imports, fields, and logic that are not related to the fix\n"
            "6. Output the COMPLETE corrected file — every line, from first import to last closing brace\n"
            "7. Output ONLY the code — no markdown fences, no explanations\n\n"
            "TYPE MISMATCH FIX GUIDE (for 'incompatible types' / 'Expected N arguments' errors):\n"
            "- Check DEPENDENCY SIGNATURES above to see the exact return type of the method\n"
            "- If a method returns a model object (e.g. User), do NOT pass it where String is expected\n"
            "- To extract a String from a model, call the appropriate getter (e.g. user.getUsername())\n"
            "- If the error says 'X cannot be converted to Y', find the variable assignment and\n"
            "  ensure the right-hand side type matches the left-hand side\n"
            "- If the error says 'Expected N arguments, but got M', check the DEPENDENCY SIGNATURES\n"
            "  for the exact parameter count of that function and fix the call to match\n"
            "- For store functions like login(token), do NOT call login(user, token) — match exactly"
            f"{syntax_note}"
        )

        fixed_code = await self._call_llm(prompt, system_override=self._get_source_system_prompt(lang))
        fixed_code = self._clean_fences(fixed_code, profile.code_fence_name)

        # ── Validate the rewrite ─────────────────────────────────────────────
        validation_result = _validate_rewrite(fixed_code, current_content, profile, file_path, "Fix")
        if validation_result is not None:
            return validation_result

        await self.repo.async_write_file(file_path, fixed_code)

        return TaskResult(
            success=True,
            output=f"Fixed {file_path} ({len(review_errors)} issue(s) addressed)",
            files_modified=[file_path],
            metrics=self.get_metrics(),
        )

    def _format_fix_context(self, context: AgentContext) -> str:
        """Minimal context for fix tasks — architecture + blueprint + dependencies.

        Dependency files (AST stubs) are included for ALL fix triggers because:
        - Build fixes often involve wrong method names on dependencies
          (e.g., calling register() when the service has registerUser())
        - Review fixes may flag incorrect API usage
        - Security/integration fixes need DTO field names and interfaces
        """
        parts: list[str] = []
        if context.architecture_summary:
            parts.append(f"## Architecture\n{context.architecture_summary[:1000]}")
        if context.file_blueprint:
            fb = context.file_blueprint
            parts.append(
                f"## File Blueprint\n"
                f"Path: {fb.path}\n"
                f"Purpose: {fb.purpose}\n"
                f"Layer: {fb.layer}\n"
                f"Depends on: {', '.join(fb.depends_on) or 'none'}\n"
                f"Exports: {', '.join(fb.exports) or 'TBD'}"
            )

        # Include dependency files so the fix agent can see actual method
        # signatures when fixing "cannot find symbol" or wrong-name errors.
        # Build-triggered fixes get a larger budget because type-mismatch errors
        # (e.g. User vs String) require seeing full type definitions.
        fix_trigger = context.task.metadata.get("fix_trigger", "") if context.task else ""
        is_build_fix = fix_trigger in ("build", "build_unattributed")
        if context.related_files:
            dep_sections: list[str] = []
            total_chars = 0
            max_dep_chars = 8000 if is_build_fix else 4000
            per_dep_chars = 3000 if is_build_fix else 1500

            for dep_path, dep_content in context.related_files.items():
                # Skip the target file itself — it's already in the prompt
                if dep_path == context.task.file:
                    continue
                if not dep_content:
                    continue
                # Truncate each dependency to keep within budget
                chunk = dep_content[:per_dep_chars]
                dep_sections.append(f"--- {dep_path} ---\n{chunk}")
                total_chars += len(chunk)
                if total_chars >= max_dep_chars:
                    dep_sections.append("... (remaining dependencies omitted)")
                    break

            if dep_sections:
                parts.append(
                    "## Related Files (for reference — do NOT modify these)\n"
                    + "\n\n".join(dep_sections)
                )

        return "\n\n".join(parts)

    async def _try_diff_fix(
        self,
        context: AgentContext,
        file_path: str,
        current_content: str,
        lang: str,
        profile: LanguageProfile,
        fix_context_header: str,
        issues_text: str,
        fix_trigger: str,
    ) -> TaskResult | None:
        """Ask the LLM for a unified diff that fixes the issues.

        Returns a TaskResult on success, or None if the diff cannot be applied
        (caller falls back to full-file rewrite).
        """
        numbered = "\n".join(
            f"{i + 1:4d}  {line}"
            for i, line in enumerate(current_content.splitlines())
        )
        prompt = (
            f"{fix_context_header}\n\n"
            f"File to fix: {file_path}\n"
            f"Language: {profile.display_name}\n\n"
            f"Current file content (line numbers for reference):\n"
            f"```\n{numbered}\n```\n\n"
            f"Issues to fix ({fix_trigger}):\n{issues_text}\n\n"
            "Generate a unified diff patch that resolves ALL the issues above.\n"
            "Output ONLY the raw diff — no markdown fences, no explanation.\n"
            f"Format:\n--- a/{file_path}\n+++ b/{file_path}\n@@ -LINE,COUNT +LINE,COUNT @@\n..."
        )
        system = (
            "You are a surgical code fix agent. "
            "Output ONLY a standard unified diff — never rewrite entire files.\n"
            "Include 3 lines of context around each changed hunk.\n"
            "Line numbers in @@ headers MUST match the actual file content exactly."
        )
        patch_text = await self._call_llm(prompt, system_override=system)

        valid, err = self._file_tools.validate_patch(file_path, patch_text)
        if not valid:
            logger.debug("Diff fix validation failed for %s: %s", file_path, err)
            return None

        applied = self._file_tools.apply_patch(file_path, patch_text)
        if not applied:
            logger.debug("Diff fix application failed for %s", file_path)
            return None

        return TaskResult(
            success=True,
            output=f"Fixed {file_path} via diff patch ({fix_trigger})",
            files_modified=[file_path],
            metrics=self.get_metrics(),
        )

    # ── Modification workflow ──────────────────────────────────────────

    def _get_modify_system_prompt(self, language: str) -> str:
        """System prompt for targeted file modification (patch editing)."""
        profile = get_language_profile(language)
        return (
            f"You are an expert {profile.display_name} developer agent specializing in "
            f"modifying existing code. You make targeted, surgical edits to existing files.\n\n"
            "CRITICAL REQUIREMENTS:\n"
            "1. Output the COMPLETE modified file — every line from start to end\n"
            "2. Preserve ALL existing code that is NOT part of the requested change\n"
            "3. Preserve existing imports, formatting, and style conventions exactly\n"
            "4. Add new imports ONLY if the modification requires them\n"
            "5. Do NOT remove, rename, or reorganize any existing code\n"
            "6. Do NOT add placeholder or TODO comments — write complete implementations\n"
            "7. Output ONLY the code — no markdown fences, no explanations\n\n"
            "PLACEMENT RULES:\n"
            "- New methods go inside their target class, after related existing methods\n"
            "- New functions go after related existing functions\n"
            "- New imports go with the existing import block at the top\n"
            f"- Follow the existing code's style and {profile.display_name} conventions exactly"
        )

    async def modify_file(self, context: AgentContext) -> TaskResult:
        """Public entry-point for targeted file modification.

        Alias kept so external callers (PatchAgent fallback) do not depend on
        a private method name.  Delegates to ``_modify_file``.
        """
        return await self._modify_file(context)

    async def _modify_file(self, context: AgentContext) -> TaskResult:
        """Modify an existing file based on a change action.

        For files above ``_LARGE_FILE_THRESHOLD`` lines, attempts a unified
        diff patch first (cheaper and more reliable for large files).  Falls
        back to full-file rewrite only when the diff fails or for small files.
        """
        file_path = context.task.file
        change_desc: str = context.task.metadata.get("change_description", "")
        change_type: str = context.task.metadata.get("change_type", "")
        target_function: str = context.task.metadata.get("target_function", "")
        target_class: str = context.task.metadata.get("target_class", "")

        # Read the current file content
        current_content = await self.repo.async_read_file(file_path)
        if current_content is None:
            if context.file_blueprint:
                return await self._generate_source(context)
            return TaskResult(
                success=False,
                errors=[f"File not found: {file_path} and no blueprint to generate from"],
            )

        fb = context.file_blueprint
        lang = (fb.language if fb else None) or context.blueprint.tech_stack.get("language", "python")
        profile = get_language_profile(lang)

        target_hint = ""
        if target_class:
            target_hint += f"Target class: {target_class}\n"
        if target_function:
            target_hint += f"Target function/method: {target_function}\n"

        line_count = len(current_content.splitlines())
        logger.info("Modifying %s (%d lines): %s", file_path, line_count, change_desc[:80])

        # Large files: try a surgical diff first to save tokens.
        if line_count > _LARGE_FILE_THRESHOLD:
            result = await self._try_diff_modification(
                context, file_path, current_content, lang, profile,
                change_desc, change_type, target_hint,
            )
            if result is not None:
                return result
            logger.warning(
                "Diff modification failed for %s (%d lines) — falling back to full rewrite",
                file_path, line_count,
            )

        # Full-file rewrite (small files, or large-file diff fallback)
        prompt = (
            f"{self._format_context(context)}\n\n"
            f"MODIFICATION REQUEST for: {file_path}\n"
            f"Change type: {change_type}\n"
            f"{target_hint}"
            f"Description: {change_desc}\n\n"
            f"Current file content:\n```{profile.code_fence_name}\n{current_content}\n```\n\n"
            f"Apply the modification. Output the COMPLETE modified file "
            f"preserving all existing code. Output only the code, nothing else."
        )
        modified_code = await self._call_llm(
            prompt, system_override=self._get_modify_system_prompt(lang),
        )
        modified_code = self._clean_fences(modified_code, profile.code_fence_name)

        # ── Validate the rewrite ─────────────────────────────────────────────
        validation_result = _validate_rewrite(modified_code, current_content, profile, file_path, "Modification")
        if validation_result is not None:
            return validation_result

        await self.repo.async_write_file(file_path, modified_code)

        return TaskResult(
            success=True,
            output=f"Modified {file_path}: {change_desc[:80]} (full rewrite, {len(modified_code)} bytes)",
            files_modified=[file_path],
            metrics=self.get_metrics(),
        )

    async def _try_diff_modification(
        self,
        context: AgentContext,
        file_path: str,
        current_content: str,
        lang: str,
        profile: LanguageProfile,
        change_desc: str,
        change_type: str,
        target_hint: str,
    ) -> TaskResult | None:
        """Ask the LLM for a unified diff patch and apply it.

        Returns a ``TaskResult`` on success, or ``None`` if the diff could not
        be validated or applied (so the caller can fall back to full rewrite).
        """
        # Show line numbers so the LLM can emit correct @@ offsets
        numbered = "\n".join(
            f"{i + 1:4d}  {line}"
            for i, line in enumerate(current_content.splitlines())
        )
        prompt = (
            f"File to modify: {file_path}\n"
            f"Language: {profile.display_name}\n"
            f"Change type: {change_type}\n"
            f"{target_hint}"
            f"Change description: {change_desc}\n\n"
            f"Current file content (line numbers for reference):\n"
            f"```\n{numbered}\n```\n\n"
            "Generate a unified diff patch. "
            "Output ONLY the raw diff — no markdown fences, no explanation.\n"
            f"Format:\n--- a/{file_path}\n+++ b/{file_path}\n@@ -LINE,COUNT +LINE,COUNT @@\n..."
        )
        system = (
            "You are a surgical code modification agent. "
            "Output ONLY a standard unified diff — never rewrite entire files.\n"
            "Include 3 lines of context around each change hunk.\n"
            "Line numbers in @@ headers MUST match the actual file content exactly."
        )

        patch_text = await self._call_llm(prompt, system_override=system)

        valid, err = self._file_tools.validate_patch(file_path, patch_text)
        if not valid:
            logger.debug("Diff validation failed for %s: %s", file_path, err)
            return None

        applied = self._file_tools.apply_patch(file_path, patch_text)
        if not applied:
            logger.debug("Diff application failed for %s", file_path)
            return None

        return TaskResult(
            success=True,
            output=f"Patched {file_path}: {change_desc[:80]} (diff on {len(current_content.splitlines())} lines)",
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
        content = content.strip() + "\n"
        # Remove duplicated content blocks caused by LLM continuation failures
        content = self._deduplicate_content(content)
        return content
