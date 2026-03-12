"""Coder agent - generates code for a specific file following the blueprint."""

from __future__ import annotations

import difflib
import logging
from pathlib import Path
from typing import Any

from agents.base_agent import BaseAgent
from core.agent_tools import CODER_TOOLS, ToolDefinition
from core.language import get_language_profile, LanguageProfile
from core.models import AgentContext, AgentRole, ChangeAction, ChangeActionType, TaskResult, TaskType

# Files with more lines than this threshold are modified via unified diff instead
# of a full-file rewrite.  Diffs are ~5–20× cheaper in tokens for large files.
_LARGE_FILE_THRESHOLD = 200

# Max chars of build/compiler output to include in fix prompts.
# Maven can dump 50K+ chars; the relevant error lines are always near the end.
_MAX_ERROR_CHARS = 3_000
# Max chars of a single related file included in fix context.
_MAX_RELATED_FILE_CHARS = 2_000

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
            f"{lang_rules}"
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

        return (
            f"{self._format_context(context)}\n\n"
            f"Generate the complete {profile.display_name} file for: {fb.path}\n"
            f"Purpose: {fb.purpose}\n"
            f"Layer: {fb.layer}\n"
            f"Must export: {', '.join(fb.exports) if fb.exports else 'appropriate classes/functions'}\n\n"
            f"Use the available tools to:\n"
            f"1. Read any dependency files you need to understand their interfaces\n"
            f"2. Search for usage patterns or symbol definitions if needed\n"
            f"3. Write the complete, working {profile.display_name} code via write_file\n\n"
            f"Output ONLY the code to write_file — no markdown fences, no explanations."
            f"{syntax_reminder}"
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
        # Build output (Maven, javac, tsc) can be 50K+ chars.  Keep the last
        # _MAX_ERROR_CHARS which contain the actual error lines, not the build
        # setup noise at the start.
        if fix_trigger == "build" and build_errors:
            raw = build_errors
            issues_text = (
                f"Compilation/build errors to fix:\n"
                + (raw[-_MAX_ERROR_CHARS:] if len(raw) > _MAX_ERROR_CHARS
                   else raw)
            )
        elif review_errors:
            issues_text = "\n".join(f"- {e}" for e in review_errors)
        else:
            raw = review_output
            issues_text = raw[-_MAX_ERROR_CHARS:] if len(raw) > _MAX_ERROR_CHARS else raw

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
            f"The following file has issues that must be fixed:\n\n"
            f"File: {file_path}\n\n"
            f"Current content:\n```{profile.code_fence_name}\n{current_content}\n```\n\n"
            f"Issues to fix ({fix_trigger}):\n{issues_text}\n\n"
            f"Output the COMPLETE corrected file with ALL issues resolved. "
            f"Output only the code, nothing else."
            f"{syntax_note}"
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

    def _format_fix_context(self, context: AgentContext) -> str:
        """Minimal context for fix tasks — architecture + blueprint only.

        Deliberately excludes related_files: during a fix the agent already has
        the broken file content in the prompt, and dumping dependency source code
        on top of it consumes tokens without improving fix quality.  The model
        can call read_file via tools if it genuinely needs an interface.
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
            "Rules:\n"
            "- You will receive the CURRENT file content and a specific modification request\n"
            "- Output the COMPLETE modified file — not just the patch\n"
            "- Preserve ALL existing code that is not being changed\n"
            "- Preserve existing imports, formatting, and style conventions\n"
            "- Add new imports only if the modification requires them\n"
            "- Place new functions/methods in the appropriate location:\n"
            "  * New methods go inside their target class\n"
            "  * New functions go after related existing functions\n"
            "  * New imports go with existing import blocks\n"
            f"- Follow the existing code's style and {profile.display_name} conventions\n"
            "- Do NOT remove or rewrite code that isn't part of the requested change\n"
            "- Do NOT add placeholder or TODO comments — write complete implementations\n"
            "- Output only the code, no markdown fences or explanations"
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
        return content.strip() + "\n"
