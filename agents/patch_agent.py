"""Patch agent — generates targeted unified-diff patches for file modification.

Unlike CoderAgent which rewrites entire files, PatchAgent generates minimal
± line diffs targeting only the requested change location.  The patch is
validated and applied via FileTools.apply_patch().

On failure the agent falls back to CoderAgent full-file rewrite mode so
the pipeline never silently produces an empty or corrupted result.
"""

from __future__ import annotations

import logging

from agents.base_agent import BaseAgent
from core.language import get_language_profile
from core.models import AgentContext, AgentRole, TaskResult, TaskType
from tools.file_tools import FileTools

logger = logging.getLogger(__name__)


class PatchAgent(BaseAgent):
    """Generates targeted unified-diff patches for MODIFY_FILE tasks.

    Workflow:
    1. Read the current target file.
    2. Send file + change request to the LLM — ask for a unified diff only.
    3. Validate the diff applies cleanly to the current file content.
    4. Apply via FileTools.apply_patch().
    5. On any failure → fall back to CoderAgent full-file rewrite.
    """

    role = AgentRole.PATCH_AGENT

    @property
    def system_prompt(self) -> str:
        return (
            "You are a surgical code modification agent. "
            "You generate ONLY unified diff patches — never rewrite entire files.\n\n"
            "Output format — standard unified diff:\n"
            "--- a/path/to/file.py\n"
            "+++ b/path/to/file.py\n"
            "@@ -LINE,COUNT +LINE,COUNT @@\n"
            " context line\n"
            "-removed line\n"
            "+added line\n"
            " context line\n\n"
            "Rules:\n"
            "- Include 3 lines of context before and after each change\n"
            "- Change ONLY what is specified — preserve all other code exactly\n"
            "- Output the raw diff only, no explanation, no markdown fences\n"
            "- Line numbers in @@ headers MUST match the actual file content\n"
            "- Do NOT add, remove, or rearrange any lines outside the diff hunks"
        )

    async def execute(self, context: AgentContext) -> TaskResult:
        """Generate a patch and apply it; fall back to full-file rewrite on failure."""
        file_path = context.task.file

        # Read current file content
        current_content = await self.repo.async_read_file(file_path)
        if current_content is None:
            # File does not yet exist — delegate entirely to CoderAgent
            logger.info("PatchAgent: %s not found, delegating to CoderAgent", file_path)
            return await self._fallback_full_rewrite(context)

        # Generate the patch via LLM
        patch_text = await self._generate_patch(context, current_content)

        # Validate and apply
        file_tools = FileTools(self.repo.workspace)
        valid, error_msg = file_tools.validate_patch(file_path, patch_text)
        if not valid:
            logger.warning(
                "PatchAgent: patch validation failed for %s (%s) — falling back to full-file rewrite",
                file_path, error_msg,
            )
            return await self._fallback_full_rewrite(context)

        applied = file_tools.apply_patch(file_path, patch_text)
        if not applied:
            logger.warning(
                "PatchAgent: patch application failed for %s — falling back to full-file rewrite",
                file_path,
            )
            return await self._fallback_full_rewrite(context)

        return TaskResult(
            success=True,
            output=f"Patched {file_path}: {context.task.description[:80]}",
            files_modified=[file_path],
            metrics=self.get_metrics(),
        )

    # ── Private helpers ──────────────────────────────────────────────────────

    async def _generate_patch(self, context: AgentContext, current_content: str) -> str:
        """Ask the LLM to generate a unified diff for the requested change."""
        file_path = context.task.file
        change_desc: str = context.task.metadata.get("change_description", context.task.description)
        change_type: str = context.task.metadata.get("change_type", "")
        target_function: str = context.task.metadata.get("target_function", "")
        target_class: str = context.task.metadata.get("target_class", "")

        fb = context.file_blueprint
        lang = (fb.language if fb else None) or context.blueprint.tech_stack.get("language", "python")
        profile = get_language_profile(lang)

        target_hint = ""
        if target_class:
            target_hint += f"Target class: {target_class}\n"
        if target_function:
            target_hint += f"Target function/method: {target_function}\n"

        prompt = (
            f"File to modify: {file_path}\n"
            f"Language: {profile.display_name}\n"
            f"Change type: {change_type}\n"
            f"{target_hint}"
            f"Change description: {change_desc}\n\n"
            f"Current file content (line numbers shown for reference ONLY — "
            f"do NOT include line numbers in your diff output):\n"
        )

        # Add line numbers to help the LLM emit correct @@ headers
        numbered_lines = "\n".join(
            f"{i + 1:4d}  {line}"
            for i, line in enumerate(current_content.splitlines())
        )
        prompt += f"```\n{numbered_lines}\n```\n\n"
        prompt += (
            "Generate a unified diff patch.\n\n"
            "FORMAT REQUIREMENTS:\n"
            f"--- a/{file_path}\n"
            f"+++ b/{file_path}\n"
            "@@ -<start>,<count> +<start>,<count> @@\n"
            " context line (3 lines before change)\n"
            "-removed line\n"
            "+added line\n"
            " context line (3 lines after change)\n\n"
            "RULES:\n"
            "- Output ONLY the raw diff — no markdown fences, no explanation\n"
            "- Line numbers in @@ headers reference the ORIGINAL file lines "
            "(not the numbered display above)\n"
            "- Include exactly 3 context lines before and after each hunk\n"
            "- The count in @@ header = context lines + changed lines in that hunk\n"
            "- Change ONLY what the description asks for — nothing else"
        )

        return await self._call_llm(prompt)

    async def _fallback_full_rewrite(self, context: AgentContext) -> TaskResult:
        """Fall back to CoderAgent full-file rewrite when patching fails."""
        from agents.coder_agent import CoderAgent
        coder = CoderAgent(llm_client=self.llm, repo_manager=self.repo)
        return await coder.modify_file(context)
