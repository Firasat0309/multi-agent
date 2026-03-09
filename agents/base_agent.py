"""Base agent class with common LLM interaction and tool usage patterns."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from core.language import get_language_profile
from core.llm_client import LLMClient, LLMResponse
from core.models import AgentContext, AgentRole, Task, TaskResult
from core.repository_manager import RepositoryManager

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class for all agents in the system."""

    role: AgentRole

    def __init__(
        self,
        llm_client: LLMClient,
        repo_manager: RepositoryManager,
    ) -> None:
        self.llm = llm_client
        self.repo = repo_manager
        self._metrics: dict[str, Any] = {"llm_calls": 0, "tokens_used": 0}

    @abstractmethod
    async def execute(self, context: AgentContext) -> TaskResult:
        """Execute the agent's task with the given context."""
        ...

    @property
    def system_prompt(self) -> str:
        """Base system prompt; subclasses should override or extend."""
        return (
            f"You are a {self.role.value} agent in an automated code generation system.\n"
            "You produce high-quality, production-ready code.\n"
            "Follow the architecture and blueprint strictly.\n"
            "Never invent files or dependencies outside the blueprint.\n"
            "Output only the requested content with no extra commentary."
        )

    # Maximum number of continuation requests when the LLM hits max_tokens.
    # Prevents infinite loops if the model keeps producing at-limit output.
    _MAX_CONTINUATIONS = 4

    async def _call_llm(
        self,
        user_prompt: str,
        system_override: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """Call LLM and track metrics. Auto-continues in a loop if output is truncated."""
        response = await self.llm.generate(
            system_prompt=system_override or self.system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
        )
        self._metrics["llm_calls"] += 1
        self._metrics["tokens_used"] += sum(response.usage.values())
        content = response.content

        # Loop continuations until the model stops naturally or we hit the cap.
        # Previously this was single-shot — large files (e.g. 500+ line Java
        # classes) could be silently truncated after one continuation.
        continuation_count = 0
        while (
            response.stop_reason in ("max_tokens", "length", "MAX_TOKENS")
            and continuation_count < self._MAX_CONTINUATIONS
        ):
            continuation_count += 1
            logger.warning(
                f"LLM output truncated (stop_reason={response.stop_reason!r}), "
                f"requesting continuation {continuation_count}/{self._MAX_CONTINUATIONS}"
            )
            continuation_prompt = (
                f"{user_prompt}\n\n"
                f"IMPORTANT: Your previous response was cut off mid-way. "
                f"Here is the end of what you wrote:\n"
                f"```\n{content[-300:]}\n```\n"
                f"Continue EXACTLY from where the output was cut off. "
                f"Output only the continuation — no preamble, no repetition."
            )
            response = await self.llm.generate(
                system_prompt=system_override or self.system_prompt,
                user_prompt=continuation_prompt,
                temperature=temperature,
            )
            self._metrics["llm_calls"] += 1
            self._metrics["tokens_used"] += sum(response.usage.values())
            continuation = response.content

            # Clean up the join point: strip markdown fences the continuation
            # may have added at its start (```java\n...) and find the overlap
            # region to avoid duplicated lines at the stitch boundary.
            continuation = continuation.strip()
            if continuation.startswith("```"):
                first_nl = continuation.find("\n")
                if first_nl != -1:
                    continuation = continuation[first_nl + 1:]
            if continuation.endswith("```"):
                continuation = continuation[:-3].rstrip()

            # Detect overlapping tail: the LLM sometimes repeats the last
            # 1-3 lines for context.  Find the longest suffix of `content`
            # that matches a prefix of `continuation` and skip it.
            overlap = 0
            tail_lines = content.rsplit("\n", 5)[-5:]  # last 5 lines
            for n in range(min(len(tail_lines), 5), 0, -1):
                tail = "\n".join(tail_lines[-n:])
                if continuation.startswith(tail):
                    overlap = len(tail)
                    break
            if overlap:
                continuation = continuation[overlap:]
                logger.debug(f"Stripped {overlap}-char overlap at continuation boundary")

            content = content + continuation

        if continuation_count >= self._MAX_CONTINUATIONS:
            logger.warning(
                f"Hit max continuation limit ({self._MAX_CONTINUATIONS}). "
                f"Output may still be incomplete ({len(content)} chars total)."
            )

        return content

    async def _call_llm_json(
        self,
        user_prompt: str,
        system_override: str | None = None,
    ) -> dict[str, Any]:
        """Call LLM expecting JSON output."""
        result = await self.llm.generate_json(
            system_prompt=system_override or self.system_prompt,
            user_prompt=user_prompt,
        )
        self._metrics["llm_calls"] += 1
        return result

    def _format_context(self, context: AgentContext) -> str:
        """Format agent context into a prompt section."""
        parts: list[str] = []

        if context.architecture_summary:
            parts.append(f"## Architecture\n{context.architecture_summary[:2000]}")

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

        if context.related_files:
            parts.append("## Related Files")
            for path, content in context.related_files.items():
                # Truncate large files
                truncated = content[:4000] if len(content) > 4000 else content
                # Detect language from file extension for correct code fencing
                lang_name = ""
                if context.file_blueprint:
                    lang_name = get_language_profile(context.file_blueprint.language).code_fence_name
                if not lang_name:
                    for ext, fence in {".py": "python", ".java": "java", ".go": "go",
                                       ".ts": "typescript", ".rs": "rust", ".cs": "csharp"}.items():
                        if path.endswith(ext):
                            lang_name = fence
                            break
                parts.append(f"### {path}\n```{lang_name}\n{truncated}\n```")

        if context.dependency_info:
            deps = context.dependency_info
            if deps.get("upstream"):
                parts.append(f"## Upstream dependencies: {', '.join(deps['upstream'])}")
            if deps.get("downstream"):
                parts.append(f"## Downstream dependents: {', '.join(deps['downstream'])}")

        return "\n\n".join(parts)

    def get_metrics(self) -> dict[str, Any]:
        return dict(self._metrics)
