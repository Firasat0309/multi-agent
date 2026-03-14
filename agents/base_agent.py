"""Base agent class with common LLM interaction and tool usage patterns."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

# Files longer than this are served in chunks when no explicit range is given.
# Matches the 100-line window used by most IDE agent loops; we use 150 to
# reduce round-trips for typical class/function sizes.
_READ_CHUNK_LINES = 150

from core.agent_tools import ToolDefinition
from core.language import get_language_profile
from core.llm_client import LLMClient, ToolCall
from core.models import AgentContext, AgentRole, TaskResult
from core.repository_manager import RepositoryManager
from core.mcp_client import MCPClient
from tools.code_search import CodeSearch
from tools.file_tools import FileTools, PatchError

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class for all agents in the system."""

    role: AgentRole

    def __init__(
        self,
        llm_client: LLMClient,
        repo_manager: RepositoryManager,
        mcp_client: MCPClient | None = None,
    ) -> None:
        self.llm = llm_client
        self.repo = repo_manager
        self._mcp_client = mcp_client
        self._mcp_tools: list[ToolDefinition] = []
        self._metrics: dict[str, Any] = {"llm_calls": 0, "tokens_used": 0}
        # Tool helpers are created once per agent instance and reused across all
        # tool dispatch calls — avoiding per-call re-instantiation overhead.
        self._code_search = CodeSearch(repo_manager.workspace)
        self._file_tools = FileTools(repo_manager.workspace)

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

    # ── Agentic tool-use loop ─────────────────────────────────────────────────

    @property
    def tools(self) -> list[ToolDefinition]:
        """Tools available to this agent. Override in subclasses to add tools.
        If an MCP client is attached, its dynamically fetched tools are appended."""
        base_tools = []
        return base_tools + self._mcp_tools

    def _build_prompt(self, context: AgentContext) -> str:
        """Build the initial user message for the agentic loop.

        Subclasses should override this to produce a task-specific prompt.
        """
        return self._format_context(context)

    async def execute_agentic(self, context: AgentContext) -> TaskResult:
        """Tool-use loop: agent may request reads, searches, writes, and MCP extensions.

        The loop continues until the LLM returns ``stop_reason == "end_turn"``
        or ``max_iterations`` is reached.  Each tool call is dispatched via
        ``_dispatch_tool()`` and the result is fed back in the next turn.
        """
        # If an MCP client is present, fetch the tools before execution starts
        if self._mcp_client and not self._mcp_tools:
            try:
                self._mcp_tools = await self._mcp_client.list_tools()
            except Exception as e:
                logger.warning(f"Agent failed to fetch MCP tools before execution: {e}")

        if not self.tools:
            # No tools registered — fall back to single-shot call
            content = await self._call_llm(self._build_prompt(context))
            return self._parse_agentic_result(context, content, [])

        messages: list[dict] = [
            {"role": "user", "content": self._build_prompt(context)}
        ]
        files_written: list[str] = []
        max_iterations = 10
        _target = (context.file_blueprint.path if context.file_blueprint else None) or "task"
        _heartbeat_interval = 15  # seconds between "still waiting" log lines

        async def _llm_call_with_heartbeat(iteration: int) -> "LLMResponseWithTools":
            """Wrap generate_with_tools with a pre-call log and a periodic heartbeat.

            This prevents the screen from going silent for 30–90 seconds during
            a large LLM response.  A background task logs every
            ``_heartbeat_interval`` seconds so the user can see progress.
            """
            import time as _time
            _t0 = _time.monotonic()
            logger.info(
                "[%s] LLM call iter %d/%d for %s — waiting for response…",
                self.__class__.__name__, iteration + 1, max_iterations, _target,
            )

            async def _heartbeat() -> None:
                while True:
                    await asyncio.sleep(_heartbeat_interval)
                    elapsed = _time.monotonic() - _t0
                    logger.info(
                        "[%s]   …still waiting (%.0fs elapsed, iter %d/%d, file: %s)",
                        self.__class__.__name__, elapsed,
                        iteration + 1, max_iterations, _target,
                    )

            hb = asyncio.create_task(_heartbeat())
            try:
                resp = await self.llm.generate_with_tools(
                    messages=messages,
                    tools=self.tools,
                    system_prompt=self.system_prompt,
                )
            finally:
                hb.cancel()
                try:
                    await hb
                except asyncio.CancelledError:
                    pass
            elapsed = _time.monotonic() - _t0
            tool_names = [tc.name for tc in resp.tool_calls] if resp.tool_calls else []
            logger.info(
                "[%s] LLM responded in %.1fs (iter %d/%d) stop=%s tools=%s",
                self.__class__.__name__, elapsed,
                iteration + 1, max_iterations,
                resp.stop_reason, tool_names or "none",
            )
            return resp

        for iteration in range(max_iterations):
            response = await _llm_call_with_heartbeat(iteration)
            self._metrics["llm_calls"] += 1
            self._metrics["tokens_used"] += sum(response.usage.values())

            if response.stop_reason == "end_turn":
                # If the LLM returned text but never called write_file for the
                # target file, inject a recovery turn rather than exiting.
                # This handles the case where the model outputs code as plain
                # text instead of routing it through the write_file tool.
                target_file = context.file_blueprint.path if context.file_blueprint else None
                if (
                    target_file
                    and target_file not in files_written
                    and response.content
                    and iteration < max_iterations - 1
                ):
                    logger.warning(
                        "%s: end_turn reached but %s not yet written — injecting write_file reminder (iter %d)",
                        self.__class__.__name__, target_file, iteration,
                    )
                    messages.append({"role": "assistant", "content": response.raw_content})
                    messages.append({
                        "role": "user",
                        "content": (
                            f"You have not called write_file yet. "
                            f"Please call write_file with path='{target_file}' "
                            f"and the complete file content now. "
                            f"Do NOT respond with plain text — use the write_file tool."
                        ),
                    })
                    continue
                return self._parse_agentic_result(context, response.content, files_written)

            if not response.tool_calls:
                # Model returned a non-tool_use stop that isn't "end_turn" — treat as done
                logger.warning(
                    "%s agentic loop: unexpected stop_reason=%r with no tool calls (iteration %d)",
                    self.__class__.__name__, response.stop_reason, iteration,
                )
                return self._parse_agentic_result(context, response.content, files_written)

            # Execute all tool calls in this response concurrently.
            # asyncio.gather preserves order, so tool_results aligns with
            # the tool_use blocks in the assistant message.
            async def _run_one(tc: ToolCall) -> str:
                return await self._dispatch_tool(context, tc)

            results = await asyncio.gather(*[_run_one(tc) for tc in response.tool_calls])

            tool_results: list[dict] = []
            for tc, result in zip(response.tool_calls, results):
                logger.debug(
                    "%s tool %s(%s) → %s",
                    self.__class__.__name__, tc.name,
                    list(tc.input.keys()), result[:120] if result else "",
                )
                if tc.name == "write_file" and not result.startswith("Error"):
                    written_path = tc.input.get("path", "")
                    if written_path:
                        files_written.append(written_path)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc.tool_use_id,
                    "content": result,
                })

            # Extend the conversation: assistant turn (with tool_use blocks) + user turn (results)
            messages.append({"role": "assistant", "content": response.raw_content})
            messages.append({"role": "user", "content": tool_results})

            # If the target file was just written, return immediately rather
            # than waiting for the model to emit end_turn.  This prevents
            # hitting max_iterations when the write happens on the last turn.
            target_file = context.file_blueprint.path if context.file_blueprint else None
            if target_file and target_file in files_written:
                return self._parse_agentic_result(context, response.content, files_written)

        raise RuntimeError(
            f"{self.__class__.__name__} exceeded max_iterations ({max_iterations}) "
            "without completing the agentic loop"
        )

    def _parse_agentic_result(
        self,
        context: AgentContext,
        final_text: str,
        files_written: list[str],
    ) -> TaskResult:
        """Convert the agentic loop outcome to a TaskResult.

        Subclasses may override to add domain-specific validation.
        """
        return TaskResult(
            success=True,
            output=(final_text[:500] if final_text else "Agentic task completed"),
            files_modified=files_written,
            metrics=self.get_metrics(),
        )

    async def _dispatch_tool(self, context: AgentContext, tc: ToolCall) -> str:
        """Dispatch a tool call to the appropriate handler.

        Returns a string result for the LLM.  Exceptions are caught and
        returned as error strings so the model can retry or adapt.
        """
        handlers = {
            "read_file":       self._tool_read_file,
            "write_file":      self._tool_write_file,
            "search_code":     self._tool_search_code,
            "find_definition": self._tool_find_definition,
            "list_files":      self._tool_list_files,
            "apply_patch":     self._tool_apply_patch,
        }
        
        handler = handlers.get(tc.name)
        if handler is None:
            # Not a local native handler. Check if it's an MCP tool.
            if self._mcp_client and any(t.name == tc.name for t in self._mcp_tools):
                try:
                    return await self._mcp_client.execute_tool(tc.name, tc.input)
                except Exception as exc:
                    logger.warning("MCP Tool %s execution raised: %s", tc.name, exc)
                    return f"Error executing MCP tool: {exc}"
            return f"Error: unknown tool '{tc.name}'"
            
        try:
            return await handler(tc.input)
        except Exception as exc:
            logger.warning("Tool %s raised: %s", tc.name, exc)
            return f"Error: {exc}"

    # ── Tool handlers ─────────────────────────────────────────────────────────

    async def _tool_read_file(self, inp: dict) -> str:
        path = inp.get("path", "")
        if not path:
            return "Error: 'path' is required"

        content = await self.repo.async_read_file(path)
        if content is None:
            return f"File not found: {path}"

        lines = content.splitlines()
        total = len(lines)

        start_line: int | None = inp.get("start_line")
        end_line: int | None = inp.get("end_line")

        if start_line is None and end_line is None:
            # No range requested — return full file for small files,
            # or the first chunk with a navigation hint for large ones.
            if total <= _READ_CHUNK_LINES:
                return content
            end_line = _READ_CHUNK_LINES
            start_line = 1

        # Clamp to valid 1-based range
        start = max(1, int(start_line or 1))
        if end_line is None:
            end = min(total, start + _READ_CHUNK_LINES - 1)
        else:
            end = min(total, int(end_line))

        chunk = lines[start - 1 : end]

        header = f"[{path}  lines {start}–{end} of {total}]"
        if end < total:
            header += f"  — use start_line={end + 1} to continue"
        return header + "\n" + "\n".join(chunk)

    async def _tool_write_file(self, inp: dict) -> str:
        path = inp.get("path", "")
        content = inp.get("content", "")
        if not path:
            return "Error: 'path' is required"
        # Security: async_write_file is scoped to workspace root
        await self.repo.async_write_file(path, content)
        msg = f"Written {len(content)} bytes to {path}"
        # Relay broken-import warnings so the LLM can correct or accept them
        broken = getattr(self.repo, "_last_broken_imports", [])
        if broken:
            msg += (
                f"\n⚠ Unresolvable imports in {path}: {broken}."
                " These paths do not match any file currently in the workspace."
                " If the missing file will be generated later, you may leave the import as-is."
                " Otherwise, fix the import path or inline the dependency."
                " Do NOT rewrite this file just to fix the import — only rewrite if the"
                " component logic itself is wrong."
            )
        return msg

    async def _tool_search_code(self, inp: dict) -> str:
        query = inp.get("query", "")
        file_pattern = inp.get("file_pattern", "**/*")
        if not query:
            return "Error: 'query' is required"
        results = self._code_search.search(query, file_pattern=file_pattern, max_results=20)
        if not results:
            return f"No results for: {query}"
        return "\n".join(f"{r.file}:{r.line}: {r.content}" for r in results[:20])

    async def _tool_find_definition(self, inp: dict) -> str:
        symbol = inp.get("symbol", "")
        if not symbol:
            return "Error: 'symbol' is required"
        results = self._code_search.find_definition(symbol)
        if not results:
            return f"No definition found for: {symbol}"
        return "\n".join(f"{r.file}:{r.line}: {r.content}" for r in results[:10])

    async def _tool_list_files(self, inp: dict) -> str:
        directory = inp.get("directory", "")
        pattern = inp.get("pattern", "**/*")
        # Always operate on the resolved absolute workspace path so that
        # relative_to() and startswith() comparisons are consistent.
        workspace = self.repo.workspace.resolve()
        base = workspace
        if directory:
            dir_path = Path(directory)
            # If the LLM supplies an absolute path, try to strip the workspace
            # prefix so it can be treated as workspace-relative.  Reject it if
            # it escapes the workspace entirely.
            if dir_path.is_absolute():
                try:
                    directory = str(dir_path.relative_to(workspace))
                except ValueError:
                    return "Error: access denied (path escapes workspace)"
            target = (workspace / directory).resolve()
            if not (target == workspace or target.is_relative_to(workspace)):
                return "Error: access denied (path escapes workspace)"
            base = target
        if not base.exists():
            return f"Directory not found: {directory or '.'}"
        if not base.is_dir():
            return f"Not a directory: {directory}"
        files = sorted(
            str(p.relative_to(workspace))
            for p in base.rglob(pattern)
            if p.is_file()
        )
        return "\n".join(files[:200]) if files else "No files found"

    async def _tool_apply_patch(self, inp: dict) -> str:
        path = inp.get("path", "")
        patch = inp.get("patch", "")
        if not path or not patch:
            return "Error: 'path' and 'patch' are required"
        try:
            return self._file_tools.apply_patch(path, patch)
        except PatchError as exc:
            return f"Patch failed: {exc}"

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
        import time as _time
        _t0 = _time.monotonic()
        _heartbeat_interval = 15

        async def _heartbeat(label: str) -> None:
            while True:
                await asyncio.sleep(_heartbeat_interval)
                logger.info("[%s] …still waiting for LLM (%.0fs, %s)",
                            self.__class__.__name__, _time.monotonic() - _t0, label)

        logger.info("[%s] LLM call (single-shot) — waiting for response…",
                    self.__class__.__name__)
        hb = asyncio.create_task(_heartbeat("initial"))
        try:
            response = await self.llm.generate(
                system_prompt=system_override or self.system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
            )
        finally:
            hb.cancel()
            try:
                await hb
            except asyncio.CancelledError:
                pass
        logger.info("[%s] LLM responded in %.1fs", self.__class__.__name__,
                    _time.monotonic() - _t0)
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
