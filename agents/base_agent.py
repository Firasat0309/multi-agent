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
    max_iterations: int = 10

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

    @staticmethod
    def _estimate_max_tokens(context: AgentContext) -> int:
        """Estimate max_tokens needed for the agentic loop based on context.

        Heuristic: a source file is roughly 3-4 tokens per character.  We need
        enough headroom for the model's reasoning *plus* the file content it
        will write via ``write_file``.  The base is 16384; we scale up when we
        can estimate a larger file.
        """
        _BASE = 16384
        _CHARS_PER_TOKEN = 3.5  # conservative estimate

        # If the target file already exists (FIX_CODE / REVIEW), use its size
        if context.related_files and context.file_blueprint:
            existing = context.related_files.get(context.file_blueprint.path, "")
            if existing:
                # Need room for reasoning + full rewrite of existing file
                file_tokens = int(len(existing) / _CHARS_PER_TOKEN)
                # 2x for reasoning overhead + full file output
                return max(_BASE, file_tokens * 2)

        # Estimate from dependency count: more deps typically → larger file
        if context.file_blueprint:
            dep_count = len(context.file_blueprint.depends_on)
            if dep_count >= 5:
                return max(_BASE, 24576)  # complex file with many deps
            if dep_count >= 3:
                return max(_BASE, 20480)

        return _BASE

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
        max_iterations = self.max_iterations
        _target = (context.file_blueprint.path if context.file_blueprint else None) or "task"
        _heartbeat_interval = 15  # seconds between "still waiting" log lines
        _max_tokens = self._estimate_max_tokens(context)

        async def _llm_call_with_heartbeat(iteration: int) -> "LLMResponseWithTools":
            """Wrap generate_with_tools with a pre-call log and a periodic heartbeat.

            This prevents the screen from going silent for 30–90 seconds during
            a large LLM response.  A background task logs every
            ``_heartbeat_interval`` seconds so the user can see progress.
            """
            import time as _time
            _t0 = _time.monotonic()
            logger.info(
                "[%s] LLM call iter %d/%d for %s (max_tokens=%d) — waiting for response…",
                self.__class__.__name__, iteration + 1, max_iterations, _target, _max_tokens,
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
                    max_tokens=_max_tokens,
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

        _consecutive_nudges = 0  # track how many budget-guard nudges fired in a row
        _end_turn_reminders = 0  # track end_turn recovery attempts
        _MAX_END_TURN_REMINDERS = 2  # cap end_turn reminders to avoid loops
        _MAX_CONSECUTIVE_NUDGES = 2  # cap budget-guard nudges before letting tools run

        target_file = context.file_blueprint.path if context.file_blueprint else None

        for iteration in range(max_iterations):
            response = await _llm_call_with_heartbeat(iteration)
            self._metrics["llm_calls"] += 1
            self._metrics["tokens_used"] += sum(response.usage.values())

            # ── end_turn: LLM stopped without tool calls ──────────────────
            if response.stop_reason == "end_turn":
                if (
                    target_file
                    and not self._path_in_written(target_file, files_written)
                    and response.content
                    and iteration < max_iterations - 1
                    and _end_turn_reminders < _MAX_END_TURN_REMINDERS
                ):
                    _end_turn_reminders += 1
                    logger.warning(
                        "%s: end_turn reached but %s not yet written — "
                        "injecting write_file reminder (%d/%d, iter %d)",
                        self.__class__.__name__, target_file,
                        _end_turn_reminders, _MAX_END_TURN_REMINDERS, iteration,
                    )
                    # Check if the LLM's text contains a code block it
                    # could use as write_file content.
                    code_hint = ""
                    text = response.content or ""
                    if "```" in text:
                        import re as _re
                        code_blocks = _re.findall(
                            r"```(?:\w+)?\n(.*?)```", text, _re.DOTALL,
                        )
                        if code_blocks:
                            longest = max(code_blocks, key=len)
                            if len(longest.strip()) > 50:
                                code_hint = (
                                    f"\n\nIt looks like you already wrote the code as "
                                    f"plain text. Use exactly that code as the content "
                                    f"argument to write_file."
                                )
                    messages.append({"role": "assistant", "content": response.raw_content})
                    messages.append({
                        "role": "user",
                        "content": (
                            f"You have not called write_file yet. "
                            f"Please call write_file with path='{target_file}' "
                            f"and the complete file content now. "
                            f"Do NOT respond with plain text — use the write_file tool."
                            f"{code_hint}"
                        ),
                    })
                    continue
                return self._parse_agentic_result(context, response.content, files_written)

            # ── Truncation recovery ───────────────────────────────────────
            # Output was cut off mid-generation — ask the model to continue.
            if response.stop_reason in ("max_tokens", "length", "MAX_TOKENS") and not response.tool_calls:
                if iteration < max_iterations - 1:
                    logger.warning(
                        "%s: output truncated (stop_reason=%r) with no tool calls — "
                        "requesting continuation (iter %d)",
                        self.__class__.__name__, response.stop_reason, iteration,
                    )
                    messages.append({"role": "assistant", "content": response.raw_content})
                    messages.append({
                        "role": "user",
                        "content": (
                            "Your response was cut off before you could call any tools. "
                            "Please continue and call the appropriate tool (e.g. write_file) "
                            "to complete the task. Do NOT repeat what you already said — "
                            "pick up from where you left off."
                        ),
                    })
                    continue

            if not response.tool_calls:
                # Non-tool_use stop that isn't "end_turn" — treat as done
                logger.warning(
                    "%s agentic loop: unexpected stop_reason=%r with no tool calls (iteration %d)",
                    self.__class__.__name__, response.stop_reason, iteration,
                )
                return self._parse_agentic_result(context, response.content, files_written)

            # ── Budget guard: nudge toward write_file if past halfway ─────
            # We always execute tool calls first so the LLM receives real
            # results.  The nudge is appended as a text block *inside* the
            # tool_results user message (not a separate user message) to
            # maintain strict role alternation.
            _has_write_call = any(
                tc.name == "write_file" for tc in response.tool_calls
            )
            nudge_text: str | None = None
            if (
                target_file
                and iteration >= max_iterations // 2
                and not self._path_in_written(target_file, files_written)
                and not _has_write_call
                and _consecutive_nudges < _MAX_CONSECUTIVE_NUDGES
            ):
                _consecutive_nudges += 1
                logger.warning(
                    "%s: iteration %d/%d with no write to %s — "
                    "scheduling write_file nudge (%d/%d)",
                    self.__class__.__name__, iteration, max_iterations, target_file,
                    _consecutive_nudges, _MAX_CONSECUTIVE_NUDGES,
                )
                nudge_text = (
                    f"WARNING: You have used {iteration + 1} of {max_iterations} iterations "
                    f"and still have not written the target file '{target_file}'. "
                    f"You MUST call write_file with path='{target_file}' on this turn. "
                    f"Do NOT read more files or deliberate further — write the complete "
                    f"component now using write_file."
                )
            elif _has_write_call or not (
                target_file
                and iteration >= max_iterations // 2
                and not self._path_in_written(target_file, files_written)
            ):
                # Reset only when the LLM is actively writing or the nudge
                # condition no longer applies (before halfway, or file written).
                _consecutive_nudges = 0

            # Execute all tool calls concurrently.
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

            # Append nudge as a text block inside the same user message
            # to maintain strict assistant/user role alternation.
            if nudge_text:
                tool_results.append({"type": "text", "text": nudge_text})

            # Extend the conversation: assistant turn + user turn (results + optional nudge)
            messages.append({"role": "assistant", "content": response.raw_content})
            messages.append({"role": "user", "content": tool_results})

            # If the target file was just written, return immediately — unless
            # the code was detected as truncated, in which case keep going so
            # the model sees the warning and can rewrite with complete content.
            has_quality_issue = any(
                "TRUNCATED CODE DETECTED" in r.get("content", "")
                or "STUB CODE DETECTED" in r.get("content", "")
                for r in tool_results
                if "content" in r  # skip text blocks
            )
            if target_file and self._path_in_written(target_file, files_written) and not has_quality_issue:
                return self._parse_agentic_result(context, response.content, files_written)

        raise RuntimeError(
            f"{self.__class__.__name__} exceeded max_iterations ({max_iterations}) "
            "without completing the agentic loop"
        )

    @staticmethod
    def _normalize_path(p: str) -> str:
        """Normalize a file path for comparison (strip './', collapse separators)."""
        import posixpath
        return posixpath.normpath(p)

    @staticmethod
    def _paths_match(a: str, b: str) -> bool:
        """Check whether two paths refer to the same file.

        Handles the common case where the LLM uses an absolute workspace path
        (e.g. ``/home/user/project/src/Foo.java``) while the blueprint stores
        a relative path (``src/Foo.java``).  We normalise both and, if they
        still differ, check whether the shorter path is a suffix of the longer
        one (separated on a ``/`` boundary so ``BarFoo.java`` does not match
        ``Foo.java``).
        """
        import posixpath
        na = posixpath.normpath(a)
        nb = posixpath.normpath(b)
        if na == nb:
            return True
        # Suffix check: "src/Foo.java" matches "/workspace/src/Foo.java"
        # Require a "/" boundary to avoid false positives.
        shorter, longer = (na, nb) if len(na) <= len(nb) else (nb, na)
        return longer.endswith("/" + shorter)

    @classmethod
    def _path_in_written(cls, target: str, written: list[str]) -> bool:
        """Check whether *target* appears in *written* using normalized comparison."""
        return any(cls._paths_match(target, w) for w in written)

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

    # Timeout in seconds for individual tool handler calls (read_file, write_file, etc.).
    # Prevents the agentic loop from hanging indefinitely if underlying I/O stalls.
    _TOOL_TIMEOUT_SECONDS: float = 60.0

    async def _dispatch_tool(self, context: AgentContext, tc: ToolCall) -> str:
        """Dispatch a tool call to the appropriate handler.

        Returns a string result for the LLM.  Exceptions are caught and
        returned as error strings so the model can retry or adapt.
        Each handler is wrapped with a timeout to prevent the agentic loop
        from hanging if underlying I/O stalls.
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
                    return await asyncio.wait_for(
                        self._mcp_client.execute_tool(tc.name, tc.input),
                        timeout=self._TOOL_TIMEOUT_SECONDS,
                    )
                except asyncio.TimeoutError:
                    logger.warning("MCP tool %s timed out after %.0fs", tc.name, self._TOOL_TIMEOUT_SECONDS)
                    return f"Error: MCP tool '{tc.name}' timed out after {self._TOOL_TIMEOUT_SECONDS:.0f}s"
                except Exception as exc:
                    logger.warning("MCP Tool %s execution raised: %s", tc.name, exc)
                    return f"Error executing MCP tool: {exc}"
            return f"Error: unknown tool '{tc.name}'"

        try:
            return await asyncio.wait_for(
                handler(tc.input),
                timeout=self._TOOL_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Tool %s timed out after %.0fs for %s",
                tc.name, self._TOOL_TIMEOUT_SECONDS,
                context.file_blueprint.path if context.file_blueprint else "unknown",
            )
            return f"Error: tool '{tc.name}' timed out after {self._TOOL_TIMEOUT_SECONDS:.0f}s"
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

    @staticmethod
    def _strip_string_literals(content: str) -> str:
        """Remove string literals from source code before structural analysis.

        Strips double-quoted, single-quoted, and backtick strings (including
        common escape sequences) so that brace/bracket counts reflect the code
        structure rather than string content.

        This is a best-effort heuristic — not a full parser.  It handles:
          - "..." and '...' with \\-escapes
          - Template literals `...` (JS/TS)
          - Triple-double-quoted strings (Python)
          - Line comments: // and #
          - Multi-line comments: /* ... */

        Multi-line strings are handled by consuming newlines within the string
        so the line count is preserved (avoids off-by-one issues in error messages).
        """
        import re as _re
        # Patterns ordered longest-match first to avoid partial matches.
        _STRING_RE = _re.compile(
            r'"""[\s\S]*?"""'          # Python triple-double
            r"|'''[\s\S]*?'''"         # Python triple-single
            r"|`(?:[^`\\]|\\.)*`"      # JS/TS template literal
            r'|"(?:[^"\\]|\\.)*"'      # double-quoted
            r"|'(?:[^'\\]|\\.)*'"      # single-quoted
            r"|//[^\n]*"               # line comment (//)
            r"|#[^\n]*"                # line comment (#)
            r"|/\*[\s\S]*?\*/"         # block comment
        )
        # Replace matched strings with spaces to preserve character positions
        # for accurate brace counting without shifting indices.
        return _STRING_RE.sub(lambda m: " " * len(m.group()), content)

    @staticmethod
    def _detect_truncated_code(content: str, path: str) -> str | None:
        """Check if written code looks truncated (unbalanced braces/brackets).

        Returns a warning message if the code appears incomplete, None otherwise.
        Applies only to languages that use braces for scoping.

        Improvements over the naive brace-count approach:
        - Strips string literals and comments before counting so that JSON-like
          structures (e.g. ``{"key": {}}`` in object literals) don't produce
          false positives.
        - Also checks parentheses to catch unclosed function signatures.
        - Threshold of 2 unclosed braces (not 1) to tolerate deliberate partial
          files (e.g. a config-only class with a single open block).
        """
        ext = path.rsplit(".", 1)[-1] if "." in path else ""
        if ext not in ("java", "ts", "tsx", "js", "jsx", "go", "rs", "cs", "kt", "scala", "c", "cpp", "h"):
            return None

        # Strip string literals and comments before counting structural tokens
        # to prevent e.g. {"key": {}} in a string from counting as an open brace.
        stripped = BaseAgent._strip_string_literals(content)

        opens_braces = stripped.count("{") - stripped.count("}")
        opens_parens = stripped.count("(") - stripped.count(")")

        reasons: list[str] = []
        if opens_braces >= 2:
            reasons.append(f"{opens_braces} unclosed braces")
        if opens_parens >= 3:
            # Parens are noisier (macros, decorators) so use a higher threshold
            reasons.append(f"{opens_parens} unclosed parentheses")

        if reasons:
            return (
                f"⚠ TRUNCATED CODE DETECTED: {', '.join(reasons)} in {path}. "
                f"The file content appears incomplete — it likely ends mid-class or mid-method. "
                f"Please call write_file again with the COMPLETE file content. "
                f"Make sure all classes, methods, and blocks are properly closed."
            )
        return None

    @staticmethod
    def _detect_stub_methods(content: str, path: str) -> str | None:
        """Check if written code has unimplemented method bodies (compiled languages only).

        Returns a warning message if stub patterns are found, None otherwise.
        Stubs compile successfully but produce non-functional code that wastes
        the entire review → build → test cycle.
        """
        import re as _re
        ext = path.rsplit(".", 1)[-1] if "." in path else ""
        if ext not in ("java", "cs"):
            return None

        # Count methods with trivial bodies: return null/0/false, throw UnsupportedOperationException
        stub_re = _re.compile(
            r"(?:public|protected|private)\s+(?!class\b|interface\b|enum\b|static\s+final\b)"
            r"\S+\s+\w+\s*\([^)]*\)\s*(?:throws\s+\S+\s*)?\{"
            r"[^}]{0,60}\b(?:return\s+(?:null|0|false|"
            r'"");|throw\s+new\s+(?:UnsupportedOperationException|NotImplementedException))'
        )
        # Abstract-style signatures (semicolon instead of body) in non-interface context
        abstract_re = _re.compile(
            r"(?:public|protected|private)\s+(?!static\s+final\b)\S+\s+\w+\s*\([^)]*\)\s*;"
        )

        stubs = stub_re.findall(content)
        abstracts = abstract_re.findall(content)

        # Only flag if multiple stubs found — a single return null might be valid
        total = len(stubs) + len(abstracts)
        if total >= 2:
            return (
                f"⚠ STUB CODE DETECTED: {total} method(s) in {path} have placeholder "
                f"implementations (return null, return 0, throw UnsupportedOperationException, "
                f"or method signatures ending with ';' instead of a body). "
                f"EVERY method must have a FULL, WORKING implementation. "
                f"Please call write_file again with the complete file where every method "
                f"has real business logic, not stubs."
            )
        return None

    async def _tool_write_file(self, inp: dict) -> str:
        path = inp.get("path", "")
        content = inp.get("content", "")
        if not path:
            return "Error: 'path' is required"
        # Deduplicate content before writing — LLMs sometimes pass the
        # full file content with the class repeated multiple times.
        content = self._deduplicate_content(content)
        # Security: async_write_file is scoped to workspace root
        await self.repo.async_write_file(path, content)
        msg = f"Written {len(content)} bytes to {path}"
        # Detect truncated code (unbalanced braces) and warn the LLM
        trunc_warning = self._detect_truncated_code(content, path)
        if trunc_warning:
            logger.warning("Truncated code detected in %s", path)
            msg += f"\n{trunc_warning}"
        # Detect stub/skeleton methods in compiled languages
        stub_warning = self._detect_stub_methods(content, path)
        if stub_warning:
            logger.warning("Stub code detected in %s", path)
            msg += f"\n{stub_warning}"
        # Relay broken-import warnings so the LLM can correct or accept them
        broken = getattr(self.repo, "_last_broken_imports", [])
        if broken:
            # Collect existing workspace files that might be the correct targets
            # so the LLM can fix the import path instead of guessing.
            repo_index = self.repo.get_repo_index()
            existing_files = sorted(f.path for f in repo_index.files)[:50]
            existing_hint = ""
            if existing_files:
                existing_hint = (
                    "\nFiles currently in the workspace:\n"
                    + "\n".join(f"  {f}" for f in existing_files)
                )
            msg += (
                f"\n⚠ Unresolvable imports in {path}: {broken}."
                " These paths do not match any file currently in the workspace."
                f"{existing_hint}"
                "\nIf the missing file will be generated later, you may leave the import as-is."
                " Otherwise, fix the import path to point to the correct existing file."
                " Do NOT rewrite this file just to fix the import — only rewrite if the"
                " import path is clearly wrong and should point to an existing file above."
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
            # Show trailing context so the LLM can find the cut-off point.
            # We intentionally omit the full original prompt to avoid
            # ballooning the context on each continuation — the system
            # prompt already provides enough background.
            tail_chars = min(800, len(content))
            continuation_prompt = (
                f"IMPORTANT: Your previous response was cut off mid-way. "
                f"Here is the end of what you wrote (last {tail_chars} chars):\n"
                f"```\n{content[-tail_chars:]}\n```\n"
                f"Continue EXACTLY from where the output was cut off. "
                f"Do NOT repeat any class, function, or method definition that "
                f"already appears above — duplicates will be rejected. "
                f"Output only the remaining code — no preamble, no repetition."
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

            # ── Full-restart detection ─────────────────────────────────
            # LLMs (especially Gemini Flash) often restart from the very
            # beginning of the file instead of continuing.  Detect this by
            # checking if the continuation starts with the same leading
            # lines (package/import block) as the already-accumulated
            # content.  When detected, keep the longer version rather than
            # blindly appending a duplicate.
            full_restart = False
            if continuation.strip():
                # Compare first non-blank line of content vs continuation
                first_orig = ""
                for _l in content.splitlines():
                    if _l.strip():
                        first_orig = _l.strip()
                        break
                first_cont = ""
                for _l in continuation.splitlines():
                    if _l.strip():
                        first_cont = _l.strip()
                        break
                if (
                    first_orig
                    and first_cont
                    and first_orig == first_cont
                    and any(first_orig.startswith(kw) for kw in (
                        "package ", "import ", "from ", "#include", "module ",
                        "namespace ", "using ", "pub mod", "pub use", "#![",
                    ))
                ):
                    full_restart = True
                    logger.warning(
                        "Full-restart detected in continuation (starts with %r) "
                        "— using longer version instead of appending duplicate",
                        first_cont[:60],
                    )
                    # Keep whichever version is longer (the continuation
                    # may have made it further before being truncated again)
                    if len(continuation) > len(content):
                        content = continuation
                    # else: keep existing content as-is

            if not full_restart:
                # Detect overlapping tail: the LLM sometimes repeats the last
                # few lines (or even larger sections) for context.  Find the
                # longest suffix of `content` that matches a prefix of
                # `continuation` and skip it.  We check up to the last 30 lines
                # because LLMs can restart from the beginning of a class or
                # function definition that was 10–20 lines back.
                overlap = 0
                max_overlap_lines = 30
                tail_lines = content.rsplit("\n", max_overlap_lines)[-max_overlap_lines:]
                for n in range(min(len(tail_lines), max_overlap_lines), 0, -1):
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

        # Final deduplication pass: detect if the content contains the same
        # class/file repeated multiple times (LLM continuation failure).
        if continuation_count > 0:
            content = self._deduplicate_content(content)

        return content

    @staticmethod
    def _deduplicate_content(content: str) -> str:
        """Detect and remove duplicated file blocks in LLM output.

        When LLM continuations fail (restart from beginning), the content
        may contain the same class/package declaration repeated 2-4 times.
        This finds the repeating unit and keeps only the longest instance.
        """
        import re as _re

        # Detect duplicate package/module/namespace declarations
        # These should appear at most once in a valid source file.
        header_patterns = [
            _re.compile(r"^package\s+[\w.]+\s*;", _re.MULTILINE),         # Java
            _re.compile(r"^from\s+__future__\s+import\b", _re.MULTILINE),  # Python
            _re.compile(r"^#include\s+[<\"]", _re.MULTILINE),              # C/C++
            _re.compile(r"^namespace\s+[\w.]+\s*\{?", _re.MULTILINE),     # C#
            _re.compile(r"^package\s+\w+\s*$", _re.MULTILINE),            # Go/Kotlin
        ]

        for pattern in header_patterns:
            matches = list(pattern.finditer(content))
            if len(matches) >= 2:
                # Found duplicate declarations — split at each occurrence
                # and keep the longest block (most complete version).
                positions = [m.start() for m in matches]
                blocks: list[str] = []
                for i, pos in enumerate(positions):
                    end = positions[i + 1] if i + 1 < len(positions) else len(content)
                    blocks.append(content[pos:end].rstrip())
                longest = max(blocks, key=len)
                logger.warning(
                    "Deduplication: found %d repeated blocks (pattern: %s), "
                    "keeping longest (%d chars)",
                    len(blocks), pattern.pattern[:40], len(longest),
                )
                return longest.rstrip() + "\n"

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
        """Format agent context into a prompt section.

        The primary file (matching the current task's file_blueprint) gets full
        content to prevent truncation of critical method signatures. Dependency
        files are truncated at 4000 chars to stay within token budget.
        """
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

        # Include API contract summary when available — critical for fullstack mode
        # so backend agents know the exact endpoint signatures they must implement.
        if context.api_contract:
            import json as _json
            ac = context.api_contract
            contract_lines = [
                f"## API Contract ({ac.title} v{ac.version})",
                f"Base URL: {ac.base_url}",
                "Endpoints:",
            ]
            for ep in ac.endpoints:
                auth = " [auth]" if ep.auth_required else ""
                contract_lines.append(f"  {ep.method} {ep.path}{auth} — {ep.description}")
                if ep.request_schema:
                    try:
                        contract_lines.append(f"    Request: {_json.dumps(ep.request_schema)}")
                    except Exception:
                        pass
                if ep.response_schema:
                    try:
                        contract_lines.append(f"    Response: {_json.dumps(ep.response_schema)}")
                    except Exception:
                        pass
            parts.append("\n".join(contract_lines))

        if context.related_files:
            parts.append("## Related Files")
            # Determine the primary file path to give it full content
            primary_path = context.file_blueprint.path if context.file_blueprint else None
            for path, content in context.related_files.items():
                # Primary file gets full content; dependencies get truncated
                if path == primary_path:
                    truncated = content
                else:
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
