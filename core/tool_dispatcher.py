"""Tool dispatch engine — routes tool calls to implementations.

Extracted from the monolithic BaseAgent to separate concerns:
  - ToolDispatcher handles routing, concurrency, permission checks
  - BaseAgent focuses on task-specific logic
  - RecoveryManager handles error recovery and retries

This is instantiated once per agent and reused across all iterations.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, TYPE_CHECKING

from core.agent_tools import ToolDefinition, validate_tool_input, validate_tool_input_strict
from core.llm_client import ToolCall
from core.permissions import ToolPermissionChecker

if TYPE_CHECKING:
    from core.repository_manager import RepositoryManager
    from core.mcp_client import MCPClient
    from tools.code_search import CodeSearch
    from tools.file_tools import FileTools, PatchError

logger = logging.getLogger(__name__)

# Files longer than this are served in chunks.
_READ_CHUNK_LINES = 150


class ToolDispatcher:
    """Routes tool calls to their implementations with permission checks.

    Handles:
    - Standard tools (read_file, write_file, search_code, etc.)
    - MCP tools (delegated to MCPClient)
    - Concurrent execution of safe tools
    - Permission validation before execution
    - File quality checks on writes (truncation, stubs)
    """

    def __init__(
        self,
        repo: RepositoryManager,
        code_search: CodeSearch,
        file_tools: FileTools,
        *,
        mcp_client: MCPClient | None = None,
        permission_checker: ToolPermissionChecker | None = None,
        quality_checker: QualityChecker | None = None,
    ) -> None:
        self.repo = repo
        self._code_search = code_search
        self._file_tools = file_tools
        self._mcp_client = mcp_client
        self._permission_checker = permission_checker
        self._quality_checker = quality_checker or QualityChecker()

    async def dispatch(self, tool_call: ToolCall) -> str:
        """Execute a single tool call and return the result string."""
        name = tool_call.name
        inp = tool_call.input

        # Permission check
        if self._permission_checker and name in ("write_file", "apply_patch"):
            ok, reason = self._permission_checker.check_write(inp.get("path", ""))
            if not ok:
                return f"Error: permission denied — {reason}"

        # Validate input schema
        validation_error = validate_tool_input(name, inp)
        if validation_error:
            return f"Error: {validation_error}"

        # Strict validation (Pydantic) for known tools
        strict_error = validate_tool_input_strict(name, inp)
        if strict_error:
            return f"Error: {strict_error}"

        # Route to handler
        handlers = {
            "read_file": self._tool_read_file,
            "write_file": self._tool_write_file,
            "search_code": self._tool_search_code,
            "find_definition": self._tool_find_definition,
            "list_files": self._tool_list_files,
            "apply_patch": self._tool_apply_patch,
        }

        handler = handlers.get(name)
        if handler:
            return await handler(inp)

        # MCP tool fallback
        if self._mcp_client:
            try:
                return await self._mcp_client.call_tool(name, inp)
            except Exception as e:
                return f"Error calling MCP tool '{name}': {e}"

        return f"Error: unknown tool '{name}'"

    async def execute_batch(
        self,
        tool_calls: list[ToolCall],
        safe_tool_names: set[str],
    ) -> list[str]:
        """Execute a batch of tool calls, running safe ones concurrently.

        Returns results in the same order as tool_calls.
        """
        safe_calls = [(i, tc) for i, tc in enumerate(tool_calls) if tc.name in safe_tool_names]
        unsafe_calls = [(i, tc) for i, tc in enumerate(tool_calls) if tc.name not in safe_tool_names]

        results: list[str | None] = [None] * len(tool_calls)

        # Run safe tools concurrently
        if safe_calls:
            safe_results = await asyncio.gather(
                *[self.dispatch(tc) for _, tc in safe_calls],
                return_exceptions=True,
            )
            for (idx, _), res in zip(safe_calls, safe_results):
                if isinstance(res, Exception):
                    results[idx] = f"Error: {res}"
                else:
                    results[idx] = res

        # Run unsafe tools sequentially
        for idx, tc in unsafe_calls:
            try:
                results[idx] = await self.dispatch(tc)
            except Exception as e:
                results[idx] = f"Error: {e}"

        return results  # type: ignore[return-value]

    # ── Tool implementations ──────────────────────────────────────────────

    async def _tool_read_file(self, inp: dict) -> str:
        path = inp.get("path", "")
        if not path:
            return "Error: 'path' is required"
        start_line = inp.get("start_line")
        end_line = inp.get("end_line")
        try:
            full = await self.repo.async_read_file(path)
        except FileNotFoundError:
            return f"Error: file not found: {path}"
        except PermissionError:
            return f"Error: access denied (path escapes workspace): {path}"
        except Exception as e:
            return f"Error reading {path}: {e}"

        lines = full.splitlines(keepends=True)
        total = len(lines)

        if start_line is not None:
            s = max(1, int(start_line)) - 1
        else:
            s = 0
        if end_line is not None:
            e = min(total, int(end_line))
        else:
            e = min(total, s + _READ_CHUNK_LINES)

        chunk = "".join(lines[s:e])
        header = f"[File: {path} | Lines {s+1}-{e} of {total}]\n"
        return header + chunk

    async def _tool_write_file(self, inp: dict) -> str:
        path = inp.get("path", "")
        content = inp.get("content", "")
        if not path:
            return "Error: 'path' is required"
        # Strip LLM artifacts before writing
        content = self._quality_checker.strip_leading_prose(content, path)
        content = self._quality_checker.strip_trailing_fence(content, path)
        content = self._quality_checker.deduplicate_content(content)
        # Write file
        await self.repo.async_write_file(path, content)
        msg = f"Written {len(content)} bytes to {path}"
        # Quality checks
        trunc_warning = self._quality_checker.detect_truncated_code(content, path)
        if trunc_warning:
            logger.warning("Truncated code detected in %s", path)
            msg += f"\n{trunc_warning}"
        stub_warning = self._quality_checker.detect_stub_methods(content, path)
        if stub_warning:
            logger.warning("Stub code detected in %s", path)
            msg += f"\n{stub_warning}"
        # Relay broken-import warnings
        broken = getattr(self.repo, "_last_broken_imports", [])
        if broken:
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
        workspace = self.repo.workspace.resolve()
        base = workspace
        if directory:
            dir_path = Path(directory)
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
        from tools.file_tools import PatchError
        path = inp.get("path", "")
        patch = inp.get("patch", "")
        if not path or not patch:
            return "Error: 'path' and 'patch' are required"
        try:
            return self._file_tools.apply_patch(path, patch)
        except PatchError as exc:
            return f"Patch failed: {exc}"


class QualityChecker:
    """Checks quality of LLM-generated code before/after writing.

    Extracted from BaseAgent to be reusable by ToolDispatcher and other
    components without depending on the full agent class hierarchy.
    """

    @staticmethod
    def strip_string_literals(content: str) -> str:
        """Strip string literals and comments for brace counting."""
        import re
        result = re.sub(r'"(?:[^"\\]|\\.)*"', '""', content)
        result = re.sub(r"'(?:[^'\\]|\\.)*'", "''", result)
        result = re.sub(r"/\*.*?\*/", "", result, flags=re.DOTALL)
        result = re.sub(r"//[^\n]*", "", result)
        result = re.sub(r"#[^\n]*", "", result)
        return result

    @staticmethod
    def detect_truncated_code(content: str, path: str) -> str | None:
        """Check if written code looks truncated (unbalanced braces/brackets)."""
        ext = path.rsplit(".", 1)[-1] if "." in path else ""
        if ext not in ("java", "ts", "tsx", "js", "jsx", "go", "rs", "cs", "kt", "scala", "c", "cpp", "h"):
            return None
        stripped = QualityChecker.strip_string_literals(content)
        opens_braces = stripped.count("{") - stripped.count("}")
        opens_parens = stripped.count("(") - stripped.count(")")
        reasons: list[str] = []
        if opens_braces >= 2:
            reasons.append(f"{opens_braces} unclosed braces")
        if opens_parens >= 3:
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
    def detect_stub_methods(content: str, path: str) -> str | None:
        """Check if written code has unimplemented method bodies."""
        import re
        ext = path.rsplit(".", 1)[-1] if "." in path else ""
        if ext not in ("java", "cs"):
            return None
        stub_re = re.compile(
            r"(?:public|protected|private)\s+(?!class\b|interface\b|enum\b|static\s+final\b)"
            r"\S+\s+\w+\s*\([^)]*\)\s*(?:throws\s+\S+\s*)?\{"
            r"[^}]{0,60}\b(?:return\s+(?:null|0|false|"
            r'"");|throw\s+new\s+(?:UnsupportedOperationException|NotImplementedException))'
        )
        abstract_re = re.compile(
            r"(?:public|protected|private)\s+(?!static\s+final\b)\S+\s+\w+\s*\([^)]*\)\s*;"
        )
        stubs = stub_re.findall(content)
        abstracts = abstract_re.findall(content)
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

    @staticmethod
    def strip_leading_prose(content: str, path: str) -> str:
        """Strip LLM chain-of-thought text that precedes actual source code."""
        import re
        from pathlib import Path as _Path

        if not content or len(content) < 30:
            return content
        ext = _Path(path).suffix.lower() if path else ""

        _FIRST_CODE_LINE: dict[str, re.Pattern[str]] = {
            ".java": re.compile(
                r"^(?:package\s|import\s|/\*|//|@\w|public\s|private\s|protected\s|abstract\s|final\s)",
                re.MULTILINE,
            ),
            ".py": re.compile(
                r'^(?:#!|\"\"\"|from\s+\S|import\s+\S|class\s+\w|def\s+\w|async\s+def\s|@\w|#\s)',
                re.MULTILINE,
            ),
            ".ts": re.compile(
                r"^(?:import\s|export\s|class\s|interface\s|type\s|const\s|let\s|var\s|function\s|//|/\*|enum\s|declare\s|namespace\s|'use |\"use )",
                re.MULTILINE,
            ),
            ".tsx": re.compile(
                r"^(?:import\s|export\s|class\s|interface\s|type\s|const\s|let\s|var\s|function\s|//|/\*|enum\s|declare\s|namespace\s|'use |\"use )",
                re.MULTILINE,
            ),
            ".go": re.compile(
                r"^(?:package\s+\w|import\s|//|/\*|func\s|type\s|var\s|const\s)",
                re.MULTILINE,
            ),
            ".cs": re.compile(
                r"^(?:using\s+\w|namespace\s|//|/\*|\[\w|public\s|private\s|protected\s|internal\s|class\s|interface\s|global\s)",
                re.MULTILINE,
            ),
            ".rs": re.compile(
                r"^(?:use\s|mod\s|//|/\*|pub\s|fn\s|struct\s|enum\s|trait\s|impl\s|#\[|extern\s|#!\[)",
                re.MULTILINE,
            ),
        }
        pattern = _FIRST_CODE_LINE.get(ext)
        if not pattern:
            return content
        m = pattern.search(content)
        if not m or m.start() == 0:
            return content
        leading = content[:m.start()]
        _PROSE_INDICATORS = re.compile(
            r"(?:\bI'll\b|\bI will\b|\bWait[,!]|\bLet me\b|\bOkay[,.]"
            r"|\bI need to\b|\bI should\b|\bI'm\b|\bHmm\b"
            r"|\*\s{2,}|`[^`]+`|\bcheck\b.*\brule\b|\bready\b)",
            re.IGNORECASE,
        )
        if _PROSE_INDICATORS.search(leading):
            logger.warning(
                "Stripped %d chars of LLM deliberation prose before code in %s",
                len(leading), path,
            )
            return content[m.start():]
        return content

    @staticmethod
    def strip_trailing_fence(content: str, path: str = "") -> str:
        """Remove trailing markdown fences and LLM commentary after code."""
        import re
        from pathlib import Path as _Path
        if "```" not in content:
            return content
        ext = _Path(path).suffix.lower() if path else ""
        if ext in {".md", ".markdown", ".mdx", ".rst", ".txt"}:
            return content
        fence_pos = content.rfind("```")
        if fence_pos <= 0:
            return content
        trailing = content[fence_pos + 3:].strip()
        if not trailing or not re.match(
            r"^(?:since|here(?:'s| is)?|note|this|that|the|i\b|let\b|okay\b|wait\b|provided\b)",
            trailing, re.IGNORECASE,
        ):
            return content
        stripped = content[:fence_pos].rstrip()
        if stripped:
            return stripped + "\n"
        return content

    @staticmethod
    def deduplicate_content(content: str) -> str:
        """Detect and remove duplicated file blocks in LLM output."""
        import re
        header_patterns = [
            re.compile(r"^package\s+[\w.]+\s*;", re.MULTILINE),
            re.compile(r"^from\s+__future__\s+import\b", re.MULTILINE),
            re.compile(r"^#include\s+[<\"]", re.MULTILINE),
            re.compile(r"^namespace\s+[\w.]+\s*\{?", re.MULTILINE),
            re.compile(r"^package\s+\w+\s*$", re.MULTILINE),
        ]
        for pattern in header_patterns:
            matches = list(pattern.finditer(content))
            if len(matches) >= 2:
                positions = [m.start() for m in matches]
                blocks: list[str] = []
                for i, pos in enumerate(positions):
                    end = positions[i + 1] if i + 1 < len(positions) else len(content)
                    blocks.append(content[pos:end].rstrip())
                longest = max(blocks, key=len)
                logger.warning(
                    "Deduplication: found %d repeated blocks (pattern: %s), keeping longest (%d chars)",
                    len(blocks), pattern.pattern[:40], len(longest),
                )
                return longest.rstrip() + "\n"
        return content

    @staticmethod
    def extract_code_block(text: str) -> str | None:
        """Return the longest fenced code block from text, or None."""
        import re
        if "```" not in text:
            return None
        blocks = re.findall(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
        if not blocks:
            return None
        longest = max(blocks, key=len)
        return longest if len(longest.strip()) > 50 else None
