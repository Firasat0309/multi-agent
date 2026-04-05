"""Self-registering tool system inspired by Claude Code's declarative tool pattern.

Instead of maintaining parallel lists of tool definitions and hardcoded handler
maps (as in ``agent_tools.py`` and ``BaseAgent._dispatch_tool()``), tools
register themselves with a global registry at import time.

Usage::

    from core.tool_registry import ToolRegistry, tool

    @tool(
        name="read_file",
        description="Read the content of a file in the workspace.",
        input_schema={...},
        is_concurrency_safe=True,
    )
    async def handle_read_file(inp: dict, *, ctx: ToolContext) -> str:
        content = await ctx.repo.async_read_file(inp["path"])
        return content or f"File not found: {inp['path']}"

    # In agent setup:
    registry = ToolRegistry.default()
    tools = registry.get_definitions(tags={"read", "search"})
    result = await registry.dispatch("read_file", inp, ctx=tool_ctx)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable

from core.llm_client import ToolDefinition

logger = logging.getLogger(__name__)

__all__ = ["ToolRegistry", "ToolContext", "tool"]


# ── Tool context (injected into handlers) ────────────────────────────────────

@runtime_checkable
class RepoLike(Protocol):
    """Minimal protocol for repository access in tool handlers."""

    async def async_read_file(self, path: str) -> str | None: ...
    async def async_write_file(self, path: str, content: str) -> None: ...

    @property
    def workspace(self) -> str: ...


@dataclass
class ToolContext:
    """Runtime context injected into every tool handler call."""
    repo: RepoLike
    workspace: str
    permission_checker: Any | None = None
    agent_name: str = "unknown"


# ── Handler type ─────────────────────────────────────────────────────────────

ToolHandler = Callable[[dict, ToolContext], Awaitable[str]]


# ── Registry entry ───────────────────────────────────────────────────────────

@dataclass
class RegisteredTool:
    """A tool definition paired with its async handler."""
    definition: ToolDefinition
    handler: ToolHandler
    tags: frozenset[str] = field(default_factory=frozenset)


# ── Global registry ──────────────────────────────────────────────────────────

class ToolRegistry:
    """Central registry for self-registering tools.

    Tools register via the module-level ``@tool`` decorator or by calling
    ``registry.register()`` directly.  The registry provides both the
    ``ToolDefinition`` list (for the LLM) and the dispatch table (for
    execution) in a single source of truth.
    """

    _default: "ToolRegistry | None" = None

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    @classmethod
    def default(cls) -> "ToolRegistry":
        """Return the global default registry (created on first access)."""
        if cls._default is None:
            cls._default = cls()
        return cls._default

    @classmethod
    def reset_default(cls) -> None:
        """Reset the default registry (for testing)."""
        cls._default = None

    # ── Registration ─────────────────────────────────────────────────

    def register(
        self,
        name: str,
        description: str,
        input_schema: dict,
        handler: ToolHandler,
        *,
        is_concurrency_safe: bool = True,
        tags: set[str] | frozenset[str] | None = None,
    ) -> None:
        """Register a tool with its handler."""
        if name in self._tools:
            logger.warning("Overwriting tool registration: %s", name)
        defn = ToolDefinition(
            name=name,
            description=description,
            input_schema=input_schema,
            is_concurrency_safe=is_concurrency_safe,
        )
        self._tools[name] = RegisteredTool(
            definition=defn,
            handler=handler,
            tags=frozenset(tags) if tags else frozenset(),
        )

    def unregister(self, name: str) -> None:
        """Remove a tool from the registry."""
        self._tools.pop(name, None)

    # ── Querying ─────────────────────────────────────────────────────

    def get(self, name: str) -> RegisteredTool | None:
        """Look up a registered tool by name."""
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def get_definitions(
        self,
        *,
        tags: set[str] | None = None,
        names: set[str] | None = None,
    ) -> list[ToolDefinition]:
        """Return tool definitions, optionally filtered by tags or names.

        If *tags* is given, returns tools that have ANY of the specified tags.
        If *names* is given, returns only tools with those names.
        Both filters are AND-combined when specified together.
        """
        result = []
        for rt in self._tools.values():
            if names is not None and rt.definition.name not in names:
                continue
            if tags is not None and not rt.tags.intersection(tags):
                continue
            result.append(rt.definition)
        return result

    def all_definitions(self) -> list[ToolDefinition]:
        """Return all registered tool definitions."""
        return [rt.definition for rt in self._tools.values()]

    def all_names(self) -> list[str]:
        """Return all registered tool names."""
        return list(self._tools.keys())

    # ── Dispatch ─────────────────────────────────────────────────────

    async def dispatch(
        self,
        name: str,
        inp: dict,
        ctx: ToolContext,
        *,
        timeout: float = 60.0,
    ) -> str:
        """Dispatch a tool call to its registered handler.

        Returns the tool result string.  Unknown tools return an error string.
        Exceptions are caught and returned as error strings.
        """
        rt = self._tools.get(name)
        if rt is None:
            return f"Error: unknown tool '{name}'"
        try:
            return await asyncio.wait_for(
                rt.handler(inp, ctx),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("Tool %s timed out after %.0fs", name, timeout)
            return f"Error: tool '{name}' timed out after {timeout:.0f}s"
        except Exception as exc:
            logger.warning("Tool %s raised: %s", name, exc)
            return f"Error: {exc}"


# ── Decorator for registration ───────────────────────────────────────────────

def tool(
    name: str,
    description: str,
    input_schema: dict,
    *,
    is_concurrency_safe: bool = True,
    tags: set[str] | None = None,
    registry: ToolRegistry | None = None,
) -> Callable[[ToolHandler], ToolHandler]:
    """Decorator to register a tool handler with the global (or given) registry.

    Usage::

        @tool(
            name="read_file",
            description="Read a file",
            input_schema={"type": "object", "properties": {...}, "required": ["path"]},
            tags={"read", "file"},
        )
        async def handle_read_file(inp: dict, ctx: ToolContext) -> str:
            ...
    """
    def decorator(fn: ToolHandler) -> ToolHandler:
        reg = registry or ToolRegistry.default()
        reg.register(
            name=name,
            description=description,
            input_schema=input_schema,
            handler=fn,
            is_concurrency_safe=is_concurrency_safe,
            tags=tags,
        )
        return fn
    return decorator
