"""Typed tool factory with fail-closed defaults.

Inspired by Claude Code's ``buildTool()`` pattern — every tool definition
goes through this factory to guarantee consistent, safe defaults.

Fail-closed means: if a tool doesn't explicitly declare itself as
concurrency-safe, read-only, or non-destructive, the factory assumes
the WORST case.  This is the same design philosophy as Claude Code's
permission system.

Usage::

    from core.tool_factory import build_tool, ToolDef

    read_file_tool = build_tool(
        name="read_file",
        description="Read workspace file contents",
        input_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        execute=_handle_read_file,
        is_read_only=True,
        is_concurrency_safe=True,
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol

logger = logging.getLogger(__name__)


# ── Input validation ─────────────────────────────────────────────────────────

class InputValidator(Protocol):
    """Protocol for tool input validators."""
    def __call__(self, tool_input: dict[str, Any]) -> str | None:
        """Return None if valid, or an error message string."""
        ...


class PermissionChecker(Protocol):
    """Protocol for tool permission checks."""
    def __call__(self, tool_input: dict[str, Any], *, agent_name: str) -> bool:
        """Return True if the operation is allowed."""
        ...


# ── Tool definition ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ToolDef:
    """Complete tool definition with safe defaults.

    All tools must go through ``build_tool()`` to guarantee defaults.
    """
    name: str
    description: str
    input_schema: dict[str, Any]
    execute: Callable[..., Any]

    # ── Safety flags (fail-closed defaults) ──────────────────────────
    is_read_only: bool = False          # assume writes by default
    is_concurrency_safe: bool = False   # assume NOT safe by default
    is_destructive: bool = False        # assume non-destructive by default
    requires_permission: bool = True    # assume permission check needed

    # ── Optional hooks ───────────────────────────────────────────────
    validate_input: InputValidator | None = None
    check_permission: PermissionChecker | None = None

    # ── Metadata ─────────────────────────────────────────────────────
    category: str = "general"           # file, search, build, mcp, etc.
    user_facing_name: str = ""          # display name (defaults to name)


def build_tool(
    *,
    name: str,
    description: str,
    input_schema: dict[str, Any],
    execute: Callable[..., Any],
    is_read_only: bool = False,
    is_concurrency_safe: bool = False,
    is_destructive: bool = False,
    requires_permission: bool = True,
    validate_input: InputValidator | None = None,
    check_permission: PermissionChecker | None = None,
    category: str = "general",
    user_facing_name: str = "",
) -> ToolDef:
    """Build a complete tool definition with fail-closed defaults.

    This is the ONLY way to create a ToolDef.  The factory ensures:
      - `is_concurrency_safe` defaults to False (assume not safe)
      - `is_read_only` defaults to False (assume writes)
      - `requires_permission` defaults to True (assume check needed)
      - `user_facing_name` defaults to `name`
    """
    return ToolDef(
        name=name,
        description=description,
        input_schema=input_schema,
        execute=execute,
        is_read_only=is_read_only,
        is_concurrency_safe=is_concurrency_safe,
        is_destructive=is_destructive,
        requires_permission=requires_permission,
        validate_input=validate_input,
        check_permission=check_permission,
        category=category,
        user_facing_name=user_facing_name or name,
    )


# ── Tool registry ────────────────────────────────────────────────────────────

class ToolRegistry:
    """Central registry for all available tools.

    Supports feature-gated registration so tools can be conditionally
    enabled/disabled without modifying the tool definition.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDef] = {}
        self._disabled: set[str] = set()

    def register(self, tool: ToolDef) -> None:
        """Register a tool. Overwrites if already registered."""
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Remove a tool from the registry."""
        self._tools.pop(name, None)

    def disable(self, name: str) -> None:
        """Conditionally disable a tool (keeps it registered but hidden)."""
        self._disabled.add(name)

    def enable(self, name: str) -> None:
        """Re-enable a previously disabled tool."""
        self._disabled.discard(name)

    def get(self, name: str) -> ToolDef | None:
        """Get a tool by name (returns None if not registered or disabled)."""
        if name in self._disabled:
            return None
        return self._tools.get(name)

    def get_all(self) -> list[ToolDef]:
        """Return all enabled tools."""
        return [
            tool for name, tool in self._tools.items()
            if name not in self._disabled
        ]

    def get_by_category(self, category: str) -> list[ToolDef]:
        """Return enabled tools in a specific category."""
        return [
            tool for tool in self.get_all()
            if tool.category == category
        ]

    def get_read_only(self) -> list[ToolDef]:
        """Return tools that are safe for read-only agents."""
        return [tool for tool in self.get_all() if tool.is_read_only]

    def get_concurrency_safe(self) -> list[ToolDef]:
        """Return tools safe for concurrent execution."""
        return [tool for tool in self.get_all() if tool.is_concurrency_safe]

    def list_names(self) -> list[str]:
        """Return names of all enabled tools."""
        return [tool.name for tool in self.get_all()]

    def __len__(self) -> int:
        return len([n for n in self._tools if n not in self._disabled])

    def __contains__(self, name: str) -> bool:
        return name in self._tools and name not in self._disabled
