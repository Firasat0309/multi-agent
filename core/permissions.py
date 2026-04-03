"""Permission and safety layer for agent tool execution.

Inspired by Claude Code's multi-layered permission system, this provides
workspace-scoped safety checks that prevent agents from:
  - Writing outside the workspace (path traversal)
  - Writing to disallowed file types
  - Executing dangerous shell commands
  - Exceeding per-agent rate limits

Usage::

    checker = ToolPermissionChecker(workspace_root=Path("workspace"))
    ok, reason = checker.check_write("src/main.py")
    ok, reason = checker.check_command("mvn package")
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from core.errors import PermissionDeniedError

logger = logging.getLogger(__name__)


# ── Permission modes ─────────────────────────────────────────────────────────

class PermissionMode(StrEnum):
    """Permission enforcement level."""
    STRICT = "strict"           # deny anything not explicitly allowed
    DEFAULT = "default"         # deny known-dangerous, allow everything else
    PERMISSIVE = "permissive"   # warn but allow (dev-only)


# ── Result types ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PermissionResult:
    """Result of a permission check."""
    allowed: bool
    reason: str = ""
    tool_name: str = ""
    path: str = ""


# ── Blocked command patterns ─────────────────────────────────────────────────
# Regex patterns that match dangerous shell commands.  Ordered roughly by
# severity.  The list is conservative — it's better to block a legitimate
# command and have the user allowlist it than to let a hallucinating LLM
# run ``rm -rf /``.

_BLOCKED_COMMAND_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\brm\s+(-[rR]f?|--recursive)\s", re.IGNORECASE),
    re.compile(r"\brm\s+-[^\s]*f[^\s]*\s+/", re.IGNORECASE),
    re.compile(r"\bsudo\b"),
    re.compile(r"\bchmod\s+777\b"),
    re.compile(r"\bcurl\b.*\|\s*(sh|bash)\b"),
    re.compile(r"\bwget\b.*\|\s*(sh|bash)\b"),
    re.compile(r"\b(mkfs|dd\s+if=|fdisk)\b"),
    re.compile(r"\b(shutdown|reboot|halt|poweroff)\b"),
    re.compile(r"\bkill\s+-9\s+1\b"),
    re.compile(r":(){.*};:"),  # fork bomb
    re.compile(r"\beval\b.*\$\("),  # command injection via eval
    re.compile(r">\s*/dev/sd[a-z]"),  # write to raw disk
    re.compile(r"\bnc\s+-[el]"),  # netcat listener (reverse shell)
]

# File extensions that are NEVER allowed to be written (binary, system, etc.)
_BLOCKED_EXTENSIONS: frozenset[str] = frozenset({
    ".exe", ".dll", ".so", ".dylib", ".bin",
    ".sh" if False else "",  # .sh is allowed — it's used for deploy scripts
    ".pem", ".key", ".p12", ".keystore",  # secrets — should use env vars
})
# Remove empty string from the set
_BLOCKED_EXTENSIONS = _BLOCKED_EXTENSIONS - {""}


# ── Rate limiter ─────────────────────────────────────────────────────────────

class _RateLimiter:
    """Simple sliding-window rate limiter for tool calls."""

    def __init__(self, max_calls: int, window_seconds: float) -> None:
        self._max_calls = max_calls
        self._window = window_seconds
        self._calls: dict[str, list[float]] = {}

    def check(self, key: str) -> bool:
        """Return True if the call is within rate limits."""
        now = time.monotonic()
        calls = self._calls.setdefault(key, [])
        # Prune expired entries
        calls[:] = [t for t in calls if now - t < self._window]
        if len(calls) >= self._max_calls:
            return False
        calls.append(now)
        return True

    def reset(self) -> None:
        self._calls.clear()


# ── Main permission checker ──────────────────────────────────────────────────

class ToolPermissionChecker:
    """Workspace-scoped permission checker for agent tool calls.

    Provides three categories of checks:
      1. **Write checks** — path traversal, extension blocklist, workspace scope
      2. **Command checks** — dangerous command blocklist
      3. **Rate limit checks** — per-agent call frequency limits

    Parameters
    ----------
    workspace_root : Path
        All file operations must resolve within this directory.
    mode : PermissionMode
        Enforcement level (strict / default / permissive).
    allowed_extensions : set[str] | None
        If set, ONLY these extensions can be written.  If None, the
        blocklist ``_BLOCKED_EXTENSIONS`` is used instead.
    max_writes_per_minute : int
        Rate limit for write operations per agent per minute.
    max_commands_per_minute : int
        Rate limit for shell command execution per agent per minute.
    """

    def __init__(
        self,
        workspace_root: Path,
        *,
        mode: PermissionMode = PermissionMode.DEFAULT,
        allowed_extensions: set[str] | None = None,
        max_writes_per_minute: int = 60,
        max_commands_per_minute: int = 30,
    ) -> None:
        self._root = workspace_root.resolve()
        self._mode = mode
        self._allowed_extensions = allowed_extensions
        self._write_limiter = _RateLimiter(max_writes_per_minute, 60.0)
        self._command_limiter = _RateLimiter(max_commands_per_minute, 60.0)

    # ── Write permission ─────────────────────────────────────────────

    def check_write(self, path: str, *, agent_name: str = "unknown") -> PermissionResult:
        """Check whether an agent may write to *path*.

        Validates:
          1. Path resolves within workspace (no traversal)
          2. Extension is not blocked
          3. Write rate limit not exceeded
        """
        try:
            resolved = (self._root / path).resolve()
        except (ValueError, OSError) as exc:
            return PermissionResult(
                allowed=False,
                reason=f"Invalid path: {exc}",
                tool_name="write_file",
                path=path,
            )

        # 1. Path traversal check
        if not resolved.is_relative_to(self._root):
            return PermissionResult(
                allowed=False,
                reason=f"Path traversal blocked: {path} resolves outside workspace",
                tool_name="write_file",
                path=path,
            )

        # 2. Extension check
        ext = resolved.suffix.lower()
        if self._allowed_extensions is not None:
            if ext and ext not in self._allowed_extensions:
                return PermissionResult(
                    allowed=False,
                    reason=f"Extension '{ext}' not in allowed set",
                    tool_name="write_file",
                    path=path,
                )
        elif ext in _BLOCKED_EXTENSIONS:
            if self._mode == PermissionMode.PERMISSIVE:
                logger.warning("Permissive mode: allowing blocked extension %s for %s", ext, path)
            else:
                return PermissionResult(
                    allowed=False,
                    reason=f"Extension '{ext}' is blocked (secrets/binaries)",
                    tool_name="write_file",
                    path=path,
                )

        # 3. Rate limit
        if not self._write_limiter.check(agent_name):
            return PermissionResult(
                allowed=False,
                reason=f"Write rate limit exceeded for agent {agent_name}",
                tool_name="write_file",
                path=path,
            )

        return PermissionResult(allowed=True, tool_name="write_file", path=path)

    # ── Command permission ───────────────────────────────────────────

    def check_command(self, command: str, *, agent_name: str = "unknown") -> PermissionResult:
        """Check whether a shell command is safe to execute.

        Validates:
          1. Command does not match blocked patterns
          2. Command rate limit not exceeded
        """
        # 1. Pattern blocklist
        for pattern in _BLOCKED_COMMAND_PATTERNS:
            if pattern.search(command):
                if self._mode == PermissionMode.PERMISSIVE:
                    logger.warning("Permissive mode: allowing blocked command pattern for: %s", command[:100])
                else:
                    return PermissionResult(
                        allowed=False,
                        reason=f"Blocked command pattern: {pattern.pattern}",
                        tool_name="execute_command",
                        path="",
                    )

        # 2. Rate limit
        if not self._command_limiter.check(agent_name):
            return PermissionResult(
                allowed=False,
                reason=f"Command rate limit exceeded for agent {agent_name}",
                tool_name="execute_command",
                path="",
            )

        return PermissionResult(allowed=True, tool_name="execute_command")

    # ── Read permission (lightweight — only path traversal) ──────────

    def check_read(self, path: str) -> PermissionResult:
        """Check whether an agent may read *path*."""
        try:
            resolved = (self._root / path).resolve()
        except (ValueError, OSError) as exc:
            return PermissionResult(
                allowed=False,
                reason=f"Invalid path: {exc}",
                tool_name="read_file",
                path=path,
            )

        if not resolved.is_relative_to(self._root):
            return PermissionResult(
                allowed=False,
                reason=f"Path traversal blocked: {path} resolves outside workspace",
                tool_name="read_file",
                path=path,
            )

        return PermissionResult(allowed=True, tool_name="read_file", path=path)

    # ── Convenience ──────────────────────────────────────────────────

    def require_write(self, path: str, *, agent_name: str = "unknown") -> None:
        """Check write permission; raise ``PermissionDeniedError`` if denied."""
        result = self.check_write(path, agent_name=agent_name)
        if not result.allowed:
            raise PermissionDeniedError(
                result.reason,
                tool_name="write_file",
                path=path,
            )

    def require_command(self, command: str, *, agent_name: str = "unknown") -> None:
        """Check command permission; raise ``PermissionDeniedError`` if denied."""
        result = self.check_command(command, agent_name=agent_name)
        if not result.allowed:
            raise PermissionDeniedError(
                result.reason,
                tool_name="execute_command",
                path="",
            )

    def reset_rate_limits(self) -> None:
        """Reset all rate limit counters (useful between pipeline phases)."""
        self._write_limiter.reset()
        self._command_limiter.reset()
