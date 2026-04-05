"""Pipeline hook system for extensibility.

Inspired by Claude Code's 15+ hook events.  Hooks allow external code
to observe and react to pipeline events without modifying core logic.

Hook events fire at key lifecycle points:
  - Pipeline start/end
  - Agent execution start/end
  - File write (pre/post)
  - Build checkpoint (pre/post)
  - Test execution (pre/post)
  - Tier completion

Handlers can be sync or async.  Exceptions in handlers are logged but
never propagate to the caller (hooks must not break the pipeline).

Usage::

    from core.hooks import HookEvent, HookRegistry

    hooks = HookRegistry()

    @hooks.on(HookEvent.POST_FILE_WRITE)
    async def log_write(file_path: str, **kwargs):
        print(f"File written: {file_path}")

    # In pipeline code:
    await hooks.fire(HookEvent.POST_FILE_WRITE, file_path="src/main.py", content="...")
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

__all__ = ["HookEvent", "HookRegistry", "HookPlugin"]


# ── Hook events ──────────────────────────────────────────────────────────────

class HookEvent(StrEnum):
    """All events that support hook handlers."""

    # Pipeline lifecycle
    PIPELINE_START = "pipeline_start"
    PIPELINE_END = "pipeline_end"
    PIPELINE_ERROR = "pipeline_error"

    # Agent execution
    PRE_AGENT_EXECUTE = "pre_agent_execute"
    POST_AGENT_EXECUTE = "post_agent_execute"
    AGENT_ERROR = "agent_error"

    # File operations
    PRE_FILE_WRITE = "pre_file_write"
    POST_FILE_WRITE = "post_file_write"
    FILE_READ = "file_read"

    # Build & test
    PRE_BUILD_CHECKPOINT = "pre_build_checkpoint"
    POST_BUILD_CHECKPOINT = "post_build_checkpoint"
    PRE_TEST = "pre_test"
    POST_TEST = "post_test"

    # Tier scheduling
    TIER_START = "tier_start"
    TIER_COMPLETE = "tier_complete"

    # Review
    REVIEW_COMPLETE = "review_complete"
    REVIEW_FIX_START = "review_fix_start"

    # Session
    SESSION_CHECKPOINT = "session_checkpoint"

    # Cost
    COST_UPDATE = "cost_update"


# ── Handler types ────────────────────────────────────────────────────────────

# Handlers receive keyword arguments matching the event's payload.
HookHandler = Callable[..., Any]
AsyncHookHandler = Callable[..., Awaitable[Any]]


@dataclass
class HookResult:
    """Result of firing a hook event."""
    event: HookEvent
    handlers_run: int
    handlers_failed: int
    elapsed_ms: float
    errors: list[str] = field(default_factory=list)


# ── Registry ─────────────────────────────────────────────────────────────────

class HookRegistry:
    """Central registry for pipeline hook handlers.

    Handlers are registered for specific events and called when that event
    fires.  Both sync and async handlers are supported.
    """

    def __init__(self) -> None:
        self._handlers: dict[HookEvent, list[HookHandler | AsyncHookHandler]] = defaultdict(list)

    # ── Registration ─────────────────────────────────────────────────

    def register(self, event: HookEvent, handler: HookHandler | AsyncHookHandler) -> Callable[[], None]:
        """Register a handler for an event.

        Returns an unsubscribe function.
        """
        self._handlers[event].append(handler)
        def unsubscribe() -> None:
            try:
                self._handlers[event].remove(handler)
            except ValueError:
                pass
        return unsubscribe

    def on(self, event: HookEvent) -> Callable:
        """Decorator to register a handler for an event.

        Usage::

            @hooks.on(HookEvent.POST_FILE_WRITE)
            async def on_write(file_path: str, **kwargs):
                ...
        """
        def decorator(fn: HookHandler | AsyncHookHandler) -> HookHandler | AsyncHookHandler:
            self.register(event, fn)
            return fn
        return decorator

    # ── Firing ───────────────────────────────────────────────────────

    async def fire(self, event: HookEvent, **kwargs: Any) -> HookResult:
        """Fire an event, calling all registered handlers.

        Handlers are called concurrently.  Exceptions are caught and logged
        but never propagate — hooks must not break the pipeline.
        """
        handlers = self._handlers.get(event, [])
        if not handlers:
            return HookResult(event=event, handlers_run=0, handlers_failed=0, elapsed_ms=0.0)

        start = time.monotonic()
        failed = 0
        errors: list[str] = []

        async def _run_handler(handler: HookHandler | AsyncHookHandler) -> None:
            nonlocal failed
            try:
                result = handler(**kwargs)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:
                failed += 1
                error_msg = f"{getattr(handler, '__qualname__', repr(handler))}: {exc}"
                errors.append(error_msg)
                logger.warning(
                    "Hook handler failed for %s: %s",
                    event.value, error_msg,
                )

        await asyncio.gather(*[_run_handler(h) for h in handlers])

        elapsed = (time.monotonic() - start) * 1000
        return HookResult(
            event=event,
            handlers_run=len(handlers),
            handlers_failed=failed,
            elapsed_ms=elapsed,
            errors=errors,
        )

    # ── Utilities ────────────────────────────────────────────────────

    def has_handlers(self, event: HookEvent) -> bool:
        """Check if any handlers are registered for an event."""
        return bool(self._handlers.get(event))

    def handler_count(self, event: HookEvent) -> int:
        """Number of handlers registered for an event."""
        return len(self._handlers.get(event, []))

    def clear(self, event: HookEvent | None = None) -> None:
        """Remove all handlers (or handlers for a specific event)."""
        if event is None:
            self._handlers.clear()
        else:
            self._handlers.pop(event, None)

    # ── Plugin loading ───────────────────────────────────────────────

    def load_plugin(self, plugin: "HookPlugin") -> None:
        """Register all hooks declared by a plugin.

        A plugin is any object with ``hook_events`` returning an iterable
        of ``(event, handler)`` pairs.  This allows packaging related hooks
        into a single unit for clean enable/disable.
        """
        for event, handler in plugin.hook_events():
            self.register(event, handler)
        logger.info("Loaded plugin: %s", plugin.name)

    def fire_sync(self, event: HookEvent, **kwargs: Any) -> HookResult:
        """Synchronous variant of ``fire()`` for non-async contexts.

        Runs handlers in a new event loop if one is not already running.
        Falls back to direct sync calls if no event loop exists.
        """
        handlers = self._handlers.get(event, [])
        if not handlers:
            return HookResult(event=event, handlers_run=0, handlers_failed=0, elapsed_ms=0.0)

        start = time.monotonic()
        failed = 0
        errors: list[str] = []

        for handler in handlers:
            try:
                result = handler(**kwargs)
                if asyncio.iscoroutine(result):
                    result.close()  # discard — can't await in sync context
            except Exception as exc:
                failed += 1
                errors.append(f"{getattr(handler, '__qualname__', repr(handler))}: {exc}")

        elapsed = (time.monotonic() - start) * 1000
        return HookResult(event=event, handlers_run=len(handlers), handlers_failed=failed, elapsed_ms=elapsed, errors=errors)


# ── Plugin protocol ──────────────────────────────────────────────────────────

class HookPlugin:
    """Base class for hook plugins.

    Subclass and implement ``hook_events()`` to return the hooks this
    plugin provides.

    Usage::

        class MetricsPlugin(HookPlugin):
            name = "metrics"

            def hook_events(self):
                return [
                    (HookEvent.POST_AGENT_EXECUTE, self._record_agent),
                    (HookEvent.POST_BUILD_CHECKPOINT, self._record_build),
                ]

            async def _record_agent(self, **kwargs):
                ...

        hooks = HookRegistry()
        hooks.load_plugin(MetricsPlugin())
    """
    name: str = "unnamed"

    def hook_events(self) -> list[tuple[HookEvent, HookHandler | AsyncHookHandler]]:
        """Return (event, handler) pairs for this plugin."""
        return []

    def reset(self) -> None:
        """Remove ALL handlers for ALL events (for testing)."""
        self._handlers.clear()
