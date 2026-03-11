"""Lightweight async event bus — lets agents react to each other's completions.

Agents subscribe to one or more ``BusEventType`` values and receive an
``AgentEvent`` payload whenever a matching event is published.  All
exceptions raised inside handlers are caught and logged so that a faulty
subscriber never crashes the publisher.

Usage::

    bus = EventBus()

    async def on_file_written(event: AgentEvent) -> None:
        print(f"File written: {event.file_path}")

    bus.subscribe(BusEventType.FILE_WRITTEN, on_file_written)
    await bus.publish(AgentEvent(type=BusEventType.FILE_WRITTEN, file_path="src/foo.py"))
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)

Handler = Callable[["AgentEvent"], Awaitable[None]]


class BusEventType(str, Enum):
    """All event types published on the agent event bus."""

    # File-level events
    FILE_WRITTEN = "file_written"

    # Task-level events (DAG and lifecycle)
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"

    # Review events
    REVIEW_PASSED = "review_passed"
    REVIEW_FAILED = "review_failed"

    # Build checkpoint events
    BUILD_PASSED = "build_passed"
    BUILD_FAILED = "build_failed"

    # Test events
    TEST_PASSED = "test_passed"
    TEST_FAILED = "test_failed"


@dataclass
class AgentEvent:
    """Payload carried by every bus event."""

    type: BusEventType
    task_id: int | None = None
    task_type: str | None = None
    file_path: str | None = None
    agent_name: str | None = None
    # Arbitrary extra data agents may attach (errors, metrics, …)
    data: dict = field(default_factory=dict)


class EventBus:
    """Async pub-sub hub shared across all agents in a pipeline run.

    *subscribe* — register a coroutine handler for one event type.
    *subscribe_all* — register a handler for every event type.
    *publish* — fire an event; all matching handlers are awaited concurrently.
    """

    def __init__(self) -> None:
        self._handlers: dict[BusEventType, list[Handler]] = defaultdict(list)
        self._catch_all: list[Handler] = []

    # ── Subscription ─────────────────────────────────────────────────────────

    def subscribe(self, event_type: BusEventType, handler: Handler) -> None:
        """Register *handler* to be called whenever *event_type* is published."""
        self._handlers[event_type].append(handler)

    def subscribe_all(self, handler: Handler) -> None:
        """Register *handler* to be called for every published event."""
        self._catch_all.append(handler)

    # ── Publishing ───────────────────────────────────────────────────────────

    async def publish(self, event: AgentEvent) -> None:
        """Publish *event* to all matching subscribers.

        Handlers for the specific event type and catch-all handlers are run
        concurrently.  Any exception raised by a handler is logged but does
        not propagate — a broken subscriber never affects other subscribers or
        the publisher.
        """
        handlers = self._handlers[event.type] + self._catch_all
        if not handlers:
            return

        async def _safe(h: Handler) -> None:
            try:
                await h(event)
            except Exception:
                logger.exception(
                    "EventBus handler %s raised for event %s",
                    getattr(h, "__qualname__", repr(h)),
                    event.type.value,
                )

        await asyncio.gather(*[_safe(h) for h in handlers])
