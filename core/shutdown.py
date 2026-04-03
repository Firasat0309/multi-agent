"""Graceful shutdown handler — tears down active sandboxes on SIGTERM/SIGINT.

Usage (in cli.py ``main()``)::

    from core.shutdown import install_signal_handlers, register_resource
    install_signal_handlers()
    ...
    register_resource(sandbox_orchestrator)  # any object with async teardown()
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class Teardownable(Protocol):
    """Any object that exposes an async ``teardown()`` method."""

    async def teardown(self) -> None: ...


# ── Global registry of resources that need cleanup ───────────────────────────

_resources: list[Teardownable] = []
_shutting_down = False


def register_resource(resource: Teardownable) -> None:
    """Register a resource that must be torn down on shutdown."""
    _resources.append(resource)


def unregister_resource(resource: Teardownable) -> None:
    """Remove a previously registered resource (e.g. after normal cleanup)."""
    try:
        _resources.remove(resource)
    except ValueError:
        pass


async def _teardown_all() -> None:
    """Destroy every registered resource, logging but not raising errors."""
    for resource in reversed(_resources):
        try:
            logger.info("Shutting down %s …", type(resource).__name__)
            await resource.teardown()
        except Exception:
            logger.warning("Error tearing down %s", type(resource).__name__, exc_info=True)
    _resources.clear()


def _handle_signal(signum: int, _frame: object) -> None:
    """Signal callback — runs teardown in the current or a new event loop."""
    global _shutting_down
    if _shutting_down:
        # Second signal → force exit
        logger.warning("Received second signal, forcing exit")
        sys.exit(128 + signum)

    _shutting_down = True
    sig_name = signal.Signals(signum).name
    logger.warning("Received %s — tearing down resources …", sig_name)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We're inside an async context — schedule teardown as a task
        loop.create_task(_teardown_and_exit(signum))
    else:
        # No running loop — create one just for teardown
        asyncio.run(_teardown_all())
        sys.exit(128 + signum)


async def _teardown_and_exit(signum: int) -> None:
    await _teardown_all()
    sys.exit(128 + signum)


def install_signal_handlers() -> None:
    """Register SIGTERM and SIGINT handlers.

    Safe to call multiple times — subsequent calls are no-ops.
    On Windows, only SIGINT (Ctrl+C) is supported; SIGTERM is
    registered only on Unix-like systems.
    """
    signal.signal(signal.SIGINT, _handle_signal)
    if sys.platform != "win32":
        signal.signal(signal.SIGTERM, _handle_signal)
    logger.debug("Graceful shutdown handlers installed")
