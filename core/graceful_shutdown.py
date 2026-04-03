"""Graceful shutdown and optional health-check HTTP endpoint.

Installs SIGINT/SIGTERM handlers that set a cancellation event so the
pipeline can drain in-flight work instead of crashing mid-generation.

Usage from the CLI entry point::

    from core.graceful_shutdown import install_shutdown_handlers, shutdown_requested

    install_shutdown_handlers()

    # In your event loop:
    if shutdown_requested():
        # stop dispatching new work, let running tasks finish
        ...
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
import threading
from typing import Any

logger = logging.getLogger(__name__)

# Module-level event: set when a termination signal is received.
_shutdown_event = threading.Event()

# asyncio-aware event (created lazily on first use in an event loop).
_async_shutdown_event: asyncio.Event | None = None


def shutdown_requested() -> bool:
    """Return True if a SIGINT/SIGTERM has been received."""
    return _shutdown_event.is_set()


def get_async_shutdown_event() -> asyncio.Event:
    """Return an asyncio.Event that is set when shutdown is requested.

    Must be called from within a running event loop.
    """
    global _async_shutdown_event
    if _async_shutdown_event is None:
        _async_shutdown_event = asyncio.Event()
    return _async_shutdown_event


def _signal_handler(signum: int, frame: Any) -> None:
    sig_name = signal.Signals(signum).name
    logger.warning("Received %s — requesting graceful shutdown", sig_name)
    _shutdown_event.set()
    if _async_shutdown_event is not None:
        # Thread-safe set from signal handler context.
        _async_shutdown_event.set()


def install_shutdown_handlers() -> None:
    """Install SIGINT and SIGTERM handlers for graceful shutdown.

    Safe to call multiple times. On Windows only SIGINT is available.
    """
    signal.signal(signal.SIGINT, _signal_handler)
    if sys.platform != "win32":
        signal.signal(signal.SIGTERM, _signal_handler)
    logger.debug("Graceful shutdown handlers installed")


# ── Minimal health-check server ──────────────────────────────────────────────

async def start_health_server(port: int = 8099) -> asyncio.Server | None:
    """Start a tiny HTTP health-check server on *port*.

    Responds to ``GET /healthz`` with ``200 OK`` (or ``503`` if shutdown
    is requested).  Returns the server object so the caller can close it
    on exit. Returns ``None`` if the port is unavailable.
    """
    async def _handle(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
    ) -> None:
        try:
            data = await asyncio.wait_for(reader.read(4096), timeout=5)
            request_line = data.split(b"\r\n", 1)[0].decode(errors="replace")

            if "GET /healthz" in request_line:
                if shutdown_requested():
                    body = b"shutting down"
                    status = "503 Service Unavailable"
                else:
                    body = b"ok"
                    status = "200 OK"
            else:
                body = b"not found"
                status = "404 Not Found"

            response = (
                f"HTTP/1.1 {status}\r\n"
                f"Content-Length: {len(body)}\r\n"
                f"Content-Type: text/plain\r\n\r\n"
            ).encode() + body
            writer.write(response)
            await writer.drain()
        except Exception:
            pass
        finally:
            writer.close()

    try:
        server = await asyncio.start_server(_handle, "127.0.0.1", port)
        logger.info("Health-check server listening on http://127.0.0.1:%d/healthz", port)
        return server
    except OSError:
        logger.debug("Could not bind health-check port %d — skipping", port)
        return None
