"""Keyed async lock — replaces the ``dict[str, Lock] + guard Lock`` pattern.

The ``SimpleLoopExecutor`` (and similar code) needs per-key locking: e.g.
build locks grouped by module so independent modules can build concurrently.
The naive pattern requires a guard lock just to safely access the lock dict:

    async with self._guard:
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        lock = self._locks[key]
    async with lock:
        ...

``KeyedLock`` encapsulates this into a single reusable utility:

    async with self._build_locks.acquire(module_key):
        await self._run_build(file_path)

Inspired by Claude Code's internal ``KeyedMutex`` used for per-conversation
locking in the multi-agent swarm coordinator.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict


class KeyedLock:
    """Async lock pool keyed by arbitrary strings.

    Each key gets its own ``asyncio.Lock``.  Locks are created lazily and
    the internal dict lookup is protected by a lightweight guard lock.
    """

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._guard = asyncio.Lock()

    async def _get(self, key: str) -> asyncio.Lock:
        """Get or create the lock for *key*."""
        async with self._guard:
            if key not in self._locks:
                self._locks[key] = asyncio.Lock()
            return self._locks[key]

    def acquire(self, key: str) -> "_KeyedLockContext":
        """Return an async context manager that holds the lock for *key*."""
        return _KeyedLockContext(self, key)

    def locked(self, key: str) -> bool:
        """Check if the lock for *key* is currently held (non-blocking)."""
        lock = self._locks.get(key)
        return lock.locked() if lock else False

    @property
    def keys(self) -> list[str]:
        """Return all keys that have ever been locked."""
        return list(self._locks.keys())


class _KeyedLockContext:
    """Async context manager for ``KeyedLock.acquire()``."""

    def __init__(self, pool: KeyedLock, key: str) -> None:
        self._pool = pool
        self._key = key
        self._lock: asyncio.Lock | None = None

    async def __aenter__(self) -> None:
        self._lock = await self._pool._get(self._key)
        await self._lock.acquire()

    async def __aexit__(self, *exc: object) -> None:
        if self._lock is not None:
            self._lock.release()
