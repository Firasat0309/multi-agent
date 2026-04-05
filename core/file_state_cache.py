"""In-memory file state cache for the agentic tool-use loop.

During a single agent iteration, the same file may be read multiple times
(e.g. the agent reads a dependency, then later re-reads it after writing
the target file).  Without caching, each ``read_file`` tool call hits disk.

``FileStateCache`` provides a write-through cache: reads are served from
memory after the first disk hit, and writes update both disk and cache so
subsequent reads see the latest content instantly.

This mirrors how Claude Code's tool system caches file contents in the
query loop to avoid redundant FS reads during a single turn.

Usage::

    from core.file_state_cache import FileStateCache

    cache = FileStateCache()
    content = await cache.read(repo, "src/models/User.java")  # disk read
    content = await cache.read(repo, "src/models/User.java")  # cached

    await cache.write(repo, "src/models/User.java", new_content)  # writes + caches
    content = await cache.read(repo, "src/models/User.java")  # returns new_content
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class RepoReader(Protocol):
    """Minimal protocol for async file I/O."""
    async def async_read_file(self, path: str) -> str | None: ...
    async def async_write_file(self, path: str, content: str) -> None: ...


@dataclass(slots=True)
class CacheEntry:
    """A single cached file."""
    content: str
    size: int
    cached_at: float
    read_count: int = 0


class FileStateCache:
    """Write-through file content cache."""

    def __init__(self, max_entries: int = 500, max_total_bytes: int = 50 * 1024 * 1024) -> None:
        self._cache: dict[str, CacheEntry] = {}
        self._max_entries = max_entries
        self._max_total_bytes = max_total_bytes
        self._total_bytes = 0
        self._hits = 0
        self._misses = 0

    async def read(self, repo: RepoReader, path: str) -> str | None:
        """Read a file, returning cached content if available."""
        entry = self._cache.get(path)
        if entry is not None:
            entry.read_count += 1
            self._hits += 1
            return entry.content

        self._misses += 1
        content = await repo.async_read_file(path)
        if content is not None:
            self._put(path, content)
        return content

    async def write(self, repo: RepoReader, path: str, content: str) -> None:
        """Write a file to disk and update the cache."""
        await repo.async_write_file(path, content)
        self._put(path, content)

    def get(self, path: str) -> str | None:
        """Synchronous cache lookup (no disk I/O)."""
        entry = self._cache.get(path)
        if entry is not None:
            entry.read_count += 1
            self._hits += 1
            return entry.content
        return None

    def invalidate(self, path: str) -> None:
        """Remove a single entry from the cache."""
        entry = self._cache.pop(path, None)
        if entry:
            self._total_bytes -= entry.size

    def clear(self) -> None:
        """Clear the entire cache."""
        self._cache.clear()
        self._total_bytes = 0

    @property
    def stats(self) -> dict[str, int]:
        return {
            "entries": len(self._cache),
            "total_bytes": self._total_bytes,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_pct": int(self._hits / max(1, self._hits + self._misses) * 100),
        }

    def _put(self, path: str, content: str) -> None:
        """Add or update a cache entry, evicting if over capacity."""
        size = len(content)

        # Remove existing entry first (to update totals correctly)
        old = self._cache.pop(path, None)
        if old:
            self._total_bytes -= old.size

        # Evict LRU entries if over limits
        while (
            self._cache
            and (len(self._cache) >= self._max_entries or self._total_bytes + size > self._max_total_bytes)
        ):
            self._evict_one()

        self._cache[path] = CacheEntry(
            content=content,
            size=size,
            cached_at=time.monotonic(),
        )
        self._total_bytes += size

    def _evict_one(self) -> None:
        """Evict the least-recently-used entry."""
        if not self._cache:
            return
        # LRU by cached_at (oldest first)
        oldest_key = min(self._cache, key=lambda k: self._cache[k].cached_at)
        entry = self._cache.pop(oldest_key)
        self._total_bytes -= entry.size
