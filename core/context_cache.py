"""Shared context cache for tier-level deduplication.

When multiple files in the same tier share dependencies, each one was
independently re-building context (reading the same dep files, computing
the same embeddings).  This cache deduplicates those reads.

Typical savings: a 10-file tier where every file depends on 3 common
modules goes from 30 file reads → 3 cached reads + 7 fast hits.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class ContextCache:
    """In-memory cache for context fragments shared across a processing tier.

    Keyed by content hash so that the same file content is never read twice
    within the same tier execution.  Entries are automatically evicted after
    ``ttl_seconds`` (default: 300s / 5 minutes) to avoid stale reads when
    files are modified during the fix loop.
    """

    def __init__(self, *, ttl_seconds: float = 300.0) -> None:
        self._store: dict[str, tuple[float, Any]] = {}
        self._ttl = ttl_seconds
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        """Retrieve a cached value, or None if expired / absent."""
        entry = self._store.get(key)
        if entry is None:
            self._misses += 1
            return None
        ts, value = entry
        if time.monotonic() - ts > self._ttl:
            del self._store[key]
            self._misses += 1
            return None
        self._hits += 1
        return value

    def put(self, key: str, value: Any) -> None:
        """Store a value with the current timestamp."""
        self._store[key] = (time.monotonic(), value)

    def invalidate(self, key: str) -> None:
        """Remove a specific key (e.g. when a file is rewritten)."""
        self._store.pop(key, None)

    def clear(self) -> None:
        """Drop all cached entries (called between tiers)."""
        self._store.clear()

    @staticmethod
    def file_key(path: str, content: str) -> str:
        """Compute a cache key from file path + content hash."""
        h = hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()[:12]
        return f"{path}:{h}"

    @property
    def stats(self) -> dict[str, int]:
        return {"hits": self._hits, "misses": self._misses, "size": len(self._store)}
