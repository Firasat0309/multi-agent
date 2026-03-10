"""Per-file asyncio locks to serialize concurrent writes.

Shared across an AgentManager instance — all agents writing to the
same file will queue rather than race.

Usage::

    manager = FileLockManager()

    async with manager.lock_for("src/foo.py"):
        # safe to write to src/foo.py
        ...
"""

from __future__ import annotations

import asyncio
from collections import defaultdict


class FileLockManager:
    """Per-file asyncio locks to serialize concurrent writes.

    ``defaultdict(asyncio.Lock)`` creates a fresh lock for each new path on
    first access.  The lock is never deleted — this is intentional because
    the number of distinct file paths in any repository is bounded and small
    compared to the memory cost of keeping the locks alive for the run.
    """

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    def lock_for(self, file_path: str) -> asyncio.Lock:
        """Return (creating if necessary) the asyncio.Lock for *file_path*."""
        return self._locks[file_path]
