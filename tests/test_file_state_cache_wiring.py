"""Tests for FileStateCache wiring into BaseAgent tool handlers.

Verifies that:
1. FileStateCache unit behavior — read, write, cache hits, eviction
2. BaseAgent._tool_read_file uses cache when FILE_STATE_CACHE is enabled
3. BaseAgent._tool_write_file updates cache on write
4. Cache is disabled (None) when feature flag is off
5. Write-through: write then read returns new content without disk I/O
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.file_state_cache import FileStateCache


def _run(coro):
    return asyncio.run(coro)


# ── FileStateCache unit tests ────────────────────────────────────────────────


class FakeRepo:
    """Minimal async repo stub for testing."""

    def __init__(self, files: dict[str, str] | None = None):
        self.files = dict(files or {})
        self.read_count = 0
        self.write_count = 0

    async def async_read_file(self, path: str) -> str | None:
        self.read_count += 1
        return self.files.get(path)

    async def async_write_file(self, path: str, content: str) -> None:
        self.write_count += 1
        self.files[path] = content


class TestFileStateCacheUnit:
    def test_read_miss_then_hit(self):
        repo = FakeRepo({"a.py": "print('hello')"})
        cache = FileStateCache()

        # First read — cache miss, hits disk
        content = _run(cache.read(repo, "a.py"))
        assert content == "print('hello')"
        assert repo.read_count == 1
        assert cache.stats["misses"] == 1

        # Second read — cache hit, no disk I/O
        content = _run(cache.read(repo, "a.py"))
        assert content == "print('hello')"
        assert repo.read_count == 1  # still 1
        assert cache.stats["hits"] == 1

    def test_read_nonexistent_file(self):
        repo = FakeRepo({})
        cache = FileStateCache()
        content = _run(cache.read(repo, "missing.py"))
        assert content is None
        assert cache.stats["entries"] == 0

    def test_write_through(self):
        repo = FakeRepo({})
        cache = FileStateCache()

        _run(cache.write(repo, "b.py", "class B: pass"))
        assert repo.write_count == 1
        assert repo.files["b.py"] == "class B: pass"

        # Read should come from cache, not disk
        content = _run(cache.read(repo, "b.py"))
        assert content == "class B: pass"
        assert repo.read_count == 0  # never hit disk

    def test_write_updates_cache(self):
        repo = FakeRepo({"c.py": "v1"})
        cache = FileStateCache()

        _run(cache.read(repo, "c.py"))  # populate cache
        _run(cache.write(repo, "c.py", "v2"))  # update

        content = _run(cache.read(repo, "c.py"))
        assert content == "v2"
        assert repo.read_count == 1  # only the initial read hit disk

    def test_invalidate(self):
        repo = FakeRepo({"d.py": "original"})
        cache = FileStateCache()

        _run(cache.read(repo, "d.py"))
        cache.invalidate("d.py")

        # After invalidation, must hit disk again
        repo.files["d.py"] = "modified on disk"
        content = _run(cache.read(repo, "d.py"))
        assert content == "modified on disk"
        assert repo.read_count == 2

    def test_clear(self):
        repo = FakeRepo({"e.py": "x", "f.py": "y"})
        cache = FileStateCache()

        _run(cache.read(repo, "e.py"))
        _run(cache.read(repo, "f.py"))
        assert cache.stats["entries"] == 2

        cache.clear()
        assert cache.stats["entries"] == 0

    def test_eviction_by_max_entries(self):
        repo = FakeRepo({f"{i}.py": f"content{i}" for i in range(5)})
        cache = FileStateCache(max_entries=3)

        for i in range(5):
            _run(cache.read(repo, f"{i}.py"))

        assert cache.stats["entries"] <= 3

    def test_sync_get(self):
        repo = FakeRepo({"g.py": "sync test"})
        cache = FileStateCache()

        # Before any read, sync get returns None
        assert cache.get("g.py") is None

        _run(cache.read(repo, "g.py"))
        assert cache.get("g.py") == "sync test"

    def test_stats_hit_rate(self):
        repo = FakeRepo({"h.py": "stats"})
        cache = FileStateCache()

        _run(cache.read(repo, "h.py"))  # miss
        _run(cache.read(repo, "h.py"))  # hit
        _run(cache.read(repo, "h.py"))  # hit

        stats = cache.stats
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate_pct"] == 66  # 2/3


# ── BaseAgent integration ────────────────────────────────────────────────────


class TestBaseAgentCacheIntegration:
    """Test that BaseAgent _tool_read_file and _tool_write_file use the cache."""

    def _make_agent(self, cache_enabled: bool):
        """Create a minimal BaseAgent with mocked dependencies."""
        from agents.base_agent import BaseAgent
        from core.models import AgentRole

        with patch.multiple(
            "agents.base_agent",
            feature=lambda name: name == "FILE_STATE_CACHE" if cache_enabled else False,
        ):
            # Create a concrete subclass since BaseAgent is abstract
            class TestAgent(BaseAgent):
                role = AgentRole.CODER
                async def execute(self, context):
                    pass

            llm = MagicMock()
            repo = MagicMock()
            repo.workspace = MagicMock()
            agent = TestAgent(llm, repo)

        # Restore real feature for runtime (the flag was only needed for __init__)
        return agent

    def test_cache_created_when_flag_on(self):
        agent = self._make_agent(cache_enabled=True)
        assert agent._file_cache is not None
        assert isinstance(agent._file_cache, FileStateCache)

    def test_cache_none_when_flag_off(self):
        agent = self._make_agent(cache_enabled=False)
        assert agent._file_cache is None

    def test_read_uses_cache(self):
        agent = self._make_agent(cache_enabled=True)
        repo = FakeRepo({"src/main.py": "hello world\nline 2"})
        agent.repo = repo

        result = _run(agent._tool_read_file({"path": "src/main.py"}))
        assert "hello world" in result
        assert repo.read_count == 1

        # Second read — should hit cache
        result2 = _run(agent._tool_read_file({"path": "src/main.py"}))
        assert "hello world" in result2
        assert repo.read_count == 1  # no new disk read

    def test_read_bypasses_cache_when_disabled(self):
        agent = self._make_agent(cache_enabled=False)
        repo = FakeRepo({"src/main.py": "content"})
        agent.repo = repo

        _run(agent._tool_read_file({"path": "src/main.py"}))
        _run(agent._tool_read_file({"path": "src/main.py"}))
        assert repo.read_count == 2  # both hit disk

    def test_write_updates_cache(self):
        agent = self._make_agent(cache_enabled=True)
        repo = FakeRepo({})
        agent.repo = repo

        result = _run(agent._tool_write_file({"path": "out.py", "content": "new code"}))
        assert "Written" in result
        assert repo.write_count == 1

        # Read should come from cache
        read_result = _run(agent._tool_read_file({"path": "out.py"}))
        assert "new code" in read_result
        assert repo.read_count == 0  # no disk read needed

    def test_write_through_consistency(self):
        """Write then read returns the written content, not stale cache."""
        agent = self._make_agent(cache_enabled=True)
        repo = FakeRepo({"x.py": "version 1"})
        agent.repo = repo

        # Populate cache with v1
        _run(agent._tool_read_file({"path": "x.py"}))

        # Write v2 — should update cache
        _run(agent._tool_write_file({"path": "x.py", "content": "version 2"}))

        # Read should return v2
        result = _run(agent._tool_read_file({"path": "x.py"}))
        assert "version 2" in result
        assert repo.read_count == 1  # only the initial read hit disk
