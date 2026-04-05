"""Tests for improvements: retry strategies, fix loop, compaction, permissions,
hooks, plugin loader, shared context, and embedding pruning.
"""

from __future__ import annotations

import asyncio
import hashlib
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ════════════════════════════════════════════════════════════════════════════
# 1. Error-specific LLM retry strategies
# ════════════════════════════════════════════════════════════════════════════


class TestRetryStrategies:
    """Tests for RETRY_STRATEGIES and _extract_retry_after."""

    def test_retry_strategies_cover_all_categories(self):
        from core.llm_client import LLMErrorCategory, RETRY_STRATEGIES

        for cat in LLMErrorCategory:
            assert cat in RETRY_STRATEGIES, f"Missing strategy for {cat.value}"

    def test_auth_errors_abort(self):
        from core.llm_client import LLMErrorCategory, RETRY_STRATEGIES

        assert RETRY_STRATEGIES[LLMErrorCategory.AUTH_FAILED].abort is True
        assert RETRY_STRATEGIES[LLMErrorCategory.AUTH_FAILED].max_retries == 0

    def test_rate_limit_uses_retry_after(self):
        from core.llm_client import LLMErrorCategory, RETRY_STRATEGIES

        strategy = RETRY_STRATEGIES[LLMErrorCategory.RATE_LIMITED]
        assert strategy.use_retry_after is True
        assert strategy.max_retries > 3  # More patient for rate limits

    def test_context_length_triggers_compaction(self):
        from core.llm_client import LLMErrorCategory, RETRY_STRATEGIES

        strategy = RETRY_STRATEGIES[LLMErrorCategory.CONTEXT_LENGTH]
        assert strategy.trigger_compaction is True

    def test_quota_exhausted_aborts(self):
        from core.llm_client import LLMErrorCategory, RETRY_STRATEGIES

        assert RETRY_STRATEGIES[LLMErrorCategory.QUOTA_EXCEEDED].abort is True

    def test_extract_retry_after_from_header(self):
        from core.llm_client import _extract_retry_after

        class FakeResponse:
            headers = {"retry-after": "30"}

        class FakeError(Exception):
            response = FakeResponse()

        result = _extract_retry_after(FakeError("rate limited"))
        assert result == 30.0

    def test_extract_retry_after_from_message(self):
        from core.llm_client import _extract_retry_after

        err = Exception("Please retry after 5.0 seconds")
        result = _extract_retry_after(err)
        assert result == 5.0

    def test_extract_retry_after_none_when_missing(self):
        from core.llm_client import _extract_retry_after

        result = _extract_retry_after(Exception("Some other error"))
        assert result is None

    def test_classify_llm_error_categories(self):
        from core.llm_client import classify_llm_error, LLMErrorCategory

        assert classify_llm_error("429 too many requests") == LLMErrorCategory.RATE_LIMITED
        assert classify_llm_error("401 Unauthorized") == LLMErrorCategory.AUTH_FAILED
        assert classify_llm_error("model not_found") == LLMErrorCategory.MODEL_NOT_FOUND
        assert classify_llm_error("context_length exceeded") == LLMErrorCategory.CONTEXT_LENGTH


# ════════════════════════════════════════════════════════════════════════════
# 2. FixLoop
# ════════════════════════════════════════════════════════════════════════════


class TestFixLoop:
    """Tests for the extracted FixLoop class."""

    def test_record_error(self):
        from core.fix_loop import FixLoop

        fl = FixLoop("src/main.java")
        rec = fl.record_error("cannot find symbol", attempt=1)
        assert rec.attempt == 1
        assert rec.error_hash == hashlib.md5(b"cannot find symbol").hexdigest()[:8]
        assert len(fl.error_history) == 1

    def test_not_stalled_with_single_error(self):
        from core.fix_loop import FixLoop

        fl = FixLoop("src/main.java")
        fl.record_error("error A", attempt=1)
        assert not fl.is_stalled()

    def test_stalled_on_same_hash(self):
        from core.fix_loop import FixLoop

        fl = FixLoop("src/main.java")
        fl.record_error("error A", attempt=1)
        fl.record_error("error A", attempt=2)
        assert fl.is_stalled()

    def test_stalled_on_oscillation(self):
        from core.fix_loop import FixLoop

        fl = FixLoop("src/main.java")
        fl.record_error("error A", attempt=1)
        fl.record_error("error B", attempt=2)
        fl.record_error("error A", attempt=3)
        assert fl.is_stalled()

    def test_escalation_prefix_level_1(self):
        from core.fix_loop import FixLoop

        fl = FixLoop("src/main.java")
        fl.record_error("error A", attempt=1)
        fl.record_error("error A", attempt=2)
        prefix = fl.get_escalation_prefix()
        assert "COMPLETELY DIFFERENT approach" in prefix

    def test_escalation_prefix_level_2(self):
        from core.fix_loop import FixLoop

        fl = FixLoop("src/main.java")
        for i in range(4):
            fl.record_error("error A", attempt=i)
        prefix = fl.get_escalation_prefix()
        assert "REWRITE" in prefix or "STALLED" in prefix

    def test_build_fix_metadata_includes_history(self):
        from core.fix_loop import FixLoop

        fl = FixLoop("src/main.java")
        fl.record_error("error A", attempt=1)
        fl.record_error("error B", attempt=2)
        meta = fl.build_fix_metadata()
        assert "build_errors" in meta
        assert "error B" in meta["build_errors"]

    def test_record_review_findings(self):
        from core.fix_loop import FixLoop

        fl = FixLoop("src/main.java")
        rec = fl.record_review_findings(["null deref", "missing validation"], attempt=0)
        assert "CODE REVIEW" in rec.errors_text
        assert "null deref" in rec.errors_text


# ════════════════════════════════════════════════════════════════════════════
# 3. Context compaction — file-state restoration
# ════════════════════════════════════════════════════════════════════════════


class TestCompactionRestoration:
    def test_restore_file_state_injects_message(self):
        from core.context_compaction import compact_messages, restore_file_state

        messages = [
            {"role": "user", "content": "Generate the service layer"},
        ]
        # Simulate a compacted history with a marker
        for i in range(20):
            messages.append({"role": "assistant", "content": f"Working on step {i}..."})
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": f"Written 500 bytes to src/service_{i}.py\n"}],
            })

        compacted = compact_messages(messages, char_budget=500, keep_tail=4)
        assert any("messages removed" in str(m.get("content", "")) for m in compacted)

        def read_fn(path: str) -> str | None:
            if "service" in path:
                return "class Service:\n    pass\n"
            return None

        restored = restore_file_state(compacted, read_fn)
        assert len(restored) >= len(compacted)
        # Should have a file restoration message
        restoration_msgs = [
            m for m in restored
            if isinstance(m.get("content"), str) and "File context restoration" in m["content"]
        ]
        assert len(restoration_msgs) == 1

    def test_restore_noop_when_no_compaction(self):
        from core.context_compaction import restore_file_state

        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        result = restore_file_state(messages, lambda p: None)
        assert result == messages

    def test_extract_recent_file_paths(self):
        from core.context_compaction import _extract_recent_file_paths

        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Written 200 bytes to src/app.py\n"}]},
            {"role": "user", "content": [{"type": "text", "text": "[File: src/model.py|Lines 1-10]"}]},
        ]
        paths = _extract_recent_file_paths(messages, max_files=5)
        assert "src/model.py" in paths or "src/app.py" in paths


# ════════════════════════════════════════════════════════════════════════════
# 4. Permission model integration
# ════════════════════════════════════════════════════════════════════════════


class TestPermissionIntegration:
    def test_command_permission_check(self):
        from pathlib import Path
        from core.permissions import ToolPermissionChecker

        checker = ToolPermissionChecker(workspace_root=Path("/tmp/project"))
        result = checker.check_command("rm -rf /")
        assert not result.allowed
        assert result.reason  # Should explain why it's blocked

    def test_write_permission_allowed(self):
        from pathlib import Path
        from core.permissions import ToolPermissionChecker

        checker = ToolPermissionChecker(workspace_root=Path("/tmp/project"))
        result = checker.check_write("src/main.py")
        assert result.allowed

    def test_write_permission_blocked_extension(self):
        from pathlib import Path
        from core.permissions import ToolPermissionChecker

        checker = ToolPermissionChecker(workspace_root=Path("/tmp/project"))
        result = checker.check_write("malware.exe")
        assert not result.allowed


# ════════════════════════════════════════════════════════════════════════════
# 5. Hook system wiring
# ════════════════════════════════════════════════════════════════════════════


class TestHookSystem:
    def test_hook_registry_fire(self):
        from core.hooks import HookEvent, HookRegistry

        registry = HookRegistry()
        fired = []

        @registry.on(HookEvent.TIER_START)
        async def on_tier(tier_index: int, **kwargs):
            fired.append(tier_index)

        result = asyncio.run(
            registry.fire(HookEvent.TIER_START, tier_index=3)
        )
        assert result.handlers_run == 1
        assert fired == [3]

    def test_hook_plugin_loading(self):
        from core.hooks import HookEvent, HookPlugin, HookRegistry

        class TestPlugin(HookPlugin):
            name = "test"

            def __init__(self):
                self.calls = []

            def hook_events(self):
                return [(HookEvent.POST_BUILD_CHECKPOINT, self._on_build)]

            async def _on_build(self, **kwargs):
                self.calls.append(kwargs)

        registry = HookRegistry()
        plugin = TestPlugin()
        registry.load_plugin(plugin)
        assert registry.handler_count(HookEvent.POST_BUILD_CHECKPOINT) == 1


# ════════════════════════════════════════════════════════════════════════════
# 6. Plugin Discovery
# ════════════════════════════════════════════════════════════════════════════


class TestPluginDiscovery:
    def test_discover_from_empty_dir(self, tmp_path):
        from core.plugin_loader import discover_plugins_from_directory

        plugins = discover_plugins_from_directory(tmp_path)
        assert plugins == []

    def test_discover_from_nonexistent_dir(self):
        from core.plugin_loader import discover_plugins_from_directory

        plugins = discover_plugins_from_directory("/nonexistent/path")
        assert plugins == []


# ════════════════════════════════════════════════════════════════════════════
# 7. Shared Context
# ════════════════════════════════════════════════════════════════════════════


class TestSharedContext:
    def test_add_and_get_insight(self):
        from core.shared_context import SharedContext

        ctx = SharedContext()
        ctx.add_insight("architect", "Using hexagonal architecture")
        insights = ctx.get_insights()
        assert len(insights) == 1
        assert insights[0].content == "Using hexagonal architecture"

    def test_add_and_get_decision(self):
        from core.shared_context import SharedContext

        ctx = SharedContext()
        ctx.add_decision("architect", "Use Spring Boot 3.2", reason="LTS support")
        decisions = ctx.get_decisions()
        assert len(decisions) == 1
        assert decisions[0].reason == "LTS support"

    def test_build_context_prompt(self):
        from core.shared_context import SharedContext

        ctx = SharedContext()
        ctx.add_decision("architect", "Use REST APIs")
        ctx.add_insight("planner", "10 files to generate")
        prompt = ctx.build_context_prompt(for_agent="coder")
        assert "REST APIs" in prompt
        assert "10 files" in prompt

    def test_excludes_self_insights(self):
        from core.shared_context import SharedContext

        ctx = SharedContext()
        ctx.add_insight("coder", "My own insight")
        ctx.add_insight("architect", "Other insight")
        prompt = ctx.build_context_prompt(for_agent="coder")
        assert "My own insight" not in prompt
        assert "Other insight" in prompt

    def test_file_notes(self):
        from core.shared_context import SharedContext

        ctx = SharedContext()
        ctx.add_file_note("src/Service.java", "Uses singleton pattern")
        notes = ctx.get_file_notes("src/Service.java")
        assert "Uses singleton pattern" in notes

    def test_serialization_roundtrip(self):
        from core.shared_context import SharedContext

        ctx = SharedContext()
        ctx.add_insight("agent1", "insight1")
        ctx.add_decision("agent2", "decision1", reason="reason1")
        ctx.add_file_note("a.py", "note1")

        data = ctx.to_dict()
        ctx2 = SharedContext.from_dict(data)
        assert len(ctx2.get_insights()) == 1
        assert len(ctx2.get_decisions()) == 1
        assert ctx2.get_file_notes("a.py") == ["note1"]

    def test_max_limits(self):
        from core.shared_context import SharedContext

        ctx = SharedContext(max_insights=5)
        for i in range(10):
            ctx.add_insight("agent", f"insight {i}")
        assert len(ctx.get_insights()) == 5


# ════════════════════════════════════════════════════════════════════════════
# 8. Embedding pruning
# ════════════════════════════════════════════════════════════════════════════


class TestEmbeddingPruning:
    def test_delete_file_returns_count(self):
        from memory.embedding_store import EmbeddingStore

        store = EmbeddingStore.__new__(EmbeddingStore)
        store._client = None
        store._collection = None
        store._persist_dir = "/tmp"
        store._collection_name = "test"
        store._embedding_model = "test"

        # With no client, should return 0
        result = store.delete_file("src/old.py")
        assert result == 0

    def test_prune_deleted_files(self):
        from memory.embedding_store import EmbeddingStore

        store = EmbeddingStore.__new__(EmbeddingStore)
        store._client = MagicMock()
        store._collection = MagicMock()
        store._persist_dir = "/tmp"
        store._collection_name = "test"
        store._embedding_model = "test"

        # Mock get_indexed_files to return stale + current files
        store._collection.get.side_effect = [
            # First call: get_indexed_files
            {"metadatas": [
                {"file": "src/current.py"},
                {"file": "src/deleted.py"},
                {"file": "src/also_deleted.py"},
            ]},
            # Second call: delete_file for "src/deleted.py"
            {"ids": ["src/deleted.py::chunk_0"]},
            # Third call: delete_file for "src/also_deleted.py"
            {"ids": ["src/also_deleted.py::chunk_0"]},
        ]

        existing = {"src/current.py"}
        pruned = store.prune_deleted_files(existing)
        assert "src/deleted.py" in pruned
        assert "src/also_deleted.py" in pruned
        assert "src/current.py" not in pruned

    def test_get_indexed_files_with_no_client(self):
        from memory.embedding_store import EmbeddingStore

        store = EmbeddingStore.__new__(EmbeddingStore)
        store._client = None
        store._collection = None
        store._persist_dir = "/tmp"
        store._collection_name = "test"
        store._embedding_model = "test"

        result = store.get_indexed_files()
        assert result == set()
