"""Tests for features added during the improvement passes.

Covers:
  - ExecutionConfig defaults and override
  - validate_tool_input (agent_tools)
  - _classify_error (llm_client)
  - BaseAgent._extract_code_block
  - BaseAgent._check_stagnation
  - DependencyGraphStore.get_layers with cycles
  - LocalSandbox timeout
  - Graceful shutdown (core/shutdown.py)
  - Language rules file loading (core/language_rules.py)
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from config.settings import ExecutionConfig


def _run(coro):
    return asyncio.run(coro)


# ── ExecutionConfig ──────────────────────────────────────────────────────────


class TestExecutionConfig:
    def test_defaults(self):
        cfg = ExecutionConfig()
        assert cfg.max_reverify_depth == 5
        assert cfg.max_context_files == 20
        assert cfg.request_timeout == 180
        assert cfg.retry_count == 4
        assert cfg.backoff_base == 2.0
        assert cfg.tool_timeout_seconds == 60.0
        assert cfg.circuit_failure_threshold == 5
        assert cfg.circuit_recovery_timeout == 60.0

    def test_frozen(self):
        cfg = ExecutionConfig()
        with pytest.raises(AttributeError):
            cfg.retry_count = 10  # type: ignore[misc]

    def test_override(self):
        from dataclasses import replace
        cfg = ExecutionConfig()
        cfg2 = replace(cfg, retry_count=8, backoff_base=1.0)
        assert cfg2.retry_count == 8
        assert cfg2.backoff_base == 1.0
        # original unchanged
        assert cfg.retry_count == 4


# ── validate_tool_input ──────────────────────────────────────────────────────


class TestValidateToolInput:
    def test_valid_read_file(self):
        from core.agent_tools import validate_tool_input
        assert validate_tool_input("read_file", {"path": "src/main.py"}) is None

    def test_missing_required_field(self):
        from core.agent_tools import validate_tool_input
        err = validate_tool_input("read_file", {})
        assert err is not None
        assert "path" in err

    def test_wrong_type(self):
        from core.agent_tools import validate_tool_input
        err = validate_tool_input("read_file", {"path": 123})
        assert err is not None
        assert "string" in err

    def test_write_file_missing_content(self):
        from core.agent_tools import validate_tool_input
        err = validate_tool_input("write_file", {"path": "a.py"})
        assert err is not None
        assert "content" in err

    def test_unknown_tool_passes(self):
        from core.agent_tools import validate_tool_input
        assert validate_tool_input("nonexistent_tool", {"foo": 1}) is None

    def test_extra_keys_tolerated(self):
        from core.agent_tools import validate_tool_input
        assert validate_tool_input("read_file", {"path": "a.py", "extra": True}) is None


# ── _classify_error ──────────────────────────────────────────────────────────


class TestClassifyError:
    def test_rate_limit_429(self):
        from core.llm_client import _classify_error
        is_rl, is_trans = _classify_error("Error 429 Too Many Requests")
        assert is_rl is True
        assert is_trans is True

    def test_rate_limit_keyword(self):
        from core.llm_client import _classify_error
        is_rl, _ = _classify_error("rate_limit_exceeded: slow down")
        assert is_rl is True

    def test_transient_500(self):
        from core.llm_client import _classify_error
        is_rl, is_trans = _classify_error("Internal Server Error 500")
        assert is_rl is False
        assert is_trans is True

    def test_transient_overloaded(self):
        from core.llm_client import _classify_error
        _, is_trans = _classify_error("The server is overloaded, try later")
        assert is_trans is True

    def test_transient_resource_exhausted(self):
        from core.llm_client import _classify_error
        _, is_trans = _classify_error("RESOURCE_EXHAUSTED: quota exceeded")
        assert is_trans is True

    def test_non_transient(self):
        from core.llm_client import _classify_error
        is_rl, is_trans = _classify_error("NotFound: model does not exist")
        assert is_rl is False
        assert is_trans is False


# ── _extract_code_block ──────────────────────────────────────────────────────


class TestExtractCodeBlock:
    @staticmethod
    def _extract(text):
        from agents.base_agent import BaseAgent
        return BaseAgent._extract_code_block(text)

    def test_no_fences(self):
        assert self._extract("just plain text") is None

    def test_short_block_ignored(self):
        assert self._extract("```python\nhi\n```") is None

    def test_valid_block(self):
        code = "x = 1\n" * 20  # > 50 chars
        text = f"Here is the code:\n```python\n{code}```\nDone."
        result = self._extract(text)
        assert result is not None
        assert "x = 1" in result

    def test_picks_longest(self):
        short = "a = 1\n"
        long_code = "def foo():\n    pass\n" * 10
        text = f"```\n{short}```\n\n```python\n{long_code}```"
        result = self._extract(text)
        assert "def foo" in result


# ── _check_stagnation ───────────────────────────────────────────────────────


class TestCheckStagnation:
    @staticmethod
    def _check(**kwargs):
        from agents.base_agent import BaseAgent
        return BaseAgent._check_stagnation(**kwargs)

    def test_write_resets(self):
        tc = MagicMock()
        tc.name = "write_file"
        count, stop = self._check(
            tool_calls=[tc],
            results=["OK"],
            stagnant_count=3,
            max_stagnant=5,
            agent_name="Test",
            files_written=["a.py"],
        )
        assert count == 0
        assert stop is False

    def test_no_write_increments(self):
        tc = MagicMock()
        tc.name = "read_file"
        count, stop = self._check(
            tool_calls=[tc],
            results=["content"],
            stagnant_count=2,
            max_stagnant=5,
            agent_name="Test",
            files_written=["a.py"],
        )
        # read_file calls count as half-stagnant (research activity)
        assert count == 2.5
        assert stop is False

    def test_stagnation_triggers_stop(self):
        tc = MagicMock()
        tc.name = "read_file"
        count, stop = self._check(
            tool_calls=[tc],
            results=["content"],
            stagnant_count=4.5,
            max_stagnant=5,
            agent_name="Test",
            files_written=["a.py"],
        )
        assert count == 5.0
        assert stop is True


# ── DependencyGraphStore.get_layers with cycles ──────────────────────────────


class TestGetLayersCycles:
    def test_acyclic(self):
        from memory.dependency_graph import DependencyGraphStore
        store = DependencyGraphStore.__new__(DependencyGraphStore)
        import networkx as nx
        store._graph = nx.DiGraph()
        store._graph.add_edge("a.py", "b.py")
        store._graph.add_edge("b.py", "c.py")
        layers = store.get_layers()
        assert layers["c.py"] == 0
        assert layers["b.py"] == 1
        assert layers["a.py"] == 2

    def test_cyclic_does_not_crash(self):
        from memory.dependency_graph import DependencyGraphStore
        import networkx as nx
        store = DependencyGraphStore.__new__(DependencyGraphStore)
        store._graph = nx.DiGraph()
        store._graph.add_edge("a.py", "b.py")
        store._graph.add_edge("b.py", "a.py")
        # Should not raise NetworkXUnfeasible
        layers = store.get_layers()
        assert isinstance(layers, dict)
        assert "a.py" in layers
        assert "b.py" in layers


# ── LocalSandbox timeout ─────────────────────────────────────────────────────


@pytest.mark.skipif(sys.platform == "win32", reason="asyncio subprocess unreliable in Windows test runner")
class TestLocalSandboxTimeout:
    def test_fast_command(self, tmp_path):
        from sandbox.sandbox_runner import LocalSandbox

        async def _test():
            sb = LocalSandbox(timeout=30)
            info = await sb.create(tmp_path)
            return await sb.execute(info.sandbox_id, 'python -c "print(\'hello\')"')

        result = _run(_test())
        assert result.exit_code == 0
        assert "hello" in result.stdout

    def test_timeout_kills(self, tmp_path):
        from sandbox.sandbox_runner import LocalSandbox

        async def _test():
            sb = LocalSandbox(timeout=2)
            info = await sb.create(tmp_path)
            return await sb.execute(
                info.sandbox_id,
                'python -c "import time; time.sleep(120)"',
            )

        result = _run(_test())
        assert result.exit_code == -1
        assert "timed out" in result.stderr

    def test_sandbox_not_found(self):
        from sandbox.sandbox_runner import LocalSandbox
        sb = LocalSandbox()
        result = _run(sb.execute("nonexistent", "echo hi"))
        assert result.exit_code == -1
        assert "not found" in result.stderr.lower()


# ── Graceful shutdown ────────────────────────────────────────────────────────


class TestGracefulShutdown:
    def test_register_and_unregister(self):
        from core.shutdown import register_resource, unregister_resource, _resources

        class _FakeResource:
            async def teardown(self):
                pass

        r = _FakeResource()
        initial = len(_resources)
        register_resource(r)
        assert len(_resources) == initial + 1
        unregister_resource(r)
        assert len(_resources) == initial

    def test_unregister_missing_is_noop(self):
        from core.shutdown import unregister_resource

        class _FakeResource:
            async def teardown(self):
                pass

        # Should not raise
        unregister_resource(_FakeResource())

    def test_teardown_all(self):
        from core.shutdown import register_resource, _teardown_all, _resources

        torn_down = []

        class _FakeResource:
            def __init__(self, name):
                self.name = name
            async def teardown(self):
                torn_down.append(self.name)

        # Save and clear existing resources
        saved = list(_resources)
        _resources.clear()

        register_resource(_FakeResource("a"))
        register_resource(_FakeResource("b"))
        _run(_teardown_all())
        assert torn_down == ["b", "a"]  # reversed order
        assert len(_resources) == 0

        # Restore
        _resources.extend(saved)

    def test_teardown_continues_on_error(self):
        from core.shutdown import register_resource, _teardown_all, _resources

        torn_down = []

        class _Bad:
            async def teardown(self):
                raise RuntimeError("boom")

        class _Good:
            async def teardown(self):
                torn_down.append("good")

        saved = list(_resources)
        _resources.clear()

        register_resource(_Good())
        register_resource(_Bad())
        _run(_teardown_all())
        assert "good" in torn_down

        _resources.extend(saved)

    def test_install_signal_handlers_idempotent(self):
        from core.shutdown import install_signal_handlers
        # Should not raise even when called twice
        install_signal_handlers()
        install_signal_handlers()


# ── Language rules file loading ──────────────────────────────────────────────


class TestLanguageRulesFileLoading:
    def test_loads_from_data_dir(self):
        from core.language_rules import get_rules
        # java rules are registered (either from data file or fallback)
        rules = get_rules("java")
        assert "Spring Boot" in rules
        assert len(rules) > 100

    def test_file_rules_override_fallback(self):
        from core.language_rules import get_rules
        # Fallback rules contain Spring Boot guidance
        rules = get_rules("java")
        assert "SecurityFilterChain" in rules

    def test_fallback_languages_present(self):
        from core.language_rules import get_rules, list_languages
        langs = list_languages()
        for lang in ("go", "rust", "python"):
            assert lang in langs, f"{lang} missing from registry"
            assert len(get_rules(lang)) > 50

    def test_load_from_custom_dir(self, tmp_path):
        from core.language_rules import clear_rules, load_rules_from_directory, get_rules

        (tmp_path / "kotlin.txt").write_text("Use data classes for DTOs", encoding="utf-8")
        saved_rules = {}

        # Capture current state
        from core import language_rules as lr
        saved_rules = dict(lr._RULES)

        clear_rules()
        count = load_rules_from_directory(tmp_path)
        assert count == 1
        assert "data classes" in get_rules("kotlin")

        # Restore
        lr._RULES.clear()
        lr._RULES.update(saved_rules)

    def test_load_nonexistent_dir(self, tmp_path):
        from core.language_rules import load_rules_from_directory
        count = load_rules_from_directory(tmp_path / "does_not_exist")
        assert count == 0

    def test_coder_agent_no_longer_has_inline_dict(self):
        """Verify _LANGUAGE_SYNTAX_RULES class attribute is removed."""
        from agents.coder_agent import CoderAgent
        assert not hasattr(CoderAgent, "_LANGUAGE_SYNTAX_RULES")
