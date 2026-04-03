"""Tests for the improvement modules added during the multi-agent analysis pass.

Covers:
  - core/errors.py — structured error hierarchy
  - core/permissions.py — ToolPermissionChecker
  - core/feature_flags.py — feature flag system
  - core/hooks.py — HookRegistry
  - core/language_rules.py — pluggable language rules
  - core/event_bus.py — unsubscribe + reset
  - core/state_machine.py — cascade_failures BFS fix
  - core/context_builder.py — file read caching
  - config/settings.py — config validation
"""

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest


def _run(coro):
    return asyncio.run(coro)


# ── core/errors.py ───────────────────────────────────────────────────────────


class TestErrorHierarchy:
    def test_base_pipeline_error(self):
        from core.errors import PipelineError
        err = PipelineError("test", context={"key": "val"})
        assert str(err) == "test"
        assert err.context == {"key": "val"}

    def test_llm_retryable_error(self):
        from core.errors import LLMRetryableError
        err = LLMRetryableError("rate limit hit")
        assert isinstance(err, Exception)
        assert err.context == {}

    def test_config_validation_error(self):
        from core.errors import ConfigValidationError
        err = ConfigValidationError("bad config", context={"errors": ["a", "b"]})
        assert err.context["errors"] == ["a", "b"]

    def test_permission_denied_error(self):
        from core.errors import PermissionDeniedError
        err = PermissionDeniedError("not allowed")
        assert "not allowed" in str(err)


# ── core/permissions.py ──────────────────────────────────────────────────────


class TestToolPermissionChecker:
    @pytest.fixture
    def checker(self, tmp_path):
        from core.permissions import ToolPermissionChecker
        return ToolPermissionChecker(workspace_root=tmp_path)

    def test_write_within_workspace(self, checker):
        result = checker.check_write("src/main.py")
        assert result.allowed is True

    def test_write_path_traversal_blocked(self, checker):
        result = checker.check_write("../../etc/passwd")
        assert result.allowed is False
        assert "traversal" in result.reason.lower()

    def test_write_blocked_extension(self, checker):
        result = checker.check_write("keys/server.pem")
        assert result.allowed is False
        assert "blocked" in result.reason.lower()

    def test_check_command_safe(self, checker):
        result = checker.check_command("mvn package")
        assert result.allowed is True

    def test_check_command_dangerous(self, checker):
        result = checker.check_command("sudo rm -rf /")
        assert result.allowed is False


# ── core/feature_flags.py ────────────────────────────────────────────────────


class TestFeatureFlags:
    def test_default_flags(self):
        import core.feature_flags as ff
        old_flags = ff._FLAGS.copy()
        old_frozen = ff._FROZEN
        ff._FROZEN = False
        try:
            ff._FLAGS.clear()
            ff.init_features_from_env()
            # By default, standard flags should be False unless env says otherwise
            assert ff.feature("NONEXISTENT") is False
        finally:
            ff._FLAGS.update(old_flags)
            ff._FROZEN = old_frozen

    def test_set_and_freeze(self):
        import core.feature_flags as ff
        old_flags = ff._FLAGS.copy()
        old_frozen = ff._FROZEN
        ff._FROZEN = False
        try:
            ff._FLAGS.clear()
            ff.set_feature("TEST_FLAG", True)
            assert ff.feature("TEST_FLAG") is True
            ff.freeze_features()
            # After freezing, set_feature logs a warning but doesn't raise
            ff.set_feature("ANOTHER", True)
            assert ff.feature("ANOTHER") is False  # not set because frozen
        finally:
            ff._FLAGS.clear()
            ff._FLAGS.update(old_flags)
            ff._FROZEN = old_frozen


# ── core/hooks.py ────────────────────────────────────────────────────────────


class TestHookRegistry:
    def test_register_and_fire(self):
        from core.hooks import HookRegistry, HookEvent
        registry = HookRegistry()
        called = []

        async def on_start(**kwargs):
            called.append(kwargs)

        registry.register(HookEvent.PIPELINE_START, on_start)
        result = _run(registry.fire(HookEvent.PIPELINE_START, key="value"))
        assert result.handlers_run == 1
        assert result.handlers_failed == 0
        assert called == [{"key": "value"}]

    def test_decorator_style(self):
        from core.hooks import HookRegistry, HookEvent
        registry = HookRegistry()
        called = []

        @registry.on(HookEvent.PIPELINE_END)
        async def on_end(**kwargs):
            called.append("ended")

        _run(registry.fire(HookEvent.PIPELINE_END))
        assert called == ["ended"]

    def test_clear_and_reset(self):
        from core.hooks import HookRegistry, HookEvent
        registry = HookRegistry()

        async def handler(**kwargs):
            pass

        registry.register(HookEvent.PIPELINE_START, handler)
        registry.clear()
        result = _run(registry.fire(HookEvent.PIPELINE_START))
        assert result.handlers_run == 0

    def test_failed_handler_counted(self):
        from core.hooks import HookRegistry, HookEvent
        registry = HookRegistry()

        async def bad_handler(**kwargs):
            raise ValueError("boom")

        registry.register(HookEvent.PIPELINE_START, bad_handler)
        result = _run(registry.fire(HookEvent.PIPELINE_START))
        assert result.handlers_run == 1
        assert result.handlers_failed == 1


# ── core/language_rules.py ───────────────────────────────────────────────────


class TestLanguageRules:
    def test_register_and_get(self):
        from core.language_rules import register_rules, get_rules, has_rules
        register_rules("kotlin", "- Use data classes for DTOs")
        assert has_rules("kotlin")
        assert "data classes" in get_rules("kotlin")

    def test_get_unknown_language(self):
        from core.language_rules import get_rules
        assert get_rules("brainfuck") == ""

    def test_case_insensitive(self):
        from core.language_rules import register_rules, get_rules
        register_rules("SWIFT", "- Use optionals")
        assert get_rules("swift") != ""


# ── core/event_bus.py — unsubscribe + reset ──────────────────────────────────


class TestEventBusImprovements:
    def test_unsubscribe(self):
        from core.event_bus import EventBus, BusEventType

        bus = EventBus()
        called = []

        async def handler(event):
            called.append(event)

        bus.subscribe(BusEventType.FILE_WRITTEN, handler)
        removed = bus.unsubscribe(BusEventType.FILE_WRITTEN, handler)
        assert removed is True

        from core.event_bus import AgentEvent
        _run(bus.publish(AgentEvent(type=BusEventType.FILE_WRITTEN)))
        assert called == []

    def test_unsubscribe_not_found(self):
        from core.event_bus import EventBus, BusEventType

        bus = EventBus()

        async def handler(event):
            pass

        removed = bus.unsubscribe(BusEventType.FILE_WRITTEN, handler)
        assert removed is False

    def test_unsubscribe_critical(self):
        from core.event_bus import EventBus, BusEventType

        bus = EventBus()

        async def handler(event):
            pass

        bus.subscribe_critical(BusEventType.FILE_WRITTEN, handler)
        removed = bus.unsubscribe_critical(BusEventType.FILE_WRITTEN, handler)
        assert removed is True

    def test_reset(self):
        from core.event_bus import EventBus, BusEventType

        bus = EventBus()
        called = []

        async def handler(event):
            called.append(1)

        bus.subscribe(BusEventType.FILE_WRITTEN, handler)
        bus.subscribe_all(handler)
        bus.reset()

        from core.event_bus import AgentEvent
        _run(bus.publish(AgentEvent(type=BusEventType.FILE_WRITTEN)))
        assert called == []


# ── core/state_machine.py — cascade_failures BFS ────────────────────────────


class TestCascadeFailuresBFS:
    def test_cascade_transitive(self):
        from core.state_machine import LifecycleEngine, EventType, FilePhase

        # A → B → C (C depends on B, B depends on A)
        engine = LifecycleEngine(
            file_paths=["a.py", "b.py", "c.py"],
            file_deps={"b.py": ["a.py"], "c.py": ["b.py"]},
        )

        # Fail A
        engine.process_event("a.py", EventType.DEPS_MET)
        engine.process_event("a.py", EventType.RETRIES_EXHAUSTED)
        assert engine.get_lifecycle("a.py").phase == FilePhase.FAILED

        # B and C should cascade
        cascaded = engine.cascade_failures()
        assert "b.py" in cascaded
        assert "c.py" in cascaded

    def test_cascade_with_scope(self):
        from core.state_machine import LifecycleEngine, EventType, FilePhase

        engine = LifecycleEngine(
            file_paths=["a.py", "b.py", "c.py"],
            file_deps={"b.py": ["a.py"], "c.py": ["a.py"]},
        )

        engine.process_event("a.py", EventType.DEPS_MET)
        engine.process_event("a.py", EventType.RETRIES_EXHAUSTED)

        # Only cascade within scope
        cascaded = engine.cascade_failures(scope={"b.py"})
        assert "b.py" in cascaded
        assert "c.py" not in cascaded

    def test_no_cascade_when_non_pending(self):
        from core.state_machine import LifecycleEngine, EventType, FilePhase

        engine = LifecycleEngine(
            file_paths=["a.py", "b.py"],
            file_deps={"b.py": ["a.py"]},
        )

        # Move B past PENDING
        engine.process_event("b.py", EventType.DEPS_MET)
        engine.process_event("b.py", EventType.CODE_GENERATED)

        # Fail A
        engine.process_event("a.py", EventType.DEPS_MET)
        engine.process_event("a.py", EventType.RETRIES_EXHAUSTED)

        # B is in REVIEWING, not PENDING — should not cascade
        cascaded = engine.cascade_failures()
        assert cascaded == []


# ── config/settings.py — config validation ──────────────────────────────────


class TestConfigValidation:
    def test_validate_missing_api_key(self):
        from config.settings import Settings, LLMConfig, LLMProvider
        from core.errors import ConfigValidationError

        settings = Settings(llm=LLMConfig(provider=LLMProvider.ANTHROPIC, api_key=""))
        with pytest.raises(ConfigValidationError):
            settings.validate()

    def test_validate_invalid_concurrent_agents(self):
        from config.settings import Settings, LLMConfig, LLMProvider
        from core.errors import ConfigValidationError

        settings = Settings(
            llm=LLMConfig(provider=LLMProvider.ANTHROPIC, api_key="test-key"),
            max_concurrent_agents=0,
        )
        with pytest.raises(ConfigValidationError):
            settings.validate()

    def test_validate_passes_with_valid_config(self):
        from config.settings import Settings, LLMConfig, LLMProvider

        settings = Settings(
            llm=LLMConfig(provider=LLMProvider.ANTHROPIC, api_key="test-key"),
        )
        # Should not raise
        settings.validate()


# ── core/checkpoint.py — TerminalRunner Protocol ────────────────────────────


class TestTerminalRunnerProtocol:
    def test_protocol_compliance(self):
        from core.checkpoint import TerminalRunner

        class MockTerminal:
            async def run_command(self, command: str, *, timeout: int = 120):
                return None

        assert isinstance(MockTerminal(), TerminalRunner)
