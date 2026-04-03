"""Shared test fixtures for isolation between test runs."""

from __future__ import annotations

import pytest

from core.event_bus import EventBus
from core.feature_flags import _FLAGS, set_feature
from core.file_metrics import reset_file_metrics
from core.hooks import HookRegistry


@pytest.fixture(autouse=True)
def _reset_singletons() -> None:
    """Reset global singletons before each test to prevent cross-test leaks."""
    # Clear feature flags (unfrozen for test flexibility)
    import core.feature_flags as ff
    ff._FROZEN = False
    ff._FLAGS.clear()

    # Reset per-file metrics
    reset_file_metrics()

    yield  # type: ignore[misc]


@pytest.fixture()
def event_bus() -> EventBus:
    """Provide a fresh EventBus for tests that need one."""
    bus = EventBus()
    yield bus  # type: ignore[misc]
    bus.reset()


@pytest.fixture()
def hook_registry() -> HookRegistry:
    """Provide a fresh HookRegistry for tests that need one."""
    registry = HookRegistry()
    yield registry  # type: ignore[misc]
    registry.clear()
