"""Compile-time-style feature flag system for progressive rollout.

Inspired by Claude Code's ``feature('FLAG_NAME')`` pattern that enables
dead-code elimination.  In Python we can't do compile-time elimination,
but we can gate code paths at startup and conditionally register tools,
agents, and pipeline phases.

Usage::

    from core.feature_flags import feature, set_feature

    # At startup:
    set_feature("STREAMING", True)
    set_feature("FULLSTACK_MODE", os.environ.get("ENABLE_FULLSTACK") == "1")

    # In code:
    if feature("STREAMING"):
        await stream_response(...)
    else:
        response = await batch_generate(...)

    # Gate tool registration:
    if feature("MCP_TOOLS"):
        registry.register(mcp_browse_tool)
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# ── Global flag store ────────────────────────────────────────────────────────
# Flags are set once at startup and never change during a run.
# This is intentionally a module-level dict, not a class, for zero-overhead
# access in hot paths (tool dispatch, agent loops).

_FLAGS: dict[str, bool] = {}
_FROZEN: bool = False


def feature(name: str) -> bool:
    """Check whether a feature flag is enabled.

    Returns False for unknown flags (fail-closed).
    """
    return _FLAGS.get(name, False)


def set_feature(name: str, enabled: bool) -> None:
    """Set a feature flag.  Must be called before ``freeze_features()``."""
    if _FROZEN:
        logger.warning("Cannot set feature '%s' after flags are frozen", name)
        return
    _FLAGS[name] = enabled
    if enabled:
        logger.info("Feature flag enabled: %s", name)


def freeze_features() -> None:
    """Prevent further flag changes (call after startup configuration)."""
    global _FROZEN
    _FROZEN = True
    logger.info("Feature flags frozen: %s", {k: v for k, v in _FLAGS.items() if v})


def get_all_features() -> dict[str, bool]:
    """Return a copy of all feature flags (for logging/debugging)."""
    return dict(_FLAGS)


def reset_features() -> None:
    """Reset all flags (for testing only)."""
    global _FROZEN
    _FLAGS.clear()
    _FROZEN = False


# ── Initialize default flags from environment ───────────────────────────────

def init_features_from_env() -> None:
    """Read feature flags from environment variables.

    Convention: ``FEATURE_<NAME>=1`` enables the flag.
    Example: ``FEATURE_STREAMING=1``, ``FEATURE_MCP_TOOLS=1``
    """
    prefix = "FEATURE_"
    for key, value in os.environ.items():
        if key.startswith(prefix):
            flag_name = key[len(prefix):]
            set_feature(flag_name, value in ("1", "true", "True", "yes"))


# ── Standard flag names ──────────────────────────────────────────────────────
# Defined as constants for IDE autocompletion and typo prevention.

STREAMING = "STREAMING"
MCP_TOOLS = "MCP_TOOLS"
FULLSTACK_MODE = "FULLSTACK_MODE"
SANDBOX_DOCKER = "SANDBOX_DOCKER"
HOOK_SYSTEM = "HOOK_SYSTEM"
SESSION_PERSISTENCE = "SESSION_PERSISTENCE"
COST_TRACKING = "COST_TRACKING"
ADVANCED_CONTEXT = "ADVANCED_CONTEXT"
