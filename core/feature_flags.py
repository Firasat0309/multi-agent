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
STREAMING_LLM = "STREAMING_LLM"
MCP_TOOLS = "MCP_TOOLS"
TOKEN_BUDGETS = "TOKEN_BUDGETS"
REACTIVE_COMPACTION = "REACTIVE_COMPACTION"
SESSION_RESUME = "SESSION_RESUME"
FILE_STATE_CACHE = "FILE_STATE_CACHE"
PROMPT_CACHING = "PROMPT_CACHING"
TOOL_REGISTRY = "TOOL_REGISTRY"

# ── All known flags with defaults (for documentation / validation) ───────────

_KNOWN_FLAGS: dict[str, bool] = {
    STREAMING: False,
    STREAMING_LLM: False,
    MCP_TOOLS: False,
    TOKEN_BUDGETS: True,
    REACTIVE_COMPACTION: True,
    SESSION_RESUME: False,
    FILE_STATE_CACHE: False,
    PROMPT_CACHING: True,
    TOOL_REGISTRY: False,
}


def init_defaults() -> None:
    """Set all known flags to their default values (unless already set)."""
    for name, default in _KNOWN_FLAGS.items():
        if name not in _FLAGS:
            _FLAGS[name] = default
FULLSTACK_MODE = "FULLSTACK_MODE"
SANDBOX_DOCKER = "SANDBOX_DOCKER"
HOOK_SYSTEM = "HOOK_SYSTEM"
SESSION_PERSISTENCE = "SESSION_PERSISTENCE"
COST_TRACKING = "COST_TRACKING"
ADVANCED_CONTEXT = "ADVANCED_CONTEXT"

# ── New flags (phase 2 improvements) ────────────────────────────────────────

# LLM response streaming — process tool calls as they arrive
STREAMING_LLM = "STREAMING_LLM"

# Reactive compaction — auto-compact on prompt-too-long errors
REACTIVE_COMPACTION = "REACTIVE_COMPACTION"

# Per-file token budget enforcement
TOKEN_BUDGETS = "TOKEN_BUDGETS"

# Blueprint validation and retry on invalid JSON
BLUEPRINT_RETRY = "BLUEPRINT_RETRY"

# OpenTelemetry span instrumentation on hot paths
OTEL_SPANS = "OTEL_SPANS"

# Cost warning at 80% budget threshold (not just hard-fail at 100%)
COST_WARNING = "COST_WARNING"

# Optional lightweight code review between generation and build.
# ReviewerAgent catches semantic bugs (null dereference, missing error
# handling, layer violations) that the compiler won't catch.
QUICK_REVIEW = "QUICK_REVIEW"

# ── Phase 3 improvement flags ───────────────────────────────────────────────

# Structured output: force CoderAgent to use write_file via tool_choice,
# eliminating nudge/recovery iterations when the LLM forgets to call it.
STRUCTURED_OUTPUT = "STRUCTURED_OUTPUT"

# AST pre-validation: fast syntax check before running expensive builds.
# Catches ~60% of syntax errors in <1s, skipping 8-15s build steps.
FAST_SYNTAX_CHECK = "FAST_SYNTAX_CHECK"

# Blueprint consistency validation: cross-file consistency check before
# code generation begins. Catches missing types, duplicate exports, etc.
BLUEPRINT_CONSISTENCY = "BLUEPRINT_CONSISTENCY"

# Tiered model selection: route simple tasks (config files, DTOs) to
# cheaper/faster models (Haiku) while using Sonnet for complex files.
MODEL_ROUTING = "MODEL_ROUTING"

# Enhanced fix memory: cross-file pattern retrieval so fix agents can
# reuse resolution strategies from similar errors in other files.
ENHANCED_FIX_MEMORY = "ENHANCED_FIX_MEMORY"

# Reviewer with tools: give ReviewerAgent read-only tools (read_file,
# search_code, find_definition) for cross-file verification.
REVIEWER_TOOLS = "REVIEWER_TOOLS"

# Prompt injection defense: sanitize user inputs and wrap user-provided
# content in delimiters to prevent prompt injection attacks.
PROMPT_GUARD = "PROMPT_GUARD"

# Selective cascade: only cascade failures to direct dependents rather
# than failing all downstream files.
SELECTIVE_CASCADE = "SELECTIVE_CASCADE"

# DAG-driven execution: replace tier-sequential execution with per-file
# dependency DAG scheduling for maximum parallelism.
DAG_EXECUTOR = "DAG_EXECUTOR"

# Streaming fix pipeline: stream LLM fix output and validate incrementally,
# cancelling bad generations early to save tokens.
STREAMING_FIX = "STREAMING_FIX"

# Batch generation: group small independent files (models, DTOs, configs)
# into a single LLM call to reduce per-call overhead in early tiers.
BATCH_GENERATION = "BATCH_GENERATION"

# Speculative execution: start generating dependent files before their
# dependencies finish, using blueprint info for interfaces.  If deps fail
# or actual exports diverge from assumptions, discard the speculative output.
SPECULATIVE_EXEC = "SPECULATIVE_EXEC"

# Typed message construction: use core/messages.py dataclasses instead of
# raw dict literals for conversation messages.  Prevents key typos and
# provides IDE autocompletion while remaining dict-compatible.
TYPED_MESSAGES = "TYPED_MESSAGES"
