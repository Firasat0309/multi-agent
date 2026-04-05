"""Tiered model selection — route tasks to appropriate model tiers.

Routes simple tasks (config files, DTOs, boilerplate) to cheaper/faster
models while reserving expensive models for complex files (services,
algorithms, integration code).

Gated behind the ``MODEL_ROUTING`` feature flag.

Design:
  - Three tiers: FAST (cheap), STANDARD (default), COMPLEX (premium).
  - Routing is based on file metadata (layer, dependency count, estimated
    complexity) — no extra LLM call needed.
  - The executor passes the recommended model to the LLM client, which
    can override its default model for that call.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.models import FileBlueprint

logger = logging.getLogger(__name__)


class ModelTier(str, Enum):
    """LLM model tier for cost/performance routing."""
    FAST = "fast"           # Haiku / GPT-4o-mini — simple files
    STANDARD = "standard"   # Sonnet / GPT-4o — typical files
    COMPLEX = "complex"     # Opus / o1 — high-complexity files


# Default model mappings per provider
_TIER_MODELS: dict[str, dict[ModelTier, str]] = {
    "anthropic": {
        ModelTier.FAST: "claude-haiku-4-20250414",
        ModelTier.STANDARD: "claude-sonnet-4-20250514",
        ModelTier.COMPLEX: "claude-sonnet-4-20250514",  # Opus when available
    },
    "openai": {
        ModelTier.FAST: "gpt-4o-mini",
        ModelTier.STANDARD: "gpt-4o",
        ModelTier.COMPLEX: "gpt-4o",
    },
    "gemini": {
        ModelTier.FAST: "gemini-2.0-flash",
        ModelTier.STANDARD: "gemini-2.5-pro",
        ModelTier.COMPLEX: "gemini-2.5-pro",
    },
}

# Layers that are typically simple enough for FAST tier
_FAST_LAYERS = {"model", "config"}

# Layers that tend to be complex
_COMPLEX_LAYERS = {"controller", "service"}


class ModelRouter:
    """Routes files to appropriate model tiers based on complexity signals."""

    def __init__(self, provider: str = "anthropic") -> None:
        self._provider = provider

    def route(self, fb: FileBlueprint) -> ModelTier:
        """Determine the appropriate model tier for a file blueprint.

        Heuristics:
          - FAST: model/config layer with ≤2 dependencies, no complex purpose
          - COMPLEX: ≥5 dependencies, or service/controller with complex patterns
          - STANDARD: everything else
        """
        dep_count = len(fb.depends_on)
        layer = fb.layer.lower() if fb.layer else ""
        purpose = fb.purpose.lower() if fb.purpose else ""

        # Complexity signals in the purpose description
        _complex_signals = (
            "algorithm", "transaction", "concurren", "stream",
            "websocket", "caching", "pagination", "authentication",
            "authorization", "middleware", "interceptor", "security",
        )
        has_complex_purpose = any(s in purpose for s in _complex_signals)

        # FAST tier: simple files with few dependencies
        if (
            layer in _FAST_LAYERS
            and dep_count <= 2
            and not has_complex_purpose
        ):
            return ModelTier.FAST

        # COMPLEX tier: many dependencies or complex patterns
        if dep_count >= 5 or has_complex_purpose:
            return ModelTier.COMPLEX

        # COMPLEX tier: service/controller with moderate deps
        if layer in _COMPLEX_LAYERS and dep_count >= 3:
            return ModelTier.COMPLEX

        return ModelTier.STANDARD

    def get_model(self, tier: ModelTier) -> str:
        """Get the model name for the given tier and provider."""
        provider_models = _TIER_MODELS.get(self._provider, _TIER_MODELS["anthropic"])
        return provider_models.get(tier, provider_models[ModelTier.STANDARD])

    def route_and_resolve(self, fb: FileBlueprint) -> tuple[ModelTier, str]:
        """Route a file and return (tier, model_name)."""
        tier = self.route(fb)
        model = self.get_model(tier)
        logger.debug(
            "Model routing: %s → %s (%s)",
            fb.path, tier.value, model,
        )
        return tier, model
