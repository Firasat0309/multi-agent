"""Loader for externalized language rule YAML files.

Provides cached access to language-specific patterns (duplicate detection,
package validation, Spring Boot defaults) without hardcoding them in agent
source code.
"""

from __future__ import annotations

import functools
import re
from pathlib import Path
from typing import Any

import yaml

_CONFIG_DIR = Path(__file__).resolve().parent


@functools.lru_cache(maxsize=1)
def load_language_rules() -> dict[str, Any]:
    """Load ``language_rules.yaml`` and compile regex patterns."""
    path = _CONFIG_DIR / "language_rules.yaml"
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    # Compile patterns for runtime use
    for lang, cfg in raw.items():
        pattern_str = cfg.get("duplicate_pattern")
        if pattern_str:
            cfg["_compiled_pattern"] = re.compile(pattern_str, re.MULTILINE)
    return raw


@functools.lru_cache(maxsize=1)
def load_java_rules() -> dict[str, Any]:
    """Load ``java_rules.yaml``."""
    path = _CONFIG_DIR / "java_rules.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def get_duplicate_pattern(language: str) -> re.Pattern[str] | None:
    """Return the compiled duplicate-detection regex for *language*, or None."""
    rules = load_language_rules()
    cfg = rules.get(language)
    if cfg is None:
        return None
    return cfg.get("_compiled_pattern")


def get_duplicates_allowed(language: str) -> list[str]:
    """Return names that are allowed to repeat for *language*."""
    rules = load_language_rules()
    cfg = rules.get(language)
    if cfg is None:
        return []
    return cfg.get("duplicates_allowed", [])
