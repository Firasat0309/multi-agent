"""Shared JSON parsing utilities for LLM output.

Consolidates the strip-fences → parse → bracket-extract → repair
pattern that was duplicated across multiple agent files.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def strip_markdown_fences(text: str) -> str:
    """Remove leading/trailing markdown code fences from *text*."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # drop opening fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text


def parse_llm_json(
    text: str,
    *,
    repair: bool = True,
    walk_inward: bool = False,
    label: str = "LLM response",
) -> dict[str, Any]:
    """Parse JSON from LLM output with progressive fallback.

    1. Strip markdown fences.
    2. ``json.loads`` (strict).
    3. Extract outermost ``{ … }`` and retry.
    4. Optionally *walk_inward* — try successively shorter ``}`` positions
       so truncated output doesn't silently succeed with an incomplete fragment.
    5. If *repair* is True, run ``LLMClient._repair_json_text`` and retry.
    6. Return ``{}`` on total failure (logged as error).
    """
    text = strip_markdown_fences(text)

    # 1. Strict parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Bracket extraction
    start = text.find("{")
    if start == -1:
        logger.error("No JSON object found in %s: %s...", label, text[:200])
        return {}

    if walk_inward:
        # Walk inward: try successively shorter closing braces so a truncated
        # response doesn't parse as a silently incomplete fragment.
        end = len(text)
        while end > start:
            end = text.rfind("}", start, end)
            if end == -1:
                break
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass  # try one character shorter
    else:
        end = text.rfind("}") + 1
        if end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass

    # 3. Repair (trailing commas, truncated JSON, etc.)
    if repair:
        try:
            from core.llm_client import LLMClient

            extracted = text[start:] if start != -1 else text
            repaired = LLMClient._repair_json_text(extracted)
            result = json.loads(repaired)
            logger.info("%s JSON repair succeeded", label)
            return result
        except (json.JSONDecodeError, Exception):
            pass

    logger.error("Could not parse %s JSON: %s...", label, text[:200])
    return {}


def parse_llm_json_strict(
    text: str,
    *,
    label: str = "LLM response",
) -> dict[str, Any]:
    """Like :func:`parse_llm_json` but raises on failure instead of returning ``{}``."""
    result = parse_llm_json(text, repair=True, label=label)
    if not result:
        raise ValueError(
            f"Failed to parse {label} JSON from LLM response. "
            "The model returned malformed or truncated JSON."
        )
    return result
