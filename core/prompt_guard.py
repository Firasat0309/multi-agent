"""Prompt injection defense — sanitize and delimit user-provided content.

When user-provided content (project descriptions, file purposes, error
messages from builds) is injected into prompts, malicious content could
hijack the LLM's instructions.  This module provides:

1. **Content delimiting** — wraps user content in unique delimiters so the
   LLM can distinguish instructions from user data.
2. **Suspicious pattern detection** — flags content that looks like prompt
   injection attempts (role overrides, instruction overrides, etc.).
3. **Sanitization** — strips or escapes dangerous patterns while preserving
   legitimate content.

Gated behind the ``PROMPT_GUARD`` feature flag.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# ── Suspicious patterns ──────────────────────────────────────────────────────
# These patterns indicate possible prompt injection attempts.
# They are checked against user-provided content BEFORE it enters the prompt.

_INJECTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Direct instruction overrides
    (re.compile(r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions|rules|prompts)", re.I),
     "instruction_override"),
    (re.compile(r"disregard\s+(all\s+)?(previous|above|prior|your)\s+(instructions|rules)", re.I),
     "instruction_override"),
    (re.compile(r"forget\s+(everything|all|your)\s+(instructions|rules|training)", re.I),
     "instruction_override"),

    # Role hijacking
    (re.compile(r"you\s+are\s+now\s+(a|an|the)\s+", re.I),
     "role_hijack"),
    (re.compile(r"pretend\s+(to\s+be|you\s+are)", re.I),
     "role_hijack"),
    (re.compile(r"act\s+as\s+(a|an|the|if)\s+", re.I),
     "role_hijack"),

    # System prompt extraction
    (re.compile(r"(show|reveal|print|output|display|repeat)\s+(your|the)\s+(system|initial)\s+(prompt|instructions|message)", re.I),
     "prompt_extraction"),
    (re.compile(r"what\s+(are|were)\s+your\s+(initial|system|original)\s+(instructions|prompt)", re.I),
     "prompt_extraction"),

    # Delimiter escape attempts
    (re.compile(r"</?system>|</?assistant>|</?human>|\[INST\]|\[/INST\]", re.I),
     "delimiter_escape"),

    # Tool/function abuse
    (re.compile(r"(call|invoke|execute|run)\s+(the\s+)?(function|tool|command)\s+", re.I),
     "tool_abuse"),
]

# Unique delimiter to wrap user content
_DELIMITER = "═══USER_CONTENT═══"


def detect_injection(text: str) -> list[tuple[str, str]]:
    """Check text for prompt injection patterns.

    Returns a list of (category, matched_text) tuples for each detected
    pattern.  Empty list means no injection detected.
    """
    findings: list[tuple[str, str]] = []
    for pattern, category in _INJECTION_PATTERNS:
        match = pattern.search(text)
        if match:
            findings.append((category, match.group(0)))
    return findings


def sanitize_user_content(text: str) -> str:
    """Remove or neutralize detected injection attempts.

    For each matched pattern, the offending substring is replaced with
    a harmless placeholder: ``[FILTERED: <category>]``

    Non-matching content is left untouched.
    """
    result = text
    for pattern, category in _INJECTION_PATTERNS:
        result = pattern.sub(f"[FILTERED: {category}]", result)
    return result


def delimit_user_content(text: str, label: str = "user input") -> str:
    """Wrap user-provided content in unique delimiters.

    This makes it explicit to the LLM that the enclosed text is data
    provided by the user, NOT additional instructions.

    Args:
        text: The user-provided content to delimit.
        label: A label describing what the content is (e.g. "project description").
    """
    return (
        f"\n{_DELIMITER} BEGIN {label.upper()} {_DELIMITER}\n"
        f"{text}\n"
        f"{_DELIMITER} END {label.upper()} {_DELIMITER}\n"
    )


def guard_prompt(text: str, label: str = "user input") -> str:
    """Full prompt guard pipeline: detect, sanitize, and delimit.

    1. Detects injection attempts (logs warnings).
    2. Sanitizes matched patterns.
    3. Wraps in delimiters.

    This is the main entry point for the PROMPT_GUARD feature.
    """
    findings = detect_injection(text)
    if findings:
        categories = [f[0] for f in findings]
        logger.warning(
            "Prompt injection detected in %s: %s",
            label, ", ".join(categories),
        )
        text = sanitize_user_content(text)

    return delimit_user_content(text, label)
