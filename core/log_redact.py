"""Logging filter that redacts API keys and secrets from log output."""

from __future__ import annotations

import logging
import re


# Patterns matching common API key formats.  Each pattern is compiled once.
_REDACT_PATTERNS: list[re.Pattern[str]] = [
    # Anthropic: sk-ant-api03-...
    re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"),
    # OpenAI: sk-proj-... or sk-...
    re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{20,}"),
    # Generic bearer tokens
    re.compile(r"Bearer\s+[A-Za-z0-9_.-]{20,}", re.IGNORECASE),
    # Google/Gemini API keys (AIza...)
    re.compile(r"AIza[A-Za-z0-9_-]{30,}"),
    # Environment variable assignments with key-like values
    re.compile(
        r"((?:ANTHROPIC|OPENAI|GEMINI|API)[_A-Z]*KEY\s*=\s*)['\"]?[A-Za-z0-9_.-]{10,}['\"]?",
        re.IGNORECASE,
    ),
]

_REPLACEMENT = "***REDACTED***"


class SecretRedactionFilter(logging.Filter):
    """Logging filter that scrubs API keys from log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: redact(v) if isinstance(v, str) else v
                    for k, v in record.args.items()
                }
            elif isinstance(record.args, tuple):
                record.args = tuple(
                    redact(a) if isinstance(a, str) else a for a in record.args
                )
        return True


def redact(text: str) -> str:
    """Replace any recognised secret patterns in *text* with a placeholder."""
    for pattern in _REDACT_PATTERNS:
        text = pattern.sub(_REPLACEMENT, text)
    return text
