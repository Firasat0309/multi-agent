"""Message history compaction for the agentic tool-use loop.

When the conversation between an agent and its LLM grows too large, older
messages need to be summarised or trimmed to stay within the context window.
This module provides a simple character-budget compactor that:

  1. Always keeps the **first** user message (the task prompt).
  2. Always keeps the **last N** messages (the recent working context).
  3. Replaces the middle with a summary marker so the model knows context
     was elided.

The approach mirrors Claude Code's ``compactMessageHistory`` but is adapted
for the batch agentic loop rather than an interactive REPL.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Rough chars-per-token estimate (conservative — avoids exceeding real limit).
_CHARS_PER_TOKEN = 3.5

# Default budget: 100 000 tokens expressed in characters.
DEFAULT_CHAR_BUDGET = int(100_000 * _CHARS_PER_TOKEN)

# Number of tail messages always preserved verbatim.
_KEEP_TAIL = 6


def _message_chars(msg: dict) -> int:
    """Estimate character count of a single message dict."""
    content = msg.get("content", "")
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for block in content:
            if isinstance(block, dict):
                total += len(block.get("text", "")) + len(block.get("content", ""))
            elif isinstance(block, str):
                total += len(block)
        return total
    return 0


def compact_messages(
    messages: list[dict],
    *,
    char_budget: int = DEFAULT_CHAR_BUDGET,
    keep_tail: int = _KEEP_TAIL,
) -> list[dict]:
    """Return a compacted copy of *messages* that fits within *char_budget*.

    If the messages already fit, returns the original list unchanged (no copy).
    Otherwise, the first message + the last *keep_tail* messages are preserved
    and everything in between is replaced by a single ``[context compacted]``
    user message.
    """
    total = sum(_message_chars(m) for m in messages)
    if total <= char_budget:
        return messages

    if len(messages) <= keep_tail + 1:
        # Not enough messages to compact — return as-is.
        return messages

    head = [messages[0]]
    tail = messages[-keep_tail:]
    removed_count = len(messages) - 1 - keep_tail

    logger.info(
        "Compacting message history: %d chars → removed %d middle messages",
        total, removed_count,
    )

    summary = {
        "role": "user",
        "content": (
            f"[{removed_count} earlier messages removed to fit context window. "
            f"The conversation started with the task prompt above and the most "
            f"recent tool interactions follow below.]"
        ),
    }

    return head + [summary] + tail
