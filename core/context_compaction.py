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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

# Rough chars-per-token estimate (conservative — avoids exceeding real limit).
_CHARS_PER_TOKEN = 3.5

# Default budget: 120 000 tokens expressed in characters.
# Raised from 100k → 120k: the extra headroom avoids premature compaction
# that was discarding critical tool_result messages mid-fix-loop.
DEFAULT_CHAR_BUDGET = int(120_000 * _CHARS_PER_TOKEN)

# Number of tail messages always preserved verbatim.
# Raised from 6 → 10: with only 6, the last write_file + its tool_result
# plus the build error were frequently compacted away, leaving the fix
# agent with no memory of what it just tried.
_KEEP_TAIL = 10


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
    user message that includes a summary of which files were read/written in
    the removed section (so the model doesn't re-read already-seen files).
    """
    total = sum(_message_chars(m) for m in messages)
    if total <= char_budget:
        return messages

    if len(messages) <= keep_tail + 1:
        # Not enough messages to compact — return as-is.
        return messages

    head = [messages[0]]
    tail = messages[-keep_tail:]
    middle = messages[1:-keep_tail]
    removed_count = len(middle)

    # Extract a brief summary of tool interactions in the removed section
    # so the model retains awareness of what files were already processed.
    files_read: list[str] = []
    files_written: list[str] = []
    for msg in middle:
        content = msg.get("content", "")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    text = block.get("content", "") or block.get("text", "")
                    if "Written" in text and " bytes to " in text:
                        # Extract path from "Written N bytes to path"
                        parts = text.split(" bytes to ", 1)
                        if len(parts) == 2:
                            files_written.append(parts[1].split("\n")[0].strip())
                    elif text.startswith("[File: ") or text.startswith("["):
                        # read_file result header
                        path = text.split("|")[0].replace("[File: ", "").replace("[", "").strip()
                        if path and "/" in path:
                            files_read.append(path)

    logger.info(
        "Compacting message history: %d chars → removed %d middle messages "
        "(read %d files, wrote %d files)",
        total, removed_count, len(files_read), len(files_written),
    )

    summary_parts = [
        f"[{removed_count} earlier messages removed to fit context window. "
        f"The conversation started with the task prompt above and the most "
        f"recent tool interactions follow below.]"
    ]
    if files_written:
        summary_parts.append(
            f"Files written in removed section: {', '.join(dict.fromkeys(files_written))}"
        )
    if files_read:
        unique_reads = list(dict.fromkeys(files_read))[:15]
        summary_parts.append(
            f"Files read in removed section: {', '.join(unique_reads)}"
        )

    summary = {
        "role": "user",
        "content": "\n".join(summary_parts),
    }

    return head + [summary] + tail


async def compact_messages_with_summary(
    messages: list[dict],
    llm_generate: "Callable[..., Awaitable[Any]]",
    *,
    char_budget: int = DEFAULT_CHAR_BUDGET,
    keep_tail: int = _KEEP_TAIL,
) -> list[dict]:
    """Compaction with an LLM-generated semantic summary of removed messages.

    Like ``compact_messages`` but instead of a bare file-list marker, calls
    *llm_generate* to produce a concise summary of the removed conversation
    section.  Falls back to the simple compaction on any LLM failure.

    Args:
        llm_generate: Async callable with signature
            ``(prompt: str, system: str, max_tokens: int) -> response``
            where ``response.content`` is the generated text.  Typically
            bound to ``LLMClient.generate``.
    """
    total = sum(_message_chars(m) for m in messages)
    if total <= char_budget:
        return messages

    if len(messages) <= keep_tail + 1:
        return messages

    head = [messages[0]]
    tail = messages[-keep_tail:]
    middle = messages[1:-keep_tail]

    # Build a condensed representation of the middle for the LLM
    middle_text_parts: list[str] = []
    for msg in middle:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if isinstance(content, list):
            texts = []
            for block in content:
                if isinstance(block, dict):
                    texts.append(block.get("text", "") or block.get("content", ""))
                elif isinstance(block, str):
                    texts.append(block)
            content = "\n".join(texts)
        # Truncate individual messages to keep the summarization prompt manageable
        if len(content) > 500:
            content = content[:250] + " ... " + content[-250:]
        middle_text_parts.append(f"[{role}] {content}")

    middle_text = "\n---\n".join(middle_text_parts)
    # Cap the total summarization input to avoid blowing the budget
    if len(middle_text) > 15_000:
        middle_text = middle_text[:7_500] + "\n...[truncated]...\n" + middle_text[-7_500:]

    prompt = (
        "Summarise the following conversation excerpt in 3-5 bullet points.\n"
        "Focus on: what files were created/modified, what errors occurred, "
        "what fixes were attempted, and the current state of progress.\n"
        "Be concise — this summary will replace the removed messages.\n\n"
        f"{middle_text}"
    )

    try:
        response = await llm_generate(
            prompt=prompt,
            system="You are a conversation summariser. Output only the bullet-point summary.",
            max_tokens=300,
        )
        llm_summary = response.content.strip() if hasattr(response, "content") else str(response).strip()
    except Exception as exc:
        logger.warning("LLM summary failed (%s), falling back to simple compaction.", exc)
        return compact_messages(messages, char_budget=char_budget, keep_tail=keep_tail)

    logger.info(
        "LLM-based compaction: removed %d middle messages, summary %d chars.",
        len(middle), len(llm_summary),
    )

    summary = {
        "role": "user",
        "content": (
            f"[{len(middle)} earlier messages removed to fit context window. "
            f"LLM-generated summary of removed section:]\n\n{llm_summary}"
        ),
    }

    return head + [summary] + tail


# ── File-state restoration ──────────────────────────────────────────────

# After compaction, recently-referenced files may have been discussed in
# the removed section.  The model no longer "sees" their contents.  This
# mirrors Claude Code's approach of re-injecting the current state of
# recently-referenced files so the agent doesn't operate on stale context.

_MAX_RESTORE_FILES = 5
_MAX_RESTORE_CHARS = 5_000  # per file


def _extract_recent_file_paths(messages: list[dict], *, max_files: int = _MAX_RESTORE_FILES) -> list[str]:
    """Extract file paths recently referenced in tool-use messages.

    Scans from newest to oldest and returns paths in most-recent-first order,
    deduplicated.
    """
    import re
    paths: list[str] = []
    seen: set[str] = set()

    for msg in reversed(messages):
        content = msg.get("content", "")
        if isinstance(content, list):
            texts = []
            for block in content:
                if isinstance(block, dict):
                    texts.append(block.get("text", "") or block.get("content", ""))
                elif isinstance(block, str):
                    texts.append(block)
            content = "\n".join(texts)

        # Match common patterns: "Written N bytes to <path>", "[File: <path>", tool input paths
        for pattern in [
            r"Written \d+ bytes to (.+?)[\n\r]",
            r"\[File: ([^\]\|]+)",
            r'"file(?:_path|name)?"\s*:\s*"([^"]+)"',
        ]:
            for match in re.finditer(pattern, content):
                path = match.group(1).strip()
                if path and path not in seen and "/" in path:
                    seen.add(path)
                    paths.append(path)
                    if len(paths) >= max_files:
                        return paths
    return paths


def restore_file_state(
    compacted_messages: list[dict],
    read_file_fn: "Callable[[str], str | None]",
    *,
    max_files: int = _MAX_RESTORE_FILES,
    max_chars_per_file: int = _MAX_RESTORE_CHARS,
) -> list[dict]:
    """Re-inject current file contents after compaction.

    Scans the *tail* messages for recently-referenced files, reads their
    current contents via *read_file_fn*, and inserts a restoration message
    right after the compaction summary so the agent has fresh context.

    Args:
        compacted_messages: Output from ``compact_messages`` or
            ``compact_messages_with_summary``.
        read_file_fn: Synchronous callable ``(path) -> content | None``.
            Should return ``None`` for missing/unreadable files.
        max_files: Maximum files to restore (default 5).
        max_chars_per_file: Truncate each file to this many characters.

    Returns:
        New message list with an optional file-restoration message inserted.
    """
    if len(compacted_messages) < 3:
        return compacted_messages

    # Only act if there's a compaction marker
    has_marker = any(
        "[" in (m.get("content", "") if isinstance(m.get("content"), str) else "")
        and "messages removed" in (m.get("content", "") if isinstance(m.get("content"), str) else "")
        for m in compacted_messages[:3]
    )
    if not has_marker:
        return compacted_messages

    # Extract paths from the tail portion (everything after the summary marker)
    tail_start = 2  # head + summary
    tail_messages = compacted_messages[tail_start:]
    paths = _extract_recent_file_paths(tail_messages, max_files=max_files)

    if not paths:
        return compacted_messages

    # Read current file contents
    restorations: list[str] = []
    for path in paths:
        try:
            content = read_file_fn(path)
            if content is None:
                continue
            if len(content) > max_chars_per_file:
                content = content[:max_chars_per_file] + "\n... [truncated]"
            restorations.append(f"### {path} (current state)\n```\n{content}\n```")
        except Exception:
            logger.debug("Could not restore %s", path, exc_info=True)

    if not restorations:
        return compacted_messages

    logger.info(
        "Restoring %d file states after compaction (%s)",
        len(restorations), [p for p in paths[:len(restorations)]],
    )

    restore_msg = {
        "role": "user",
        "content": (
            "[File context restoration — current contents of recently-modified files:]\n\n"
            + "\n\n".join(restorations)
        ),
    }

    # Insert after the compaction summary (position 2) and before tail
    return compacted_messages[:tail_start] + [restore_msg] + compacted_messages[tail_start:]
