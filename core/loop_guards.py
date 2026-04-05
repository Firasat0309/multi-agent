"""Loop guard heuristics extracted from BaseAgent.

These stateless helper functions implement the budget-guard nudges,
stagnation detection, and truncation recovery that keep the agentic
tool-use loop on track.  Extracting them reduces BaseAgent by ~150
lines and makes each heuristic independently testable.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def build_budget_nudge(
    *,
    target_file: str | None,
    iteration: int,
    max_iterations: int,
    files_written: list[str],
    has_write_call: bool,
    consecutive_nudges: int,
    max_consecutive_nudges: int,
    path_in_written_fn: Any,
) -> str | None:
    """Return a nudge message if the agent is past halfway without writing.

    Returns ``None`` if no nudge is needed.
    """
    # Single-target file nudge
    if (
        target_file
        and iteration >= max_iterations // 2
        and not path_in_written_fn(target_file, files_written)
        and not has_write_call
        and consecutive_nudges < max_consecutive_nudges
    ):
        return (
            f"WARNING: You have used {iteration + 1} of {max_iterations} iterations "
            f"and still have not written the target file '{target_file}'. "
            f"You MUST call write_file with path='{target_file}' on this turn. "
            f"Do NOT read more files or deliberate further — write the complete "
            f"component now using write_file."
        )

    # Multi-file budget nudge (no single target)
    if (
        target_file is None
        and files_written
        and iteration >= max_iterations // 2
    ):
        return (
            f"You have written {len(files_written)} file(s) so far: "
            f"{', '.join(files_written[-5:])}. "
            f"You have used {iteration + 1} of {max_iterations} iterations. "
            "If you still have files to write, write them NOW using write_file. "
            "When ALL files are written, stop — do not call any more tools."
        )

    # Zero-writes nudge for multi-file agents
    if (
        target_file is None
        and not files_written
        and iteration >= max_iterations // 3
    ):
        return (
            f"WARNING: You have used {iteration + 1} of {max_iterations} iterations "
            "and have NOT written any files yet. "
            "You MUST start calling write_file NOW to produce the required files. "
            "Do not read more files — use the context you already have."
        )

    return None


def check_stagnation(
    tool_calls: list[Any],
    results: list[str],
    stagnant_count: float,
    max_stagnant: int,
    agent_name: str,
    files_written: list[str],
) -> tuple[float, bool]:
    """Update stagnation counter and return ``(new_count, should_stop)``.

    A stagnant iteration is one where the agent has files written but
    produces no new writes this turn AND made no read/search calls.
    """
    wrote_this_iter = any(
        tc.name == "write_file" and not result.startswith("Error")
        for tc, result in zip(tool_calls, results)
    )
    if wrote_this_iter:
        return 0, False

    read_calls = sum(
        1 for tc in tool_calls
        if tc.name in ("read_file", "search_code", "find_definition", "list_files")
    )
    if read_calls > 0:
        stagnant_count += 0.5
    else:
        stagnant_count += 1

    logger.warning(
        "%s: no new file written this iteration (%.1f/%d stagnant, %d files so far)",
        agent_name, stagnant_count, max_stagnant, len(files_written),
    )
    if int(stagnant_count) >= max_stagnant:
        logger.info(
            "%s: stagnation limit reached — treating %d written file(s) as complete",
            agent_name, len(files_written),
        )
        return stagnant_count, True
    return stagnant_count, False


def build_end_turn_reminder(
    target_file: str,
    response_content: str,
    end_turn_reminders: int,
    max_reminders: int,
    extract_code_block_fn: Any,
) -> tuple[str | None, bool]:
    """Build a write_file reminder for end_turn without file written.

    Returns ``(reminder_text, is_exhausted)`` where ``is_exhausted`` means
    all reminders have been used and the caller should try auto-write.
    """
    if end_turn_reminders >= max_reminders:
        return None, True

    code_block = extract_code_block_fn(response_content or "")
    code_hint = ""
    if code_block:
        code_hint = (
            "\n\nIt looks like you already wrote the code "
            "as plain text. Use exactly that code as the "
            "content argument to write_file."
        )
    reminder = (
        f"You have not called write_file yet. "
        f"Please call write_file with "
        f"path='{target_file}' and the complete file "
        f"content now. Do NOT respond with plain text "
        f"— use the write_file tool.{code_hint}"
    )
    return reminder, False
