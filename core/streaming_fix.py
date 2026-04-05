"""Streaming fix pipeline — stream LLM output and validate incrementally.

When fixing a build error, the normal path waits for the entire LLM
response to complete before writing the file and validating.  This module
intercepts the streaming fix response and performs lightweight syntax
checks on the accumulating output.

If the partial output already contains obvious fatal patterns (e.g.,
the same syntax error being reproduced, incomplete control structures
at >80% of expected output), the generation can be cancelled early to
save tokens.

Gated behind the ``STREAMING_FIX`` feature flag.

Usage from the executor::

    from core.streaming_fix import streaming_fix_file

    if feature("STREAMING_FIX"):
        content, cancelled = await streaming_fix_file(
            llm_client=self._am.llm,
            system_prompt=system_prompt,
            user_prompt=fix_prompt,
            language=language,
            known_bad_patterns=previous_error_signatures,
        )
        if cancelled:
            logger.info("Streaming fix cancelled early — bad pattern detected")
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.llm_client import LLMClient

logger = logging.getLogger(__name__)

# Minimum # of chunks before we start checking for problems
_MIN_CHUNKS_BEFORE_CHECK = 10
# Check interval — validate every N chunks to avoid excessive overhead
_CHECK_INTERVAL = 5
# Fraction of output that must be accumulated before early-cancel is allowed
_MIN_PROGRESS_FOR_CANCEL = 0.3


class _StreamingValidator:
    """Incrementally validates streaming LLM output for a code fix."""

    def __init__(
        self,
        language: str,
        known_bad_patterns: list[str] | None = None,
        expected_lines: int = 100,
    ) -> None:
        self.language = language.lower()
        self.known_bad_patterns = known_bad_patterns or []
        self.expected_lines = max(expected_lines, 20)
        self._chunks: list[str] = []
        self._chunk_count = 0
        self._cancelled = False
        self._cancel_reason = ""

    @property
    def accumulated(self) -> str:
        return "".join(self._chunks)

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    @property
    def cancel_reason(self) -> str:
        return self._cancel_reason

    def add_chunk(self, text: str) -> bool:
        """Add a text chunk. Returns False if generation should be cancelled."""
        self._chunks.append(text)
        self._chunk_count += 1

        # Don't check too early — let the LLM build up some output
        if self._chunk_count < _MIN_CHUNKS_BEFORE_CHECK:
            return True

        # Only check periodically to avoid overhead
        if self._chunk_count % _CHECK_INTERVAL != 0:
            return True

        return self._validate_accumulated()

    def _validate_accumulated(self) -> bool:
        """Check accumulated output for fatal problems."""
        text = self.accumulated
        lines = text.count("\n")
        progress = lines / self.expected_lines if self.expected_lines else 0.5

        # Don't cancel if we haven't produced enough output
        if progress < _MIN_PROGRESS_FOR_CANCEL:
            return True

        # Check 1: Known bad patterns from previous errors being reproduced
        for pattern in self.known_bad_patterns:
            if pattern and pattern in text:
                self._cancelled = True
                self._cancel_reason = f"Known bad pattern reproduced: {pattern[:80]}"
                logger.info(
                    "[StreamingFix] Cancelling: %s", self._cancel_reason,
                )
                return False

        # Check 2: Obvious structural problems
        if self.language in ("python", "py"):
            if not self._check_python_structure(text, progress):
                return False
        elif self.language in ("java",):
            if not self._check_java_structure(text, progress):
                return False
        elif self.language in ("typescript", "ts", "javascript", "js"):
            if not self._check_js_structure(text, progress):
                return False

        return True

    def _check_python_structure(self, text: str, progress: float) -> bool:
        """Check Python code structure for obvious problems."""
        # Excessive indentation errors (likely garbled output)
        indent_errors = len(re.findall(r"^\S+.*\n {4,}\S", text, re.MULTILINE))
        if indent_errors > 10 and progress > 0.5:
            self._cancelled = True
            self._cancel_reason = f"Python indentation chaos ({indent_errors} issues)"
            return False

        # Unclosed triple quotes repeated (LLM stuck in docstring)
        triple_quotes = text.count('"""') + text.count("'''")
        if triple_quotes > 6 and triple_quotes % 2 != 0 and progress > 0.6:
            self._cancelled = True
            self._cancel_reason = "Stuck in unclosed docstring/triple-quote"
            return False

        return True

    def _check_java_structure(self, text: str, progress: float) -> bool:
        """Check Java code structure for obvious problems."""
        opens = text.count("{")
        closes = text.count("}")
        # If we're >70% through and braces are wildly unbalanced
        if progress > 0.7 and opens > 0 and abs(opens - closes) > opens * 0.5:
            self._cancelled = True
            self._cancel_reason = f"Brace imbalance: {opens} open vs {closes} close"
            return False
        return True

    def _check_js_structure(self, text: str, progress: float) -> bool:
        """Check JS/TS code structure for obvious problems."""
        opens = text.count("{")
        closes = text.count("}")
        if progress > 0.7 and opens > 0 and abs(opens - closes) > opens * 0.5:
            self._cancelled = True
            self._cancel_reason = f"Brace imbalance: {opens} open vs {closes} close"
            return False

        # Check for import loop (LLM repeating imports)
        import_lines = re.findall(r"^import .+$", text, re.MULTILINE)
        if len(import_lines) > 30 and progress > 0.4:
            unique_imports = set(import_lines)
            if len(unique_imports) < len(import_lines) * 0.5:
                self._cancelled = True
                self._cancel_reason = "Excessive duplicate imports"
                return False
        return True


async def streaming_fix_file(
    llm_client: "LLMClient",
    system_prompt: str,
    user_prompt: str,
    language: str,
    known_bad_patterns: list[str] | None = None,
    expected_lines: int = 100,
) -> tuple[str, bool]:
    """Stream an LLM fix response with incremental validation.

    Args:
        llm_client: The LLM client to use for generation.
        system_prompt: System prompt for the fix.
        user_prompt: User prompt containing error context and fix instructions.
        language: Programming language of the file being fixed.
        known_bad_patterns: Short error signatures from previous failed attempts.
            If any of these appear in the streaming output, cancel early.
        expected_lines: Rough estimate of expected output lines.

    Returns:
        (content, cancelled): The accumulated content and whether generation
        was cancelled early.
    """
    validator = _StreamingValidator(
        language=language,
        known_bad_patterns=known_bad_patterns,
        expected_lines=expected_lines,
    )

    cancel_event = asyncio.Event()

    def _on_chunk(text: str) -> None:
        """Called for each streaming text delta."""
        should_continue = validator.add_chunk(text)
        if not should_continue:
            cancel_event.set()

    try:
        response = await llm_client.generate_streaming(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            on_chunk=_on_chunk,
        )
        # If we got the full response without cancellation, use it
        if not validator.cancelled:
            return response.content, False
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.debug("[StreamingFix] Generation error", exc_info=True)

    # Return whatever we accumulated
    content = validator.accumulated
    cancelled = validator.cancelled
    if cancelled:
        logger.info(
            "[StreamingFix] Cancelled after %d chunks: %s",
            validator._chunk_count,
            validator.cancel_reason,
        )

    return content, cancelled


def extract_error_signature(errors_text: str, max_signatures: int = 3) -> list[str]:
    """Extract short, distinctive error signatures for streaming detection.

    These are used as ``known_bad_patterns`` to catch the LLM reproducing
    the same errors in its fix output.

    Example: ``"cannot find symbol: variable userService"``
    """
    signatures = []
    for line in errors_text.splitlines():
        line = line.strip()
        if not line or len(line) < 10:
            continue
        # Look for lines that look like error messages
        if any(kw in line.lower() for kw in (
            "error:", "cannot", "undefined", "not found",
            "expected", "unexpected", "invalid", "missing",
        )):
            # Take the core error text (strip file paths and line numbers)
            # e.g., "src/Foo.java:12: error: cannot find symbol" → "cannot find symbol"
            match = re.search(r"(?:error|warning):\s*(.+)", line, re.IGNORECASE)
            sig = match.group(1).strip() if match else line[-80:]
            if sig and sig not in signatures:
                signatures.append(sig)
                if len(signatures) >= max_signatures:
                    break
    return signatures
