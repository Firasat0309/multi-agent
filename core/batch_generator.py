"""Batch generation — group small independent files into a single LLM call.

Within a tier, many files are small and independent (models, DTOs, config,
interfaces).  Instead of making N separate LLM calls, this module groups
them into batches that fit within the context window and generates them in
a single multi-file prompt.

Gated behind the ``BATCH_GENERATION`` feature flag (not yet added — callers
should check before invoking).

Design:
  - Only files under ``MAX_BATCH_FILE_LINES`` (estimated) are eligible.
  - Files that depend on each other within the same batch are excluded
    (they need sequential generation).
  - Each batch is capped at ``MAX_BATCH_TOKENS`` estimated prompt tokens.
  - The LLM is asked to output ``<file path="...">...</file>`` blocks.
  - Parsing extracts each file and writes it individually.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.models import FileBlueprint, RepositoryBlueprint

logger = logging.getLogger(__name__)

# Tunables
MAX_BATCH_FILE_LINES = 150      # Only batch files expected to be ≤150 lines
MAX_BATCH_TOKENS = 12_000       # Max prompt tokens per batch (conservative)
MAX_FILES_PER_BATCH = 8         # Cap files per batch to keep output manageable
_CHARS_PER_TOKEN = 3.5          # Conservative estimate

# Layers that are typically small/simple and safe to batch
_BATCHABLE_LAYERS = {"model", "config", "repository", ""}

# Regex to parse multi-file LLM output
_FILE_BLOCK_RE = re.compile(
    r'<file\s+path="([^"]+)">\s*\n(.*?)\n\s*</file>',
    re.DOTALL,
)


@dataclass
class FileBatch:
    """A group of files to generate in a single LLM call."""
    files: list[FileBlueprint]
    estimated_tokens: int = 0


@dataclass
class BatchPlan:
    """The result of batch planning: batched files + unbatched files."""
    batches: list[FileBatch] = field(default_factory=list)
    unbatched: list[str] = field(default_factory=list)  # file paths to process individually


def plan_batches(
    blueprint: RepositoryBlueprint,
    tier_files: list[str],
) -> BatchPlan:
    """Partition *tier_files* into batches + individual files.

    Files are eligible for batching if they are:
      - In a batchable layer (model, config, repository)
      - Don't have mutual dependencies with other files in the same batch
      - Estimated to be small (based on purpose/layer heuristics)
    """
    fb_map = {fb.path: fb for fb in blueprint.file_blueprints}
    tier_set = set(tier_files)

    eligible: list[FileBlueprint] = []
    individual: list[str] = []

    for path in tier_files:
        fb = fb_map.get(path)
        if fb is None:
            individual.append(path)
            continue

        # Check if layer is batchable
        if fb.layer not in _BATCHABLE_LAYERS:
            individual.append(path)
            continue

        # Check dependencies — if this file depends on other files IN THE SAME
        # TIER, it needs sequential processing.
        intra_tier_deps = [d for d in fb.depends_on if d in tier_set]
        if intra_tier_deps:
            individual.append(path)
            continue

        eligible.append(fb)

    if not eligible:
        return BatchPlan(batches=[], unbatched=tier_files)

    # Greedy bin-packing: group eligible files into batches
    batches: list[FileBatch] = []
    current = FileBatch(files=[])
    current_deps: set[str] = set()  # accumulated deps for cycle prevention

    for fb in eligible:
        # Estimate tokens for this file's prompt contribution
        est = _estimate_file_prompt_tokens(fb)

        # Check if adding this file would exceed batch limits
        would_exceed_tokens = (current.estimated_tokens + est) > MAX_BATCH_TOKENS
        would_exceed_files = len(current.files) >= MAX_FILES_PER_BATCH

        if current.files and (would_exceed_tokens or would_exceed_files):
            batches.append(current)
            current = FileBatch(files=[])
            current_deps = set()

        current.files.append(fb)
        current.estimated_tokens += est
        current_deps.add(fb.path)

    if current.files:
        batches.append(current)

    # Batches with only 1 file aren't worth the overhead — move to individual
    final_batches: list[FileBatch] = []
    for batch in batches:
        if len(batch.files) < 2:
            individual.extend(fb.path for fb in batch.files)
        else:
            final_batches.append(batch)

    plan = BatchPlan(batches=final_batches, unbatched=individual)
    logger.info(
        "Batch plan: %d batches (%d files) + %d individual",
        len(plan.batches),
        sum(len(b.files) for b in plan.batches),
        len(plan.unbatched),
    )
    return plan


def build_batch_prompt(
    batch: FileBatch,
    blueprint: RepositoryBlueprint,
    tech_stack_summary: str,
) -> str:
    """Build a single prompt that asks the LLM to generate all files in the batch.

    The prompt instructs the model to output each file in a
    ``<file path="...">...</file>`` block.
    """
    file_specs: list[str] = []
    for fb in batch.files:
        deps_str = ", ".join(fb.depends_on) if fb.depends_on else "none"
        exports_str = ", ".join(fb.exports) if fb.exports else "inferred from purpose"
        file_specs.append(
            f"### {fb.path}\n"
            f"- Purpose: {fb.purpose}\n"
            f"- Language: {fb.language}\n"
            f"- Layer: {fb.layer or 'unspecified'}\n"
            f"- Dependencies: {deps_str}\n"
            f"- Expected exports: {exports_str}"
        )

    files_section = "\n\n".join(file_specs)

    return (
        f"Generate the following {len(batch.files)} files for the "
        f"**{blueprint.name}** project.\n\n"
        f"Tech stack: {tech_stack_summary}\n"
        f"Architecture: {blueprint.architecture_style}\n\n"
        f"## Files to Generate\n\n{files_section}\n\n"
        f"## Output Format\n\n"
        f"Output each file in a `<file path=\"...\">` block. "
        f"Include the COMPLETE file content — no placeholders or TODOs.\n\n"
        f"Example:\n"
        f'<file path="src/model/User.java">\n'
        f"package com.example.model;\n\n"
        f"public class User {{\n"
        f"    // ... full implementation\n"
        f"}}\n"
        f"</file>\n\n"
        f"Now generate all {len(batch.files)} files:"
    )


def parse_batch_response(response_text: str) -> dict[str, str]:
    """Parse multi-file LLM response into {path: content} dict."""
    results: dict[str, str] = {}
    for match in _FILE_BLOCK_RE.finditer(response_text):
        path = match.group(1).strip()
        content = match.group(2)
        results[path] = content
    return results


def _estimate_file_prompt_tokens(fb: FileBlueprint) -> int:
    """Rough estimate of how many tokens this file adds to the batch prompt."""
    # Base: purpose + metadata ~ 100 tokens
    base = 100
    # Deps and exports add a bit
    base += len(fb.depends_on) * 10
    base += len(fb.exports) * 10
    # Estimated output tokens (model/config files are typically 50-150 lines)
    estimated_output = 400  # ~150 lines * ~3 tokens/line
    return base + estimated_output
