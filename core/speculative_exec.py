"""Speculative execution — start dependent files before deps complete.

In the standard pipeline, a service file must wait for its model/repository
dependencies to fully generate and build before starting.  With speculative
execution, we start generating the service file using BLUEPRINT information
(expected exports, interfaces) while the dependencies are still in progress.

If the dependencies succeed, the speculative output is validated against
the actual generated code.  If they fail, the speculative output is discarded.

This can save 15-30s per file in deep dependency chains.

Gated behind the ``SPECULATIVE_EXEC`` feature flag.

Design:
  - Files with ALL dependencies in GENERATING/BUILDING state (not FAILED)
    are eligible for speculative generation.
  - The speculative prompt includes blueprint exports/interfaces instead
    of actual file content.
  - When deps complete, a lightweight validation check compares speculative
    assumptions against actual exports.
  - If validation passes → skip generation, go straight to build.
  - If validation fails → discard speculative output, generate normally.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.models import FileBlueprint, RepositoryBlueprint
    from core.state_machine import LifecycleEngine, FilePhase

logger = logging.getLogger(__name__)


class SpeculativeResult:
    """Result of a speculative generation attempt."""

    def __init__(
        self,
        file_path: str,
        content: str,
        assumed_exports: dict[str, list[str]],
    ) -> None:
        self.file_path = file_path
        self.content = content
        # {dep_path: [assumed_export_names]} — what we assumed deps would export
        self.assumed_exports = assumed_exports
        self.valid: bool | None = None  # None = not yet validated

    def validate_against_actuals(
        self,
        actual_contents: dict[str, str],
        blueprint: RepositoryBlueprint,
    ) -> bool:
        """Check if speculative assumptions still hold.

        Compares assumed exports against the actual generated file contents.
        If all assumed symbols are found in the actual files, the speculative
        output is valid.
        """
        fb_map = {fb.path: fb for fb in blueprint.file_blueprints}

        for dep_path, assumed in self.assumed_exports.items():
            actual = actual_contents.get(dep_path, "")
            if not actual:
                # Dependency didn't generate — speculative output invalid
                self.valid = False
                return False

            # Check if assumed exports appear in the actual content
            for symbol in assumed:
                if symbol not in actual:
                    logger.info(
                        "[Speculative] %s: assumed export '%s' from %s not found in actual",
                        self.file_path, symbol, dep_path,
                    )
                    self.valid = False
                    return False

        self.valid = True
        return True


def get_eligible_files(
    engine: "LifecycleEngine",
    all_files: list[str],
) -> list[str]:
    """Return files eligible for speculative generation.

    A file is eligible if:
    1. It is in PENDING state.
    2. ALL its dependencies are currently in-progress (not failed).
    3. It has at least one dependency that hasn't completed yet.
    """
    from core.state_machine import FilePhase

    eligible = []
    for fp in all_files:
        lc = engine.get_lifecycle(fp)
        if lc.phase != FilePhase.PENDING:
            continue

        deps = lc.depends_on
        if not deps:
            continue

        has_incomplete = False
        all_nonfailed = True
        for dep in deps:
            if dep not in engine._lifecycles:
                continue
            dep_lc = engine.get_lifecycle(dep)
            if dep_lc.phase == FilePhase.FAILED:
                all_nonfailed = False
                break
            if not dep_lc.is_terminal:
                has_incomplete = True

        if all_nonfailed and has_incomplete:
            eligible.append(fp)

    return eligible


def build_speculative_prompt(
    file_bp: "FileBlueprint",
    blueprint: "RepositoryBlueprint",
) -> tuple[str, dict[str, list[str]]]:
    """Build a generation prompt using blueprint info instead of actual dep content.

    Returns (prompt_text, assumed_exports) where assumed_exports maps each
    dependency to the list of exports we told the LLM about.
    """
    fb_map = {fb.path: fb for fb in blueprint.file_blueprints}
    assumed_exports: dict[str, list[str]] = {}

    dep_descriptions = []
    for dep_path in file_bp.depends_on:
        dep_fb = fb_map.get(dep_path)
        if dep_fb is None:
            continue
        exports = dep_fb.exports or []
        assumed_exports[dep_path] = exports
        dep_descriptions.append(
            f"- {dep_path} (layer: {dep_fb.layer})\n"
            f"  Purpose: {dep_fb.purpose}\n"
            f"  Exports: {', '.join(exports) if exports else 'TBD'}"
        )

    deps_section = "\n".join(dep_descriptions) if dep_descriptions else "No dependencies."

    prompt = (
        f"Generate the file: {file_bp.path}\n"
        f"Purpose: {file_bp.purpose}\n"
        f"Layer: {file_bp.layer}\n"
        f"Language: {file_bp.language}\n\n"
        f"## Dependencies (from blueprint — actual files may differ slightly)\n"
        f"{deps_section}\n\n"
        f"NOTE: Use the EXPECTED interfaces from the blueprint. "
        f"The actual dependency files are still being generated. "
        f"Focus on implementing correct logic using the declared exports above."
    )

    return prompt, assumed_exports
