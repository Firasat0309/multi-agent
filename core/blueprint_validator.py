"""Blueprint consistency validation — P1.1.

Performs deep cross-file consistency checks beyond the basic structural
validation in ``RunPipeline._validate_blueprint``.  This catches issues that
would otherwise surface only during code generation (e.g. import cycles,
dangling exports, layer violations).

Gated behind the ``BLUEPRINT_CONSISTENCY`` feature flag.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.models import RepositoryBlueprint, FileBlueprint

logger = logging.getLogger(__name__)

# Layer ordering (lower index = lower layer)
# Files in a lower layer should NOT depend on files in a higher layer.
_LAYER_ORDER = {
    "model": 0,
    "repository": 1,
    "service": 2,
    "controller": 3,
    "config": 0,        # config is infrastructure-level
    "test": 99,          # tests can depend on anything
    "": 99,              # unspecified layer — skip checks
}


class BlueprintConsistencyValidator:
    """Validates cross-file consistency of a RepositoryBlueprint.

    Checks performed:
    1. Import/export coherence — every ``depends_on`` target must export
       something the depending file could use.
    2. Cycle detection — detects circular dependency chains.
    3. Layer violations — e.g. a model file depending on a controller.
    4. Orphan detection — files that nothing depends on and that export
       nothing (potential dead code in the blueprint).
    5. Missing language — files without a resolved language field.
    """

    def validate(self, blueprint: RepositoryBlueprint) -> list[str]:
        """Return a list of consistency issues (empty = valid)."""
        issues: list[str] = []

        fb_map: dict[str, FileBlueprint] = {
            fb.path: fb for fb in blueprint.file_blueprints
        }
        path_set = set(fb_map.keys())

        issues.extend(self._check_import_export_coherence(fb_map, path_set))
        issues.extend(self._check_cycles(fb_map))
        issues.extend(self._check_layer_violations(fb_map))
        issues.extend(self._check_orphans(fb_map, path_set))
        issues.extend(self._check_missing_language(fb_map))

        if issues:
            logger.warning(
                "Blueprint consistency: %d issue(s) found", len(issues),
            )
        return issues

    # ── Private checks ────────────────────────────────────────────────────

    @staticmethod
    def _check_import_export_coherence(
        fb_map: dict[str, FileBlueprint],
        path_set: set[str],
    ) -> list[str]:
        """Every depends_on target should exist and export at least one symbol."""
        issues: list[str] = []
        for path, fb in fb_map.items():
            for dep in fb.depends_on:
                if dep not in path_set:
                    # Already caught by structural validation — skip duplicate.
                    continue
                dep_fb = fb_map[dep]
                if not dep_fb.exports:
                    issues.append(
                        f"{path} depends on {dep} which declares no exports"
                    )
        return issues

    @staticmethod
    def _check_cycles(fb_map: dict[str, FileBlueprint]) -> list[str]:
        """Detect circular dependency chains using iterative DFS."""
        issues: list[str] = []
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {p: WHITE for p in fb_map}
        parent: dict[str, str | None] = {p: None for p in fb_map}

        for start in fb_map:
            if color[start] != WHITE:
                continue
            stack = [start]
            while stack:
                node = stack[-1]
                if color[node] == WHITE:
                    color[node] = GRAY
                    fb = fb_map[node]
                    for dep in fb.depends_on:
                        if dep not in fb_map:
                            continue
                        if color[dep] == GRAY:
                            # Found cycle — reconstruct it
                            cycle = [dep, node]
                            cur = node
                            while cur != dep and parent.get(cur) is not None:
                                cur = parent[cur]  # type: ignore[assignment]
                                cycle.append(cur)
                            cycle.reverse()
                            issues.append(
                                f"Circular dependency: {' → '.join(cycle)}"
                            )
                        elif color[dep] == WHITE:
                            parent[dep] = node
                            stack.append(dep)
                elif color[node] == GRAY:
                    color[node] = BLACK
                    stack.pop()
                else:
                    stack.pop()
        return issues

    @staticmethod
    def _check_layer_violations(fb_map: dict[str, FileBlueprint]) -> list[str]:
        """Lower layers should not depend on higher layers."""
        issues: list[str] = []
        for path, fb in fb_map.items():
            src_layer = _LAYER_ORDER.get(fb.layer, 99)
            if src_layer >= 99:
                continue  # unknown/test layer — skip
            for dep in fb.depends_on:
                dep_fb = fb_map.get(dep)
                if dep_fb is None:
                    continue
                dep_layer = _LAYER_ORDER.get(dep_fb.layer, 99)
                if dep_layer >= 99:
                    continue
                if src_layer < dep_layer:
                    issues.append(
                        f"Layer violation: {path} ({fb.layer}, L{src_layer}) "
                        f"depends on {dep} ({dep_fb.layer}, L{dep_layer})"
                    )
        return issues

    @staticmethod
    def _check_orphans(
        fb_map: dict[str, FileBlueprint],
        path_set: set[str],
    ) -> list[str]:
        """Warn about files that are never depended on and export nothing.

        These might be dead-weight in the blueprint.  Excludes entry-points
        (e.g. main.*, Application.*, index.*) and test files.
        """
        depended_on: set[str] = set()
        for fb in fb_map.values():
            depended_on.update(fb.depends_on)

        _ENTRY_PATTERNS = ("main.", "Main.", "application.", "Application.",
                           "index.", "Index.", "app.", "App.", "server.", "Server.")

        issues: list[str] = []
        for path, fb in fb_map.items():
            if path in depended_on:
                continue
            if fb.exports:
                continue
            if fb.layer == "test":
                continue
            basename = path.rsplit("/", 1)[-1] if "/" in path else path
            if any(basename.startswith(pat) for pat in _ENTRY_PATTERNS):
                continue
            issues.append(
                f"Orphan file: {path} has no exports and nothing depends on it"
            )
        return issues

    @staticmethod
    def _check_missing_language(fb_map: dict[str, FileBlueprint]) -> list[str]:
        """Files should have a resolved language before entering execution."""
        return [
            f"{path}: no language resolved"
            for path, fb in fb_map.items()
            if not fb.language
        ]
