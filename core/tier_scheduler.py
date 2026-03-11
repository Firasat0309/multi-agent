"""Dependency-tier scheduler for incremental build verification.

Groups files into tiers based on the dependency graph so that foundational
files (models, interfaces) are generated and verified before files that
depend on them (services, controllers).

Tier 0: Files with zero dependencies
Tier 1: Files that depend only on Tier 0
Tier 2: Files that depend only on Tier 0 and Tier 1
...

This enables repo-level build checkpoints between tiers — after all Tier 0
files are generated and reviewed, a build checkpoint verifies they compile
cleanly before Tier 1 files start generating.  Errors in foundational types
are caught and fixed before dependent files even begin, eliminating cascading
build failures.

Usage::

    scheduler = TierScheduler()
    tiers = scheduler.compute_tiers(
        file_paths=["User.java", "UserRepo.java", "UserService.java"],
        file_deps={
            "User.java": [],
            "UserRepo.java": ["User.java"],
            "UserService.java": ["UserRepo.java", "User.java"],
        },
    )
    # tiers = [{"User.java"}, {"UserRepo.java"}, {"UserService.java"}]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Tier:
    """A group of files at the same dependency depth."""

    index: int
    files: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.files)

    def __repr__(self) -> str:
        return f"Tier({self.index}, {len(self.files)} files)"


class TierScheduler:
    """Computes dependency tiers for incremental generation and verification."""

    def compute_tiers(
        self,
        file_paths: list[str],
        file_deps: dict[str, list[str]],
    ) -> list[Tier]:
        """Group files into tiers by dependency depth.

        Args:
            file_paths: All files to be generated.
            file_deps: Mapping of file → list of dependency file paths.
                       Dependencies not in ``file_paths`` are treated as
                       external (already available) and ignored.

        Returns:
            Ordered list of ``Tier`` objects from shallowest to deepest.
            Cycle members are placed in the tier with their lowest-depth
            dependency to avoid deadlock.
        """
        all_files = set(file_paths)

        # Filter deps to only include internal files
        internal_deps: dict[str, set[str]] = {}
        for path in file_paths:
            deps = file_deps.get(path, [])
            internal_deps[path] = {d for d in deps if d in all_files and d != path}

        tiers: list[Tier] = []
        remaining = set(file_paths)
        assigned: set[str] = set()

        tier_index = 0
        max_iterations = len(file_paths) + 1  # safety guard

        while remaining and tier_index < max_iterations:
            # Files whose internal dependencies are all in previous tiers
            tier_files = [
                f for f in remaining
                if internal_deps[f].issubset(assigned)
            ]

            if not tier_files:
                # Cycle detected — break it by picking files with the fewest
                # unresolved dependencies (greedy cycle breaking)
                logger.warning(
                    "Dependency cycle detected among %d remaining files — "
                    "breaking cycle by including least-blocked files",
                    len(remaining),
                )
                unresolved_counts = {
                    f: len(internal_deps[f] - assigned)
                    for f in remaining
                }
                min_unresolved = min(unresolved_counts.values())
                tier_files = [
                    f for f, count in unresolved_counts.items()
                    if count == min_unresolved
                ]

            tier = Tier(index=tier_index, files=sorted(tier_files))
            tiers.append(tier)

            assigned.update(tier_files)
            remaining -= set(tier_files)
            tier_index += 1

        logger.info(
            "Computed %d dependency tiers for %d files: %s",
            len(tiers),
            len(file_paths),
            [(t.index, len(t)) for t in tiers],
        )
        return tiers

    def should_checkpoint(
        self,
        tier: Tier,
        compiled: bool,
        min_tier_size: int = 1,
    ) -> bool:
        """Determine whether a build checkpoint should run after this tier.

        Checkpoints are only useful for compiled languages.  For interpreted
        languages, per-file syntax checks (linting) are sufficient.

        Args:
            tier: The tier that just completed.
            compiled: Whether the target language requires compilation.
            min_tier_size: Minimum number of files in a tier to warrant a
                          checkpoint (avoids overhead for tiny tiers).

        Returns:
            True if a build checkpoint should run after this tier.
        """
        if not compiled:
            return False
        return len(tier) >= min_tier_size

    def get_tier_for_file(
        self,
        file_path: str,
        tiers: list[Tier],
    ) -> int | None:
        """Return the tier index for a given file path, or None if not found."""
        for tier in tiers:
            if file_path in tier.files:
                return tier.index
        return None
