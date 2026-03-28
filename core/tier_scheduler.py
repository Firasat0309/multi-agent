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
from pathlib import PurePosixPath

logger = logging.getLogger(__name__)

# Build/project config files that should be generated AFTER all source files
# so the LLM can see actual imports and include correct dependencies.
_BUILD_CONFIG_NAMES: frozenset[str] = frozenset({
    "pom.xml", "build.gradle", "build.gradle.kts",
    "settings.gradle", "settings.gradle.kts",
    "package.json", "cargo.toml", "go.mod", "go.sum",
    "requirements.txt", "pyproject.toml", "setup.py",
})
_BUILD_CONFIG_EXTENSIONS: frozenset[str] = frozenset({".csproj", ".fsproj", ".vbproj"})


def _is_build_config(file_path: str) -> bool:
    """Check if a file is a build/project configuration file."""
    name = PurePosixPath(file_path.replace("\\", "/")).name.lower()
    if name in _BUILD_CONFIG_NAMES:
        return True
    suffix = PurePosixPath(name).suffix.lower()
    return suffix in _BUILD_CONFIG_EXTENSIONS


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

        # Move build config files (pom.xml, build.gradle, package.json, etc.)
        # to the final tier so they're generated AFTER all source files.
        # This lets the LLM see actual imports and include correct dependencies
        # instead of guessing what the source code will need.
        config_set = {f for f in file_paths if _is_build_config(f)}
        if config_set:
            for tier in tiers:
                tier.files = [f for f in tier.files if f not in config_set]
            # Remove empty tiers and renumber
            tiers = [t for t in tiers if t.files]
            for i, t in enumerate(tiers):
                t.index = i
            # Add config files as the last tier
            config_tier = Tier(index=len(tiers), files=sorted(config_set))
            tiers.append(config_tier)
            logger.info(
                "Moved %d build config file(s) to final tier %d: %s",
                len(config_set), config_tier.index, sorted(config_set),
            )

        # Merge adjacent single-file tiers that form linear dependency chains.
        # A→B→C→D produces 4 tiers of 1 file each; they can safely be compiled
        # together in a single tier because each file only depends on the one
        # before it.  This reduces the number of build checkpoints from N to ~1
        # for deep linear chains while preserving correctness for fan-out tiers
        # (multiple files at the same depth must still build together).
        tiers = self._merge_linear_tiers(tiers)

        logger.info(
            "Computed %d dependency tiers for %d files: %s",
            len(tiers),
            len(file_paths),
            [(t.index, len(t)) for t in tiers],
        )
        return tiers

    @staticmethod
    def _merge_linear_tiers(tiers: list[Tier]) -> list[Tier]:
        """NO-OP: tier merging is disabled to preserve cross-file consistency.

        Previously, adjacent single-file tiers were merged so that a linear
        chain like Model → Service → Controller would become one tier and
        generate concurrently.  This caused cross-file method name mismatches:
        Controller couldn't read Service's actual generated code because
        Service hadn't been written to disk yet (they were generating in
        parallel within the same tier).

        Keeping each dependency depth as its own tier ensures that when file B
        depends on file A, A is fully generated (and its AST stub shows real
        method signatures) before B starts generating.

        The tradeoff is more build checkpoints, but each is fast for small
        tiers (incremental compile), and correctness >> speed.
        """
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
