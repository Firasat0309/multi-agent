"""Tests for fixes 9-13: tier merging, virtual stubs, incremental indexing,
import validation caching, and incremental dependency graph."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.tier_scheduler import Tier, TierScheduler
from core.stub_generator import StubGenerator
from core.import_validator import ImportValidator


# ── Fix 9: Tier merging ──────────────────────────────────────────────────────

class TestTierMerging:
    def test_linear_chain_merged(self):
        """A→B→C→D should merge from 4 single-file tiers into 1."""
        scheduler = TierScheduler()
        tiers = scheduler.compute_tiers(
            file_paths=["A.java", "B.java", "C.java", "D.java"],
            file_deps={
                "A.java": [],
                "B.java": ["A.java"],
                "C.java": ["B.java"],
                "D.java": ["C.java"],
            },
        )
        # All 4 single-file tiers should merge into 1
        assert len(tiers) == 1
        assert len(tiers[0].files) == 4
        assert tiers[0].index == 0

    def test_fan_out_not_merged(self):
        """A→{B,C} should keep B and C in their own tier (fan-out)."""
        scheduler = TierScheduler()
        tiers = scheduler.compute_tiers(
            file_paths=["A.java", "B.java", "C.java"],
            file_deps={
                "A.java": [],
                "B.java": ["A.java"],
                "C.java": ["A.java"],
            },
        )
        # Tier 0: A (1 file), Tier 1: B+C (2 files, fan-out)
        # The single-file tier 0 stays alone, tier 1 has 2 files
        assert len(tiers) == 2
        assert len(tiers[0].files) == 1
        assert len(tiers[1].files) == 2

    def test_mixed_linear_and_fanout(self):
        """A→B→{C,D}→E should produce: [A,B], [C,D], [E]."""
        scheduler = TierScheduler()
        tiers = scheduler.compute_tiers(
            file_paths=["A.java", "B.java", "C.java", "D.java", "E.java"],
            file_deps={
                "A.java": [],
                "B.java": ["A.java"],
                "C.java": ["B.java"],
                "D.java": ["B.java"],
                "E.java": ["C.java", "D.java"],
            },
        )
        # Before merging: [A], [B], [C,D], [E]
        # After merging: [A,B] (two consecutive single-file tiers), [C,D], [E]
        assert len(tiers) == 3
        assert len(tiers[0].files) == 2  # A+B merged
        assert len(tiers[1].files) == 2  # C,D fan-out
        assert len(tiers[2].files) == 1  # E

    def test_no_deps_all_in_one_tier(self):
        """Files with no deps should all be in tier 0 (fan-out), no merging."""
        scheduler = TierScheduler()
        tiers = scheduler.compute_tiers(
            file_paths=["A.java", "B.java", "C.java"],
            file_deps={"A.java": [], "B.java": [], "C.java": []},
        )
        assert len(tiers) == 1
        assert len(tiers[0].files) == 3

    def test_single_file(self):
        scheduler = TierScheduler()
        tiers = scheduler.compute_tiers(
            file_paths=["A.java"],
            file_deps={"A.java": []},
        )
        assert len(tiers) == 1

    def test_tier_indices_renumbered(self):
        """After merging, tier indices should be sequential."""
        scheduler = TierScheduler()
        tiers = scheduler.compute_tiers(
            file_paths=["A.java", "B.java", "C.java", "D.java", "E.java"],
            file_deps={
                "A.java": [],
                "B.java": ["A.java"],
                "C.java": ["B.java"],
                "D.java": ["B.java"],
                "E.java": ["C.java", "D.java"],
            },
        )
        for i, tier in enumerate(tiers):
            assert tier.index == i

    def test_merge_linear_tiers_static(self):
        """Test the static merge method directly."""
        tiers = [
            Tier(0, ["A"]),
            Tier(1, ["B"]),
            Tier(2, ["C"]),
            Tier(3, ["D", "E"]),
            Tier(4, ["F"]),
        ]
        merged = TierScheduler._merge_linear_tiers(tiers)
        assert len(merged) == 3
        assert merged[0].files == ["A", "B", "C"]
        assert merged[1].files == ["D", "E"]
        assert merged[2].files == ["F"]


# ── Fix 10: Virtual stub overlay ─────────────────────────────────────────────

class TestVirtualStubOverlay:
    def test_generate_stubs_are_in_memory_only(self, tmp_path):
        gen = StubGenerator("java", tmp_path)
        created = gen.generate_stubs(["src/Foo.java"])
        assert len(created) == 1
        # Not yet on disk
        assert not (tmp_path / "src/Foo.java").exists()
        # But in the overlay
        assert "src/Foo.java" in gen._overlay

    def test_flush_writes_to_disk(self, tmp_path):
        gen = StubGenerator("java", tmp_path)
        gen.generate_stubs(["src/Foo.java"])
        written = gen.flush()
        assert written == 1
        assert (tmp_path / "src/Foo.java").exists()

    def test_flush_skips_existing_files(self, tmp_path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "Foo.java").write_text("real code")
        gen = StubGenerator("java", tmp_path)
        gen._overlay["src/Foo.java"] = "stub content"
        written = gen.flush()
        assert written == 0
        assert (tmp_path / "src" / "Foo.java").read_text() == "real code"

    def test_cleanup_removes_from_overlay_and_disk(self, tmp_path):
        gen = StubGenerator("java", tmp_path)
        gen.generate_stubs(["src/Foo.java"])
        gen.flush()
        assert (tmp_path / "src/Foo.java").exists()
        gen.cleanup_stubs(["src/Foo.java"])
        assert not (tmp_path / "src/Foo.java").exists()
        assert "src/Foo.java" not in gen._overlay

    def test_cleanup_all(self, tmp_path):
        gen = StubGenerator("java", tmp_path)
        gen.generate_stubs(["src/A.java", "src/B.java"])
        gen.flush()
        assert (tmp_path / "src/A.java").exists()
        gen.cleanup_all()
        assert not (tmp_path / "src/A.java").exists()
        assert not (tmp_path / "src/B.java").exists()
        assert len(gen._overlay) == 0

    def test_duplicate_generate_skips(self, tmp_path):
        gen = StubGenerator("java", tmp_path)
        first = gen.generate_stubs(["src/Foo.java"])
        second = gen.generate_stubs(["src/Foo.java"])
        assert len(first) == 1
        assert len(second) == 0  # Already in overlay


# ── Fix 12: Import validation caching ────────────────────────────────────────

class TestImportValidatorCache:
    def _make_python_lang(self):
        lang = MagicMock()
        lang.name = "python"
        return lang

    def test_cache_hit(self):
        validator = ImportValidator()
        lang = self._make_python_lang()
        content = "import os\n"
        known = {"os.py"}
        result1 = validator.validate("test.py", content, known, lang)
        result2 = validator.validate("test.py", content, known, lang)
        assert result1 == result2
        # Cache should have exactly 1 entry
        assert len(validator._cache) == 1

    def test_cache_miss_on_content_change(self):
        validator = ImportValidator()
        lang = self._make_python_lang()
        known = {"os.py"}
        validator.validate("test.py", "import os\n", known, lang)
        validator.validate("test.py", "import sys\n", known, lang)
        assert len(validator._cache) == 2

    def test_clear_cache(self):
        validator = ImportValidator()
        lang = self._make_python_lang()
        validator.validate("test.py", "import os\n", {"os.py"}, lang)
        assert len(validator._cache) == 1
        validator.clear_cache()
        assert len(validator._cache) == 0

    def test_cache_eviction_on_overflow(self):
        validator = ImportValidator()
        lang = self._make_python_lang()
        # Fill cache beyond the 500 limit
        for i in range(510):
            validator.validate(f"test_{i}.py", f"# file {i}\n", set(), lang)
        # After eviction, should be under 500
        assert len(validator._cache) <= 500


# ── Fix 13: Incremental dependency graph ─────────────────────────────────────

class TestIncrementalDepGraph:
    def _make_repo_manager(self, tmp_path):
        from core.repository_manager import RepositoryManager
        from core.models import FileIndex
        rm = RepositoryManager(tmp_path)

        # Simulate indexed files
        fi_a = FileIndex(path="A.java", exports=["ClassA"], imports=[], checksum="aaa")
        fi_b = FileIndex(path="B.java", exports=["ClassB"], imports=["import ClassA;"], checksum="bbb")
        fi_c = FileIndex(path="C.java", exports=["ClassC"], imports=["import ClassB;"], checksum="ccc")
        rm._repo_index.add_or_update(fi_a)
        rm._repo_index.add_or_update(fi_b)
        rm._repo_index.add_or_update(fi_c)
        return rm

    def test_full_rebuild(self, tmp_path):
        rm = self._make_repo_manager(tmp_path)
        graph = rm.rebuild_dependency_graph()
        assert graph["B.java"] == ["A.java"]
        assert graph["C.java"] == ["B.java"]
        assert graph["A.java"] == []

    def test_incremental_update_changed_file(self, tmp_path):
        rm = self._make_repo_manager(tmp_path)
        # First do a full rebuild
        rm.rebuild_dependency_graph()

        # Now change B's imports
        from core.models import FileIndex
        fi_b_new = FileIndex(
            path="B.java", exports=["ClassB"],
            imports=["import ClassA;", "import ClassC;"], checksum="bbb2"
        )
        rm._repo_index.add_or_update(fi_b_new)

        graph = rm.update_dependency_graph(changed_files=["B.java"])
        assert "A.java" in graph["B.java"]
        assert "C.java" in graph["B.java"]

    def test_incremental_update_reverse_deps(self, tmp_path):
        rm = self._make_repo_manager(tmp_path)
        rm.rebuild_dependency_graph()

        # Change A's exports (rename ClassA → ClassANew)
        from core.models import FileIndex
        fi_a = FileIndex(path="A.java", exports=["ClassANew"], imports=[], checksum="aaa2")
        rm._repo_index.add_or_update(fi_a)

        graph = rm.update_dependency_graph(changed_files=["A.java"])
        # B previously depended on A via ClassA — now ClassA no longer exists,
        # so B should lose the dep on A (its import "ClassA" no longer matches).
        assert "A.java" not in graph["B.java"]

    def test_incremental_falls_back_to_full(self, tmp_path):
        rm = self._make_repo_manager(tmp_path)
        # No existing graph file — should fall back to full rebuild
        graph = rm.update_dependency_graph(changed_files=["A.java"])
        assert "B.java" in graph
        assert (tmp_path / "dependency_graph.json").exists()

    def test_stale_files_removed(self, tmp_path):
        rm = self._make_repo_manager(tmp_path)
        rm.rebuild_dependency_graph()

        # Remove C from the index
        rm._repo_index.files = [f for f in rm._repo_index.files if f.path != "C.java"]

        graph = rm.update_dependency_graph(changed_files=["B.java"])
        assert "C.java" not in graph
