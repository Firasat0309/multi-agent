"""Tests for the dependency tier scheduler."""

from core.tier_scheduler import Tier, TierScheduler


class TestTierScheduler:
    def setup_method(self):
        self.scheduler = TierScheduler()

    # ── Basic tier computation ─────────────────────────────────────

    def test_no_deps_single_tier(self):
        tiers = self.scheduler.compute_tiers(
            file_paths=["a.java", "b.java", "c.java"],
            file_deps={"a.java": [], "b.java": [], "c.java": []},
        )
        assert len(tiers) == 1
        assert set(tiers[0].files) == {"a.java", "b.java", "c.java"}

    def test_linear_chain(self):
        tiers = self.scheduler.compute_tiers(
            file_paths=["model.java", "repo.java", "service.java"],
            file_deps={
                "model.java": [],
                "repo.java": ["model.java"],
                "service.java": ["repo.java"],
            },
        )
        assert len(tiers) == 3
        assert tiers[0].files == ["model.java"]
        assert tiers[1].files == ["repo.java"]
        assert tiers[2].files == ["service.java"]

    def test_diamond_deps(self):
        tiers = self.scheduler.compute_tiers(
            file_paths=["base.java", "left.java", "right.java", "top.java"],
            file_deps={
                "base.java": [],
                "left.java": ["base.java"],
                "right.java": ["base.java"],
                "top.java": ["left.java", "right.java"],
            },
        )
        assert len(tiers) == 3
        assert tiers[0].files == ["base.java"]
        assert set(tiers[1].files) == {"left.java", "right.java"}
        assert tiers[2].files == ["top.java"]

    def test_external_deps_ignored(self):
        """Dependencies not in file_paths are treated as external."""
        tiers = self.scheduler.compute_tiers(
            file_paths=["app.java"],
            file_deps={"app.java": ["spring-framework"]},
        )
        assert len(tiers) == 1
        assert tiers[0].files == ["app.java"]

    def test_self_deps_ignored(self):
        tiers = self.scheduler.compute_tiers(
            file_paths=["a.java"],
            file_deps={"a.java": ["a.java"]},
        )
        assert len(tiers) == 1

    # ── Cycle handling ─────────────────────────────────────────────

    def test_cycle_is_broken(self):
        """Circular deps don't cause infinite loops."""
        tiers = self.scheduler.compute_tiers(
            file_paths=["a.java", "b.java"],
            file_deps={
                "a.java": ["b.java"],
                "b.java": ["a.java"],
            },
        )
        # Both should eventually be scheduled
        all_files = set()
        for t in tiers:
            all_files.update(t.files)
        assert all_files == {"a.java", "b.java"}

    # ── Tier indices ───────────────────────────────────────────────

    def test_tier_indices_are_sequential(self):
        tiers = self.scheduler.compute_tiers(
            file_paths=["a.java", "b.java", "c.java"],
            file_deps={
                "a.java": [],
                "b.java": ["a.java"],
                "c.java": ["b.java"],
            },
        )
        for i, tier in enumerate(tiers):
            assert tier.index == i

    # ── should_checkpoint ──────────────────────────────────────────

    def test_checkpoint_for_compiled(self):
        tier = Tier(index=0, files=["a.java", "b.java"])
        assert self.scheduler.should_checkpoint(tier, compiled=True)

    def test_no_checkpoint_for_interpreted(self):
        tier = Tier(index=0, files=["a.py", "b.py"])
        assert not self.scheduler.should_checkpoint(tier, compiled=False)

    # ── get_tier_for_file ──────────────────────────────────────────

    def test_get_tier_for_file(self):
        tiers = [
            Tier(index=0, files=["model.java"]),
            Tier(index=1, files=["service.java"]),
        ]
        assert self.scheduler.get_tier_for_file("model.java", tiers) == 0
        assert self.scheduler.get_tier_for_file("service.java", tiers) == 1
        assert self.scheduler.get_tier_for_file("unknown.java", tiers) is None

    # ── Real-world scenario ────────────────────────────────────────

    def test_spring_boot_project(self):
        """Typical Spring Boot project tiers correctly."""
        tiers = self.scheduler.compute_tiers(
            file_paths=[
                "models/User.java",
                "models/Order.java",
                "repos/UserRepo.java",
                "repos/OrderRepo.java",
                "services/UserService.java",
                "services/OrderService.java",
                "controllers/UserController.java",
                "controllers/OrderController.java",
                "config/AppConfig.java",
                "Application.java",
            ],
            file_deps={
                "models/User.java": [],
                "models/Order.java": ["models/User.java"],
                "repos/UserRepo.java": ["models/User.java"],
                "repos/OrderRepo.java": ["models/Order.java"],
                "services/UserService.java": ["repos/UserRepo.java"],
                "services/OrderService.java": ["repos/OrderRepo.java", "services/UserService.java"],
                "controllers/UserController.java": ["services/UserService.java"],
                "controllers/OrderController.java": ["services/OrderService.java"],
                "config/AppConfig.java": [],
                "Application.java": ["config/AppConfig.java"],
            },
        )
        # Should have at least 3 tiers: models → repos → services → controllers
        assert len(tiers) >= 3

        # Model files should be in tier 0
        tier0_files = set(tiers[0].files)
        assert "models/User.java" in tier0_files
        assert "config/AppConfig.java" in tier0_files

        # Controllers should be in a later tier than services
        ctrl_tier = self.scheduler.get_tier_for_file("controllers/UserController.java", tiers)
        svc_tier = self.scheduler.get_tier_for_file("services/UserService.java", tiers)
        assert ctrl_tier is not None and svc_tier is not None
        assert ctrl_tier > svc_tier
