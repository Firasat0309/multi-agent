"""Task DAG engine with topological execution ordering.

Two execution modes:
  1. **Legacy DAG** — ``TaskGraphBuilder.build_from_blueprint()`` creates a static
     graph with pre-baked FIX_CODE tasks.  Used when lifecycle mode is disabled.
  2. **Lifecycle + global DAG** — ``LifecyclePlanBuilder.build()`` creates:
     - A ``LifecycleEngine`` for per-file Generate→Review→Fix→Test cycles
     - A slim ``TaskGraph`` for global phases (security, module review, deploy, docs)
     The per-file lifecycle is event-driven; the global DAG runs after all files
     reach a terminal state.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Iterator

import networkx as nx

from core.models import Task, TaskStatus, TaskType, RepositoryBlueprint, FileBlueprint, ChangePlan, ChangeAction
from core.state_machine import LifecycleEngine

if TYPE_CHECKING:
    from memory.dependency_graph import DependencyGraphStore

logger = logging.getLogger(__name__)


class TaskGraph:
    """Manages the task DAG and provides topological execution ordering."""

    def __init__(self) -> None:
        self._graph = nx.DiGraph()
        self._tasks: dict[int, Task] = {}
        self._next_id = 1

    @property
    def tasks(self) -> dict[int, Task]:
        return dict(self._tasks)

    def add_task(self, task: Task) -> None:
        self._tasks[task.task_id] = task
        self._graph.add_node(task.task_id)
        for dep_id in task.dependencies:
            self._graph.add_edge(dep_id, task.task_id)
        self._next_id = max(self._next_id, task.task_id + 1)

    def get_ready_tasks(self) -> list[Task]:
        """Return tasks whose dependencies are all completed."""
        ready = []
        for task_id, task in self._tasks.items():
            if task.status != TaskStatus.PENDING:
                continue
            deps_met = all(
                self._tasks[d].status == TaskStatus.COMPLETED
                for d in task.dependencies
                if d in self._tasks
            )
            if deps_met:
                task.status = TaskStatus.READY
                ready.append(task)
        return ready

    def get_execution_order(self) -> list[int]:
        """Return task IDs in topological order."""
        try:
            return list(nx.topological_sort(self._graph))
        except nx.NetworkXUnfeasible:
            logger.error("Cycle detected in task graph!")
            raise ValueError("Task graph contains a cycle")

    def mark_completed(self, task_id: int) -> None:
        if task_id in self._tasks:
            self._tasks[task_id].status = TaskStatus.COMPLETED

    def mark_failed(self, task_id: int) -> None:
        if task_id in self._tasks:
            task = self._tasks[task_id]
            task.status = TaskStatus.FAILED
            # Block downstream tasks
            for downstream in nx.descendants(self._graph, task_id):
                if downstream in self._tasks:
                    self._tasks[downstream].status = TaskStatus.BLOCKED

    def has_remaining_tasks(self) -> bool:
        return any(
            t.status in (TaskStatus.PENDING, TaskStatus.READY, TaskStatus.IN_PROGRESS)
            for t in self._tasks.values()
        )

    def get_task(self, task_id: int) -> Task | None:
        return self._tasks.get(task_id)

    def get_stats(self) -> dict[str, int]:
        stats: dict[str, int] = defaultdict(int)
        for task in self._tasks.values():
            stats[task.status.value] += 1
        return dict(stats)

    def validate(self) -> list[str]:
        """Validate the task graph for issues."""
        errors: list[str] = []
        if not nx.is_directed_acyclic_graph(self._graph):
            errors.append("Task graph contains cycles")
        for task_id, task in self._tasks.items():
            for dep in task.dependencies:
                if dep not in self._tasks:
                    errors.append(f"Task {task_id} depends on missing task {dep}")
        return errors


class TaskGraphBuilder:
    """Builds a task graph from a repository blueprint."""

    def __init__(self) -> None:
        self._next_id = 1

    def _alloc_id(self) -> int:
        tid = self._next_id
        self._next_id += 1
        return tid

    def build_from_blueprint(self, blueprint: RepositoryBlueprint) -> TaskGraph:
        graph = TaskGraph()
        file_task_map: dict[str, int] = {}

        # Pre-validate: warn about depends_on references to files not in the blueprint
        known_paths = {fb.path for fb in blueprint.file_blueprints}
        for fb in blueprint.file_blueprints:
            for dep_path in fb.depends_on:
                if dep_path not in known_paths:
                    logger.warning(
                        f"Blueprint: {fb.path} depends on '{dep_path}' which is "
                        f"not in the blueprint — dependency will be ignored"
                    )

        # Phase 1: Generate code files (respecting dependency order)
        for fb in blueprint.file_blueprints:
            deps = [file_task_map[d] for d in fb.depends_on if d in file_task_map]
            task = Task(
                task_id=self._alloc_id(),
                task_type=TaskType.GENERATE_FILE,
                file=fb.path,
                description=f"Generate {fb.path}: {fb.purpose}",
                dependencies=deps,
            )
            graph.add_task(task)
            file_task_map[fb.path] = task.task_id

        # Phase 2: File-level reviews
        review_tasks: list[int] = []
        review_task_map: dict[str, int] = {}
        for fb in blueprint.file_blueprints:
            gen_task_id = file_task_map[fb.path]
            task = Task(
                task_id=self._alloc_id(),
                task_type=TaskType.REVIEW_FILE,
                file=fb.path,
                description=f"Review {fb.path}",
                dependencies=[gen_task_id],
            )
            graph.add_task(task)
            review_tasks.append(task.task_id)
            review_task_map[fb.path] = task.task_id

        # Phase 2.5: Fix code based on review findings
        fix_task_map: dict[str, int] = {}
        for fb in blueprint.file_blueprints:
            review_task_id = review_task_map[fb.path]
            fix_task = Task(
                task_id=self._alloc_id(),
                task_type=TaskType.FIX_CODE,
                file=fb.path,
                description=f"Fix {fb.path} based on review",
                dependencies=[review_task_id],
                metadata={"review_task_id": review_task_id},
            )
            graph.add_task(fix_task)
            fix_task_map[fb.path] = fix_task.task_id

        # Phase 3: Generate tests
        # IMPORTANT: Each test task depends on ALL fix tasks, not just its own.
        # For compiled languages (Java, Go, Rust, C#), the test command compiles
        # the entire project.  If UserService.java hasn't been generated yet when
        # UserTest runs `mvn test`, compilation fails and the fix loop wastes all
        # attempts on errors from files that don't exist yet.
        # By depending on all fix tasks, we guarantee every source file is written
        # and review-fixed before ANY test task begins.
        all_fix_ids = list(fix_task_map.values())
        test_tasks: list[int] = []
        for fb in blueprint.file_blueprints:
            if fb.layer in ("test", "config", "deploy"):
                continue
            task = Task(
                task_id=self._alloc_id(),
                task_type=TaskType.GENERATE_TEST,
                file=fb.path,
                description=f"Generate tests for {fb.path}",
                dependencies=all_fix_ids,
            )
            graph.add_task(task)
            test_tasks.append(task.task_id)

        # Phase 4: Security scan (after all fixes applied)
        all_gen_ids = list(fix_task_map.values())
        sec_task = Task(
            task_id=self._alloc_id(),
            task_type=TaskType.SECURITY_SCAN,
            file="*",
            description="Security scan of entire codebase",
            dependencies=all_gen_ids,
        )
        graph.add_task(sec_task)

        # Phase 5: Module-level review (after all per-file fixes)
        all_fix_ids = list(fix_task_map.values())
        mod_review = Task(
            task_id=self._alloc_id(),
            task_type=TaskType.REVIEW_MODULE,
            file="*",
            description="Module consistency review",
            dependencies=all_fix_ids,
        )
        graph.add_task(mod_review)

        # Phase 5.5: Module-level fix — fix cross-file consistency issues
        # Each source file gets a fix task that sees the module review findings
        mod_fix_ids: list[int] = []
        for fb in blueprint.file_blueprints:
            if fb.layer in ("config", "deploy"):
                continue
            mod_fix = Task(
                task_id=self._alloc_id(),
                task_type=TaskType.FIX_CODE,
                file=fb.path,
                description=f"Fix {fb.path} based on module review",
                dependencies=[mod_review.task_id],
                metadata={"review_task_id": mod_review.task_id},
            )
            graph.add_task(mod_fix)
            mod_fix_ids.append(mod_fix.task_id)

        # Phase 6: Architecture review (after module fixes + security)
        arch_review = Task(
            task_id=self._alloc_id(),
            task_type=TaskType.REVIEW_ARCHITECTURE,
            file="*",
            description="Architecture review for dependency cycles and layer violations",
            dependencies=mod_fix_ids + [sec_task.task_id],
        )
        graph.add_task(arch_review)

        # Phase 7: Deployment artifacts
        deploy_task = Task(
            task_id=self._alloc_id(),
            task_type=TaskType.GENERATE_DEPLOY,
            file="deploy/",
            description="Generate Dockerfile and Kubernetes manifests",
            dependencies=[arch_review.task_id],
        )
        graph.add_task(deploy_task)

        # Phase 8: Documentation
        docs_task = Task(
            task_id=self._alloc_id(),
            task_type=TaskType.GENERATE_DOCS,
            file="docs/",
            description="Generate README, changelog, API docs",
            dependencies=[arch_review.task_id],
        )
        graph.add_task(docs_task)

        errors = graph.validate()
        if errors:
            raise ValueError(f"Task graph validation failed: {errors}")

        logger.info(f"Built task graph with {len(graph.tasks)} tasks")
        return graph


class LifecyclePlanBuilder:
    """Builds a LifecycleEngine + global TaskGraph from a blueprint.

    The per-file lifecycle (Generate → Review → Fix → Test) is handled by
    the ``LifecycleEngine``.  The global phases (security scan, module review,
    architecture review, deploy, docs) are still managed as a small DAG.
    """

    def __init__(self) -> None:
        self._next_id = 1

    def _alloc_id(self) -> int:
        tid = self._next_id
        self._next_id += 1
        return tid

    def build(
        self,
        blueprint: RepositoryBlueprint,
        *,
        max_review_fixes: int = 2,
        max_test_fixes: int = 3,
    ) -> tuple[LifecycleEngine, TaskGraph]:
        """Build a lifecycle engine + global task graph.

        Returns:
            (lifecycle_engine, global_task_graph)
        """
        # ── Validate blueprint deps ─────────────────────────────────
        known_paths = {fb.path for fb in blueprint.file_blueprints}
        for fb in blueprint.file_blueprints:
            for dep_path in fb.depends_on:
                if dep_path not in known_paths:
                    logger.warning(
                        "Blueprint: %s depends on '%s' which is not in the "
                        "blueprint — dependency will be ignored",
                        fb.path, dep_path,
                    )

        # ── Build per-file lifecycle engine ──────────────────────────
        file_paths = [fb.path for fb in blueprint.file_blueprints]
        file_deps = {
            fb.path: [d for d in fb.depends_on if d in known_paths]
            for fb in blueprint.file_blueprints
        }

        engine = LifecycleEngine(
            file_paths=file_paths,
            file_deps=file_deps,
            max_review_fixes=max_review_fixes,
            max_test_fixes=max_test_fixes,
        )

        # Mark config/deploy/test files to skip testing
        for fb in blueprint.file_blueprints:
            if fb.layer in ("test", "config", "deploy"):
                engine.skip_testing(fb.path)

        # ── Build global task graph (runs after all files terminal) ──
        global_graph = TaskGraph()

        # Sentinel task: marks lifecycle completion (auto-completed by executor)
        lifecycle_done = Task(
            task_id=self._alloc_id(),
            task_type=TaskType.REVIEW_FILE,  # placeholder type
            file="*",
            description="[sentinel] All per-file lifecycles completed",
            dependencies=[],
        )
        global_graph.add_task(lifecycle_done)

        # Security scan
        sec_task = Task(
            task_id=self._alloc_id(),
            task_type=TaskType.SECURITY_SCAN,
            file="*",
            description="Security scan of entire codebase",
            dependencies=[lifecycle_done.task_id],
        )
        global_graph.add_task(sec_task)

        # Module review
        mod_review = Task(
            task_id=self._alloc_id(),
            task_type=TaskType.REVIEW_MODULE,
            file="*",
            description="Module consistency review",
            dependencies=[lifecycle_done.task_id],
        )
        global_graph.add_task(mod_review)

        # Module fix (per-file, depends on module review)
        mod_fix_ids: list[int] = []
        for fb in blueprint.file_blueprints:
            if fb.layer in ("config", "deploy"):
                continue
            mod_fix = Task(
                task_id=self._alloc_id(),
                task_type=TaskType.FIX_CODE,
                file=fb.path,
                description=f"Fix {fb.path} based on module review",
                dependencies=[mod_review.task_id],
                metadata={"review_task_id": mod_review.task_id},
            )
            global_graph.add_task(mod_fix)
            mod_fix_ids.append(mod_fix.task_id)

        # Architecture review
        arch_review = Task(
            task_id=self._alloc_id(),
            task_type=TaskType.REVIEW_ARCHITECTURE,
            file="*",
            description="Architecture review for dependency cycles and layer violations",
            dependencies=mod_fix_ids + [sec_task.task_id],
        )
        global_graph.add_task(arch_review)

        # Deploy
        deploy_task = Task(
            task_id=self._alloc_id(),
            task_type=TaskType.GENERATE_DEPLOY,
            file="deploy/",
            description="Generate Dockerfile and Kubernetes manifests",
            dependencies=[arch_review.task_id],
        )
        global_graph.add_task(deploy_task)

        # Docs
        docs_task = Task(
            task_id=self._alloc_id(),
            task_type=TaskType.GENERATE_DOCS,
            file="docs/",
            description="Generate README, changelog, API docs",
            dependencies=[arch_review.task_id],
        )
        global_graph.add_task(docs_task)

        errors = global_graph.validate()
        if errors:
            raise ValueError(f"Global task graph validation failed: {errors}")

        logger.info(
            "Built lifecycle plan: %d files (lifecycle) + %d global tasks",
            len(file_paths), len(global_graph.tasks),
        )
        return engine, global_graph


class ModificationTaskGraphBuilder:
    """Builds a task DAG for modifying an existing repository.

    The modification flow is:
      1. MODIFY_FILE tasks for each change in the plan (graph-ordered via dep_store)
      2. GENERATE_FILE tasks for brand-new files in the plan
      3. REVIEW_FILE for each modified/new file
      4. FIX_CODE for each review
      5. GENERATE_TEST for affected test files (expanded by dep_store impact analysis)
      6. REVIEW_MODULE for cross-file consistency

    When ``dep_store`` is provided the builder:
    - Reorders modification tasks topologically so dependencies are modified first
    - Expands affected_tests using graph-derived impact analysis (not just heuristics)
    - Logs any detected dependency cycles as warnings before building
    - Stores per-file impact metadata in task.metadata for downstream context use
    """

    def __init__(self, dep_store: DependencyGraphStore | None = None) -> None:
        self._next_id = 1
        self._dep_store = dep_store

    def _alloc_id(self) -> int:
        tid = self._next_id
        self._next_id += 1
        return tid

    def build_from_change_plan(
        self,
        change_plan: ChangePlan,
        blueprint: RepositoryBlueprint,
    ) -> TaskGraph:
        """Build a task graph from a ChangePlan.

        Each ChangeAction becomes a MODIFY_FILE task; each new_file becomes
        a GENERATE_FILE task.  Reviews and tests follow.
        """
        graph = TaskGraph()
        file_task_map: dict[str, int] = {}  # file_path → last task_id for that file

        # ── Pre-flight: cycle detection & safe ordering ────────────────
        ordered_changes = self._order_changes(change_plan.changes)

        # ── Phase 1: Modification tasks (graph-safe order) ────────────
        for change in ordered_changes:
            deps = [
                file_task_map[d]
                for d in change.depends_on
                if d in file_task_map
            ]
            impact_meta = self._get_impact_meta(change.file)
            task = Task(
                task_id=self._alloc_id(),
                task_type=TaskType.MODIFY_FILE,
                file=change.file,
                description=f"Modify {change.file}: {change.description}",
                dependencies=deps,
                metadata={
                    "change_type": change.type.value,
                    "change_description": change.description,
                    "target_function": change.function,
                    "target_class": change.class_name,
                    **impact_meta,
                },
            )
            graph.add_task(task)
            file_task_map[change.file] = task.task_id

        # ── Phase 1b: Generate brand-new files ────────────────────────
        for nf in change_plan.new_files:
            deps = [
                file_task_map[d]
                for d in nf.depends_on
                if d in file_task_map
            ]
            task = Task(
                task_id=self._alloc_id(),
                task_type=TaskType.GENERATE_FILE,
                file=nf.path,
                description=f"Create new file {nf.path}: {nf.purpose}",
                dependencies=deps,
            )
            graph.add_task(task)
            file_task_map[nf.path] = task.task_id

        # ── Phase 2: Review each modified/new file ────────────────────
        review_task_map: dict[str, int] = {}
        for file_path, gen_task_id in file_task_map.items():
            task = Task(
                task_id=self._alloc_id(),
                task_type=TaskType.REVIEW_FILE,
                file=file_path,
                description=f"Review changes in {file_path}",
                dependencies=[gen_task_id],
            )
            graph.add_task(task)
            review_task_map[file_path] = task.task_id

        # ── Phase 2.5: Fix based on review ────────────────────────────
        fix_task_ids: list[int] = []
        for file_path, review_id in review_task_map.items():
            fix_task = Task(
                task_id=self._alloc_id(),
                task_type=TaskType.FIX_CODE,
                file=file_path,
                description=f"Fix {file_path} based on review",
                dependencies=[review_id],
                metadata={"review_task_id": review_id},
            )
            graph.add_task(fix_task)
            fix_task_ids.append(fix_task.task_id)

        # ── Phase 3: Tests for affected files ─────────────────────────
        # Merge plan's explicit test list with graph-derived impacted tests.
        test_targets = self._expand_test_targets(
            change_plan.affected_tests, list(file_task_map.keys())
        )
        for file_path in test_targets:
            task = Task(
                task_id=self._alloc_id(),
                task_type=TaskType.GENERATE_TEST,
                file=file_path,
                description=f"Generate/update tests for {file_path}",
                dependencies=fix_task_ids,
            )
            graph.add_task(task)

        # ── Phase 4: Module-level review ──────────────────────────────
        mod_review = Task(
            task_id=self._alloc_id(),
            task_type=TaskType.REVIEW_MODULE,
            file="*",
            description="Module consistency review of modifications",
            dependencies=fix_task_ids,
        )
        graph.add_task(mod_review)

        errors = graph.validate()
        if errors:
            raise ValueError(f"Modification task graph validation failed: {errors}")

        logger.info(
            "Built modification task graph: %d tasks (%d modify, %d new, %d review, %d test)",
            len(graph.tasks),
            len(ordered_changes),
            len(change_plan.new_files),
            len(review_task_map),
            len(test_targets),
        )
        return graph

    def _order_changes(self, changes: list[ChangeAction]) -> list[ChangeAction]:
        """Return changes in safe modification order using the dependency graph.

        When dep_store is available:
          1. Detect and warn about cycles in the affected subgraph.
          2. Use get_modification_order() to topologically sort the target files.
          3. Map back to the original ChangeAction objects.

        Falls back to original order if dep_store is absent or graph is sparse.
        """
        if self._dep_store is None or not changes:
            return changes

        file_paths = [c.file for c in changes]

        # Warn about cycles before they cause silent ordering problems
        cycles = self._dep_store.detect_cycles()
        affected_files = set(file_paths)
        affecting_cycles = [
            cycle for cycle in cycles
            if any(f in affected_files for f in cycle)
        ]
        if affecting_cycles:
            logger.warning(
                "Dependency cycles detected involving modification targets: %s",
                affecting_cycles,
            )

        ordered_paths = self._dep_store.get_modification_order(file_paths)

        # Map back to ChangeAction, preserving duplicates (multiple changes per file)
        path_to_changes: dict[str, list[ChangeAction]] = {}
        for change in changes:
            path_to_changes.setdefault(change.file, []).append(change)

        result: list[ChangeAction] = []
        seen: set[str] = set()
        for path in ordered_paths:
            if path not in seen and path in path_to_changes:
                result.extend(path_to_changes[path])
                seen.add(path)
        # Append any changes for files not in the graph (new files, etc.)
        for change in changes:
            if change.file not in seen:
                result.append(change)
                seen.add(change.file)

        if result != changes:
            original = [c.file for c in changes]
            reordered = [c.file for c in result]
            logger.info(
                "Modification order resequenced by dependency graph: %s → %s",
                original, reordered,
            )

        return result

    def _get_impact_meta(self, file_path: str) -> dict[str, Any]:
        """Return impact analysis metadata for a file to embed in task.metadata."""
        if self._dep_store is None:
            return {}
        impact = self._dep_store.get_impact_analysis(file_path)
        return {
            "impact_direct_dependents": impact.get("direct_dependents", []),
            "impact_transitive_dependents": impact.get("transitive_dependents", []),
            "impact_affected_tests": impact.get("test_files", []),
        }

    def _expand_test_targets(
        self, plan_tests: list[str], changed_files: list[str]
    ) -> set[str]:
        """Merge plan test list with graph-derived impacted test files."""
        targets: set[str] = set(plan_tests) | set(changed_files)

        if self._dep_store is None:
            return targets

        for file_path in changed_files:
            impact = self._dep_store.get_impact_analysis(file_path)
            graph_tests = impact.get("test_files", [])
            if graph_tests:
                logger.debug(
                    "Graph impact: %s affects tests %s", file_path, graph_tests
                )
            targets.update(graph_tests)

        return targets
