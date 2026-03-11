"""Task DAG engine with topological execution ordering.

Execution model: **Lifecycle + global DAG** — ``LifecyclePlanBuilder.build()`` creates:
  - A ``LifecycleEngine`` for per-file Generate→Review→Fix→Test cycles
  - A slim ``TaskGraph`` for global phases (security, module review, deploy, docs)
  The per-file lifecycle is event-driven; the global DAG runs after all files
  reach a terminal state.

``TaskGraphBuilder`` is retained for modification workflows only.
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
    """Task graph builder for modification workflows.

    ``build_from_blueprint()`` has been removed — use ``LifecyclePlanBuilder``
    for new-project generation.  This class is kept for ``ModificationTaskGraphBuilder``
    which still uses it internally for helper utilities.
    """

    def __init__(self) -> None:
        self._next_id = 1

    def _alloc_id(self) -> int:
        tid = self._next_id
        self._next_id += 1
        return tid


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

        # Build a suffix→full-path lookup so short dep references like
        # "models/Greeting.java" resolve to their full blueprint paths
        # (e.g. "src/main/java/com/example/helloworld/models/Greeting.java").
        # Progressively shorter suffixes are added; the first (longest) match wins.
        _suffix_map: dict[str, str] = {}
        for p in known_paths:
            _suffix_map[p] = p  # exact match always wins
            parts = p.replace("\\", "/").split("/")
            for i in range(1, len(parts)):
                suffix = "/".join(parts[i:])
                if suffix not in _suffix_map:
                    _suffix_map[suffix] = p

        def _resolve_dep(dep: str) -> str | None:
            return _suffix_map.get(dep.replace("\\", "/"))

        for fb in blueprint.file_blueprints:
            for dep_path in fb.depends_on:
                if _resolve_dep(dep_path) is None:
                    logger.warning(
                        "Blueprint: %s depends on '%s' which is not in the "
                        "blueprint — dependency will be ignored",
                        fb.path, dep_path,
                    )

        # ── Build per-file lifecycle engine ──────────────────────────
        file_paths = [fb.path for fb in blueprint.file_blueprints]
        file_deps = {
            fb.path: [
                _resolve_dep(d) for d in fb.depends_on
                if _resolve_dep(d) is not None
            ]
            for fb in blueprint.file_blueprints
        }

        from core.language import detect_language_from_blueprint
        lang_profile = detect_language_from_blueprint(blueprint.tech_stack)
        compiled = bool(lang_profile.build_command)

        engine = LifecycleEngine(
            file_paths=file_paths,
            file_deps=file_deps,
            max_review_fixes=max_review_fixes,
            max_test_fixes=max_test_fixes,
            compiled=compiled,
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
            metadata={"sentinel": True},
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

        # Integration tests — run after all per-file fixes and module review
        integration_test = Task(
            task_id=self._alloc_id(),
            task_type=TaskType.GENERATE_INTEGRATION_TEST,
            file="tests/integration/",
            description="Generate integration tests verifying cross-module interactions",
            dependencies=mod_fix_ids + [sec_task.task_id],
        )
        global_graph.add_task(integration_test)

        # Architecture review (after integration tests + security)
        arch_review = Task(
            task_id=self._alloc_id(),
            task_type=TaskType.REVIEW_ARCHITECTURE,
            file="*",
            description="Architecture review for dependency cycles and layer violations",
            dependencies=[integration_test.task_id],
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

        # ── Pre-flight: conflict detection & resolution ───────────────
        conflicts = self._detect_conflicts(change_plan.changes)
        if conflicts:
            for conflict in conflicts:
                logger.warning("Change conflict detected: %s", conflict)
            change_plan.risk_notes.extend(conflicts)

        # Merge changes that target the same symbol in the same file into a
        # single MODIFY_FILE task.  Without this, the second task reads the
        # file (with task 1's edits in place), asks the LLM to apply only
        # its change, and writes the full file back — silently dropping task
        # 1's changes if the LLM doesn't faithfully preserve them.
        resolved_changes = self._merge_conflicting_changes(change_plan.changes)

        # ── Pre-flight: cycle detection & safe ordering ────────────────
        ordered_changes = self._order_changes(resolved_changes)

        # Files that are brand-new (in new_files) must be created by a
        # GENERATE_FILE task (Phase 1b), not a MODIFY_FILE task — the file
        # doesn't exist yet and PatchAgent cannot patch a non-existent file.
        # Routing them through MODIFY_FILE also causes a duplicate task when
        # the same path appears in both change_plan.changes and new_files.
        new_file_paths = {nf.path for nf in change_plan.new_files}

        # ── Phase 1: Modification tasks (graph-safe order) ────────────
        for change in ordered_changes:
            # Skip files that are truly new — handled by GENERATE_FILE below.
            if change.file in new_file_paths:
                continue

            deps = [
                file_task_map[d]
                for d in change.depends_on
                if d in file_task_map
            ]
            # Implicit same-file chain: if a previous task already targets
            # this file, make the new task depend on it.  This guarantees
            # sequential execution even when the change plan omits the
            # self-dependency in its explicit depends_on list.
            if change.file in file_task_map:
                prev_id = file_task_map[change.file]
                if prev_id not in deps:
                    deps.append(prev_id)

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

    @staticmethod
    def _is_test_file(path: str) -> bool:
        """Heuristic: return True if path looks like an existing test file.

        Test files should NOT be passed to TestAgent as source targets because
        TestAgent will compute a 'test for the test file' path (e.g.
        test_test_foo.py) and fail with 'No blueprint' for the test file.
        """
        stem = path.replace("\\", "/").split("/")[-1].lower()
        return (
            stem.startswith("test_")
            or stem.endswith("_test.py")
            or stem.endswith(".test.ts")
            or stem.endswith("_test.go")
            or stem.endswith("_test.rs")
            or stem.endswith("tests.cs")
            or stem.endswith("test.java")
            or "/test/" in path.lower()
            or "/tests/" in path.lower()
        )

    def _expand_test_targets(
        self, plan_tests: list[str], changed_files: list[str]
    ) -> set[str]:
        """Merge plan test list with graph-derived impacted test files.

        ``plan_tests`` (``change_plan.affected_tests``) are test-file paths.
        Adding them directly would cause TestAgent to generate a 'test for a
        test file', which always fails with 'No blueprint'.  They are filtered
        out here; the corresponding source files are already in
        ``changed_files`` so their tests will be regenerated regardless.
        """
        # Only include source files as test generation targets; skip files
        # that are already test files (they'd be double-processed).
        targets: set[str] = {
            f for f in changed_files if not self._is_test_file(f)
        }

        if self._dep_store is None:
            return targets

        for file_path in changed_files:
            impact = self._dep_store.get_impact_analysis(file_path)
            graph_tests = impact.get("test_files", [])
            if graph_tests:
                logger.debug(
                    "Graph impact: %s affects tests %s", file_path, graph_tests
                )
            # graph_tests are test-file paths; filter them out for the same
            # reason as plan_tests — TestAgent cannot re-generate a test file
            # by receiving a test file path (it would compute test_test_foo.py).
            targets.update(f for f in graph_tests if not self._is_test_file(f))

        return targets

    def _merge_conflicting_changes(
        self, changes: list[ChangeAction]
    ) -> list[ChangeAction]:
        """Merge changes that target the same symbol in the same file.

        Two changes with identical ``(file, class_name, function)`` keys are
        collapsed into a single ``ChangeAction`` whose description combines
        both intents.  The merged task is given to the agent once, so it can
        apply all sub-changes in a single read-modify-write pass rather than
        having the second pass overwrite (and potentially lose) the first's
        edits.

        Changes whose target key contains only blank fields — i.e. file-level
        changes with no named symbol — are never merged because there is no
        reliable way to determine whether they conflict.
        """
        # Preserve insertion order so the task graph stays deterministic.
        seen: dict[tuple[str, str, str], list[ChangeAction]] = {}
        for change in changes:
            key = (change.file, change.class_name or "", change.function or "")
            seen.setdefault(key, []).append(change)

        merged: list[ChangeAction] = []
        for (file, cls, fn), group in seen.items():
            if len(group) == 1:
                merged.append(group[0])
                continue

            # Blank-target changes (no class AND no function) stay separate —
            # they likely touch different parts of the file.
            if not cls and not fn:
                merged.extend(group)
                continue

            # Merge: combine descriptions; union depends_on; keep first type.
            combined_desc = "; ".join(
                f"({i + 1}) {c.description}" for i, c in enumerate(group)
            )
            seen_deps: set[str] = set()
            all_deps: list[str] = []
            for c in group:
                for d in c.depends_on:
                    if d not in seen_deps:
                        all_deps.append(d)
                        seen_deps.add(d)

            symbol = f"{cls}.{fn}" if fn else (cls or fn)
            logger.info(
                "Merging %d conflicting changes for '%s' in %s into one task",
                len(group), symbol, file,
            )
            merged.append(ChangeAction(
                type=group[0].type,
                file=file,
                description=combined_desc,
                function=fn,
                class_name=cls,
                depends_on=all_deps,
                details=group[0].details,
            ))

        return merged

    def _detect_conflicts(self, changes: list[ChangeAction]) -> list[str]:
        """Detect when multiple changes target the same function/class in the same file.

        Returns a list of human-readable conflict descriptions.  Conflicts are
        promoted to ``ChangePlan.risk_notes`` and logged as warnings so the
        operator (or the approval gate) can decide whether to proceed.

        Serialization: conflicting changes are left in their original order;
        the dependency graph in ``build_from_change_plan`` will emit them as
        sequential tasks by virtue of using ``file_task_map`` (each file slot
        holds only the last task_id, so each subsequent task for the same file
        implicitly depends on the previous one).
        """
        conflicts: list[str] = []
        # file → list of (class_name, function) tuples already seen
        file_targets: dict[str, list[tuple[str, str]]] = {}

        for change in changes:
            target = (change.class_name or "", change.function or "")
            previous = file_targets.setdefault(change.file, [])

            # A conflict is two changes targeting the *same named symbol*
            # in the same file (ignoring blank/unnamed targets).
            if any(t != ("", "") and t == target and target != ("", "") for t in previous):
                qualifier = (
                    f"{change.class_name}.{change.function}"
                    if change.function
                    else (change.class_name or change.function or "unnamed symbol")
                )
                conflicts.append(
                    f"Multiple changes targeting '{qualifier}' in {change.file} — "
                    "changes will be serialized but may produce merge conflicts"
                )

            previous.append(target)

        return conflicts


class EnhanceLifecyclePlanBuilder:
    """Builds a LifecycleEngine + global TaskGraph from a ChangePlan.

    This is the Enhance-pipeline counterpart of ``LifecyclePlanBuilder``.
    Instead of creating GENERATE_FILE lifecycles for every blueprint file, it
    creates:
      - MODIFY_FILE lifecycles for existing files being changed
      - GENERATE_FILE lifecycles for brand-new files
      - A global TaskGraph with just a module review sentinel

    The resulting ``(LifecycleEngine, TaskGraph)`` can be passed directly to
    ``AgentManager.execute_with_checkpoints()`` so the Enhance pipeline uses
    the same unified executor, tier-based scheduling, build checkpoints, and
    review→fix cycles as the Generate pipeline.
    """

    def __init__(self, dep_store: DependencyGraphStore | None = None) -> None:
        self._next_id = 1
        self._dep_store = dep_store

    def _alloc_id(self) -> int:
        tid = self._next_id
        self._next_id += 1
        return tid

    def build(
        self,
        change_plan: ChangePlan,
        blueprint: RepositoryBlueprint,
        *,
        max_review_fixes: int = 2,
        max_test_fixes: int = 3,
        compiled: bool = False,
    ) -> tuple[LifecycleEngine, TaskGraph]:
        """Build a lifecycle engine + global task graph from a change plan.

        Returns:
            (lifecycle_engine, global_task_graph)
        """
        # ── Collect files and their metadata ────────────────────────
        new_file_paths = {nf.path for nf in change_plan.new_files}

        # Merge changes that target the same symbol in the same file.
        mod_builder = ModificationTaskGraphBuilder(dep_store=self._dep_store)
        conflicts = mod_builder._detect_conflicts(change_plan.changes)
        if conflicts:
            for conflict in conflicts:
                logger.warning("Change conflict detected: %s", conflict)
            change_plan.risk_notes.extend(conflicts)
        resolved_changes = mod_builder._merge_conflicting_changes(change_plan.changes)

        # ── Build per-file overrides ─────────────────────────────────
        # Group changes by file so we can build consolidated metadata.
        changes_by_file: dict[str, list[ChangeAction]] = {}
        for change in resolved_changes:
            if change.file not in new_file_paths:
                changes_by_file.setdefault(change.file, []).append(change)

        file_overrides: dict[str, dict[str, Any]] = {}
        file_paths: list[str] = []
        file_deps: dict[str, list[str]] = {}

        # Existing files → MODIFY_FILE lifecycle
        for file_path, changes in changes_by_file.items():
            file_paths.append(file_path)
            combined_desc = "; ".join(c.description for c in changes)
            # Union all change dependencies
            deps: list[str] = []
            seen_deps: set[str] = set()
            for c in changes:
                for d in c.depends_on:
                    if d not in seen_deps and d != file_path:
                        deps.append(d)
                        seen_deps.add(d)
            file_deps[file_path] = deps

            file_overrides[file_path] = {
                "generation_task_type": "modify_file",
                "change_metadata": {
                    "change_type": changes[0].type.value,
                    "change_description": combined_desc,
                    "target_function": changes[0].function,
                    "target_class": changes[0].class_name,
                },
            }

        # New files → GENERATE_FILE lifecycle (default).
        # Do NOT filter nf.depends_on here — the post-loop all_file_set filter
        # below handles it correctly for all cases, including cross-new-file
        # deps where B depends on A but B's iteration runs before A's append.
        for nf in change_plan.new_files:
            file_paths.append(nf.path)
            file_deps[nf.path] = list(nf.depends_on)

        # ── Filter deps to only internal files ───────────────────────
        all_file_set = set(file_paths)
        file_deps = {
            fp: [d for d in deps if d in all_file_set]
            for fp, deps in file_deps.items()
        }

        # ── Build lifecycle engine ───────────────────────────────────
        engine = LifecycleEngine(
            file_paths=file_paths,
            file_deps=file_deps,
            max_review_fixes=max_review_fixes,
            max_test_fixes=max_test_fixes,
            compiled=compiled,
            file_overrides=file_overrides,
        )

        # Mark test files and config files to skip testing
        known_bp_paths = {fb.path: fb for fb in blueprint.file_blueprints}
        for fp in file_paths:
            fb = known_bp_paths.get(fp)
            if fb and fb.layer in ("test", "config", "deploy"):
                engine.skip_testing(fp)
            elif ModificationTaskGraphBuilder._is_test_file(fp):
                engine.skip_testing(fp)

        # ── Build global task graph ──────────────────────────────────
        global_graph = TaskGraph()

        # Sentinel task: marks lifecycle completion
        lifecycle_done = Task(
            task_id=self._alloc_id(),
            task_type=TaskType.REVIEW_FILE,
            file="*",
            description="[sentinel] All per-file lifecycles completed",
            dependencies=[],
            metadata={"sentinel": True},
        )
        global_graph.add_task(lifecycle_done)

        # Module consistency review
        mod_review = Task(
            task_id=self._alloc_id(),
            task_type=TaskType.REVIEW_MODULE,
            file="*",
            description="Module consistency review of modifications",
            dependencies=[lifecycle_done.task_id],
        )
        global_graph.add_task(mod_review)

        errors = global_graph.validate()
        if errors:
            raise ValueError(f"Enhance global task graph validation failed: {errors}")

        logger.info(
            "Built enhance lifecycle plan: %d files (%d modify, %d new) + %d global tasks",
            len(file_paths),
            len(changes_by_file),
            len(change_plan.new_files),
            len(global_graph.tasks),
        )
        return engine, global_graph
