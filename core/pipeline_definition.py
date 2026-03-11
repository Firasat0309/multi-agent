"""Declarative pipeline definitions for the unified execution engine.

Both the Generate and Enhance pipelines are expressed as a sequence of
``Phase`` objects, each containing per-file tasks (run in parallel, respecting
dependencies) and an optional repo-level ``Checkpoint``.

This replaces the two hard-coded execution paths (``execute_with_lifecycle``
and ``execute_graph``) with a single configurable structure.

Example::

    GENERATE_PIPELINE = PipelineDefinition(
        name="generate",
        phases=[
            Phase(
                name="code_generation",
                file_tasks=[
                    FileTaskDef(TaskType.GENERATE_FILE, review=True, max_review_fixes=3),
                ],
                checkpoint=CheckpointDef(
                    name="build_verification",
                    max_retries=4,
                ),
            ),
            Phase(
                name="testing",
                file_tasks=[
                    FileTaskDef(TaskType.GENERATE_TEST, max_test_fixes=3),
                ],
            ),
        ],
        global_tasks=[TaskType.SECURITY_SCAN, TaskType.GENERATE_DEPLOY],
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.models import TaskType


@dataclass
class CheckpointDef:
    """Definition for a repo-level build checkpoint between phases.

    The checkpoint runs after all per-file tasks in the phase complete.
    On failure, it attributes errors to files and dispatches fix tasks,
    then re-runs the build.
    """

    name: str = "build_verification"
    max_retries: int = 4
    timeout: int = 180


@dataclass
class FileTaskDef:
    """Definition for a per-file task within a phase.

    ``review``: If True, the file goes through a Review → Fix cycle after
    this task type completes.

    ``max_review_fixes`` / ``max_test_fixes``: Cycle limits for the
    review-fix and test-fix loops respectively.
    """

    task_type: TaskType
    review: bool = False
    max_review_fixes: int = 3
    max_test_fixes: int = 3


@dataclass
class Phase:
    """A phase in the pipeline — contains per-file work + optional checkpoint.

    Per-file tasks run in parallel (respecting dependency order).
    The checkpoint runs after all per-file tasks in this phase complete
    successfully.
    """

    name: str
    file_tasks: list[FileTaskDef] = field(default_factory=list)
    checkpoint: CheckpointDef | None = None
    skip_for_interpreted: bool = False  # Skip this entire phase for non-compiled langs


@dataclass
class PipelineDefinition:
    """Complete declarative pipeline specification.

    Consumed by the unified ``PipelineExecutor`` which replaces both
    ``execute_with_lifecycle()`` and ``execute_graph()``.
    """

    name: str
    phases: list[Phase] = field(default_factory=list)
    global_tasks: list[TaskType] = field(default_factory=list)

    @property
    def has_checkpoints(self) -> bool:
        return any(p.checkpoint is not None for p in self.phases)

    def with_retries(self, max_retries: int) -> "PipelineDefinition":
        """Return a copy of this pipeline with all checkpoint max_retries overridden."""
        import dataclasses
        new_phases = [
            dataclasses.replace(
                phase,
                checkpoint=dataclasses.replace(phase.checkpoint, max_retries=max_retries)
                if phase.checkpoint else None,
            )
            for phase in self.phases
        ]
        return dataclasses.replace(self, phases=new_phases)


# ── Pre-built pipeline definitions ──────────────────────────────────────────

GENERATE_PIPELINE = PipelineDefinition(
    name="generate",
    phases=[
        Phase(
            name="code_generation",
            file_tasks=[
                FileTaskDef(
                    task_type=TaskType.GENERATE_FILE,
                    review=True,
                    max_review_fixes=3,
                ),
            ],
            checkpoint=CheckpointDef(
                name="build_verification",
                max_retries=4,
            ),
        ),
        Phase(
            name="testing",
            file_tasks=[
                FileTaskDef(
                    task_type=TaskType.GENERATE_TEST,
                    max_test_fixes=3,
                ),
            ],
        ),
    ],
    global_tasks=[
        TaskType.SECURITY_SCAN,
        TaskType.REVIEW_MODULE,
        TaskType.GENERATE_INTEGRATION_TEST,
        TaskType.REVIEW_ARCHITECTURE,
        TaskType.GENERATE_DEPLOY,
        TaskType.GENERATE_DOCS,
    ],
)


ENHANCE_PIPELINE = PipelineDefinition(
    name="enhance",
    phases=[
        Phase(
            name="modification",
            file_tasks=[
                FileTaskDef(
                    task_type=TaskType.MODIFY_FILE,
                    review=True,
                    max_review_fixes=2,
                ),
            ],
            checkpoint=CheckpointDef(
                name="build_verification",
                max_retries=4,
            ),
        ),
        Phase(
            name="testing",
            file_tasks=[
                FileTaskDef(
                    task_type=TaskType.GENERATE_TEST,
                    max_test_fixes=3,
                ),
            ],
            checkpoint=CheckpointDef(
                name="test_verification",
                max_retries=2,
                timeout=300,
            ),
        ),
    ],
    global_tasks=[
        TaskType.REVIEW_MODULE,
    ],
)
