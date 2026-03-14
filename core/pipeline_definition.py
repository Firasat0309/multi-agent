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
    On failure, it attributes errors to specific files using
    ``CompilerErrorAttributor`` and dispatches targeted fix tasks, then
    re-runs the build — up to ``max_retries`` times.

    Checkpoint timing within the executor::

        Tier N files: Generate → Review → Fix loops
            ↓ (all complete)
        Forward-reference stubs created for Tier N+1 files
            ↓
        BuildCheckpoint.run_once()  ←── this checkpoint
            ↓ pass → clean stubs, advance to Tier N+1
            ↓ fail → attribute errors → fix affected → retry

    Attributes:
        name: Identifier for logging and metrics (e.g. "build_verification").
        max_retries: Maximum number of build→fix→rebuild cycles. Each cycle
            only fixes files with newly attributed errors.
        timeout: Seconds before the build command is killed. Should account
            for dependency resolution on first build.
    """

    name: str = "build_verification"
    max_retries: int = 4
    timeout: int = 180


@dataclass
class FileTaskDef:
    """Definition for a per-file task within a phase.

    Each ``FileTaskDef`` describes one kind of work to be done per file.
    The lifecycle FSM (``FileLifecycle``) drives the actual state transitions;
    these fields declare the *intent* of the phase.

    Attributes:
        task_type: The ``TaskType`` to dispatch for each file (e.g.
            ``GENERATE_FILE``, ``MODIFY_FILE``, ``GENERATE_TEST``).
        review: If True, the file enters a Review → Fix cycle after the
            primary task completes.  Currently driven by the lifecycle FSM
            rather than this flag — reserved for future phase customization.
        max_review_fixes: Maximum review→fix iterations before the FSM
            exhausts retries and skips to the next phase.
        max_test_fixes: Maximum test→fix iterations before the FSM marks
            the file as DEGRADED (quality warning, not hard failure).
    """

    task_type: TaskType
    review: bool = False
    max_review_fixes: int = 3
    max_test_fixes: int = 3


@dataclass
class Phase:
    """A phase in the pipeline — contains per-file work + optional checkpoint.

    Per-file tasks run in parallel (respecting dependency tiers).  For
    generation phases (``GENERATE_FILE`` / ``MODIFY_FILE``), files are
    processed tier-by-tier with an optional build checkpoint between tiers.
    For test phases (``GENERATE_TEST``), all files run in a flat pass.

    Execution order within a generation phase::

        for tier in tiers:
            1. Run per-file lifecycles (tasks in parallel, respecting deps)
            2. If checkpoint defined and language is compiled:
               a. Generate forward-reference stubs for later tiers
               b. Run build checkpoint (with retry/fix loop)
               c. Clean up stubs
            3. If checkpoint fails after all retries:
               - Fail files with unresolvable errors
               - Block downstream files that depend exclusively on failed files

    Attributes:
        name: Human-readable phase identifier for logging.
        file_tasks: Per-file task definitions dispatched during this phase.
        checkpoint: Optional build checkpoint config.  Checkpoints only run
            when the target language has a build command (i.e., for compiled
            languages); they are always disabled for interpreted languages.
        skip_for_interpreted: If True, the entire phase is skipped when the
            target language has no build command (Python, Ruby, JS, etc.),
            regardless of whether a checkpoint is configured.
    """

    name: str
    file_tasks: list[FileTaskDef] = field(default_factory=list)
    checkpoint: CheckpointDef | None = None
    skip_for_interpreted: bool = False  # Skip this entire phase when no build command


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


# ── Frontend pipeline definition ──────────────────────────────────────────────

FRONTEND_PIPELINE = PipelineDefinition(
    name="frontend",
    phases=[
        Phase(
            name="component_generation",
            file_tasks=[
                # Each component file is generated individually.
                # The FrontendPipeline uses ComponentDAGAgent to order tiers;
                # this definition captures the per-file task contract.
                FileTaskDef(
                    task_type=TaskType.GENERATE_COMPONENT,
                    review=False,
                    max_review_fixes=0,
                ),
            ],
            # Frontend components are interpreted TypeScript — no compile checkpoint
            checkpoint=None,
            skip_for_interpreted=False,
        ),
    ],
    global_tasks=[
        TaskType.INTEGRATE_API,
        TaskType.MANAGE_STATE,
        TaskType.GENERATE_DOCS,
    ],
)

