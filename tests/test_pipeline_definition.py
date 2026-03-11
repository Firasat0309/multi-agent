"""Tests for the declarative pipeline definitions."""

from core.models import TaskType
from core.pipeline_definition import (
    CheckpointDef,
    FileTaskDef,
    Phase,
    PipelineDefinition,
    GENERATE_PIPELINE,
    ENHANCE_PIPELINE,
)


class TestPipelineDefinition:
    def test_generate_pipeline_structure(self):
        assert GENERATE_PIPELINE.name == "generate"
        assert len(GENERATE_PIPELINE.phases) == 2
        assert GENERATE_PIPELINE.has_checkpoints

    def test_generate_pipeline_phases(self):
        gen_phase = GENERATE_PIPELINE.phases[0]
        assert gen_phase.name == "code_generation"
        assert gen_phase.checkpoint is not None
        assert gen_phase.checkpoint.name == "build_verification"

        test_phase = GENERATE_PIPELINE.phases[1]
        assert test_phase.name == "testing"
        assert test_phase.checkpoint is None

    def test_generate_pipeline_global_tasks(self):
        assert TaskType.SECURITY_SCAN in GENERATE_PIPELINE.global_tasks
        assert TaskType.GENERATE_DEPLOY in GENERATE_PIPELINE.global_tasks
        assert TaskType.GENERATE_DOCS in GENERATE_PIPELINE.global_tasks

    def test_enhance_pipeline_structure(self):
        assert ENHANCE_PIPELINE.name == "enhance"
        assert len(ENHANCE_PIPELINE.phases) == 2
        assert ENHANCE_PIPELINE.has_checkpoints

    def test_enhance_pipeline_uses_modify(self):
        mod_phase = ENHANCE_PIPELINE.phases[0]
        assert mod_phase.file_tasks[0].task_type == TaskType.MODIFY_FILE

    def test_enhance_pipeline_has_test_checkpoint(self):
        test_phase = ENHANCE_PIPELINE.phases[1]
        assert test_phase.name == "testing"
        assert test_phase.checkpoint is not None
        assert test_phase.checkpoint.name == "test_verification"
        assert test_phase.checkpoint.max_retries == 2

    def test_enhance_pipeline_build_checkpoint(self):
        mod_phase = ENHANCE_PIPELINE.phases[0]
        assert mod_phase.checkpoint is not None
        assert mod_phase.checkpoint.name == "build_verification"
        assert mod_phase.checkpoint.max_retries == 4

    def test_custom_pipeline(self):
        pipeline = PipelineDefinition(
            name="custom",
            phases=[
                Phase(
                    name="build_only",
                    file_tasks=[FileTaskDef(TaskType.GENERATE_FILE)],
                    checkpoint=CheckpointDef(max_retries=2),
                ),
            ],
        )
        assert pipeline.name == "custom"
        assert pipeline.has_checkpoints
        assert len(pipeline.global_tasks) == 0

    def test_pipeline_without_checkpoints(self):
        pipeline = PipelineDefinition(
            name="simple",
            phases=[
                Phase(name="gen", file_tasks=[FileTaskDef(TaskType.GENERATE_FILE)]),
            ],
        )
        assert not pipeline.has_checkpoints

    def test_file_task_def_defaults(self):
        ftd = FileTaskDef(TaskType.GENERATE_FILE)
        assert not ftd.review
        assert ftd.max_review_fixes == 3
        assert ftd.max_test_fixes == 3

    def test_checkpoint_def_defaults(self):
        cd = CheckpointDef()
        assert cd.name == "build_verification"
        assert cd.max_retries == 4
        assert cd.timeout == 180
