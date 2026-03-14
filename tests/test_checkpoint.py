"""Tests for the build checkpoint system."""

import asyncio

from unittest.mock import AsyncMock, MagicMock
from core.checkpoint import BuildCheckpoint, CheckpointResult, CheckpointCycleResult
from core.error_attributor import AttributedError, AttributionResult


class TestCheckpointResult:
    def test_passed_result(self):
        result = CheckpointResult(passed=True, attempt=1)
        assert result.errors_by_file == {}
        assert result.affected_files == []

    def test_failed_result_with_errors(self):
        attribution = AttributionResult(
            errors_by_file={
                "User.java": [
                    AttributedError(file_path="User.java", line=10, message="missing ;"),
                ],
            }
        )
        result = CheckpointResult(
            passed=False,
            attempt=1,
            attribution=attribution,
        )
        assert result.affected_files == ["User.java"]
        assert result.errors_by_file == {"User.java": ["missing ;"]}


class TestCheckpointCycleResult:
    def test_passed_cycle(self):
        cycle = CheckpointCycleResult(
            passed=True,
            total_attempts=1,
            history=[CheckpointResult(passed=True, attempt=1)],
        )
        assert cycle.final_result is not None
        assert cycle.final_result.passed
        assert cycle.remaining_errors == {}

    def test_failed_cycle(self):
        attribution = AttributionResult(
            errors_by_file={
                "Svc.java": [
                    AttributedError(file_path="Svc.java", message="type error"),
                ],
            }
        )
        cycle = CheckpointCycleResult(
            passed=False,
            total_attempts=3,
            history=[
                CheckpointResult(passed=False, attempt=1, attribution=attribution),
                CheckpointResult(passed=False, attempt=2, attribution=attribution),
                CheckpointResult(passed=False, attempt=3, attribution=attribution),
            ],
        )
        assert not cycle.passed
        assert cycle.remaining_errors == {"Svc.java": ["type error"]}

    def test_empty_cycle(self):
        cycle = CheckpointCycleResult(passed=True, total_attempts=0)
        assert cycle.final_result is None
        assert cycle.remaining_errors == {}


class TestBuildCheckpoint:
    def _make_terminal(self, exit_code=0, stdout="", stderr=""):
        terminal = MagicMock()
        terminal.run_command = AsyncMock(return_value=MagicMock(
            exit_code=exit_code, stdout=stdout, stderr=stderr
        ))
        return terminal

    def test_build_passes(self):
        terminal = self._make_terminal(exit_code=0, stdout="BUILD SUCCESS")
        checkpoint = BuildCheckpoint(
            build_command="mvn compile",
            terminal=terminal,
        )
        result = asyncio.run(checkpoint.run_once())

        assert result.passed
        assert result.attempt == 1
        terminal.run_command.assert_called_once_with("mvn compile", timeout=180)

    def test_build_fails_with_errors(self):
        terminal = self._make_terminal(
            exit_code=1,
            stderr="User.java:10: error: missing ;\n",
        )
        checkpoint = BuildCheckpoint(
            build_command="mvn compile",
            terminal=terminal,
        )
        result = asyncio.run(checkpoint.run_once())

        assert not result.passed
        assert result.affected_files  # at least one file attributed

    def test_fix_context_for_file(self):
        terminal = self._make_terminal()
        attribution = AttributionResult(
            errors_by_file={
                "User.java": [
                    AttributedError(file_path="User.java", line=10, message="missing ;"),
                ],
            }
        )
        cp_result = CheckpointResult(
            passed=False,
            attempt=1,
            attribution=attribution,
        )

        checkpoint = BuildCheckpoint(
            build_command="mvn compile",
            terminal=terminal,
        )
        ctx = checkpoint.get_fix_context_for_file("User.java", cp_result)

        assert ctx["fix_trigger"] == "build"
        assert "missing ;" in ctx["build_errors"]
        assert ctx["build_command"] == "mvn compile"

    def test_custom_timeout(self):
        terminal = self._make_terminal(exit_code=0, stdout="OK")
        checkpoint = BuildCheckpoint(
            build_command="cargo build",
            terminal=terminal,
            timeout=300,
        )
        asyncio.run(checkpoint.run_once())

        terminal.run_command.assert_called_once_with("cargo build", timeout=300)
