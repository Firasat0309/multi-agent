"""Repo-level build checkpoint — replaces per-file build verification.

A ``BuildCheckpoint`` runs the language's compile command against the entire
workspace (or a specific module), parses the output using ``ErrorAttributor``,
and returns the list of files that need fixing.

The checkpoint supports a retry loop: fix → rebuild → check again, up to
``max_retries`` times.  On each iteration only the files with new errors are
sent to the fix agent, avoiding redundant work.

Usage by the orchestrator::

    checkpoint = BuildCheckpoint(
        build_command="mvn package -DskipTests",
        terminal=build_terminal,
        attributor=CompilerErrorAttributor(),
        known_files={"User.java", "UserService.java", ...},
    )

    result = await checkpoint.run()
    if not result.passed:
        for path, errors in result.errors_by_file.items():
            # dispatch fix agent for each affected file
            ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from core.error_attributor import AttributionResult, BaseErrorAttributor, CompilerErrorAttributor

logger = logging.getLogger(__name__)


@dataclass
class CheckpointResult:
    """Result of a single checkpoint run."""

    passed: bool
    attempt: int
    attribution: AttributionResult | None = None
    raw_output: str = ""

    @property
    def errors_by_file(self) -> dict[str, list[str]]:
        """Convenience: file → list of error message strings."""
        if not self.attribution:
            return {}
        return {
            path: [e.message for e in errors]
            for path, errors in self.attribution.errors_by_file.items()
        }

    @property
    def affected_files(self) -> list[str]:
        """Files that have build errors."""
        if not self.attribution:
            return []
        return self.attribution.affected_files


@dataclass
class CheckpointCycleResult:
    """Result of the full checkpoint cycle (all retries)."""

    passed: bool
    total_attempts: int
    history: list[CheckpointResult] = field(default_factory=list)
    files_fixed: list[str] = field(default_factory=list)

    @property
    def final_result(self) -> CheckpointResult | None:
        return self.history[-1] if self.history else None

    @property
    def remaining_errors(self) -> dict[str, list[str]]:
        """Errors that were never resolved."""
        final = self.final_result
        if final and not final.passed:
            return final.errors_by_file
        return {}


class BuildCheckpoint:
    """Repo-level build verification checkpoint.

    Runs the language's build command, attributes errors to files, and
    supports a fix-rebuild cycle up to ``max_retries`` times.

    For multi-module projects, pass ``module_path`` to scope the build
    to a specific module (e.g., ``mvn compile -pl :module-name``).
    """

    # Module-scoped build flags for common build tools.
    # The value is appended to the build command with the module path.
    _MODULE_FLAGS: dict[str, str] = {
        "mvn": "-pl",           # Maven: mvn compile -pl :module-name
        "gradle": "-p",         # Gradle: gradle build -p module
        "dotnet": "--project",  # dotnet build --project Module/
        "cargo": "-p",          # cargo build -p crate_name
    }

    def __init__(
        self,
        build_command: str,
        terminal: Any,  # TerminalTools
        *,
        attributor: BaseErrorAttributor | None = None,
        known_files: set[str] | None = None,
        max_retries: int = 4,
        timeout: int = 180,
        checkpoint_name: str = "build",
        module_path: str | None = None,
    ) -> None:
        self.base_build_command = build_command
        self.build_command = self._scope_command(build_command, module_path)
        self.terminal = terminal
        self.attributor = attributor or CompilerErrorAttributor()
        self.known_files = known_files
        self.max_retries = max_retries
        self.timeout = timeout
        self.name = checkpoint_name
        self.module_path = module_path

    @classmethod
    def _scope_command(cls, command: str, module_path: str | None) -> str:
        """Append module-scoping flag to the build command if applicable.

        Examples:
            _scope_command("mvn compile", "user-service")
            → "mvn compile -pl :user-service"

            _scope_command("go build ./...", "pkg/handlers")
            → "go build ./pkg/handlers/..."
        """
        if not module_path:
            return command

        parts = command.split()
        if not parts:
            return command

        tool = parts[0]

        # Go has special module syntax
        if tool == "go":
            # Replace ./... with ./module_path/...
            return command.replace("./...", f"./{module_path}/...")

        # Check for known module flags
        for tool_prefix, flag in cls._MODULE_FLAGS.items():
            if tool_prefix in tool:
                return f"{command} {flag} {module_path}"

        # Fallback: just append as an argument
        logger.debug(
            "No module-scoping rule for '%s' — appending module path as argument",
            tool,
        )
        return f"{command} {module_path}"

    async def run_once(self, attempt: int = 1) -> CheckpointResult:
        """Run the build command once and attribute any errors.

        Returns a ``CheckpointResult`` with pass/fail and attributed errors.
        """
        logger.info(
            "[Checkpoint:%s] Running build (attempt %d): %s",
            self.name, attempt, self.build_command,
        )

        result = await self.terminal.run_command(
            self.build_command, timeout=self.timeout,
        )

        if result.exit_code == 0:
            logger.info("[Checkpoint:%s] Build passed (attempt %d)", self.name, attempt)
            return CheckpointResult(
                passed=True,
                attempt=attempt,
                raw_output=result.stdout[:500],
            )

        # Build failed — attribute errors
        raw_output = "\n".join(filter(None, [result.stdout, result.stderr])).strip()
        attribution = self.attributor.attribute(
            raw_output, known_files=self.known_files,
        )

        logger.warning(
            "[Checkpoint:%s] Build failed (attempt %d): %d errors in %d files",
            self.name, attempt, attribution.total_errors,
            len(attribution.affected_files),
        )
        if attribution.unattributed_errors:
            logger.debug(
                "[Checkpoint:%s] %d unattributed error lines",
                self.name, len(attribution.unattributed_errors),
            )

        return CheckpointResult(
            passed=False,
            attempt=attempt,
            attribution=attribution,
            raw_output=raw_output[:8000],  # cap for context size
        )

    def get_fix_context_for_file(
        self,
        file_path: str,
        checkpoint_result: CheckpointResult,
    ) -> dict[str, Any]:
        """Build metadata for a fix agent targeting a specific file.

        This is passed as ``task.metadata`` to the CoderAgent (FIX_CODE task)
        so it knows exactly what build errors to fix.
        """
        if not checkpoint_result.attribution:
            return {}

        errors = checkpoint_result.attribution.summary_for_file(file_path)
        return {
            "fix_trigger": "build",
            "build_errors": errors,
            "checkpoint_name": self.name,
            "checkpoint_attempt": checkpoint_result.attempt,
            "build_command": self.build_command,
        }
