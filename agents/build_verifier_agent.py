"""Build verifier agent — runs the language's compile/build command.

Sits between REVIEWING and TESTING in the per-file lifecycle for compiled
languages (Java, Go, Rust, C#, TypeScript).  Catches compilation errors
before the test phase so the fix cycle receives accurate compiler output
rather than confusing test-runner errors.

For interpreted languages (Python) the BUILDING phase is auto-skipped by
the LifecycleEngine; this agent is never invoked.
"""

from __future__ import annotations

import logging
from typing import Any

from agents.base_agent import BaseAgent
from core.language import detect_language_from_blueprint
from core.models import AgentContext, AgentRole, TaskResult
from tools.terminal_tools import TerminalTools

logger = logging.getLogger(__name__)


class BuildVerifierAgent(BaseAgent):
    """Compiles / type-checks the workspace and reports errors.

    Requires a ``TerminalTools`` instance that can run the build command
    (the build-tier sandbox, which has network access for dependency
    downloads such as ``mvn package`` or ``go mod download``).
    """

    role = AgentRole.BUILD_VERIFIER

    def __init__(self, *args: Any, terminal: TerminalTools | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.terminal = terminal

    @property
    def system_prompt(self) -> str:
        return (
            "You are a build verification agent. "
            "You run compile commands and interpret compiler output."
        )

    async def execute(self, context: AgentContext) -> TaskResult:
        """Run the build command and return success/failure with compiler output."""
        lang_profile = detect_language_from_blueprint(context.blueprint.tech_stack)
        build_cmd = lang_profile.build_command

        if not build_cmd:
            # Interpreted language — should have been auto-skipped by LifecycleEngine
            return TaskResult(
                success=True,
                output="No build step required for this language",
                metrics=self.get_metrics(),
            )

        if self.terminal is None:
            logger.warning(
                "BuildVerifierAgent has no terminal — skipping build verification for %s",
                context.task.file,
            )
            return TaskResult(
                success=True,
                output="Build verification skipped (no terminal configured)",
                metrics=self.get_metrics(),
            )

        logger.info("Running build verification: %s", build_cmd)
        result = await self.terminal.run_command(build_cmd, timeout=180)

        if result.exit_code == 0:
            return TaskResult(
                success=True,
                output=f"Build passed: {build_cmd}\n{result.stdout[:500]}",
                metrics=self.get_metrics(),
            )

        compiler_output = "\n".join(filter(None, [result.stdout, result.stderr])).strip()
        logger.warning(
            "Build failed for %s:\n%s", context.task.file, compiler_output[:500]
        )
        return TaskResult(
            success=False,
            output=f"Build failed: {build_cmd}",
            errors=[compiler_output or "Build command exited with non-zero status"],
            metrics=self.get_metrics(),
        )
