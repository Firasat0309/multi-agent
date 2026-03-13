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

        # Build a coder-friendly payload for FIX_CODE prompts.
        # Compiler logs are often very large/noisy; the actionable diagnostics
        # are typically near the end.
        max_tail = 4000
        output_tail = compiler_output[-max_tail:] if compiler_output else ""
        # Target path comes from the lifecycle task currently being verified.
        # For FilePhase.BUILDING, AgentManager creates Task(file=<current file>)
        # and passes it into this agent as context.task.file.
        # Fallback to file_blueprint.path for non-lifecycle invocations.
        target_path = context.task.file or (
            context.file_blueprint.path if context.file_blueprint else ""
        )
        mentions_target = bool(target_path and target_path in compiler_output)
        summary = (
            f"Build failed for {target_path} using '{build_cmd}' "
            f"(exit={result.exit_code}, target_mentioned={mentions_target})"
        )
        prompt_payload = (
            f"{summary}\n\n"
            f"--- compiler_output_tail ---\n{output_tail or '(no output captured)'}"
        )

        logger.warning("%s", summary)
        return TaskResult(
            success=False,
            output=summary,
            errors=[prompt_payload],
            metrics={
                **self.get_metrics(),
                "build_command": build_cmd,
                "exit_code": result.exit_code,
                "target_path": target_path,
                "target_mentioned": mentions_target,
            },
        )
