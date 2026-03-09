"""Test agent - generates unit and integration tests, runs them, and fixes failures."""

from __future__ import annotations

import logging
from typing import Any

from agents.base_agent import BaseAgent
from core.language import get_language_profile, LanguageProfile
from core.models import AgentContext, AgentRole, TaskResult
from tools.terminal_tools import TerminalTools

logger = logging.getLogger(__name__)


class TestAgent(BaseAgent):
    role = AgentRole.TESTER

    def __init__(self, *args: Any, terminal: TerminalTools | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.terminal = terminal

    def _get_system_prompt(self, language: str) -> str:
        profile = get_language_profile(language)
        return (
            f"You are a test engineering agent for {profile.display_name} projects.\n\n"
            "Rules:\n"
            f"- Generate ONLY the {profile.display_name} test file content, no markdown fences\n"
            f"- Use the standard test framework for {profile.display_name}\n"
            "- Mock external dependencies (database, HTTP, etc.)\n"
            "- Test both happy paths and error cases\n"
            "- Test edge cases and boundary conditions\n"
            "- Use descriptive test names\n"
            "- Use fixtures/setup for common setup\n"
            f"- Follow idiomatic {profile.display_name} testing conventions\n"
            "- Generate both unit tests and integration tests where appropriate"
        )

    @property
    def system_prompt(self) -> str:
        return self._get_system_prompt("python")

    async def execute(self, context: AgentContext) -> TaskResult:
        """Generate tests for a file and optionally run them."""
        if not context.file_blueprint:
            return TaskResult(
                success=False,
                errors=[f"No blueprint for {context.task.file}"],
            )

        fb = context.file_blueprint
        lang = fb.language or "python"
        profile = get_language_profile(lang)
        logger.info(f"Generating {profile.display_name} tests for {fb.path}")

        formatted = self._format_context(context)
        prompt = (
            f"{formatted}\n\n"
            f"Generate comprehensive {profile.display_name} tests for: {fb.path}\n"
            f"The file's purpose: {fb.purpose}\n"
            f"The file exports: {', '.join(fb.exports)}\n\n"
            f"Generate complete, working test code. Output only the code."
        )

        test_code = await self._call_llm(prompt, system_override=self._get_system_prompt(lang))
        test_code = self._clean_code(test_code, profile)

        # Compute test file path based on language conventions
        test_path = self._compute_test_path(fb.path, profile)
        self.repo.write_test_file(test_path, test_code)

        # Run tests if terminal available (autonomous debug loop)
        if self.terminal:
            result = await self._run_and_fix(test_path, test_code, context, profile)
            return result

        return TaskResult(
            success=True,
            output=f"Generated tests: {test_path}",
            files_modified=[test_path],
            metrics=self.get_metrics(),
        )

    def _compute_test_path(self, source_path: str, profile: LanguageProfile) -> str:
        """Compute test file path following language conventions."""
        filename = source_path.split("/")[-1]
        directory = source_path.rsplit("/", 1)[0] if "/" in source_path else ""

        name, ext = filename.rsplit(".", 1) if "." in filename else (filename, "")

        if profile.name == "python":
            test_name = f"test_{name}.{ext}"
        elif profile.name == "java":
            # Strip src/main/ prefix if present — test_root ("src/test") will be prepended
            # by write_test_file via the RepositoryManager.
            # Keep the sub-path (e.g. "java/com/example") so package structure is preserved.
            if directory.startswith("src/main/"):
                directory = directory[len("src/main/"):]
            elif directory == "src/main":
                directory = ""
            test_name = f"{name}Test.{ext}"
        elif profile.name == "go":
            test_name = f"{name}_test.{ext}"
        elif profile.name in ("typescript", "ts"):
            test_name = f"{name}.test.{ext}"
        elif profile.name == "rust":
            test_name = f"{name}_test.{ext}"
        elif profile.name in ("csharp", "c#"):
            test_name = f"{name}Tests.{ext}"
        else:
            test_name = f"test_{name}.{ext}"

        return f"{directory}/{test_name}" if directory else test_name

    async def _run_and_fix(
        self, test_path: str, test_code: str, context: AgentContext,
        profile: LanguageProfile, max_attempts: int = 4,
    ) -> TaskResult:
        """Autonomous debug loop: run tests, fix tests or source, rerun.

        Strategy per attempt:
          0 → fix test file  (LLM may have written bad assertions / imports)
          1 → fix source file (bug in the implementation exposed by tests)
          2 → fix test file  (adapt to corrected source)
          3 → fix source file (one more pass)
        """
        lang = context.file_blueprint.language if context.file_blueprint else "python"
        source_path = context.file_blueprint.path if context.file_blueprint else ""
        last_error = ""

        for attempt in range(max_attempts):
            cmd_result = await self.terminal.run_command(profile.test_command)

            if cmd_result.exit_code == 0:
                return TaskResult(
                    success=True,
                    output=f"Tests passing after {attempt + 1} attempt(s)",
                    files_modified=[test_path],
                    metrics={**self.get_metrics(), "fix_attempts": attempt},
                )

            error_output = f"{cmd_result.stdout}\n{cmd_result.stderr}".strip()
            last_error = error_output[:500]
            logger.info(f"Test failed (attempt {attempt + 1}/{max_attempts})")

            fix_test = attempt % 2 == 0  # even → fix test, odd → fix source

            if fix_test:
                logger.info("Fixing test file based on error output")
                fix_prompt = (
                    f"The following test file failed:\n\n"
                    f"```{profile.code_fence_name}\n{test_code}\n```\n\n"
                    f"Error output:\n```\n{error_output}\n```\n\n"
                    f"Fix the test code so it compiles and passes. "
                    f"Output only the complete corrected test file."
                )
                test_code = await self._call_llm(
                    fix_prompt, system_override=self._get_system_prompt(lang)
                )
                test_code = self._clean_code(test_code, profile)
                self.repo.write_test_file(test_path, test_code)
            else:
                logger.info("Fixing source file based on test failure")
                source_content = self.repo.read_file(source_path) or ""
                fix_prompt = (
                    f"The following source file has a bug exposed by failing tests:\n\n"
                    f"Source file ({source_path}):\n"
                    f"```{profile.code_fence_name}\n{source_content}\n```\n\n"
                    f"Test error output:\n```\n{error_output}\n```\n\n"
                    f"Fix the source file to make the tests pass. "
                    f"Output only the complete corrected source file."
                )
                fixed_source = await self._call_llm(
                    fix_prompt,
                    system_override=(
                        f"You are an expert {profile.display_name} developer. "
                        f"Fix bugs in the source file so the tests pass. "
                        f"Output only the complete corrected code, no markdown fences."
                    ),
                )
                fixed_source = self._clean_code(fixed_source, profile)
                if source_path:
                    self.repo.write_file(source_path, fixed_source)

        # Exhausted attempts — still mark success=True so the task is not retried.
        # The errors are logged for visibility.
        return TaskResult(
            success=True,
            output=f"Tests written (still failing after {max_attempts} fix attempts — manual review needed)",
            errors=[last_error],
            files_modified=[test_path, source_path] if source_path else [test_path],
            metrics={**self.get_metrics(), "fix_attempts": max_attempts, "tests_passing": False},
        )

    def _clean_code(self, code: str, profile: LanguageProfile) -> str:
        code = code.strip()
        fence = f"```{profile.code_fence_name}"
        if code.startswith(fence):
            code = code[len(fence):]
        elif code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
        return code.strip() + "\n"
