"""Test agent - generates unit and integration tests, runs them, and fixes failures."""

from __future__ import annotations

import logging
from typing import Any

from agents.base_agent import BaseAgent
from core.models import AgentContext, AgentRole, TaskResult
from tools.terminal_tools import TerminalTools

logger = logging.getLogger(__name__)


class TestAgent(BaseAgent):
    role = AgentRole.TESTER

    def __init__(self, *args: Any, terminal: TerminalTools | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.terminal = terminal

    @property
    def system_prompt(self) -> str:
        return (
            "You are a test engineering agent. You generate comprehensive test suites.\n\n"
            "Rules:\n"
            "- Generate ONLY the Python test file content, no markdown fences\n"
            "- Use pytest as the test framework\n"
            "- Use pytest-asyncio for async tests\n"
            "- Mock external dependencies (database, HTTP, etc.)\n"
            "- Test both happy paths and error cases\n"
            "- Test edge cases and boundary conditions\n"
            "- Use descriptive test names\n"
            "- Use fixtures for common setup\n"
            "- Include type hints\n"
            "- Generate both unit tests and integration tests where appropriate"
        )

    async def execute(self, context: AgentContext) -> TaskResult:
        """Generate tests for a file and optionally run them."""
        if not context.file_blueprint:
            return TaskResult(
                success=False,
                errors=[f"No blueprint for {context.task.file}"],
            )

        fb = context.file_blueprint
        logger.info(f"Generating tests for {fb.path}")

        formatted = self._format_context(context)
        prompt = (
            f"{formatted}\n\n"
            f"Generate comprehensive pytest tests for: {fb.path}\n"
            f"The file's purpose: {fb.purpose}\n"
            f"The file exports: {', '.join(fb.exports)}\n\n"
            "Generate complete, working pytest test code. Output only the code."
        )

        test_code = await self._call_llm(prompt)
        test_code = self._clean_code(test_code)

        # Compute test file path
        test_path = f"test_{fb.path.split('/')[-1]}"
        if "/" in fb.path:
            test_dir = fb.path.rsplit("/", 1)[0]
            test_path = f"{test_dir}/{test_path}"

        self.repo.write_test_file(test_path, test_code)

        # Run tests if terminal available (autonomous debug loop)
        if self.terminal:
            result = await self._run_and_fix(test_path, test_code, context)
            return result

        return TaskResult(
            success=True,
            output=f"Generated tests: {test_path}",
            files_modified=[test_path],
            metrics=self.get_metrics(),
        )

    async def _run_and_fix(
        self, test_path: str, test_code: str, context: AgentContext, max_attempts: int = 3
    ) -> TaskResult:
        """Autonomous debug loop: run tests, inspect errors, fix, rerun."""
        for attempt in range(max_attempts):
            result = await self.terminal.run_tests(test_path)

            if result.exit_code == 0:
                return TaskResult(
                    success=True,
                    output=f"Tests passing after {attempt + 1} attempt(s)",
                    files_modified=[test_path],
                    metrics={**self.get_metrics(), "fix_attempts": attempt},
                )

            # Ask LLM to fix the test
            logger.info(f"Test failed (attempt {attempt + 1}), asking LLM to fix")
            fix_prompt = (
                f"The following test file failed:\n\n"
                f"```python\n{test_code}\n```\n\n"
                f"Error output:\n```\n{result.stdout}\n{result.stderr}\n```\n\n"
                f"Fix the test code. Output only the complete corrected test file."
            )
            test_code = await self._call_llm(fix_prompt)
            test_code = self._clean_code(test_code)
            self.repo.write_test_file(test_path, test_code)

        return TaskResult(
            success=False,
            output=f"Tests still failing after {max_attempts} attempts",
            errors=[result.stderr[:500]],
            files_modified=[test_path],
            metrics={**self.get_metrics(), "fix_attempts": max_attempts},
        )

    def _clean_code(self, code: str) -> str:
        code = code.strip()
        if code.startswith("```python"):
            code = code[len("```python"):]
        elif code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
        return code.strip() + "\n"
