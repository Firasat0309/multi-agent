"""Integration Test Agent — generates end-to-end tests verifying cross-module interactions."""

from __future__ import annotations

import logging
from typing import Any

from agents.base_agent import BaseAgent
from core.language import get_language_profile, LanguageProfile
from core.models import AgentContext, AgentRole, TaskResult
from tools.terminal_tools import TerminalTools

logger = logging.getLogger(__name__)


class IntegrationTestAgent(BaseAgent):
    """Generates integration tests that verify interactions between modules.

    Produces tests that:
    - Invoke a controller/handler endpoint end-to-end
    - Verify the service layer is called correctly
    - Verify the repository/data layer is exercised
    - Mock *only* external boundaries (database, 3rd-party APIs, message queues)

    For REST APIs  : generates httpx/requests-based tests against the running app.
    For gRPC       : generates stub-based integration tests.
    For message queues: generates producer/consumer roundtrip tests.
    For CLI / batch: generates subprocess-level integration tests.
    """

    role = AgentRole.INTEGRATION_TESTER

    # Output directory for integration tests — language-agnostic fallback;
    # language-specific paths are computed in _compute_output_dir().
    _DEFAULT_OUTPUT_DIR = "tests/integration"

    def __init__(self, *args: Any, terminal: TerminalTools | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.terminal = terminal

    @property
    def system_prompt(self) -> str:
        return (
            "You are an integration test engineering agent.\n\n"
            "Rules:\n"
            "- Output ONLY the test file content — no markdown fences, no explanations\n"
            "- Tests must call real application code end-to-end, mocking ONLY external\n"
            "  boundaries (database connections, HTTP clients, message brokers)\n"
            "- Each test must have a clear arrange / act / assert structure\n"
            "- Cover at least: the happy path, one error/edge case, and one boundary value\n"
            "- Use the same test framework as the project's unit tests\n"
            "- Keep tests independent — each test sets up and tears down its own state\n"
            "- Prefer realistic data over minimal stubs to catch data-shape mismatches\n"
        )

    async def execute(self, context: AgentContext) -> TaskResult:
        """Generate integration tests for the entire project."""
        lang = context.blueprint.tech_stack.get("language", "python")
        profile = get_language_profile(lang)
        logger.info("Generating %s integration tests", profile.display_name)

        # Collect all source files for context (truncated to stay in token budget)
        source_summary = self._summarise_sources(context)
        arch_summary = context.architecture_summary or ""

        output_dir = self._compute_output_dir(profile)
        test_file = f"{output_dir}/test_integration.{profile.file_extensions[0].lstrip('.')}"

        prompt = (
            f"Generate integration tests for a {profile.display_name} project.\n\n"
            f"Architecture style: {context.blueprint.architecture_style}\n"
            f"Tech stack: {context.blueprint.tech_stack}\n\n"
        )
        if arch_summary:
            prompt += f"Architecture overview:\n{arch_summary[:2000]}\n\n"
        prompt += (
            f"Source modules:\n{source_summary}\n\n"
            f"Requirements:\n"
            f"- Test file: {test_file}\n"
            f"- Test the primary entry points / public APIs end-to-end\n"
            f"- Mock only external I/O (DB, HTTP, queues) at the boundary\n"
            f"- Include at least 3 integration test cases\n"
            f"- Output only the complete test file, no fences or commentary\n"
        )

        test_code = await self._call_llm(prompt)
        test_code = self._strip_fences(test_code, profile)

        await self.repo.async_write_test_file(test_file, test_code)
        logger.info("Integration test written: %s", test_file)

        # Optionally run integration tests when a terminal is available
        run_result: dict[str, Any] = {}
        if self.terminal:
            run_result = await self._run_integration_tests(test_file, profile)

        return TaskResult(
            success=True,
            output=f"Integration tests written: {test_file}",
            files_modified=[test_file],
            metrics={
                **self.get_metrics(),
                "integration_test_file": test_file,
                **run_result,
            },
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _compute_output_dir(self, profile: LanguageProfile) -> str:
        """Return the integration test directory for the given language."""
        mapping = {
            "python": "tests/integration",
            "java": "src/test/java/integration",
            "go": "integration",
            "typescript": "tests/integration",
            "rust": "tests",
            "csharp": "tests/Integration",
        }
        return mapping.get(profile.name, self._DEFAULT_OUTPUT_DIR)

    def _summarise_sources(self, context: AgentContext, max_chars: int = 8000) -> str:
        """Build a compact module list from the blueprint for the LLM prompt."""
        lines: list[str] = []
        total = 0
        for fb in context.blueprint.file_blueprints:
            if fb.layer in ("test", "config", "deploy"):
                continue
            line = f"  {fb.path} [{fb.layer}] — {fb.purpose}"
            if fb.exports:
                line += f"  exports: {', '.join(fb.exports[:6])}"
            lines.append(line)
            total += len(line)
            if total >= max_chars:
                lines.append("  ... (truncated)")
                break
        return "\n".join(lines) if lines else "(no source modules found)"

    def _strip_fences(self, code: str, profile: LanguageProfile) -> str:
        code = code.strip()
        fence = f"```{profile.code_fence_name}"
        if code.startswith(fence):
            code = code[len(fence):]
        elif code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
        return code.strip() + "\n"

    async def _run_integration_tests(
        self, test_file: str, profile: LanguageProfile
    ) -> dict[str, Any]:
        """Run integration tests if a terminal is available. Non-blocking on failure."""
        cmd = self._build_run_command(test_file, profile)
        logger.info("Running integration tests: %s", cmd)
        try:
            result = await self.terminal.run_command(cmd)
            passed = result.exit_code == 0
            if not passed:
                logger.warning(
                    "Integration tests did not pass (exit=%d): %s",
                    result.exit_code,
                    (result.stdout + result.stderr)[:400],
                )
            return {"integration_tests_passed": passed, "integration_exit_code": result.exit_code}
        except Exception as e:
            logger.warning("Integration test run failed (non-critical): %s", e)
            return {"integration_tests_passed": False, "integration_run_error": str(e)}

    def _build_run_command(self, test_file: str, profile: LanguageProfile) -> str:
        if profile.name == "python":
            return f"pytest -v --tb=short {profile.test_root}/{test_file}"
        elif profile.name == "java":
            return "mvn failsafe:integration-test"
        elif profile.name == "go":
            return f"go test -v -tags=integration ./{test_file.rsplit('/', 1)[0]}/..."
        elif profile.name in ("typescript", "ts"):
            return "npx jest --verbose --testPathPattern=integration"
        elif profile.name == "rust":
            return "cargo test --test '*'"
        elif profile.name in ("csharp", "c#"):
            return "dotnet test --filter Category=Integration"
        return profile.test_command
