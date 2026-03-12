"""Integration Test Agent — generates end-to-end tests verifying cross-module interactions."""

from __future__ import annotations

import logging
from typing import Any

from agents.base_agent import BaseAgent
from core.language import detect_language_from_blueprint, get_language_profile, LanguageProfile
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
            "YOUR TASK: Generate complete, compilable, runnable integration tests that "
            "verify cross-module interactions end-to-end.\n\n"
            "CRITICAL REQUIREMENTS:\n"
            "1. Output ONLY the test file content — no markdown fences, no explanations\n"
            "2. The test file MUST compile and run independently\n"
            "3. Every import MUST match the actual package/module path of the source file\n"
            "4. Tests must call real application code end-to-end through "
            "controller → service → repository layers\n"
            "5. Mock ONLY external boundaries (database connections, HTTP clients to "
            "external services, message brokers) — NEVER mock the class under test\n\n"
            "TEST STRUCTURE:\n"
            "- Each test must have a clear arrange / act / assert structure\n"
            "- Include at least:\n"
            "  a) Happy-path test for each main endpoint (create, read, update, delete)\n"
            "  b) One error-path test (e.g., 404 for missing resource, 400 for invalid input)\n"
            "  c) One boundary test (empty collection, max-length input, concurrent access)\n"
            "- Each test must set up its own data and clean up after itself\n"
            "- Use realistic test data (real names, valid emails, proper dates) — not "
            "'test', 'foo', 'bar'\n"
            "- Keep tests independent — no test should depend on another test's state\n\n"
            "ASSERTION RULES:\n"
            "- Assert on HTTP status codes, response body structure, AND specific field values\n"
            "- Assert collection sizes and contents, not just non-emptiness\n"
            "- Verify error responses include appropriate error messages and status codes\n"
            "- NEVER use assertTrue(true) or assertNotNull(result) as the only assertion"
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
            f"Generate integration tests for a {profile.display_name} "
            f"{context.blueprint.architecture_style} project.\n\n"
            f"Project: {context.blueprint.name}\n"
            f"Tech stack: {context.blueprint.tech_stack}\n\n"
        )
        if arch_summary:
            prompt += f"Architecture overview:\n{arch_summary[:3000]}\n\n"

        # Include actual controller/endpoint code for accurate test generation
        endpoint_code = self._extract_endpoint_code(context)
        if endpoint_code:
            prompt += f"Endpoint definitions (use these EXACT signatures):\n{endpoint_code}\n\n"

        # Add language-specific test hints
        test_hints = self._build_test_hints(profile)

        prompt += (
            f"Source modules:\n{source_summary}\n\n"
            f"Language-specific guidance:\n{test_hints}\n\n"
            f"REQUIREMENTS:\n"
            f"- Test file: {test_file}\n"
            f"- Test the PRIMARY API endpoints end-to-end through "
            f"controller → service → repository\n"
            f"- Mock ONLY external boundaries: database connections, HTTP clients "
            f"to external services\n"
            f"- Include at least:\n"
            f"  1. Happy-path test for each main endpoint (create, read, update, delete)\n"
            f"  2. One error-path test (e.g., 404 for missing resource, 400 for invalid input)\n"
            f"  3. One boundary test (empty collection, max-length input)\n"
            f"- Each test must set up its own data and clean up after itself\n"
            f"- Use realistic test data (real names, valid emails, proper dates)\n"
            f"- Assert on HTTP status codes, response body structure, and specific field values\n"
            f"- Output ONLY the complete test file — no fences, no commentary\n"
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

    def _extract_endpoint_code(self, context: AgentContext) -> str:
        """Extract controller/endpoint source code from related files for the prompt."""
        endpoint_keywords = ("controller", "router", "handler", "endpoint", "route", "api")
        sections: list[str] = []
        total = 0
        max_chars = 6000

        for fb in context.blueprint.file_blueprints:
            if fb.layer == "controller" or any(kw in fb.path.lower() for kw in endpoint_keywords):
                content = context.related_files.get(fb.path, "")
                if content:
                    chunk = content[:2000]
                    sections.append(f"--- {fb.path} ---\n{chunk}")
                    total += len(chunk)
                    if total >= max_chars:
                        break

        return "\n\n".join(sections) if sections else ""

    def _build_test_hints(self, profile: LanguageProfile) -> str:
        """Return language-specific integration test guidance."""
        hints = {
            "python": (
                "- Use pytest with httpx.AsyncClient or TestClient (FastAPI/Starlette)\n"
                "- Use unittest.mock.patch or pytest-mock for mocking external services\n"
                "- Use fixtures for test data setup and teardown\n"
                "- For Django, use django.test.TestCase with self.client"
            ),
            "java": (
                "- Use @SpringBootTest with @AutoConfigureMockMvc\n"
                "- Use MockMvc for HTTP endpoint testing\n"
                "- Use @MockBean for external service mocking\n"
                "- Use @Transactional for automatic rollback after each test\n"
                "- Assert with MockMvcResultMatchers: status(), jsonPath(), content()"
            ),
            "go": (
                "- Use net/http/httptest.NewServer for test HTTP server\n"
                "- Use httptest.NewRecorder for response recording\n"
                "- Use testify/assert for assertions\n"
                "- Use interfaces for dependency injection and mocking"
            ),
            "typescript": (
                "- Use supertest for HTTP endpoint testing\n"
                "- Use jest.mock() for external service mocking\n"
                "- Use beforeEach/afterEach for setup/teardown\n"
                "- Assert response.status, response.body, and specific fields"
            ),
            "rust": (
                "- Use actix_web::test or axum::test for HTTP testing\n"
                "- Use #[tokio::test] for async tests\n"
                "- Use mockall for trait mocking"
            ),
            "csharp": (
                "- Use WebApplicationFactory<Program> for integration testing\n"
                "- Use HttpClient from factory.CreateClient()\n"
                "- Use Moq for external service mocking\n"
                "- Use [Fact] or [Theory] attributes for test methods"
            ),
        }
        return hints.get(profile.name, "- Use the standard test framework for this language")

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
            return f"npx jest --verbose --testPathPattern=integration"
        elif profile.name == "rust":
            return "cargo test --test '*'"
        elif profile.name in ("csharp", "c#"):
            return "dotnet test --filter Category=Integration"
        return profile.test_command
