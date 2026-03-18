"""Test agent - generates unit and integration tests, runs them, and fixes failures."""

from __future__ import annotations

import logging
import re
from typing import Any

from agents.base_agent import BaseAgent
from core.coverage_runner import CoverageRunner
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
            f"You are a senior test engineer specializing in {profile.display_name}.\n\n"

            "YOUR TASK: Generate a focused, compilable, runnable test file that "
            "covers the CORE functionality of the source file under test.\n\n"

            "OUTPUT FORMAT:\n"
            f"- Output ONLY raw {profile.display_name} code — no markdown fences, "
            "no explanations, no commentary, no text before or after the code\n"
            "- The very first line must be a valid language statement "
            f"(package declaration, import, or {profile.display_name} comment)\n\n"

            "CRITICAL REQUIREMENTS:\n"
            "1. The test file MUST compile and run independently with zero modifications\n"
            "2. Every import MUST match the EXACT package/module path of the source file — "
            "copy import paths directly from the source code provided\n"
            "3. Every test method MUST have ≥1 specific assertion that checks a concrete "
            "value, type, exception, or state change — NEVER use assertTrue(true), "
            "assertNotNull(result), or any trivial placeholder as the ONLY assertion\n"
            "4. Test REAL method signatures from the source code — parameter names, types, "
            "and return types must match the source EXACTLY\n"
            "5. Focus on the MAIN public API — test only the most important public "
            "methods/endpoints that define the file's core responsibility\n"
            "6. Write 1 happy-path test per core method. Add 1 error-case test ONLY for "
            "methods that handle user input, I/O, or critical business logic\n"
            "7. Skip edge cases, boundary tests, and exhaustive permutations — keep it lean\n"
            "8. Aim for 3–8 total tests per file, NOT 3 per method\n\n"

            "TEST STRUCTURE (follow this order):\n"
            "1. Package/module declaration (if required by language)\n"
            "2. All imports (source under test + test framework + mocking library)\n"
            "3. Test class/suite declaration with setup/teardown\n"
            f"4. Use the standard {profile.display_name} test framework\n"
            "5. For the CORE public methods, write:\n"
            "   a) HAPPY PATH: valid input → verify exact expected return value\n"
            "   b) ERROR CASE (only for critical methods): invalid input → verify "
            "exact exception type\n\n"

            "MOCKING RULES:\n"
            "- Mock ONLY external dependencies (DB, HTTP, filesystem, message queues)\n"
            "- NEVER mock the class/module under test\n"
            "- When dependency interfaces are provided in the prompt, mock those EXACT "
            "signatures — do not guess method names or parameter types\n"
            "- Configure mock return values for happy-path tests\n"
            "- Configure mock exceptions for error-case tests\n"
            "- Verify mock interactions: assert mocks were called with expected arguments\n\n"

            "ASSERTION RULES:\n"
            "- assertEqual/assertEquals with SPECIFIC expected values (e.g., "
            "assertEquals(42, result) not assertNotNull(result))\n"
            "- assertRaises/assertThrows with SPECIFIC exception types\n"
            "- Assert collection sizes AND specific elements, not just non-emptiness\n"
            "- Assert state changes on mocks (e.g., verify save() was called with correct entity)\n\n"

            "TEST ISOLATION & DETERMINISM:\n"
            "- Each test must be independent: set up its own state, clean up after\n"
            "- Use fixtures/setUp for common initialization code\n"
            "- NEVER depend on test execution order\n"
            "- NEVER use real time (sleep, current timestamp) — mock time-dependent behavior\n"
            "- NEVER use random values without seeding — use fixed test data\n"
            "- NEVER access real filesystem, network, or databases — mock all I/O\n\n"

            "TEST NAMING:\n"
            "- Names must describe: WHAT is tested + WHAT input + WHAT is expected\n"
            "- Example patterns: test_methodName_whenCondition_thenExpectedResult, "
            "test_createUser_withNullEmail_throwsValidationError"
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

        # Compute test file path based on language conventions
        test_path = self._compute_test_path(fb.path, profile)

        # Get the actual source code for explicit inclusion in the prompt
        source_code = context.related_files.get(fb.path, "")

        # Build language-specific hints (package declaration, framework, etc.)
        lang_hints = self._build_test_hints(test_path, fb, profile)

        # ── Build explicit dependency interface section ────────────────────
        # Extract interfaces of files the source depends on so the LLM can
        # create accurate mocks instead of guessing method signatures.
        dep_interfaces: list[str] = []
        if fb.depends_on and context.related_files:
            for dep_path in fb.depends_on:
                dep_content = context.related_files.get(dep_path, "")
                if dep_content and dep_path != fb.path:
                    # AST stubs (from context_builder) are already compact
                    # and contain full method signatures — use them as-is.
                    # Only truncate non-stub content (full source fallback).
                    is_stub = dep_content.startswith("// AST stub")
                    cap = 4000 if is_stub else 2000
                    dep_interfaces.append(
                        f"### {dep_path} (dependency — mock this interface)\n"
                        f"```{profile.code_fence_name}\n{dep_content[:cap]}\n```"
                    )

        # ── Build exports checklist ───────────────────────────────────────
        exports_list = fb.exports or []
        exports_section = ""
        if exports_list:
            items = "\n".join(f"  - [ ] {export}" for export in exports_list)
            exports_section = (
                f"PUBLIC API THAT MUST BE TESTED (every item must have ≥3 tests):\n"
                f"{items}\n\n"
            )

        prompt = (
            f"Generate comprehensive tests for: {fb.path}\n"
            f"Purpose: {fb.purpose}\n"
            f"Layer: {fb.layer}\n\n"
        )

        # Include source code — this is the primary input
        if source_code:
            prompt += (
                f"SOURCE CODE TO TEST (use EXACT method signatures from this):\n"
                f"```{profile.code_fence_name}\n{source_code}\n```\n\n"
            )

        # Include dependency interfaces for accurate mocking
        if dep_interfaces:
            prompt += (
                "DEPENDENCY INTERFACES (mock these — use exact method signatures):\n"
                + "\n".join(dep_interfaces) + "\n\n"
            )

        # When an API contract is available and this is a controller / handler file,
        # embed the endpoint schemas so the LLM generates tests that use realistic
        # request bodies and validates the documented response shapes.
        if context.api_contract and fb.layer.lower() in (
            "controller", "handler", "router", "route", "api", "resource"
        ):
            import json as _json
            ac = context.api_contract
            contract_lines = [
                "API CONTRACT — generate tests that exercise these endpoints with "
                "the documented request/response shapes:",
                f"  Base URL: {ac.base_url}",
            ]
            for ep in ac.endpoints:
                auth = " [auth required]" if ep.auth_required else ""
                contract_lines.append(f"  {ep.method} {ep.path}{auth} — {ep.description}")
                if ep.request_schema:
                    try:
                        contract_lines.append(f"    Request example: {_json.dumps(ep.request_schema)}")
                    except Exception:
                        pass
                if ep.response_schema:
                    try:
                        contract_lines.append(f"    Expected response: {_json.dumps(ep.response_schema)}")
                    except Exception:
                        pass
            prompt += "\n".join(contract_lines) + "\n\n"

        prompt += exports_section

        prompt += (
            f"{lang_hints}\n\n"
            "MANDATORY CHECKLIST:\n"
            "1. Read the source code above — use EXACT class/method/function names\n"
            "2. Generate ≥3 test methods per public method (happy + error + edge)\n"
            f"3. Total minimum tests: {max(len(exports_list) * 3, 6)}\n"
            "4. Every test has ≥1 specific assertion (not assertNotNull or assertTrue(true))\n"
            "5. Mock all dependencies using the interfaces provided above\n"
            "6. Output ONLY the complete test file code — nothing else"
        )

        test_code = await self._call_llm(prompt, system_override=self._get_system_prompt(lang))
        test_code = self._clean_code(test_code, profile)
        await self.repo.async_write_test_file(test_path, test_code)

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

    def _build_targeted_test_command(self, test_path: str, profile: LanguageProfile) -> str:
        """Build a command that runs ONLY the specific test file/class.

        Instead of running all tests (e.g. ``mvn test``, ``pytest``), this
        targets the individual test so that errors from other test files do not
        contaminate the output or waste fix attempts.
        """
        filename = test_path.split("/")[-1]
        class_name = filename.rsplit(".", 1)[0] if "." in filename else filename

        # Workspace-relative path (prepend test_root when the language uses one)
        if profile.test_root and not test_path.startswith(profile.test_root):
            ws_test_path = f"{profile.test_root}/{test_path}"
        else:
            ws_test_path = test_path

        if profile.name == "python":
            return f"pytest -v --tb=short {ws_test_path}"
        elif profile.name == "java":
            # -Dtest targets a specific test class; Maven still compiles everything
            # but only RUNS the targeted test, keeping error output relevant.
            return f"mvn test -Dtest={class_name} -DfailIfNoTests=false"
        elif profile.name == "go":
            pkg_dir = ws_test_path.rsplit("/", 1)[0] if "/" in ws_test_path else "."
            return f"go test -v ./{pkg_dir}/"
        elif profile.name in ("typescript", "ts"):
            return f"npx jest --verbose {ws_test_path}"
        elif profile.name == "rust":
            return f"cargo test {class_name}"
        elif profile.name in ("csharp", "c#"):
            return f"dotnet test --filter FullyQualifiedName~{class_name}"
        else:
            return profile.test_command

    def _compute_java_package(self, test_path: str) -> str:
        """Extract Java package declaration from test path.

        E.g. ``java/com/example/service/UserServiceTest.java`` → ``com.example.service``
        """
        parts = test_path.replace("\\", "/").split("/")
        # Strip leading "java/" directory (it's a source-root, not a package)
        if parts and parts[0] == "java":
            parts = parts[1:]
        if parts:
            parts = parts[:-1]  # remove filename
        return ".".join(parts) if parts else ""

    def _build_test_hints(self, test_path: str, fb: Any, profile: LanguageProfile) -> str:
        """Build language-specific hints for test generation quality."""
        hints = [
            f"- Test file will be saved as: {test_path}",
            "- Import the classes/functions from the source file being tested",
            "- Test both happy paths and error/edge cases",
            "- Use descriptive test method names",
        ]

        if profile.name == "java":
            pkg = self._compute_java_package(test_path)
            if pkg:
                hints.insert(0, f"- Start with: package {pkg};")
            hints.append("- Use JUnit 5 (@Test, @BeforeEach, Assertions.assertEquals, etc.)")
            hints.append("- Use Mockito for mocking dependencies (@Mock, @InjectMocks, @ExtendWith(MockitoExtension.class))")
            hints.append("- Import from the correct package matching the source file")
        elif profile.name == "python":
            hints.append("- Use pytest assertions (assert x == y) and fixtures")
            hints.append("- Use unittest.mock (patch, MagicMock) for mocking dependencies")
        elif profile.name == "go":
            hints.append("- Use the standard testing package (testing.T)")
            hints.append("- Test file MUST be in the same package as the source")
        elif profile.name in ("typescript", "ts"):
            hints.append("- Use Jest describe/it/expect blocks")
            hints.append("- Use jest.mock() for mocking dependencies")
        elif profile.name == "rust":
            hints.append("- Use #[cfg(test)] mod tests { ... } or separate test file")
            hints.append("- Use assert_eq!, assert_ne!, assert! macros")
        elif profile.name in ("csharp", "c#"):
            hints.append("- Use xUnit or NUnit [Fact]/[Test] attributes")
            hints.append("- Use Moq for mocking dependencies")

        return "Requirements:\n" + "\n".join(hints)

    # Error messages that indicate a build/project configuration issue, not a code bug.
    _BUILD_CONFIG_ERRORS = (
        # Java / Maven / Gradle
        "no pom", "build failure", "could not determine",
        # Go
        "no go files", "cannot find module", "go.mod",
        # Node / TypeScript
        "package.json", "cannot find module", "enoent",
        # Rust
        "cargo.toml", "could not find cargo.toml",
        # C# / .NET
        "no project found", "msbuild", ".csproj",
        # Python
        "no module named", "modulenotfounderror",
        # Generic
        "could not find", "no such file or directory", "not a valid",
        "no build file", "compileerror", "cannot resolve",
    )

    def _is_build_config_error(self, error_output: str) -> bool:
        """Detect if test failure is caused by missing build config, not a code bug."""
        lower = error_output.lower()
        return any(marker in lower for marker in self._BUILD_CONFIG_ERRORS)

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

        Safeguards:
          - Runs only the SPECIFIC test file, not the entire suite
          - Build/config errors (no pom.xml etc.) → only fix tests, never touch source
          - Size guard: never replace source with content < 50% of original (prevents destruction)
          - Error output truncated to 3000 chars to prevent token explosion
          - Fix prompts re-include original requirements to prevent test simplification
        """
        lang = context.file_blueprint.language if context.file_blueprint else "python"
        source_path = context.file_blueprint.path if context.file_blueprint else ""
        exports_list = context.file_blueprint.exports if context.file_blueprint else []
        last_error = ""
        source_modified = False

        # Build targeted command that runs ONLY this specific test
        targeted_cmd = self._build_targeted_test_command(test_path, profile)
        logger.info(f"Test command: {targeted_cmd}")

        # Read source code once for context in fix prompts (non-blocking)
        source_for_context = await self.repo.async_read_file(source_path) or ""
        # Truncate for prompt inclusion (avoid token bloat)
        source_snippet = source_for_context[:4000]

        # Pre-build the requirements reminder to include in every fix prompt
        requirements_reminder = (
            "REQUIREMENTS (do NOT reduce test count or simplify tests to make them pass):\n"
            f"- Must test these exports: {', '.join(exports_list) if exports_list else 'all public methods'}\n"
            f"- Minimum {max(len(exports_list) * 3, 6)} test methods total\n"
            "- Each test must have ≥1 specific assertion (no assertTrue(true) or assertNotNull-only)\n"
            "- Happy path + error case + edge case for each public method\n"
        )

        for attempt in range(max_attempts):
            cmd_result = await self.terminal.run_command(targeted_cmd)

            if cmd_result.exit_code == 0:
                quality = self._check_test_quality(test_code, profile, exports_list)
                quality_note = ""
                if quality["test_count"] < 2:
                    quality_note = " (WARNING: only {n} test(s) — consider adding more)".format(
                        n=quality["test_count"]
                    )
                if quality.get("untested_exports"):
                    quality_note += (
                        f" (WARNING: untested exports: "
                        f"{', '.join(quality['untested_exports'])})"
                    )

                # ── Coverage measurement ──────────────────────────────
                coverage_metrics: dict[str, Any] = {}
                if self.terminal and source_path:
                    try:
                        cov = await CoverageRunner().run_with_coverage(
                            test_file=test_path,
                            source_file=source_path,
                            terminal=self.terminal,
                            lang=profile,
                            min_coverage=0.80,
                        )
                        coverage_metrics = {
                            "line_coverage": cov.line_coverage,
                            "branch_coverage": cov.branch_coverage,
                            "coverage_gate_passed": cov.passed_gate,
                        }
                        if not cov.passed_gate:
                            quality_note += (
                                f" (coverage {cov.line_pct} below 80% gate)"
                            )
                            logger.warning(
                                "Coverage gate failed for %s: %s < 80%%",
                                source_path, cov.line_pct,
                            )
                    except Exception as cov_err:
                        logger.debug("Coverage measurement skipped: %s", cov_err)

                return TaskResult(
                    success=True,
                    output=f"Tests passing after {attempt + 1} attempt(s){quality_note}",
                    files_modified=[test_path],
                    metrics={
                        **self.get_metrics(),
                        "fix_attempts": attempt,
                        **quality,
                        **coverage_metrics,
                    },
                )

            error_output = f"{cmd_result.stdout}\n{cmd_result.stderr}".strip()
            last_error = error_output[:500]
            # Truncate to prevent token explosion in fix prompts (Maven logs can be huge)
            error_for_prompt = error_output[:3000]
            logger.info(f"Test failed (attempt {attempt + 1}/{max_attempts})")

            # If the error is a build/config issue (no pom.xml etc.),
            # only fix tests — touching source code won't help.
            is_build_error = self._is_build_config_error(error_output)
            fix_test = attempt % 2 == 0 or is_build_error

            if is_build_error and attempt == 0:
                logger.warning(
                    "Build configuration error detected — fixing tests only, "
                    "source files will not be modified"
                )

            if fix_test:
                logger.info("Fixing test file based on error output")
                fix_prompt = (
                    f"The following test file failed:\n\n"
                    f"```{profile.code_fence_name}\n{test_code}\n```\n\n"
                )
                # Include source code so the LLM knows what correct behavior looks like
                if source_snippet:
                    fix_prompt += (
                        f"Source code being tested ({source_path}) — this is the SOURCE OF TRUTH:\n"
                        f"```{profile.code_fence_name}\n{source_snippet}\n```\n\n"
                    )
                fix_prompt += (
                    f"Error output:\n```\n{error_for_prompt}\n```\n\n"
                    f"{requirements_reminder}\n"
                    "FIX INSTRUCTIONS:\n"
                    "1. The SOURCE CODE above is correct — adapt the TESTS to match it\n"
                    "2. Fix import paths to match the actual source file location\n"
                    "3. Fix method calls to match actual method signatures in the source\n"
                    "4. Fix assertions to match the actual return types and values\n"
                    "5. Do NOT remove or simplify tests — fix them so they pass while "
                    "keeping the same test count and coverage\n"
                    "6. If a test method is fundamentally wrong, rewrite it for the same "
                    "scenario (happy/error/edge) — do not delete it\n"
                    "7. Output the COMPLETE corrected test file, nothing else"
                )
                test_code = await self._call_llm(
                    fix_prompt, system_override=self._get_system_prompt(lang)
                )
                test_code = self._clean_code(test_code, profile)
                await self.repo.async_write_test_file(test_path, test_code)
            else:
                source_content = await self.repo.async_read_file(source_path) or ""
                original_size = len(source_content)
                logger.info("Fixing source file based on test failure")

                fix_prompt = (
                    f"The following source file has a bug exposed by failing tests:\n\n"
                    f"Source file ({source_path}):\n"
                    f"```{profile.code_fence_name}\n{source_content}\n```\n\n"
                    f"Failing test file ({test_path}):\n"
                    f"```{profile.code_fence_name}\n{test_code}\n```\n\n"
                    f"Test error output:\n```\n{error_for_prompt}\n```\n\n"
                    f"FIX RULES:\n"
                    f"1. Fix ONLY the specific bug(s) that cause the test failures\n"
                    f"2. Do NOT remove, rename, or simplify any methods/classes/functions\n"
                    f"3. Do NOT change method signatures (names, parameters, return types)\n"
                    f"4. Keep the file structure, size, and all existing functionality intact\n"
                    f"5. The fix should be minimal — change only the lines needed\n"
                    f"6. Output the COMPLETE corrected source file, no markdown fences"
                )
                fixed_source = await self._call_llm(
                    fix_prompt,
                    system_override=(
                        f"You are an expert {profile.display_name} developer. "
                        f"Fix the specific bug in the source file so the tests pass. "
                        f"CRITICAL: Do NOT simplify, shorten, or restructure the file. "
                        f"Do NOT change any method signatures. Keep all existing methods, "
                        f"classes, and functionality intact. Only change the minimum lines "
                        f"necessary to fix the failing test. "
                        f"Output only the complete corrected code, no markdown fences."
                    ),
                )
                fixed_source = self._clean_code(fixed_source, profile)

                # Size guard: reject rewrites that destroy content (< 50% of original)
                if source_path and original_size > 0 and len(fixed_source) < original_size * 0.5:
                    logger.warning(
                        f"Source fix rejected: new size ({len(fixed_source)}) is less than "
                        f"50% of original ({original_size}). Keeping original to prevent "
                        f"content destruction."
                    )
                # Size guard: reject rewrites that inflate content (> 135% of original)
                elif source_path and original_size > 0 and len(fixed_source) > original_size * 1.35:
                    logger.warning(
                        f"Source fix rejected: new size ({len(fixed_source)}) exceeds 135% "
                        f"of original ({original_size}). Keeping original to prevent "
                        f"content duplication."
                    )
                elif source_path:
                    await self.repo.async_write_file(source_path, fixed_source)
                    source_modified = True
                    # Refresh source snippet for next fix iteration
                    source_snippet = fixed_source[:4000]

        # Exhausted attempts — report partial success so downstream agents know
        # tests exist but are not passing. success=False signals the pipeline
        # that manual intervention is needed.
        return TaskResult(
            success=False,
            output=f"Tests written but still failing after {max_attempts} fix attempts — manual review needed",
            errors=[last_error],
            files_modified=[test_path, source_path] if source_modified else [test_path],
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
        code = code.strip() + "\n"
        # Remove duplicated content blocks caused by LLM continuation failures
        code = self._deduplicate_content(code)
        return code

    # ── Test quality gate ────────────────────────────────────────────────

    # Language-specific regex patterns that match individual test functions/methods
    _TEST_FUNCTION_PATTERNS: dict[str, re.Pattern[str]] = {
        "python": re.compile(r"^\s*(?:def|async\s+def)\s+test_\w+", re.MULTILINE),
        "java": re.compile(r"@Test\b", re.MULTILINE),
        "go": re.compile(r"^func\s+Test\w+\(", re.MULTILINE),
        "typescript": re.compile(r"(?:it|test)\s*\(", re.MULTILINE),
        "rust": re.compile(r"#\[test\]", re.MULTILINE),
        "csharp": re.compile(r"\[(?:Fact|Test|Theory)\]", re.MULTILINE),
    }

    # Patterns that indicate a trivial / placeholder assertion
    _TRIVIAL_ASSERTION = re.compile(
        r"assert\s+True|assertTrue\s*\(\s*true\s*\)|expect\s*\(\s*true\s*\)",
        re.IGNORECASE,
    )

    def _check_test_quality(self, test_code: str, profile: LanguageProfile,
                           exports: list[str] | None = None) -> dict[str, Any]:
        """Quality gate — count test functions, flag trivial assertions, check export coverage.

        Returns a metrics dict that gets merged into TaskResult.metrics.
        """
        pattern = self._TEST_FUNCTION_PATTERNS.get(profile.name)
        test_count = len(pattern.findall(test_code)) if pattern else 0

        trivial_count = len(self._TRIVIAL_ASSERTION.findall(test_code))
        has_assertions = bool(re.search(
            r"assert|expect|should|assertEquals|assertEqual|assert_eq",
            test_code, re.IGNORECASE,
        ))

        # Check which exported symbols have corresponding test methods
        exports = exports or []
        tested_exports: list[str] = []
        untested_exports: list[str] = []
        test_code_lower = test_code.lower()
        for export in exports:
            # Check if the export name appears in any test method name or body
            if export.lower() in test_code_lower:
                tested_exports.append(export)
            else:
                untested_exports.append(export)

        export_coverage = len(tested_exports) / len(exports) if exports else 1.0

        if test_count == 0:
            logger.warning("Quality gate: no test functions detected in generated test")
        elif trivial_count > 0:
            logger.warning(
                "Quality gate: %d trivial assertion(s) (e.g. assert True) detected", trivial_count
            )
        elif not has_assertions:
            logger.warning("Quality gate: no assertions found in test code")
        if untested_exports:
            logger.warning(
                "Quality gate: %d export(s) not covered by tests: %s",
                len(untested_exports), ", ".join(untested_exports),
            )

        return {
            "test_count": test_count,
            "trivial_assertions": trivial_count,
            "has_assertions": has_assertions,
            "export_coverage": round(export_coverage, 2),
            "untested_exports": untested_exports,
        }
