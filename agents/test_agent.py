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

        # Compute test file path based on language conventions
        test_path = self._compute_test_path(fb.path, profile)

        # Get the actual source code for explicit inclusion in the prompt
        source_code = context.related_files.get(fb.path, "")

        # Build language-specific hints (package declaration, framework, etc.)
        lang_hints = self._build_test_hints(test_path, fb, profile)

        formatted = self._format_context(context)
        prompt = (
            f"{formatted}\n\n"
            f"Generate comprehensive {profile.display_name} tests for: {fb.path}\n"
            f"The file's purpose: {fb.purpose}\n"
            f"The file exports: {', '.join(fb.exports)}\n\n"
        )

        # Explicitly include source code so the LLM knows what to test
        if source_code:
            prompt += (
                f"Here is the actual source code to test:\n"
                f"```{profile.code_fence_name}\n{source_code}\n```\n\n"
            )

        prompt += (
            f"{lang_hints}\n\n"
            f"Generate complete, working test code. Output only the code."
        )

        test_code = await self._call_llm(prompt, system_override=self._get_system_prompt(lang))
        test_code = self._clean_code(test_code, profile)
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
          - Error output truncated to 2000 chars to prevent token explosion
        """
        lang = context.file_blueprint.language if context.file_blueprint else "python"
        source_path = context.file_blueprint.path if context.file_blueprint else ""
        last_error = ""
        source_modified = False

        # Build targeted command that runs ONLY this specific test
        targeted_cmd = self._build_targeted_test_command(test_path, profile)
        logger.info(f"Test command: {targeted_cmd}")

        # Read source code once for context in fix prompts
        source_for_context = self.repo.read_file(source_path) or ""
        # Truncate for prompt inclusion (avoid token bloat)
        source_snippet = source_for_context[:4000]

        for attempt in range(max_attempts):
            cmd_result = await self.terminal.run_command(targeted_cmd)

            if cmd_result.exit_code == 0:
                return TaskResult(
                    success=True,
                    output=f"Tests passing after {attempt + 1} attempt(s)",
                    files_modified=[test_path],
                    metrics={**self.get_metrics(), "fix_attempts": attempt},
                )

            error_output = f"{cmd_result.stdout}\n{cmd_result.stderr}".strip()
            last_error = error_output[:500]
            # Truncate to prevent token explosion in fix prompts (Maven logs can be huge)
            error_for_prompt = error_output[:2000]
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
                        f"Source code being tested ({source_path}):\n"
                        f"```{profile.code_fence_name}\n{source_snippet}\n```\n\n"
                    )
                fix_prompt += (
                    f"Error output:\n```\n{error_for_prompt}\n```\n\n"
                    f"Fix the test code so it compiles and passes. "
                    f"The tests should accurately test the source code shown above. "
                    f"Output only the complete corrected test file."
                )
                test_code = await self._call_llm(
                    fix_prompt, system_override=self._get_system_prompt(lang)
                )
                test_code = self._clean_code(test_code, profile)
                self.repo.write_test_file(test_path, test_code)
            else:
                source_content = self.repo.read_file(source_path) or ""
                original_size = len(source_content)
                logger.info("Fixing source file based on test failure")

                fix_prompt = (
                    f"The following source file has a bug exposed by failing tests:\n\n"
                    f"Source file ({source_path}):\n"
                    f"```{profile.code_fence_name}\n{source_content}\n```\n\n"
                    f"Test error output:\n```\n{error_for_prompt}\n```\n\n"
                    f"Fix ONLY the specific bug(s) — do NOT simplify, remove methods, "
                    f"or reduce functionality. Keep the file structure and size intact. "
                    f"Output only the complete corrected source file."
                )
                fixed_source = await self._call_llm(
                    fix_prompt,
                    system_override=(
                        f"You are an expert {profile.display_name} developer. "
                        f"Fix the specific bug in the source file so the tests pass. "
                        f"IMPORTANT: Do NOT simplify or shorten the file. Keep all existing "
                        f"methods, classes, and functionality intact. Only change what is "
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
                elif source_path:
                    self.repo.write_file(source_path, fixed_source)
                    source_modified = True
                    # Refresh source snippet for next fix iteration
                    source_snippet = fixed_source[:4000]

        # Exhausted attempts — still mark success=True so the task is not retried.
        return TaskResult(
            success=True,
            output=f"Tests written (still failing after {max_attempts} fix attempts — manual review needed)",
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
        return code.strip() + "\n"
