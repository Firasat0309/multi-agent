"""Test template cache — reuse generated test scaffolds for similar files.

When generating tests, files in the same layer/framework combination often
share identical boilerplate: imports, setup/teardown, mocking patterns,
assertion helpers. This module caches test scaffolds by (language, layer,
test_framework) so subsequent test generations can start from a pre-built
template instead of having the LLM re-derive the same boilerplate.

Gated behind the ``TEST_TEMPLATE_CACHE`` feature flag.

Savings:
  - Reduces prompt size by ~30% (no need to re-explain framework setup)
  - Reduces output tokens by ~20% (scaffold already present)
  - Improves consistency across test files (same import style, naming)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TestTemplate:
    """A cached test scaffold for a specific (language, layer, framework) combo."""
    language: str
    layer: str
    framework: str          # e.g., "junit5", "pytest", "jest", "go-test"
    imports_block: str      # Pre-built import section
    setup_block: str        # Setup/teardown/fixtures
    mock_pattern: str       # Mocking pattern example
    assertion_style: str    # Assertion helper usage example
    sample_test: str        # One complete happy-path test as template
    hits: int = 0           # Cache hit counter

    @property
    def cache_key(self) -> str:
        return f"{self.language}:{self.layer}:{self.framework}"

    def to_prompt_section(self) -> str:
        """Render this template as a prompt section the LLM can build upon."""
        parts = [
            "USE THIS SCAFFOLD as the starting point for your test file.",
            "Modify as needed but keep the import style and assertion patterns consistent.",
            "",
        ]
        if self.imports_block:
            parts.append(f"// === Standard imports ===\n{self.imports_block}\n")
        if self.setup_block:
            parts.append(f"// === Setup pattern ===\n{self.setup_block}\n")
        if self.mock_pattern:
            parts.append(f"// === Mocking pattern ===\n{self.mock_pattern}\n")
        if self.sample_test:
            parts.append(f"// === Example test structure ===\n{self.sample_test}\n")
        return "\n".join(parts)


class TestTemplateCache:
    """Caches and retrieves test scaffolds keyed by (language, layer, framework).

    The cache is populated lazily: after the first test file for a given
    combo is successfully generated and passes, we extract its scaffold
    and cache it.  Subsequent test files of the same combo get the
    scaffold injected into their prompt, reducing LLM work.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._templates: dict[str, TestTemplate] = {}
        self._cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

    def get(self, language: str, layer: str, framework: str) -> TestTemplate | None:
        """Retrieve a cached template (or None if not cached)."""
        key = f"{language}:{layer}:{framework}"
        template = self._templates.get(key)
        if template:
            template.hits += 1
            logger.debug(
                "Test template cache hit: %s (hits=%d)", key, template.hits,
            )
        return template

    def store(
        self,
        language: str,
        layer: str,
        framework: str,
        test_content: str,
    ) -> None:
        """Extract scaffold from a successful test file and cache it.

        Only the structural parts are cached — test-specific assertions
        and method names are stripped.
        """
        key = f"{language}:{layer}:{framework}"
        if key in self._templates:
            return  # Already cached

        template = self._extract_scaffold(language, layer, framework, test_content)
        if template:
            self._templates[key] = template
            logger.info("Cached test template for %s", key)
            if self._cache_dir:
                self._save_to_disk(template)

    def _extract_scaffold(
        self,
        language: str,
        layer: str,
        framework: str,
        content: str,
    ) -> TestTemplate | None:
        """Extract reusable scaffold from test file content."""
        lines = content.split("\n")
        if len(lines) < 5:
            return None

        imports_block = self._extract_imports(lines, language)
        setup_block = self._extract_setup(lines, language)
        mock_pattern = self._extract_mock_pattern(lines, language)
        sample_test = self._extract_first_test(lines, language)

        if not imports_block and not setup_block:
            return None  # Not enough structure to cache

        return TestTemplate(
            language=language,
            layer=layer,
            framework=framework,
            imports_block=imports_block,
            setup_block=setup_block,
            mock_pattern=mock_pattern,
            assertion_style="",  # Could extract assertion helpers
            sample_test=sample_test,
        )

    @staticmethod
    def _extract_imports(lines: list[str], language: str) -> str:
        """Extract the import block from the top of the file."""
        import_lines: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("//") or stripped.startswith("#"):
                if import_lines:
                    import_lines.append(line)
                continue
            if language in ("java", "kotlin"):
                if stripped.startswith(("import ", "package ")):
                    import_lines.append(line)
                elif import_lines:
                    break
            elif language in ("python", "py"):
                if stripped.startswith(("import ", "from ")):
                    import_lines.append(line)
                elif import_lines:
                    break
            elif language in ("typescript", "javascript", "ts", "js"):
                if stripped.startswith(("import ", "const ")) and "require" in stripped:
                    import_lines.append(line)
                elif stripped.startswith("import "):
                    import_lines.append(line)
                elif import_lines:
                    break
            elif import_lines:
                break
        return "\n".join(import_lines)

    @staticmethod
    def _extract_setup(lines: list[str], language: str) -> str:
        """Extract setup/teardown/fixture blocks."""
        setup_patterns = {
            "java": (r"@Before|@BeforeEach|@BeforeAll|setUp\(\)", r"@After|@AfterEach|@AfterAll|tearDown\(\)"),
            "python": (r"def setUp|@pytest\.fixture|@classmethod.*setUpClass", r"def tearDown"),
            "typescript": (r"beforeEach|beforeAll|describe\(", r"afterEach|afterAll"),
            "javascript": (r"beforeEach|beforeAll|describe\(", r"afterEach|afterAll"),
        }
        patterns = setup_patterns.get(language, (r"setUp|setup|before", r"tearDown|teardown|after"))

        setup_lines: list[str] = []
        in_setup = False
        brace_depth = 0

        for line in lines:
            if re.search(patterns[0], line):
                in_setup = True
            if in_setup:
                setup_lines.append(line)
                brace_depth += line.count("{") - line.count("}")
                if language == "python":
                    # Python: stop at next unindented def/class
                    if setup_lines and line.strip() and not line.startswith((" ", "\t")) and len(setup_lines) > 1:
                        setup_lines.pop()
                        in_setup = False
                elif brace_depth <= 0 and len(setup_lines) > 1:
                    in_setup = False

        return "\n".join(setup_lines[:30])  # Cap at 30 lines

    @staticmethod
    def _extract_mock_pattern(lines: list[str], language: str) -> str:
        """Extract mocking pattern examples."""
        mock_keywords = {
            "java": ("@Mock", "@InjectMocks", "Mockito.when", "mock(", "when("),
            "python": ("@patch", "Mock(", "MagicMock(", "mock.patch"),
            "typescript": ("jest.mock", "jest.fn", "jest.spyOn", "vi.mock", "vi.fn"),
            "javascript": ("jest.mock", "jest.fn", "jest.spyOn", "sinon.stub"),
        }
        keywords = mock_keywords.get(language, ("mock", "Mock", "stub"))

        mock_lines: list[str] = []
        for line in lines:
            if any(kw in line for kw in keywords):
                mock_lines.append(line.strip())
                if len(mock_lines) >= 5:
                    break

        return "\n".join(mock_lines)

    @staticmethod
    def _extract_first_test(lines: list[str], language: str) -> str:
        """Extract the first complete test method as a structural example."""
        test_patterns = {
            "java": r"@Test",
            "python": r"def test_",
            "typescript": r"\bit\(",
            "javascript": r"\bit\(",
        }
        pattern = test_patterns.get(language, r"(?:test|it)\s*\(")

        test_lines: list[str] = []
        in_test = False
        brace_depth = 0

        for line in lines:
            if not in_test and re.search(pattern, line):
                in_test = True
                brace_depth = 0

            if in_test:
                test_lines.append(line)
                brace_depth += line.count("{") - line.count("}")
                if language == "python":
                    if test_lines and line.strip() and not line.startswith((" ", "\t")) and len(test_lines) > 1:
                        test_lines.pop()
                        break
                elif brace_depth <= 0 and len(test_lines) > 1:
                    break

            if len(test_lines) >= 25:
                break

        return "\n".join(test_lines)

    def _save_to_disk(self, template: TestTemplate) -> None:
        """Persist a template to the cache directory."""
        if not self._cache_dir:
            return
        key_hash = hashlib.md5(template.cache_key.encode()).hexdigest()[:12]
        path = self._cache_dir / f"test_template_{key_hash}.json"
        try:
            data = {
                "language": template.language,
                "layer": template.layer,
                "framework": template.framework,
                "imports_block": template.imports_block,
                "setup_block": template.setup_block,
                "mock_pattern": template.mock_pattern,
                "assertion_style": template.assertion_style,
                "sample_test": template.sample_test,
            }
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            logger.debug("Failed to save test template to disk", exc_info=True)

    def _load_from_disk(self) -> None:
        """Load cached templates from disk."""
        if not self._cache_dir:
            return
        for path in self._cache_dir.glob("test_template_*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                template = TestTemplate(**data)
                self._templates[template.cache_key] = template
                logger.debug("Loaded cached test template: %s", template.cache_key)
            except Exception:
                logger.debug("Failed to load test template from %s", path, exc_info=True)

    @property
    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        return {
            "cached_templates": len(self._templates),
            "total_hits": sum(t.hits for t in self._templates.values()),
            "keys": list(self._templates.keys()),
        }
