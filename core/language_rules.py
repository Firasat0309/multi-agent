"""Pluggable language-specific rules registry.

Extracts hard-coded ``_LANGUAGE_SYNTAX_RULES`` dicts from ``coder_agent.py``
into a registry that can be extended without modifying agent code.

Rules are injected into the coder system prompt to guide the LLM toward
correct, framework-specific code generation.

Usage::

    from core.language_rules import register_rules, get_rules

    # At startup:
    register_rules("java", '''
    - Spring Boot 3.x: Use SecurityFilterChain, not WebSecurityConfigurerAdapter
    - Always use constructor injection for @Service/@Repository beans
    ''')

    # In CoderAgent:
    rules = get_rules("java")
    if rules:
        system_prompt += f"\\n\\n## Language-specific rules\\n{rules}"
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Data directory for external rule files ────────────────────────────────────

_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "language_rules"

# ── Registry ─────────────────────────────────────────────────────────────────

_RULES: dict[str, str] = {}


def register_rules(language: str, rules: str) -> None:
    """Register or replace language-specific rules."""
    _RULES[language.lower()] = rules.strip()


def get_rules(language: str) -> str:
    """Get rules for a language, or empty string if none registered."""
    return _RULES.get(language.lower(), "")


def has_rules(language: str) -> bool:
    """Check if rules are registered for a language."""
    return language.lower() in _RULES


def list_languages() -> list[str]:
    """Return languages with registered rules."""
    return sorted(_RULES.keys())


def clear_rules() -> None:
    """Remove all rules (for testing)."""
    _RULES.clear()


def load_rules_from_directory(directory: Path | None = None) -> int:
    """Load ``*.txt`` rule files from *directory* (default: ``data/language_rules/``).

    Each file is named ``<language>.txt`` and its contents are registered
    under the corresponding language key.  Returns the number of files loaded.
    """
    d = directory or _DATA_DIR
    if not d.is_dir():
        logger.debug("Language rules directory not found: %s", d)
        return 0
    count = 0
    for path in sorted(d.glob("*.txt")):
        lang = path.stem.lower()
        try:
            text = path.read_text(encoding="utf-8").strip()
            if text:
                register_rules(lang, text)
                count += 1
        except Exception:
            logger.warning("Failed to load language rules from %s", path, exc_info=True)
    return count


# ── Default rules ────────────────────────────────────────────────────────────
# First, load from external data files (preferred — full, detailed rules).
# Then register compact fallbacks for languages that don't have a data file.

load_rules_from_directory()


def _fallback(lang: str, text: str) -> None:
    """Register *text* only if *lang* was not already loaded from a data file."""
    if not has_rules(lang):
        register_rules(lang, text)


_fallback("java", """\
CRITICAL Java / Spring Boot rules:
- Spring Boot 3.x removed WebSecurityConfigurerAdapter. Use a @Bean method returning SecurityFilterChain.
- Do NOT inject PasswordEncoder into the same @Configuration class that defines the PasswordEncoder @Bean — it causes a circular dependency. Instead, inline BCryptPasswordEncoder creation or use a separate config class.
- Always use constructor injection (not @Autowired fields) for @Service, @Repository, @RestController beans.
- Use jakarta.* package imports (not javax.*) for Spring Boot 3.x / Jakarta EE 10.
- @Entity classes must have a no-arg constructor (can be protected).
- Record types cannot be @Entity — use a class with @Data or explicit getters.
- @SpringBootApplication class must be in the root package (above all component packages).
- Use @RequiredArgsConstructor (Lombok) when available; otherwise write the constructor explicitly.
- Repository interfaces extend JpaRepository<EntityType, IdType> — never implement them.
- Test classes use @SpringBootTest + @AutoConfigureMockMvc for integration tests.
""")

_fallback("typescript", """\
CRITICAL TypeScript rules:
- Always use semicolons at the end of statements.
- Specify return types on all exported functions and methods.
- Use `readonly` for properties that should not be reassigned.
- Decorators go ABOVE the class/method definition, not on the same line.
- Use `import type { ... }` for type-only imports.
- Prefer `interface` over `type` for object shapes (better error messages).
- Use `strict: true` in tsconfig.json — never use `any` without explicit annotation.
- Express route handlers must call `next()` or send a response — never leave hanging.
- NestJS: @Injectable() decorators required on all service classes.
- Use `enum` for string unions with more than 3 values.
""")

_fallback("csharp", """\
CRITICAL C# / .NET rules:
- Use file-scoped namespaces (namespace Foo; not namespace Foo { ... }).
- All using directives at the top of the file, before the namespace declaration.
- Attributes go on separate lines above the target.
- Use auto-properties (get; set;) not backing fields unless logic is needed.
- Use `record` types for immutable DTOs.
- ASP.NET Core: Use builder.Services.AddXxx() pattern in Program.cs.
- Entity Framework: DbContext must have a constructor accepting DbContextOptions<T>.
- Use `async Task<T>` not `async void` (except event handlers).
- Prefer `IActionResult` return type for controllers.
- Use `[ApiController]` attribute on API controllers for automatic model validation.
""")

_fallback("go", """\
CRITICAL Go rules:
- Package declaration must match the directory name.
- Exported identifiers start with uppercase.
- Error handling: always check `err != nil` immediately after the call.
- Use `context.Context` as the first parameter for functions that do I/O.
- Use `defer` for cleanup (closing files, releasing locks).
- Struct embedding for composition, not inheritance.
- Use `errors.Is()` and `errors.As()` for error comparison (not ==).
- Use `http.HandlerFunc` for simple handlers, `http.Handler` interface for complex ones.
- Prefer table-driven tests with `t.Run()` subtests.
- Use `make()` for slices and maps, not `var` with nil initial value.
""")

_fallback("rust", """\
CRITICAL Rust rules:
- Use `Result<T, E>` for fallible functions, not panics.
- Prefer `&str` over `String` for function parameters.
- Use `impl Trait` for return types when the caller doesn't need to name the type.
- Derive Clone, Debug on all public structs.
- Use `thiserror` for library error types, `anyhow` for application error types.
- Actix-web: extractors are function parameters (web::Path, web::Json, web::Data).
- Axum: use `Router::new().route()` for route registration.
- Use `#[cfg(test)]` mod for unit tests in the same file.
- Prefer `Vec<T>` over arrays unless the size is known at compile time.
- Lifetimes: prefer owned types in structs unless performance requires borrowing.
""")

_fallback("python", """\
CRITICAL Python rules:
- Use type hints on all function signatures.
- Use `dataclass` or `pydantic.BaseModel` for data containers, not plain dicts.
- FastAPI: use `Depends()` for dependency injection, not global state.
- Django: models go in models.py, views in views.py, serializers in serializers.py.
- Flask: use Blueprints for route organization in multi-module apps.
- Use `async def` for I/O-bound functions; use `def` for CPU-bound.
- Use `pathlib.Path` not `os.path` for path manipulation.
- Use `logging` module not `print()` for diagnostic output.
- Test files: prefix with `test_`, test functions with `test_`.
- Use `@pytest.fixture` for test setup, not `setUp()` methods.
""")
