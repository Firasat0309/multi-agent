"""Shared prompt template system for agents.

Eliminates prompt duplication across ReviewerAgent, SecurityAgent, TestAgent,
CoderAgent, etc. by providing composable template fragments that agents
combine into their system prompts.

Usage::

    from core.prompt_templates import PromptTemplates

    class MyAgent(BaseAgent):
        @property
        def system_prompt(self) -> str:
            return PromptTemplates.compose(
                PromptTemplates.role("code reviewer"),
                PromptTemplates.code_constraints(language="Java"),
                PromptTemplates.output_json(),
            )
"""

from __future__ import annotations

from string import Template
from typing import Any


class PromptTemplates:
    """Registry of composable prompt fragments shared across all agents."""

    # ── Role & Identity ──────────────────────────────────────────────────

    _ROLE = Template(
        "You are an expert $role in an automated code generation system.\n"
        "You produce high-quality, production-ready output.\n"
        "Follow the architecture and blueprint strictly."
    )

    # ── Constraints ──────────────────────────────────────────────────────

    _CODE_CONSTRAINTS = Template(
        "Follow these constraints strictly:\n"
        "- Output ONLY the requested content, no extra commentary\n"
        "- Follow the blueprint architecture exactly\n"
        "- Do not invent files or dependencies outside the plan\n"
        "- Use idiomatic $language patterns and conventions\n"
        "- Prefer standard library solutions over external dependencies"
    )

    _NO_HALLUCINATION = (
        "IMPORTANT: Do NOT hallucinate APIs, endpoints, classes, or methods "
        "that are not defined in the blueprint or related files provided. "
        "Use ONLY what is explicitly available in the context."
    )

    _FIX_CONSTRAINTS = Template(
        "You are fixing build errors. Rules:\n"
        "- Fix ONLY the errors described — do not refactor unrelated code\n"
        "- Preserve all existing functionality\n"
        "- If the same error appeared before, try a COMPLETELY DIFFERENT approach\n"
        "- Read the error message carefully — the fix is usually in the details\n"
        "- Max fix attempts remaining: $remaining"
    )

    # ── Output Format ────────────────────────────────────────────────────

    _OUTPUT_JSON = (
        "Respond with valid JSON only. No markdown fences, no explanations, "
        "no text outside the JSON object."
    )

    _OUTPUT_CODE = Template(
        "Respond with complete, compilable $language source code. "
        "No markdown fences, no explanations, no placeholder comments like "
        "'// TODO' or '// implement here'. Every function must have a real implementation."
    )

    _OUTPUT_DIFF = (
        "Respond with a unified diff patch. Include only changed lines with "
        "sufficient context (3 lines before/after). Do not include unchanged files."
    )

    # ── Security ─────────────────────────────────────────────────────────

    _SECURITY_RULES = (
        "Security requirements (OWASP Top 10):\n"
        "- Validate and sanitize ALL user inputs\n"
        "- Use parameterized queries for database access (never string concatenation)\n"
        "- Implement proper authentication and authorization checks\n"
        "- Never log sensitive data (passwords, tokens, PII)\n"
        "- Use secure defaults for cryptographic operations\n"
        "- Implement proper error handling that doesn't leak internals\n"
        "- Set appropriate CORS, CSP, and security headers"
    )

    # ── Testing ──────────────────────────────────────────────────────────

    _TEST_RULES = Template(
        "Test generation rules:\n"
        "- Generate focused, targeted tests — 3-8 per file, not per method\n"
        "- Test behavior and contracts, not implementation details\n"
        "- Use $framework as the test framework\n"
        "- Include both happy-path and error-path tests\n"
        "- Mock external dependencies (DB, HTTP, filesystem)\n"
        "- Tests must be deterministic and independent"
    )

    # ── Review ───────────────────────────────────────────────────────────

    _REVIEW_RULES = (
        "Code review guidelines:\n"
        "- Check for correctness, not style preferences\n"
        "- Flag potential bugs, race conditions, and edge cases\n"
        "- Verify error handling covers all failure modes\n"
        "- Check that public APIs match the blueprint specification\n"
        "- Severity levels: critical (must fix), warning (should fix), info (suggestion)"
    )

    # ── Tool Usage ───────────────────────────────────────────────────────

    _TOOL_USAGE = (
        "You have access to tools for reading files, writing files, and searching code. "
        "Use them to:\n"
        "1. Read related files to understand dependencies and APIs\n"
        "2. Write the complete file using write_file (NEVER output code as plain text)\n"
        "3. Search for definitions when you need to verify signatures or types\n\n"
        "CRITICAL: Always use write_file to produce output. Plain text code is NOT accepted."
    )

    # ── Public API ───────────────────────────────────────────────────────

    @classmethod
    def role(cls, role_name: str) -> str:
        """Generate a role identity prompt fragment."""
        return cls._ROLE.substitute(role=role_name)

    @classmethod
    def code_constraints(cls, language: str = "the target") -> str:
        """Generate code constraint rules for a language."""
        return cls._CODE_CONSTRAINTS.substitute(language=language)

    @classmethod
    def fix_constraints(cls, remaining: int = 3) -> str:
        """Generate fix-specific constraint rules."""
        return cls._FIX_CONSTRAINTS.substitute(remaining=remaining)

    @classmethod
    def no_hallucination(cls) -> str:
        return cls._NO_HALLUCINATION

    @classmethod
    def output_json(cls) -> str:
        return cls._OUTPUT_JSON

    @classmethod
    def output_code(cls, language: str = "source") -> str:
        return cls._OUTPUT_CODE.substitute(language=language)

    @classmethod
    def output_diff(cls) -> str:
        return cls._OUTPUT_DIFF

    @classmethod
    def security_rules(cls) -> str:
        return cls._SECURITY_RULES

    @classmethod
    def test_rules(cls, framework: str = "the standard") -> str:
        return cls._TEST_RULES.substitute(framework=framework)

    @classmethod
    def review_rules(cls) -> str:
        return cls._REVIEW_RULES

    @classmethod
    def tool_usage(cls) -> str:
        return cls._TOOL_USAGE

    @classmethod
    def compose(cls, *fragments: str) -> str:
        """Join multiple prompt fragments with double newlines."""
        return "\n\n".join(f for f in fragments if f)
