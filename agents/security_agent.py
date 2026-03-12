"""Security agent - scans generated code for vulnerabilities."""

from __future__ import annotations

import json
import logging
from typing import Any

from agents.base_agent import BaseAgent
from core.models import AgentContext, AgentRole, TaskResult
from tools.terminal_tools import TerminalTools

logger = logging.getLogger(__name__)


class SecurityAgent(BaseAgent):
    role = AgentRole.SECURITY

    def __init__(self, *args: Any, terminal: TerminalTools | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.terminal = terminal

    @property
    def system_prompt(self) -> str:
        return (
            "You are a security review agent. You analyze source code for real, "
            "exploitable vulnerabilities.\n\n"

            "You MUST respond with a JSON object (no markdown fences, no prose):\n"
            "{\n"
            '  "passed": true or false,\n'
            '  "vulnerabilities": [\n'
            "    {\n"
            '      "severity": "critical" | "high" | "medium" | "low",\n'
            '      "file": "exact/path/to/file",\n'
            '      "line": <line-number>,\n'
            '      "type": "<one of the allowed types below>",\n'
            '      "description": "Specific description of the vulnerability",\n'
            '      "remediation": "Specific code change to fix it"\n'
            "    }\n"
            "  ],\n"
            '  "summary": "One-sentence overall security assessment"\n'
            "}\n\n"

            "ALLOWED VULNERABILITY TYPES:\n"
            "SQL_INJECTION, COMMAND_INJECTION, XSS, SSRF, PATH_TRAVERSAL, "
            "INSECURE_DESERIALIZATION, HARDCODED_SECRET, INSECURE_CRYPTO, "
            "MISSING_AUTH, MISSING_INPUT_VALIDATION, INSECURE_DEPENDENCY, "
            "INFORMATION_DISCLOSURE, RACE_CONDITION\n\n"

            "SEVERITY DEFINITIONS:\n"
            "- critical: Directly exploitable, leads to data breach or RCE "
            "(e.g., unsanitized SQL concatenation, command injection with user input)\n"
            "- high: Exploitable with some effort (e.g., XSS in user-facing endpoint, "
            "missing authentication on admin route)\n"
            "- medium: Potential risk requiring specific conditions (e.g., weak crypto, "
            "verbose error messages leaking stack traces)\n"
            "- low: Best practice violation, minimal real-world risk "
            "(e.g., missing rate limiting, debug mode enabled)\n\n"

            "PASS/FAIL RULE:\n"
            "- Set \"passed\": false if ANY critical or high severity vulnerability exists\n"
            "- Set \"passed\": true if only medium/low findings or no findings\n\n"

            "FALSE POSITIVE AVOIDANCE:\n"
            "- Do NOT flag framework-managed security features as vulnerabilities "
            "(e.g., Spring Security CSRF protection, Django ORM parameterization, "
            "Express.js helmet middleware)\n"
            "- Do NOT flag parameterized queries / prepared statements as SQL injection\n"
            "- Do NOT flag dependencies without evidence of a known vulnerability\n"
            "- Only report vulnerabilities you can specifically identify in the actual code shown"
        )

    async def execute(self, context: AgentContext) -> TaskResult:
        """Run security scan on the codebase."""
        logger.info("Running security scan")

        # Step 1: Static analysis with bandit (if terminal available)
        bandit_findings: list[dict[str, Any]] = []
        if self.terminal:
            bandit_result = await self.terminal.run_security_scan()
            if bandit_result.exit_code == 0 and bandit_result.stdout:
                try:
                    bandit_data = json.loads(bandit_result.stdout)
                    bandit_findings = bandit_data.get("results", [])
                except json.JSONDecodeError:
                    pass

        # Step 2: LLM-based security review
        formatted = self._format_context(context)
        bandit_section = ""
        if bandit_findings:
            bandit_section = (
                "\n\n## Bandit Static Analysis Findings\n"
                + json.dumps(bandit_findings[:20], indent=2)
            )

        prompt = (
            f"{formatted}{bandit_section}\n\n"
            "Perform a security review of the codebase shown above.\n\n"
            "For each file, check:\n"
            "1. User input handling: Is input validated and sanitized before use in "
            "queries, commands, file paths, or HTML output?\n"
            "2. Authentication/authorization: Are sensitive endpoints protected?\n"
            "3. Secrets: Are credentials, API keys, or tokens hardcoded?\n"
            "4. Data exposure: Do error messages or logs leak sensitive information?\n"
            "5. Cryptography: Is encryption/hashing using modern algorithms?\n\n"
            "Only report vulnerabilities you can identify in the actual code. "
            "Use the allowed vulnerability types from your instructions."
        )

        data = await self._call_llm_json(prompt)

        # Validate and normalise — raises ValidationError if required fields
        # (passed, summary) are missing.
        from pydantic import ValidationError
        from core.llm_schema import validate_security_response
        try:
            validated = validate_security_response(data)
        except ValidationError as e:
            return TaskResult(
                success=False,
                errors=[f"LLM output validation failed: {e}"],
            )

        vulns = validated.vulnerabilities
        critical = [v for v in vulns if v.severity in ("critical", "high")]
        passed = validated.passed
        summary = validated.summary or "Security scan complete"
        status = "PASSED" if passed else f"NEEDS ATTENTION ({len(critical)} critical/high)"

        return TaskResult(
            # Security scan completing is always a successful task execution.
            # Vulnerabilities are findings, not task failures — they don't block the pipeline.
            success=True,
            output=f"[{status}] {summary}",
            errors=[v.description for v in critical],
            metrics={
                "total_vulnerabilities": len(vulns),
                "critical_count": len(critical),
                "passed": passed,
                "bandit_findings": len(bandit_findings),
                **self.get_metrics(),
            },
        )
