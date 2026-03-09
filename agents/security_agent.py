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
            "You are a security review agent. You analyze code for vulnerabilities.\n\n"
            "You must respond with JSON:\n"
            "{\n"
            '  "passed": true/false,\n'
            '  "vulnerabilities": [\n'
            "    {\n"
            '      "severity": "critical|high|medium|low",\n'
            '      "file": "path",\n'
            '      "line": 42,\n'
            '      "type": "SQL_INJECTION|XSS|SSRF|...",\n'
            '      "description": "...",\n'
            '      "remediation": "how to fix"\n'
            "    }\n"
            "  ],\n"
            '  "summary": "overall security assessment"\n'
            "}\n\n"
            "Check for:\n"
            "- SQL injection\n"
            "- Command injection\n"
            "- XSS vulnerabilities\n"
            "- SSRF vulnerabilities\n"
            "- Insecure deserialization\n"
            "- Hardcoded secrets/credentials\n"
            "- Path traversal\n"
            "- Improper input validation\n"
            "- Insecure cryptography\n"
            "- Missing authentication/authorization"
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
            "Perform a comprehensive security review of the codebase."
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
