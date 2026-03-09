"""Reviewer agent - performs hierarchical code review."""

from __future__ import annotations

import json
import logging
from typing import Any

from agents.base_agent import BaseAgent
from core.models import (
    AgentContext,
    AgentRole,
    ReviewFinding,
    ReviewLevel,
    ReviewResult,
    TaskResult,
    TaskType,
)

logger = logging.getLogger(__name__)


class ReviewerAgent(BaseAgent):
    role = AgentRole.REVIEWER

    @property
    def system_prompt(self) -> str:
        return (
            "You are a senior code reviewer agent. You perform thorough code reviews.\n\n"
            "You must respond with a JSON object:\n"
            "{\n"
            '  "passed": true/false,\n'
            '  "summary": "overall assessment",\n'
            '  "findings": [\n'
            "    {\n"
            '      "severity": "critical|warning|info",\n'
            '      "file": "path/to/file.py",\n'
            '      "line": 42,\n'
            '      "message": "description of issue",\n'
            '      "suggestion": "how to fix"\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Review criteria:\n"
            "- Code correctness and logic errors\n"
            "- Type safety and proper error handling\n"
            "- Adherence to the architecture blueprint\n"
            "- Proper imports and dependency usage\n"
            "- Security vulnerabilities (injection, XSS, etc.)\n"
            "- Performance issues\n"
            "- Code style and readability\n"
            "- Critical = must fix, Warning = should fix, Info = suggestion"
        )

    async def execute(self, context: AgentContext) -> TaskResult:
        """Execute the appropriate review level."""
        task_type = context.task.task_type

        if task_type == TaskType.REVIEW_FILE:
            review = await self._review_file(context)
        elif task_type == TaskType.REVIEW_MODULE:
            review = await self._review_module(context)
        elif task_type == TaskType.REVIEW_ARCHITECTURE:
            review = await self._review_architecture(context)
        else:
            return TaskResult(success=False, errors=[f"Unknown review type: {task_type}"])

        critical_count = sum(1 for f in review.findings if f.severity == "critical")
        status = "PASSED" if review.passed else f"NEEDS WORK ({critical_count} critical)"

        return TaskResult(
            # Review completing (even with findings) is a successful task execution.
            # Findings are reported as output/warnings, not task failures.
            success=True,
            output=f"[{status}] {review.summary}",
            errors=[f.message for f in review.findings if f.severity == "critical"],
            metrics={
                "review_level": review.level.value,
                "findings_count": len(review.findings),
                "critical_count": critical_count,
                "passed": review.passed,
            },
        )

    async def _review_file(self, context: AgentContext) -> ReviewResult:
        """Review a single file."""
        formatted = self._format_context(context)
        prompt = (
            f"{formatted}\n\n"
            f"Review the file: {context.task.file}\n"
            "Focus on correctness, security, and adherence to the blueprint."
        )

        data = await self._call_llm_json(prompt)
        return self._parse_review(data, ReviewLevel.FILE)

    async def _review_module(self, context: AgentContext) -> ReviewResult:
        """Review module-level consistency."""
        formatted = self._format_context(context)
        prompt = (
            f"{formatted}\n\n"
            "Perform a MODULE-LEVEL review checking:\n"
            "- Consistency between controllers, services, and repositories\n"
            "- Proper interface contracts between layers\n"
            "- No missing or broken imports between files\n"
            "- Consistent error handling patterns\n"
            "- Proper dependency injection usage"
        )

        data = await self._call_llm_json(prompt)
        return self._parse_review(data, ReviewLevel.MODULE)

    async def _review_architecture(self, context: AgentContext) -> ReviewResult:
        """Review architecture-level concerns."""
        formatted = self._format_context(context)
        prompt = (
            f"{formatted}\n\n"
            "Perform an ARCHITECTURE-LEVEL review checking:\n"
            "- No dependency cycles\n"
            "- Proper layer separation (controllers don't access repositories directly)\n"
            "- No layer violations\n"
            "- Consistent patterns across the codebase\n"
            "- Proper separation of concerns\n"
            "- Scalability concerns"
        )

        data = await self._call_llm_json(prompt)
        return self._parse_review(data, ReviewLevel.ARCHITECTURE)

    def _parse_review(self, data: dict[str, Any], level: ReviewLevel) -> ReviewResult:
        from core.llm_schema import validate_review_response

        # Validate and normalise — catches wrong key names, missing fields, bad types
        validated = validate_review_response(data)

        if not data:
            return ReviewResult(
                level=level,
                passed=True,
                findings=[],
                summary="Review skipped: LLM did not return a valid response.",
            )

        findings = [
            ReviewFinding(
                level=level,
                severity=f["severity"],
                file=f["file"],
                line=f["line"],
                message=f["message"],
                suggestion=f["suggestion"],
            )
            for f in validated["findings"]
        ]
        return ReviewResult(
            level=level,
            passed=validated["passed"],
            findings=findings,
            summary=validated["summary"],
        )
