"""Coder agent - generates code for a specific file following the blueprint."""

from __future__ import annotations

import logging
from typing import Any

from agents.base_agent import BaseAgent
from core.models import AgentContext, AgentRole, TaskResult

logger = logging.getLogger(__name__)


class CoderAgent(BaseAgent):
    role = AgentRole.CODER

    @property
    def system_prompt(self) -> str:
        return (
            "You are an expert Python developer agent. You generate production-quality code.\n\n"
            "Rules:\n"
            "- Generate ONLY the Python file content, no markdown fences or explanations\n"
            "- Follow the file blueprint exactly (purpose, exports, dependencies)\n"
            "- Import from the correct modules as specified in the dependency graph\n"
            "- Use type hints throughout\n"
            "- Use async/await where appropriate (FastAPI endpoints, DB operations)\n"
            "- Include proper error handling\n"
            "- Follow PEP 8 and clean code principles\n"
            "- Use Pydantic for data validation\n"
            "- Use dependency injection patterns\n"
            "- Do NOT add placeholder or TODO comments - write complete implementations\n"
            "- Do NOT import modules that aren't in the dependency graph"
        )

    async def execute(self, context: AgentContext) -> TaskResult:
        """Generate code for the assigned file."""
        if not context.file_blueprint:
            return TaskResult(
                success=False,
                errors=[f"No blueprint found for {context.task.file}"],
            )

        fb = context.file_blueprint
        logger.info(f"Generating code for {fb.path}")

        formatted_context = self._format_context(context)

        prompt = (
            f"{formatted_context}\n\n"
            f"Generate the complete Python file for: {fb.path}\n"
            f"Purpose: {fb.purpose}\n"
            f"This file must export: {', '.join(fb.exports) if fb.exports else 'appropriate classes/functions'}\n"
            f"Layer: {fb.layer}\n\n"
            "Generate the complete, working Python code. Output only the code, nothing else."
        )

        code = await self._call_llm(prompt)
        code = self._clean_code(code)

        # Write the file
        self.repo.write_file(fb.path, code)

        return TaskResult(
            success=True,
            output=f"Generated {fb.path} ({len(code)} bytes)",
            files_modified=[fb.path],
            metrics=self.get_metrics(),
        )

    def _clean_code(self, code: str) -> str:
        """Strip markdown fences and leading/trailing whitespace."""
        code = code.strip()
        if code.startswith("```python"):
            code = code[len("```python"):]
        elif code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
        return code.strip() + "\n"
