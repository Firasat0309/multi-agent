"""Change Planner agent - plans targeted modifications to existing repositories."""

from __future__ import annotations

import json
import logging
from typing import Any

from agents.base_agent import BaseAgent
from core.models import (
    AgentContext,
    AgentRole,
    ChangeAction,
    ChangeActionType,
    ChangePlan,
    FileBlueprint,
    ModuleInfo,
    RepoAnalysis,
    TaskResult,
)

logger = logging.getLogger(__name__)

# Valid action type strings the LLM may return → enum mapping
_ACTION_TYPE_MAP: dict[str, ChangeActionType] = {
    "add_function": ChangeActionType.ADD_FUNCTION,
    "add_method": ChangeActionType.ADD_METHOD,
    "add_endpoint": ChangeActionType.ADD_ENDPOINT,
    "add_class": ChangeActionType.ADD_CLASS,
    "add_import": ChangeActionType.ADD_IMPORT,
    "modify_function": ChangeActionType.MODIFY_FUNCTION,
    "add_field": ChangeActionType.ADD_FIELD,
    "create_file": ChangeActionType.CREATE_FILE,
}


class ChangePlannerAgent(BaseAgent):
    """Plans targeted modifications to an existing codebase.

    Given a user request and a repository analysis, this agent produces a
    structured ChangePlan that lists exactly which files to modify, what
    functions/methods to add or change, and in what order.
    """

    role = AgentRole.CHANGE_PLANNER

    @property
    def system_prompt(self) -> str:
        return (
            "You are a change planning agent for an existing codebase. Your job is to "
            "analyze a modification request and produce a precise plan of targeted changes.\n\n"
            "You MUST produce a JSON response with this exact structure:\n"
            "{\n"
            '  "summary": "Brief description of the overall change",\n'
            '  "changes": [\n'
            "    {\n"
            '      "type": "add_function|add_method|add_endpoint|add_class|add_import|modify_function|add_field|create_file",\n'
            '      "file": "path/to/file.py",\n'
            '      "description": "What this change does",\n'
            '      "function": "function_name",\n'
            '      "class_name": "ClassName (if adding method to class)",\n'
            '      "depends_on": ["path/to/other_file.py"]\n'
            "    }\n"
            "  ],\n"
            '  "new_files": [\n'
            "    {\n"
            '      "path": "path/to/new_file.py",\n'
            '      "purpose": "Why this file is needed",\n'
            '      "depends_on": ["path/to/existing.py"],\n'
            '      "exports": ["ClassName", "function_name"],\n'
            '      "layer": "service|controller|model|repository|util|test"\n'
            "    }\n"
            "  ],\n"
            '  "affected_tests": ["path/to/test_file.py"],\n'
            '  "risk_notes": ["Any potential risks or considerations"]\n'
            "}\n\n"
            "Rules:\n"
            "- Each change should be the SMALLEST atomic modification needed\n"
            "- Order changes so dependencies come first\n"
            "- Prefer modifying existing files over creating new ones\n"
            "- Always list test files that need updating\n"
            "- Include import changes if new dependencies are introduced\n"
            "- Be specific: name the exact function/class/method to add or modify\n"
            "- Only create_file when the change truly needs a new module\n"
            "- Respond with valid JSON only. No markdown fences or explanations."
        )

    async def execute(self, context: AgentContext) -> TaskResult:
        """Not used directly — use plan_changes() instead."""
        return TaskResult(success=False, errors=["Use plan_changes() method"])

    async def plan_changes(
        self,
        user_request: str,
        repo_analysis: RepoAnalysis,
        file_contents: dict[str, str] | None = None,
    ) -> ChangePlan:
        """Produce a structured change plan for a modification request.

        Args:
            user_request: The user's natural-language description of the desired change.
            repo_analysis: Output of RepositoryAnalyzerAgent.analyze_repository().
            file_contents: Optional dict of file_path → content for targeted files.
        """
        logger.info("Planning changes for request: %s", user_request[:100])

        # Build context about the repository for the LLM
        repo_context = self._build_repo_context(repo_analysis, file_contents)

        prompt = (
            f"The user wants to modify an existing codebase.\n\n"
            f"User request:\n{user_request}\n\n"
            f"Repository analysis:\n{repo_context}\n\n"
            f"Plan the exact changes needed. Be precise about which files to modify "
            f"and what to add/change in each file."
        )

        response = await self.llm.generate(
            system_prompt=self.system_prompt,
            user_prompt=prompt,
            max_tokens=8192,
        )
        self._metrics["llm_calls"] += 1
        self._metrics["tokens_used"] += sum(response.usage.values())

        data = self._parse_json(response.content)
        return self._parse_change_plan(data)

    def _build_repo_context(
        self,
        analysis: RepoAnalysis,
        file_contents: dict[str, str] | None,
    ) -> str:
        """Build a compact text representation of the repository for LLM context."""
        lines: list[str] = []

        lines.append(f"Tech stack: {json.dumps(analysis.tech_stack)}")
        lines.append(f"Architecture: {analysis.architecture_style}")
        if analysis.summary:
            lines.append(f"Summary: {analysis.summary}")
        lines.append(f"Entry points: {analysis.entry_points}")
        lines.append("")

        lines.append(f"Modules ({len(analysis.modules)} total):")
        for m in analysis.modules:
            funcs = ", ".join(m.functions[:10])
            classes = ", ".join(m.classes[:5])
            line = f"  - {m.file} [{m.layer}]"
            if classes:
                line += f" classes=[{classes}]"
            if funcs:
                line += f" funcs=[{funcs}]"
            lines.append(line)

        # If specific file contents provided, include them for precision
        if file_contents:
            lines.append("")
            lines.append("Key file contents:")
            total_chars = 0
            for path, content in file_contents.items():
                if total_chars > 30_000:
                    break
                truncated = content[:4000]
                if len(content) > 4000:
                    truncated += "\n# ... (truncated)"
                lines.append(f"\n--- {path} ---")
                lines.append(truncated)
                total_chars += len(truncated)

        return "\n".join(lines)

    def _parse_change_plan(self, data: dict[str, Any]) -> ChangePlan:
        """Parse raw JSON into a validated ChangePlan."""
        changes: list[ChangeAction] = []
        for c in data.get("changes", []):
            action_type_str = c.get("type", "add_function")
            action_type = _ACTION_TYPE_MAP.get(action_type_str, ChangeActionType.ADD_FUNCTION)
            changes.append(ChangeAction(
                type=action_type,
                file=c.get("file", ""),
                description=c.get("description", ""),
                function=c.get("function", ""),
                class_name=c.get("class_name", ""),
                depends_on=c.get("depends_on", []),
                details=c.get("details", {}),
            ))

        new_files: list[FileBlueprint] = []
        for nf in data.get("new_files", []):
            new_files.append(FileBlueprint(
                path=nf.get("path", ""),
                purpose=nf.get("purpose", ""),
                depends_on=nf.get("depends_on", []),
                exports=nf.get("exports", []),
                layer=nf.get("layer", ""),
            ))

        return ChangePlan(
            summary=data.get("summary", ""),
            changes=changes,
            new_files=new_files,
            affected_tests=data.get("affected_tests", []),
            risk_notes=data.get("risk_notes", []),
        )

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        """Parse JSON from LLM output, handling fences."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
            logger.error("Could not parse change plan JSON: %s...", text[:200])
            return {}
