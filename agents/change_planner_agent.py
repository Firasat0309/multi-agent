"""Change Planner agent - plans targeted modifications to existing repositories."""

from __future__ import annotations

import json
import logging
from typing import Any, TYPE_CHECKING

from agents.base_agent import BaseAgent
from core.ast_extractor import ASTExtractor
from core.language import detect_language_from_blueprint
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

if TYPE_CHECKING:
    from memory.embedding_store import EmbeddingStore

logger = logging.getLogger(__name__)

# Maximum modules included in the repo context sent to the LLM.
# Modules are ranked by keyword overlap with the user request so the most
# relevant ones fill the budget rather than an arbitrary positional slice.
_MAX_CONTEXT_MODULES = 60

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

# Shared extractor — stateless aside from the parse cache.
_ast_extractor = ASTExtractor()


class ChangePlannerAgent(BaseAgent):
    """Plans targeted modifications to an existing codebase.

    Given a user request and a repository analysis, this agent produces a
    structured ChangePlan that lists exactly which files to modify, what
    functions/methods to add or change, and in what order.
    """

    role = AgentRole.CHANGE_PLANNER

    def __init__(
        self,
        llm_client: Any,
        repo_manager: Any,
        *,
        embedding_store: EmbeddingStore | None = None,
    ) -> None:
        super().__init__(llm_client=llm_client, repo_manager=repo_manager)
        self._embedding_store = embedding_store

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
            "- Respond with valid JSON only. No markdown fences or explanations.\n\n"

            "ORDERING RULE:\n"
            "- The 'changes' array MUST be in execution order: if change B depends on "
            "change A, A must appear first in the array\n"
            "- depends_on in each change lists the FILE PATHS that must be modified before "
            "this change can be applied\n\n"

            "COMPLETENESS RULE:\n"
            "- If you add a new method to a service, you MUST also include a change for "
            "the controller that calls it and the test that covers it\n"
            "- If you add a new class, you MUST include import changes in files that need it\n"
            "- Every change must reference an actual file that exists in the repository analysis"
        )

    async def execute(self, context: AgentContext) -> TaskResult:
        """Plan changes from task metadata (keys: 'user_request', 'repo_analysis')."""
        user_request: str = context.task.metadata.get("user_request", context.task.description)
        repo_analysis: RepoAnalysis | None = context.task.metadata.get("repo_analysis")
        if repo_analysis is None:
            return TaskResult(
                success=False,
                errors=["'repo_analysis' missing from task metadata — run RepositoryAnalyzerAgent first"],
            )
        try:
            plan = await self.plan_changes(user_request, repo_analysis)
            return TaskResult(
                success=True,
                output=f"Change plan created: {len(plan.changes)} change(s), {len(plan.new_files)} new file(s)",
                metrics=self.get_metrics(),
            )
        except Exception as exc:
            logger.exception("ChangePlannerAgent.execute failed")
            return TaskResult(success=False, errors=[str(exc)])

    async def plan_changes(
        self,
        user_request: str,
        repo_analysis: RepoAnalysis,
    ) -> ChangePlan:
        """Produce a structured change plan for a modification request.

        Args:
            user_request: The user's natural-language description of the desired change.
            repo_analysis: Output of RepositoryAnalyzerAgent.analyze_repository().

        File contents are fetched lazily — only the top-ranked modules (up to
        ``_MAX_CONTEXT_MODULES``) are read from disk, so this method is safe
        to call on repos of any size without OOM risk.
        """
        logger.info("Planning changes for request: %s", user_request[:100])

        repo_context = self._build_repo_context(repo_analysis, user_request)

        prompt = (
            f"The user wants to modify an existing codebase.\n\n"
            f"User request:\n{user_request}\n\n"
            f"Repository analysis:\n{repo_context}\n\n"
            f"Plan the exact changes needed. Be precise about which files to modify "
            f"and what to add/change in each file."
        )

        # Use _call_llm so the max_tokens continuation loop fires if the JSON
        # is truncated — direct llm.generate() bypasses that safety net.
        raw = await self._call_llm(prompt)

        data = self._parse_json(raw)
        if not data:
            logger.error(
                "ChangePlannerAgent produced empty plan (JSON parse failed). "
                "Raw response (first 500 chars): %s", raw[:500],
            )
        return self._parse_change_plan(data)

    def _build_repo_context(
        self,
        analysis: RepoAnalysis,
        user_request: str = "",
    ) -> str:
        """Build a compact text representation of the repository for LLM context.

        Modules are ranked by keyword overlap with *user_request* so the
        capped list contains the most relevant modules rather than an
        arbitrary positional slice of the full module list.

        File contents are fetched lazily — only the top ``_MAX_CONTEXT_MODULES``
        files are read from disk via ``self.repo.read_source_files()``, so
        this method never loads more than ~60 files into memory regardless of
        how large the repository is.

        File contents are rendered as AST stubs (signatures only) when the
        language is supported by tree-sitter, falling back to 4 000-char
        truncation otherwise.  Stubs are 5-20× smaller than raw source and
        provide richer structural information per token.
        """
        lines: list[str] = []

        lines.append(f"Tech stack: {json.dumps(analysis.tech_stack)}")
        lines.append(f"Architecture: {analysis.architecture_style}")
        if analysis.summary:
            lines.append(f"Summary: {analysis.summary}")
        lines.append(f"Entry points: {analysis.entry_points}")
        lines.append("")

        # Rank modules by a hybrid of keyword overlap and semantic similarity.
        # Embedding search surfaces conceptually related code even when the
        # user's wording doesn't match the symbol names.
        request_tokens = set(user_request.lower().split())

        # Semantic boost: query the embedding store for files relevant to the request.
        embedding_boost: dict[str, float] = {}
        if self._embedding_store and user_request:
            try:
                hits = self._embedding_store.search(user_request, n_results=20)
                if hits:
                    max_dist = max(h["distance"] for h in hits) or 1.0
                    for hit in hits:
                        fp = hit.get("file", "")
                        # Normalise distance to 0-1 score (lower distance = higher score)
                        score = 1.0 - (hit["distance"] / max_dist) if max_dist else 0.0
                        # Keep the best score per file across chunks
                        embedding_boost[fp] = max(embedding_boost.get(fp, 0.0), score)
            except Exception:
                logger.debug("Embedding search failed, falling back to keyword-only ranking")

        def _module_score(m: ModuleInfo) -> float:
            # Keyword overlap score (integer count)
            symbols = " ".join([m.file] + m.classes + m.functions).lower()
            keyword_score = sum(1 for tok in request_tokens if tok in symbols)
            # Embedding similarity bonus (0-5 range to be comparable with keyword hits)
            semantic_score = embedding_boost.get(m.file, 0.0) * 5
            return keyword_score + semantic_score

        ranked_modules = sorted(analysis.modules, key=_module_score, reverse=True)
        shown = ranked_modules[:_MAX_CONTEXT_MODULES]
        omitted = len(analysis.modules) - len(shown)

        lines.append(
            f"Modules ({len(analysis.modules)} total"
            + (f", showing top {len(shown)} by relevance):" if omitted else "):")
        )
        for m in shown:
            funcs = ", ".join(m.functions[:10])
            classes = ", ".join(m.classes[:5])
            line = f"  - {m.file} [{m.layer}]"
            if classes:
                line += f" classes=[{classes}]"
            if funcs:
                line += f" funcs=[{funcs}]"
            lines.append(line)
        if omitted:
            lines.append(f"  ... and {omitted} more modules omitted")

        # Fetch only the ranked files — bounded by _MAX_CONTEXT_MODULES.
        # self.repo.read_source_files() reads exactly the paths listed so
        # no more than len(shown) files are ever held in memory at once.
        top_paths = [m.file for m in shown]
        file_contents = self.repo.read_source_files(top_paths)

        if file_contents:
            lines.append("")
            lines.append("Key file contents (AST stubs where available):")
            lang_profile = detect_language_from_blueprint(analysis.tech_stack)
            total_chars = 0
            for path in top_paths:  # preserve relevance order
                content = file_contents.get(path)
                if content is None:
                    continue
                if total_chars > 30_000:
                    lines.append("  ... remaining files omitted (token budget reached)")
                    break
                stub = _ast_extractor.extract_stub(path, content, lang_profile.name)
                if stub is not None:
                    body = f"// AST stub (signatures only)\n{stub}"
                else:
                    body = content[:4_000]
                    if len(content) > 4_000:
                        body += "\n# ... (truncated)"
                lines.append(f"\n--- {path} ---")
                lines.append(body)
                total_chars += len(body)

        return "\n".join(lines)

    def _parse_change_plan(self, data: dict[str, Any]) -> ChangePlan:
        """Parse raw JSON into a validated ChangePlan."""
        changes: list[ChangeAction] = []
        for c in data.get("changes", []):
            action_type_str = c.get("type", "add_function")
            action_type = _ACTION_TYPE_MAP.get(action_type_str)
            if action_type is None:
                logger.warning(
                    "ChangePlannerAgent: unknown action type %r, defaulting to ADD_FUNCTION",
                    action_type_str,
                )
                action_type = ChangeActionType.ADD_FUNCTION
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
        """Parse JSON from LLM output, handling markdown fences and truncation.

        The fallback bracket-scan walks inward from the outermost ``{`` and
        ``}`` so a truncated response that ends mid-array or mid-string is
        rejected cleanly rather than parsed as a silently incomplete plan.
        """
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
            pass

        # Walk inward: find outermost { … } pair where both ends parse cleanly.
        # rfind("}") alone can match an interior "}" in truncated output,
        # producing an incomplete but parse-successful fragment.
        start = text.find("{")
        if start == -1:
            logger.error("No JSON object found in change plan response: %s...", text[:200])
            return {}

        end = len(text)
        while end > start:
            end = text.rfind("}", start, end)
            if end == -1:
                break
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass  # try one character shorter

        logger.error("Could not parse change plan JSON: %s...", text[:200])
        return {}
