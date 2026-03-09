"""Context builder that assembles minimal relevant context for each agent task."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from core.ast_extractor import ASTExtractor
from core.language import detect_language_from_blueprint
from core.models import (
    AgentContext,
    FileBlueprint,
    RepositoryBlueprint,
    RepositoryIndex,
    Task,
    TaskType,
)

logger = logging.getLogger(__name__)

# Hard cap on files included in wide-scope reviews to avoid context overflow.
_MAX_REVIEW_FILES = 20
# Hard cap on total characters across all related files sent to the LLM.
_MAX_CONTEXT_CHARS = 120_000

# Shared extractor instance (stateless aside from cache)
_ast_extractor = ASTExtractor()


class ContextBuilder:
    """Builds focused context for agents, avoiding full-repo dumps."""

    def __init__(
        self,
        workspace_dir: Path,
        blueprint: RepositoryBlueprint,
        repo_index: RepositoryIndex,
    ) -> None:
        self.workspace = workspace_dir
        self.blueprint = blueprint
        self.repo_index = repo_index

    def build(self, task: Task) -> AgentContext:
        """Build context for a specific task."""
        file_bp = self._find_blueprint(task.file)
        related = self._collect_related_files(task, file_bp)
        dep_info = self._build_dependency_info(task.file)

        return AgentContext(
            task=task,
            blueprint=self.blueprint,
            file_blueprint=file_bp,
            related_files=related,
            architecture_summary=self.blueprint.architecture_doc,
            dependency_info=dep_info,
        )

    def _find_blueprint(self, file_path: str) -> FileBlueprint | None:
        for fb in self.blueprint.file_blueprints:
            if fb.path == file_path:
                return fb
        return None

    def _collect_related_files(
        self, task: Task, file_bp: FileBlueprint | None
    ) -> dict[str, str]:
        """Read only the files this task depends on, respecting token budget.

        For dependency files (not the task's own target), we use AST-extracted
        stubs when the language is supported by tree-sitter.  This gives the
        LLM the public API surface (class/method/field signatures) without
        implementation bodies — typically a 5-10x context reduction.

        The target file itself is always included in full so the agent has
        complete source when reviewing, testing, or fixing it.
        """
        related: dict[str, str] = {}
        paths_to_read: set[str] = set()

        # Files that should be included as AST stubs (dependencies)
        dependency_paths: set[str] = set()
        # Files that MUST be included in full (target file being worked on)
        full_source_paths: set[str] = set()

        if file_bp:
            dependency_paths.update(file_bp.depends_on)

        # For review/fix/test tasks, read the target file in full
        if task.task_type in (TaskType.REVIEW_FILE, TaskType.GENERATE_TEST, TaskType.FIX_CODE):
            full_source_paths.add(task.file)

        # For module/architecture review, read files up to the cap
        if task.task_type in (TaskType.REVIEW_MODULE, TaskType.REVIEW_ARCHITECTURE):
            for fb in self.blueprint.file_blueprints[:_MAX_REVIEW_FILES]:
                full_source_paths.add(fb.path)

        # Detect language for AST extraction
        lang = detect_language_from_blueprint(self.blueprint.tech_stack)

        # Process full-source files first (highest priority)
        total_chars = 0
        for p in full_source_paths:
            if total_chars >= _MAX_CONTEXT_CHARS:
                break
            content = self._read_file(p)
            if content is not None:
                if len(content) > 8_000:
                    content = content[:8_000] + "\n# ... (truncated)"
                related[p] = content
                total_chars += len(content)

        # Process dependency files — use AST stubs where possible
        for p in dependency_paths:
            if p in related:
                continue  # already included as full source
            if total_chars >= _MAX_CONTEXT_CHARS:
                logger.debug(
                    "Context budget reached (%d chars), skipping remaining files", total_chars
                )
                break
            content = self._read_file(p)
            if content is None:
                continue

            # Try AST stub extraction — much more compact than raw source
            file_index = self.repo_index.get_file(p)
            checksum = file_index.checksum if file_index else ""
            stub = _ast_extractor.extract_stub(
                p, content, lang.name, checksum=checksum,
            )

            if stub is not None:
                # Stub is typically 5-10x smaller than full source
                related[p] = f"// AST stub (signatures only)\n{stub}"
                total_chars += len(stub)
                logger.debug(
                    "Using AST stub for %s (%d→%d chars, %.0f%% reduction)",
                    p, len(content), len(stub),
                    (1 - len(stub) / len(content)) * 100 if content else 0,
                )
            else:
                # Fallback: raw truncation for unsupported languages
                if len(content) > 8_000:
                    content = content[:8_000] + "\n# ... (truncated)"
                related[p] = content
                total_chars += len(content)

        return related

    def _read_file(self, rel_path: str) -> str | None:
        # Search across all language source roots — order matters: most specific first
        candidates = [
            self.workspace / rel_path,              # absolute path (Java: src/main/java/...)
            self.workspace / "src" / "main" / rel_path,  # Java short path (java/com/...)
            self.workspace / "src" / rel_path,      # Python / TypeScript
        ]
        for path in candidates:
            if path.exists():
                try:
                    return path.read_text(encoding="utf-8")
                except Exception:
                    logger.warning(f"Failed to read {path}")
        return None

    def _build_dependency_info(self, file_path: str) -> dict[str, Any]:
        """Get dependency information from the repo index."""
        lang = detect_language_from_blueprint(self.blueprint.tech_stack)
        info: dict[str, Any] = {"upstream": [], "downstream": []}
        target = self.repo_index.get_file(file_path)
        if not target:
            return info

        for f in self.repo_index.files:
            if f.path == file_path:
                continue
            # Check if this file imports from our target
            for imp in f.imports:
                module = lang.to_module_path(file_path)
                if module in imp or file_path in imp:
                    info["downstream"].append(f.path)
            # Check if our target imports from this file
            for imp in target.imports:
                module = lang.to_module_path(f.path)
                if module in imp or f.path in imp:
                    info["upstream"].append(f.path)

        return info
