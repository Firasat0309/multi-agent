"""Context builder that assembles minimal relevant context for each agent task."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

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
        """Read only the files this task depends on."""
        related: dict[str, str] = {}
        paths_to_read: set[str] = set()

        if file_bp:
            paths_to_read.update(file_bp.depends_on)

        # For review tasks, read the target file
        if task.task_type in (TaskType.REVIEW_FILE, TaskType.GENERATE_TEST):
            paths_to_read.add(task.file)

        # For module/architecture review, read all files
        if task.task_type in (TaskType.REVIEW_MODULE, TaskType.REVIEW_ARCHITECTURE):
            for fb in self.blueprint.file_blueprints:
                paths_to_read.add(fb.path)

        for p in paths_to_read:
            content = self._read_file(p)
            if content is not None:
                related[p] = content

        return related

    def _read_file(self, rel_path: str) -> str | None:
        src_path = self.workspace / "src" / rel_path
        if src_path.exists():
            try:
                return src_path.read_text(encoding="utf-8")
            except Exception:
                logger.warning(f"Failed to read {src_path}")
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
