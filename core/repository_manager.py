"""Repository manager handling workspace files and structure."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from core.language import LanguageProfile, detect_language_from_blueprint, PYTHON
from core.models import FileIndex, RepositoryBlueprint, RepositoryIndex

logger = logging.getLogger(__name__)


# Files that must always live at the workspace root, never under src_dir
_ROOT_LEVEL_FILES = {
    "pom.xml", "build.gradle", "build.gradle.kts",
    "settings.gradle", "settings.gradle.kts",
    "go.mod", "go.sum",
    "cargo.toml", "cargo.lock",
    "package.json", "package-lock.json", "tsconfig.json",
    "requirements.txt", "pyproject.toml", "setup.py", "setup.cfg",
    "makefile", "rakefile", "gemfile",
    "dockerfile", "docker-compose.yml", "docker-compose.yaml",
    ".gitignore", ".editorconfig",
}


class RepositoryManager:
    """Manages the generated repository workspace."""

    def __init__(self, workspace_dir: Path) -> None:
        self.workspace = workspace_dir
        self.deploy_dir = workspace_dir / "deploy"
        self.docs_dir = workspace_dir / "docs"
        self._repo_index = RepositoryIndex()
        self._lang_profile: LanguageProfile = PYTHON

    @property
    def src_dir(self) -> Path:
        """Source root — language-aware (e.g. workspace/src for Python, workspace for Java)."""
        sr = self._lang_profile.source_root
        return self.workspace / sr if sr else self.workspace

    @property
    def test_dir(self) -> Path:
        """Test root — language-aware (e.g. workspace/tests for Python, workspace for Java)."""
        tr = self._lang_profile.test_root
        return self.workspace / tr if tr else self.workspace

    def initialize(self, blueprint: RepositoryBlueprint) -> None:
        """Create workspace directory structure from blueprint."""
        self._lang_profile = detect_language_from_blueprint(blueprint.tech_stack)
        self.workspace.mkdir(parents=True, exist_ok=True)
        # Only create explicit src/test dirs when the language uses dedicated roots
        if self._lang_profile.source_root:
            self.src_dir.mkdir(parents=True, exist_ok=True)
        if self._lang_profile.test_root:
            self.test_dir.mkdir(parents=True, exist_ok=True)
        self.deploy_dir.mkdir(exist_ok=True)
        self.docs_dir.mkdir(exist_ok=True)

        # Create subdirectories from blueprint
        for folder in blueprint.folder_structure:
            self._resolve_write_path(self.src_dir, folder).mkdir(parents=True, exist_ok=True)

        # Write architecture doc
        arch_path = self.workspace / "architecture.md"
        arch_path.write_text(blueprint.architecture_doc, encoding="utf-8")

        # Write blueprint files
        bp_data = {
            "name": blueprint.name,
            "architecture_style": blueprint.architecture_style,
            "tech_stack": blueprint.tech_stack,
            "files": [
                {
                    "path": fb.path,
                    "purpose": fb.purpose,
                    "depends_on": fb.depends_on,
                    "exports": fb.exports,
                    "language": fb.language,
                    "layer": fb.layer,
                }
                for fb in blueprint.file_blueprints
            ],
        }
        bp_path = self.workspace / "file_blueprints.json"
        bp_path.write_text(json.dumps(bp_data, indent=2), encoding="utf-8")

        # Write dependency graph
        dep_graph = self._build_dep_graph(blueprint)
        dep_path = self.workspace / "dependency_graph.json"
        dep_path.write_text(json.dumps(dep_graph, indent=2), encoding="utf-8")

        logger.info(f"Initialized workspace at {self.workspace}")

    def _resolve_write_path(self, root: Path, rel_path: str) -> Path:
        """Resolve the full write path, avoiding double-prefix when rel_path already starts with root.

        Handles the case where blueprints may include the root prefix in file paths
        (e.g. 'src/main/java/...' when source_root is already 'src/main').
        Root-level files (pom.xml, go.mod, etc.) always go to workspace root.
        """
        # Root-level project files always live at workspace root
        filename = Path(rel_path).name.lower()
        if filename in _ROOT_LEVEL_FILES:
            return self.workspace / Path(rel_path).name  # preserve original case

        if root == self.workspace:
            return self.workspace / rel_path
        root_prefix = str(root.relative_to(self.workspace)).replace("\\", "/")
        if rel_path.startswith(root_prefix + "/") or rel_path == root_prefix:
            # Path already includes the root prefix — write relative to workspace
            return self.workspace / rel_path
        return root / rel_path

    def write_file(self, rel_path: str, content: str) -> Path:
        """Write a file to the source root and update index."""
        file_path = self._resolve_write_path(self.src_dir, rel_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")

        # Ensure __init__.py exists in all parent packages (Python only)
        self._ensure_init_files(file_path)

        # Update repo index
        self._index_file(rel_path, content)

        logger.info(f"Wrote {rel_path} ({len(content)} bytes)")
        return file_path

    def write_test_file(self, rel_path: str, content: str) -> Path:
        """Write a test file to the test root."""
        file_path = self._resolve_write_path(self.test_dir, rel_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        self._ensure_init_files(file_path)
        logger.info(f"Wrote test {rel_path}")
        return file_path

    def write_deploy_file(self, rel_path: str, content: str) -> Path:
        """Write a deployment file."""
        file_path = self.deploy_dir / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        logger.info(f"Wrote deploy artifact {rel_path}")
        return file_path

    def write_doc_file(self, rel_path: str, content: str) -> Path:
        """Write a documentation file."""
        file_path = self.docs_dir / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        logger.info(f"Wrote doc {rel_path}")
        return file_path

    def read_file(self, rel_path: str) -> str | None:
        """Read a file from the source root."""
        file_path = self._resolve_write_path(self.src_dir, rel_path)
        if not file_path.exists():
            return None
        return file_path.read_text(encoding="utf-8")

    def list_files(self, directory: str = "") -> list[str]:
        """List source files in a subdirectory of src."""
        target = self.src_dir / directory
        if not target.exists():
            return []
        results: list[str] = []
        for ext in self._lang_profile.file_extensions:
            for p in target.rglob(f"*{ext}"):
                if p.is_file():
                    results.append(str(p.relative_to(self.src_dir)).replace("\\", "/"))
        return results

    def search_code(self, keyword: str) -> list[dict[str, Any]]:
        """Search for a keyword across all source files."""
        results: list[dict[str, Any]] = []
        for ext in self._lang_profile.file_extensions:
            for src_file in self.src_dir.rglob(f"*{ext}"):
                try:
                    content = src_file.read_text(encoding="utf-8")
                    for i, line in enumerate(content.splitlines(), 1):
                        if keyword in line:
                            results.append({
                                "file": str(src_file.relative_to(self.src_dir)).replace("\\", "/"),
                                "line": i,
                                "content": line.strip(),
                            })
                except Exception:
                    continue
        return results

    def get_repo_index(self) -> RepositoryIndex:
        return self._repo_index

    def save_repo_index(self) -> None:
        """Persist the repo index to disk."""
        index_data = {
            "files": [
                {
                    "path": f.path,
                    "exports": f.exports,
                    "imports": f.imports,
                    "classes": f.classes,
                    "functions": f.functions,
                    "checksum": f.checksum,
                }
                for f in self._repo_index.files
            ]
        }
        index_path = self.workspace / "repo_index.json"
        index_path.write_text(json.dumps(index_data, indent=2), encoding="utf-8")

    def _index_file(self, rel_path: str, content: str) -> None:
        """Parse and index a source file using language-aware heuristics."""
        import re
        exports: list[str] = []
        imports: list[str] = []
        classes: list[str] = []
        functions: list[str] = []

        for line in content.splitlines():
            stripped = line.strip()
            # Detect imports using language profile patterns
            for pattern in self._lang_profile.import_patterns:
                if re.match(pattern, stripped):
                    imports.append(stripped)
                    break
            # Detect definitions using language profile patterns
            for pattern in self._lang_profile.definition_patterns:
                m = re.match(pattern, stripped)
                if m:
                    name = m.group(1)
                    # Heuristic: class-like names are uppercase-start
                    if name[0].isupper():
                        classes.append(name)
                    else:
                        functions.append(name)
                    exports.append(name)
                    break

        checksum = hashlib.md5(content.encode()).hexdigest()
        file_index = FileIndex(
            path=rel_path,
            exports=exports,
            imports=imports,
            classes=classes,
            functions=functions,
            checksum=checksum,
        )
        self._repo_index.add_or_update(file_index)

    def _ensure_init_files(self, file_path: Path) -> None:
        """Create package init files if required by the language (Python only)."""
        if not self._lang_profile.package_init_file:
            return
        current = file_path.parent
        roots = {self.src_dir, self.test_dir}
        while current not in roots and current != current.parent:
            init_file = current / self._lang_profile.package_init_file
            if not init_file.exists():
                init_file.write_text("", encoding="utf-8")
            current = current.parent

    def _build_dep_graph(self, blueprint: RepositoryBlueprint) -> dict[str, list[str]]:
        graph: dict[str, list[str]] = {}
        for fb in blueprint.file_blueprints:
            graph[fb.path] = fb.depends_on
        return graph
