"""Repository manager handling workspace files and structure."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from core.language import LanguageProfile, detect_language_from_blueprint, PYTHON
from core.import_validator import ImportValidator
from core.models import FileIndex, RepositoryBlueprint, RepositoryIndex

# Avoid circular imports: EmbeddingStore is imported lazily in write_file()
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from memory.embedding_store import EmbeddingStore

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

    def __init__(
        self,
        workspace_dir: Path,
        embedding_store: EmbeddingStore | None = None,
    ) -> None:
        self.workspace = workspace_dir
        self.deploy_dir = workspace_dir / "deploy"
        self.docs_dir = workspace_dir / "docs"
        self._repo_index = RepositoryIndex()
        self._lang_profile: LanguageProfile = PYTHON
        self._import_validator = ImportValidator()
        self._embedding_store = embedding_store

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

    @staticmethod
    def _write_atomic(file_path: Path, content: str) -> None:
        """Write *content* to *file_path* atomically via a temp-file + rename.

        Prevents half-written files from being seen by concurrent readers or
        by agents that re-scan the workspace after a crash / KeyboardInterrupt.
        On Windows, ``os.replace`` is used (atomic on NTFS for same-volume
        moves) instead of ``Path.rename`` which can raise if the destination
        already exists.
        """
        file_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=file_path.parent, prefix=".~", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(content)
            os.replace(tmp_path, file_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def write_file(self, rel_path: str, content: str) -> Path:
        """Write a file to the source root and update index (sync version)."""
        file_path = self._resolve_write_path(self.src_dir, rel_path)
        self._write_atomic(file_path, content)

        # Ensure __init__.py exists in all parent packages (Python only)
        self._ensure_init_files(file_path)

        # Update repo index
        self._index_file(rel_path, content)

        # Validate imports immediately after write — surface broken imports
        # before they reach test time.  Non-blocking: warnings only.
        known_files = {f.path for f in self._repo_index.files}
        broken = self._import_validator.validate(
            rel_path, content, known_files, self._lang_profile
        )
        if broken:
            logger.warning(
                "Broken imports detected in %s: %s", rel_path, broken
            )

        # Incremental embedding update — keeps vector index current as files
        # are written rather than deferring everything to finalization.
        if self._embedding_store is not None:
            try:
                self._embedding_store.index_file(rel_path, content)
            except Exception as e:
                logger.warning("Embedding index update failed for %s: %s", rel_path, e)

        logger.info(f"Wrote {rel_path} ({len(content)} bytes)")
        return file_path

    async def async_write_file(self, rel_path: str, content: str) -> Path:
        """Write a file without blocking the event loop.

        Delegates the actual I/O (write_text, mkdir, md5 hashing, __init__.py
        creation) to a thread so the event loop stays responsive while agents
        are generating files concurrently.
        """
        return await asyncio.to_thread(self.write_file, rel_path, content)

    def write_test_file(self, rel_path: str, content: str) -> Path:
        """Write a test file to the test root."""
        file_path = self._resolve_write_path(self.test_dir, rel_path)
        self._write_atomic(file_path, content)
        self._ensure_init_files(file_path)
        logger.info(f"Wrote test {rel_path}")
        return file_path

    async def async_write_test_file(self, rel_path: str, content: str) -> Path:
        """Write a test file without blocking the event loop."""
        return await asyncio.to_thread(self.write_test_file, rel_path, content)

    def write_deploy_file(self, rel_path: str, content: str) -> Path:
        """Write a deployment file."""
        file_path = self.deploy_dir / rel_path
        self._write_atomic(file_path, content)
        logger.info(f"Wrote deploy artifact {rel_path}")
        return file_path

    async def async_write_deploy_file(self, rel_path: str, content: str) -> Path:
        """Write a deployment file without blocking the event loop."""
        return await asyncio.to_thread(self.write_deploy_file, rel_path, content)

    def write_doc_file(self, rel_path: str, content: str) -> Path:
        """Write a documentation file."""
        file_path = self.docs_dir / rel_path
        self._write_atomic(file_path, content)
        logger.info(f"Wrote doc {rel_path}")
        return file_path

    async def async_write_doc_file(self, rel_path: str, content: str) -> Path:
        """Write a documentation file without blocking the event loop."""
        return await asyncio.to_thread(self.write_doc_file, rel_path, content)

    def read_file(self, rel_path: str) -> str | None:
        """Read a file from the source root."""
        file_path = self._resolve_write_path(self.src_dir, rel_path)
        if not file_path.exists():
            return None
        return file_path.read_text(encoding="utf-8")

    async def async_read_file(self, rel_path: str) -> str | None:
        """Read a file without blocking the event loop."""
        return await asyncio.to_thread(self.read_file, rel_path)

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
        """Parse and index a source file.

        Uses tree-sitter AST extraction for supported languages (currently
        Java), falling back to regex heuristics for others.  The AST path
        gives 100% accurate class/method/field extraction; the regex path
        handles ~90% of common patterns.
        """
        import re

        # ── Try AST-based extraction first (accurate, no regex guessing) ──
        from core.ast_extractor import ASTExtractor
        checksum = hashlib.md5(content.encode()).hexdigest()
        _extractor = ASTExtractor()
        sig = _extractor.extract(rel_path, content, self._lang_profile.name, checksum=checksum)

        if sig is not None:
            exports: list[str] = []
            imports: list[str] = [imp for imp in sig.imports]
            classes: list[str] = []
            functions: list[str] = []

            for t in sig.types:
                exports.append(t.name)
                if t.kind in ("class", "interface", "enum", "record"):
                    classes.append(t.name)
                for m in t.methods:
                    if not m.is_constructor:
                        functions.append(m.name)
                        exports.append(m.name)

            file_index = FileIndex(
                path=rel_path,
                exports=exports,
                imports=imports,
                classes=classes,
                functions=functions,
                checksum=checksum,
            )
            self._repo_index.add_or_update(file_index)
            return

        # ── Fallback: regex-based extraction for unsupported languages ────
        exports: list[str] = []
        imports: list[str] = []
        classes: list[str] = []
        functions: list[str] = []

        # ── Line-by-line pass (imports + single-line definitions) ────────
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

        # ── Multi-line pass (catch patterns spanning multiple lines) ─────
        # These catch decorated/annotated definitions, multi-line exports,
        # and other patterns the line-by-line pass misses.
        _MULTILINE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
            # Python: @decorator\n(async) def/class
            ("python", re.compile(
                r"@\w+[^\n]*\n\s*(?:async\s+)?(?:def|class)\s+(\w+)", re.MULTILINE
            )),
            # Java/C#: @Annotation\npublic class/interface/enum
            ("java", re.compile(
                r"@\w+[^\n]*\n\s*(?:public\s+)?(?:class|interface|enum|record)\s+(\w+)",
                re.MULTILINE,
            )),
            ("csharp", re.compile(
                r"\[\w+[^\]]*\]\s*\n\s*(?:public\s+)?(?:class|interface|struct|enum|record)\s+(\w+)",
                re.MULTILINE,
            )),
            # TypeScript/JS: multi-line export { ... }
            ("typescript", re.compile(
                r"export\s*\{([^}]+)\}", re.MULTILINE | re.DOTALL
            )),
            # Go: multi-line type block
            ("go", re.compile(
                r"type\s*\(\s*((?:\s*\w+\s+\w+[^\)]*\n?)+)\s*\)", re.MULTILINE
            )),
            # Rust: pub(crate) fn / pub struct across lines
            ("rust", re.compile(
                r"(?:#\[\w+[^\]]*\]\s*\n\s*)?pub(?:\([^)]*\))?\s+(?:fn|struct|enum|trait)\s+(\w+)",
                re.MULTILINE,
            )),
        ]

        lang = self._lang_profile.name.lower()
        seen_names = set(exports)  # avoid duplicates from line-by-line pass

        for pattern_lang, pattern in _MULTILINE_PATTERNS:
            if pattern_lang != lang:
                continue
            for m in pattern.finditer(content):
                captured = m.group(1)
                if pattern_lang == "typescript" and "{" not in captured:
                    # export { A, B, C } — extract individual names
                    for name in re.split(r"[,\n]", captured):
                        name = name.strip().split(" as ")[-1].strip()
                        if name and name not in seen_names:
                            exports.append(name)
                            seen_names.add(name)
                            if name[0].isupper():
                                classes.append(name)
                            else:
                                functions.append(name)
                elif pattern_lang == "go" and "\n" in captured:
                    # type ( \n Name Type \n ... ) — extract each name
                    for type_line in captured.strip().splitlines():
                        parts = type_line.strip().split()
                        if parts and parts[0].isidentifier() and parts[0] not in seen_names:
                            name = parts[0]
                            exports.append(name)
                            seen_names.add(name)
                            if name[0].isupper():
                                classes.append(name)
                            else:
                                functions.append(name)
                else:
                    name = captured.strip()
                    if name and name not in seen_names:
                        exports.append(name)
                        seen_names.add(name)
                        if name[0].isupper():
                            classes.append(name)
                        else:
                            functions.append(name)

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

    def rebuild_dependency_graph(self) -> dict[str, list[str]]:
        """Re-derive the dependency graph from the current repo index.

        Called after code generation phases to reflect actual imports rather
        than the static blueprint dependencies.  Writes updated
        ``dependency_graph.json`` to the workspace.
        """
        graph: dict[str, list[str]] = {}
        known_exports: dict[str, str] = {}  # export_name -> file_path

        # First pass: build export→file mapping
        for fi in self._repo_index.files:
            for exp in fi.exports:
                known_exports[exp] = fi.path

        # Second pass: resolve imports to file paths
        for fi in self._repo_index.files:
            deps: list[str] = []
            for imp_line in fi.imports:
                # Extract imported names from the import statement
                for exp_name, exp_file in known_exports.items():
                    if exp_name in imp_line and exp_file != fi.path:
                        if exp_file not in deps:
                            deps.append(exp_file)
            graph[fi.path] = deps

        dep_path = self.workspace / "dependency_graph.json"
        dep_path.write_text(json.dumps(graph, indent=2), encoding="utf-8")
        logger.info(f"Rebuilt dependency graph: {len(graph)} files, "
                    f"{sum(len(v) for v in graph.values())} edges")
        return graph

    def _build_dep_graph(self, blueprint: RepositoryBlueprint) -> dict[str, list[str]]:
        graph: dict[str, list[str]] = {}
        for fb in blueprint.file_blueprints:
            graph[fb.path] = fb.depends_on
        return graph

    # ── Existing-repo scanning ───────────────────────────────────────

    def scan_existing_repo(self) -> RepositoryIndex:
        """Scan an existing workspace and build the repo index from actual files.

        Unlike ``initialize()`` which creates a workspace from a blueprint,
        this reads whatever is already on disk — for modification workflows
        where the repo already exists.
        """
        logger.info("Scanning existing repository at %s", self.workspace)

        # Auto-detect language from file extensions in the workspace
        ext_counts: dict[str, int] = {}
        skip_dirs = {
            ".git", ".svn", "node_modules", "__pycache__", ".venv", "venv",
            "target", "build", "dist", ".idea", ".vscode", ".gradle",
            ".mypy_cache", ".pytest_cache",
        }
        for f in self.workspace.rglob("*"):
            if f.is_file() and not any(p in skip_dirs for p in f.parts):
                ext_counts[f.suffix.lower()] = ext_counts.get(f.suffix.lower(), 0) + 1

        from core.language import detect_language_from_extensions
        self._lang_profile = detect_language_from_extensions(ext_counts)
        logger.info("Detected language: %s", self._lang_profile.display_name)

        # Index all source files
        file_count = 0
        for ext in self._lang_profile.file_extensions:
            for src_file in self.workspace.rglob(f"*{ext}"):
                if not src_file.is_file():
                    continue
                if any(p in skip_dirs for p in src_file.parts):
                    continue
                try:
                    content = src_file.read_text(encoding="utf-8")
                    rel_path = str(src_file.relative_to(self.workspace)).replace("\\", "/")
                    self._index_file(rel_path, content)
                    file_count += 1
                except Exception as e:
                    logger.warning("Failed to index %s: %s", src_file, e)

        logger.info("Indexed %d files into repo index", file_count)
        return self._repo_index

    def read_all_source_files(self) -> dict[str, str]:
        """Read all source files in the workspace into a dict of path → content.

        Useful for the repository analyzer agent which needs file contents
        to produce module summaries.
        """
        skip_dirs = {
            ".git", "node_modules", "__pycache__", ".venv", "venv",
            "target", "build", "dist", ".idea", ".vscode",
        }
        result: dict[str, str] = {}
        for ext in self._lang_profile.file_extensions:
            for src_file in self.workspace.rglob(f"*{ext}"):
                if not src_file.is_file():
                    continue
                if any(p in skip_dirs for p in src_file.parts):
                    continue
                try:
                    content = src_file.read_text(encoding="utf-8")
                    rel_path = str(src_file.relative_to(self.workspace)).replace("\\", "/")
                    result[rel_path] = content
                except Exception:
                    pass
        return result
