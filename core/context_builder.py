"""Context builder that assembles minimal relevant context for each agent task."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.ast_extractor import ASTExtractor
from core.language import detect_language_from_blueprint
from core.models import (
    AgentContext,
    APIContract,
    FileBlueprint,
    RepositoryBlueprint,
    RepositoryIndex,
    Task,
    TaskType,
    ChangePlan,
)

if TYPE_CHECKING:
    from memory.dependency_graph import DependencyGraphStore
    from memory.embedding_store import EmbeddingStore

logger = logging.getLogger(__name__)

# Hard cap on files included in wide-scope reviews to avoid context overflow.
_MAX_REVIEW_FILES = 20
# Hard cap on total characters across all related files sent to the LLM.
_MAX_CONTEXT_CHARS = 120_000
# Max semantic hits to include from vector search per task.
_MAX_SEMANTIC_HITS = 3

# Shared extractor instance (stateless aside from cache)
_ast_extractor = ASTExtractor()


def _smart_truncate(content: str, max_chars: int) -> str:
    """Truncate source code preserving imports and tail signatures.

    When AST extraction fails, this is far better than blind ``[:max_chars]``
    which cuts at an arbitrary point and loses critical declarations that
    appear late in the file (e.g., public methods at line 300+).

    Strategy:
      1. Always keep imports/package/use lines (top of file)
      2. Keep the first section of code (class header, early fields)
      3. Keep the LAST section (public methods the dependent needs)
      4. Insert a ``... (N lines omitted)`` marker in the middle

    This ensures the LLM sees the API surface at both ends of the file.
    """
    if len(content) <= max_chars:
        return content

    lines = content.split("\n")

    # Phase 1: collect import/package lines (always at top)
    import_end = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("//") or stripped.startswith("#") or stripped.startswith("/*"):
            continue
        if any(stripped.startswith(kw) for kw in (
            "import ", "from ", "package ", "namespace ", "use ", "using ",
            "require(", "const ", "let ", "var ",  # JS/TS requires
        )):
            import_end = i + 1
        elif import_end > 0:
            break  # past the import block

    # Phase 2: split budget — 60% for head (imports + early code), 40% for tail
    head_budget = int(max_chars * 0.6)
    tail_budget = max_chars - head_budget

    # Build head: always include ALL import lines, then fill remaining budget
    # with early code.  This guarantees the LLM sees every import even when
    # the import block is large (common in Java files with 30+ imports).
    head_lines: list[str] = []
    head_chars = 0
    for i, line in enumerate(lines):
        if i < import_end:
            # Import lines are mandatory — include even if over budget
            head_lines.append(line)
            head_chars += len(line) + 1
        elif head_chars + len(line) + 1 > head_budget:
            break
        else:
            head_lines.append(line)
            head_chars += len(line) + 1

    # Recalculate tail budget: subtract what the head actually consumed.
    # When mandatory imports exceed the 60% head budget, the tail gets
    # whatever is left so we still show some API surface from the file end.
    remaining_budget = max(0, max_chars - head_chars)
    effective_tail_budget = min(tail_budget, remaining_budget)

    # Build tail: work backwards from end of file
    tail_lines: list[str] = []
    tail_chars = 0
    if effective_tail_budget > 0:
        for line in reversed(lines):
            if tail_chars + len(line) + 1 > effective_tail_budget:
                break
            tail_lines.append(line)
            tail_chars += len(line) + 1
        tail_lines.reverse()

    # Check for overlap (small files might have head and tail overlap)
    head_end = len(head_lines)
    tail_start = len(lines) - len(tail_lines) if tail_lines else len(lines)
    if head_end >= tail_start:
        # Overlap — just return head (imports are most critical)
        result = "\n".join(head_lines)
        if len(result) > max_chars:
            return result[:max_chars]
        return result

    omitted = tail_start - head_end
    marker = f"\n// ... ({omitted} lines omitted) ...\n"

    return "\n".join(head_lines) + marker + "\n".join(tail_lines)


# ── Context priority model ───────────────────────────────────────────────────

@dataclass
class ContextFile:
    """A candidate file for inclusion in an agent's context window."""
    path: str
    content: str
    priority: int           # 1=target file, 2=direct dep, 3=semantic hit, 4=module review
    relevance_score: float  # 0.0–1.0 (higher is more relevant); from semantic distance
    is_stub: bool           # True when content is an AST stub rather than full source


class ContextBuilder:
    """Builds focused context for agents, avoiding full-repo dumps."""

    def __init__(
        self,
        workspace_dir: Path,
        blueprint: RepositoryBlueprint,
        repo_index: RepositoryIndex,
        dep_store: DependencyGraphStore | None = None,
        embedding_store: EmbeddingStore | None = None,
        api_contract: APIContract | None = None,
    ) -> None:
        self.workspace = workspace_dir
        self.blueprint = blueprint
        self.repo_index = repo_index
        self._dep_store = dep_store
        self._embedding_store = embedding_store
        self._api_contract = api_contract

    def build(self, task: Task) -> AgentContext:
        """Build context for a specific task using priority-ordered file collection."""
        file_bp = self._find_blueprint(task.file)
        related = self._collect_ranked(task, file_bp)
        dep_info = self._build_dependency_info(task.file)

        return AgentContext(
            task=task,
            blueprint=self.blueprint,
            file_blueprint=file_bp,
            related_files=related,
            architecture_summary=self.blueprint.architecture_doc,
            dependency_info=dep_info,
            api_contract=self._api_contract,
        )

    def _find_blueprint(self, file_path: str) -> FileBlueprint | None:
        for fb in self.blueprint.file_blueprints:
            if fb.path == file_path:
                return fb
        return None

    def _collect_ranked(
        self, task: Task, file_bp: FileBlueprint | None
    ) -> dict[str, str]:
        """Collect context files in relevance order, stopping at budget.

        Priority ordering:
          1. Target file (full source, always included first)
          2. Direct imports of the target (AST stubs)
          3. Semantic search hits (AST stubs, ordered by distance)
          4. Module review files (AST stubs, ordered by dep layer)

        Within the same priority tier, files are sorted by descending
        ``relevance_score`` so the most pertinent content is always included
        before the budget runs out.
        """
        candidates: list[ContextFile] = []
        lang = detect_language_from_blueprint(self.blueprint.tech_stack)

        # ── Priority 1: target file in full ──────────────────────────────────
        target_full_paths: set[str] = set()
        if task.task_type in (
            TaskType.REVIEW_FILE, TaskType.GENERATE_TEST, TaskType.FIX_CODE
        ):
            target_full_paths.add(task.file)

        for p in target_full_paths:
            content = self._read_file(p)
            if content is not None:
                candidates.append(ContextFile(
                    path=p,
                    content=content,
                    priority=1,
                    relevance_score=1.0,
                    is_stub=False,
                ))

        # ── Priority 2: direct blueprint dependencies (AST stubs) ────────────
        # AST stubs include full method signatures (name, params, return type)
        # which is exactly what the coder agent needs to match cross-file APIs.
        if file_bp:
            for dep_path in file_bp.depends_on:
                content = self._read_file(dep_path)
                if content is None:
                    continue
                file_index = self.repo_index.get_file(dep_path)
                checksum = file_index.checksum if file_index else ""
                stub = _ast_extractor.extract_stub(dep_path, content, lang.name, checksum)
                if stub is not None:
                    body = f"// AST stub — use these EXACT method names and signatures\n{stub}"
                    logger.debug(
                        "AST stub for %s (%d→%d chars, %.0f%% reduction)",
                        dep_path, len(content), len(stub),
                        (1 - len(stub) / len(content)) * 100 if content else 0,
                    )
                    candidates.append(ContextFile(
                        path=dep_path, content=body,
                        priority=2, relevance_score=0.9, is_stub=True,
                    ))
                else:
                    trunc = _smart_truncate(content, 8_000)
                    candidates.append(ContextFile(
                        path=dep_path, content=trunc,
                        priority=2, relevance_score=0.9, is_stub=False,
                    ))

        # ── Priority 3: semantic search hits ─────────────────────────────────
        # Guard: skip if the model warmup thread hasn't finished yet.  Calling
        # search() before _ensure_client() completes would block the event-loop
        # thread on a threading.Lock acquired by the background warmup thread.
        if self._embedding_store is not None and self._embedding_store.is_ready:
            try:
                hits = self._embedding_store.search(
                    task.description,
                    n_results=_MAX_SEMANTIC_HITS + 10,
                )
                already = {c.path for c in candidates}
                sem_added = 0
                for hit in hits:
                    if sem_added >= _MAX_SEMANTIC_HITS:
                        break
                    fp = hit.get("file", "")
                    if not fp or fp in already:
                        continue
                    content = self._read_file(fp)
                    if content is None:
                        continue
                    file_index = self.repo_index.get_file(fp)
                    checksum = file_index.checksum if file_index else ""
                    stub = _ast_extractor.extract_stub(fp, content, lang.name, checksum)
                    body = (
                        f"// Semantic match\n{stub}"
                        if stub is not None
                        else _smart_truncate(content, 4_000)
                    )
                    dist = hit.get("distance", 0.5)
                    score = max(0.0, 1.0 - dist)
                    candidates.append(ContextFile(
                        path=fp, content=body,
                        priority=3, relevance_score=score, is_stub=stub is not None,
                    ))
                    already.add(fp)
                    sem_added += 1
            except Exception:
                logger.debug("Semantic search failed (non-critical)", exc_info=True)

        # ── Priority 4: module review files ──────────────────────────────────
        if task.task_type in (TaskType.REVIEW_MODULE, TaskType.REVIEW_ARCHITECTURE):
            already = {c.path for c in candidates}
            ranked = self._rank_review_files(task)
            n_ranked = len(ranked)
            for rank, fb in enumerate(ranked):
                if fb.path in already:
                    continue
                content = self._read_file(fb.path)
                if content is None:
                    continue
                file_index = self.repo_index.get_file(fb.path)
                checksum = file_index.checksum if file_index else ""
                stub = _ast_extractor.extract_stub(fb.path, content, lang.name, checksum)
                body = (
                    f"// AST stub (signatures only)\n{stub}"
                    if stub is not None
                    else _smart_truncate(content, 8_000)
                )
                # Relevance score descends linearly from 0.9 (rank 0) to 0.5 (last).
                relevance = 0.9 - 0.4 * (rank / max(n_ranked - 1, 1))
                candidates.append(ContextFile(
                    path=fb.path, content=body,
                    priority=4, relevance_score=relevance, is_stub=stub is not None,
                ))

        # ── Sort: (priority asc, relevance_score desc) ───────────────────────
        candidates.sort(key=lambda c: (c.priority, -c.relevance_score))

        # ── Fill budget ───────────────────────────────────────────────────────
        result: dict[str, str] = {}
        total = 0
        for cf in candidates:
            if cf.path in result:
                continue  # already included (e.g. target file appears as dep too)
            if total + len(cf.content) > _MAX_CONTEXT_CHARS:
                logger.debug(
                    "Context budget reached (%d chars), skipping %s (priority=%d)",
                    total, cf.path, cf.priority,
                )
                continue
            result[cf.path] = cf.content
            total += len(cf.content)

        return result

    # ── Review-file ranking ───────────────────────────────────────────────────

    def _rank_review_files(self, task: Task) -> list[FileBlueprint]:
        """Return up to ``_MAX_REVIEW_FILES`` blueprint files ranked by relevance.

        REVIEW_MODULE — prioritises by directory proximity to ``task.file``,
        then by same architectural layer, then by dependency overlap.

        REVIEW_ARCHITECTURE — samples proportionally across all layers so the
        LLM sees the full stack rather than an arbitrary positional slice.
        """
        if task.task_type == TaskType.REVIEW_MODULE:
            return self._rank_module_review_files(task.file)
        return self._rank_architecture_review_files()

    def _rank_module_review_files(self, target_file: str) -> list[FileBlueprint]:
        """Rank files for REVIEW_MODULE by closeness to ``target_file``.

        When ``target_file`` is ``"*"`` (whole-project review), we must avoid
        mixing backend and frontend files into the same context window —
        the reviewer would flag Java code as being "in a TypeScript file."
        Instead, pick the dominant language and include only those files.
        """
        # Whole-project review: focus on the primary language to avoid
        # false "language mismatch" findings between BE and FE modules.
        if target_file == "*":
            return self._rank_module_review_by_language()

        target_dir = str(Path(target_file).parent)
        target_fb = self._find_blueprint(target_file)
        target_layer = target_fb.layer if target_fb else ""
        target_deps = set(target_fb.depends_on) if target_fb else set()

        def _score(fb: FileBlueprint) -> float:
            fb_dir = str(Path(fb.path).parent)
            if fb_dir == target_dir:
                return 3.0
            # Parent or child directory
            if fb_dir.startswith(target_dir + "/") or target_dir.startswith(fb_dir + "/"):
                return 2.0
            if target_layer and fb.layer == target_layer:
                return 1.5
            overlap = len(set(fb.depends_on) & target_deps)
            if overlap:
                return 1.0 + overlap * 0.1
            return 0.5

        ranked = sorted(self.blueprint.file_blueprints, key=_score, reverse=True)
        return ranked[:_MAX_REVIEW_FILES]

    def _rank_module_review_by_language(self) -> list[FileBlueprint]:
        """For whole-project module reviews, group files by language and
        return the largest language group so the reviewer checks cross-file
        consistency within one language, not across BE/FE boundaries.
        """
        by_lang: dict[str, list[FileBlueprint]] = {}
        for fb in self.blueprint.file_blueprints:
            lang = fb.language or "unknown"
            # Normalise: treat typescript/tsx as one group
            if lang in ("tsx", "jsx"):
                lang = "typescript"
            by_lang.setdefault(lang, []).append(fb)

        if not by_lang:
            return self.blueprint.file_blueprints[:_MAX_REVIEW_FILES]

        # Pick the largest language group (primary backend language)
        primary_lang = max(by_lang, key=lambda k: len(by_lang[k]))
        primary_files = by_lang[primary_lang]

        # Within the primary language, prioritise by dependency depth:
        # controllers/services first (they have cross-file calls to verify).
        _LAYER_PRIORITY = {
            "controller": 0, "service": 1, "repository": 2,
            "model": 3, "dto": 4, "config": 5,
        }

        def _layer_score(fb: FileBlueprint) -> int:
            return _LAYER_PRIORITY.get(fb.layer or "", 10)

        primary_files.sort(key=_layer_score)
        return primary_files[:_MAX_REVIEW_FILES]

    def _rank_architecture_review_files(self) -> list[FileBlueprint]:
        """Sample files proportionally across all architectural layers.

        For fullstack projects, restrict to the primary language group first
        so the reviewer doesn't flag BE/FE language mismatches.
        """
        # ── Language filtering (same logic as module review) ──────────
        source_files = self.blueprint.file_blueprints
        by_lang: dict[str, list[FileBlueprint]] = {}
        for fb in source_files:
            lang = fb.language or "unknown"
            if lang in ("tsx", "jsx"):
                lang = "typescript"
            by_lang.setdefault(lang, []).append(fb)

        if len(by_lang) > 1:
            # Fullstack: pick dominant language to avoid cross-language noise
            primary_lang = max(by_lang, key=lambda k: len(by_lang[k]))
            source_files = by_lang[primary_lang]

        # ── Layer-proportional sampling ──────────────────────────────
        by_layer: dict[str, list[FileBlueprint]] = {}
        for fb in source_files:
            by_layer.setdefault(fb.layer or "other", []).append(fb)

        n_layers = len(by_layer)
        if n_layers == 0:
            return source_files[:_MAX_REVIEW_FILES]

        per_layer = max(1, _MAX_REVIEW_FILES // n_layers)
        result: list[FileBlueprint] = []
        seen: set[str] = set()

        for layer_files in by_layer.values():
            for fb in layer_files[:per_layer]:
                result.append(fb)
                seen.add(fb.path)

        # Fill remaining budget from whichever layers had extras
        if len(result) < _MAX_REVIEW_FILES:
            for fb in source_files:
                if len(result) >= _MAX_REVIEW_FILES:
                    break
                if fb.path not in seen:
                    result.append(fb)

        return result

    def _collect_related_files(
        self, task: Task, file_bp: FileBlueprint | None
    ) -> dict[str, str]:
        """Legacy method — delegates to ``_collect_ranked``."""
        return self._collect_ranked(task, file_bp)

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
        """Get dependency information from the repo index, enriched by dep_store."""
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

        # Enrich with graph-derived impact analysis when dep_store is available.
        # This adds transitive dependents and heuristic test files that the
        # simple import scan above cannot derive.
        if self._dep_store is not None:
            impact = self._dep_store.get_impact_analysis(file_path)
            info["transitive_downstream"] = impact.get("transitive_dependents", [])
            info["affected_tests"] = impact.get("test_files", [])
            # Fill gaps: if graph knows direct deps that import scan missed, merge them
            for p in impact.get("direct_dependents", []):
                if p not in info["downstream"]:
                    info["downstream"].append(p)
            for p in impact.get("direct_dependencies", []):
                if p not in info["upstream"]:
                    info["upstream"].append(p)

        return info

    def _enrich_with_semantic_hits(
        self, task: Task, related: dict[str, str]
    ) -> None:
        """Add semantically similar files via vector search (in-place).

        Uses the task description as the query so the agent receives code
        that is conceptually related even when no explicit import exists.
        Respects the overall context budget and never overwrites files already
        included via blueprint dependencies.
        """
        # Guard: same readiness check as in _collect_ranked — avoid blocking
        # the event-loop thread if the background warmup hasn't completed yet.
        if self._embedding_store is None or not self._embedding_store.is_ready:
            return

        total_chars = sum(len(v) for v in related.values())
        if total_chars >= _MAX_CONTEXT_CHARS:
            return

        try:
            hits = self._embedding_store.search(
                task.description, n_results=_MAX_SEMANTIC_HITS + len(related)
            )
        except Exception:
            logger.debug("Semantic search failed (non-critical)", exc_info=True)
            return

        lang = detect_language_from_blueprint(self.blueprint.tech_stack)
        added = 0
        for hit in hits:
            if added >= _MAX_SEMANTIC_HITS:
                break
            if total_chars >= _MAX_CONTEXT_CHARS:
                break
            file_path = hit.get("file", "")
            if not file_path or file_path in related:
                continue
            content = self._read_file(file_path)
            if content is None:
                continue
            # Use AST stub where possible to keep token budget low
            file_index = self.repo_index.get_file(file_path)
            checksum = file_index.checksum if file_index else ""
            stub = _ast_extractor.extract_stub(
                file_path, content, lang.name, checksum=checksum,
            )
            snippet = (
                f"// Semantic match (similarity search)\n{stub}"
                if stub is not None
                else _smart_truncate(content, 4_000)
            )
            related[file_path] = snippet
            total_chars += len(snippet)
            added += 1
            logger.debug(
                "Semantic hit for '%s': %s (dist=%.3f)",
                task.description[:60], file_path, hit.get("distance", 0),
            )

    # ── Modification-aware context building ───────────────────────────

    def build_modification_context(
        self,
        task: Task,
        change_plan: ChangePlan | None = None,
    ) -> AgentContext:
        """Build context specifically for file modification tasks.

        Unlike ``build()``, this includes:
        - The FULL content of the target file (not truncated)
        - Related files that will be affected by the change
        - Dependency info so the agent knows what depends on the target
        - The change plan summary in architecture_summary for reference
        """
        file_bp = self._find_blueprint(task.file)
        related = self._collect_modification_context(task, change_plan)
        dep_info = self._build_dependency_info(task.file)

        # Include the change plan summary so the agent understands the bigger picture
        arch_summary = self.blueprint.architecture_doc
        if change_plan:
            arch_summary = (
                f"MODIFICATION CONTEXT\n"
                f"Change plan: {change_plan.summary}\n"
                f"Total changes: {len(change_plan.changes)}\n"
                f"Risk notes: {'; '.join(change_plan.risk_notes)}\n\n"
                f"{arch_summary}"
            )

        # Semantic enrichment for modification tasks — finds code that is
        # conceptually similar to the modification description.
        self._enrich_with_semantic_hits(task, related)

        return AgentContext(
            task=task,
            blueprint=self.blueprint,
            file_blueprint=file_bp,
            related_files=related,
            architecture_summary=arch_summary,
            dependency_info=dep_info,
        )

    def _collect_modification_context(
        self,
        task: Task,
        change_plan: ChangePlan | None,
    ) -> dict[str, str]:
        """Collect context files for a modification task.

        Prioritizes:
        1. The target file (full content, not truncated)
        2. Files the target depends on (AST stubs where possible)
        3. Files that depend on the target (they may need updates)
        4. Other files being changed in the same plan
        """
        related: dict[str, str] = {}
        total_chars = 0
        lang = detect_language_from_blueprint(self.blueprint.tech_stack)

        # 1. Target file — always in full, never truncated
        target_content = self._read_file(task.file)
        if target_content is not None:
            related[task.file] = target_content
            total_chars += len(target_content)

        # 2. Upstream dependencies (files the target imports from)
        dep_info = self._build_dependency_info(task.file)
        for up_path in dep_info.get("upstream", []):
            if total_chars >= _MAX_CONTEXT_CHARS:
                break
            if up_path in related:
                continue
            content = self._read_file(up_path)
            if content is None:
                continue
            # Use AST stub for dependencies
            file_index = self.repo_index.get_file(up_path)
            checksum = file_index.checksum if file_index else ""
            stub = _ast_extractor.extract_stub(up_path, content, lang.name, checksum=checksum)
            if stub is not None:
                related[up_path] = f"// AST stub (signatures only)\n{stub}"
                total_chars += len(stub)
            else:
                truncated = _smart_truncate(content, 6_000)
                related[up_path] = truncated
                total_chars += len(truncated)

        # 3. Downstream dependents (files that import from the target)
        for down_path in dep_info.get("downstream", [])[:5]:
            if total_chars >= _MAX_CONTEXT_CHARS:
                break
            if down_path in related:
                continue
            content = self._read_file(down_path)
            if content is None:
                continue
            file_index = self.repo_index.get_file(down_path)
            checksum = file_index.checksum if file_index else ""
            stub = _ast_extractor.extract_stub(down_path, content, lang.name, checksum=checksum)
            if stub is not None:
                related[down_path] = f"// AST stub (dependents)\n{stub}"
                total_chars += len(stub)
            else:
                truncated = _smart_truncate(content, 4_000)
                related[down_path] = truncated
                total_chars += len(truncated)

        # 4. Other files in the change plan (so the agent sees the full picture)
        if change_plan:
            for change in change_plan.changes:
                if change.file == task.file or change.file in related:
                    continue
                if total_chars >= _MAX_CONTEXT_CHARS:
                    break
                content = self._read_file(change.file)
                if content is None:
                    continue
                file_index = self.repo_index.get_file(change.file)
                checksum = file_index.checksum if file_index else ""
                stub = _ast_extractor.extract_stub(
                    change.file, content, lang.name, checksum=checksum,
                )
                if stub is not None:
                    related[change.file] = f"// AST stub (related change)\n{stub}"
                    total_chars += len(stub)
                else:
                    truncated = _smart_truncate(content, 4_000)
                    related[change.file] = truncated
                    total_chars += len(truncated)

        return related
