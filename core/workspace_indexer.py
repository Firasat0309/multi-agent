"""Final-pass workspace indexer — persists all generated files into memory stores.

Optimisations over the naive sequential approach:
  - **Incremental indexing**: only files whose checksum changed since the last
    indexing pass are re-embedded, avoiding redundant ChromaDB upserts.
  - **Batched embedding upserts**: files are collected into batches (default 20)
    and upserted in one ChromaDB call per batch instead of one call per file.
  - **Parallel file reads**: uses ``concurrent.futures.ThreadPoolExecutor`` so
    disk I/O for reading file content doesn't block sequentially.
"""

from __future__ import annotations

import concurrent.futures
import logging
from typing import TYPE_CHECKING

from config.settings import Settings
from memory.dependency_graph import DependencyGraphStore
from memory.embedding_store import EmbeddingStore
from memory.repo_index import RepoIndexStore

if TYPE_CHECKING:
    from core.repository_manager import RepositoryManager

logger = logging.getLogger(__name__)

# Maximum number of files to batch in a single ChromaDB upsert.
_EMBED_BATCH_SIZE = 20
# Maximum number of parallel file reads.
_MAX_READ_WORKERS = 8


def index_workspace(repo_manager: RepositoryManager, settings: Settings) -> None:
    """Index all files in *repo_manager* into the persistent memory stores.

    Called as a finalisation step by both the generation and modification
    pipelines.  The incremental write-time indexing (via
    ``RepositoryManager._embedding_store``) keeps the vector index current
    during execution; this pass persists the dependency graph and repo index
    to disk so subsequent runs (and the enhance pipeline) can load them.
    """
    index_store = RepoIndexStore(settings.workspace_dir)
    dep_store = DependencyGraphStore(settings.workspace_dir)
    embedding_store = EmbeddingStore(
        persist_dir=settings.memory.chroma_persist_dir,
        embedding_model=settings.memory.embedding_model,
    )

    repo_index = repo_manager.get_repo_index()
    known_files = {f.path for f in repo_index.files}
    lang = repo_manager._lang_profile

    # Track which checksums were already indexed in the embedding store
    # on a previous run.  Read the persisted set (if any) so we can skip
    # files whose content hasn't changed.
    _indexed_checksums: set[str] = getattr(repo_manager, "_indexed_checksums", set())

    # Collect files that need re-embedding (changed since last pass).
    files_to_embed: list[str] = []

    for file_info in repo_index.files:
        index_store.update_file(file_info)

        for imp in file_info.imports:
            resolved = lang.resolve_import_to_path(imp, known_files)
            if resolved:
                dep_store.add_dependency(file_info.path, resolved)
            else:
                logger.debug("Unresolved import '%s' in %s", imp, file_info.path)

        # Only re-embed if the file's checksum changed.
        if file_info.checksum and file_info.checksum in _indexed_checksums:
            continue
        files_to_embed.append(file_info.path)

    # ── Parallel file reads + batched embedding upserts ────────────────────
    if files_to_embed:
        def _read_file(path: str) -> tuple[str, str | None]:
            return path, repo_manager.read_file(path)

        with concurrent.futures.ThreadPoolExecutor(max_workers=_MAX_READ_WORKERS) as pool:
            results = list(pool.map(_read_file, files_to_embed))

        batch_paths: list[str] = []
        batch_contents: list[str] = []

        for path, content in results:
            if not content:
                continue
            batch_paths.append(path)
            batch_contents.append(content)

            if len(batch_paths) >= _EMBED_BATCH_SIZE:
                _embed_batch(embedding_store, batch_paths, batch_contents)
                batch_paths, batch_contents = [], []

        if batch_paths:
            _embed_batch(embedding_store, batch_paths, batch_contents)

        # Update the checksum set so subsequent calls skip unchanged files.
        embed_set = set(files_to_embed)
        new_checksums = {
            fi.checksum for fi in repo_index.files
            if fi.checksum and fi.path in embed_set
        }
        _indexed_checksums.update(new_checksums)
        repo_manager._indexed_checksums = _indexed_checksums  # type: ignore[attr-defined]

    dep_store.save()
    index_store.save()
    logger.info(
        "Workspace indexed: %d files (%d re-embedded), %d dependency edges",
        len(repo_index.files),
        len(files_to_embed),
        sum(1 for _ in dep_store.get_graph().edges()),
    )


def _embed_batch(
    store: EmbeddingStore,
    paths: list[str],
    contents: list[str],
) -> None:
    """Index a batch of files into the embedding store."""
    for path, content in zip(paths, contents):
        try:
            store.index_file(path, content)
        except Exception as e:
            logger.warning("Embedding index failed for %s: %s", path, e)
