"""Final-pass workspace indexer — persists all generated files into memory stores."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from config.settings import Settings
from memory.dependency_graph import DependencyGraphStore
from memory.embedding_store import EmbeddingStore
from memory.repo_index import RepoIndexStore

if TYPE_CHECKING:
    from core.repository_manager import RepositoryManager

logger = logging.getLogger(__name__)


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

    for file_info in repo_index.files:
        index_store.update_file(file_info)

        for imp in file_info.imports:
            resolved = lang.resolve_import_to_path(imp, known_files)
            if resolved:
                dep_store.add_dependency(file_info.path, resolved)
            else:
                logger.debug("Unresolved import '%s' in %s", imp, file_info.path)

        # Re-index embeddings for files written before the embedding store was
        # wired (e.g. workspace init files) and to handle any missed updates.
        content = repo_manager.read_file(file_info.path)
        if content:
            embedding_store.index_file(file_info.path, content)

    dep_store.save()
    index_store.save()
    logger.info(
        "Workspace indexed: %d files, %d dependency edges",
        len(repo_index.files),
        sum(1 for _ in dep_store.get_graph().edges()),
    )
