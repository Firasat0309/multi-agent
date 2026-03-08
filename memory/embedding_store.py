"""Vector memory system using ChromaDB for semantic code search."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class EmbeddingStore:
    """Semantic vector store for code chunks using ChromaDB."""

    def __init__(self, persist_dir: str = ".chroma", collection_name: str = "code") -> None:
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._client: Any = None
        self._collection: Any = None

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            import chromadb
            # chromadb >= 0.4 uses PersistentClient; the old Settings-based
            # constructor was removed.
            self._client = chromadb.PersistentClient(path=self._persist_dir)
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except ImportError:
            logger.warning("chromadb not installed, vector memory disabled")
            self._client = None

    def index_file(self, file_path: str, content: str) -> None:
        """Index a file's content as chunks."""
        self._ensure_client()
        if self._collection is None:
            return

        chunks = self._chunk_code(content)
        ids = [f"{file_path}::chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"file": file_path, "chunk_index": i} for i in range(len(chunks))]

        # Upsert to handle re-indexing
        self._collection.upsert(
            ids=ids,
            documents=chunks,
            metadatas=metadatas,
        )

    def search(self, query: str, n_results: int = 5) -> list[dict[str, Any]]:
        """Semantic search across indexed code."""
        self._ensure_client()
        if self._collection is None:
            return []

        results = self._collection.query(
            query_texts=[query],
            n_results=n_results,
        )

        hits: list[dict[str, Any]] = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                dist = results["distances"][0][i] if results["distances"] else 0
                hits.append({
                    "content": doc,
                    "file": meta.get("file", ""),
                    "chunk_index": meta.get("chunk_index", 0),
                    "distance": dist,
                })
        return hits

    def _chunk_code(self, content: str, chunk_size: int = 40) -> list[str]:
        """Split code into chunks by logical blocks (functions/classes or line groups)."""
        lines = content.splitlines()
        chunks: list[str] = []
        current_chunk: list[str] = []

        for line in lines:
            # Start new chunk on top-level definitions
            if (line.startswith("class ") or line.startswith("def ")) and current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
            current_chunk.append(line)
            if len(current_chunk) >= chunk_size:
                chunks.append("\n".join(current_chunk))
                current_chunk = []

        if current_chunk:
            chunks.append("\n".join(current_chunk))

        return chunks if chunks else [content]
