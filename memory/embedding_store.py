"""Vector memory system using ChromaDB for semantic code search."""

from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Module-level lock: ChromaDB's PersistentClient uses a shared class-level
# dictionary keyed by persist_dir path.  Concurrent threads hitting the same
# path simultaneously trigger a KeyError race condition inside ChromaDB.  A
# single lock serialises all first-time initialisations across all instances.
_chroma_init_lock = threading.Lock()

# Top-level definition boundary patterns, keyed by file extension.
# A line matching one of these signals the start of a new logical chunk.
# Extensions not in this table fall back to fixed-size window splitting.
_TOPLEVEL_PATTERNS: dict[str, re.Pattern[str]] = {
    ".py":   re.compile(r"^(class |def |async def )"),
    ".go":   re.compile(r"^func "),
    ".rs":   re.compile(r"^(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn |^(?:pub\s+)?(?:struct|enum|trait|impl)\s"),
    ".rb":   re.compile(r"^(?:def |class |module )"),
    ".java": re.compile(r"^(\s*(@\w+)\s*)*\s*(public|private|protected|static|abstract|final|synchronized|native)\b"),
    ".ts":   re.compile(r"^(?:export\s+)?(?:default\s+)?(?:async\s+)?function\b|^(?:export\s+)?(?:abstract\s+)?(?:class|interface|type|enum)\b"),
    ".tsx":  re.compile(r"^(?:export\s+)?(?:default\s+)?(?:async\s+)?function\b|^(?:export\s+)?(?:abstract\s+)?(?:class|interface|type|enum)\b"),
    ".js":   re.compile(r"^(?:export\s+)?(?:default\s+)?(?:async\s+)?function\b|^(?:export\s+)?class\b"),
    ".jsx":  re.compile(r"^(?:export\s+)?(?:default\s+)?(?:async\s+)?function\b|^(?:export\s+)?class\b"),
    ".cs":   re.compile(r"^\s*(?:public|private|protected|internal|static|abstract|sealed|override|virtual)\b"),
    ".swift":re.compile(r"^(?:(?:public|private|internal|open|fileprivate)\s+)?(?:func|class|struct|enum|protocol|extension)\b"),
    ".kt":   re.compile(r"^(?:fun |(?:(?:data|sealed|abstract|open|inner)\s+)?class |object |interface )"),
    ".php":  re.compile(r"^(?:(?:public|private|protected|static|abstract|final)\s+)*(?:function|class|interface|trait)\b"),
    ".cpp":  re.compile(r"^[A-Za-z_][\w:* <>]*\s+[A-Za-z_]\w*\s*\("),
    ".c":    re.compile(r"^[A-Za-z_][\w* ]*\s+[A-Za-z_]\w*\s*\("),
}


class EmbeddingStore:
    """Semantic vector store for code chunks using ChromaDB."""

    def __init__(
        self,
        persist_dir: str = ".chroma",
        collection_name: str = "code",
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._embedding_model = embedding_model
        self._client: Any = None
        self._collection: Any = None

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        with _chroma_init_lock:
            # Re-check inside the lock — another thread may have initialised
            # while we were waiting.
            if self._client is not None:
                return
            self._ensure_client_locked()

    def _ensure_client_locked(self) -> None:
        """Inner init that runs under ``_chroma_init_lock``."""
        try:
            import chromadb
            from chromadb.utils.embedding_functions import (
                SentenceTransformerEmbeddingFunction,
            )

            # chromadb >= 0.4 uses PersistentClient; the old Settings-based
            # constructor was removed.
            self._client = chromadb.PersistentClient(path=self._persist_dir)

            # Use the configured embedding model so collection semantics stay
            # consistent across runs and match the user's hardware budget.
            try:
                ef = SentenceTransformerEmbeddingFunction(
                    model_name=self._embedding_model
                )
            except Exception as e:
                logger.warning(
                    "Could not load embedding model '%s' (%s); falling back to ChromaDB default",
                    self._embedding_model, e,
                )
                ef = None  # chromadb will use its built-in default

            kwargs: dict[str, Any] = {
                "name": self._collection_name,
                "metadata": {"hnsw:space": "cosine"},
            }
            if ef is not None:
                kwargs["embedding_function"] = ef

            self._collection = self._client.get_or_create_collection(**kwargs)
        except ImportError:
            logger.warning("chromadb not installed, vector memory disabled")
            self._client = None
        except Exception as e:
            logger.warning("ChromaDB init failed (%s) — vector memory disabled", e)
            self._client = None

    def index_file(self, file_path: str, content: str) -> None:
        """Index a file's content as chunks."""
        self._ensure_client()
        if self._collection is None:
            return

        chunks = self._chunk_code(content, file_path=file_path)
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

    def _chunk_code(self, content: str, chunk_size: int = 40, file_path: str = "") -> list[str]:
        """Split code into chunks at top-level definition boundaries.

        Uses language-appropriate patterns derived from *file_path*'s extension.
        Falls back to fixed *chunk_size*-line windows for unknown file types.
        """
        ext = Path(file_path).suffix.lower() if file_path else ""
        pattern = _TOPLEVEL_PATTERNS.get(ext)

        lines = content.splitlines()
        chunks: list[str] = []
        current_chunk: list[str] = []

        for line in lines:
            # Flush the current chunk when we hit a top-level definition boundary.
            if current_chunk and pattern is not None and pattern.match(line):
                chunks.append("\n".join(current_chunk))
                current_chunk = []
            current_chunk.append(line)
            # Also flush when the fixed-size window fills up.
            if len(current_chunk) >= chunk_size:
                chunks.append("\n".join(current_chunk))
                current_chunk = []

        if current_chunk:
            chunks.append("\n".join(current_chunk))

        return chunks if chunks else [content]
