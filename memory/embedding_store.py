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

    @property
    def is_ready(self) -> bool:
        """Return True if the ChromaDB client has been initialised.

        Callers that run on the asyncio event-loop thread should check this
        before calling ``search()`` to avoid blocking the loop on the
        threading lock inside ``_ensure_client()``.
        """
        return self._client is not None

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
        except ImportError:
            logger.warning("chromadb not installed, vector memory disabled")
            self._client = None
            return

        try:
            # chromadb >= 0.4 uses PersistentClient; the old Settings-based
            # constructor was removed.
            self._client = chromadb.PersistentClient(path=self._persist_dir)

            # Use the configured embedding model so collection semantics stay
            # consistent across runs and match the user's hardware budget.
            ef = None
            try:
                from chromadb.utils.embedding_functions import (
                    SentenceTransformerEmbeddingFunction,
                )
                ef = SentenceTransformerEmbeddingFunction(
                    model_name=self._embedding_model
                )
            except (ImportError, Exception) as e:
                logger.warning(
                    "Could not load embedding model '%s' (%s); falling back to ChromaDB default",
                    self._embedding_model, e,
                )

            kwargs: dict[str, Any] = {
                "name": self._collection_name,
                "metadata": {"hnsw:space": "cosine"},
            }
            if ef is not None:
                kwargs["embedding_function"] = ef

            self._collection = self._client.get_or_create_collection(**kwargs)
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

        For Java and Python files, uses AST-extracted signatures so each chunk
        carries semantic meaning (type + method name + annotations + parameter
        types) rather than arbitrary raw-line windows.  Falls back to the
        pattern-based splitter for unsupported languages or when the AST parse
        fails.
        """
        ext = Path(file_path).suffix.lower() if file_path else ""

        # AST-based chunking for Java and Python
        if ext in (".java", ".py") and file_path:
            ast_chunks = self._chunk_via_ast(content, file_path, ext.lstrip("."))
            if ast_chunks:
                return ast_chunks

        # Pattern-based fallback (all other languages / AST failures)
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

    def _chunk_via_ast(self, content: str, file_path: str, language: str) -> list[str]:
        """Return embedding chunks built from AST signatures (Java / Python).

        Produces two kinds of chunks per type:

        * **Class chunk** — file path, package, type header, annotations,
          and all field declarations.  Gives the search index a surface that
          matches queries like "UserService fields" or "class extending BaseEntity".
        * **Method chunk** — file path, class name, annotations, and the full
          method signature.  One chunk per method so retrieval is fine-grained.

        Returns an empty list on any failure so the caller can fall back to the
        pattern-based splitter.
        """
        try:
            # Lazy import avoids a circular dependency between memory/ and core/
            from core.ast_extractor import ASTExtractor  # noqa: PLC0415

            file_sig = ASTExtractor().extract(file_path, content, language)
            if file_sig is None or not file_sig.types:
                return []

            chunks: list[str] = []
            pkg_line = f"Package: {file_sig.package}" if file_sig.package else ""

            for type_sig in file_sig.types:
                # ── Class-level chunk ──────────────────────────────────────
                lines: list[str] = [f"File: {file_path}"]
                if pkg_line:
                    lines.append(pkg_line)
                header = f"{type_sig.kind} {type_sig.name}".strip()
                if type_sig.modifiers:
                    header = f"{type_sig.modifiers} {header}".strip()
                if type_sig.extends:
                    header += f" extends {type_sig.extends}"
                if type_sig.implements:
                    header += f" implements {', '.join(type_sig.implements)}"
                lines.append(header)
                if type_sig.annotations:
                    lines.append("Annotations: " + " ".join(type_sig.annotations))
                for fld in type_sig.fields:
                    field_line = f"  field: {fld.modifiers} {fld.type_name} {fld.name}".strip()
                    lines.append(field_line)
                if type_sig.enum_constants:
                    lines.append("  constants: " + ", ".join(type_sig.enum_constants))
                chunks.append("\n".join(lines))

                # ── Per-method chunks ──────────────────────────────────────
                for method in type_sig.methods:
                    mlines: list[str] = [f"File: {file_path}", f"Class: {type_sig.name}"]
                    if method.annotations:
                        mlines.append("Annotations: " + " ".join(method.annotations))
                    if method.is_constructor:
                        sig = f"{method.modifiers} {method.name}{method.parameters}".strip()
                    else:
                        ret = method.return_type or ""
                        sig = f"{method.modifiers} {ret} {method.name}{method.parameters}".strip()
                    mlines.append(sig)
                    chunks.append("\n".join(mlines))

            return chunks if chunks else []

        except Exception:
            logger.debug(
                "AST chunking failed for %s — falling back to pattern chunking", file_path
            )
            return []
