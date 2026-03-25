# chroma_store.py
"""
Chroma-backed vector store — drop-in replacement for VectorStore (FAISS).

Advantages over FAISS:
  - True deletion: delete_by_file() removes stale vectors immediately
  - Auto-persistence: no manual .save()/.load(); Chroma writes on every upsert
  - Upsert semantics: re-indexing a file updates existing vectors in-place
  - Per-repo isolation: each repo uses its own persist directory

Metadata serialization
-----------------------
Chroma only accepts scalar values (str, int, float, bool) in metadata dicts.
List fields (calls, called_by, depends_on, etc.) are JSON-encoded to strings
on write and transparently decoded back to lists on read.

Usage:
    from chroma_store import ChromaStore

    store = ChromaStore(persist_dir="/path/to/repo/.kael_index/chroma")
    store.add_vectors(embeddings, metadata_list)
    results = store.search(query_vector, k=5)   # [(meta, similarity, rank), ...]
    store.delete_by_file("src/auth.py")         # removes all chunks from that file
    store.clear()                               # wipe collection for a full rebuild
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import chromadb

try:

    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

# ---------------------------------------------------------------------------
# Metadata serialization helpers
# ---------------------------------------------------------------------------

# Fields that hold Python lists — must be JSON-encoded for Chroma storage.
_LIST_FIELDS: frozenset[str] = frozenset({
    "calls", "called_by", "external_calls",
    "depends_on", "imported_by", "params",
})


def _serialize(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a metadata dict to Chroma-compatible scalars."""
    out: Dict[str, Any] = {}
    for k, v in meta.items():
        if k in _LIST_FIELDS:
            out[k] = json.dumps(v) if v else "[]"
        elif isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif v is None:
            out[k] = ""
        else:
            out[k] = str(v)
    return out


def _deserialize(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Restore list fields from JSON strings after Chroma retrieval."""
    out: Dict[str, Any] = dict(meta)
    for k in _LIST_FIELDS:
        if k in out and isinstance(out[k], str):
            try:
                out[k] = json.loads(out[k])
            except (json.JSONDecodeError, TypeError):
                out[k] = []
    return out


def _chunk_id(meta: Dict[str, Any]) -> str:
    """
    Stable, deterministic ID for a chunk.

    Two chunks representing the same symbol in the same file always produce
    the same ID, so upsert correctly updates in-place on re-index rather than
    creating a duplicate.
    """
    key = "::".join([
        meta.get("file_path", ""),
        meta.get("chunk_type", ""),
        meta.get("function_name") or meta.get("class_name") or meta.get("symbol_name", ""),
        str(meta.get("line_number", "")),
    ])
    return hashlib.md5(key.encode()).hexdigest()


# ---------------------------------------------------------------------------
# ChromaStore
# ---------------------------------------------------------------------------

class ChromaStore:
    """
    Wraps a single Chroma collection and exposes the same interface as
    VectorStore so it can be used as a drop-in replacement throughout the
    indexing pipeline and ContextBuilder / HybridSearcher.

    Each instance manages exactly one collection (named "codebase") inside
    *persist_dir*.  Isolate different repositories by using different
    persist_dir values.
    """

    _COLLECTION = "codebase"

    def __init__(self, persist_dir: str, dimension: int = 768):  # 768 = CodeBERT hidden size
        if not CHROMA_AVAILABLE:
            raise ImportError(
                "chromadb is not installed. Run:  pip install chromadb"
            )
        self.dimension = dimension
        self._persist_dir = persist_dir
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=persist_dir)
        self._col = self._client.get_or_create_collection(
            name=self._COLLECTION,
            # cosine distance → similarity = 1 − distance
            metadata={"hnsw:space": "cosine"},
        )

        # Lazy cache — invalidated on every write
        self._metadata_cache: Optional[List[Dict[str, Any]]] = None

    # ------------------------------------------------------------------
    # VectorStore-compatible interface
    # ------------------------------------------------------------------

    @property
    def metadata(self) -> List[Dict[str, Any]]:
        """All stored metadata dicts (fetched once, then cached)."""
        if self._metadata_cache is None:
            result = self._col.get(include=["metadatas"])
            metas = result.get("metadatas") or []
            self._metadata_cache = [_deserialize(m) for m in metas]
        return self._metadata_cache

    def add_vectors(
        self,
        vectors: List[List[float]],
        metadata: List[Dict[str, Any]],
    ) -> None:
        """Upsert vectors with their metadata.  Idempotent for unchanged chunks."""
        if not vectors:
            return
        ids = [_chunk_id(m) for m in metadata]
        serialized = [_serialize(m) for m in metadata]
        self._col.upsert(ids=ids, embeddings=vectors, metadatas=serialized)
        self._metadata_cache = None  # invalidate cache
        print(f"Upserted {len(vectors)} vectors. Total: {self.get_total_vectors()}")

    def search(
        self,
        query_vector: List[float],
        k: int = 5,
    ) -> List[Tuple[Dict[str, Any], float, int]]:
        """
        Return top-k results as (metadata, cosine_similarity, rank) tuples.

        Chroma returns cosine *distance* ∈ [0, 2].  We convert to cosine
        *similarity* ∈ [-1, 1] via similarity = 1 − distance so that the
        return format matches VectorStore.search().
        """
        total = self.get_total_vectors()
        if total == 0:
            return []
        k = min(k, total)
        result = self._col.query(
            query_embeddings=[query_vector],
            n_results=k,
            include=["metadatas", "distances"],
        )
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        return [
            (_deserialize(m), 1.0 - d, i)
            for i, (m, d) in enumerate(zip(metas, dists))
        ]

    def get_total_vectors(self) -> int:
        return self._col.count()

    def save(self, filepath: str) -> None:
        """No-op — Chroma auto-persists to disk on every write."""

    def load(self, filepath: str) -> None:
        """No-op — data is already loaded from persist_dir in __init__."""

    # ------------------------------------------------------------------
    # Chroma-specific extras (not in VectorStore)
    # ------------------------------------------------------------------

    def delete_by_file(self, file_path: str) -> int:
        """
        Remove all chunks that belong to *file_path*.

        Returns the number of deleted entries.  Call this before re-indexing
        a modified file to avoid stale duplicate vectors.
        """
        result = self._col.get(
            where={"file_path": {"$eq": file_path}},
            include=["metadatas"],
        )
        ids = result.get("ids") or []
        if ids:
            self._col.delete(ids=ids)
            self._metadata_cache = None
        return len(ids)

    def clear(self) -> None:
        """
        Delete the entire collection and recreate it empty.

        Used at the start of a full rebuild so no stale vectors remain from
        previously indexed files that no longer exist.
        """
        self._client.delete_collection(self._COLLECTION)
        self._col = self._client.get_or_create_collection(
            name=self._COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self._metadata_cache = None

    @property
    def persist_dir(self) -> str:
        return self._persist_dir
