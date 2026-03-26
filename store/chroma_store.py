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
import shutil
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import chromadb

try:

    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

# ---------------------------------------------------------------------------
# Single shared PersistentClient per persist_dir
# ---------------------------------------------------------------------------
# chromadb.PersistentClient opens an exclusive SQLite connection. Opening
# multiple clients to the same path in the same process causes
# SQLITE_READONLY_DBMOVED (code 1032) when one client writes while another
# has a stale connection to the same file.  Caching one client per path
# ensures all ChromaStore instances share a single connection.

_client_lock = threading.Lock()
_client_cache: Dict[str, chromadb.PersistentClient] = {}


def _get_or_create_client(persist_dir: str) -> chromadb.PersistentClient:
    """Return the cached PersistentClient for *persist_dir*, creating it once."""
    # Normalise path so "/a/b/" and "/a/b" map to the same key
    key = str(Path(persist_dir).resolve())
    with _client_lock:
        if key not in _client_cache:
            _client_cache[key] = chromadb.PersistentClient(path=persist_dir)
        return _client_cache[key]


def reset_chroma_dir(persist_dir: str) -> None:
    """Fully reset a ChromaDB persist directory before a fresh index run.

    Three things must happen in order to avoid SQLITE_READONLY_DBMOVED
    (ChromaDB InternalError code 1032):

    1. Evict our module-level PersistentClient cache entry for this path so
       _get_or_create_client() will create a new connection afterward.

    2. Call SharedSystemClient.clear_system_cache() — the *actual* fix.
       In ChromaDB ≥1.x, PersistentClient() is a plain function, not a
       class, so chromadb.PersistentClient.clear_system_cache() silently
       raises AttributeError and does nothing.  The real method lives on
       SharedSystemClient and zeroes _identifier_to_system, which is the
       dict that keeps background threads (WAL checkpoint, segment GC, index
       compaction) alive after our Python reference is dropped.  Without
       this call those threads keep running, write back into the directory
       moments after we delete it, and race with the new client's first
       upsert — producing SQLITE_READONLY_DBMOVED every time.

    3. Delete and recreate the directory so the new PersistentClient opens
       a brand-new inode with no shared history.
    """
    key = str(Path(persist_dir).resolve())

    # Step 1 — evict our cache entry.
    with _client_lock:
        _client_cache.pop(key, None)

    # Step 2 — stop ChromaDB's singleton server + background threads.
    try:
        from chromadb.api.client import SharedSystemClient
        SharedSystemClient.clear_system_cache()
    except Exception:
        pass

    # Step 3 — fresh directory (new inode, zero stale state).
    persist_path = Path(persist_dir)
    shutil.rmtree(persist_path, ignore_errors=True)
    persist_path.mkdir(parents=True, exist_ok=True)

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

        self._client = _get_or_create_client(persist_dir)
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

    # ChromaDB hard-caps a single upsert at this many items.
    _UPSERT_BATCH = 5000

    def add_vectors(
        self,
        vectors: List[List[float]],
        metadata: List[Dict[str, Any]],
        replace_all: bool = False,
    ) -> None:
        """Upsert vectors with their metadata.

        When *replace_all* is True this method provides full-replace semantics
        without calling delete_collection():

        • delete_collection() internally issues SQLite VACUUM INTO which
          renames the database file, changing its inode.  Any open connection
          (including the one that triggered the VACUUM) then gets
          SQLITE_READONLY_DBMOVED (ChromaDB InternalError code 1032) on the
          very next write.

        • Instead we upsert all new chunks (idempotent — same chunk_id means
          same vector is updated in-place) and then delete only the IDs that
          are in the collection but absent from the new batch.  These are
          chunks that belonged to files deleted or renamed since the last run.
          No VACUUM, no file rename, no background-thread race.
        """
        if not vectors:
            if replace_all:
                result = self._col.get(include=[])
                all_ids = result.get("ids") or []
                for i in range(0, len(all_ids), self._UPSERT_BATCH):
                    self._col.delete(ids=all_ids[i : i + self._UPSERT_BATCH])
                self._metadata_cache = None
            return

        ids = [_chunk_id(m) for m in metadata]
        serialized = [_serialize(m) for m in metadata]

        for start in range(0, len(ids), self._UPSERT_BATCH):
            end = start + self._UPSERT_BATCH
            self._col.upsert(
                ids=ids[start:end],
                embeddings=vectors[start:end],
                metadatas=serialized[start:end],
            )

        if replace_all:
            new_id_set = set(ids)
            result = self._col.get(include=[])
            existing_ids = result.get("ids") or []
            stale = [eid for eid in existing_ids if eid not in new_id_set]
            if stale:
                for i in range(0, len(stale), self._UPSERT_BATCH):
                    self._col.delete(ids=stale[i : i + self._UPSERT_BATCH])
                print(f"Removed {len(stale)} stale vectors from previous index.")

        self._metadata_cache = None
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
        """Wipe all data and reopen a fresh collection on this directory."""
        reset_chroma_dir(self._persist_dir)
        self._client = _get_or_create_client(self._persist_dir)
        self._col = self._client.get_or_create_collection(
            name=self._COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self._metadata_cache = None

    @property
    def persist_dir(self) -> str:
        return self._persist_dir
