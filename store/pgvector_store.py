from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import psycopg2
from psycopg2.extras import execute_values

import db


def _chunk_id(meta: Dict[str, Any]) -> str:
    key = "::".join([
        meta.get("file_path", ""),
        meta.get("chunk_type", ""),
        meta.get("function_name") or meta.get("class_name") or meta.get("symbol_name", ""),
        str(meta.get("line_number", "")),
    ])
    return hashlib.md5(key.encode()).hexdigest()


def _vector_str(vec: List[float]) -> str:
    return "[" + ",".join(str(float(x)) for x in vec) + "]"


_INSERT_BATCH = 200

# Expected embedding dimension — validated on first write if any vectors present.
_EXPECTED_DIM = 768


class PgVectorStore:
    def __init__(self, namespace: str, project_id: str) -> None:
        self.namespace = namespace
        self.project_id = project_id
        self._dim: Optional[int] = None

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _connection(self):
        """Yield a pooled connection, guaranteeing putconn on exit."""
        conn = db.get_pool().getconn()
        try:
            yield conn
        finally:
            db.get_pool().putconn(conn)

    def _get_dimension(self) -> Optional[int]:
        """Query pg_attribute for the embedding column dimension (cached)."""
        if self._dim is not None:
            return self._dim
        try:
            conn = db.get_pool().getconn()
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT a.atttypmod
                        FROM pg_attribute a
                        JOIN pg_class c ON c.oid = a.attrelid
                        JOIN pg_namespace n ON n.oid = c.relnamespace
                        WHERE c.relname = 'embeddings'
                          AND a.attname = 'embedding'
                          AND NOT a.attisdropped
                          AND n.nspname = current_schema()
                    """)
                    row = cur.fetchone()
                    if row is not None and row[0] != -1:
                        self._dim = row[0]
                    else:
                        self._dim = _EXPECTED_DIM
            finally:
                db.get_pool().putconn(conn)
        except Exception:
            self._dim = _EXPECTED_DIM
        return self._dim

    def _validate_dimension(self, vectors: List[List[float]]) -> None:
        """Raise ValueError if vector dimensions don't match the column."""
        if not vectors:
            return
        dim = self._get_dimension()
        actual = len(vectors[0])
        if actual != dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {dim}, got {actual}. "
                f"Ensure the correct embedding model is being used."
            )

    # ------------------------------------------------------------------
    # Streaming add_vectors — accepts lists or generators
    # ------------------------------------------------------------------

    def add_vectors(
        self,
        vectors: Union[List[List[float]], Iterator[List[float]]],
        metadata: Union[List[Dict[str, Any]], Iterator[Dict[str, Any]]],
        replace_all: bool = False,
    ) -> None:
        if replace_all:
            self.clear()

        # Materialise into a list to support multiple passes (validation + insert)
        # and maintain backward compatibility with callers passing plain lists.
        if not isinstance(vectors, list):
            vectors = list(vectors)
            metadata = list(metadata)

        if not vectors:
            return

        self._validate_dimension(vectors)

        rows = [
            (
                self.namespace,
                self.project_id,
                _chunk_id(meta),
                meta.get("file_path"),
                meta.get("chunk_type") or meta.get("type"),
                meta.get("symbol_name"),
                meta.get("line_number"),
                meta.get("content"),
                json.dumps(meta),
                _vector_str(vec),
            )
            for vec, meta in zip(vectors, metadata)
        ]

        for i in range(0, len(rows), _INSERT_BATCH):
            batch = rows[i : i + _INSERT_BATCH]
            with self._connection() as conn:
                with conn.cursor() as cur:
                    execute_values(
                        cur,
                        """
                        INSERT INTO embeddings (
                            namespace, project_id, chunk_id, file_path, chunk_type,
                            symbol_name, line_number, content, metadata, embedding, created_at
                        ) VALUES %s
                        ON CONFLICT (namespace, project_id, chunk_id)
                        DO UPDATE SET
                            file_path = EXCLUDED.file_path,
                            chunk_type = EXCLUDED.chunk_type,
                            symbol_name = EXCLUDED.symbol_name,
                            line_number = EXCLUDED.line_number,
                            content = EXCLUDED.content,
                            metadata = EXCLUDED.metadata,
                            embedding = EXCLUDED.embedding
                        """,
                        batch,
                        template="(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::vector, NOW())",
                    )
                    conn.commit()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: List[float],
        k: int = 5,
        offset: int = 0,
        limit: Optional[int] = None,
    ) -> List[Tuple[Dict[str, Any], float, int]]:
        qv = _vector_str(query_vector)
        effective_limit = limit if limit is not None else k
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT metadata, 1 - (embedding <=> %s::vector) AS similarity
                    FROM embeddings
                    WHERE namespace = %s AND project_id = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s OFFSET %s
                    """,
                    (qv, self.namespace, self.project_id, qv, effective_limit, offset),
                )
                rows = cur.fetchall()
                results = []
                for i, (meta_json, sim) in enumerate(rows):
                    meta = meta_json if isinstance(meta_json, dict) else json.loads(meta_json)
                    results.append((meta, float(sim), i))
                return results

    def get_total_vectors(self) -> int:
        return db.count_embeddings(self.namespace, self.project_id)

    def clear(self) -> None:
        db.delete_embeddings(self.namespace, self.project_id)
