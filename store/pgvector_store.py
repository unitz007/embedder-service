from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Tuple

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


class PgVectorStore:
    def __init__(self, namespace: str, project_id: str) -> None:
        self.namespace = namespace
        self.project_id = project_id

    def add_vectors(self, vectors: List[List[float]], metadata: List[Dict[str, Any]], replace_all: bool = False) -> None:
        if replace_all:
            self.clear()

        if not vectors:
            return

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
            conn = db.get_pool().getconn()
            try:
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
            finally:
                db.get_pool().putconn(conn)

    def search(self, query_vector: List[float], k: int = 5) -> List[Tuple[Dict[str, Any], float, int]]:
        qv = _vector_str(query_vector)
        conn = db.get_pool().getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT metadata, 1 - (embedding <=> %s::vector) AS similarity
                    FROM embeddings
                    WHERE namespace = %s AND project_id = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (qv, self.namespace, self.project_id, qv, k),
                )
                rows = cur.fetchall()
                results = []
                for i, (meta_json, sim) in enumerate(rows):
                    meta = meta_json if isinstance(meta_json, dict) else json.loads(meta_json)
                    results.append((meta, float(sim), i))
                return results
        finally:
            db.get_pool().putconn(conn)

    def get_total_vectors(self) -> int:
        return db.count_embeddings(self.namespace, self.project_id)

    def clear(self) -> None:
        db.delete_embeddings(self.namespace, self.project_id)
