import os
from datetime import datetime
from typing import Optional

import hashlib
import json
import os
from datetime import datetime
from typing import Optional, List

import psycopg2
from psycopg2.extras import execute_values
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.pool import ThreadedConnectionPool


_POOL: Optional[ThreadedConnectionPool] = None

_POOL: Optional[ThreadedConnectionPool] = None


def _vector_str(vec: List[float]) -> str:
    return "[" + ",".join(str(float(x)) for x in vec) + "]"


def _normalize_db_url(url: str, app_env: str) -> str:
    if app_env != "prod":
        return url
    if "sslmode=" in url:
        return url
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}sslmode=require"


def get_database_url() -> str:
    app_env = os.getenv("APP_ENV", "dev").lower()
    url = os.getenv("DATABASE_URL", "").strip()
    if url:
        return _normalize_db_url(url, app_env)

    if app_env == "prod":
        raise RuntimeError("DATABASE_URL must be set in production")

    local_url = os.getenv("LOCAL_DATABASE_URL", "").strip()
    if local_url:
        return local_url

    return "postgresql://postgres:postgres@localhost:5432/indexer"


def get_pool() -> ThreadedConnectionPool:
    global _POOL
    if _POOL is not None:
        return _POOL
    maxconn = int(os.getenv("DB_POOL_MAX", "20"))
    _POOL = ThreadedConnectionPool(
        minconn=2,
        maxconn=maxconn,
        dsn=get_database_url(),
        connect_timeout=10,
        options='-c statement_timeout=30000',
    )
    return _POOL


def cache_get_embeddings(project_id: str, file_path: str) -> Optional[list]:
    """Get cached embeddings for a file from the database."""
    conn = get_pool().getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunk_id, embedding 
                FROM embeddings 
                WHERE project_id = %s AND file_path = %s
                ORDER BY chunk_id
                """,
                (project_id, file_path),
            )
            rows = cur.fetchall()
            if not rows:
                return None
            return [(row[0], row[1]) for row in rows]
    except Exception as e:
        print(f"Error fetching cached embeddings: {e}")
        return None
    finally:
        get_pool().putconn(conn)


def cache_set_embeddings(project_id: str, file_path: str, embeddings: list) -> None:
    """Cache computed embeddings for a file in the database."""
    if not embeddings:
        return
    
    conn = get_pool().getconn()
    try:
        with conn.cursor() as cur:
            # Delete existing embeddings for this file
            cur.execute(
                "DELETE FROM embeddings WHERE project_id = %s AND file_path = %s",
                (project_id, file_path),
            )
            
            # Insert new embeddings
            execute_values(
                cur,
                """
                INSERT INTO embeddings (
                    project_id, file_path, chunk_id, embedding, created_at
                ) VALUES %s
                """,
                [
                    (
                        project_id,
                        file_path,
                        emb[0],  # chunk_id
                        _vector_str(emb[1]),  # embedding
                        datetime.now(),
                    )
                    for emb in embeddings
                ],
                template="(%s,%s,%s,%s::vector,%s),
            )
            conn.commit()
    except Exception as e:
        print(f"Error setting cached embeddings: {e}")
        conn.rollback()
    finally:
        get_pool().putconn(conn)


def count_embeddings(namespace: str, project_id: str) -> int:
    """Count the number of embeddings for a project."""
    conn = get_pool().getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM embeddings WHERE namespace = %s AND project_id = %s",
                (namespace, project_id),
            )
            row = cur.fetchone()
            return row[0] if row else 0
    finally:
        get_pool().putconn(conn)

def cache_get_embeddings(keys: List[Tuple[bytes, str]]) -> Dict[bytes, List[float]]:
    """Get cached embeddings from the database.
    
    Args:
        keys: List of (hash, model_name) tuples
        
    Returns:
        Dictionary mapping hash -> embedding vector
    """
    if not keys:
        return {}
    
    conn = get_pool().getconn()
    try:
        with conn.cursor() as cur:
            # Create a temporary table with our keys for efficient lookup
            cur.execute("""
                CREATE TEMP TABLE temp_keys (
                    hash BYTEA,
                    model_name TEXT
                )
            """)
            
            # Insert keys
            execute_values(
                cur,
                "INSERT INTO temp_keys (hash, model_name) VALUES %s",
                keys
            )
            
            # Join with embeddings table
            cur.execute("""
                SELECT e.hash, e.embedding
                FROM embeddings_cache e
                INNER JOIN temp_keys t ON e.hash = t.hash AND e.model_name = t.model_name
            """)
            
            results = {}
            for hash_val, embedding in cur.fetchall():
                results[hash_val] = embedding
            
            return results
    except Exception as e:
        print(f"Error getting cached embeddings: {e}")
        return {}
    finally:
        get_pool().putconn(conn)


def cache_store_embeddings(rows: List[Tuple[bytes, str, List[float]]]) -> None:
    """Store embeddings in the cache.
    
    Args:
        rows: List of (hash, model_name, embedding) tuples
    """
    if not rows:
        return
    
    conn = get_pool().getconn()
    try:
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO embeddings_cache (hash, model_name, embedding, created_at)
                VALUES %s
                ON CONFLICT (hash, model_name) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    created_at = EXCLUDED.created_at
                """,
                [
                    (hash_val, model_name, embedding, datetime.now())
                    for hash_val, model_name, embedding in rows
                ],
                template="(%s,%s,%s,%s)"
            )
            conn.commit()
    except Exception as e:
        print(f"Error storing embeddings in cache: {e}")
        conn.rollback()
    finally:
        get_pool().putconn(conn)


def close_pool() -> None:
    """Gracefully close and drop the connection pool.

    Closes all connections and clears the global pool reference.  Safe to call
    multiple times; idempotent.
    """
    global _POOL
    if _POOL is not None:
        _POOL.closeall()
        _POOL = None


def init_db() -> None:
    conn = get_pool().getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    filename TEXT,
                    status TEXT NOT NULL,
                    source TEXT,
                    owner TEXT,
                    repo TEXT,
                    ref TEXT,
                    total_vectors INTEGER,
                    persist_dir TEXT,
                    embedder TEXT,
                    webhook_url TEXT,
                    error TEXT,
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL
                )
                """
            )
            cur.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS embedder TEXT")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS index_meta (
                    namespace TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    embedder TEXT,
                    model TEXT,
                    embedding_dim INTEGER,
                    use_cloud BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL,
                    PRIMARY KEY (namespace, project_id)
                )
                """
            )
            # Per-tenant embedder override
            cur.execute(
                "ALTER TABLE index_meta ADD COLUMN IF NOT EXISTS preferred_embedder TEXT"
            )
            # Migrate: drop embeddings table if it was created with wrong dimension (e.g. 1024)
            cur.execute("""
                SELECT a.atttypmod
                FROM pg_attribute a
                JOIN pg_class c ON c.oid = a.attrelid
                WHERE c.relname = 'embeddings' AND a.attname = 'embedding' AND NOT a.attisdropped
            """)
            dim_row = cur.fetchone()
            if dim_row is not None and dim_row[0] != 768:
                cur.execute("DROP TABLE IF EXISTS embeddings CASCADE")

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    id BIGSERIAL PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    file_path TEXT,
                    chunk_type TEXT,
                    symbol_name TEXT,
                    line_number INTEGER,
                    content TEXT,
                    metadata JSONB,
                    embedding vector(768),
                    created_at TIMESTAMPTZ NOT NULL,
                    UNIQUE (namespace, project_id, chunk_id)
                )
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS embeddings_vec_idx
                ON embeddings USING ivfflat (embedding vector_cosine_ops)
                """
            )            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS embeddings_vec_idx
                ON embeddings USING ivfflat (embedding vector_cosine_ops)
                """
            )
            
            # Create embeddings cache table for storing cached embeddings
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings_cache (
                    hash BYTEA NOT NULL,
                    model_name TEXT NOT NULL,
                    embedding VECTOR(1024), -- Support both 768 and 1024 dims
                    created_at TIMESTAMPTZ NOT NULL,
                    PRIMARY KEY (hash, model_name)
                )
                """
            )
            
            # Add index for faster lookups
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS embeddings_cache_hash_idx 
                ON embeddings_cache (hash)
                """
            )
            conn.commit()
    finally:
        get_pool().putconn(conn)


def get_job(job_id: str) -> Optional[dict]:
    conn = get_pool().getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT job_id, namespace, project_id, filename, status, source,
                       owner, repo, ref, total_vectors, persist_dir, embedder,
                       webhook_url, error, created_at, updated_at
                FROM jobs WHERE job_id = %s
                """,
                (job_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return {
                "job_id": row[0],
                "namespace": row[1],
                "project_id": row[2],
                "filename": row[3],
                "status": row[4],
                "source": row[5],
                "owner": row[6],
                "repo": row[7],
                "ref": row[8],
                "total_vectors": row[9],
                "persist_dir": row[10],
                "embedder": row[11],
                "webhook_url": row[12],
                "error": row[13],
                "created_at": row[14].isoformat() if row[14] else None,
                "updated_at": row[15].isoformat() if row[15] else None,
            }
    finally:
        get_pool().putconn(conn)


def create_job(**kwargs) -> None:
    conn = get_pool().getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO jobs (
                    job_id, namespace, project_id, filename, status, source,
                    owner, repo, ref, total_vectors, persist_dir, embedder,
                    webhook_url, error, created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    kwargs["job_id"],
                    kwargs["namespace"],
                    kwargs["project_id"],
                    kwargs.get("filename"),
                    kwargs["status"],
                    kwargs.get("source"),
                    kwargs.get("owner"),
                    kwargs.get("repo"),
                    kwargs.get("ref"),
                    kwargs.get("total_vectors"),
                    kwargs.get("persist_dir"),
                    kwargs.get("embedder"),
                    kwargs.get("webhook_url"),
                    kwargs.get("error"),
                    kwargs["created_at"],
                    kwargs["updated_at"],
                ),
            )
            conn.commit()
    finally:
        get_pool().putconn(conn)


def update_job(job_id: str, **kwargs) -> dict:
    conn = get_pool().getconn()
    try:
        with conn.cursor() as cur:
            # Build dynamic UPDATE clause
            fields = []
            values = []
            for k, v in kwargs.items():
                fields.append(f"{k} = %s")
                values.append(v)
            values.append(job_id)

            cur.execute(
                f"UPDATE jobs SET {', '.join(fields)} WHERE job_id = %s RETURNING *",
                values,
            )
            row = cur.fetchone()
            conn.commit()

            if not row:
                raise ValueError(f"Job {job_id} not found")

            return {
                "job_id": row[0],
                "namespace": row[1],
                "project_id": row[2],
                "filename": row[3],
                "status": row[4],
                "source": row[5],
                "owner": row[6],
                "repo": row[7],
                "ref": row[8],
                "total_vectors": row[9],
                "persist_dir": row[10],
                "embedder": row[11],
                "webhook_url": row[12],
                "error": row[13],
                "created_at": row[14].isoformat() if row[14] else None,
                "updated_at": row[15].isoformat() if row[15] else None,
            }
    finally:
        get_pool().putconn(conn)


def delete_embeddings(namespace: str, project_id: str) -> None:
    conn = get_pool().getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM embeddings WHERE namespace = %s AND project_id = %s",
                (namespace, project_id),
            )
            conn.commit()
    finally:
        get_pool().putconn(conn)


def get_index_meta(namespace: str, project_id: str) -> Optional[dict]:
    conn = get_pool().getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT embedder, model, embedding_dim, use_cloud, created_at, updated_at
                FROM index_meta WHERE namespace = %s AND project_id = %s
                """,
                (namespace, project_id),
            )
            row = cur.fetchone()
            if not row:
                return None
            return {
                "embedder": row[0],
                "model": row[1],
                "embedding_dim": row[2],
                "use_cloud": row[3],
                "created_at": row[4].isoformat() if row[4] else None,
                "updated_at": row[5].isoformat() if row[5] else None,
            }
    finally:
        get_pool().putconn(conn)


def upsert_index_meta(namespace: str, project_id: str, meta: dict) -> None:
    conn = get_pool().getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO index_meta (
                    namespace, project_id, embedder, model, embedding_dim, use_cloud,
                    created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (namespace, project_id)
                DO UPDATE SET
                    embedder = EXCLUDED.embedder,
                    model = EXCLUDED.model,
                    embedding_dim = EXCLUDED.embedding_dim,
                    use_cloud = EXCLUDED.use_cloud,
                    updated_at = EXCLUDED.updated_at
                """,
                (
                    namespace,
                    project_id,
                    meta["embedder"],
                    meta["model"],
                    meta["embedding_dim"],
                    meta.get("use_cloud", False),
                    meta["created_at"],
                    meta["updated_at"],
                ),
            )
            conn.commit()
    finally:
        get_pool().putconn(conn)