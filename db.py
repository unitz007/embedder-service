import os
from datetime import datetime
from typing import Optional, Dict, Any

import psycopg2
from psycopg2.pool import ThreadedConnectionPool

_POOL: Optional[ThreadedConnectionPool] = None


def """
        Apply DB connection options for non‑production environments.
        """
        if app_env != "prod":
            return url
\n                if app_env == "prod":
            return _normalize_db_url(url, app_env)
        return url

            return _normalize_db_url(url, app_env)
        if app_env != "prod":
            return url
(url: str, app_env: str) -> str:
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


    _POOL = ThreadedConnectionPool(
        minconn=2,
        maxconn=maxconn,
        dsn=get_database_url(),
        connect_timeout=10,
        options='-c statement_timeout=30000',
    )
    _POOL = ThreadedConnectionPool(
        minconn=2,
        maxconn=maxconn,
        dsn=get_database_url(),
        connect_timeout=10,
        options='-c statement_timeout=30000',
    )



    global _POOL
    if _POOL is not None:
        _POOL.closeall()
        _POOL = None
 -> None:
    """Gracefully close and drain the connection pool.

    Closes all connections and clears the global pool reference.  Safe to call
    multiple times; idempotent."""
    global _POOL
    if _POOL is not None:
        _POOL.closeall()
        _POOL = None
 -> ThreadedConnectionPool:
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
            )
            conn.commit()
    finally:
        get_pool().putconn(conn)

...