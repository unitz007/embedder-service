import os
from datetime import datetime
from typing import Optional, Dict, Any

import psycopg2
from psycopg2.pool import ThreadedConnectionPool

_POOL: Optional[ThreadedConnectionPool] = None


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
    if _POOL is None:
        maxconn = int(os.getenv("DB_POOL_MAX", "20"))
        _POOL = ThreadedConnectionPool(
            minconn=2,
            maxconn=maxconn,
            dsn=get_database_url(),
        )
    return _POOL


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


def _row_to_job(row) -> Dict[str, Any]:
    if not row:
        return {}
    keys = [
        "job_id",
        "namespace",
        "project_id",
        "filename",
        "status",
        "source",
        "owner",
        "repo",
        "ref",
        "total_vectors",
        "persist_dir",
        "embedder",
        "webhook_url",
        "error",
        "created_at",
        "updated_at",
    ]
    job = dict(zip(keys, row))
    if isinstance(job.get("created_at"), datetime):
        job["created_at"] = job["created_at"].isoformat() + "Z"
    if isinstance(job.get("updated_at"), datetime):
        job["updated_at"] = job["updated_at"].isoformat() + "Z"
    return job


def create_job(
    job_id: str,
    namespace: str,
    project_id: str,
    filename: str,
    status: str,
    created_at: str,
    updated_at: str,
    webhook_url: Optional[str] = None,
    source: Optional[str] = None,
    owner: Optional[str] = None,
    repo: Optional[str] = None,
    ref: Optional[str] = None,
) -> Dict[str, Any]:
    conn = get_pool().getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO jobs (
                    job_id, namespace, project_id, filename, status,
                    source, owner, repo, ref, webhook_url,
                    created_at, updated_at
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                RETURNING job_id, namespace, project_id, filename, status,
                          source, owner, repo, ref, total_vectors,
                          persist_dir, embedder, webhook_url, error,
                          created_at, updated_at
                """,
                (
                    job_id, namespace, project_id, filename, status,
                    source, owner, repo, ref, webhook_url,
                    created_at, updated_at,
                ),
            )
            row = cur.fetchone()
            conn.commit()
            return _row_to_job(row)
    finally:
        get_pool().putconn(conn)


def update_job(job_id: str, **updates) -> Dict[str, Any]:
    if not updates:
        return get_job(job_id)
    updates["updated_at"] = updates.get("updated_at")
    fields = []
    values = []
    for k, v in updates.items():
        fields.append(f"{k} = %s")
        values.append(v)
    values.append(job_id)

    conn = get_pool().getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE jobs
                SET {", ".join(fields)}
                WHERE job_id = %s
                RETURNING job_id, namespace, project_id, filename, status,
                          source, owner, repo, ref, total_vectors,
                          persist_dir, embedder, webhook_url, error,
                          created_at, updated_at
                """,
                values,
            )
            row = cur.fetchone()
            conn.commit()
            return _row_to_job(row)
    finally:
        get_pool().putconn(conn)


def get_job(job_id: str) -> Dict[str, Any]:
    conn = get_pool().getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT job_id, namespace, project_id, filename, status,
                       source, owner, repo, ref, total_vectors,
                       persist_dir, embedder, webhook_url, error,
                       created_at, updated_at
                FROM jobs
                WHERE job_id = %s
                """,
                (job_id,),
            )
            row = cur.fetchone()
            return _row_to_job(row)
    finally:
        get_pool().putconn(conn)


def upsert_index_meta(namespace: str, project_id: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    now = datetime.utcnow().isoformat() + "Z"
    conn = get_pool().getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO index_meta (
                    namespace, project_id, embedder, model,
                    embedding_dim, use_cloud, preferred_embedder,
                    created_at, updated_at
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (namespace, project_id)
                DO UPDATE SET
                    embedder = EXCLUDED.embedder,
                    model = EXCLUDED.model,
                    embedding_dim = EXCLUDED.embedding_dim,
                    use_cloud = EXCLUDED.use_cloud,
                    preferred_embedder = EXCLUDED.preferred_embedder,
                    updated_at = EXCLUDED.updated_at
                RETURNING namespace, project_id, embedder, model,
                          embedding_dim, use_cloud, preferred_embedder,
                          created_at, updated_at
                """,
                (
                    namespace,
                    project_id,
                    meta.get("embedder"),
                    meta.get("model"),
                    meta.get("embedding_dim"),
                    bool(meta.get("use_cloud")),
                    meta.get("preferred_embedder"),
                    now,
                    now,
                ),
            )
            row = cur.fetchone()
            conn.commit()
            return {
                "namespace": row[0],
                "project_id": row[1],
                "embedder": row[2],
                "model": row[3],
                "embedding_dim": row[4],
                "use_cloud": row[5],
                "preferred_embedder": row[6],
                "created_at": row[7].isoformat() + "Z" if isinstance(row[7], datetime) else row[7],
                "updated_at": row[8].isoformat() + "Z" if isinstance(row[8], datetime) else row[8],
            }
    finally:
        get_pool().putconn(conn)


def get_index_meta(namespace: str, project_id: str) -> Dict[str, Any]:
    conn = get_pool().getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT namespace, project_id, embedder, model, embedding_dim,
                       use_cloud, preferred_embedder, created_at, updated_at
                FROM index_meta
                WHERE namespace = %s AND project_id = %s
                """,
                (namespace, project_id),
            )
            row = cur.fetchone()
            if not row:
                return {}
            return {
                "namespace": row[0],
                "project_id": row[1],
                "embedder": row[2],
                "model": row[3],
                "embedding_dim": row[4],
                "use_cloud": row[5],
                "preferred_embedder": row[6],
                "created_at": row[7].isoformat() + "Z" if isinstance(row[7], datetime) else row[7],
                "updated_at": row[8].isoformat() + "Z" if isinstance(row[8], datetime) else row[8],
            }
    finally:
        get_pool().putconn(conn)


def update_preferred_embedder(namespace: str, project_id: str, preferred_embedder: Optional[str]) -> Dict[str, Any]:
    """Update only the preferred_embedder column for a tenant.

    If *preferred_embedder* is ``None`` the column is set to ``NULL``,
    reverting to automatic embedder selection.
    """
    now = datetime.utcnow().isoformat() + "Z"
    conn = get_pool().getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO index_meta (
                    namespace, project_id, preferred_embedder,
                    created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (namespace, project_id)
                DO UPDATE SET
                    preferred_embedder = EXCLUDED.preferred_embedder,
                    updated_at = EXCLUDED.updated_at
                RETURNING namespace, project_id, embedder, model, embedding_dim,
                          use_cloud, preferred_embedder, created_at, updated_at
                """,
                (namespace, project_id, preferred_embedder, now, now),
            )
            row = cur.fetchone()
            conn.commit()
            return {
                "namespace": row[0],
                "project_id": row[1],
                "embedder": row[2],
                "model": row[3],
                "embedding_dim": row[4],
                "use_cloud": row[5],
                "preferred_embedder": row[6],
                "created_at": row[7].isoformat() + "Z" if isinstance(row[7], datetime) else row[7],
                "updated_at": row[8].isoformat() + "Z" if isinstance(row[8], datetime) else row[8],
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
            cur.execute(
                "DELETE FROM index_meta WHERE namespace = %s AND project_id = %s",
                (namespace, project_id),
            )
            conn.commit()
    finally:
        get_pool().putconn(conn)


def count_embeddings(namespace: str, project_id: str) -> int:
    conn = get_pool().getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM embeddings WHERE namespace = %s AND project_id = %s",
                (namespace, project_id),
            )
            return int(cur.fetchone()[0])
    finally:
        get_pool().putconn(conn)
