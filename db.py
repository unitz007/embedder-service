import os
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

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

            # ── Embedding cache table ──────────────────────────────────
            # The embedding column uses an untyped vector (no dimension)
            # because the PK is (content_hash, model_name), guaranteeing
            # that any given row's vector is always consumed with the
            # correct model.  A fixed dimension would break multi-model
            # caching (e.g. CodeBERT 768-d vs Voyage 1024-d).
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    content_hash BYTEA NOT NULL,
                    model_name   TEXT    NOT NULL,
                    embedding    vector  NOT NULL,
                    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (content_hash, model_name)
                )
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS embedding_cache_model_idx
                ON embedding_cache (model_name)
                """
            )

            conn.commit()
    finally:
        get_pool().putconn(conn)


# ── Embedding cache helpers ────────────────────────────────────────────────

def cache_get_embeddings(
    hashes_and_models: List[Tuple[bytes, str]],
) -> Dict[bytes, List[float]]:
    """Batch-fetch cached embeddings.

    Args:
        hashes_and_models: List of ``(content_hash, model_name)`` tuples.

    Returns:
        Dict mapping ``content_hash`` (bytes) → embedding (list of floats).
        Only hashes that were found in the cache are included.
    """
    if not hashes_and_models:
        return {}

    # Deduplicate input — same (hash, model) pair may appear multiple times
    unique_keys = list(set(hashes_and_models))

    # Build a composite VALUES clause for a single query.
    # psycopg2 %s placeholders work with BYTEA and TEXT transparently.
    rows = ", ".join("(%s, %s)" for _ in unique_keys)
    sql = f"""
        SELECT content_hash, embedding
        FROM embedding_cache
        WHERE (content_hash, model_name) IN ({rows})
    """
    flat_args = []
    for h, m in unique_keys:
        flat_args.extend([h, m])

    conn = get_pool().getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, flat_args)
            result: Dict[bytes, List[float]] = {}
            for content_hash, embedding in cur.fetchall():
                # pgvector returns the vector as a list of floats
                if isinstance(embedding, str):
                    # Strip brackets and parse
                    result[content_hash] = [float(x) for x in embedding.strip("[]").split(",")]
                else:
                    result[content_hash] = list(embedding)
            return result
    finally:
        get_pool().putconn(conn)


def cache_store_embeddings(
    rows: List[Tuple[bytes, str, List[float]]],
) -> None:
    """Batch-insert embeddings into the cache.

    Args:
        rows: List of ``(content_hash, model_name, embedding_list)`` tuples.
              Items whose key already exists are silently ignored (ON CONFLICT DO NOTHING).
    """
    if not rows:
        return

    from psycopg2.extras import execute_values

    conn = get_pool().getconn()
    try:
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO embedding_cache (content_hash, model_name, embedding)
                VALUES %s
                ON CONFLICT (content_hash, model_name) DO NOTHING
                """,
                rows,
                template="(%s, %s, %s::vector)",
            )
            conn.commit()
    finally:
        get_pool().putconn(conn)


# ── Job helpers ────────────────────────────────────────────────────────────

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


def update_job_index_meta(
    namespace: str,
    project_id: str,
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """Update embedder-related metadata on the most recent job for a project.

    This is a best-effort helper: if no job exists for the given
    ``(namespace, project_id)`` pair the function returns an empty dict
    without raising.
    """
    if not data:
        return {}

    # Only allow known columns to prevent SQL injection via key names.
    allowed = {
        "embedder", "model", "embedding_dim", "use_cloud",
        "total_vectors", "updated_at",
    }
    safe_fields = {k: v for k, v in data.items() if k in allowed}
    if not safe_fields:
        return {}

    fields_sql = ", ".join(f"{k} = %s" for k in safe_fields)
    values = list(safe_fields.values())

    conn = get_pool().getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE jobs
                SET {fields_sql}
                WHERE job_id = (
                    SELECT job_id FROM jobs
                    WHERE namespace = %s AND project_id = %s
                    ORDER BY created_at DESC
                    LIMIT 1
                )
                RETURNING job_id, namespace, project_id, filename, status,
                          source, owner, repo, ref, total_vectors,
                          persist_dir, embedder, webhook_url, error,
                          created_at, updated_at
                """,
                values + [namespace, project_id],
            )
            row = cur.fetchone()
            conn.commit()
            return _row_to_job(row) if row else {}
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
                    embedding_dim, use_cloud, created_at, updated_at
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (namespace, project_id)
                DO UPDATE SET
                    embedder = EXCLUDED.embedder,
                    model = EXCLUDED.model,
                    embedding_dim = EXCLUDED.embedding_dim,
                    use_cloud = EXCLUDED.use_cloud,
                    updated_at = EXCLUDED.updated_at
                RETURNING namespace, project_id, embedder, model, embedding_dim, use_cloud, created_at, updated_at
                """,
                (
                    namespace,
                    project_id,
                    meta.get("embedder"),
                    meta.get("model"),
                    meta.get("embedding_dim"),
                    bool(meta.get("use_cloud")),
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
                "created_at": row[6].isoformat() + "Z" if isinstance(row[6], datetime) else row[6],
                "updated_at": row[7].isoformat() + "Z" if isinstance(row[7], datetime) else row[7],
            }
    finally:
        get_pool().putconn(conn)


def get_index_meta(namespace: str, project_id: str) -> Dict[str, Any]:
    conn = get_pool().getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT namespace, project_id, embedder, model, embedding_dim, use_cloud, created_at, updated_at
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
                "created_at": row[6].isoformat() + "Z" if isinstance(row[6], datetime) else row[6],
                "updated_at": row[7].isoformat() + "Z" if isinstance(row[7], datetime) else row[7],
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
