"""
Pipeline for embedding and storing code chunks.

Exposes a single ``run_indexing`` entry-point consumed by the CLI and the
Flask API.  Re-indexing a project that was previously embedded with a
different model will automatically re-embed all chunks (via the
``clear()`` call on the pgvector store).
"""
from __future__ import annotations

import logging
import os
import shutil
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import db
from lib import embedder
from lib import chunker
from store.pgvector_store import PgVectorStore

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------


def _resolve_embedder(
    *,
    force_embedder: Optional[str] = None,
    use_cloud: bool = False,
    cache: Optional[embedder.EmbeddingCache] = None,
):
    """Return the embedder instance to use."""
    if force_embedder == "voyage" or use_cloud:
        return embedder.VoyageEmbedder(cache=cache)
    if force_embedder and force_embedder != "local":
        return embedder.CodeEmbedder(model_name=force_embedder, cache=cache)
    # Default — local CodeBERT
    return embedder.CodeEmbedder(cache=cache)


# -------------------------------------------------------------------
# Pipeline
# -------------------------------------------------------------------


def run_indexing(
    *,
    namespace: str,
    project_id: str,
    input_path: str,
    file_type: str = "code",
    model_name: Optional[str] = None,
    embedder_name: Optional[str] = None,
    use_cloud: bool = False,
    force: bool = False,
) -> Dict[str, Any]:
    """Chunk, embed and store *input_path* into pgvector.

    Parameters
    ----------
    namespace, project_id : str
        Partitioning keys for the embedding store.
    input_path : str
        Path to a file or directory on disk.
    file_type : str
        ``"code"`` (default) or ``"text"`` — determines the chunker used.
    model_name : str | None
        Deprecated synonym for *embedder_name*.
    embedder_name : str | None
        Explicit embedder choice, e.g. ``"voyage"``, ``"local"``,
        or any HuggingFace model identifier.
    use_cloud : bool
        When *True* use Voyage AI even if the chunk count is below the
        large-codebase threshold.
    force : bool
        When *True* re-embed from scratch even if a prior index exists
        with the same embedder.

    Returns
    -------
    dict
        Summary including embedder used, chunk / vector counts and
        wall-clock time.
    """
    t0 = time.time()

    # ── Initialise Postgres schema ─────────────────────────────────
    db.init_db()

    # ── Resolve embedder ───────────────────────────────────────────
    eff_embedder_name = embedder_name or model_name or "local"
    use_cloud = use_cloud or eff_embedder_name == "voyage"

    # Create semantic embedding cache
    cache = embedder.EmbeddingCache()

    embedder_instance = _resolve_embedder(
        force_embedder=eff_embedder_name,
        use_cloud=use_cloud,
        cache=cache,
    )

    # ── Chunking ───────────────────────────────────────────────────
    logger.info("Chunking %s …", input_path)
    chunks = chunker.chunk_file_or_directory(input_path, file_type=file_type)

    if not chunks:
        logger.warning("No chunks produced from %s — nothing to embed.", input_path)
        return {
            "status": "skipped",
            "namespace": namespace,
            "project_id": project_id,
            "chunks": 0,
            "vectors": 0,
            "embedder": eff_embedder_name,
            "elapsed_seconds": round(time.time() - t0, 2),
        }

    logger.info("Produced %d chunks.", len(chunks))

    # ── Determine effective embedder name for model-dimension tracking
    if isinstance(embedder_instance, embedder.VoyageEmbedder):
        resolved_model = embedder.VOYAGE_MODEL
        resolved_dim = embedder_instance.dim
    else:
        resolved_model = embedder_instance.model_name
        resolved_dim = embedder_instance.dim

    # ── Re-embed if embedder changed ───────────────────────────────
    meta = db.get_index_meta(namespace, project_id)
    if meta and not force:
        prev_embedder = meta.get("embedder", "local")
        prev_dim = meta.get("embedding_dim")
        if (
            prev_embedder != eff_embedder_name
            or (prev_dim is not None and prev_dim != resolved_dim)
        ):
            logger.info(
                "Embedder changed (%s→%s) — clearing old embeddings.",
                prev_embedder,
                eff_embedder_name,
            )
            store = PgVectorStore(namespace, project_id)
            store.clear()

    if force:
        store = PgVectorStore(namespace, project_id)
        store.clear()

    # ── Embedding ──────────────────────────────────────────────────
    logger.info("Embedding %d chunks (model: %s) …", len(chunks), resolved_model)
    embedded_chunks = embedder_instance.embed_chunks(chunks)

    # ── Persist to pgvector ────────────────────────────────────────
    vectors = [c["embedding"] for c in embedded_chunks]
    metas = [
        {k: v for k, v in c.items() if k != "embedding"}
        for c in embedded_chunks
    ]

    store = PgVectorStore(namespace, project_id)
    store.add_vectors(vectors, metas, replace_all=True)

    # ── Update index meta ──────────────────────────────────────────
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    db.upsert_index_meta(namespace, project_id, {
        "embedder": eff_embedder_name,
        "model": resolved_model,
        "embedding_dim": resolved_dim,
        "use_cloud": use_cloud,
    })
    db.update_job_index_meta(namespace, project_id, {
        "embedder": eff_embedder_name,
        "model": resolved_model,
        "embedding_dim": resolved_dim,
        "use_cloud": use_cloud,
        "updated_at": now,
    })

    total_vectors = store.get_total_vectors()

    elapsed = round(time.time() - t0, 2)
    logger.info(
        "Done — %d vectors stored in %.2fs (embedder: %s)",
        total_vectors,
        elapsed,
        eff_embedder_name,
    )

    return {
        "status": "success",
        "namespace": namespace,
        "project_id": project_id,
        "chunks": len(chunks),
        "vectors": total_vectors,
        "embedder": eff_embedder_name,
        "elapsed_seconds": elapsed,
    }


# -------------------------------------------------------------------
# Legacy helper — keep for backward compat with older API routes
# -------------------------------------------------------------------

def update_job_index_meta(namespace: str, project_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Persist embedder metadata onto the jobs table (best-effort)."""
    return db.update_job_index_meta(namespace, project_id, data)
