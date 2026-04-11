# main.py
"""FastAPI application for code-indexing service."""

import json
import os
import uuid
import shutil
from datetime import datetime, timezone
from typing import Optional

import psycopg2
from fastapi import FastAPI, Request, HTTPException, Path as PathParam
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import db
from indexing.full_pipeline import full_pipeline_pgvector

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class IndexRequest(BaseModel):
    namespace: str
    project_id: str
    source: Optional[str] = None
    owner: Optional[str] = None
    repo: Optional[str] = None
    ref: Optional[str] = None
    webhook_url: Optional[str] = None
    include_dotfiles: Optional[bool] = False


class EmbedderConfig(BaseModel):
    preferred_embedder: Optional[str] = Field(
        None,
        description=(
            "HuggingFace model name or 'voyage-code-3' / any string "
            "containing 'voyage'. Set to null to clear and revert to "
            "the default size-based heuristic."
        ),
    )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Code Indexer",
    version="0.1.0",
    description="Index code repositories into pgvector for semantic search.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PERSIST_ROOT = os.getenv("PERSIST_ROOT", "/tmp/indexer")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _clone_dir(namespace: str, project_id: str) -> str:
    return os.path.join(PERSIST_ROOT, namespace, project_id)


def _ensure_clone_dir(namespace: str, project_id: str) -> str:
    d = _clone_dir(namespace, project_id)
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Webhook delivery
# ---------------------------------------------------------------------------

def _deliver_webhook(url: str, payload: dict) -> None:
    import requests as http_requests
    try:
        http_requests.post(url, json=payload, timeout=10)
    except Exception:
        pass  # best-effort


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
def on_startup() -> None:
    db.init_db()
    os.makedirs(PERSIST_ROOT, exist_ok=True)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/index", status_code=202)
async def start_indexing(req: IndexRequest):
    """Start an indexing job (synchronous for now)."""
    job_id = uuid.uuid4().hex[:12]
    now = _now_iso()

    persist_dir = _ensure_clone_dir(req.namespace, req.project_id)

    job = db.create_job(
        job_id=job_id,
        namespace=req.namespace,
        project_id=req.project_id,
        filename="full",
        status="running",
        created_at=now,
        updated_at=now,
        webhook_url=req.webhook_url,
        source=req.source,
        owner=req.owner,
        repo=req.repo,
        ref=req.ref,
    )

    # ------------------------------------------------------------------
    # Run pipeline
    # ------------------------------------------------------------------
    try:
        # Load tenant embedder preference before indexing
        existing_meta = db.get_index_meta(req.namespace, req.project_id)
        preferred_embedder = existing_meta.get("preferred_embedder")

        store, call_graph, import_graph, index_meta = full_pipeline_pgvector(
            repo_path=persist_dir,
            namespace=req.namespace,
            project_id=req.project_id,
            include_dotfiles=req.include_dotfiles or False,
            preferred_embedder=preferred_embedder,
        )

        # Persist index_meta to DB (includes actual embedder used)
        db.upsert_index_meta(req.namespace, req.project_id, index_meta)

        total_vectors = db.count_embeddings(req.namespace, req.project_id)

        job = db.update_job(
            job_id,
            status="completed",
            total_vectors=total_vectors,
            persist_dir=persist_dir,
            embedder=index_meta.get("embedder"),
        )

        if req.webhook_url:
            _deliver_webhook(req.webhook_url, {
                "event": "index.completed",
                "job_id": job_id,
                "namespace": req.namespace,
                "project_id": req.project_id,
                "total_vectors": total_vectors,
            })

        return job

    except Exception as exc:
        error_msg = str(exc)[:2000]
        job = db.update_job(
            job_id,
            status="failed",
            error=error_msg,
        )

        if req.webhook_url:
            _deliver_webhook(req.webhook_url, {
                "event": "index.failed",
                "job_id": job_id,
                "namespace": req.namespace,
                "project_id": req.project_id,
                "error": error_msg,
            })

        raise HTTPException(status_code=500, detail=job)


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.delete("/embeddings/{namespace}/{project_id}")
def delete_project(
    namespace: str,
    project_id: str,
):
    """Delete all embeddings and metadata for a project."""
    db.delete_embeddings(namespace, project_id)
    clone_dir = _clone_dir(namespace, project_id)
    if os.path.isdir(clone_dir):
        shutil.rmtree(clone_dir, ignore_errors=True)
    return {"status": "deleted", "namespace": namespace, "project_id": project_id}


# ---------------------------------------------------------------------------
# Tenant embedder configuration
# ---------------------------------------------------------------------------

@app.get("/embeddings/{namespace}/{project_id}/config")
def get_embedder_config(
    namespace: str,
    project_id: str,
):
    """Return the current embedder config for a tenant project.

    The response includes the ``preferred_embedder`` (tenant override) as
    well as the ``embedder``, ``model``, and ``embedding_dim`` that were
    actually used during the most recent indexing run.
    """
    meta = db.get_index_meta(namespace, project_id)
    if not meta:
        return {
            "namespace": namespace,
            "project_id": project_id,
            "preferred_embedder": None,
            "embedder": None,
            "model": None,
            "embedding_dim": None,
            "use_cloud": None,
            "last_indexed_at": None,
        }
    return {
        "namespace": meta["namespace"],
        "project_id": meta["project_id"],
        "preferred_embedder": meta.get("preferred_embedder"),
        "embedder": meta.get("embedder"),
        "model": meta.get("model"),
        "embedding_dim": meta.get("embedding_dim"),
        "use_cloud": meta.get("use_cloud"),
        "last_indexed_at": meta.get("updated_at"),
    }


@app.put("/embeddings/{namespace}/{project_id}/config")
def set_embedder_config(
    namespace: str,
    project_id: str,
    body: EmbedderConfig,
):
    """Set (or clear) the preferred embedder for a tenant project.

    * **Voyage AI** — set ``preferred_embedder`` to any string containing
      ``"voyage"`` (e.g. ``"voyage-code-3"``).
    * **Custom local model** — set it to a HuggingFace model identifier
      (e.g. ``"microsoft/codebert-base"``).
    * **Revert to automatic** — set ``preferred_embedder`` to ``null``.

    The preference takes effect on the *next* indexing run.
    """
    result = db.update_tenant_embedder_config(
        namespace=namespace,
        project_id=project_id,
        preferred_embedder=body.preferred_embedder,
    )
    return result
