import os
import uuid
import json
import time
import threading
import traceback
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Any

from flask import Flask, request, jsonify, g
from flask_cors import CORS

import db

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

try:
    from config import (
        DEFAULT_NAMESPACE,
        API_KEY,
        UNAUTHORIZED_RESPONSE,
        LOCAL_INDEXER_BASE,
    )
except ImportError:
    # Graceful fallback for local development
    DEFAULT_NAMESPACE = None
    API_KEY = None
    UNAUTHORIZED_RESPONSE = {"error": "Unauthorized"}
    LOCAL_INDEXER_BASE = None

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ---------------------------------------------------------------------------
# Background-job bookkeeping
# ---------------------------------------------------------------------------

_jobs: dict[str, dict[str, Any]] = {}


def _record_job(job_id: str, status: str, **extra):
    _jobs[job_id] = {"job_id": job_id, "status": status, **extra}


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

@app.before_request
def _auth():
    if API_KEY and request.path.startswith("/api/"):
        key = request.headers.get("Authorization", "").replace("Bearer ", "")
        if key != API_KEY:
            return jsonify(UNAUTHORIZED_RESPONSE), 401


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/api/jobs", methods=["GET"])
def list_jobs():
    try:
        rows = db.get_jobs()
        return jsonify(rows)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/jobs/<job_id>", methods=["GET"])
def get_job(job_id: str):
    job = db.get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/api/jobs/<job_id>/status", methods=["GET"])
def get_job_status(job_id: str):
    job = db.get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify({"job_id": job_id, "status": job["status"]})


@app.route("/api/index", methods=["POST"])
def start_indexing():
    data = request.get_json(silent=True) or {}
    source = data.get("source", "git")
    namespace = data.get("namespace") or DEFAULT_NAMESPACE or "default"
    project_id = data.get("project_id")
    url = data.get("url")
    ref = data.get("ref", "HEAD")
    webhook_url = data.get("webhook_url")

    if not project_id or not url:
        return jsonify({"error": "project_id and url required"}), 400

    job_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat() + "Z"
    db.create_job(
        job_id=job_id,
        namespace=namespace,
        project_id=project_id,
        filename=url,
        status="queued",
        created_at=now,
        updated_at=now,
        webhook_url=webhook_url,
        source=source,
    )
    _record_job(job_id, "queued")

    thread = threading.Thread(
        target=_run_index_job,
        args=(job_id, source, namespace, project_id, url, ref, webhook_url, data),
        daemon=True,
    )
    thread.start()
    return jsonify({"job_id": job_id, "status": "queued"})


@app.route("/api/index/<namespace>/<project_id>/model", methods=["GET"])
def get_index_model(namespace: str, project_id: str):
    """Return the current embedder model stored for a tenant's index."""
    try:
        meta = db.get_index_meta(namespace, project_id)
        if not meta:
            return jsonify({"error": "No index found for this project"}), 404
        return jsonify(meta)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/index/<namespace>/<project_id>/model", methods=["PUT"])
def set_index_model(namespace: str, project_id: str):
    """Override (or clear) the preferred embedding model for a tenant.

    Body:
        preferred_embedder: string | null
            - ``"voyage-code-3"`` → Voyage AI cloud embedder
            - ``"microsoft/codebert-base"`` → local CodeBERT embedder
            - ``null`` → revert to automatic selection
    """
    data = request.get_json(silent=True) or {}
    preferred = data.get("preferred_embedder")

    if preferred is not None and not isinstance(preferred, str):
        return jsonify({"error": "preferred_embedder must be a string or null"}), 400

    if preferred:
        preferred = preferred.strip() or None

    try:
        meta = db.update_preferred_embedder(namespace, project_id, preferred)
        return jsonify(meta)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/search", methods=["POST"])
def search_code():
    data = request.get_json(silent=True) or {}
    namespace = data.get("namespace") or DEFAULT_NAMESPACE or "default"
    project_id = data.get("project_id")
    query = data.get("query")
    top_k = data.get("top_k", 5)

    if not project_id or not query:
        return jsonify({"error": "project_id and query required"}), 400

    try:
        # Lazy import so the heavy ML libs are only loaded at search time
        from store.pgvector_store import PgVectorStore
        from lib.embedder import CodeEmbedder, VoyageEmbedder
        from utils.context_builder import ContextBuilder

        meta = db.get_index_meta(namespace, project_id)
        if not meta:
            return jsonify({"error": "No index found for this project"}), 404

        is_cloud = bool(meta.get("use_cloud"))
        if is_cloud:
            embedder = VoyageEmbedder()
        else:
            model_name = meta.get("model", "microsoft/codebert-base")
            embedder = CodeEmbedder(model_name)

        query_embedding = embedder.encode([query])[0].tolist()

        store = PgVectorStore(namespace=namespace, project_id=project_id)
        results = store.search(query_embedding, k=top_k)

        ctx = ContextBuilder()
        ctx.store = store
        hits = [
            {
                "file_path": r.get("file_path", ""),
                "chunk_type": r.get("chunk_type", ""),
                "symbol_name": r.get("symbol_name", ""),
                "line_number": r.get("line_number", 0),
                "content": r.get("content", ""),
                "metadata": r.get("metadata", {}),
                "score": float(score),
            }
            for r, score in results
        ]

        return jsonify({"results": hits, "total": len(hits)})
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# Background indexing job
# ---------------------------------------------------------------------------

def _run_index_job(
    job_id: str,
    source: str,
    namespace: str,
    project_id: str,
    url: str,
    ref: str,
    webhook_url: Optional[str],
    data: dict,
):
    now = datetime.utcnow().isoformat() + "Z"
    db.update_job(job_id, status="running", updated_at=now)
    _record_job(job_id, "running")

    tmp_dir: Optional[str] = None

    try:
        # Clone repo
        tmp_dir = tempfile.mkdtemp(prefix="indexer_")
        _clone_repo(url, ref, tmp_dir)

        # Determine preferred embedder
        preferred_embedder = data.get("preferred_embedder")
        if preferred_embedder and not isinstance(preferred_embedder, str):
            preferred_embedder = None

        # Run the pgvector pipeline
        from indexing.full_pipeline import full_pipeline_pgvector
        store, call_graph, import_graph, index_meta = full_pipeline_pgvector(
            repo_path=tmp_dir,
            namespace=namespace,
            project_id=project_id,
            preferred_embedder=preferred_embedder,
        )

        # Persist index metadata
        db.upsert_index_meta(namespace, project_id, index_meta)

        # Record stats
        total_vectors = store.count()
        now = datetime.utcnow().isoformat() + "Z"
        db.update_job(
            job_id,
            status="complete",
            total_vectors=total_vectors,
            embedder=index_meta.get("embedder"),
            updated_at=now,
        )
        _record_job(job_id, "complete", total_vectors=total_vectors)

        # Fire webhook
        if webhook_url:
            import requests as _requests
            try:
                _requests.post(
                    webhook_url,
                    json={
                        "job_id": job_id,
                        "status": "complete",
                        "namespace": namespace,
                        "project_id": project_id,
                        "total_vectors": total_vectors,
                    },
                    timeout=10,
                )
            except Exception:
                pass

    except Exception as exc:
        traceback.print_exc()
        now = datetime.utcnow().isoformat() + "Z"
        db.update_job(job_id, status="failed", error=str(exc), updated_at=now)
        _record_job(job_id, "failed", error=str(exc))

    finally:
        if tmp_dir and os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)


def _clone_repo(url: str, ref: str, dest: str):
    """Clone a git repository into *dest* at the given *ref*."""
    # Delegate to an external indexer service when configured
    if LOCAL_INDEXER_BASE:
        import requests as _requests
        resp = _requests.post(
            f"{LOCAL_INDEXER_BASE}/api/clone",
            json={"url": url, "ref": ref},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        path = data.get("path")
        if not path or not os.path.isdir(path):
            raise RuntimeError(f"Clone service returned invalid path: {path}")
        # Copy into our temp dir so the caller can safely delete it
        shutil.copytree(path, dest, dirs_exist_ok=True)
        return

    import subprocess
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", ref, url, dest],
        check=True,
        timeout=300,
    )


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8001"))
    app.run(host="0.0.0.0", port=port, debug=False)
