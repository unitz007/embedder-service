import os
import shutil
import zipfile
import json
import hmac
import hashlib
from datetime import datetime
from uuid import uuid4
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Form
from pydantic import BaseModel, Field
import requests

from pathlib import Path

from indexing.full_pipeline import full_pipeline_chroma
from store.chroma_store import ChromaStore, reset_chroma_dir
from lib.embedder import CodeEmbedder, VoyageEmbedder, DEFAULT_MODEL

app = FastAPI()

CHROMA_BASE_DIR = os.getenv("CHROMA_DIR") or os.path.abspath(os.path.join(os.getcwd(), "chroma_data"))
GITHUB_API_BASE = os.getenv("GITHUB_API_BASE", "https://api.github.com")
GITHUB_API_VERSION = os.getenv("GITHUB_API_VERSION", "2022-11-28")
GITHUB_PAT = os.getenv("GITHUB_PAT", "")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "")

# Simple in-memory job tracker for personal use (resets on restart)
JOBS = {}

# Reuse a single local embedder instance to avoid reload per request
EMBEDDER = CodeEmbedder(DEFAULT_MODEL)



class SearchRequest(BaseModel):
    namespace: str
    project_id: str
    embedding: List[float] = Field(..., description="Embedding vector to search with")
    k: Optional[int] = Field(5, description="Number of results to return")
    text: Optional[str] = Field(
        None,
        description=(
            "Original query text. When the embedding dimension does not match "
            "the index the server re-embeds this text with the correct model "
            "automatically, so the caller does not need to know which embedder "
            "was used at index time."
        ),
    )


class TextSearchRequest(BaseModel):
    namespace: str
    project_id: str
    query: str = Field(..., description="Raw query text to search with")
    k: Optional[int] = Field(5, description="Number of results to return")


class EmbedRequest(BaseModel):
    text: str = Field(..., description="Text to embed using the indexing model")
    use_cloud: bool = Field(
        False,
        description=(
            "Set to true to embed with Voyage AI (voyage-code-3). "
            "Required when the target index was built from a large codebase."
        ),
    )
    namespace: Optional[str] = Field(
        None,
        description=(
            "When provided together with project_id the server looks up the "
            "index metadata and automatically selects the correct embedding "
            "model, so use_cloud does not need to be set manually."
        ),
    )
    project_id: Optional[str] = Field(None, description="See namespace.")

class GitHubIndexRequest(BaseModel):
    owner: str
    repo: str
    ref: Optional[str] = None
    namespace: Optional[str] = None
    project_id: Optional[str] = None
    webhook_url: Optional[str] = None



def _read_index_meta(persist_dir: str) -> dict:
    """Return the index_meta.json for a persisted index, or {} if missing."""
    meta_path = Path(persist_dir) / "index_meta.json"
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _job_update(job_id, **updates):
    job = JOBS.get(job_id, {})
    job.update(updates)
    job["updated_at"] = datetime.utcnow().isoformat() + "Z"
    JOBS[job_id] = job
    return job


def _job_create(namespace, project_id, filename, webhook_url=None):
    job_id = uuid4().hex
    JOBS[job_id] = {
        "job_id": job_id,
        "namespace": namespace,
        "project_id": project_id,
        "filename": filename,
        "status": "queued",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "webhook_url": webhook_url,
    }
    return job_id


def _notify_webhook(job: dict) -> None:
    webhook_url = job.get("webhook_url")
    if not webhook_url:
        return
    payload = json.dumps(job)
    headers = {"Content-Type": "application/json"}
    if WEBHOOK_SECRET:
        sig = hmac.new(WEBHOOK_SECRET.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()
        headers["X-Indexer-Signature"] = f"sha256={sig}"
    try:
        requests.post(webhook_url, data=payload, headers=headers, timeout=10)
    except Exception as e:
        print(f"Webhook notify failed: {e}")


def _require_pat() -> str:
    if not GITHUB_PAT:
        raise RuntimeError("GITHUB_PAT is not set")
    return GITHUB_PAT


def _github_download_zip(pat: str, owner: str, repo: str, ref: Optional[str], zip_path: str) -> None:
    if ref:
        url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/zipball/{ref}"
    else:
        url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/zipball"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {pat}",
        "X-GitHub-Api-Version": GITHUB_API_VERSION,
    }
    with requests.get(url, headers=headers, stream=True, allow_redirects=True, timeout=60) as r:
        if r.status_code not in (200, 302):
            raise RuntimeError(f"GitHub zip download error {r.status_code}: {r.text}")
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)




def safe_extract(zip_ref, path):
    for member in zip_ref.namelist():
        member_path = os.path.abspath(os.path.join(path, member))
        if not member_path.startswith(os.path.abspath(path)):
            raise Exception("Unsafe zip file detected")

    zip_ref.extractall(path)


def get_repo_root(path):
    items = os.listdir(path)
    if len(items) == 1:
        return os.path.join(path, items[0])
    return path


def process_zip(temp_dir, zip_path, namespace, project_id, job_id):
    extract_path = os.path.join(temp_dir, "extracted")

    try:
        _job_update(job_id, status="running")

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            safe_extract(zip_ref, extract_path)

        repo_root = get_repo_root(extract_path)
        persist_dir = os.path.join(CHROMA_BASE_DIR, namespace, project_id)
        os.makedirs(persist_dir, exist_ok=True)

        store, call_graph, import_graph, embedder_name = full_pipeline_chroma(
            repo_path=repo_root,
            chroma_dir=persist_dir,
            graphs_prefix=os.path.join(persist_dir, "vector_store"),
        )

        job = _job_update(
            job_id,
            status="completed",
            total_vectors=store.get_total_vectors(),
            persist_dir=persist_dir,
            embedder=embedder_name,
        )
        _notify_webhook(job)
    except Exception as e:
        job = _job_update(job_id, status="failed", error=str(e))
        _notify_webhook(job)
        raise


@app.get("/")
def read_root():
    return {"status": "ok", "service": "indexer"}

@app.post("/embeddings/{namespace}/{project_id}")
async def post_embeddings(
        namespace: str,
        project_id: str,
        background_tasks: BackgroundTasks,
        code_zip: UploadFile = File(...),
        webhook_url: Optional[str] = Form(None),
):
    temp_dir = f"/tmp/{namespace}/{project_id}"
    os.makedirs(temp_dir, exist_ok=True)

    zip_path = os.path.join(temp_dir, code_zip.filename)

    with open(zip_path, "wb") as f:
        shutil.copyfileobj(code_zip.file, f)

    job_id = _job_create(namespace, project_id, code_zip.filename, webhook_url=webhook_url)
    background_tasks.add_task(process_zip, temp_dir, zip_path, namespace, project_id, job_id)

    return {
        "status": "queued",
        "job_id": job_id,
        "namespace": namespace,
        "project_id": project_id
    }

@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return {"error": "job not found"}
    return job


@app.delete("/embeddings/{namespace}/{project_id}")
def delete_embeddings(namespace: str, project_id: str):
    persist_dir = os.path.join(CHROMA_BASE_DIR, namespace, project_id)
    if not os.path.exists(persist_dir):
        return {"error": "index not found", "namespace": namespace, "project_id": project_id}

    # reset_chroma_dir evicts the client cache AND calls
    # SharedSystemClient.clear_system_cache() to stop ChromaDB's background
    # threads (WAL checkpoint, segment GC).  Without that second step, those
    # threads write back into the directory moments after rmtree, causing
    # SQLITE_READONLY_DBMOVED (code 1032) on the next operation.
    # reset_chroma_dir recreates the directory at the end, so we remove it again.
    reset_chroma_dir(persist_dir)
    shutil.rmtree(persist_dir, ignore_errors=True)

    # Also clean up any leftover temp upload files for this project.
    temp_dir = f"/tmp/{namespace}/{project_id}"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)

    return {"deleted": True, "namespace": namespace, "project_id": project_id}


@app.get("/embeddings/{namespace}/{project_id}")
def get_embeddings(namespace: str, project_id: str):
    persist_dir = os.path.join(CHROMA_BASE_DIR, namespace, project_id)
    if not os.path.exists(persist_dir):
        return {"error": "index not found", "namespace": namespace, "project_id": project_id}
    store = ChromaStore(persist_dir=persist_dir)
    index_meta = _read_index_meta(persist_dir)
    return {
        "namespace": namespace,
        "project_id": project_id,
        "total_vectors": store.get_total_vectors(),
        "persist_dir": persist_dir,
        **index_meta,
    }


@app.post("/search")
def search_embeddings(req: SearchRequest):
    persist_dir = os.path.join(CHROMA_BASE_DIR, req.namespace, req.project_id)
    if not os.path.exists(persist_dir):
        return {"error": "index not found", "namespace": req.namespace, "project_id": req.project_id}
    if not req.embedding:
        return {"error": "embedding is empty"}

    index_meta = _read_index_meta(persist_dir)
    expected_dim = index_meta.get("embedding_dim")
    embedding = req.embedding

    if expected_dim and len(embedding) != expected_dim:
        # Dimension mismatch — the caller embedded with the wrong model.
        # Rather than returning an error, re-embed the query text here if
        # the search request carries it; otherwise surface a clear message.
        if not req.text:
            return {
                "error": (
                    f"Embedding dimension mismatch: index was built with "
                    f"{index_meta.get('embedder', 'unknown')} "
                    f"({index_meta.get('model', '?')}, dim={expected_dim}) "
                    f"but the query has dim={len(embedding)}. "
                    f"Re-embed your query with use_cloud="
                    f"{'true' if index_meta.get('use_cloud') else 'false'} "
                    f"or pass the raw text in the 'text' field."
                )
            }
        if index_meta.get("use_cloud"):
            if not VOYAGE_API_KEY:
                return {"error": "VOYAGE_API_KEY is not set on this server"}
            embedding = VoyageEmbedder(api_key=VOYAGE_API_KEY, input_type="query").encode([req.text])[0].tolist()
        else:
            embedding = EMBEDDER.encode([req.text])[0].tolist()

    store = ChromaStore(persist_dir=persist_dir)
    results = store.search(embedding, k=req.k or 5)
    return {
        "namespace": req.namespace,
        "project_id": req.project_id,
        "k": req.k or 5,
        "results": [
            {
                "metadata": meta,
                "similarity_score": score,
                "rank": rank + 1,
            }
            for rank, (meta, score, _) in enumerate(results)
        ],
    }


@app.post("/search/text")
def search_by_text(req: TextSearchRequest):
    """Search using raw query text — the server picks the correct embedding model
    automatically by reading the index metadata. No pre-computed embedding needed."""
    if not req.query.strip():
        return {"error": "query is empty"}

    persist_dir = os.path.join(CHROMA_BASE_DIR, req.namespace, req.project_id)
    if not os.path.exists(persist_dir):
        return {"error": "index not found", "namespace": req.namespace, "project_id": req.project_id}

    index_meta = _read_index_meta(persist_dir)

    if index_meta.get("use_cloud"):
        if not VOYAGE_API_KEY:
            return {"error": "VOYAGE_API_KEY is not set on this server"}
        embedding = VoyageEmbedder(api_key=VOYAGE_API_KEY, input_type="query").encode([req.query])[0].tolist()
    else:
        embedding = EMBEDDER.encode([req.query])[0].tolist()

    store = ChromaStore(persist_dir=persist_dir)
    results = store.search(embedding, k=req.k or 5)
    return {
        "namespace": req.namespace,
        "project_id": req.project_id,
        "k": req.k or 5,
        "embedder": index_meta.get("embedder", "unknown"),
        "results": [
            {
                "metadata": meta,
                "similarity_score": score,
                "rank": rank + 1,
            }
            for rank, (meta, score, _) in enumerate(results)
        ],
    }


@app.post("/embed")
def embed_text(req: EmbedRequest):
    if not req.text.strip():
        return {"error": "text is empty"}

    # Auto-detect the right model from the project's index metadata so the
    # caller doesn't have to set use_cloud manually.
    use_cloud = req.use_cloud
    if not use_cloud and req.namespace and req.project_id:
        persist_dir = os.path.join(CHROMA_BASE_DIR, req.namespace, req.project_id)
        meta = _read_index_meta(persist_dir)
        use_cloud = bool(meta.get("use_cloud", False))

    if use_cloud:
        if not VOYAGE_API_KEY:
            return {"error": "VOYAGE_API_KEY is not set on this server"}
        embedder = VoyageEmbedder(api_key=VOYAGE_API_KEY, input_type="query")
        embedding = embedder.encode([req.text])[0].tolist()
        return {"embedding": embedding, "model": "voyage-code-3"}

    embedding = EMBEDDER.encode([req.text])[0].tolist()
    return {"embedding": embedding, "model": DEFAULT_MODEL}


@app.post("/github/index")
def github_index(req: GitHubIndexRequest, background_tasks: BackgroundTasks):
    namespace = req.namespace or req.owner
    project_id = req.project_id or req.repo

    temp_dir = f"/tmp/{namespace}/{project_id}"
    os.makedirs(temp_dir, exist_ok=True)

    zip_filename = f"{req.owner}-{req.repo}.zip"
    zip_path = os.path.join(temp_dir, zip_filename)

    job_id = _job_create(namespace, project_id, zip_filename, webhook_url=req.webhook_url)
    _job_update(job_id, status="queued", source="github", owner=req.owner, repo=req.repo, ref=req.ref)

    def _download_and_process():
        try:
            _job_update(job_id, status="downloading")
            pat = _require_pat()
            _github_download_zip(pat, req.owner, req.repo, req.ref, zip_path)
            process_zip(temp_dir, zip_path, namespace, project_id, job_id)
        except Exception as e:
            _job_update(job_id, status="failed", error=str(e))
            raise

    background_tasks.add_task(_download_and_process)
    return {
        "status": "queued",
        "job_id": job_id,
        "namespace": namespace,
        "project_id": project_id,
    }


if __name__ == "__main__":
    uvicorn.run("main:app", reload=False)
