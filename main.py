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
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Form, Request, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import requests

from indexing.full_pipeline import full_pipeline_pgvector
from store.pgvector_store import PgVectorStore
from lib.embedder import CodeEmbedder, VoyageEmbedder, DEFAULT_MODEL
import db

app = FastAPI()

GITHUB_API_BASE = os.getenv("GITHUB_API_BASE", "https://api.github.com")
GITHUB_API_VERSION = os.getenv("GITHUB_API_VERSION", "2022-11-28")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "")

# Jobs are persisted in Postgres (local for dev, Supabase for prod)

# Reuse a single local embedder instance to avoid reload per request
EMBEDDER = CodeEmbedder(DEFAULT_MODEL)


@app.on_event("startup")
def _startup():
    db.init_db()



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
    owner: Optional[str] = None
    repo: Optional[str] = None
    ref: Optional[str] = None
    namespace: Optional[str] = None
    project_id: Optional[str] = None
    webhook_url: Optional[str] = None
    access_token: Optional[str] = Field(None, description="OAuth2 access token for GitHub")
    path: Optional[str] = Field(None, description="Absolute path to a local directory to index instead of fetching from GitHub")


def _require_bearer_token(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization: Bearer token")
    return auth.split(" ", 1)[1].strip()



def _read_index_meta(namespace: str, project_id: str) -> dict:
    """Return index metadata from the database, or {} if missing."""
    return db.get_index_meta(namespace, project_id) or {}


def _job_update(job_id, **updates):
    updates["updated_at"] = datetime.utcnow().isoformat() + "Z"
    return db.update_job(job_id, **updates)


def _job_create(namespace, project_id, filename, webhook_url=None, source=None, owner=None, repo=None, ref=None):
    job_id = uuid4().hex
    created_at = datetime.utcnow().isoformat() + "Z"
    db.create_job(
        job_id=job_id,
        namespace=namespace,
        project_id=project_id,
        filename=filename,
        status="queued",
        created_at=created_at,
        updated_at=created_at,
        webhook_url=webhook_url,
        source=source,
        owner=owner,
        repo=repo,
        ref=ref,
    )
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


def _resolve_github_token(req_token: Optional[str], auth_header: Optional[str]) -> Optional[str]:
    """Return the GitHub token if one was provided, or None for public-repo access."""
    if req_token:
        return req_token
    if auth_header and auth_header.lower().startswith("bearer "):
        return auth_header.split(" ", 1)[1].strip()
    return None


def _github_headers(token: Optional[str]) -> dict:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": GITHUB_API_VERSION,
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _github_download_zip(token: Optional[str], owner: str, repo: str, ref: Optional[str], zip_path: str) -> None:
    if ref:
        url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/zipball/{ref}"
    else:
        url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/zipball"
    with requests.get(url, headers=_github_headers(token), stream=True, allow_redirects=True, timeout=60) as r:
        if r.status_code not in (200, 302):
            raise RuntimeError(f"GitHub zip download error {r.status_code}: {r.text}")
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def _github_validate_access(token: Optional[str], owner: str, repo: str) -> None:
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}"
    resp = requests.get(url, headers=_github_headers(token), timeout=15)
    if resp.status_code == 200:
        return
    if resp.status_code == 404 and not token:
        raise HTTPException(
            status_code=404,
            detail=f"Repository {owner}/{repo} not found or is private. Provide an access_token to index private repositories.",
        )
    raise HTTPException(status_code=resp.status_code, detail=f"GitHub auth failed: {resp.text}")




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

        store, call_graph, import_graph, index_meta = full_pipeline_pgvector(
            repo_path=repo_root,
            namespace=namespace,
            project_id=project_id,
        )
        db.upsert_index_meta(namespace, project_id, index_meta)

        job = _job_update(
            job_id,
            status="completed",
            total_vectors=store.get_total_vectors(),
            persist_dir="pgvector",
            embedder=index_meta.get("embedder"),
        )
        _notify_webhook(job)
    except Exception as e:
        job = _job_update(job_id, status="failed", error=str(e))
        _notify_webhook(job)
        raise


@app.get("/")
def read_root():
    return FileResponse(os.path.join(os.path.dirname(__file__), "web", "index.html"))

@app.post("/embeddings/{namespace}/{project_id}")
async def post_embeddings(
        request: Request,
        namespace: str,
        project_id: str,
        background_tasks: BackgroundTasks,
        code_zip: UploadFile = File(...),
        webhook_url: Optional[str] = Form(None),
):
    _require_bearer_token(request)
    temp_dir = f"/tmp/{namespace}/{project_id}"
    os.makedirs(temp_dir, exist_ok=True)

    zip_path = os.path.join(temp_dir, code_zip.filename)

    with open(zip_path, "wb") as f:
        shutil.copyfileobj(code_zip.file, f)

    job_id = _job_create(namespace, project_id, code_zip.filename, webhook_url=webhook_url, source="upload")
    background_tasks.add_task(process_zip, temp_dir, zip_path, namespace, project_id, job_id)

    return {
        "status": "queued",
        "job_id": job_id,
        "namespace": namespace,
        "project_id": project_id
    }

@app.get("/jobs/{job_id}")
def get_job(job_id: str, request: Request):
    _require_bearer_token(request)
    job = db.get_job(job_id)
    if not job:
        return {"error": "job not found"}
    return job


@app.delete("/embeddings/{namespace}/{project_id}")
def delete_embeddings(namespace: str, project_id: str, request: Request):
    _require_bearer_token(request)
    db.delete_embeddings(namespace, project_id)

    # Also clean up any leftover temp upload files for this project.
    temp_dir = f"/tmp/{namespace}/{project_id}"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)

    return {"deleted": True, "namespace": namespace, "project_id": project_id}


@app.get("/embeddings/{namespace}/{project_id}")
def get_embeddings(namespace: str, project_id: str, request: Request):
    _require_bearer_token(request)
    store = PgVectorStore(namespace=namespace, project_id=project_id)
    index_meta = _read_index_meta(namespace, project_id)
    return {
        "namespace": namespace,
        "project_id": project_id,
        "total_vectors": store.get_total_vectors(),
        "persist_dir": "pgvector",
        **index_meta,
    }


@app.post("/search")
def search_embeddings(req: SearchRequest):
    if not req.embedding:
        return {"error": "embedding is empty"}

    index_meta = _read_index_meta(req.namespace, req.project_id)
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

    store = PgVectorStore(namespace=req.namespace, project_id=req.project_id)
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

    index_meta = _read_index_meta(req.namespace, req.project_id)

    if index_meta.get("use_cloud"):
        if not VOYAGE_API_KEY:
            return {"error": "VOYAGE_API_KEY is not set on this server"}
        embedding = VoyageEmbedder(api_key=VOYAGE_API_KEY, input_type="query").encode([req.query])[0].tolist()
    else:
        embedding = EMBEDDER.encode([req.query])[0].tolist()

    store = PgVectorStore(namespace=req.namespace, project_id=req.project_id)
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
        meta = _read_index_meta(req.namespace, req.project_id)
        use_cloud = bool(meta.get("use_cloud", False))

    if use_cloud:
        if not VOYAGE_API_KEY:
            return {"error": "VOYAGE_API_KEY is not set on this server"}
        embedder = VoyageEmbedder(api_key=VOYAGE_API_KEY, input_type="query")
        embedding = embedder.encode([req.text])[0].tolist()
        return {"embedding": embedding, "model": "voyage-code-3"}

    embedding = EMBEDDER.encode([req.text])[0].tolist()
    return {"embedding": embedding, "model": DEFAULT_MODEL}


def _process_local_path(repo_path: str, namespace: str, project_id: str, job_id: str):
    try:
        _job_update(job_id, status="running")
        store, call_graph, import_graph, index_meta = full_pipeline_pgvector(
            repo_path=repo_path,
            namespace=namespace,
            project_id=project_id,
        )
        db.upsert_index_meta(namespace, project_id, index_meta)
        job = _job_update(
            job_id,
            status="completed",
            total_vectors=store.get_total_vectors(),
            persist_dir="pgvector",
            embedder=index_meta.get("embedder"),
        )
        _notify_webhook(job)
    except Exception as e:
        job = _job_update(job_id, status="failed", error=str(e))
        _notify_webhook(job)
        raise


@app.post("/github/index")
def github_index(req: GitHubIndexRequest, background_tasks: BackgroundTasks, request: Request):
    # --- Local path branch ---
    if req.path:
        if not os.path.isdir(req.path):
            raise HTTPException(status_code=400, detail=f"Local path does not exist or is not a directory: {req.path}")
        folder_name = os.path.basename(req.path.rstrip("/"))
        namespace = req.namespace or folder_name
        project_id = req.project_id or folder_name
        job_id = _job_create(namespace, project_id, req.path, webhook_url=req.webhook_url, source="local")
        _job_update(job_id, status="queued", source="local")
        background_tasks.add_task(_process_local_path, req.path, namespace, project_id, job_id)
        return {"status": "queued", "job_id": job_id, "namespace": namespace, "project_id": project_id}

    # --- GitHub branch ---
    if not req.owner or not req.repo:
        raise HTTPException(status_code=400, detail="Provide either 'path' (local directory) or both 'owner' and 'repo' (GitHub repository)")

    token = _resolve_github_token(req.access_token, request.headers.get("authorization"))
    _github_validate_access(token, req.owner, req.repo)
    namespace = req.namespace or req.owner
    project_id = req.project_id or req.repo

    temp_dir = f"/tmp/{namespace}/{project_id}"
    os.makedirs(temp_dir, exist_ok=True)

    zip_filename = f"{req.owner}-{req.repo}.zip"
    zip_path = os.path.join(temp_dir, zip_filename)

    job_id = _job_create(
        namespace,
        project_id,
        zip_filename,
        webhook_url=req.webhook_url,
        source="github",
        owner=req.owner,
        repo=req.repo,
        ref=req.ref,
    )
    _job_update(job_id, status="queued", source="github", owner=req.owner, repo=req.repo, ref=req.ref)

    def _download_and_process():
        try:
            _job_update(job_id, status="downloading")
            _github_download_zip(token, req.owner, req.repo, req.ref, zip_path)
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
    db.init_db()
    uvicorn.run("main:app", reload=True)
