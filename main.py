import os
import shutil
import zipfile
from datetime import datetime
from uuid import uuid4
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from pydantic import BaseModel, Field
import requests

from indexing.full_pipeline import full_pipeline_chroma
from store.chroma_store import ChromaStore
from lib.embedder import CodeEmbedder, DEFAULT_MODEL

app = FastAPI()

CHROMA_BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), "chroma_data"))
GITHUB_API_BASE = os.getenv("GITHUB_API_BASE", "https://api.github.com")
GITHUB_API_VERSION = os.getenv("GITHUB_API_VERSION", "2022-11-28")
GITHUB_PAT = os.getenv("GITHUB_PAT", "")

# Simple in-memory job tracker for personal use (resets on restart)
JOBS = {}

# Reuse a single embedder instance to avoid reload per request
EMBEDDER = CodeEmbedder(DEFAULT_MODEL)



class SearchRequest(BaseModel):
    namespace: str
    project_id: str
    embedding: List[float] = Field(..., description="Embedding vector to search with")
    k: Optional[int] = Field(5, description="Number of results to return")


class EmbedRequest(BaseModel):
    text: str = Field(..., description="Text to embed using the indexing model")

class GitHubIndexRequest(BaseModel):
    owner: str
    repo: str
    ref: Optional[str] = None
    namespace: Optional[str] = None
    project_id: Optional[str] = None



def _job_update(job_id, **updates):
    job = JOBS.get(job_id, {})
    job.update(updates)
    job["updated_at"] = datetime.utcnow().isoformat() + "Z"
    JOBS[job_id] = job


def _job_create(namespace, project_id, filename):
    job_id = uuid4().hex
    JOBS[job_id] = {
        "job_id": job_id,
        "namespace": namespace,
        "project_id": project_id,
        "filename": filename,
        "status": "queued",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }
    return job_id


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

        store, call_graph, import_graph = full_pipeline_chroma(
            repo_path=repo_root,
            chroma_dir=persist_dir,
            graphs_prefix=os.path.join(persist_dir, "vector_store"),
        )

        _job_update(
            job_id,
            status="completed",
            total_vectors=store.get_total_vectors(),
            persist_dir=persist_dir,
        )
    except Exception as e:
        _job_update(job_id, status="failed", error=str(e))
        raise


@app.get("/")
def read_root():
    return {"status": "ok", "service": "indexer"}

@app.post("/embeddings/{namespace}/{project_id}")
async def post_embeddings(
        namespace: str,
        project_id: str,
        background_tasks: BackgroundTasks,
        code_zip: UploadFile = File(...)
):
    temp_dir = f"/tmp/{namespace}/{project_id}"
    os.makedirs(temp_dir, exist_ok=True)

    zip_path = os.path.join(temp_dir, code_zip.filename)

    with open(zip_path, "wb") as f:
        shutil.copyfileobj(code_zip.file, f)

    job_id = _job_create(namespace, project_id, code_zip.filename)
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


@app.get("/embeddings/{namespace}/{project_id}")
def get_embeddings(namespace: str, project_id: str):
    persist_dir = os.path.join(CHROMA_BASE_DIR, namespace, project_id)
    if not os.path.exists(persist_dir):
        return {"error": "index not found", "namespace": namespace, "project_id": project_id}
    store = ChromaStore(persist_dir=persist_dir)
    return {
        "namespace": namespace,
        "project_id": project_id,
        "total_vectors": store.get_total_vectors(),
        "persist_dir": persist_dir,
    }


@app.post("/search")
def search_embeddings(req: SearchRequest):
    persist_dir = os.path.join(CHROMA_BASE_DIR, req.namespace, req.project_id)
    if not os.path.exists(persist_dir):
        return {"error": "index not found", "namespace": req.namespace, "project_id": req.project_id}
    if not req.embedding:
        return {"error": "embedding is empty"}

    store = ChromaStore(persist_dir=persist_dir)
    results = store.search(req.embedding, k=req.k or 5)
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


@app.post("/embed")
def embed_text(req: EmbedRequest):
    if not req.text.strip():
        return {"error": "text is empty"}
    embedding = EMBEDDER.encode([req.text])[0].tolist()
    return {"embedding": embedding}


@app.post("/github/index")
def github_index(req: GitHubIndexRequest, background_tasks: BackgroundTasks):
    namespace = req.namespace or req.owner
    project_id = req.project_id or req.repo

    temp_dir = f"/tmp/{namespace}/{project_id}"
    os.makedirs(temp_dir, exist_ok=True)

    zip_filename = f"{req.owner}-{req.repo}.zip"
    zip_path = os.path.join(temp_dir, zip_filename)

    job_id = _job_create(namespace, project_id, zip_filename)
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
    uvicorn.run("main:app", reload=True)
