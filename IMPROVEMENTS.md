# Production-Grade Improvements

> **Status**: Proposed · **Scope**: Documentation-only (no code changes)
>
> A comprehensive audit of the embedder-service codebase identifying 20 novel,
> production-grade improvements organised by domain. Each item references real
> file paths, describes the current limitation, and proposes a concrete change.

---

## Table of Contents

- [1. Reliability & Resilience](#1-reliability--resilience)
  - [1.1 Incremental Indexing via File-Hash Tracking](#11-incremental-indexing-via-file-hash-tracking)
  - [1.2 Durable Background Task Processing](#12-durable-background-task-processing)
  - [1.3 Connection Pool Lifecycle Management](#13-connection-pool-lifecycle-management)
  - [1.4 Retry with Exponential Backoff for Voyage AI](#14-retry-with-exponential-backoff-for-voyage-ai)
- [2. Security](#2-security)
  - [2.1 Zip Slip & Symlink Hardening in `safe_extract`](#21-zip-slip--symlink-hardening-in-safe_extract)
  - [2.2 Zip Bomb Protection via Uncompressed-Size Cap](#22-zip-bomb-protection-via-uncompressed-size-cap)
  - [2.3 Per-Client Rate Limiting](#23-per-client-rate-limiting)
  - [2.4 API Key Scoping and Least-Privilege Tokens](#24-api-key-scoping-and-least-privilege-tokens)
- [3. Performance & Scalability](#3-performance--scalability)
  - [3.1 Parallel File Analysis in `index_repository`](#31-parallel-file-analysis-in-index_repository)
  - [3.2 Streaming Embedding Batches to pgvector](#32-streaming-embedding-batches-to-pgvector)
  - [3.3 PgVectorStore Connection Pooling Per-Request](#33-pgvectorstore-connection-pooling-per-request)
  - [3.4 IVFFlat → HNSW Index for Better Search Latency](#34-ivfflat--hnsw-index-for-better-search-latency)
  - [3.5 Search Pagination & Cursor Support](#35-search-pagination--cursor-support)
- [4. Observability & Operations](#4-observability--operations)
  - [4.1 Structured Logging with Correlation IDs](#41-structured-logging-with-correlation-ids)
  - [4.2 Health & Readiness Endpoints](#42-health--readiness-endpoints)
  - [4.3 Indexing Metrics & Prometheus Instrumentation](#43-indexing-metrics--prometheus-instrumentation)
  - [4.4 Webhook Delivery Retry & Dead-Letter Queue](#44-webhook-delivery-retry--dead-letter-queue)
- [5. Correctness & Coverage](#5-correctness--coverage)
  - [5.1 Call Graph Support for Generic Tree-Sitter Languages](#51-call-graph-support-for-generic-tree-sitter-languages)
  - [5.2 Idempotency Key on Indexing Jobs](#52-idempotency-key-on-indexing-jobs)
  - [5.3 Embedding Dimension Guard in PgVectorStore](#53-embedding-dimension-guard-in-pgvectorstore)
  - [5.4 Graceful Handling of Partial Chroma-Dir Corruption](#54-graceful-handling-of-partial-chroma-dir-corruption)
- [6. Developer Experience](#6-developer-experience)
  - [6.1 OpenAPI Tags, Response Models, and Example Values](#61-openapi-tags-response-models-and-example-values)
  - [6.2 Configuration Validation at Startup](#62-configuration-validation-at-startup)
  - [6.3 Request / Response Logging Middleware](#63-request--response-logging-middleware)

---

## 1. Reliability & Resilience

### 1.1 Incremental Indexing via File-Hash Tracking

| Field | Value |
|---|---|
| **Files** | `indexing/full_pipeline.py` (`full_pipeline_pgvector()`), `lib/chunker.py`, `store/pgvector_store.py` |
| **Complexity** | **High** |

**Current state**: Every call to `full_pipeline_pgvector()` re-scans every file, re-parses every AST, re-chunks, re-embeds, and replaces all vectors via `add_vectors(replace_all=True)`. For a repository with 10 000 files, the entire pipeline runs from scratch even if only a single file changed.

**Proposed change**:

1. Before analysis, compute a content hash (e.g. SHA-256) of each file discovered by `scan_repository()`.
2. Store a `{namespace, project_id, file_path → content_hash}` mapping in a lightweight Postgres table (`file_hashes`) or a local JSON sidecar.
3. On subsequent runs, skip files whose hash is unchanged — reuse their existing chunks and embeddings from the vector store.
4. Only delete vectors for files that were removed or whose hash changed.
5. Embed and upsert only the delta set of new/changed chunks.

This reduces indexing time for large repositories from linear in total file count to linear in *changed* file count, which is critical for CI-triggered re-indexes.

---

### 1.2 Durable Background Task Processing

| Field | Value |
|---|---|
| **Files** | `main.py` (`BackgroundTasks` usage in `post_embeddings`, `github_index`) |
| **Complexity** | **High** |

**Current state**: Indexing jobs are dispatched via `FastAPI.BackgroundTasks`, which runs tasks in the same process. If the server restarts, crashes, or is horizontally scaled, in-flight jobs are silently lost with no recovery mechanism.

**Proposed change**:

1. Introduce a task queue (Celery + Redis, or RQ) and replace all `BackgroundTasks.add_task(...)` calls with queue.enqueue calls.
2. Workers pick up jobs by `job_id` and atomically transition the job status (`queued` → `running` → `completed`).
3. On server start, scan the `jobs` table for any stuck in `running` or `downloading` and reset them to `queued` so they get retried.
4. Add a configurable `max_retries` column to the `jobs` table and honour it in the worker.

This makes indexing jobs survive deploys, OOM kills, and multi-replica deployments.

---

### 1.3 Connection Pool Lifecycle Management

| Field | Value |
|---|---|
| **Files** | `db.py` (`ThreadedConnectionPool`), `main.py` (FastAPI app) |
| **Complexity** | Low |

**Current state**: `db.get_pool()` lazily creates a `ThreadedConnectionPool` that is never closed. The pool holds up to `DB_POOL_MAX` (default 20) connections that leak on shutdown, potentially triggering `FATAL: sorry, too many clients already` on the Postgres side during rapid restarts.

**Proposed change**:

1. Replace the `@app.on_event("startup")` / shutdown pattern with a modern FastAPI `lifespan` context manager.
2. In the shutdown phase, call `_POOL.closeall()` to gracefully drain all connections.
3. Set `psycopg2`'s connection `connect_timeout` and `options='-c statement_timeout=30s'` to prevent hung connections.

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    yield
    db.close_pool()  # new function

app = FastAPI(lifespan=lifespan)
```

---

### 1.4 Retry with Exponential Backoff for Voyage AI

| Field | Value |
|---|---|
| **Files** | `lib/embedder.py` (`VoyageEmbedder.encode()`) |
| **Complexity** | Medium |

**Current state**: `VoyageEmbedder.encode()` calls the Voyage AI API with a single `requests.post()` and a 120-second timeout. If the API returns a 429 (rate limit) or 5xx error, the exception propagates up and the entire indexing pipeline fails — even though the error may be transient.

**Proposed change**:

1. Wrap the HTTP call in a retry loop with exponential backoff (e.g. 1 s → 2 s → 4 s → 8 s, max 4 retries).
2. Only retry on retriable status codes: 429, 500, 502, 503, 504.
3. Respect the `Retry-After` header when the API returns 429.
4. Add a circuit breaker (e.g. via `tenacity` or a simple state machine) that skips the cloud embedder entirely after N consecutive failures, falling back to the local `CodeEmbedder` with a warning.
5. Log each retry with the batch offset so operators can identify which chunks need re-indexing.

---

## 2. Security

### 2.1 Zip Slip & Symlink Hardening in `safe_extract`

| Field | Value |
|---|---|
| **Files** | `main.py` (`safe_extract()`, `process_zip()`) |
| **Complexity** | Medium |

**Current state**: `safe_extract()` validates that no extracted member path escapes the target directory (classic Zip Slip protection). However, it does **not** check for symlink entries inside the zip. A malicious zip can contain a symlink that points to `/etc/passwd` or `../../.env`; when extracted, the symlink itself is safe, but a subsequent file written through the symlink can overwrite arbitrary paths.

**Proposed change**:

1. Iterate zip entries and reject any entry whose `external_attr` indicates a symlink (`stat.S_ISLNK(entry.external_attr >> 16)`).
2. After extraction, walk the extracted tree and verify no symlinks exist (defense in depth).
3. Add a configurable maximum number of entries (e.g. 100 000) to prevent CPU-exhaustion attacks from zip files with millions of tiny files.

---

### 2.2 Zip Bomb Protection via Uncompressed-Size Cap

| Field | Value |
|---|---|
| **Files** | `main.py` (`process_zip()`) |
| **Complexity** | Medium |

**Current state**: A 42 KB zip file can decompress to 4 GB (the classic "42.zip" bomb). The current code extracts the entire zip to disk before analysis begins, which can fill the disk, trigger OOM, or cause the indexing process to hang.

**Proposed change**:

1. Before extraction, iterate `zip_ref.infolist()` and sum `file.file_size` for all members. If the total exceeds a configurable cap (e.g. 2 GB default), reject the upload with HTTP 413.
2. Also cap the compression ratio: if any single file's `file_size / compress_size` exceeds 100×, reject it.
3. Track cumulative extracted bytes during extraction and abort if the running total exceeds the cap.

---

### 2.3 Per-Client Rate Limiting

| Field | Value |
|---|---|
| **Files** | `main.py` (all POST endpoints) |
| **Complexity** | Medium |

**Current state**: All endpoints require a Bearer token but impose no per-token rate limits. A compromised or aggressive client can exhaust Voyage AI API quota, Postgres connections, or CPU by flooding `/embed`, `/search`, or `/github/index`.

**Proposed change**:

1. Introduce a rate-limiting middleware (e.g. `slowapi` built on `limits`) keyed on the Bearer token.
2. Define different rate limits per endpoint class:
   - `/embed`, `/search`, `/search/text`: 60 req/min per token.
   - `/github/index`, `/embeddings/{ns}/{pid}` (POST): 5 req/min per token.
3. Return standard `429 Too Many Requests` with `Retry-After` header.
4. Expose per-token usage metrics via an admin endpoint.

---

### 2.4 API Key Scoping and Least-Privilege Tokens

| Field | Value |
|---|---|
| **Files** | `main.py` (`_require_bearer_token()`), `db.py` (no tokens table) |
| **Complexity** | **High** |

**Current state**: A single Bearer token grants access to all operations across all namespaces — upload, delete, search, embed, GitHub index. There is no token scoping, rotation, or revocation. The token is a static secret with no audit trail.

**Proposed change**:

1. Create an `api_keys` table: `{key_hash, label, scopes[], namespaces[], created_at, last_used_at, revoked_at}`.
2. Replace `_require_bearer_token()` with a function that resolves the key hash, checks revocation, and verifies the required scope for the endpoint.
3. Scopes: `read:search`, `write:index`, `write:delete`, `admin`.
4. Namespace scoping: restrict a key to a subset of namespaces so a tenant token cannot query another tenant's data.
5. Hash tokens with SHA-256 before storage (never store plaintext).

---

## 3. Performance & Scalability

### 3.1 Parallel File Analysis in `index_repository`

| Field | Value |
|---|---|
| **Files** | `indexing/full_pipeline.py` (`index_repository()`) |
| **Complexity** | Medium |

**Current state**: `index_repository()` analyses files sequentially in a for-loop. For a monorepo with thousands of files, this CPU-bound step dominates wall-clock time despite ample multi-core capacity.

**Proposed change**:

1. Use `concurrent.futures.ProcessPoolExecutor` (CPU-bound work needs processes, not threads due to the GIL) to parallelise `analyze_file()` across files.
2. Batch files into chunks of ~50 to amortise process startup cost.
3. Preserve ordering of results so downstream graph construction is deterministic.
4. Make the worker count configurable via `INDEXING_WORKERS` env var, defaulting to `min(os.cpu_count(), 8)`.

---

### 3.2 Streaming Embedding Batches to pgvector

| Field | Value |
|---|---|
| **Files** | `store/pgvector_store.py` (`PgVectorStore.add_vectors()`) |
| **Complexity** | Medium |

**Current state**: `add_vectors()` collects **all** (embedding, metadata) tuples into a single `rows` list before writing. For a repository producing 50 000 chunks, this holds the entire embedding set in memory (~50 000 × 768 floats × 4 bytes ≈ 146 MB) plus the metadata list before any SQL is executed.

**Proposed change**:

1. Accept a **generator/iterator** of `(vector, metadata)` pairs instead of pre-materialised lists.
2. Consume the iterator in `_INSERT_BATCH` (200) chunks, flushing each batch to Postgres immediately.
3. This reduces peak memory from O(total_chunks × dim) to O(batch_size × dim).
4. Update callers in `full_pipeline_pgvector()` to pass a generator that yields chunks as they are embedded, rather than waiting for the full list.

---

### 3.3 PgVectorStore Connection Pooling Per-Request

| Field | Value |
|---|---|
| **Files** | `store/pgvector_store.py` (`add_vectors()`, `search()`, `get_total_vectors()`, `clear()`), `db.py` |
| **Complexity** | Low |

**Current state**: Every `PgVectorStore` method calls `db.get_pool().getconn()` and `putconn()` individually. The `search()` method uses two connections (one for the query, one implicitly via `get_total_vectors()` if called). For the `/search/text` endpoint, which creates a `PgVectorStore` per request, this is fine at low traffic but becomes a bottleneck under load.

**Proposed change**:

1. Add a context-manager method `connection()` to `PgVectorStore` that wraps `getconn/putconn` in a `try/finally`.
2. Compose `search()` and other methods to accept an optional `conn` parameter so callers can share a connection across multiple operations in a single request.
3. Ensure `putconn` is always called even on exceptions (already done with `try/finally`, but add logging on unexpected errors).

---

### 3.4 IVFFlat → HNSW Index for Better Search Latency

| Field | Value |
|---|---|
| **Files** | `db.py` (`init_db()`, index creation) |
| **Complexity** | Low |

**Current state**: `init_db()` creates the vector index as:

```sql
CREATE INDEX embeddings_vec_idx ON embeddings
USING ivfflat (embedding vector_cosine_ops)
```

IVFFlat requires periodic `VACUUM ANALYZE` to keep the index accurate after bulk inserts and has higher recall variance at low `probes` settings. HNSW provides better recall at lower latency without needing a training step.

**Proposed change**:

1. Switch the index type to HNSW:

```sql
CREATE INDEX embeddings_vec_idx ON embeddings
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64)
```

2. Add a migration step that drops the old IVFFlat index and creates the HNSW one.
3. HNSW is supported in pgvector ≥ 0.5.0; add a version check or document the minimum requirement.

---

### 3.5 Search Pagination & Cursor Support

| Field | Value |
|---|---|
| **Files** | `main.py` (`search_embeddings()`, `search_by_text()`), `store/pgvector_store.py` (`search()`) |
| **Complexity** | Medium |

**Current state**: `/search` and `/search/text` accept a `k` parameter and return exactly `k` results. There is no way to fetch the next page of results or to browse beyond the top-5. Clients needing exhaustive recall must guess a high `k` value, wasting bandwidth and compute.

**Proposed change**:

1. Add `offset` (integer, default 0) and `limit` (integer, default 10) query parameters to both search endpoints.
2. In `PgVectorStore.search()`, translate these to `LIMIT x OFFSET y` in the SQL.
3. Return a `total_count` field (from a separate `COUNT(*)` query or pre-computed) so clients can build pagination UIs.
4. Alternatively, implement cursor-based pagination by encoding the last result's `(similarity, chunk_id)` as an opaque cursor string for more stable pagination under concurrent inserts.

---

## 4. Observability & Operations

### 4.1 Structured Logging with Correlation IDs

| Field | Value |
|---|---|
| **Files** | Entire codebase — pervasive `print()` calls in `lib/embedder.py`, `store/chroma_store.py`, `indexing/full_pipeline.py`, `analyzers/*.py`, `main.py` |
| **Complexity** | **High** |

**Current state**: The entire codebase logs via `print()` statements. There is no structured logging (JSON), no log levels (debug/info/warn/error), no correlation between a search request and the indexing job that produced its vectors, and no way to filter logs by request in a multi-tenant deployment.

**Proposed change**:

1. Replace all `print()` calls with the Python `logging` module (already imported in `lib/embedder.py` but used inconsistently).
2. Configure a `structlog` or `python-json-logger` formatter that emits JSON with fields: `timestamp`, `level`, `request_id`, `job_id`, `namespace`, `project_id`, `message`.
3. Add a FastAPI middleware that generates a `request_id` (UUID4) per incoming request and injects it into a `contextvars.ContextVar` accessible from any logger.
4. For background jobs, propagate the `job_id` as the correlation key.
5. Suppress noisy third-party loggers (already partially done in `lib/embedder.py` but should be centralised).

---

### 4.2 Health & Readiness Endpoints

| Field | Value |
|---|---|
| **Files** | `main.py` |
| **Complexity** | Low |

**Current state**: The only GET endpoint is `/` which serves the HTML dashboard. There are no `/health` or `/ready` endpoints. Container orchestrators (Kubernetes, ECS, Docker Compose health checks) cannot determine whether the application is alive or whether its dependencies (Postgres, Voyage AI) are reachable.

**Proposed change**:

1. Add `GET /health` — returns 200 if the process is running (liveness probe).
2. Add `GET /ready` — returns 200 only if:
   - Postgres connection can be established (`db.get_pool().getconn()` succeeds).
   - Optional: Voyage AI API key is valid (if `VOYAGE_API_KEY` is set).
   Returns 503 with a JSON body listing which dependency failed.
3. Expose a `/metrics` endpoint (see §4.3) or at minimum return basic stats: pool size, uptime, last index time.

---

### 4.3 Indexing Metrics & Prometheus Instrumentation

| Field | Value |
|---|---|
| **Files** | `indexing/full_pipeline.py`, `main.py`, `store/pgvector_store.py` |
| **Complexity** | Medium |

**Current state**: There are no operational metrics. Operators cannot answer: "How long does indexing take?", "How many vectors were upserted today?", "What is the p95 search latency?". The only signal is `print()` output in `store/chroma_store.py` ("Upserted N vectors").

**Proposed change**:

1. Integrate `prometheus-fastapi-instrumentator` to auto-collect HTTP metrics (request count, latency histograms, error rate).
2. Add custom counters/histograms:
   - `indexer_files_scanned_total{namespace, project_id}` — number of files found.
   - `indexer_chunks_created_total{namespace, project_id}` — number of chunks produced.
   - `indexer_embedding_duration_seconds{embedder, model}` — histogram of batch embedding time.
   - `indexer_vectors_stored_total{namespace, project_id, store}` — total vectors in the store.
   - `indexer_search_duration_seconds{namespace, project_id}` — search latency.
3. Expose at `/metrics` (standard Prometheus endpoint).

---

### 4.4 Webhook Delivery Retry & Dead-Letter Queue

| Field | Value |
|---|---|
| **Files** | `main.py` (`_notify_webhook()`) |
| **Complexity** | Medium |

**Current state**: `_notify_webhook()` makes a single fire-and-forget `requests.post()` with a 10-second timeout. If the webhook URL is unreachable, returns 5xx, or times out, the notification is silently lost. The caller has no way to know the webhook was not delivered.

**Proposed change**:

1. Implement retry with exponential backoff (3 retries: 1 s, 5 s, 30 s).
2. Only retry on 5xx and network errors; do **not** retry on 4xx.
3. After exhausting retries, write the failed payload to a `webhook_dead_letters` table: `{job_id, webhook_url, payload, attempts, last_status, created_at}`.
4. Add a `GET /admin/webhooks/failed` endpoint (admin-scoped) to inspect and manually replay dead-lettered webhooks.
5. Validate the webhook URL scheme (only `https://` allowed in production).

---

## 5. Correctness & Coverage

### 5.1 Call Graph Support for Generic Tree-Sitter Languages

| Field | Value |
|---|---|
| **Files** | `lib/graph.py` (`build_call_graph()`, `_extract_all_calls_*`), `utils/language_router.py` (`_GENERIC_TS_LANGUAGES`) |
| **Complexity** | **High** |

**Current state**: `build_call_graph()` only extracts call information for Python (`_extract_all_calls_python`), Go (`_extract_all_calls_go`), and JavaScript (`_extract_all_calls_js`). All other languages routed through the generic tree-sitter analyzer (Rust, Java, Ruby, C, C++, Lua) get an empty `raw_calls_map`, which means:
- Their function chunks never have `calls`, `called_by`, or `external_calls` metadata.
- Graph-enriched context is unavailable for these languages, reducing search quality.

**Proposed change**:

1. Create a generic `_extract_all_calls_generic()` function in `lib/graph.py` that uses the `LanguageProfile` definitions from `analyzers/generic_ts_analyzer.py` (specifically `func_types`) to walk the tree-sitter AST and collect call expressions.
2. Define call-expression patterns per language profile (e.g. Rust: `call_expression` → `function` field; Java: `method_invocation`; C: `call_expression`).
3. Add a `call_expr_types` field to `LanguageProfile` so each language declares which node types represent function calls.
4. Wire `_extract_all_calls_generic()` into `build_call_graph()` for languages in `_GENERIC_TS_LANGUAGES`.

---

### 5.2 Idempotency Key on Indexing Jobs

| Field | Value |
|---|---|
| **Files** | `main.py` (`post_embeddings()`, `github_index()`), `db.py` (`create_job()`) |
| **Complexity** | Medium |

**Current state**: Submitting the same zip file or GitHub repo twice creates two independent jobs that both run to completion, potentially racing on vector writes. There is no way for a client to safely retry a request that may or may not have been received.

**Proposed change**:

1. Accept an optional `Idempotency-Key` header on `POST /embeddings/{ns}/{pid}` and `POST /github/index`.
2. Before creating a new job, check if a job with the same `idempotency_key` already exists for the same `(namespace, project_id)`.
3. If found, return the existing job's status instead of creating a duplicate.
4. Add an `idempotency_key` column to the `jobs` table with a unique constraint on `(namespace, project_id, idempotency_key)`.

---

### 5.3 Embedding Dimension Guard in PgVectorStore

| Field | Value |
|---|---|
| **Files** | `store/pgvector_store.py` (`add_vectors()`), `db.py` (migration logic) |
| **Complexity** | Low |

**Current state**: `db.py` has a one-time migration that drops and recreates the `embeddings` table if the vector dimension is not 768. However, `PgVectorStore.add_vectors()` does not validate the dimension of incoming vectors against the expected dimension. If a client sends 1024-dimensional Voyage embeddings to a 768-dimensional column, Postgres raises a cryptic `expected 768 dimensions, not 1024` error deep in the batch insert, making diagnosis difficult.

**Proposed change**:

1. In `PgVectorStore.__init__()`, query the column's dimension from `pg_attribute` and store it as `self.dimension`.
2. In `add_vectors()`, validate `len(vectors[0]) == self.dimension` before any SQL and raise a clear `ValueError` with the actual and expected dimensions.
3. Return the dimension via a new `GET /embeddings/{ns}/{pid}` field so clients can pre-check.

---

### 5.4 Graceful Handling of Partial Chroma-Dir Corruption

| Field | Value |
|---|---|
| **Files** | `store/chroma_store.py` (`ChromaStore.__init__()`, `reset_chroma_dir()`) |
| **Complexity** | Medium |

**Current state**: `reset_chroma_dir()` does a full `shutil.rmtree` + `mkdir` to recover from ChromaDB's `SQLITE_READONLY_DBMOVED` error. This works but destroys the entire index, forcing a complete re-index. If corruption is limited to a single WAL segment or lock file, a full wipe is wasteful.

**Proposed change**:

1. Before `rmtree`, attempt a lightweight recovery: delete only `.lock`, `WAL`, and `SHM` files in the Chroma persist directory while keeping the main SQLite database intact.
2. Test if the client can successfully open and query the collection after WAL cleanup.
3. Fall back to full `reset_chroma_dir()` only if the lightweight recovery fails.
4. Log the recovery path taken so operators can monitor for recurring corruption.

---

## 6. Developer Experience

### 6.1 OpenAPI Tags, Response Models, and Example Values

| Field | Value |
|---|---|
| **Files** | `main.py` (all endpoint definitions) |
| **Complexity** | Low |

**Current state**: All endpoints return raw `dict` responses with no Pydantic response models. The auto-generated OpenAPI schema (`/docs`, `/openapi.json`) has no example values, no response schemas, and no tags, making it difficult for API consumers to understand expected payloads without reading the source code.

**Proposed change**:

1. Define Pydantic response models: `JobResponse`, `SearchResponse`, `SearchResult`, `EmbeddingInfoResponse`, `DeleteResponse`.
2. Annotate each endpoint with `response_model=...` and `tags=["search"]`, `tags=["indexing"]`, etc.
3. Add `examples` to Pydantic `Field()` definitions and to `Body()` parameters so the Swagger UI shows realistic request/response samples.
4. Add `summary` and `description` kwargs to each route decorator.

---

### 6.2 Configuration Validation at Startup

| Field | Value |
|---|---|
| **Files** | `main.py` (`_startup()`), `db.py` (`get_database_url()`) |
| **Complexity** | Low |

**Current state**: Configuration is read lazily from environment variables scattered across `main.py`, `db.py`, and `lib/embedder.py`. A missing or invalid `DATABASE_URL` is only discovered when the first request arrives (or never — it falls back to `postgresql://postgres:postgres@localhost:5432/indexer`). In production, silent fallbacks to local defaults can cause data loss.

**Proposed change**:

1. Create a `config.py` module that loads all environment variables into a typed `Settings` class (using `pydantic-settings`).
2. Define required vs. optional settings with clear validation rules:
   - `DATABASE_URL`: required in prod, optional in dev (with local default).
   - `VOYAGE_API_KEY`: optional, but if `LARGE_CODEBASE_THRESHOLD` is low, warn that cloud embeddings may be needed.
3. Call `Settings.validate()` in the `lifespan` startup and abort with a clear error message listing all missing required settings.
4. Log all effective settings (with secrets masked) at `INFO` level on startup.

---

### 6.3 Request / Response Logging Middleware

| Field | Value |
|---|---|
| **Files** | `main.py` (FastAPI app) |
| **Complexity** | Low |

**Current state**: There is no visibility into individual API requests. When debugging a search quality issue or a failed indexing job, operators must grep unstructured `print()` output and manually correlate log lines.

**Proposed change**:

1. Add a FastAPI middleware that logs every request with:
   - Method, path, query params, request ID.
   - Response status code, response time (ms).
   - Client IP (from `X-Forwarded-For` or direct).
2. Skip logging for `/health` and `/metrics` to avoid noise.
3. Mask sensitive fields (`access_token`, `Authorization` header) in logs.
4. Use the structured JSON logger from §4.1 so all request logs are machine-parseable.
