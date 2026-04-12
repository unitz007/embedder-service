# Embedder Service

A **Flask** micro-service that indexes source code repositories, generates semantic embeddings, and exposes a REST API for natural-language code search.

Built around a multi-stage pipeline:

> **Repository → File Discovery → Language Detection → AST Parsing → Symbol Extraction → Code Chunking → Embedding Generation → Vector Store → Semantic Search**

---

## Features

- **Multi-language AST analysis** — Python, Go, JavaScript/TypeScript, Shell, Rust, Java, Ruby, C, C++, Lua, plus config files (YAML, TOML, JSON, INI, .env, dotfiles)
- **Two embedding models** — local `microsoft/codebert-base` (768-dim) for small codebases; cloud `voyage-code-3` (1024-dim) via Voyage AI for large ones
- **Per-tenant model routing** — override the embedder per namespace/project via API
- **pgvector storage** — production-ready, multi-tenant Postgres with cosine similarity (IVFFlat index)
- **ChromaDB fallback** — local/embedded single-tenant vector store for offline use
- **Import & call graphs** — file-level dependency graph and symbol-level call graph injected into chunk metadata for LLM context
- **Background indexing** — git clone → parse → embed runs in a background thread; poll status via job ID
- **Webhook notifications** — plain HTTP POST to a configurable URL on job completion

---

## Quick Start

### Docker

```bash
# Build and run (Postgres must be reachable at DATABASE_URL)
docker build -t embedder-service .
docker run --rm \
  -e DATABASE_URL="postgresql://postgres:postgres@host.docker.internal:5432/indexer" \
  -p 8001:8001 \
  embedder-service
```

> A `Dockerfile` is not yet shipped — create one that installs `requirements.txt`, copies the source tree, and runs `python main.py`. See [`.dockerignore`](.dockerignore) for paths excluded from the image.

### Local

```bash
# 1. Create a virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Start PostgreSQL with the pgvector extension
#    (skip if using an external database)

# 3. Set required environment variables
export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/indexer"

# 4. Initialise the database schema and start the service
python -c "import db; db.init_db()"
python main.py
```

The service listens on **`http://localhost:8001`** by default (configurable via `PORT`).

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `PORT` | No | `8001` | HTTP listen port |
| `DATABASE_URL` | Yes (prod) | `postgresql://postgres:postgres@localhost:5432/indexer` | PostgreSQL connection string |
| `LOCAL_DATABASE_URL` | No | — | Override `DATABASE_URL` in non-prod environments |
| `APP_ENV` | No | `dev` | Set to `prod` to enforce `sslmode=require` on the database URL |
| `DB_POOL_MAX` | No | `20` | Max connections in the `ThreadedConnectionPool` |
| `API_KEY` | No | — | Bearer token for authenticating `/api/*` requests. When set, all `/api/` routes require `Authorization: Bearer <key>` |
| `VOYAGE_API_KEY` | No | — | Voyage AI API key. Required when indexing large codebases (≥ 100 000 chunks) or when using `voyage-code-3` |
| `WEBHOOK_SECRET` | No | — | Loaded via `config.py` for webhook signature verification |
| `LOCAL_INDEXER_BASE` | No | — | Base URL of an external clone service (`POST /api/clone`). When set, cloning is delegated to this service instead of local `git` |

---

## API Reference

All endpoints are prefixed with `/api/`. Authentication (when `API_KEY` is set) uses `Authorization: Bearer <key>`.

### `GET /api/health`

Health check.

```bash
curl http://localhost:8001/api/health
```

```json
{"status": "ok"}
```

---

### `POST /api/index`

Queue a repository indexing job. The service clones the repo, parses all files, generates embeddings, and stores them in pgvector. Runs asynchronously — use the returned `job_id` to poll status.

```bash
curl -X POST http://localhost:8001/api/index \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <key>" \
  -d '{
    "project_id": "my-service",
    "namespace": "acme",
    "url": "https://github.com/acme/my-service.git",
    "ref": "main",
    "webhook_url": "https://my-ci.example.com/hooks/index-done",
    "preferred_embedder": "voyage-code-3"
  }'
```

**Request body:**

| Field | Type | Required | Description |
|---|---|---|---|
| `project_id` | string | Yes | Project identifier |
| `url` | string | Yes | Git repository URL to clone |
| `namespace` | string | No | Storage namespace (defaults to `DEFAULT_NAMESPACE` or `"default"`) |
| `ref` | string | No | Branch, tag, or commit SHA (default: `"HEAD"`) |
| `source` | string | No | Source type label (default: `"git"`) |
| `webhook_url` | string | No | URL to POST job completion notification to |
| `preferred_embedder` | string | No | Embedder override: `"voyage-code-3"` (cloud), `"microsoft/codebert-base"` (local), or `null` for auto-select |

**Response:**

```json
{"job_id": "a3f9c1...", "status": "queued"}
```

---

### `GET /api/jobs`

List all indexing jobs.

```bash
curl http://localhost:8001/api/jobs
```

---

### `GET /api/jobs/<job_id>`

Get details for a specific job.

```bash
curl http://localhost:8001/api/jobs/a3f9c1...
```

```json
{
  "job_id": "a3f9c1...",
  "namespace": "acme",
  "project_id": "my-service",
  "status": "complete",
  "total_vectors": 2841,
  "embedder": "VoyageEmbedder",
  "created_at": "2024-01-15T10:32:00Z",
  "updated_at": "2024-01-15T10:34:18Z"
}
```

Job status values: `queued` → `running` → `complete` | `failed`

---

### `GET /api/jobs/<job_id>/status`

Lightweight status check (returns only `job_id` and `status`).

```bash
curl http://localhost:8001/api/jobs/a3f9c1.../status
```

```json
{"job_id": "a3f9c1...", "status": "complete"}
```

---

### `POST /api/search`

Semantic code search. The server reads the project's index metadata, loads the correct embedding model, embeds the query, and returns the top-k most similar code chunks.

```bash
curl -X POST http://localhost:8001/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "my-service",
    "namespace": "acme",
    "query": "how does authentication work",
    "top_k": 5
  }'
```

**Request body:**

| Field | Type | Required | Description |
|---|---|---|---|
| `project_id` | string | Yes | Project identifier |
| `query` | string | Yes | Natural language search query |
| `namespace` | string | No | Storage namespace (defaults to `DEFAULT_NAMESPACE` or `"default"`) |
| `top_k` | integer | No | Number of results (default: `5`) |

**Response:**

```json
{
  "results": [
    {
      "file_path": "src/auth/handler.py",
      "chunk_type": "function",
      "symbol_name": "authenticate",
      "line_number": 42,
      "content": "def authenticate(request): ...",
      "metadata": { ... },
      "score": 0.91
    }
  ],
  "total": 1
}
```

---

### `GET /api/index/<namespace>/<project_id>/model`

Get the embedding model configuration for a project's index.

```bash
curl http://localhost:8001/api/index/acme/my-service/model
```

```json
{
  "namespace": "acme",
  "project_id": "my-service",
  "embedder": "VoyageEmbedder",
  "model": "voyage-code-3",
  "embedding_dim": 1024,
  "use_cloud": true,
  "preferred_embedder": "voyage-code-3"
}
```

---

### `PUT /api/index/<namespace>/<project_id>/model`

Set or clear the preferred embedding model for a project. This override is honoured on subsequent re-indexes.

```bash
curl -X PUT http://localhost:8001/api/index/acme/my-service/model \
  -H "Content-Type: application/json" \
  -d '{"preferred_embedder": "voyage-code-3"}'
```

**Request body:**

| Field | Type | Description |
|---|---|---|
| `preferred_embedder` | string or null | Model override: `"voyage-code-3"` (cloud), `"microsoft/codebert-base"` (local), or `null` to revert to auto-select |

---

## Supported Languages

| Language | Extensions | Analyzer | Parser |
|---|---|---|---|
| **Python** | `.py` | `python_analyzer.py` | `ast` (stdlib) + regex fallback |
| **Go** | `.go` | `go_analyzer.py` | tree-sitter-go |
| **JavaScript / TypeScript** | `.js`, `.ts`, `.jsx`, `.tsx`, `.mjs`, `.cjs` | `js_analyzer.py` | tree-sitter-javascript |
| **Shell** | `.sh`, `.bash`, `.zsh`, `.fish`, plus dotfiles | `shell_analyzer.py` | regex |
| **Rust** | `.rs` | `generic_ts_analyzer.py` | tree-sitter-rust |
| **Java** | `.java` | `generic_ts_analyzer.py` | tree-sitter-java |
| **Ruby** | `.rb`, `.rake`, `Rakefile`, `Gemfile`, … | `generic_ts_analyzer.py` | tree-sitter-ruby |
| **C** | `.c`, `.h` | `generic_ts_analyzer.py` | tree-sitter-c |
| **C++** | `.cpp`, `.cc`, `.cxx`, `.hh`, `.hpp`, `.hxx` | `generic_ts_analyzer.py` | tree-sitter-cpp |
| **Lua** | `.lua` | `generic_ts_analyzer.py` | tree-sitter-lua |
| **Config** | `.yaml`, `.yml`, `.toml`, `.json`, `.ini`, `.env`, dotfiles, `Makefile`, `Dockerfile`, … | `config_analyzer.py` | PyYAML / tomli / stdlib / regex |

## Embedding Models

| Model | Type | Dimensions | When Used |
|---|---|---|---|
| `microsoft/codebert-base` | Local (HuggingFace / sentence-transformers) | 768 | Default for small codebases (< 100 000 chunks) |
| `voyage-code-3` | Cloud (Voyage AI API) | 1024 | Large codebases (≥ 100 000 chunks) or explicit per-tenant override |

Model selection is automatic based on chunk count, or can be overridden per-tenant via `PUT /api/index/<namespace>/<project_id>/model` or the `preferred_embedder` field in `POST /api/index`.

---

## Project Structure

```
├── main.py              # Flask app, REST routes, background job runner
├── db.py                # Postgres connection pool, schema init, CRUD for jobs/index_meta/embeddings
├── models.py            # Dataclasses: FunctionInfo, ClassInfo, FieldInfo, VariableInfo, FileAnalysis
├── pipeline.py          # Standalone pipeline entry-point (chunk → embed → store)
├── requirements.txt     # Python dependencies
├── analyzers/           # Per-language AST parsers
│   ├── python_analyzer.py
│   ├── go_analyzer.py
│   ├── js_analyzer.py
│   ├── shell_analyzer.py
│   ├── generic_ts_analyzer.py   # Rust, Java, Ruby, C, C++, Lua
│   └── config_analyzer.py       # YAML, TOML, JSON, INI, .env, dotfiles
├── lib/
│   ├── chunker.py       # Code chunking (function, class, variable, imports, file-summary chunks)
│   ├── embedder.py      # CodeEmbedder (local) and VoyageEmbedder (cloud)
│   └── graph.py         # Import graph + call graph builder
├── store/
│   ├── pgvector_store.py    # PgVectorStore — production vector store
│   ├── chroma_store.py      # ChromaStore — local/embedded vector store
│   └── project_store.py     # ProjectStore — JSON-file project registry
├── utils/
│   ├── file_scanner.py      # Repository file discovery with ignore patterns
│   └── language_router.py   # Language detection + analyzer dispatch
├── indexing/
│   ├── full_pipeline.py     # End-to-end pipeline (pgvector + Chroma variants)
│   └── USAGE_GUIDE.md       # Standalone script usage guide
└── web/
    └── index.html           # Static API documentation page (not served by Flask)
```

For a detailed architecture breakdown, see [ARCHITECTURE.md](ARCHITECTURE.md).  
For the changelog of shipped features, see [IMPROVEMENTS.md](IMPROVEMENTS.md).  
For the standalone indexing script guide, see [indexing/USAGE_GUIDE.md](indexing/USAGE_GUIDE.md).
