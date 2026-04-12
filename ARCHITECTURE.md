# Architecture

## System Overview

```
                        ┌─────────────────────────────────────────────┐
                        │              Client (curl / app)            │
                        └─────────────┬───────────────┬─────────────┘
                                      │               │
                              POST /api/index   POST /api/search
                                      │               │
                        ┌─────────────▼───────────────▼─────────────┐
                        │            Flask Application               │
                        │              (main.py)                     │
                        │                                            │
                        │  ┌──────────┐  ┌───────────┐  ┌────────┐  │
                        │  │ /api/*   │  │ Background │  │ Health │  │
                        │  │ Routes   │  │ Job Thread │  │  Check │  │
                        │  └────┬─────┘  └─────┬─────┘  └────────┘  │
                        └───────┼──────────────┼────────────────────┘
                                │              │
                 ┌──────────────▼──┐   ┌───────▼────────────────────┐
                 │  File Scanner   │   │  Indexing Pipeline         │
                 │  (utils/)       │   │  (indexing/full_pipeline)  │
                 │                 │   │                            │
                 │  ┌───────────┐  │   │  1. scan_repository()     │
                 │  │ Ignore    │  │   │  2. analyze_file() × N    │
                 │  │ Patterns  │  │   │  3. build_import_graph()  │
                 │  └───────────┘  │   │  4. build_call_graph()    │
                 │                 │   │  5. chunk_repository_...() │
                 └────────┬────────┘   │  6. embed_chunks()        │
                          │            │  7. store.add_vectors()   │
                 ┌────────▼────────┐   └────────┬──────────────────┘
                 │ Language Router │            │
                 │ (utils/)        │   ┌────────▼────────┐
                 │                 │   │  Vector Store    │
                 │ .py → Python    │   │                  │
                 │ .go → Go        │   │ ┌─────────────┐ │
                 │ .js → JS/TS     │   │ │  pgvector   │ │
                 │ .sh → Shell     │   │ │  (prod)     │ │
                 │ .rs → Rust …    │   │ └─────────────┘ │
                 └────────┬────────┘   │ ┌─────────────┐ │
                          │            │ │  ChromaDB   │ │
                 ┌────────▼────────┐   │ │  (local)    │ │
                 │   Analyzers     │   │ └─────────────┘ │
                 │ (analyzers/)    │   └────────┬────────┘
                 │                 │            │
                 │ Python: ast     │   ┌────────▼────────┐
                 │ Go: tree-sitter │   │  PostgreSQL     │
                 │ JS: tree-sitter │   │                 │
                 │ Generic: TS     │   │  • jobs table   │
                 │ Config: stdlib  │   │  • index_meta   │
                 └─────────────────┘   │  • embeddings   │
                                       └─────────────────┘
```

## Tech Stack

| Component | Technology | Notes |
|---|---|---|
| Web framework | **Flask** + flask-cors | Single-process, threaded, runs on port 8001 |
| Database | **PostgreSQL** + pgvector extension | `ThreadedConnectionPool` via psycopg2 |
| Local embeddings | `microsoft/codebert-base` | HuggingFace transformers + sentence-transformers, 768-dim |
| Cloud embeddings | `voyage-code-3` via Voyage AI API | 1024-dim, batched (128 texts/request) |
| Local vector store | **ChromaDB** | Embedded, cosine distance, SQLite-backed |
| Production vector store | **pgvector** | `vector(768)`, IVFFlat index with cosine ops |
| AST parsing | **tree-sitter** (Go, JS, Rust, Java, Ruby, C, C++, Lua) + Python `ast` stdlib | Regex fallback for Python syntax errors |

## Module Breakdown

| File / Directory | Responsibility |
|---|---|
| `main.py` | Flask application, route definitions, `before_request` auth middleware, background indexing job runner (`_run_index_job`), git clone helper (`_clone_repo`) |
| `db.py` | `ThreadedConnectionPool` management, `init_db()` schema creation, CRUD operations for `jobs`, `index_meta`, and `embeddings` tables |
| `models.py` | Shared dataclasses: `FunctionInfo`, `ClassInfo`, `FieldInfo`, `VariableInfo`, `FileAnalysis` |
| `pipeline.py` | Standalone `run_indexing()` entry-point for programmatic use (chunk → embed → pgvector) |
| `requirements.txt` | Python dependency declarations |
| `analyzers/python_analyzer.py` | Python AST analysis via `ast` module with regex fallback on `SyntaxError`. Extracts functions, classes, decorators, module-level variables, and class-level variables |
| `analyzers/go_analyzer.py` | Go analysis via tree-sitter-go. Extracts functions, methods, structs (with struct tags), interfaces, type aliases, and const/var declarations |
| `analyzers/js_analyzer.py` | JavaScript/TypeScript analysis via tree-sitter-javascript. Extracts functions, classes, methods, arrow functions, and const/let/var variables |
| `analyzers/shell_analyzer.py` | Shell script analysis via regex. Extracts functions, aliases, exports, and source/include paths |
| `analyzers/generic_ts_analyzer.py` | Config-driven tree-sitter analyzer for Rust, Java, Ruby, C, C++, and Lua. Uses `LanguageProfile` dataclass to describe each language's node types and naming strategies |
| `analyzers/config_analyzer.py` | Configuration file analysis. Supports YAML (PyYAML), TOML (tomllib/tomli), JSON (stdlib), INI (configparser), and a generic key-value fallback for .env, .gitignore, SSH config, etc. |
| `lib/chunker.py` | Splits `FileAnalysis` objects into typed chunks: `function`, `class`, `variable`, `imports`, `file_summary`, and `file`. Each chunk carries a `content` string (for embedding) and a `metadata` dict (for LLM context) |
| `lib/embedder.py` | `CodeEmbedder` (local HuggingFace) and `VoyageEmbedder` (cloud API). Both implement `encode(texts)` and `embed_chunks(chunks)`. Includes `EmbeddingCache` for cross-project deduplication |
| `lib/graph.py` | Builds two complementary graphs: (1) **import graph** — file-level `depends_on`/`imported_by` edges; (2) **call graph** — symbol-level `calls`/`called_by`/`external_calls` edges. Graph data is injected into chunk metadata |
| `store/pgvector_store.py` | `PgVectorStore` — inserts embeddings via `INSERT … ON CONFLICT DO UPDATE`, cosine similarity search via `<=>` operator, batch size of 200 |
| `store/chroma_store.py` | `ChromaStore` — wraps a ChromaDB collection with upsert, cosine distance search, per-file deletion, and SQLite connection caching to avoid `SQLITE_READONLY_DBMOVED` |
| `store/project_store.py` | `ProjectStore` — thread-safe, JSON-file-backed project registry (not used by the Flask API; available for tooling) |
| `utils/file_scanner.py` | `scan_repository()` — walks a directory tree, skipping VCS dirs, build outputs, IDE files, binaries, generated files, and files > 512 KB |
| `utils/language_router.py` | `detect_language()` — maps file paths to language names via extension map, filename map, and shebang detection. `analyze_file()` dispatches to the correct analyzer |
| `indexing/full_pipeline.py` | `full_pipeline_pgvector()` and `full_pipeline_chroma()` — end-to-end pipeline functions that orchestrate scanning, analysis, chunking, graph building, embedding, and storage |
| `web/index.html` | Static API documentation page (not served by Flask; meant for standalone viewing or deployment behind a static file server) |

## Data Flow — Indexing Pipeline

```
1. scan_repository(repo_path)
   └── Walk directory tree → list of source file paths

2. analyze_file(file_path)  × N files
   └── language_router detects language
   └── dispatches to correct analyzer
   └── returns FileAnalysis (functions, classes, imports, variables, package)

3. build_import_graph(analyses)
   └── Resolves import strings to local file paths
   └── Returns {file_path: {depends_on: [...], imported_by: [...]}}

4. build_call_graph(analyses)
   └── For Python: walks AST Call nodes
   └── For Go/JS: walks tree-sitter call_expression nodes
   └── Returns {"file::func": {calls: [...], called_by: [...], external_calls: [...]}}

5. chunk_repository_analyses(analyses)
   └── Creates chunks: imports, function, class, variable, file_summary, file
   └── Each chunk = {content: str, type: str, metadata: dict}

6. enrich_chunks_with_graph() + enrich_chunks_with_call_graph()
   └── Injects import graph and call graph data into chunk metadata

7. embedder.embed_chunks(chunks)
   └── CodeEmbedder: local HuggingFace inference (batched, mean-pooled, L2-normalised)
   └── VoyageEmbedder: cloud API (batched 128 texts, 1024-dim)
   └── Model selected by: per-tenant override > chunk count heuristic (> 99 999 → cloud)

8. store.add_vectors(embeddings, metadata, replace_all=True)
   └── pgvector: INSERT ... ON CONFLICT DO UPDATE in batches of 200
   └── ChromaDB: upsert in batches of 5000, then delete stale IDs
```

## Database Schema

### `jobs` table

Tracks indexing job lifecycle.

| Column | Type | Description |
|---|---|---|
| `job_id` | `TEXT PK` | UUID v4 job identifier |
| `namespace` | `TEXT NOT NULL` | Tenant namespace |
| `project_id` | `TEXT NOT NULL` | Project identifier |
| `filename` | `TEXT` | Repository URL |
| `status` | `TEXT NOT NULL` | `queued`, `running`, `complete`, or `failed` |
| `source` | `TEXT` | Source type (e.g. `"git"`) |
| `owner` | `TEXT` | Repository owner (reserved) |
| `repo` | `TEXT` | Repository name (reserved) |
| `ref` | `TEXT` | Git ref used for cloning |
| `total_vectors` | `INTEGER` | Vector count after successful indexing |
| `persist_dir` | `TEXT` | Chroma persist directory (legacy) |
| `embedder` | `TEXT` | Embedder class name used |
| `webhook_url` | `TEXT` | Webhook URL for job notifications |
| `error` | `TEXT` | Error message on failure |
| `created_at` | `TIMESTAMPTZ NOT NULL` | Job creation time |
| `updated_at` | `TIMESTAMPTZ NOT NULL` | Last status update time |

### `index_meta` table

Stores per-project embedding model configuration.

| Column | Type | Description |
|---|---|---|
| `namespace` | `TEXT PK` | Tenant namespace |
| `project_id` | `TEXT PK` | Project identifier |
| `embedder` | `TEXT` | Embedder class name (e.g. `"CodeEmbedder"`, `"VoyageEmbedder"`) |
| `model` | `TEXT` | Model identifier (e.g. `"microsoft/codebert-base"`, `"voyage-code-3"`) |
| `embedding_dim` | `INTEGER` | Vector dimensionality (768 or 1024) |
| `use_cloud` | `BOOLEAN DEFAULT FALSE` | Whether cloud embedder was used |
| `preferred_embedder` | `TEXT` | Per-tenant model override (e.g. `"voyage-code-3"`, `null`) |
| `created_at` | `TIMESTAMPTZ NOT NULL` | First index creation time |
| `updated_at` | `TIMESTAMPTZ NOT NULL` | Last index update time |

Primary key: `(namespace, project_id)`.

### `embeddings` table

Stores embedding vectors with their metadata.

| Column | Type | Description |
|---|---|---|
| `id` | `BIGSERIAL PK` | Auto-incrementing row ID |
| `namespace` | `TEXT NOT NULL` | Tenant namespace |
| `project_id` | `TEXT NOT NULL` | Project identifier |
| `chunk_id` | `TEXT NOT NULL` | MD5 hash of `file_path::chunk_type::symbol_name::line_number` |
| `file_path` | `TEXT` | Source file path |
| `chunk_type` | `TEXT` | `function`, `class`, `variable`, `imports`, `file_summary`, or `file` |
| `symbol_name` | `TEXT` | Symbol identifier (function name, class name, etc.) |
| `line_number` | `INTEGER` | Source line number |
| `content` | `TEXT` | Source text of the chunk |
| `metadata` | `JSONB` | Full metadata dict (package, signature, docstring, params, etc.) |
| `embedding` | `vector(768)` | Normalised embedding vector |
| `created_at` | `TIMESTAMPTZ NOT NULL` | Insertion time |

Indexes:
- Unique: `(namespace, project_id, chunk_id)`
- IVFFlat: `embeddings_vec_idx` on `embedding vector_cosine_ops`

## Vector Store Comparison

| Feature | pgvector (production) | ChromaDB (local) |
|---|---|---|
| **Storage backend** | PostgreSQL table | Embedded SQLite |
| **Isolation** | Multi-tenant via `namespace` + `project_id` columns | Single collection per persist directory |
| **Vector index** | IVFFlat (cosine) | HNSW (cosine) |
| **Deletion** | `DELETE WHERE namespace = %s AND project_id = %s` | `delete_by_file()` or `clear()` |
| **Persistence** | Automatic (Postgres WAL) | Automatic (Chroma writes on every upsert) |
| **Upsert** | `ON CONFLICT DO UPDATE` | Chroma `upsert` by ID |
| **Used by** | Flask API (`POST /api/index`, `POST /api/search`) | Standalone pipeline (`full_pipeline_chroma()`) |
| **Best for** | Production deployments, multi-tenant SaaS | Local development, offline/embedded use |

## Authentication & Webhooks

### Authentication

All `/api/*` routes are protected by a `before_request` hook in Flask. When the `API_KEY` environment variable is set, every request must include:

```
Authorization: Bearer <API_KEY>
```

If the header is missing or incorrect, the server returns `401 Unauthorized` with `{"error": "Unauthorized"}`.

### Webhook Notifications

When a `webhook_url` is provided in the `POST /api/index` request body, the service sends a plain HTTP POST (no HMAC signing) to that URL on job completion:

```json
{
  "job_id": "a3f9c1...",
  "status": "complete",
  "namespace": "acme",
  "project_id": "my-service",
  "total_vectors": 2841
}
```

The webhook request has a 10-second timeout. Failures are silently ignored.

## Web UI

`web/index.html` is a static single-page API documentation site. It is **not served by the Flask application** — it is a standalone HTML file that can be opened directly in a browser or deployed behind a static file server (nginx, S3, etc.). Note: the endpoint documentation in this HTML file describes a different API surface than what `main.py` implements; always refer to this `ARCHITECTURE.md` and the source code as the source of truth.
