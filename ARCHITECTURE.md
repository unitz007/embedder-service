# Architecture

## Overview

Embedder Service is a FastAPI HTTP application that indexes source code repositories and provides semantic code search. The system extracts structured information from source files via per-language AST parsers, chunks code into semantically meaningful units enriched with import and call graph metadata, generates vector embeddings using either a local model (CodeBERT) or a cloud model (Voyage AI), and stores the results in PostgreSQL with the pgvector extension for efficient approximate-nearest-neighbour search.

```
┌──────────────┐
│   Client     │  HTTP (JSON)
│  (Web UI /   │◄──────────────────────────────────────────┐
│   curl / SDK)│──────────────────────────────────────────►│
└──────┬───────┘                                           │
       │                                                   │
       ▼                                                   │
┌──────────────┐      ┌──────────────┐      ┌────────────┐ │
│   FastAPI    │      │   pgvector   │      │  Job Queue │ │
│   main.py    │─────►│   db.py      │◄─────│  (in-proc) │ │
│   Routes     │      │   store/     │      │  Webhooks  │ │
└──────┬───────┘      └──────────────┘      └────────────┘ │
       │                                                   │
       ▼                                                   │
┌──────────────┐      ┌──────────────┐      ┌────────────┐ │
│   Indexing   │      │   Chunking   │      │  Embedding │ │
│   Pipeline   │─────►│   lib/       │─────►│  lib/      │ │
│   indexing/  │      │   chunker.py │      │  embedder  │ │
└──────────────┘      └──────────────┘      └────────────┘ │
       │                                                   │
       ▼                                                   │
┌──────────────┐      ┌──────────────┐                     │
│    AST       │      │    Graph     │                     │
│   Analyzers  │─────►│  lib/graph   │─────────────────────┘
│   analyzers/ │      │              │  (enriches chunks
│   utils/     │      └──────────────┘   before embedding)
│  language_   │
│  router.py   │
└──────────────┘
```

## Request Lifecycle

### Indexing a Repository

1. **Job creation** — The client submits a POST request to either `/embeddings/{namespace}/{project_id}` (zip upload) or `/github/index` (GitHub repo or local directory). The server creates a job record in the `jobs` table and returns the `job_id` immediately.

2. **Background indexing** — The indexing pipeline runs synchronously within the request handler (not in a separate worker process). The pipeline executes these steps in order:
   - **File discovery** (`utils/file_scanner.py`) — Recursively walks the repository, respecting `.gitignore` rules and applying language-specific filters to select indexable source files.
   - **Language detection & analysis** (`utils/language_router.py` → `analyzers/`) — Each file is classified by language extension and dispatched to the appropriate analyzer. Analyzers produce `FileAnalysis` objects containing extracted symbols (functions, classes, imports, etc.).
   - **Graph construction** (`lib/graph.py`) — Two graphs are built:
     - **Import graph** — Maps each file to the set of files it imports (file-level dependency).
     - **Call graph** — Maps each function/class to the set of symbols it references (symbol-level dependency).
   - **Chunking** (`lib/chunker.py`) — Source code is split into semantically meaningful chunks (e.g., one per function/class). Each chunk receives a unique `chunk_id` and metadata including file path, symbol name, line number, chunk type, and extracted imports.
   - **Graph enrichment** — Chunk metadata is augmented with related symbols from the import and call graphs, providing richer context for retrieval.
   - **Embedding generation** (`lib/embedder.py`) — Chunks are embedded as dense vectors. The system automatically selects the embedding model:
     - **Local**: `CodeEmbedder` using `microsoft/codebert-base` (768-dimensional) via HuggingFace transformers with masked mean pooling. Used when chunk count is below `LARGE_CODEBASE_THRESHOLD` (currently 99 999).
     - **Cloud**: `VoyageEmbedder` using Voyage AI's `voyage-code-3` (1024-dimensional) API. Used for large codebases. Requires `VOYAGE_API_KEY`.
   - **Vector storage** (`store/pgvector_store.py`) — Embeddings and their metadata are upserted into the `embeddings` table with `replace_all=True`, which removes stale vectors from previous indexing runs. An `index_meta` record captures the embedding model name, dimension, and cloud/local flag to enable correct model selection at query time.

3. **Job completion** — The job status is updated to `"completed"` (or `"failed"` on error), and a webhook notification is sent if `webhook_url` was provided. The total vector count is recorded on the job.

### Searching Code

Two search endpoints are available:

- **`POST /search`** — Accepts a pre-computed embedding vector (`list[float]`) plus namespace/project ID and result count (`k`). The `PgVectorStore.search()` method performs a cosine similarity query against the `embeddings` table using the ivfflat index.
- **`POST /search/text`** — Accepts raw query text. The server reads the `index_meta` record for the target project to determine which embedding model was used during indexing, instantiates the corresponding embedder, encodes the query, and delegates to the vector search path. This prevents dimension mismatches when the project was indexed with Voyage AI (1024-d) but the default CodeBERT model (768-d) would be used otherwise.

### Generating Embeddings

`POST /embed` accepts arbitrary text and returns its embedding vector, using the local CodeBERT model by default. This is useful for pre-computing query embeddings on the client side.

## Data Model

### Tables

#### `jobs`

Tracks indexing operations and their lifecycle.

| Column | Type | Description |
|---|---|---|
| `job_id` | `TEXT PK` | Unique job identifier (UUID) |
| `namespace` | `TEXT` | Tenant/project namespace |
| `project_id` | `TEXT` | Project identifier within the namespace |
| `filename` | `TEXT` | Original filename or repo identifier |
| `status` | `TEXT` | Job status: `queued`, `processing`, `completed`, `failed` |
| `source` | `TEXT` | Source type: `github`, `upload`, `local` |
| `owner` | `TEXT` | Repository owner (GitHub sources) |
| `repo` | `TEXT` | Repository name (GitHub sources) |
| `ref` | `TEXT` | Git reference (branch/tag/commit) |
| `total_vectors` | `INTEGER` | Number of vectors stored |
| `persist_dir` | `TEXT` | Local persist directory path (legacy) |
| `embedder` | `TEXT` | Embedder class name used |
| `webhook_url` | `TEXT` | URL for completion notifications |
| `error` | `TEXT` | Error message (if failed) |
| `created_at` | `TIMESTAMPTZ` | Job creation timestamp |
| `updated_at` | `TIMESTAMPTZ` | Last update timestamp |

#### `embeddings`

Stores vector embeddings with their associated code chunk metadata.

| Column | Type | Description |
|---|---|---|
| `id` | `BIGSERIAL PK` | Auto-incrementing row ID |
| `namespace` | `TEXT` | Tenant namespace |
| `project_id` | `TEXT` | Project identifier |
| `chunk_id` | `TEXT` | Unique chunk identifier (function name, class name, etc.) |
| `file_path` | `TEXT` | Relative source file path |
| `chunk_type` | `TEXT` | Type: `function`, `class`, `module` |
| `symbol_name` | `TEXT` | Primary symbol name |
| `line_number` | `INTEGER` | Start line of the chunk in the source file |
| `content` | `TEXT` | Source code text of the chunk |
| `metadata` | `JSONB` | Additional metadata (imports, called functions, graph data) |
| `embedding` | `vector(768)` | Dense embedding vector |
| `created_at` | `TIMESTAMPTZ` | Insertion timestamp |

**Unique constraint**: `(namespace, project_id, chunk_id)` — ensures one embedding per chunk per project.

**Index**: `embeddings_vec_idx` — ivfflat index on the `embedding` column using `vector_cosine_ops` for approximate nearest-neighbour search.

#### `index_meta`

Records which embedding model was used for a project's index.

| Column | Type | Description |
|---|---|---|
| `namespace` | `TEXT PK` | Tenant namespace |
| `project_id` | `TEXT PK` | Project identifier |
| `embedder` | `TEXT` | Embedder class name: `CodeEmbedder` or `VoyageEmbedder` |
| `model` | `TEXT` | Model name: `microsoft/codebert-base` or `voyage-code-3` |
| `embedding_dim` | `INTEGER` | Vector dimension: 768 (local) or 1024 (cloud) |
| `use_cloud` | `BOOLEAN` | Whether cloud embedding was used |
| `created_at` | `TIMESTAMPTZ` | Index creation timestamp |
| `updated_at` | `TIMESTAMPTZ` | Last re-index timestamp |

### Data Flow Diagram

```
Repository (zip / git clone / local dir)
        │
        ▼
  File Discovery ──── scan_repository()
        │              • Recursive walk
        │              • .gitignore filtering
        │              • Language extension filtering
        ▼
  Language Detection ── analyze_file()
        │              • Extension → language mapping
        │              • Analyzer dispatch
        ▼
  AST Analysis ────── analyzers/*.py
        │              • Function extraction
        │              • Class extraction
        │              • Import collection
        │              • Docstring extraction
        ▼
  FileAnalysis objects (list)
        │
        ├──► Import Graph ──── build_import_graph()
        │       • file → set[imported_files]
        │
        ├──► Call Graph ───── build_call_graph()
        │       • symbol → set[called_symbols]
        │
        ▼
  Code Chunking ───── chunk_repository_analyses()
        │              • One chunk per function/class
        │              • Module-level chunks for top-level code
        │              • Metadata: file_path, symbol, line, imports
        ▼
  Graph Enrichment ── enrich_chunks_with_graph()
        │              • Add related symbols from import graph
        │              enrich_chunks_with_call_graph()
        │              • Add called functions from call graph
        ▼
  List[Dict] chunks
        │
        ▼
  Embedding ────────── CodeEmbedder / VoyageEmbedder
        │              • encode(texts) → np.ndarray (float32)
        │              • Batched processing
        ▼
  List[Dict] chunks with "embedding" key
        │
        ▼
  pgvector Storage ── PgVectorStore.add_vectors()
                       • Upsert embeddings table
                       • Update index_meta
                       • Return vector count
```

## Component Details

### `main.py` — FastAPI Application & HTTP Routes

The single-file web server that:
- Creates the FastAPI app and configures CORS.
- Initialises the database schema via `db.init_db()` on startup.
- Defines all HTTP route handlers for indexing, searching, embedding generation, and job status.
- Handles authentication via `Authorization: Bearer` tokens.
- Processes zip file uploads, GitHub repository cloning (via subprocess calls to `git`), and local directory indexing.
- Constructs the indexing pipeline (calling into `indexing/full_pipeline.py`) and the search pipeline (calling into `store/pgvector_store.py` and `lib/embedder.py`).
- Sends webhook notifications on job completion with HMAC-SHA256 signatures.
- Serves the static web UI from `web/index.html` at the root path.

### `db.py` — Database Layer

- Manages a `ThreadedConnectionPool` backed by `psycopg2`.
- Provides `init_db()` which creates the `vector` extension and all tables/indexes.
- Handles the `DATABASE_URL` / `LOCAL_DATABASE_URL` fallback chain.
- Automatically appends `sslmode=require` for production connections.
- Includes a migration that drops and recreates the `embeddings` table if the vector dimension is not 768 (to handle past schemas with wrong dimensions).
- Exposes CRUD functions: `create_job`, `update_job`, `get_job`, `upsert_index_meta`, `get_index_meta`, `delete_embeddings`, `count_embeddings`.

### `analyzers/` — Per-Language AST Analysis

Each analyzer module exports an `analyze_file(file_path: str) -> FileAnalysis` function. The `FileAnalysis` dataclass contains:

- `file_path`: Relative path to the source file.
- `language`: Detected programming language.
- `functions`: List of `FunctionInfo` objects (name, parameters, return type, docstring, line range).
- `classes`: List of `ClassInfo` objects (name, methods, docstring, line range).
- `imports`: List of imported module/symbol names.
- `symbols`: Flat list of all top-level symbol names.

**Python** (`python_analyzer.py`) uses the built-in `ast` module. All other languages use [tree-sitter](https://tree-sitter.github.io/) grammars for robust parsing.

### `utils/language_router.py` — Language Detection & Dispatch

Maps file extensions to analyzer modules and language names. Provides the `analyze_file(file_path: str) -> FileAnalysis | None` entry point that all indexing code uses.

### `utils/file_scanner.py` — File Discovery

Recursively scans a directory for source files, applying filters:
- Respects `.gitignore` rules (via `pathspec` library).
- Filters by a configurable set of indexable file extensions.
- Skips common non-code directories (`node_modules`, `__pycache__`, `.git`, `venv`, etc.).

### `lib/chunker.py` — Code Chunking

Converts `FileAnalysis` objects into embedding-ready chunks. Each function and class becomes a separate chunk with its source code as the `content` field. Module-level code (outside any function/class) is captured as a "module" chunk. Metadata includes file path, symbol name, line number, chunk type, and extracted imports.

### `lib/graph.py` — Import & Call Graphs

- **`build_import_graph(analyses)`** — Constructs a dictionary mapping each file path to the set of file paths it imports. Uses the imports extracted during AST analysis and resolves them against the scanned file list.
- **`build_call_graph(analyses)`** — Constructs a dictionary mapping each symbol name to the set of symbol names it references (calls, instantiates, or accesses). Uses a combination of AST-based reference extraction and heuristic name matching.
- **`enrich_chunks_with_graph(chunks, import_graph)`** — Adds an `imported_files` field to each chunk's metadata listing the files that the chunk's source file imports.
- **`enrich_chunks_with_call_graph(chunks, call_graph)`** — Adds a `called_functions` field to each chunk's metadata listing the symbols referenced by the chunk.

### `lib/embedder.py` — Embedding Models

- **`CodeEmbedder`** — Local embedding using HuggingFace transformers. Loads `AutoTokenizer` and `AutoModel`, encodes text with masked mean pooling and L2 normalisation. Falls back to `SentenceTransformer` for non-CodeBERT models. Gracefully degrades to random embeddings if models fail to load.
- **`VoyageEmbedder`** — Cloud embedding via Voyage AI REST API. Batches requests (up to 128 texts per request) and returns 1024-dimensional embeddings. Requires `VOYAGE_API_KEY`.
- **Model selection** — The pipeline compares the chunk count against `LARGE_CODEBASE_THRESHOLD` to decide which embedder to use. This decision and the resulting model metadata are persisted in `index_meta` so that the search endpoint can recreate the correct embedder.

### `store/pgvector_store.py` — Vector Storage

Wraps PostgreSQL/pgvector operations for the indexing and search paths:

- **`add_vectors(embeddings, metadata, replace_all=True)`** — When `replace_all=True`, deletes all existing vectors for the namespace/project before inserting. This ensures a clean index on each re-run without stale chunks from deleted or renamed files.
- **`search(query_vector, k)`** — Performs a cosine similarity query using the ivfflat index and returns the top-k results with their metadata and similarity scores.

### `store/chroma_store.py` — Alternative Vector Storage

Provides a ChromaDB-backed vector store as an alternative to pgvector. Used by the `full_pipeline_chroma()` pipeline. Stores call and import graphs as pickle files alongside the Chroma directory.

### `store/project_store.py` — JSON Project Records

Simple file-based store for project metadata, used by the Chroma pipeline.

### `indexing/full_pipeline.py` — Pipeline Orchestration

Exposes two high-level functions:

- **`full_pipeline_pgvector(repo_path, namespace, project_id, ...)`** — End-to-end indexing into pgvector. Returns the store, call graph, import graph, and index metadata.
- **`full_pipeline_chroma(repo_path, chroma_dir, ...)`** — End-to-end indexing into ChromaDB. Returns the store, call graph, and import graph. Also persists graphs and index metadata as local files.

Both follow the same internal steps (file discovery → analysis → graphs → chunking → enrichment → embedding → storage) but differ in where the final vectors are stored.

## Embedding Model Selection

```
chunks.length > LARGE_CODEBASE_THRESHOLD?
       │
       ├── YES ──► VoyageEmbedder (cloud)
       │           • voyage-code-3 model
       │           • 1024 dimensions
       │           • Requires VOYAGE_API_KEY
       │           • Batched API calls (128 per request)
       │
       └── NO  ──► CodeEmbedder (local)
                   • microsoft/codebert-base model
                   • 768 dimensions
                   • No external dependencies
                   • GPU-accelerated when available
```

The chosen model and its dimension are recorded in `index_meta` so that `/search/text` can instantiate the correct embedder at query time, preventing dimension-mismatch errors.

## Authentication

API endpoints (except `/search`, `/search/text`, `/embed`, and the web UI) require an `Authorization: Bearer <token>` header. Token validation is performed in the route handlers.

Webhook notifications are signed with HMAC-SHA256 using the `WEBHOOK_SECRET` environment variable. The `X-Signature-256` header contains the hex-encoded digest, computed over the JSON response body. Recipients can verify authenticity by recomputing the digest with the shared secret.

## Webhooks

When a job completes (success or failure) and a `webhook_url` was provided, the service sends a POST request with:

- **Headers**: `Content-Type: application/json`, `X-Signature-256: <hmac-sha256-hex>`
- **Body**: JSON containing `job_id`, `status`, `total_vectors`, `error` (if any), and other job fields.

## Error Handling

- **Indexing failures** are caught at the pipeline level, the job status is set to `"failed"`, and the error message is stored in the `jobs.error` column.
- **Embedding model load failures** fall back to random embeddings (local embedder) or raise a clear error (cloud embedder requiring API key).
- **Database dimension migrations** are handled automatically: if the `embeddings` table has the wrong vector dimension, it is dropped and recreated on `init_db()`.
- **ChromaDB SQLITE_READONLY_DBMOVED** is avoided by using `replace_all=True` upserts instead of `clear()` + `add_vectors()` (which triggers SQLite VACUUM file renaming).
