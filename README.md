# Code Indexer Service

Semantic search infrastructure for codebases. Index any GitHub repository or local
project, generate embeddings with CodeBERT or Voyage AI, and query with natural
language over function signatures, docstrings, import graphs, and call graphs.

Built with **Python 3**, **FastAPI**, **Postgres/pgvector**, and **tree-sitter**.

## Features

- **Multi-language AST analysis** — Parses Python, Go, JavaScript/TypeScript,
  Rust, Java, Ruby, C, C++, Lua, shell scripts, and config files using
  tree-sitter grammars and Python's `ast` module.
- **Symbol extraction** — Extracts functions, methods, classes, structs,
  interfaces, imports, and docstrings from each source file.
- **Import & call graphs** — Builds file-level import dependency graphs and
  symbol-level call graphs across the entire codebase.
- **Semantic code search** — Embeds code chunks with
  [microsoft/codebert-base](https://huggingface.co/microsoft/codebert-base)
  (local, 768-d) or [Voyage AI voyage-code-3](https://docs.voyageai.com/)
  (cloud, 1024-d) and stores them in Postgres/pgvector for cosine-similarity
  search.
- **Automatic model selection** — Uses the local CodeBERT embedder for small
  codebases and automatically switches to Voyage AI for large ones.
- **GitHub integration** — Clone and index public or private GitHub repos by
  owner/repo with OAuth token support.
- **Async job pipeline** — Indexing runs in background tasks with full job
  lifecycle tracking (queued → downloading → running → completed/failed) and
  webhook notifications.
- **Dimension-mismatch handling** — The `/search/text` endpoint reads the
  stored index metadata and automatically picks the correct embedding model at
  query time, so callers never need to know which model was used at index time.
- **Built-in web UI** — A single-page documentation and API explorer served at
  the root URL.

## Quick Start

### Prerequisites

- Python 3.10+
- PostgreSQL with the [pgvector extension](https://github.com/pgvector/pgvector)
- (Optional) A [Voyage AI API key](https://docs.voyageai.com/docs/api-key) for
  indexing large codebases

### Install dependencies

```bash
pip install -r requirements.txt
```

### Database setup

```bash
# Ensure the pgvector extension is available in your Postgres instance.
# If using Supabase or a managed Postgres, it may already be installed.

# Set the database connection URL:
export DATABASE_URL="postgresql://user:password@localhost:5432/indexer"

# Or for local development without setting DATABASE_URL, the service
# defaults to: postgresql://postgres:postgres@localhost:5432/indexer
```

### Run the service

```bash
uvicorn main:app --reload --port 8080
```

The API is available at `http://localhost:8080` and the web UI at the root URL.

### Index a GitHub repository

```bash
curl -X POST http://localhost:8080/github/index \
  -H "Content-Type: application/json" \
  -d '{"owner": "vercel", "repo": "next.js"}'
```

Response:

```json
{
  "status": "queued",
  "job_id": "a3f9c1...",
  "namespace": "vercel",
  "project_id": "next.js"
}
```

Poll the job status:

```bash
curl http://localhost:8080/jobs/a3f9c1...
```

### Search with natural language

```bash
curl -X POST http://localhost:8080/search/text \
  -H "Content-Type: application/json" \
  -d '{
    "namespace": "vercel",
    "project_id": "next.js",
    "query": "how does file-system routing work",
    "k": 5
  }'
```

## API Overview

All endpoints are documented in the built-in web UI. Here's a summary:

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/github/index` | Open | Queue a GitHub repo or local path for indexing |
| `POST` | `/embeddings/{ns}/{pid}` | Bearer | Upload a zip archive and index it |
| `POST` | `/search/text` | Open | Search with raw query text (recommended) |
| `POST` | `/search` | Open | Search with a pre-computed embedding vector |
| `POST` | `/embed` | Open | Embed arbitrary text with the correct model |
| `GET` | `/embeddings/{ns}/{pid}` | Bearer | Get index metadata and vector count |
| `DELETE` | `/embeddings/{ns}/{pid}` | Bearer | Delete a project's index |
| `GET` | `/jobs/{job_id}` | Bearer | Poll indexing job status |

### Authentication

Most endpoints are open. Management endpoints (upload, delete, status) require
an `Authorization: Bearer <token>` header. The token value is not validated
against any specific identity — it simply must be present.

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | Postgres connection string | `postgresql://postgres:postgres@localhost:5432/indexer` |
| `LOCAL_DATABASE_URL` | Fallback for non-prod environments | (same as `DATABASE_URL`) |
| `APP_ENV` | Application environment (`dev`, `prod`) | `dev` |
| `VOYAGE_API_KEY` | Voyage AI API key (for large codebases) | — |
| `WEBHOOK_SECRET` | HMAC secret for webhook notifications | — |
| `GITHUB_API_BASE` | GitHub API base URL | `https://api.github.com` |
| `GITHUB_API_VERSION` | GitHub API version header | `2022-11-28` |
| `DB_POOL_MAX` | Postgres connection pool size | `20` |

## Supported Languages

| Language | Parser | Extracts |
|----------|--------|----------|
| Python | `ast` (stdlib) | Functions, async functions, classes, methods, imports, docstrings |
| Go | tree-sitter-go | Functions, methods (with receiver), structs, interfaces, type aliases, imports |
| JavaScript / TypeScript | tree-sitter-javascript | Function declarations, arrow functions, classes, methods, exports, imports |
| Rust | tree-sitter-rust | Functions, impl methods, structs, enums, traits, use declarations |
| Java | tree-sitter-java | Methods, constructors, classes, interfaces, enums, imports |
| Ruby | tree-sitter-ruby | Methods, singleton methods, classes, modules, requires |
| C | tree-sitter-c | Functions, structs, enums, unions, `#include` directives |
| C++ | tree-sitter-cpp | Functions, classes, structs, namespaces, enums, `#include` directives |
| Lua | tree-sitter-lua | Functions, local functions, require calls |
| Shell (bash/zsh/fish) | Regex | Functions, aliases, exports, source/include paths |
| Config files | stdlib parsers | Sections, key-value pairs (YAML, TOML, JSON, INI, .env, .gitignore, etc.) |

## Project Structure

```
.
├── main.py                     # FastAPI application and all HTTP endpoints
├── db.py                       # Postgres connection pool, schema init, CRUD
├── models.py                   # Dataclasses: FileAnalysis, FunctionInfo, ClassInfo
├── requirements.txt            # Python dependencies
├── analyzers/                  # Per-language AST/symbol analyzers
│   ├── python_analyzer.py      #   Python (ast module)
│   ├── go_analyzer.py          #   Go (tree-sitter)
│   ├── js_analyzer.py          #   JavaScript (tree-sitter)
│   ├── generic_ts_analyzer.py  #   Rust, Java, Ruby, C, C++, Lua (config-driven)
│   ├── shell_analyzer.py       #   Bash/zsh/fish (regex)
│   └── config_analyzer.py      #   YAML, TOML, JSON, INI, .env, etc.
├── indexing/
│   ├── full_pipeline.py        # Orchestrates: scan → parse → chunk → embed → store
│   └── USAGE_GUIDE.md          # Standalone LLM analysis guide
├── lib/
│   ├── chunker.py              # Splits FileAnalysis into embedding-ready chunks
│   ├── embedder.py             # CodeBERT (local) and Voyage AI (cloud) embedders
│   ├── graph.py                # Import graph and call graph builders
│   └── indexer.py              # Thin wrapper: scan + analyze
├── store/
│   ├── pgvector_store.py       # pgvector-backed vector store (primary)
│   ├── chroma_store.py         # ChromaDB-backed vector store (alternative)
│   └── project_store.py        # JSON-file project metadata store
├── utils/
│   ├── file_scanner.py         # Repository walker with ignore rules
│   └── language_router.py      # File extension → language → analyzer dispatch
├── web/
│   └── index.html              # Built-in API documentation UI
├── .dockerignore
├── .gitignore
└── ARCHITECTURE.md             # Detailed architecture documentation
```

## How Indexing Works

1. **Scan** — Walk the repository, skip VCS/build/generated files (configurable
   ignore rules in `utils/file_scanner.py`).
2. **Detect language** — Map each file to a language via its extension or name
   (`utils/language_router.py`).
3. **Analyze** — Run the appropriate parser to extract functions, classes,
   imports, and docstrings into `FileAnalysis` objects.
4. **Build graphs** — Construct a file-level import graph and a symbol-level
   call graph across the entire codebase (`lib/graph.py`).
5. **Chunk** — Split each file into embedding-ready chunks: one per function,
   one per class, one for imports, and a file-level summary (`lib/chunker.py`).
   Graph metadata (callers, callees, dependencies) is injected into each chunk.
6. **Embed** — Generate vector embeddings. Small codebases use local CodeBERT
   (768-d); large ones use Voyage AI voyage-code-3 (1024-d) (`lib/embedder.py`).
7. **Store** — Upsert vectors into Postgres/pgvector with cosine-similarity
   indexing (`store/pgvector_store.py`). Metadata includes file path, symbol
   name, signature, docstring, params, return type, and graph relationships.

## Development

```bash
# Run with auto-reload
uvicorn main:app --reload

# Run on a different port
uvicorn main:app --port 3000
```

## License

This project is licensed under the MIT License.
