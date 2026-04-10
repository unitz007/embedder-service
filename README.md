# Embedder Service

A FastAPI-based HTTP service that indexes source code repositories, generates vector embeddings from parsed AST structures, and provides semantic code search via PostgreSQL/pgvector. The service supports indexing from uploaded zip files, local directories, or GitHub repositories, and offers both local (CodeBERT) and cloud (Voyage AI) embedding models.

## Features

- **Multi-language AST analysis** — Parses Python, Go, JavaScript, TypeScript, Rust, Java, Ruby, C/C++, Lua, Bash, and C# source files to extract functions, classes, imports, and file-level metadata.
- **Dual embedding backends** — Uses `microsoft/codebert-base` for local inference (768-d) or `voyage-code-3` via Voyage AI for large codebases (1024-d), with automatic selection based on chunk count.
- **GitHub integration** — Download and index GitHub repositories directly via the API, supporting private repos with OAuth tokens.
- **Semantic search** — Query indexed codebases with raw text or pre-computed embeddings, with automatic model detection and dimension-mismatch handling.
- **Job system** — Asynchronous indexing jobs with status tracking, webhook notifications, and persistent state in PostgreSQL.
- **Import & call graphs** — Builds file-level import graphs and symbol-level call graphs that enrich chunk metadata for richer retrieval context.
- **Multi-tenant storage** — Namespace/project-scoped embeddings stored in pgvector, suitable for serving multiple teams or organisations.

## Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL 14+ with the [pgvector extension](https://github.com/pgvector/pgvector)
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/unitz007/embedder-service.git
cd embedder-service

# Install dependencies
pip install -r requirements.txt
```

### Database Setup

```bash
# Enable the pgvector extension in your PostgreSQL database
psql -c "CREATE EXTENSION IF NOT EXISTS vector;" your_database
```

### Configuration

<!-- TODO: fill in once deployment environment is confirmed -->

Set the following environment variables before starting the service:

| Variable | Description | Default |
|---|---|---|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://postgres:postgres@localhost:5432/indexer` |
| `LOCAL_DATABASE_URL` | Fallback database URL for non-production environments | — |
| `APP_ENV` | Application environment (`dev` or `prod`) | `dev` |
| `VOYAGE_API_KEY` | Voyage AI API key for cloud embeddings | — |
| `WEBHOOK_SECRET` | HMAC secret for signing webhook notifications | — |
| `GITHUB_API_BASE` | GitHub API base URL | `https://api.github.com` |
| `GITHUB_API_VERSION` | GitHub API version header | `2022-11-28` |
| `DB_POOL_MAX` | Maximum database connection pool size | `20` |

### Running the Service

```bash
# Development (with auto-reload)
python main.py

# Or using uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The web UI is available at `http://localhost:8000/`.

## API Overview

<!-- TODO: fill in once OpenAPI schema is finalised -->

All mutating endpoints (except search/embed) require an `Authorization: Bearer <token>` header.

### Indexing

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/embeddings/{namespace}/{project_id}` | Upload a zip file and index its contents (async) |
| `POST` | `/github/index` | Index a GitHub repository or local directory (async) |
| `GET` | `/embeddings/{namespace}/{project_id}` | Get index metadata and vector count |
| `DELETE` | `/embeddings/{namespace}/{project_id}` | Delete all embeddings for a project |

### Search

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/search` | Semantic search using a pre-computed embedding vector |
| `POST` | `/search/text` | Semantic search using raw query text (server picks the model) |
| `POST` | `/embed` | Generate an embedding for arbitrary text |

### Jobs

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/jobs/{job_id}` | Get the status and result of an indexing job |

### Example Requests

```bash
# Index a GitHub repository
curl -X POST http://localhost:8000/github/index \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"owner": "unitz007", "repo": "embedder-service"}'

# Search by text
curl -X POST http://localhost:8000/search/text \
  -H "Content-Type: application/json" \
  -d '{"namespace": "unitz007", "project_id": "embedder-service", "query": "How does embedding work?", "k": 5}'

# Generate an embedding
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "def hello_world(): pass"}'
```

## Supported Languages

| Language | Parser | Features |
|---|---|---|
| Python | `ast` (stdlib) | Functions, classes, imports, docstrings, parameters |
| Go | tree-sitter-go | Functions, methods, structs, interfaces |
| JavaScript | tree-sitter-javascript | Functions, classes, imports |
| TypeScript | tree-sitter-typescript | Functions, classes, imports, interfaces |
| Rust | tree-sitter-rust | Functions, structs, traits, impls |
| Java | tree-sitter-java | Classes, methods, interfaces, enums |
| Ruby | tree-sitter-ruby | Classes, modules, methods |
| C | tree-sitter-c | Functions, structs, enums |
| C++ | tree-sitter-cpp | Classes, functions, namespaces, templates |
| Lua | tree-sitter-lua | Functions |
| Bash | tree-sitter-bash | Functions |
| C# | tree-sitter-c-sharp | Classes, methods, interfaces, structs |

## Development

<!-- TODO: fill in once test suite and CI are established -->

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ -v --cov=lib --cov=store --cov=analyzers --cov=utils --cov=indexing
```

### Project Structure

```
.
├── main.py                   # FastAPI application, routes, and request handling
├── db.py                     # PostgreSQL connection pool, schema init, CRUD operations
├── models.py                 # Data models (FileAnalysis, FunctionInfo, ClassInfo)
├── requirements.txt          # Python dependencies
├── analyzers/                # Per-language AST analysis modules
│   ├── python_analyzer.py
│   ├── go_analyzer.py
│   ├── js_analyzer.py
│   ├── generic_ts_analyzer.py
│   ├── config_analyzer.py
│   └── shell_analyzer.py
├── indexing/
│   └── full_pipeline.py      # End-to-end indexing pipeline orchestration
├── lib/
│   ├── embedder.py           # CodeEmbedder (local) and VoyageEmbedder (cloud)
│   ├── chunker.py            # Code chunking with metadata extraction
│   ├── graph.py              # Import graph and call graph construction
│   └── indexer.py            # Indexing utility functions
├── store/
│   ├── pgvector_store.py     # PostgreSQL/pgvector vector storage
│   ├── chroma_store.py       # ChromaDB vector storage (alternative backend)
│   └── project_store.py      # JSON-file-backed project record store
├── utils/
│   ├── file_scanner.py       # Recursive file discovery with ignore rules
│   └── language_router.py    # Language detection and analyzer dispatch
├── web/
│   └── index.html            # Web UI for the service
├── README.md                 # This file
└── ARCHITECTURE.md           # Architecture documentation
```

## License

<!-- TODO: add license file and update this link -->

See [LICENSE](./LICENSE) for details.
