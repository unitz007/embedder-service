# Architecture

## Overview

Code Indexer is a modular pipeline that transforms source code repositories into structured, embedding-ready chunks suitable for retrieval-augmented generation (RAG). The system operates in discrete stages — scanning, analysis, graphing, and chunking — with each stage producing output that the next stage consumes.

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Scanner    │────▶│   Analyzer   │────▶│    Graph     │────▶│   Chunker    │
│ (file list)  │     │  (AST parse) │     │  (deps/calls)│     │  (embed text)│
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                                                                   │
                                                                   ▼
                                                           ┌──────────────┐
                                                           │ LangChain    │
                                                           │ Indexer      │
                                                           │ (vector DB)  │
                                                           └──────────────┘
```

## Pipeline Stages

### 1. File Scanning (`utils/file_scanner.py`)

Entry point for discovering source files in a repository.

- Recursively walks a directory tree using `os.walk`
- Prunes ignored directories (VCS, build artifacts, node_modules, vendor, etc.)
- Filters out binary files by extension, generated/minified files by suffix, and files exceeding 512 KB
- Optionally includes hidden/dotfiles for dotfile repository support
- Outputs: `List[str]` — absolute file paths

### 2. AST Analysis (`lib/analyzer.py`)

Parses each source file to extract structured metadata.

**Per-language parsers:**

| Language   | Parser                   | Technology          |
|------------|--------------------------|---------------------|
| Python     | `_analyze_python`        | `ast` (stdlib)      |
| Go         | `_analyze_go`            | tree-sitter-go      |
| JavaScript | `_analyze_javascript`    | tree-sitter-js      |
| TypeScript | `_analyze_typescript`    | tree-sitter-ts      |

**Extracted per file:**
- **Functions**: name, line number, signature, parameters, return type, docstring
- **Classes/structs/interfaces**: name, line number, kind (class/struct/interface), docstring
- **Imports**: raw import statement strings
- **Package**: module or package path
- **Language**: detected language identifier

**Outputs:** `List[FileAnalysis]` — Pydantic model instances defined in `models/__init__.py`

### 3. Graph Construction (`lib/graph.py`)

Builds two complementary dependency graphs from the analysis results.

#### Import Graph (file-level)

```
FileA.py ──depends_on──▶ FileB.py
FileC.py ──depends_on──▶ FileA.py
```

- Resolves import strings to actual file paths within the repository
- Language-specific resolution logic (Python dotted imports, Go package paths, JS relative paths)
- Produces bidirectional edges: `depends_on` and `imported_by`
- Tracks unresolved imports (stdlib, third-party)

#### Call Graph (symbol-level)

```
UserService.authenticate() ──calls──▶ DbConnector.query()
AuthService.verify() ────────calls──▶ UserService.authenticate()
```

- Parses each function body to extract called names
- Resolves calls to known functions across the codebase using a symbol lookup table
- Handles qualified calls (e.g., `obj.method()`) by extracting the method name
- Produces bidirectional edges: `calls` and `called_by`
- Tracks external/unresolved calls separately

**Graph enrichment:** Both graphs are injected into chunk metadata via `enrich_chunks_with_graph()` and `enrich_chunks_with_call_graph()`.

### 4. Chunking (`lib/chunker.py`)

Transforms `FileAnalysis` objects into embedding-ready chunks.

**Chunking strategy:**
1. Read each source file once (shared across all extraction calls)
2. Extract imports as a dedicated chunk
3. Extract each function body with surrounding context lines (configurable, default 10 lines before, 20 after)
4. Extract each class/struct/interface body with context (10 before, 30 after)
5. Always generate a compact file-level summary chunk listing all symbols
6. If no symbols are found, fall back to full-file content chunk

**Body extraction:**
- Python: Indent-based boundary detection
- Go/JS/TS: Brace-matching with recursive tree-sitter node walking

**Each chunk contains:**
- `content`: Source text used for generating embeddings
- `type`: One of `imports`, `function`, `class`, `file_summary`, `file`
- `metadata`: Structured fields including:
  - `file_path`, `language`, `package`
  - `function_name`, `class_name`, `line_number`
  - `signature`, `docstring`, `params`, `return_type`
  - `depends_on`, `imported_by` (from import graph)
  - `calls`, `called_by`, `external_calls` (from call graph)
  - `symbol_type`, `symbol_name` (normalized identifiers)

### 5. LangChain Integration (`lib/langchain_indexer.py`)

Provides LangChain-compatible abstractions for embedding and retrieval.

- **Document loaders**: Convert chunks into LangChain `Document` objects with metadata preserved
- **Vector store indexer**: Build and query a vector index using any LangChain-compatible embedding model
- Enables RAG workflows where code chunks are retrieved based on semantic similarity to user queries

### 6. Query Helpers (`lib/query.py`)

Provides filtered retrieval utilities for accessing specific chunk types or metadata-filtered subsets of the index.

## Data Models (`models/`)

### `FileAnalysis`
Core analysis result per source file. Fields: `file_path`, `language`, `package`, `functions` (List[FunctionInfo]), `classes` (List[ClassInfo]), `imports` (List[str]).

### `FunctionInfo`
Metadata per function/method. Fields: `name`, `line`, `signature`, `docstring`, `params`, `return_type`.

### `ClassInfo`
Metadata per type definition. Fields: `name`, `line`, `kind` (class/struct/interface), `docstring`.

### `ChunkMetadata`
Structured metadata attached to each chunk for LLM context injection and BM25 scoring. Defined in `models/chunk_models.py`.

## Storage (`store/`)

### Project Store (`store/project_store.py`)

Thread-safe, JSON-file-backed CRUD store for project records. Projects are first-class entities that can optionally be linked to GitHub repositories.

- Storage: Single JSON file at `{data_dir}/projects.json`
- Thread safety: `threading.Lock` around all read/write operations
- Operations: `create`, `get`, `list_all`, `update`, `delete`
- Fields: `id`, `name`, `namespace`, `description`, `github_owner`, `github_repo`, `github_ref`, `created_at`, `updated_at`

## Data Flow Diagram

```
Repository
    │
    ▼
scan_repository()
    │  List[str] (file paths)
    ▼
analyze_files()
    │  List[FileAnalysis]
    ▼
┌─────────────────────────────┐
│ build_import_graph()        │──▶ enrich_chunks_with_graph()
│ build_call_graph()          │──▶ enrich_chunks_with_call_graph()
└─────────────────────────────┘
    │
    ▼
chunk_repository_analyses()
    │  List[Dict] (chunks with content + metadata)
    ▼
LangChain Indexer
    │  Vector store (embeddings + metadata index)
    ▼
RAG Query → Retrieved chunks → LLM context
```

## Design Decisions

1. **Single-pass file reads**: Each source file is read exactly once during chunking, shared across all extraction calls for that file.

2. **Separate graph and chunk stages**: Graphs are built from analyses and then injected into chunks, keeping the graph logic independent and testable.

3. **Metadata over embedding**: Structured metadata (signatures, params, graph edges) is stored alongside embedding vectors, enabling both semantic search and structured filtering.

4. **Always-present file summary**: Even when individual symbol searches miss, the file summary chunk provides a fallback for locating relevant files.

5. **Language-specific parsers**: Each language uses the most appropriate parsing tool — stdlib `ast` for Python, tree-sitter for Go/JS/TS — rather than a one-size-fits-all approach.

6. **JSON file storage**: The project store uses plain JSON files for simplicity and zero operational dependencies, suitable for single-node deployments.
