# Code Indexer

A modular Python codebase indexer that parses source files across multiple languages (Python, Go, JavaScript, TypeScript), extracts structured metadata via AST analysis, builds import and call graphs, and produces embedding-ready chunks for retrieval-augmented generation (RAG) systems.

## Features

- **Multi-language AST analysis** — Extracts functions, classes, imports, and file-level metadata from Python, Go, JavaScript, and TypeScript source files.
- **Dependency graphs** — Builds both file-level import graphs and symbol-level call graphs to capture codebase structure and relationships.
- **Smart chunking** — Breaks analyzed files into embedding-ready chunks by symbol (function, class) with optional context lines, preserving structural metadata for retrieval.
- **LangChain integration** — Includes LangChain-compatible document loaders and vector store indexers for seamless integration with LLM-powered RAG pipelines.
- **LLM-friendly metadata** — Every chunk carries structured metadata (signatures, docstrings, parameters, return types, graph edges) that can be injected into LLM context for better code understanding.
- **Flexible file scanning** — Recursive repository scanner with configurable ignore lists for version control, build artifacts, generated code, binaries, and IDE files.

## Project Structure

```
.
├── lib/
│   ├── analyzer.py        # Multi-language AST analysis (functions, classes, imports)
│   ├── chunker.py         # File & symbol chunking with metadata extraction
│   ├── graph.py           # Import graph (file-level) and call graph (symbol-level)
│   ├── langchain_indexer.py  # LangChain document loader & vector store integration
│   └── query.py           # Query helpers for filtered retrieval
├── models/
│   ├── __init__.py        # Pydantic models: FileAnalysis, FunctionInfo, ClassInfo
│   └── chunk_models.py    # Chunk metadata schema definitions
├── store/
│   └── project_store.py   # Thread-safe JSON-file-backed project record store
├── utils/
│   └── file_scanner.py    # Recursive file discovery with ignore rules
├── tests/
│   ├── test_chunker.py    # Chunker unit tests
│   ├── test_analyzer.py   # Analyzer unit tests
│   └── test_file_scanner.py  # File scanner unit tests
├── Dockerfile             # Multi-stage Docker build
├── .dockerignore          # Docker build exclusions
├── pyproject.toml         # Project metadata & dependencies
└── README.md              # This file
```

## Quick Start

### Prerequisites

- Python 3.11+
- pip

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd code-indexer

# Install dependencies
pip install -e .
```

### Basic Usage

```python
from utils.file_scanner import scan_repository
from models import FileAnalysis
from lib.analyzer import analyze_files
from lib.chunker import chunk_repository_analyses
from lib.graph import build_import_graph, build_call_graph, enrich_chunks_with_graph

# 1. Scan a repository for source files
files = scan_repository("/path/to/repo")

# 2. Analyze files to extract AST metadata
analyses = analyze_files(files)

# 3. Build dependency graphs
import_graph = build_import_graph(analyses)
call_graph = build_call_graph(analyses)

# 4. Chunk the analyses into embedding-ready pieces
chunks = chunk_repository_analyses(analyses)

# 5. Enrich chunks with graph data
chunks = enrich_chunks_with_graph(chunks, import_graph)
chunks = enrich_chunks_with_call_graph(chunks, call_graph)

# Each chunk has:
#   - content: source text for embedding
#   - type: "function", "class", "imports", "file_summary", or "file"
#   - metadata: structured fields (file_path, language, signature, params, etc.)
```

### LangChain Integration

```python
from lib.langchain_indexer import CodeIndexer

# Create a LangChain-compatible vector store index
indexer = CodeIndexer(embedding_model="text-embedding-3-small")
indexer.build_index("/path/to/repo")

# Query the index
results = indexer.similarity_search("How does authentication work?", k=5)
```

## Supported Languages

| Language   | AST Parser         | Features                                       |
|------------|--------------------|-------------------------------------------------|
| Python     | `ast` (stdlib)     | Functions, classes, imports, docstrings, params |
| Go         | tree-sitter-go     | Functions, methods, structs, interfaces         |
| JavaScript | tree-sitter-javascript | Functions, classes, imports               |
| TypeScript | tree-sitter-typescript | Functions, classes, imports, interfaces    |

## Docker

```bash
# Build the Docker image
docker build -t code-indexer .

# Run indexing in a container
docker run -v /path/to/repo:/workspace code-indexer
```

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ -v --cov=lib --cov=store --cov=utils --cov=models
```

## Key Concepts

### Chunk Types

| Type           | Description                                              |
|----------------|----------------------------------------------------------|
| `imports`      | All import statements from a file                        |
| `function`     | A single function/method with context lines              |
| `class`        | A class/struct/interface with context lines              |
| `file_summary` | Compact file-level overview (functions, types, imports)  |
| `file`         | Full file content (fallback when no symbols are found)   |

### Graph Types

- **Import Graph** — File-level dependencies showing `depends_on` and `imported_by` relationships between source files.
- **Call Graph** — Symbol-level relationships showing which functions call which, with `calls`, `called_by`, and `external_calls` edges.

## License

See [LICENSE](./LICENSE) for details.
