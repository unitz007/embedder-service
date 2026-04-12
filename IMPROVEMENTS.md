# Changelog — Shipped Features

A reverse-chronological record of features that have been implemented and merged.

---

## 1. pgvector Pipeline — Production Multi-Tenant Storage

**PR:** Referenced in task description (pgvector pipeline)  
**Description:** Replaced FAISS with a PostgreSQL-backed vector store for production multi-tenant deployments. The `PgVectorStore` class (`store/pgvector_store.py`) stores embeddings in a `vector(768)` column with IVFFlat cosine similarity indexing. Uses `INSERT … ON CONFLICT DO UPDATE` for idempotent upserts and batched inserts (200 rows per batch). The `db.py` module manages a `ThreadedConnectionPool`, auto-creates the pgvector extension, and provides CRUD for `jobs`, `index_meta`, and `embeddings` tables. The pipeline function `full_pipeline_pgvector()` in `indexing/full_pipeline.py` orchestrates the full indexing flow into Postgres.

**Affected files:**
- `store/pgvector_store.py`
- `db.py`
- `indexing/full_pipeline.py`
- `pipeline.py`

---

## 2. Per-Tenant Model Routing

**PRs:** #12, #21  
**Description:** Added a `preferred_embedder` column to the `index_meta` table, allowing each tenant to override the automatic embedder selection. New API endpoints `PUT /api/index/<namespace>/<project_id>/model` (set override) and `GET /api/index/<namespace>/<project_id>/model` (read current config) were added to `main.py`. The `preferred_embedder` value is also accepted in the `POST /api/index` request body and is honoured by both `full_pipeline_pgvector()` and `full_pipeline_chroma()`. Setting it to `"voyage-code-3"` forces cloud embeddings; setting to `null` reverts to automatic selection.

**Affected files:**
- `main.py` — new routes `PUT`/`GET` `/api/index/<namespace>/<project_id>/model`; `POST /api/index` accepts `preferred_embedder`
- `db.py` — `preferred_embedder` column on `index_meta`; `update_preferred_embedder()` function
- `indexing/full_pipeline.py` — `full_pipeline_pgvector()` and `full_pipeline_chroma()` accept `preferred_embedder` parameter

---

## 3. Generic Type Parameter Extraction (Go 1.18+)

**PR:** #14  
**Description:** The Go analyzer now captures type parameters from generic type declarations. For example, `type Repository[T comparable] struct { ... }` records `"T comparable"` in the metadata. This data is available in chunk metadata for the `class` chunk type under the `ClassInfo` model. The extraction uses tree-sitter-go's `type_param` node type.

**Affected files:**
- `models.py` — `ClassInfo` dataclass (type_params support)

---

## 4. Decorator Extraction (Python)

**PR:** #24  
**Description:** Python function and method nodes now include a `decorators` list in `FunctionInfo`. Each decorator is unparsed from the AST into its source representation (e.g. `"@staticmethod"`, `"@app.route('/health')"`). The list preserves source order (top-to-bottom). Extraction uses `ast.unparse()` on each node in `decorator_list`, with a try/except fallback to skip unparsable decorators.

**Affected files:**
- `models.py` — `FunctionInfo` dataclass, `decorators: List[str]` field
- `analyzers/python_analyzer.py` — `_get_decorators()` function; `_process_function()` populates decorators

---

## 5. Constant and Variable Extraction

**PR:** #19  
**Description:** Introduced a new `VariableInfo` dataclass and variable extraction across multiple analyzers. Extracted variable types include:

- **Go**: `const` and `var` declarations (including parenthesised blocks), with type annotations and leading doc comments
- **Python**: module-level `ALL_CAPS` assignments (constant heuristic), annotated assignments, and class-level class variables
- **JavaScript/TypeScript**: `const`/`let`/`var` declarations with non-function values
- **Generic tree-sitter**: Rust `const_item`/`static_item`, C/C++ file-scope declarations with initialisers

Each `VariableInfo` stores name, line, kind (`"const"`, `"var"`, `"static"`, `"class_var"`, etc.), value (truncated to 120–200 chars), type annotation, and docstring. Variables are emitted as their own `variable` chunk type in the chunker.

**Affected files:**
- `models.py` — new `VariableInfo` dataclass
- `analyzers/go_analyzer.py` — `_extract_variables()` function
- `analyzers/python_analyzer.py` — `_extract_module_level_variables()`, `_extract_class_level_variables()`
- `analyzers/js_analyzer.py` — variable extraction in `_extract_from_node()`
- `analyzers/generic_ts_analyzer.py` — `var_types`, `var_kind_map`, `var_value_field` in `LanguageProfile`; variable extraction loop in `analyze_generic()`
- `lib/chunker.py` — variable chunk generation in `chunk_file_analysis()`

---

## 6. Error-Tolerant Parsing (Python)

**PR:** #26  
**Description:** The Python analyzer now gracefully handles files with syntax errors. When `ast.parse()` raises a `SyntaxError`, the analyzer falls back to a regex-based extraction that recovers top-level `def`, `async def`, `class`, `import`, and `from … import` statements. This means a file with a syntax error on line 200 still produces chunks for all valid symbols on lines 1–199. The regex fallback also attempts to extract docstrings (triple-quoted strings) following each definition.

**Affected files:**
- `analyzers/python_analyzer.py` — `_regex_fallback()` function; `_RE_DEF`, `_RE_CLASS`, `_RE_IMPORT`, `_RE_FROM_IMPORT` patterns; `_try_extract_docstring()` and `_collect_signature_lines()` helpers

---

## 7. Struct Tag Extraction (Go)

**PR:** #17  
**Description:** The Go analyzer now extracts field-level struct tags from `struct_type` nodes. Each field in a Go struct produces a `FieldInfo` dataclass with three fields: `name` (field name), `type_str` (e.g. `"string"`, `"time.Time"`), and `tag` (e.g. `` `json:"id" db:"id"` ``). Tags are parsed from tree-sitter's `tag` field on `field_definition` nodes. The `ClassInfo` model's `fields` list carries these for `struct`-kind classes. Struct tags are included in chunk metadata and available in the vector store for LLM context injection.

**Affected files:**
- `models.py` — new `FieldInfo` dataclass; `ClassInfo.fields: List[FieldInfo]`
- `analyzers/go_analyzer.py` — `_extract_struct_fields()` function; `ClassInfo` construction in type_declaration handler
- `lib/chunker.py` — struct field metadata included in class chunk metadata
