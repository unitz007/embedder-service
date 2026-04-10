# Potential Improvements & Known Limitations

This document catalogues actionable improvements and known limitations across
the indexing and analysis codebase.  Each entry includes the **why**, a
suggested approach, the files that would need to change, and an
**effort-impact rating**.

---

## Table of Contents

1. [Innovative & Forward-Looking Proposals](#1-innovative--forward-looking-proposals)
2. [Python Analyzer](#2-python-analyzer)
3. [Go Analyzer](#3-go-analyzer)
4. [JavaScript Analyzer](#4-javascript-analyzer)
5. [Generic Tree-Sitter Analyzer](#5-generic-tree-sitter-analyzer)
6. [Shell Analyzer](#6-shell-analyzer)
7. [Config Analyzer](#7-config-analyzer)
8. [Language Router](#8-language-router)
9. [Models & Data Layer](#9-models--data-layer)
10. [Indexing Pipeline](#10-indexing-pipeline)
11. [Search & Retrieval](#11-search--retrieval)
12. [LLM Integration](#12-llm-integration)
13. [Testing](#13-testing)
14. [Operational & DevEx](#14-operational--devex)

---

## 1. Innovative & Forward-Looking Proposals

### 1.1. Semantic caching layer for embedding reuse
*[Impact: High | Effort: Medium]*

**Problem:** During re-indexing, unchanged code chunks are re-embedded every
time. For a 10,000-chunk codebase where only 200 files changed, this wastes
~95% of embedding compute and, when using the Voyage AI cloud embedder,
unnecessary API spend.

**Approach:** Maintain a content-hash → embedding cache (backed by the
existing Postgres store or a Redis sidecar). Before calling the embedder,
compute a SHA-256 hash of each chunk's `content` field and look it up in
the cache. On a cache hit, reuse the previous embedding vector without
calling the model. On a miss, embed normally and store the result keyed by
hash. This turns incremental indexing (§10.1) from "skip files with
unchanged hashes" into "skip individual *chunks* with unchanged content,"
which is more granular — a file where only one function changed reuses
embeddings for every other function.

**Files:** `lib/embedder.py` (add cache layer in `CodeEmbedder.encode` and
`VoyageEmbedder.encode`), `db.py` (new `embedding_cache` table), `indexing/full_pipeline.py`

### 1.2. Per-tenant model routing
*[Impact: High | Effort: Low]*

**Problem:** The multi-tenancy model already scopes all data by
`(namespace, project_id)` via `store/pgvector_store.py`, but model selection
in `indexing/full_pipeline.py` is purely based on codebase size
(`len(chunks) > LARGE_CODEBASE_THRESHOLD`). A paying customer who wants
Voyage AI embeddings for a small-but-critical codebase has no way to
override this, and a cost-sensitive tenant with a large codebase cannot opt
out of the cloud embedder.

**Approach:** Add an optional `preferred_embedder` column to the
`index_meta` table (`db.py`). When set (e.g. `"voyage-code-3"` or
`"microsoft/codebert-base"`), the pipeline uses that model regardless of
chunk count. When unset, the existing size-based heuristic applies. Expose
this via a `PUT /embeddings/{ns}/{pid}/config` endpoint in `main.py` that
accepts `{"preferred_embedder": "voyage-code-3"}` and stores it in
`index_meta`. This lets each tenant self-serve their embedding quality vs.
cost trade-off without any service restart or configuration file change.

**Files:** `db.py`, `indexing/full_pipeline.py`, `main.py`,
`store/pgvector_store.py`

### 1.3. Embedding cost attribution per tenant
*[Impact: Medium | Effort: Low]*

**Problem:** When using the Voyage AI cloud embedder, there is no tracking of
how many tokens or characters each tenant consumes. With multiple namespaces
and projects, it is impossible to attribute API costs back to specific
customers or teams, making pricing and budgeting opaque.

**Approach:** In `VoyageEmbedder.encode` (`lib/embedder.py`), accumulate
total character count per call and emit a structured log line with
`namespace`, `project_id`, `model`, `character_count`, and `timestamp`.
Store these records in a new `embedding_usage` table in `db.py` with
`(namespace, project_id, month)` granularity. Expose a
`GET /embeddings/{ns}/{pid}/usage` endpoint that returns cumulative
character/token counts and an estimated cost (based on Voyage AI's published
per-token pricing). This enables the operator to bill per-tenant and set
budget alerts.

**Files:** `lib/embedder.py`, `db.py`, `main.py`

### 1.4. RAG pipeline hooks (pre/post indexing)
*[Impact: High | Effort: Medium]*

**Problem:** The indexing pipeline in `indexing/full_pipeline.py` is a fixed
sequence: scan → analyze → chunk → embed → store. External systems that
need to inject custom processing — such as PII scrubbing before embedding,
domain-specific custom chunking strategies, watermarking metadata, or
triggering downstream ML pipelines — have no integration point.

**Approach:** Introduce a lightweight hook system into the pipeline. Define
two hook points — `pre_embed(chunks)` and `post_store(store, metadata)` —
as Python callables registered via a simple plugin discovery mechanism (e.g.
`entry_points` in `pyproject.toml` or a `hooks/` directory with convention
over configuration). Each hook receives and returns the pipeline's data,
enabling transformation, filtering, or enrichment. For example, a PII hook
could scan chunk content for social security numbers and redact them before
embedding. A domain hook for a medical codebase could inject ICD code
annotations into chunk metadata. The hooks run in order and are fully
optional — without any hooks registered, the pipeline behaves exactly as
today.

**Files:** `indexing/full_pipeline.py` (new `PipelineHooks` class), new
`hooks/` directory, `pyproject.toml`

### 1.5. Streaming search endpoint
*[Impact: Medium | Effort: Medium]*

**Problem:** The `POST /search/text` endpoint in `main.py` blocks until all
k results are scored and returned. For queries that trigger a large vector
scan (high k, many candidate chunks), the client waits with no feedback.
This is particularly noticeable for the pgvector store where the cosine
similarity computation happens server-side.

**Approach:** Add a `POST /search/stream` endpoint that uses Server-Sent
Events (SSE) to return results incrementally. As each result is scored
above a configurable similarity threshold, it is pushed to the client
immediately rather than waiting for the full top-k set. The client sees
results arriving one-by-one, starting with the highest-similarity match,
which dramatically improves perceived latency. This requires converting
the synchronous `PgVectorStore.search` method to an async generator or
using a background thread with a shared queue. The Chroma store path can
follow the same pattern.

**Files:** `main.py` (new SSE endpoint), `store/pgvector_store.py` (async or
threaded search variant), `store/chroma_store.py`

---

## 2. Python Analyzer

### 2.1. Decorator extraction
*[Impact: Medium | Effort: Low]*

**Problem:** Decorators (`@staticmethod`, `@property`, `@abstractmethod`,
`@app.route`, custom decorators) are completely ignored.  They carry
important semantic information such as framework routing, access control,
and typing intent.

**Approach:** In `_process_function`, iterate `node.decorator_list` and
collect the decorator chain as a list of strings.  Add a `decorators:
List[str]` field to `FunctionInfo`.  Preserve order so that
`@staticmethod @abstractmethod` remains meaningful.

**Files:** `analyzers/python_analyzer.py`, `models.py`

### 2.2. Nested class support
*[Impact: Low | Effort: Low]*

**Problem:** Only top-level `ast.ClassDef` nodes are visited.  Classes
defined inside other classes or inside functions are missed entirely.

**Approach:** After extracting methods from a class body, recurse into any
nested `ast.ClassDef` nodes the same way.  Qualify names with the
enclosing class (`Outer.Inner`).

**Files:** `analyzers/python_analyzer.py`

### 2.3. Type alias and TypedDict extraction
*[Impact: Low | Effort: Medium]*

**Problem:** `TypeAlias`, `NamedTuple`, `Protocol`, and `TypedDict`
assignments (e.g. `MyType = Union[int, str]`) are not captured.

**Approach:** Detect assignments where the value is an `ast.Subscript`,
`ast.Call` to `NamedTuple`/`TypedDict`, or `ast.Constant` with an
`ast.AnnAssign`.  Map these to `ClassInfo` with an appropriate `kind`.

**Files:** `analyzers/python_analyzer.py`

### 2.4. Package name reliability
*[Impact: Low | Effort: Low]*

**Problem:** `_get_package` walks upward looking for `__init__.py`.  It
fails for namespace packages (PEP 420), standalone scripts, and
directories that happen to lack `__init__.py` (e.g. some test fixtures).

**Approach:** Accept an optional `--package-root` override.  When not
provided, keep the current heuristic but mark the result as
"heuristic" so downstream consumers can choose to ignore it.

**Files:** `analyzers/python_analyzer.py`

### 2.5. Import alias handling
*[Impact: Low | Effort: Low]*

**Problem:** `import numpy as np` records `"numpy"` but discards the
alias `"np"`.  `from os.path import join as path_join` records only
`"os.path"`.

**Approach:** Record `f"{alias.name} as {alias.asname}"` when
`alias.asname` is set, or simply record `alias.asname` if callers
only need the local name.

**Files:** `analyzers/python_analyzer.py`

### 2.6. Error-tolerant parsing
*[Impact: High | Effort: Medium]*

**Problem:** A `SyntaxError` causes the entire file to return an empty
analysis.  Files with partial syntax errors (e.g. a missing closing
parenthesis on one line) yield nothing.

**Approach:** Fall back to a regex-based extraction pass when
`ast.parse` fails — scan for `def `, `class `, and `import ` lines at
minimum.  Alternatively, use `astroid` which supports incremental /
error-tolerant parsing.

**Files:** `analyzers/python_analyzer.py`

---

## 3. Go Analyzer

### 3.1. Test function detection
*[Impact: Low | Effort: Low]*

**Problem:** Functions named `TestFoo` in `_test.go` files are treated
identically to production code.  No `kind` or annotation distinguishes
benchmarks (`BenchmarkFoo`) or fuzz targets (`FuzzFoo`).

**Approach:** Add a `kind` field to `FunctionInfo` (or reuse `metadata`).
Detect `Test`, `Benchmark`, `Example`, and `Fuzz` prefixes and the
`_test.go` suffix, then tag accordingly.

**Files:** `analyzers/go_analyzer.py`

### 3.2. Interface embedding resolution
*[Impact: Medium | Effort: High]*

**Problem:** When an interface embeds another (`type Foo interface { Bar }`),
the embedded methods are not listed under `Foo`.

**Approach:** After collecting interface methods, resolve embedded
interfaces by looking up their method sets.  This requires cross-file
analysis or at least single-file multi-pass resolution.

**Files:** `analyzers/go_analyzer.py`

### 3.3. Generic type parameter extraction
*[Impact: Low | Effort: Low]*

**Problem:** Go 1.18+ type parameters (`[T any]`) are silently dropped
from type declarations and function signatures.

**Approach:** After extracting the `type_spec` node, check for a
`type_parameter_list` child and include its text in the signature or
a dedicated `type_params` field.

**Files:** `analyzers/go_analyzer.py`

### 3.4. Struct tag extraction
*[Impact: Medium | Effort: Low]*

**Problem:** Struct field tags (`json:"name,omitempty"`) are not captured,
even though they are critical for serialization, validation, and ORM
mapping.

**Approach:** Walk `field_definition` children inside `struct_type`.  For
each field, extract the `tag` literal if present.  Store as part of a
new `fields` list on `ClassInfo` or in `metadata`.

**Files:** `analyzers/go_analyzer.py`, `models.py`

### 3.5. Constant and variable extraction
*[Impact: Medium | Effort: Low]*

**Problem:** `const` and `var` blocks are ignored.  Package-level
identifiers with significant values (e.g. API version strings, default
configs) are invisible to search.

**Approach:** Add a new `VariableInfo` model or reuse `FunctionInfo`
with a `kind="const"` / `kind="var"` tag.  Walk `const_declaration` /
`var_declaration` nodes.

**Files:** `analyzers/go_analyzer.py`, `models.py`

---

## 4. JavaScript Analyzer

### 4.1. TypeScript support
*[Impact: High | Effort: Medium]*

**Problem:** The analyzer uses `tree_sitter_javascript`.  TypeScript
files (`.ts`, `.tsx`) are not handled.

**Approach:** Add `tree_sitter_typescript` as a dependency.  Create a
parallel or unified analyzer that handles TypeScript-specific node types
(`type_alias_declaration`, `interface_declaration`, `enum_declaration`,
`abstract_class_declaration`, `as_expression`, etc.).

**Files:** `analyzers/js_analyzer.py`, `analyzers/ts_analyzer.py` (new),
`utils/language_router.py`

### 4.2. React / JSX component extraction
*[Impact: Medium | Effort: Medium]*

**Problem:** React functional components (`export default function App()`) are
recognized as plain functions.  Class components and JSX are not
distinguished.

**Approach:** Detect `return <JSX…>` patterns and tag the function with
`kind="component"`.  For class components, detect `extends React.Component`
or `extends Component` in the heritage clause.

**Files:** `analyzers/js_analyzer.py`

### 4.3. Destructured exports
*[Impact: Medium | Effort: Low]*

**Problem:** `export { foo, bar }` and `export { default as Alias }` are
missed because the analyzer only handles `export_statement` with a
`declaration` field.

**Approach:** Walk `export_specifier` children inside `export_statement`.
Record each exported name and its local binding.

**Files:** `analyzers/js_analyzer.py`

### 4.4. Async function detection
*[Impact: Medium | Effort: Low]*

**Problem:** `async function` and `async arrow` are not tagged, losing
concurrency information that is important for understanding application
behavior.

**Approach:** Check the `async_modifier` child or `node.type` for
`async_function_declaration` / `arrow_function` within an `await`
context.  Add an `is_async` flag to `FunctionInfo`.

**Files:** `analyzers/js_analyzer.py`, `models.py`

### 4.5. Default vs named export distinction
*[Impact: Low | Effort: Low]*

**Problem:** All exports are treated equally.  Downstream consumers cannot
tell whether a symbol is the module's public API surface (named) or its
entry point (default).

**Approach:** Tag functions/classes with `export_kind` in metadata:
`"default"`, `"named"`, or `"none"`.

**Files:** `analyzers/js_analyzer.py`

---

## 5. Generic Tree-Sitter Analyzer

### 5.1. Kotlin support
*[Impact: Medium | Effort: Low]*

**Problem:** Kotlin (`.kt`) files are not handled.  Kotlin is a top-10
language on GitHub.

**Approach:** Add a `LanguageProfile` for Kotlin using
`tree_sitter_kotlin`.  Map `function_declaration`, `class_declaration`,
`object_declaration`, `companion_object`, `fun`, `import_list`.

**Files:** `analyzers/generic_ts_analyzer.py`, `utils/language_router.py`

### 5.2. Swift support
*[Impact: Medium | Effort: Low]*

**Problem:** Swift (`.swift`) files are not handled.

**Approach:** Add a `LanguageProfile` for Swift using
`tree_sitter_swift`.  Map `function_declaration`,
`class_declaration`, `struct_declaration`, `protocol_declaration`,
`extension_declaration`, `import_declaration`.

**Files:** `analyzers/generic_ts_analyzer.py`, `utils/language_router.py`

### 5.3. PHP support
*[Impact: Medium | Effort: Low]*

**Problem:** PHP (`.php`) files are not handled.

**Approach:** Add a `LanguageProfile` for PHP using
`tree_sitter_php`.  Map `function_definition`,
`class_declaration`, `interface_declaration`, `trait_declaration`,
`namespace_definition`, `use_declaration`.

**Files:** `analyzers/generic_ts_analyzer.py`, `utils/language_router.py`

### 5.4. Nested symbol qualification
*[Impact: Medium | Effort: Medium]*

**Problem:** Functions inside namespaces, impls, or modules are
generally not qualified with their enclosing scope (except for the
Java and Rust special cases).  For C++ namespaces, Lua modules, etc.,
this results in flat, ambiguous names.

**Approach:** Generalize the existing Rust/Java qualification logic into
a reusable helper.  Walk parent nodes looking for qualifying scopes
(namespace, module, impl, class) and prefix the name.

**Files:** `analyzers/generic_ts_analyzer.py`

### 5.5. C/C++ preprocessor macro extraction
*[Impact: Medium | Effort: Medium]*

**Problem:** `#define MACRO(x) …` and `#ifdef` guards are ignored.
Macros are a critical part of C/C++ APIs.

**Approach:** Add `preproc_def` and `preproc_if` to `import_types` for
C/C++ profiles, or create a separate regex pass for preprocessor
directives since tree-sitter may not capture them fully.

**Files:** `analyzers/generic_ts_analyzer.py`

### 5.6. Ruby module nesting
*[Impact: Low | Effort: Low]*

**Problem:** Ruby classes inside modules (e.g. `module Foo; class Bar; end; end`)
are captured as flat `"Bar"` without the `Foo::` prefix.

**Approach:** Maintain a stack of enclosing `module` / `class` nodes during
AST traversal.  Prefix each symbol with the full nesting path.

**Files:** `analyzers/generic_ts_analyzer.py`

### 5.7. Lazy grammar loading race condition
*[Impact: Medium | Effort: Low]*

**Problem:** `_loaded_parsers` is a module-level dict with no locking.
Concurrent calls (e.g. from a multi-threaded indexer) could cause
double-loading or corruption.

**Approach:** Use `threading.Lock` around the load-and-cache path, or
use `functools.lru_cache` on a parameterless function per language.

**Files:** `analyzers/generic_ts_analyzer.py`

### 5.8. Java inner class qualification
*[Impact: Low | Effort: Low]*

**Problem:** Java inner classes are not qualified with their outer class
(e.g. `Map.Entry`).

**Approach:** During function/class traversal, detect when a class
declaration's parent is a `class_declaration` body and prefix the name.

**Files:** `analyzers/generic_ts_analyzer.py`

### 5.9. Rust trait bound extraction
*[Impact: Low | Effort: Low]*

**Problem:** Rust trait bounds on generics (`fn foo<T: Clone + Debug>`) are
not extracted, losing important type constraint information.

**Approach:** Parse the `type_parameters` child of function/type nodes.
Extract trait bound names and attach to metadata or signature.

**Files:** `analyzers/generic_ts_analyzer.py`

---

## 6. Shell Analyzer

### 6.1. Heredoc detection
*[Impact: Low | Effort: Low]*

**Problem:** Multi-line heredocs (`cat <<EOF … EOF`) are not recognized.
Functions defined inside heredocs (unusual but possible) would be
false positives.

**Approach:** During the regex pass, track heredoc delimiters and skip
content between them.  Alternatively, warn on matches found inside
heredoc regions.

**Files:** `analyzers/shell_analyzer.py`

### 6.2. Subshell function detection
*[Impact: Low | Effort: Low]*

**Problem:** `foo() ( … )` (subshell function syntax) is not matched by the
existing regex patterns, which expect `{`.

**Approach:** Extend `_FUNC_BARE` to also accept `)` as a closing
delimiter: `r"^([\w][\w.-]*)\s*\(\s*\)\s*[\(]"`.

**Files:** `analyzers/shell_analyzer.py`

### 6.3. `local` variable tracking
*[Impact: Low | Effort: Low]*

**Problem:** `local var=value` inside functions is not captured.  These are
scoped variables that are important for understanding function behavior.

**Approach:** Add a regex for `local\s+(\w+)=` and record as function-local
metadata when inside a function body (tracked by line ranges).

**Files:** `analyzers/shell_analyzer.py`

### 6.4. POSIX compliance
*[Impact: Low | Effort: Low]*

**Problem:** The parser assumes bash/zsh extensions.  Strict POSIX shells
(`dash`, `sh`) may use different syntax that is partially missed.

**Approach:** Test against POSIX-only patterns.  Document which patterns
are bash-specific and which are POSIX-compatible.

**Files:** `analyzers/shell_analyzer.py`

---

## 7. Config Analyzer

### 7.1. Dockerfile instruction extraction
*[Impact: Low | Effort: Medium]*

**Problem:** Dockerfiles are routed to the INI parser, which treats each
`FROM`, `RUN`, `COPY` line as a generic text entry.  No structured
understanding of stages, args, or multi-stage builds.

**Approach:** Add a dedicated Dockerfile analyzer that maps instructions
to structured data: `FROM` → base image, `ARG` → build args,
`COPY`/`ADD` → file operations, `EXPOSE` → ports, `ENTRYPOINT`/`CMD`
→ entry points.  Record as `FunctionInfo` with `kind` tags.

**Files:** `analyzers/config_analyzer.py`

### 7.2. Makefile target extraction
*[Impact: Low | Effort: Medium]*

**Problem:** Makefiles fall through to `_analyze_text_kv`, which misses
target dependencies, phony targets, and recursive make calls.

**Approach:** Add a dedicated Makefile parser.  Extract targets
(`target: deps`), `.PHONY` declarations, `include` directives, and
variable assignments.  Store targets as `FunctionInfo` with dependencies
in metadata.

**Files:** `analyzers/config_analyzer.py`

### 7.3. YAML anchor and alias resolution
*[Impact: Low | Effort: Medium]*

**Problem:** YAML anchors (`&anchor`) and aliases (`*anchor`) are not
resolved.  The raw YAML structure is analyzed without expanding
references, leading to incomplete or misleading data.

**Approach:** After composing the YAML node, resolve aliases using
`yaml.compose` with a custom resolver or post-processing step.  Or
use `yaml.safe_load` (which resolves aliases natively) and map the
resolved dict back to line numbers where possible.

**Files:** `analyzers/config_analyzer.py`

### 7.4. TOML deep nesting
*[Impact: Low | Effort: Low]*

**Problem:** The TOML analyzer only walks one level deep.  Nested tables
(`[a.b.c]`) beyond the first level are collapsed or lost.

**Approach:** Recursively walk nested dicts to arbitrary depth.  Qualify
each key with its full dotted path (`a.b.c.key`).

**Files:** `analyzers/config_analyzer.py`

### 7.5. JSON array-of-objects handling
*[Impact: Low | Effort: Low]*

**Problem:** When the top-level JSON value is an array of objects, the
analyzer returns empty results.  Files like `tsconfig.json` with a
`compilerOptions` nested inside are handled, but `package.json`'s
top-level object works fine — the issue is with arrays at the root.

**Approach:** When the root is an array, iterate each element and apply
the same object analysis, prefixing keys with the array index
(`[0].key`).

**Files:** `analyzers/config_analyzer.py`

### 7.6. XML support
*[Impact: Low | Effort: Medium]*

**Problem:** XML files (`.xml`, `.svg`, `.xsl`, pom.xml, Android manifests)
have no dedicated analyzer.  They fall through to the generic
text parser.

**Approach:** Use `xml.etree.ElementTree` to parse.  Map elements to
`ClassInfo` (with tag names) and attributes to `FunctionInfo`.

**Files:** `analyzers/config_analyzer.py`

---

## 8. Language Router

### 8.1. Extension coverage gaps
*[Impact: Medium | Effort: Low]*

**Problem:** Several common extensions are not routed:
- `.kt` / `.kts` → no Kotlin support
- `.swift` → no Swift support
- `.php` → no PHP support
- `.cs` → no C# support (despite `tree-sitter-c-sharp` in requirements.txt)
- `.m` / `.mm` → no Objective-C support
- `.proto` → no Protocol Buffers support
- `.tf` / `.tfvars` → no Terraform / HCL support
- `.cypher` → no Cypher support

**Approach:** Systematically add mappings for each extension.  For
languages without tree-sitter grammars, provide a basic regex or
text-based fallback.

**Files:** `utils/language_router.py`

### 8.2. Polyglot file detection
*[Impact: Medium | Effort: High]*

**Problem:** Files like `.vue` (SFC), `.astro`, `.svelte` contain
embedded `<script>`, `<style>`, and template sections in different
languages.  A single language label is insufficient.

**Approach:** Add a "polyglot" mode that splits the file into sections
(by language fence or custom delimiter) and runs the appropriate
analyzer on each section.  Merge results, prefixing names with the
section language.

**Files:** `utils/language_router.py`, possibly new `analyzers/polyglot_analyzer.py`

### 8.3. Shebang-based routing enhancement
*[Impact: Low | Effort: Low]*

**Problem:** While shebang detection exists in `utils/language_router.py`,
files with misleading extensions or no extension at all that carry
unusual shebangs (e.g. `#!/usr/bin/env perl`, `#!/usr/bin/lua5.3`) may
still be missed.

**Approach:** Expand the shebang mapping in `_detect_from_shebang` to
cover `perl`, `node` variants, and other interpreters.  Ensure the
fallback path from extension detection to shebang detection is complete.

**Files:** `utils/language_router.py`

### 8.4. Encoding detection
*[Impact: Medium | Effort: Low]*

**Problem:** Files are assumed to be UTF-8.  Files with Latin-1, Shift-JIS,
or other encodings may cause `UnicodeDecodeError` or garbled text.

**Approach:** Use `chardet` or `charset-normalizer` to detect encoding
before opening.  Fall back to UTF-8 with `errors="ignore"` as today.

**Files:** `utils/language_router.py` (or a shared utility)

---

## 9. Models & Data Layer

### 9.1. `FunctionInfo` lacks `params` across analyzers
*[Impact: Medium | Effort: Medium]*

**Problem:** Only the Python analyzer populates `params`.  Go, JavaScript,
and the generic analyzer leave it empty because tree-sitter does not
provide a structured parameter list.

**Approach:** For tree-sitter-based analyzers, walk the `parameters` child
node and collect `identifier` children.  This works for Go (parameters
field), JavaScript (formal_parameters), Rust (parameters), Java
(formal_parameters), etc.

**Files:** `models.py` (ensure field exists), `analyzers/go_analyzer.py`,
`analyzers/js_analyzer.py`, `analyzers/generic_ts_analyzer.py`

### 9.2. `ClassInfo` lacks `members` / `fields`
*[Impact: Medium | Effort: Medium]*

**Problem:** Classes are recorded with name and kind but no member list.
Consumers cannot tell what methods or fields a class has without
cross-referencing the functions list.

**Approach:** Add `members: List[str]` to `ClassInfo`.  For Python,
extract attribute assignments in `__init__`.  For Go, extract struct
fields.  For Java, extract field declarations.

**Files:** `models.py`, all analyzers

### 9.3. No visibility/access modifier
*[Impact: Medium | Effort: Medium]*

**Problem:** There is no way to distinguish `public` vs `private` vs
`protected` symbols.  In Go (unexported = lowercase), Python
(`_prefix`), Java (`private` keyword), this information is readily
available.

**Approach:** Add `visibility: str` to `FunctionInfo` and `ClassInfo`.
Infer from naming conventions (Go/Python) or explicit keywords
(Java/C++).

**Files:** `models.py`, all analyzers

### 9.4. `metadata` dict conventions
*[Impact: Medium | Effort: Low]*

**Problem:** Language-specific data (decorators, annotations, modifiers)
has nowhere to go in the current model.  A `metadata` field could hold
this but no conventions exist.

**Approach:** Define conventions for common metadata keys:
`{"decorators": [...], "annotations": [...], "modifiers": [...],
"overrides": bool}`.  Populate them in each analyzer.

**Files:** `models.py`, all analyzers

### 9.5. Import and call graph improvements
*[Impact: High | Effort: High]*

**Problem:** The import graph (`lib/graph.py`) only resolves first-order
dependencies.  Transitive dependencies (A imports B which imports C)
are not tracked.  The call graph resolves calls within a single run
but does not handle indirect call chains.

**Approach:** Add a transitive closure pass to `build_import_graph` that
computes full dependency chains (A → B → C).  Store both direct and
transitive dependencies in the graph.  For the call graph, add a BFS
pass to compute reachable functions from any starting symbol.

**Files:** `lib/graph.py`

---

## 10. Indexing Pipeline

### 10.1. Incremental indexing
*[Impact: High | Effort: Medium]*

**Problem:** The pipeline re-indexes the entire repository on every run.
For large codebases, this is slow and wasteful.

**Approach:** Store file hashes (e.g. SHA-256) alongside embeddings.  On
subsequent runs, skip files whose hash hasn't changed.  Remove
embeddings for deleted files. This is complementary to the semantic
caching layer (§1.1) which operates at the chunk level.

**Files:** `indexing/full_pipeline.py`, `db.py`

### 10.2. Parallel file processing
*[Impact: High | Effort: Medium]*

**Problem:** Files are processed sequentially in `index_repository`
(`indexing/full_pipeline.py`).  On multi-core machines, this leaves
significant performance on the table.

**Approach:** Use `concurrent.futures.ProcessPoolExecutor` or `ThreadPoolExecutor`
to process files in parallel.  Ensure thread-safe access to the vector
store and database connection pool.

**Files:** `indexing/full_pipeline.py`

### 10.3. File size limits already addressed
*[Impact: N/A | Effort: N/A]*

**Note:** This has already been implemented in `utils/file_scanner.py`
with `MAX_FILE_SIZE_BYTES = 512 * 1024` and comprehensive binary
extension filtering.

### 10.4. Binary file detection already addressed
*[Impact: N/A | Effort: N/A]*

**Note:** This has already been implemented in `utils/file_scanner.py`
via the `BINARY_EXTENSIONS` set and `IGNORE_FILE_SUFFIXES` tuple.

### 10.5. Git-aware indexing
*[Impact: High | Effort: Medium]*

**Problem:** The pipeline does not use git information.  It indexes the
working tree, which may include uncommitted changes, build artifacts,
and `.gitignore`-d files.

**Approach:** Use `git ls-files` to enumerate tracked files.  Optionally,
use `git diff` to detect changes since the last index for true
incremental updates.

**Files:** `indexing/full_pipeline.py`, `utils/file_scanner.py`

---

## 11. Search & Retrieval

### 11.1. Fuzzy matching
*[Impact: Medium | Effort: Medium]*

**Problem:** Search is exact or embedding-based.  Typos and minor naming
variations (`getUser` vs `get_user` vs `GetUser`) may not match.

**Approach:** Add a fuzzy matching layer (e.g. `thefuzz` / `rapidfuzz`)
that supplements embedding search.  Use it as a fallback when
embedding similarity is below a threshold.

**Files:** `main.py`, `store/pgvector_store.py`

### 11.2. Hybrid result ranking
*[Impact: Medium | Effort: High]*

**Problem:** Search results are returned without explicit ranking beyond
embedding similarity.  Code symbols that match the query directly
by name should rank higher than code that merely mentions the term.

**Approach:** Implement a hybrid scoring function that combines embedding
cosine similarity with text-match boosts (exact name match, file
path relevance, recency/usage frequency).

**Files:** `main.py`, `store/pgvector_store.py`, `store/chroma_store.py`

### 11.3. File path and metadata filters
*[Impact: Medium | Effort: Low]*

**Problem:** Search cannot be scoped to specific directories, file types,
chunk types, or language filters.

**Approach:** Add optional filter parameters to the search function:
`path_prefix`, `extension`, `language`, `chunk_type`.  Apply as
post-filters on pgvector results or as `WHERE` clause additions.

**Files:** `main.py`, `store/pgvector_store.py`

### 11.4. Cross-project search
*[Impact: Medium | Effort: Medium]*

**Problem:** Search is scoped to a single `(namespace, project_id)` pair.
Users cannot search across all projects in a namespace or across
namespaces.

**Approach:** Add a `POST /search/text/global` endpoint that accepts an
optional `namespace` without `project_id`.  When only namespace is
provided, search across all projects in that namespace.  Aggregate
and deduplicate results.

**Files:** `main.py`, `store/pgvector_store.py`, `db.py`

---

## 12. LLM Integration

### 12.1. Context window management
*[Impact: High | Effort: Medium]*

**Problem:** Large repositories may produce a context that exceeds the LLM's
context window.  There is no mechanism to truncate or summarize
selectively.

**Approach:** Implement a prioritized truncation strategy: keep the
repository overview, configuration files, and the most relevant code
sections (by embedding similarity to the query).  Drop low-relevance
sections first.

**Files:** LLM context assembly layer

### 12.2. Multi-provider support
*[Impact: Medium | Effort: Medium]*

**Problem:** The system may be tightly coupled to a single LLM provider.
Using OpenAI, Anthropic, or other providers requires code changes.

**Approach:** Abstract behind an `LLMProvider` interface with
implementations for multiple providers and a mock provider for
testing.  Select via configuration.

**Files:** LLM client layer

### 12.3. Streaming responses
*[Impact: Medium | Effort: Medium]*

**Problem:** LLM responses may be received in full after the model finishes.
For long analyses, the user sees no output until completion.

**Approach:** Use streaming APIs and return tokens as they arrive via
SSE, similar to the streaming search proposal (§1.5).

**Files:** LLM client layer

### 12.4. Prompt caching
*[Impact: Medium | Effort: Low]*

**Problem:** The same repository context is re-sent on every query.  For
repositories that don't change between queries, this is wasteful.

**Approach:** Cache the context portion of the prompt keyed by
`(namespace, project_id)`.  Only re-send the query portion.  Some
providers (Anthropic, OpenAI) support explicit prompt caching APIs.

**Files:** LLM client layer

---

## 13. Testing

### 13.1. No test suite
*[Impact: Very High | Effort: Medium]*

**Problem:** There are no automated tests.  Refactoring or adding features
relies entirely on manual verification.

**Approach:** Add a `tests/` directory with:
- Unit tests per analyzer using fixture files
- Property-based tests for edge cases
- Integration tests for the full pipeline
- Snapshot tests for `FileAnalysis` output

Use `pytest` with `pytest-cov` for coverage reporting.

**Files:** New `tests/` directory

### 13.2. Golden file fixtures
*[Impact: High | Effort: Medium]*

**Problem:** Without test fixtures, it is difficult to verify that analyzer
changes don't regress on real-world code patterns.

**Approach:** Create a `tests/fixtures/` directory with small, representative
source files for each supported language.  Record expected
`FileAnalysis` output as JSON "golden" files.  Compare on each run.

**Files:** New `tests/fixtures/`, `tests/` test files

### 13.3. Performance benchmarks
*[Impact: Medium | Effort: Medium]*

**Problem:** There are no benchmarks.  Performance regressions from new
features go undetected.

**Approach:** Add `pytest-benchmark` tests that time analysis of large
fixture files.  Track over time with CI.

**Files:** New `tests/benchmarks/`

---

## 14. Operational & DevEx

### 14.1. Structured logging
*[Impact: Medium | Effort: Low]*

**Problem:** The codebase uses `print()` and `warnings.warn()` for output.
There is no structured logging with severity levels.

**Approach:** Replace `print` and `warnings` with Python's `logging` module.
Use JSON-structured logging for machine-parseable output in production.

**Files:** Throughout (`indexing/full_pipeline.py`, `lib/embedder.py`,
`store/chroma_store.py`, `main.py`)

### 14.2. CLI interface
*[Impact: Medium | Effort: Medium]*

**Problem:** The pipeline is invoked programmatically.  There are
no subcommands, help text, or argument validation for standalone use.

**Approach:** Wrap in `click` or `argparse` with subcommands:
`index`, `search`, `analyze`, `serve`.  Add `--help`, `--verbose`,
`--config` flags.

**Files:** New `cli.py`

### 14.3. Configuration file
*[Impact: Medium | Effort: Medium]*

**Problem:** Settings (model name, file size limits, excluded
paths) are hardcoded or scattered across modules.

**Approach:** Support a `pyproject.toml` / `.code-index.toml` config file
with sensible defaults.  Allow CLI flags and environment variables to
override.

**Files:** New `config.py`, update all consumers

### 14.4. Progress reporting
*[Impact: Medium | Effort: Low]*

**Problem:** For large repositories, the indexing pipeline provides no
progress feedback.  The user sees nothing until completion.

**Approach:** Add a progress bar (`tqdm`) or periodic status prints
showing files processed, total progress, and estimated time remaining.

**Files:** `indexing/full_pipeline.py`

### 14.5. Error reporting
*[Impact: Medium | Effort: Low]*

**Problem:** Files that fail to parse are silently skipped.  The user has
no visibility into which files failed and why.

**Approach:** Collect parse failures in a structured report.  At the end
of the pipeline, print a summary: "X files succeeded, Y files failed"
with file paths and error messages for failures.

**Files:** `indexing/full_pipeline.py`

### 14.6. Health check endpoint
*[Impact: Medium | Effort: Low]*

**Problem:** The FastAPI service in `main.py` has no health check endpoint.
Orchestrators (Kubernetes, load balancers) and monitoring systems have
no way to probe liveness or readiness.

**Approach:** Add a `GET /health` endpoint that checks database
connectivity and returns `{"status": "ok"}`.  Add a `GET /ready`
endpoint that also verifies the embedding model is loaded.

**Files:** `main.py`

### 14.7. Request rate limiting
*[Impact: Medium | Effort: Low]*

**Problem:** The `main.py` API has no rate limiting. A burst of indexing
requests could overwhelm the database or embedding model.

**Approach:** Add a per-IP or per-API-key rate limiter using a simple
in-memory sliding window or Redis-backed token bucket.  Apply to
indexing and embedding endpoints.

**Files:** `main.py`

---

## Priority Matrix

| # | Area | Impact | Effort | Priority |
|---|------|--------|--------|----------|
| 1.1 | Semantic caching layer | High | Medium | **P0** |
| 1.2 | Per-tenant model routing | High | Low | **P0** |
| 1.4 | RAG pipeline hooks | High | Medium | **P0** |
| 13.1 | Test suite foundation | Very High | Medium | **P0** |
| 10.1 | Incremental indexing | High | Medium | **P0** |
| 4.1 | TypeScript support | High | Medium | **P1** |
| 2.6 | Python error-tolerant parsing | High | Medium | **P1** |
| 14.1 | Structured logging | Medium | Low | **P1** |
| 10.5 | Git-aware indexing | High | Medium | **P1** |
| 10.2 | Parallel file processing | High | Medium | **P1** |
| 12.1 | Context window management | High | Medium | **P1** |
| 1.5 | Streaming search endpoint | Medium | Medium | **P1** |
| 1.3 | Embedding cost attribution | Medium | Low | **P1** |
| 14.6 | Health check endpoint | Medium | Low | **P1** |
| 14.7 | Request rate limiting | Medium | Low | **P1** |
| 8.2 | Polyglot file detection | Medium | High | **P2** |
| 5.1–5.3 | Kotlin / Swift / PHP | Medium | Low each | **P2** |
| 11.1 | Fuzzy matching | Medium | Medium | **P2** |
| 11.4 | Cross-project search | Medium | Medium | **P2** |
| 9.5 | Graph improvements | High | High | **P2** |
| 12.2 | Multi-provider LLM | Medium | Medium | **P2** |
| 7.1, 7.2 | Makefile / Dockerfile analyzers | Low | Medium | **P3** |
| 9.3 | Visibility modifiers | Medium | Medium | **P3** |
| 6.1–6.4 | Shell analyzer improvements | Low | Low | **P3** |
| 13.2 | Golden file fixtures | High | Medium | **P1** |
| 13.3 | Performance benchmarks | Medium | Medium | **P2** |
| 14.2 | CLI interface | Medium | Medium | **P2** |
| 14.3 | Configuration file | Medium | Medium | **P2** |
| 14.4 | Progress reporting | Medium | Low | **P2** |
| 14.5 | Error reporting | Medium | Low | **P2** |
| 9.4 | Metadata conventions | Medium | Low | **P2** |
| 9.1 | Params across analyzers | Medium | Medium | **P2** |
| 9.2 | ClassInfo members | Medium | Medium | **P2** |

---

*This document is a living artifact.  Update it as improvements are
implemented or new limitations are discovered.*
