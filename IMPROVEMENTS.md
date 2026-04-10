# Potential Improvements & Known Limitations

This document catalogues actionable improvements and known limitations across
the indexing and analysis codebase.  Each entry includes the **why**, a
suggested approach, and the files that would need to change.

---

## Table of Contents

1. [Python Analyzer](#1-python-analyzer)
2. [Go Analyzer](#3-go-analyzer)
3. [JavaScript Analyzer](#4-javascript-analyzer)
4. [Generic Tree-Sitter Analyzer](#5-generic-tree-sitter-analyzer)
5. [Shell Analyzer](#6-shell-analyzer)
6. [Config Analyzer](#7-config-analyzer)
7. [Language Router](#8-language-router)
8. [Models & Data Layer](#9-models--data-layer)
9. [Indexing Pipeline](#10-indexing-pipeline)
10. [Search & Retrieval](#11-search--retrieval)
11. [LLM Integration](#12-llm-integration)
12. [Testing](#13-testing)
13. [Operational & DevEx](#14-operational--devex)

---

## 1. Python Analyzer

### 1.1. Decorator extraction
**Problem:** Decorators (`@staticmethod`, `@property`, `@abstractmethod`,
`@app.route`, custom decorators) are completely ignored.  They carry
important semantic information such as framework routing, access control,
and typing intent.

**Approach:** In `_process_function`, iterate `node.decorator_list` and
collect the decorator chain as a list of strings.  Add a `decorators:
List[str]` field to `FunctionInfo`.  Preserve order so that
`@staticmethod @abstractmethod` remains meaningful.

**Files:** `analyzers/python_analyzer.py`, `models.py`

### 1.2. Nested class support
**Problem:** Only top-level `ast.ClassDef` nodes are visited.  Classes
defined inside other classes or inside functions are missed entirely.

**Approach:** After extracting methods from a class body, recurse into any
nested `ast.ClassDef` nodes the same way.  Qualify names with the
enclosing class (`Outer.Inner`).

**Files:** `analyzers/python_analyzer.py`

### 1.3. Type alias and TypedDict extraction
**Problem:** `TypeAlias`, `NamedTuple`, `Protocol`, and `TypedDict`
assignments (e.g. `MyType = Union[int, str]`) are not captured.

**Approach:** Detect assignments where the value is an `ast.Subscript`,
`ast.Call` to `NamedTuple`/`TypedDict`, or `ast.Constant` with an
`ast.AnnAssign`.  Map these to `ClassInfo` with an appropriate `kind`.

**Files:** `analyzers/python_analyzer.py`

### 1.4. Package name reliability
**Problem:** `_get_package` walks upward looking for `__init__.py`.  It
fails for namespace packages (PEP 420), standalone scripts, and
directories that happen to lack `__init__.py` (e.g. some test fixtures).

**Approach:** Accept an optional `--package-root` override.  When not
provided, keep the current heuristic but mark the result as
"heuristic" so downstream consumers can choose to ignore it.

**Files:** `analyzers/python_analyzer.py`

### 1.5. Import alias handling
**Problem:** `import numpy as np` records `"numpy"` but discards the
alias `"np"`.  `from os.path import join as path_join` records only
`"os.path"`.

**Approach:** Record `f"{alias.name} as {alias.asname}"` when
`alias.asname` is set, or simply record `alias.asname` if callers
only need the local name.

**Files:** `analyzers/python_analyzer.py`

### 1.6. Error-tolerant parsing
**Problem:** A `SyntaxError` causes the entire file to return an empty
analysis.  Files with partial syntax errors (e.g. a missing closing
parenthesis on one line) yield nothing.

**Approach:** Fall back to a regex-based extraction pass when
`ast.parse` fails — scan for `def `, `class `, and `import ` lines at
minimum.  Alternatively, use `astroid` which supports incremental /
error-tolerant parsing.

**Files:** `analyzers/python_analyzer.py`

---

## 2. Go Analyzer

### 2.1. Test function detection
**Problem:** Functions named `TestFoo` in `_test.go` files are treated
identically to production code.  No `kind` or annotation distinguishes
benchmarks (`BenchmarkFoo`) or fuzz targets (`FuzzFoo`).

**Approach:** Add a `kind` field to `FunctionInfo` (or reuse `metadata`).
Detect `Test`, `Benchmark`, `Example`, and `Fuzz` prefixes and the
`_test.go` suffix, then tag accordingly.

**Files:** `analyzers/go_analyzer.py`

### 2.2. Interface embedding resolution
**Problem:** When an interface embeds another (`type Foo interface { Bar }`),
the embedded methods are not listed under `Foo`.

**Approach:** After collecting interface methods, resolve embedded
interfaces by looking up their method sets.  This requires cross-file
analysis or at least single-file multi-pass resolution.

**Files:** `analyzers/go_analyzer.py`

### 2.3. Generic type parameter extraction
**Problem:** Go 1.18+ type parameters (`[T any]`) are silently dropped
from type declarations and function signatures.

**Approach:** After extracting the `type_spec` node, check for a
`type_parameter_list` child and include its text in the signature or
a dedicated `type_params` field.

**Files:** `analyzers/go_analyzer.py`

### 2.4. Struct tag extraction
**Problem:** Struct field tags (`json:"name,omitempty"`) are not captured,
even though they are critical for serialization, validation, and ORM
mapping.

**Approach:** Walk `field_definition` children inside `struct_type`.  For
each field, extract the `tag` literal if present.  Store as part of a
new `fields` list on `ClassInfo` or in `metadata`.

**Files:** `analyzers/go_analyzer.py`, `models.py`

### 2.5. Constant and variable extraction
**Problem:** `const` and `var` blocks are ignored.  Package-level
identifiers with significant values (e.g. API version strings, default
configs) are invisible to search.

**Approach:** Add a new `VariableInfo` model or reuse `FunctionInfo`
with a `kind="const"` / `kind="var"` tag.  Walk `const_declaration` /
`var_declaration` nodes.

**Files:** `analyzers/go_analyzer.py`, `models.py`

---

## 3. JavaScript Analyzer

### 3.1. TypeScript support
**Problem:** The analyzer uses `tree_sitter_javascript`.  TypeScript
files (`.ts`, `.tsx`) are not handled.

**Approach:** Add `tree_sitter_typescript` as a dependency.  Create a
parallel or unified analyzer that handles TypeScript-specific node types
(`type_alias_declaration`, `interface_declaration`, `enum_declaration`,
`abstract_class_declaration`, `as_expression`, etc.).

**Files:** `analyzers/js_analyzer.py`, `analyzers/ts_analyzer.py` (new),
`language_router.py`

### 3.2. React / JSX component extraction
**Problem:** React functional components (`export default function App()`) are
recognized as plain functions.  Class components and JSX are not
distinguished.

**Approach:** Detect `return <JSX…>` patterns and tag the function with
`kind="component"`.  For class components, detect `extends React.Component`
or `extends Component` in the heritage clause.

**Files:** `analyzers/js_analyzer.py`

### 3.3. Destructured exports
**Problem:** `export { foo, bar }` and `export { default as Alias }` are
missed because the analyzer only handles `export_statement` with a
`declaration` field.

**Approach:** Walk `export_specifier` children inside `export_statement`.
Record each exported name and its local binding.

**Files:** `analyzers/js_analyzer.py`

### 3.4. Async function detection
**Problem:** `async function` and `async arrow` are not tagged, losing
concurrency information that is important for understanding application
behavior.

**Approach:** Check the `async_modifier` child or `node.type` for
`async_function_declaration` / `arrow_function` within an `await`
context.  Add an `is_async` flag to `FunctionInfo`.

**Files:** `analyzers/js_analyzer.py`, `models.py`

### 3.5. Default vs named export distinction
**Problem:** All exports are treated equally.  Downstream consumers cannot
tell whether a symbol is the module's public API surface (named) or its
entry point (default).

**Approach:** Tag functions/classes with `export_kind` in metadata:
`"default"`, `"named"`, or `"none"`.

**Files:** `analyzers/js_analyzer.py`

---

## 4. Generic Tree-Sitter Analyzer

### 4.1. Kotlin support
**Problem:** Kotlin (`.kt`) files are not handled.  Kotlin is a top-10
language on GitHub.

**Approach:** Add a `LanguageProfile` for Kotlin using
`tree_sitter_kotlin`.  Map `function_declaration`, `class_declaration`,
`object_declaration`, `companion_object`, `fun`, `import_list`.

**Files:** `analyzers/generic_ts_analyzer.py`, `language_router.py`

### 4.2. Swift support
**Problem:** Swift (`.swift`) files are not handled.

**Approach:** Add a `LanguageProfile` for Swift using
`tree_sitter_swift`.  Map `function_declaration`,
`class_declaration`, `struct_declaration`, `protocol_declaration`,
`extension_declaration`, `import_declaration`.

**Files:** `analyzers/generic_ts_analyzer.py`, `language_router.py`

### 4.3. PHP support
**Problem:** PHP (`.php`) files are not handled.

**Approach:** Add a `LanguageProfile` for PHP using
`tree_sitter_php`.  Map `function_definition`,
`class_declaration`, `interface_declaration`, `trait_declaration`,
`namespace_definition`, `use_declaration`.

**Files:** `analyzers/generic_ts_analyzer.py`, `language_router.py`

### 4.4. Nested symbol qualification
**Problem:** Functions inside namespaces, impls, or modules are
generally not qualified with their enclosing scope (except for the
Java and Rust special cases).  For C++ namespaces, Lua modules, etc.,
this results in flat, ambiguous names.

**Approach:** Generalize the existing Rust/Java qualification logic into
a reusable helper.  Walk parent nodes looking for qualifying scopes
(namespace, module, impl, class) and prefix the name.

**Files:** `analyzers/generic_ts_analyzer.py`

### 4.5. C/C++ preprocessor macro extraction
**Problem:** `#define MACRO(x) …` and `#ifdef` guards are ignored.
Macros are a critical part of C/C++ APIs.

**Approach:** Add `preproc_def` and `preproc_if` to `import_types` for
C/C++ profiles, or create a separate regex pass for preprocessor
directives since tree-sitter may not capture them fully.

**Files:** `analyzers/generic_ts_analyzer.py`

### 4.6. Ruby module nesting
**Problem:** Ruby classes inside modules (e.g. `module Foo; class Bar; end; end`)
are captured as flat `"Bar"` without the `Foo::` prefix.

**Approach:** Maintain a stack of enclosing `module` / `class` nodes during
AST traversal.  Prefix each symbol with the full nesting path.

**Files:** `analyzers/generic_ts_analyzer.py`

### 4.7. Lazy grammar loading race condition
**Problem:** `_loaded_parsers` is a module-level dict with no locking.
Concurrent calls (e.g. from a multi-threaded indexer) could cause
double-loading or corruption.

**Approach:** Use `threading.Lock` around the load-and-cache path, or
use `functools.lru_cache` on a parameterless function per language.

**Files:** `analyzers/generic_ts_analyzer.py`

### 4.8. Java inner class qualification
**Problem:** Java inner classes are not qualified with their outer class
(e.g. `Map.Entry`).

**Approach:** During function/class traversal, detect when a class
declaration's parent is a `class_declaration` body and prefix the name.

**Files:** `analyzers/generic_ts_analyzer.py`

### 4.9. Rust trait bound extraction
**Problem:** Rust trait bounds on generics (`fn foo<T: Clone + Debug>`) are
not extracted, losing important type constraint information.

**Approach:** Parse the `type_parameters` child of function/type nodes.
Extract trait bound names and attach to metadata or signature.

**Files:** `analyzers/generic_ts_analyzer.py`

---

## 5. Shell Analyzer

### 5.1. Heredoc detection
**Problem:** Multi-line heredocs (`cat <<EOF … EOF`) are not recognized.
Functions defined inside heredocs (unusual but possible) would be
false positives.

**Approach:** During the regex pass, track heredoc delimiters and skip
content between them.  Alternatively, warn on matches found inside
heredoc regions.

**Files:** `analyzers/shell_analyzer.py`

### 5.2. Subshell function detection
**Problem:** `foo() ( … )` (subshell function syntax) is not matched by the
existing regex patterns, which expect `{`.

**Approach:** Extend `_FUNC_BARE` to also accept `)` as a closing
delimiter: `r"^([\w][\w.-]*)\s*\(\s*\)\s*[\({]"`.

**Files:** `analyzers/shell_analyzer.py`

### 5.3. `local` variable tracking
**Problem:** `local var=value` inside functions is not captured.  These are
scoped variables that are important for understanding function behavior.

**Approach:** Add a regex for `local\s+(\w+)=` and record as function-local
metadata when inside a function body (tracked by line ranges).

**Files:** `analyzers/shell_analyzer.py`

### 5.4. POSIX compliance
**Problem:** The parser assumes bash/zsh extensions.  Strict POSIX shells
(`dash`, `sh`) may use different syntax that is partially missed.

**Approach:** Test against POSIX-only patterns.  Document which patterns
are bash-specific and which are POSIX-compatible.

**Files:** `analyzers/shell_analyzer.py`

---

## 6. Config Analyzer

### 6.1. Dockerfile instruction extraction
**Problem:** Dockerfiles are routed to the INI parser, which treats each
`FROM`, `RUN`, `COPY` line as a generic text entry.  No structured
understanding of stages, args, or multi-stage builds.

**Approach:** Add a dedicated Dockerfile analyzer that maps instructions
to structured data: `FROM` → base image, `ARG` → build args,
`COPY`/`ADD` → file operations, `EXPOSE` → ports, `ENTRYPOINT`/`CMD`
→ entry points.  Record as `FunctionInfo` with `kind` tags.

**Files:** `analyzers/config_analyzer.py`

### 6.2. Makefile target extraction
**Problem:** Makefiles fall through to `_analyze_text_kv`, which misses
target dependencies, phony targets, and recursive make calls.

**Approach:** Add a dedicated Makefile parser.  Extract targets
(`target: deps`), `.PHONY` declarations, `include` directives, and
variable assignments.  Store targets as `FunctionInfo` with dependencies
in metadata.

**Files:** `analyzers/config_analyzer.py`

### 6.3. YAML anchor and alias resolution
**Problem:** YAML anchors (`&anchor`) and aliases (`*anchor`) are not
resolved.  The raw YAML structure is analyzed without expanding
references, leading to incomplete or misleading data.

**Approach:** After composing the YAML node, resolve aliases using
`yaml.compose` with a custom resolver or post-processing step.  Or
use `yaml.safe_load` (which resolves aliases natively) and map the
resolved dict back to line numbers where possible.

**Files:** `analyzers/config_analyzer.py`

### 6.4. TOML deep nesting
**Problem:** The TOML analyzer only walks one level deep.  Nested tables
(`[a.b.c]`) beyond the first level are collapsed or lost.

**Approach:** Recursively walk nested dicts to arbitrary depth.  Qualify
each key with its full dotted path (`a.b.c.key`).

**Files:** `analyzers/config_analyzer.py`

### 6.5. JSON array-of-objects handling
**Problem:** When the top-level JSON value is an array of objects, the
analyzer returns empty results.  Files like `tsconfig.json` with a
`compilerOptions` nested inside are handled, but `package.json`'s
top-level object works fine — the issue is with arrays at the root.

**Approach:** When the root is an array, iterate each element and apply
the same object analysis, prefixing keys with the array index
(`[0].key`).

**Files:** `analyzers/config_analyzer.py`

### 6.6. XML support
**Problem:** XML files (`.xml`, `.svg`, `.xsl`, pom.xml, Android manifests)
have no dedicated analyzer.  They fall through to the generic
text parser.

**Approach:** Use `xml.etree.ElementTree` to parse.  Map elements to
`ClassInfo` (with tag names) and attributes to `FunctionInfo`.

**Files:** `analyzers/config_analyzer.py`

---

## 7. Language Router

### 7.1. Extension coverage gaps
**Problem:** Several common extensions are not routed:
- `.jsx` / `.tsx` → no TypeScript analyzer
- `.kt` / `.kts` → no Kotlin support
- `.swift` → no Swift support
- `.php` → no PHP support
- `.cs` → no C# support
- `.m` / `.mm` → no Objective-C support
- `.rs` → routed to generic but requires `tree_sitter_rust` to be installed
- `.proto` → no Protocol Buffers support
- `.tf` / `.tfvars` → no Terraform / HCL support
- `.cypher` → no Cypher support

**Approach:** Systematically add mappings for each extension.  For
languages without tree-sitter grammars, provide a basic regex or
text-based fallback.

**Files:** `analyzers/language_router.py`

### 7.2. Polyglot file detection
**Problem:** Files like `.vue` (SFC), `.astro`, `.svelte` contain
embedded `<script>`, `<style>`, and template sections in different
languages.  A single language label is insufficient.

**Approach:** Add a "polyglot" mode that splits the file into sections
(by language fence or custom delimiter) and runs the appropriate
analyzer on each section.  Merge results, prefixing names with the
section language.

**Files:** `analyzers/language_router.py`, possibly new `analyzers/polyglot_analyzer.py`

### 7.3. Shebang-based routing
**Problem:** Files without extensions (e.g. `Makefile`, `Dockerfile`,
`Vagrantfile`) or with misleading extensions are routed by name or
fall through.  Files with `#!/usr/bin/env python3` shebangs are not
detected.

**Approach:** Read the first line of the file when extension-based routing
fails.  Map common shebangs (`python`, `bash`, `node`, `ruby`, `perl`,
`lua`) to the appropriate analyzer.

**Files:** `analyzers/language_router.py`

### 7.4. Encoding detection
**Problem:** Files are assumed to be UTF-8.  Files with Latin-1, Shift-JIS,
or other encodings may cause `UnicodeDecodeError` or garbled text.

**Approach:** Use `chardet` or `charset-normalizer` to detect encoding
before opening.  Fall back to UTF-8 with `errors="ignore"` as today.

**Files:** `analyzers/language_router.py` (or a shared utility)

---

## 8. Models & Data Layer

### 8.1. `FunctionInfo` lacks `params`
**Problem:** Only the Python analyzer populates `params`.  Go, JavaScript,
and the generic analyzer leave it empty because tree-sitter does not
provide a structured parameter list.

**Approach:** For tree-sitter-based analyzers, walk the `parameters` child
node and collect `identifier` children.  This works for Go (parameters
field), JavaScript (formal_parameters), Rust (parameters), Java
(formal_parameters), etc.

**Files:** `models.py` (ensure field exists), `analyzers/go_analyzer.py`,
`analyzers/js_analyzer.py`, `analyzers/generic_ts_analyzer.py`

### 8.2. `ClassInfo` lacks `members` / `fields`
**Problem:** Classes are recorded with name and kind but no member list.
Consumers cannot tell what methods or fields a class has without
cross-referencing the functions list.

**Approach:** Add `members: List[str]` to `ClassInfo`.  For Python,
extract attribute assignments in `__init__`.  For Go, extract struct
fields.  For Java, extract field declarations.

**Files:** `models.py`, all analyzers

### 8.3. No visibility/access modifier
**Problem:** There is no way to distinguish `public` vs `private` vs
`protected` symbols.  In Go (unexported = lowercase), Python
(`_prefix`), Java (`private` keyword), this information is readily
available.

**Approach:** Add `visibility: str` to `FunctionInfo` and `ClassInfo`.
Infer from naming conventions (Go/Python) or explicit keywords
(Java/C++).

**Files:** `models.py`, all analyzers

### 8.4. `metadata` dict is unused
**Problem:** If `FunctionInfo` or `ClassInfo` has a `metadata` field, it is
not populated by any analyzer.  Language-specific data (decorators,
annotations, modifiers) has nowhere to go.

**Approach:** Define conventions for common metadata keys:
`{"decorators": [...], "annotations": [...], "modifiers": [...],
"overrides": bool}`.  Populate them in each analyzer.

**Files:** `models.py`, all analyzers

### 8.5. No relationship / dependency graph
**Problem:** The output is a flat list of symbols per file.  Cross-file
relationships (imports, inheritance, interface implementation) are
not modeled.

**Approach:** Add a post-processing pass that builds a dependency graph
from the collected `FileAnalysis` results.  Link imports to the files
that define the imported symbols.

**Files:** New `indexing/dependency_graph.py`

---

## 9. Indexing Pipeline

### 9.1. Incremental indexing
**Problem:** The pipeline re-indexes the entire repository on every run.
For large codebases, this is slow and wasteful.

**Approach:** Store file hashes (e.g. SHA-256) alongside embeddings.  On
subsequent runs, skip files whose hash hasn't changed.  Remove
embeddings for deleted files.

**Files:** `indexing/full_pipeline.py`, `indexing/embedder.py`

### 9.2. Parallel file processing
**Problem:** Files are processed sequentially.  On multi-core machines,
this leaves significant performance on the table.

**Approach:** Use `concurrent.futures.ProcessPoolExecutor` or `ThreadPoolExecutor`
to process files in parallel.  Ensure thread-safe access to the vector
store.

**Files:** `indexing/full_pipeline.py`

### 9.3. File size limits
**Problem:** Very large files (generated code, minified bundles, vendored
dependencies) can dominate indexing time and storage.

**Approach:** Skip files exceeding a configurable size limit (e.g. 1 MB).
Optionally, skip common generated-file directories (`vendor/`,
`node_modules/`, `dist/`, `build/`, `.git/`).

**Files:** `indexing/full_pipeline.py`

### 9.4. Binary file detection
**Problem:** Binary files (images, compiled objects, `.wasm`) are not
explicitly detected.  Attempting to parse them wastes time and may
produce garbage.

**Approach:** Check for null bytes in the first 8 KB of each file.  If
found, skip the file.  Also skip by extension (`.png`, `.jpg`, `.so`,
`.dll`, `.wasm`, etc.).

**Files:** `indexing/full_pipeline.py`

### 9.5. Git-aware indexing
**Problem:** The pipeline does not use git information.  It indexes the
working tree, which may include uncommitted changes, build artifacts,
and `.gitignore`-d files.

**Approach:** Use `git ls-files` to enumerate tracked files.  Optionally,
use `git diff` to detect changes since the last index for true
incremental updates.

**Files:** `indexing/full_pipeline.py`

---

## 10. Search & Retrieval

### 10.1. Fuzzy matching
**Problem:** Search is exact or embedding-based.  Typos and minor naming
variations (`getUser` vs `get_user` vs `GetUser`) may not match.

**Approach:** Add a fuzzy matching layer (e.g. `thefuzz` / `rapidfuzz`)
that supplements embedding search.  Use it as a fallback when
embedding similarity is below a threshold.

**Files:** `indexing/embedder.py`, `indexing/search.py` (or equivalent)

### 10.2. Result ranking
**Problem:** Search results are returned without explicit ranking beyond
embedding similarity.  Code symbols that match the query directly
by name should rank higher than code that merely mentions the term.

**Approach:** Implement a hybrid scoring function that combines embedding
cosine similarity with text-match boosts (exact name match, file
path relevance, recency/usage frequency).

**Files:** Search / retrieval layer

### 10.3. File path filters
**Problem:** Search cannot be scoped to specific directories, file types,
or glob patterns.

**Approach:** Add optional filter parameters to the search function:
`path_prefix`, `extension`, `language`.  Apply as pre- or
post-filters on results.

**Files:** Search / retrieval layer

---

## 11. LLM Integration

### 11.1. Context window management
**Problem:** Large repositories may produce a context that exceeds the LLM's
context window.  There is no mechanism to truncate or summarize
selectively.

**Approach:** Implement a prioritized truncation strategy: keep the
repository overview, configuration files, and the most relevant code
sections (by embedding similarity to the query).  Drop low-relevance
sections first.

**Files:** `indexing/analyze_any_repo.py` or equivalent

### 11.2. Multi-provider support
**Problem:** The system is tightly coupled to Ollama.  Using OpenAI,
Anthropic, or other providers requires code changes.

**Approach:** Abstract behind an `LLMProvider` interface with
implementations for Ollama, OpenAI, Anthropic, and a mock provider for
testing.  Select via configuration.

**Files:** LLM client layer

### 11.3. Streaming responses
**Problem:** LLM responses are received in full after the model finishes.
For long analyses, the user sees no output until completion.

**Approach:** Use streaming APIs (`stream=True` in OpenAI, equivalent in
Ollama) and print tokens as they arrive.

**Files:** LLM client layer

### 11.4. Prompt caching
**Problem:** The same repository context is re-sent on every query.  For
repositories that don't change between queries, this is wasteful.

**Approach:** Cache the context portion of the prompt.  Only re-send the
query portion.  Some providers (Anthropic, OpenAI) support explicit
prompt caching APIs.

**Files:** LLM client layer

---

## 12. Testing

### 12.1. No test suite
**Problem:** There are no automated tests.  Refactoring or adding features
relies entirely on manual verification.

**Approach:** Add a `tests/` directory with:
- Unit tests per analyzer using fixture files
- Property-based tests for edge cases
- Integration tests for the full pipeline
- Snapshot tests for `FileAnalysis` output

Use `pytest` with `pytest-cov` for coverage reporting.

**Files:** New `tests/` directory

### 12.2. Golden file fixtures
**Problem:** Without test fixtures, it is difficult to verify that analyzer
changes don't regress on real-world code patterns.

**Approach:** Create a `tests/fixtures/` directory with small, representative
source files for each supported language.  Record expected
`FileAnalysis` output as JSON "golden" files.  Compare on each run.

**Files:** New `tests/fixtures/`, `tests/` test files

### 12.3. Performance benchmarks
**Problem:** There are no benchmarks.  Performance regressions from new
features go undetected.

**Approach:** Add `pytest-benchmark` tests that time analysis of large
fixture files.  Track over time with CI.

**Files:** New `tests/benchmarks/`

---

## 13. Operational & DevEx

### 13.1. Structured logging
**Problem:** The codebase uses `print()` and `warnings.warn()` for output.
There is no structured logging with severity levels.

**Approach:** Replace `print` and `warnings` with Python's `logging` module.
Use JSON-structured logging for machine-parseable output in production.

**Files:** Throughout

### 13.2. CLI interface
**Problem:** `analyze_any_repo.py` is a script, not a CLI tool.  There are
no subcommands, help text, or argument validation.

**Approach:** Wrap in `click` or `argparse` with subcommands:
`index`, `search`, `analyze`, `serve`.  Add `--help`, `--verbose`,
`--config` flags.

**Files:** New `cli.py` or refactor `analyze_any_repo.py`

### 13.3. Configuration file
**Problem:** Settings (model name, Ollama URL, file size limits, excluded
paths) are hardcoded or scattered across scripts.

**Approach:** Support a `pyproject.toml` / `.code-index.toml` config file
with sensible defaults.  Allow CLI flags to override.

**Files:** New `config.py`, update all consumers

### 13.4. Progress reporting
**Problem:** For large repositories, the indexing pipeline provides no
progress feedback.  The user sees nothing until completion.

**Approach:** Add a progress bar (`tqdm`) or periodic status prints
showing files processed, total progress, and estimated time remaining.

**Files:** `indexing/full_pipeline.py`

### 13.5. Error reporting
**Problem:** Files that fail to parse are silently skipped.  The user has
no visibility into which files failed and why.

**Approach:** Collect parse failures in a structured report.  At the end
of the pipeline, print a summary: "X files succeeded, Y files failed"
with file paths and error messages for failures.

**Files:** `indexing/full_pipeline.py`

### 13.6. Documentation completeness
**Problem:** The `USAGE_GUIDE.md` focuses on `analyze_any_repo.py`.  There
is no API documentation, contributor guide, or architecture overview.

**Approach:** Add:
- `ARCHITECTURE.md` — system design and data flow
- `CONTRIBUTING.md` — how to add a new language analyzer
- `API.md` — programmatic API reference
- Inline docstrings with NumPy-style parameter documentation

**Files:** Documentation files throughout

---

## Priority Matrix

| Area | Impact | Effort | Suggested Priority |
|------|--------|--------|--------------------|
| Testing (#12) | Very High | Medium | **P0** — foundation for all other work |
| Incremental indexing (#9.1) | High | Medium | **P0** — performance bottleneck |
| TypeScript support (#3.1) | High | Medium | **P1** — very common language |
| Python error-tolerant parsing (#1.6) | High | Low | **P1** — affects many repos |
| Structured logging (#13.1) | Medium | Low | **P1** — operational hygiene |
| File size / binary limits (#9.3, #9.4) | Medium | Low | **P1** — easy wins |
| Polyglot file detection (#7.2) | Medium | High | **P2** — complex but valuable |
| Kotlin / Swift / PHP (#4.1–4.3) | Medium | Low each | **P2** — add incrementally |
| Dependency graph (#8.5) | High | High | **P2** — significant new component |
| Multi-provider LLM (#11.2) | Medium | Medium | **P2** — flexibility |
| Makefile / Dockerfile analyzers (#6.1, #6.2) | Low | Low | **P3** — niche improvement |
| Visibility modifiers (#8.3) | Medium | Medium | **P3** — nice to have |

---

*This document is a living artifact.  Update it as improvements are
implemented or new limitations are discovered.*
