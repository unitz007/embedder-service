# graph.py
"""
Two complementary graphs built from the AST analysis:

1. Import Graph (file-level)
   - depends_on:  local files this file imports
   - imported_by: local files that import this file

2. Call Graph (symbol-level)
   - calls:       local functions this function calls
   - called_by:   local functions that call this function
   - external_calls: unresolved / stdlib / third-party calls

Both graphs are built in a single pipeline pass and injected into chunk metadata
so the LLM can answer relational questions like:
  "What calls authenticate()?"
  "What does UserService.Create depend on?"
"""

import ast
import os
from typing import Dict, List, Optional, Set, Any
from models import FileAnalysis


# ---------------------------------------------------------------------------
# Import graph
# ---------------------------------------------------------------------------

def _resolve_import(
    imp: str,
    current_file: str,
    all_files: Set[str],
    language: str,
) -> Optional[str]:
    """
    Attempt to resolve an import string to an actual file path in the repo.
    Returns the matching path or None if the import is external/unresolvable.
    """
    if language == "javascript":
        if not (imp.startswith("./") or imp.startswith("../")):
            return None
        base_dir = os.path.dirname(current_file)
        candidate = os.path.normpath(os.path.join(base_dir, imp))
        for suffix in ("", ".js", ".ts", "/index.js", "/index.ts"):
            if candidate + suffix in all_files:
                return candidate + suffix

    elif language == "python":
        parts = imp.split(".")
        module_suffix = os.path.join(*parts) + ".py"
        pkg_suffix = os.path.join(*parts, "__init__.py")
        for f in all_files:
            if f.endswith(module_suffix) or f.endswith(pkg_suffix):
                return f

    elif language == "go":
        pkg_name = imp.split("/")[-1]
        for f in all_files:
            if f.endswith(".go") and os.path.basename(os.path.dirname(f)) == pkg_name:
                return f

    return None


def build_import_graph(analyses: List[FileAnalysis]) -> Dict[str, Dict]:
    """
    Build a file-level import dependency graph.

    Returns:
        {
            file_path: {
                "depends_on": [file_path, ...],
                "imported_by": [file_path, ...],
                "unresolved_imports": [str, ...],
                "package": str,
            }
        }
    """
    all_files: Set[str] = {a.file_path for a in analyses}
    graph: Dict[str, Dict] = {
        a.file_path: {
            "depends_on": [],
            "imported_by": [],
            "unresolved_imports": [],
            "package": a.package,
        }
        for a in analyses
    }

    for analysis in analyses:
        for imp in analysis.imports:
            resolved = _resolve_import(imp, analysis.file_path, all_files, analysis.language)
            if resolved and resolved != analysis.file_path and resolved in graph:
                if resolved not in graph[analysis.file_path]["depends_on"]:
                    graph[analysis.file_path]["depends_on"].append(resolved)
                if analysis.file_path not in graph[resolved]["imported_by"]:
                    graph[resolved]["imported_by"].append(analysis.file_path)
            else:
                graph[analysis.file_path]["unresolved_imports"].append(imp)

    return graph


def enrich_chunks_with_graph(
    chunks: List[Dict], graph: Dict[str, Dict]
) -> List[Dict]:
    """Inject import graph data (depends_on, imported_by, package) into chunk metadata."""
    for chunk in chunks:
        file_path = chunk["metadata"].get("file_path", "")
        if file_path in graph:
            node = graph[file_path]
            chunk["metadata"]["depends_on"] = node["depends_on"]
            chunk["metadata"]["imported_by"] = node["imported_by"]
            chunk["metadata"]["package"] = node["package"]
    return chunks


# ---------------------------------------------------------------------------
# Call graph — helpers per language
# ---------------------------------------------------------------------------

def _collect_calls_python(func_node: ast.AST) -> List[str]:
    """Walk a Python function AST node and collect all call target names."""
    calls = []
    for node in ast.walk(func_node):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.append(node.func.attr)
    return calls


def _extract_all_calls_python(analysis: FileAnalysis) -> Dict[str, List[str]]:
    """
    Parse the Python file once and return {func_name: [called_names]} for every
    function in the analysis. Functions are looked up by line number.
    """
    result: Dict[str, List[str]] = {}
    try:
        with open(analysis.file_path, "r", encoding="utf-8", errors="ignore") as f:
            source = f.read()
        tree = ast.parse(source)
    except Exception:
        return result

    # Build line → AST node map for all function defs in the file
    line_to_node: Dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            line_to_node[node.lineno] = node

    for func in analysis.functions:
        func_node = line_to_node.get(func.line)
        if func_node is None:
            continue
        result[func.name] = _collect_calls_python(func_node)

    return result


def _collect_calls_recursive(node: Any, call_type: str, func_field: str, method_type: str, method_field: str) -> List[str]:
    """
    Generic recursive call collector for tree-sitter nodes.

    Args:
        node: tree-sitter node to walk
        call_type:   node type for call expressions (e.g. "call_expression")
        func_field:  field name for the callee (e.g. "function")
        method_type: node type for qualified calls (e.g. "selector_expression" in Go,
                     "member_expression" in JS)
        method_field: field name for the method name within qualified calls
    """
    calls = []
    if node.type == call_type:
        callee = node.child_by_field_name(func_field)
        if callee:
            if callee.type == "identifier":
                calls.append(callee.text.decode("utf-8", errors="ignore"))
            elif callee.type == method_type:
                field = callee.child_by_field_name(method_field)
                if field:
                    calls.append(field.text.decode("utf-8", errors="ignore"))
    for child in node.children:
        calls.extend(_collect_calls_recursive(child, call_type, func_field, method_type, method_field))
    return calls


def _extract_all_calls_go(analysis: FileAnalysis) -> Dict[str, List[str]]:
    """Parse the Go file once and return {func_name: [called_names]}."""
    result: Dict[str, List[str]] = {}
    try:
        import tree_sitter_go
        from tree_sitter import Language, Parser

        parser = Parser()
        parser.language = Language(tree_sitter_go.language())

        with open(analysis.file_path, "rb") as f:
            code = f.read()

        root = parser.parse(code).root_node

        # Build start_line → node map for function and method declarations
        line_to_node: Dict[int, Any] = {}
        for child in root.children:
            if child.type in ("function_declaration", "method_declaration"):
                line_to_node[child.start_point[0]] = child

        for func in analysis.functions:
            func_node = line_to_node.get(func.line)
            if func_node is None:
                continue
            result[func.name] = _collect_calls_recursive(
                func_node,
                call_type="call_expression",
                func_field="function",
                method_type="selector_expression",
                method_field="field",
            )
    except Exception:
        pass
    return result


def _collect_function_nodes_js(node: Any, out: Dict[int, Any]) -> None:
    """Recursively collect all function-like nodes keyed by the start line."""
    if node.type in (
        "function_declaration",
        "generator_function_declaration",
        "function_expression",
        "arrow_function",
        "method_definition",
    ):
        out[node.start_point[0]] = node
    for child in node.children:
        _collect_function_nodes_js(child, out)


def _extract_all_calls_js(analysis: FileAnalysis) -> Dict[str, List[str]]:
    """Parse JS/TS file once and return {func_name: [called_names]}."""
    result: Dict[str, List[str]] = {}
    try:
        import tree_sitter_javascript
        from tree_sitter import Language, Parser

        parser = Parser()
        parser.language = Language(tree_sitter_javascript.language())

        with open(analysis.file_path, "rb") as f:
            code = f.read()

        root = parser.parse(code).root_node

        line_to_node: Dict[int, Any] = {}
        _collect_function_nodes_js(root, line_to_node)

        for func in analysis.functions:
            func_node = line_to_node.get(func.line)
            if func_node is None:
                continue
            result[func.name] = _collect_calls_recursive(
                func_node,
                call_type="call_expression",
                func_field="function",
                method_type="member_expression",
                method_field="property",
            )
    except Exception:
        pass
    return result


# ---------------------------------------------------------------------------
# Call graph — main builder
# ---------------------------------------------------------------------------

def build_call_graph(analyses: List[FileAnalysis]) -> Dict[str, Dict]:
    """
    Build a symbol-level call graph across the entire codebase.

    Each function is identified by a unique key: "file_path::func_name"

    Returns:
        {
            "file_path::func_name": {
                "calls": ["file_path::func_name", ...], # resolved local calls
                "called_by": ["file_path::func_name", ...], # resolved callers
                "external_calls": ["name", ...], # unresolved calls
                "file_path": str,
                "function_name": str,
                "line": int,
            }
        }
    """
    # Build a lookup: simple_name → [full_key, ...]
    # Used to resolve call names to known function keys
    symbol_lookup: Dict[str, List[str]] = {}
    for analysis in analyses:
        for func in analysis.functions:
            key = f"{analysis.file_path}::{func.name}"
            # Index by full name (e.g. "MyClass.method") and simple name (e.g. "method")
            symbol_lookup.setdefault(func.name, []).append(key)
            simple = func.name.split(".")[-1]
            if simple != func.name:
                symbol_lookup.setdefault(simple, []).append(key)

    call_graph: Dict[str, Dict] = {}

    for analysis in analyses:
        # Extract raw calls for all functions in this file in a single parse
        if analysis.language == "python":
            raw_calls_map = _extract_all_calls_python(analysis)
        elif analysis.language == "go":
            raw_calls_map = _extract_all_calls_go(analysis)
        elif analysis.language == "javascript":
            raw_calls_map = _extract_all_calls_js(analysis)
        else:
            raw_calls_map = {}

        for func in analysis.functions:
            func_key = f"{analysis.file_path}::{func.name}"
            raw_calls = raw_calls_map.get(func.name, [])

            resolved: List[str] = []
            external: List[str] = []

            for call_name in set(raw_calls):
                # Skip self-calls
                if call_name == func.name or call_name == func.name.split(".")[-1]:
                    continue
                candidates = symbol_lookup.get(call_name, [])
                if candidates:
                    # Prefer same-file matches; fall back to all matches (capped at 5)
                    same_file = [c for c in candidates if c.startswith(analysis.file_path + "::")]
                    resolved.extend(same_file if same_file else candidates[:5])
                else:
                    external.append(call_name)

            call_graph[func_key] = {
                "calls": list(set(resolved)),
                "called_by": [],          # populated in the reverse pass below
                "external_calls": external,
                "file_path": analysis.file_path,
                "function_name": func.name,
                "line": func.line,
            }

    # Build reverse edges (called_by)
    for func_key, data in call_graph.items():
        for callee_key in data["calls"]:
            if callee_key in call_graph and func_key not in call_graph[callee_key]["called_by"]:
                call_graph[callee_key]["called_by"].append(func_key)

    return call_graph


def enrich_chunks_with_call_graph(
    chunks: List[Dict], call_graph: Dict[str, Dict]
) -> List[Dict]:
    """
    Inject call graph data into function chunk metadata.

    Adds to each function chunk:
      - calls:          list of "func_name" strings this function calls locally
      - called_by:      list of "func_name" strings that call this function locally
      - external_calls: list of unresolved call names (stdlib / third-party)
    """
    for chunk in chunks:
        if chunk["metadata"].get("chunk_type") != "function":
            continue
        file_path = chunk["metadata"].get("file_path", "")
        func_name = chunk["metadata"].get("function_name", "")
        func_key = f"{file_path}::{func_name}"

        if func_key not in call_graph:
            continue

        node = call_graph[func_key]
        # Store human-readable names (strip the file path prefix)
        chunk["metadata"]["calls"] = [k.split("::", 1)[-1] for k in node["calls"]]
        chunk["metadata"]["called_by"] = [k.split("::", 1)[-1] for k in node["called_by"]]
        chunk["metadata"]["external_calls"] = node["external_calls"]

    return chunks
