# analyzers/generic_ts_analyzer.py
"""
Config-driven tree-sitter analyzer for languages beyond Python/Go/JavaScript.

Supported languages (install the matching pip package):
  Rust    → pip install tree-sitter-rust
  Java    → pip install tree-sitter-java
  Ruby    → pip install tree-sitter-ruby
  C       → pip install tree-sitter-c
  C++     → pip install tree-sitter-cpp
  Lua     → pip install tree-sitter-lua

Each language is described by a LanguageProfile dict so that the same
walking logic handles all of them without per-language code.

Usage (called by language_router):
    from analyzers.generic_ts_analyzer import analyze_generic
    result = analyze_generic(file_path, "rust")
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from models import FileAnalysis, FunctionInfo, ClassInfo, VariableInfo


# ---------------------------------------------------------------------------
# Language profile schema
# ---------------------------------------------------------------------------

@dataclass
class LanguageProfile:
    # pip / importlib module name, e.g. "tree_sitter_rust"
    module: str
    # Display name used in FileAnalysis.language
    language_name: str
    # Tree-sitter node types that represent functions/methods
    func_types: List[str]
    # Tree-sitter node types that represent types/classes/impls
    class_types: List[str]
    # Tree-sitter node types that represent import/use/require statements
    import_types: List[str]
    # Tree-sitter node types that represent variable/constant declarations
    var_types: List[str] = field(default_factory=list)
    # How to get the name from a func/class/node.
    # "field:<name>"       → node.child_by_field_name("<name>")
    # "first_identifier"   → first child with type "identifier"
    # "text"               → entire node text (for import statements)
    func_name_strategy: str = "field:name"
    class_name_strategy: str = "field:name"
    import_text_strategy: str = "text"
    var_name_strategy: str = "field:name"
    # Map node.type → human-readable kind label shown in metadata
    kind_map: Dict[str, str] = field(default_factory=dict)
    # Map var node.type → kind label for VariableInfo (e.g. "const", "static")
    var_kind_map: Dict[str, str] = field(default_factory=dict)
    # Node types considered doc comments (looked for immediately before decls)
    doc_types: List[str] = field(default_factory=list)
    # Field name on var node that holds the value (if any)
    var_value_field: Optional[str] = None


# ---------------------------------------------------------------------------
# Language profiles
# ---------------------------------------------------------------------------

LANGUAGE_PROFILES: Dict[str, LanguageProfile] = {
    "rust": LanguageProfile(
        module="tree_sitter_rust",
        language_name="rust",
        func_types=["function_item"],
        class_types=["struct_item", "enum_item", "trait_item", "impl_item"],
        import_types=["use_declaration"],
        var_types=["const_item", "static_item"],
        func_name_strategy="field:name",
        class_name_strategy="field:name",
        import_text_strategy="text",
        var_name_strategy="field:name",
        var_value_field="value",
        kind_map={
            "struct_item": "struct",
            "enum_item": "enum",
            "trait_item": "trait",
            "impl_item": "impl",
        },
        var_kind_map={
            "const_item": "const",
            "static_item": "static",
        },
        doc_types=["line_comment", "block_comment"],
    ),
    "java": LanguageProfile(
        module="tree_sitter_java",
        language_name="java",
        func_types=["method_declaration", "constructor_declaration"],
        class_types=["class_declaration", "interface_declaration", "enum_declaration"],
        import_types=["import_declaration"],
        var_types=["field_declaration"],
        func_name_strategy="field:name",
        class_name_strategy="field:name",
        import_text_strategy="text",
        var_name_strategy="field:name",
        var_value_field="value",
        kind_map={
            "class_declaration": "class",
            "interface_declaration": "interface",
            "enum_declaration": "enum",
        },
        var_kind_map={
            "field_declaration": "field",
        },
        doc_types=["block_comment"],
    ),
    "ruby": LanguageProfile(
        module="tree_sitter_ruby",
        language_name="ruby",
        func_types=["method", "singleton_method"],
        class_types=["class", "module"],
        import_types=["call"],          # require / require_relative calls
        func_name_strategy="field:name",
        class_name_strategy="field:name",
        import_text_strategy="text",
        kind_map={
            "class": "class",
            "module": "module",
        },
        doc_types=["comment"],
    ),
    "c": LanguageProfile(
        module="tree_sitter_c",
        language_name="c",
        func_types=["function_definition"],
        class_types=["struct_specifier", "enum_specifier", "union_specifier"],
        import_types=["preproc_include"],
        var_types=["declaration"],
        func_name_strategy="nested:declarator/function_declarator/declarator",
        class_name_strategy="field:name",
        import_text_strategy="text",
        var_name_strategy="first_identifier",
        var_value_field="value",
        kind_map={
            "struct_specifier": "struct",
            "enum_specifier": "enum",
            "union_specifier": "union",
        },
        var_kind_map={
            "declaration": "var",
        },
        doc_types=["comment"],
    ),
    "cpp": LanguageProfile(
        module="tree_sitter_cpp",
        language_name="cpp",
        func_types=["function_definition"],
        class_types=[
            "class_specifier", "struct_specifier",
            "enum_specifier", "namespace_definition",
        ],
        import_types=["preproc_include"],
        var_types=["declaration"],
        func_name_strategy="nested:declarator/function_declarator/declarator",
        class_name_strategy="field:name",
        import_text_strategy="text",
        var_name_strategy="first_identifier",
        var_value_field="value",
        kind_map={
            "class_specifier": "class",
            "struct_specifier": "struct",
            "enum_specifier": "enum",
            "namespace_definition": "namespace",
        },
        var_kind_map={
            "declaration": "var",
        },
        doc_types=["comment"],
    ),
    "lua": LanguageProfile(
        module="tree_sitter_lua",
        language_name="lua",
        func_types=["function_declaration", "local_function"],
        class_types=[],
        import_types=["function_call"],   # require("...") calls
        func_name_strategy="field:name",
        class_name_strategy="field:name",
        import_text_strategy="text",
        kind_map={},
        doc_types=["comment"],
    ),
}


# ---------------------------------------------------------------------------
# Tree-sitter loading (lazy, per-language)
# ---------------------------------------------------------------------------

_loaded_parsers: Dict[str, object] = {}


def _load_parser(profile: LanguageProfile):
    """
    Lazily import and cache the tree-sitter parser for a language.
    Returns (parser, Language) or raises ImportError with an install hint.
    """
    lang = profile.language_name
    if lang in _loaded_parsers:
        return _loaded_parsers[lang]

    try:
        import importlib
        ts_module = importlib.import_module(profile.module)
    except ImportError:
        raise ImportError(
            f"tree-sitter grammar for '{lang}' is not installed.\n"
            f"  pip install {profile.module.replace('_', '-')}"
        )

    from tree_sitter import Language, Parser

    language_ptr = ts_module.language()
    ts_language = Language(language_ptr)
    parser = Parser()
    parser.language = ts_language

    _loaded_parsers[lang] = (parser, ts_language)
    return parser, ts_language


# ---------------------------------------------------------------------------
# AST walking helpers
# ---------------------------------------------------------------------------

def _find_all(node, types: List[str]) -> List:
    """Iteratively collect all descendant nodes whose type is in `types`.

    Uses an explicit stack instead of recursion to avoid hitting Python's
    recursion limit on deeply nested ASTs (e.g. heavily nested C++ templates).
    """
    type_set = set(types)
    results = []
    stack = [node]
    while stack:
        current = stack.pop()
        if current.type in type_set:
            results.append(current)
        # Extend with children — reversed so leftmost is processed first
        children = current.children
        if children:
            stack.extend(reversed(children))
    return results


def _first_identifier(node) -> Optional[str]:
    """Return the text of the first `identifier` or `type_identifier` child."""
    for child in node.children:
        if child.type in ("identifier", "type_identifier", "name"):
            return child.text.decode("utf-8", errors="ignore")
    for child in node.children:
        result = _first_identifier(child)
        if result:
            return result
    return None


def _resolve_name(node, strategy: str, code: bytes) -> Optional[str]:
    """
    Extract the symbol name from a node using the given strategy string.

    Strategies:
      "field:<name>"          → node.child_by_field_name("<name>")
      "first_identifier"      → first identifier child (recursive)
      "text"                  → full node text, decoded
      "nested:<f1>/<f2>/..."  → chain of child_by_field_name lookups
    """
    if strategy.startswith("field:"):
        field_name = strategy[len("field:"):]
        child = node.child_by_field_name(field_name)
        if child:
            return child.text.decode("utf-8", errors="ignore")
        return None

    if strategy == "first_identifier":
        return _first_identifier(node)

    if strategy == "text":
        return node.text.decode("utf-8", errors="ignore").strip()

    if strategy.startswith("nested:"):
        # e.g. "nested:declarator/function_declarator/declarator"
        steps = strategy[len("nested:"):].split("/")
        cur = node
        for step in steps:
            if cur is None:
                return None
            cur = cur.child_by_field_name(step)
        if cur:
            return _first_identifier(cur) or cur.text.decode("utf-8", errors="ignore")
        return None

    return None


def _get_signature(node, code: bytes) -> str:
    """Extract the signature: everything up to but not including the body block."""
    body = node.child_by_field_name("body")
    if body:
        sig_bytes = code[node.start_byte: body.start_byte]
        return " ".join(sig_bytes.decode("utf-8", errors="ignore").split())
    first_line = code[node.start_byte: node.end_byte].decode("utf-8", errors="ignore")
    return first_line.splitlines()[0].strip()


def _get_leading_doc(node, code: bytes, doc_types: List[str]) -> str:
    """Return the text of the preceding sibling if it is a doc comment."""
    parent = node.parent
    if parent is None:
        return ""
    siblings = list(parent.children)
    idx = next((i for i, c in enumerate(siblings) if c.id == node.id), -1)
    if idx <= 0:
        return ""
    prev = siblings[idx - 1]
    if prev.type in doc_types:
        raw = prev.text.decode("utf-8", errors="ignore")
        # Strip comment markers
        raw = re.sub(r"^[/*#\s]+|[/*\s]+$", "", raw, flags=re.MULTILINE)
        return raw.strip()
    return ""


# Ruby-specific: is this a require/require_relative call?
def _is_ruby_require(node) -> bool:
    if node.type != "call":
        return False
    method = node.child_by_field_name("method")
    if method and method.text.decode("utf-8", errors="ignore") in (
        "require", "require_relative", "require_all"
    ):
        return True
    return False


# Lua-specific: is this a require("...") call?
def _is_lua_require(node) -> bool:
    if node.type != "function_call":
        return False
    # First child should be the function name
    for child in node.children:
        if child.type in ("identifier", "name") and child.text.decode(
            "utf-8", errors="ignore"
        ) == "require":
            return True
    return False


def _is_var_node_interesting(node, profile: LanguageProfile) -> bool:
    """
    Filter out variable/const nodes that aren't useful to index.
    For C/C++ declarations, skip function declarations and pointer-to-function
    declarations that have a "type:function_declarator" child (already captured
    as functions).
    """
    if profile.language_name in ("c", "cpp"):
        # Skip declarations that are function definitions (already captured)
        for child in node.children:
            if child.type == "function_declarator":
                return False
        # Skip declarations without an initializer (just forward declarations)
        if node.child_by_field_name("value") is None:
            return False
        # Skip pointer-to-function declarations
        type_node = node.child_by_field_name("type")
        if type_node and b"(" in type_node.text:
            return False
    return True


import re  # noqa: E402 (needed for _get_leading_doc)


# ---------------------------------------------------------------------------
# Core analyzer
# ---------------------------------------------------------------------------

def analyze_generic(file_path: str, language: str) -> Optional[FileAnalysis]:
    """
    Parse `file_path` using the tree-sitter grammar registered for `language`.

    Returns a FileAnalysis or None if the grammar is not installed or the
    file cannot be parsed.
    """
    profile = LANGUAGE_PROFILES.get(language)
    if profile is None:
        return None

    try:
        parser, _ = _load_parser(profile)
    except ImportError as e:
        # Grammar not installed — log once and return None
        import warnings
        warnings.warn(str(e), stacklevel=2)
        return None

    try:
        with open(file_path, "rb") as f:
            code = f.read()
    except Exception:
        return None

    tree = parser.parse(code)
    root = tree.root_node

    functions: List[FunctionInfo] = []
    classes: List[ClassInfo] = []
    imports: List[str] = []
    variables: List[VariableInfo] = []

    # --- Functions ---
    for node in _find_all(root, profile.func_types):
        # Ruby require() calls can appear in func_types list — skip them here
        if profile.language_name == "ruby" and _is_ruby_require(node):
            continue
        if profile.language_name == "lua" and _is_lua_require(node):
            continue

        name = _resolve_name(node, profile.func_name_strategy, code)
        if not name:
            continue

        # For Java, qualify method with enclosing class name if present
        if profile.language_name == "java" and node.parent:
            parent_name = _resolve_name(
                node.parent, profile.class_name_strategy, code
            )
            if parent_name:
                name = f"{parent_name}.{name}"

        # For Rust impl items, try to get the impl type as the receiver
        if profile.language_name == "rust" and node.parent and node.parent.type == "declaration_list":
            impl_node = node.parent.parent
            if impl_node and impl_node.type == "impl_item":
                impl_type = impl_node.child_by_field_name("type")
                if impl_type:
                    receiver = impl_type.text.decode("utf-8", errors="ignore").strip()
                    name = f"{receiver}.{name}"

        start_line = node.start_point[0]
        sig = _get_signature(node, code)
        doc = _get_leading_doc(node, code, profile.doc_types)

        functions.append(
            FunctionInfo(
                name=name,
                line=start_line,
                signature=sig,
                docstring=doc,
            )
        )

    # --- Classes / types ---
    for node in _find_all(root, profile.class_types):
        name = _resolve_name(node, profile.class_name_strategy, code)
        if not name:
            # impl items might not have a name field — use the type field instead
            if node.type == "impl_item":
                type_node = node.child_by_field_name("type")
                if type_node:
                    name = f"impl({type_node.text.decode('utf-8', errors='ignore').strip()})"
        if not name:
            continue

        start_line = node.start_point[0]
        kind = profile.kind_map.get(node.type, node.type)
        doc = _get_leading_doc(node, code, profile.doc_types)

        classes.append(
            ClassInfo(
                name=name,
                line=start_line,
                kind=kind,
                docstring=doc,
            )
        )

    # --- Variables / constants ---
    if profile.var_types:
        for node in _find_all(root, profile.var_types):
            if not _is_var_node_interesting(node, profile):
                continue

            name = _resolve_name(node, profile.var_name_strategy, code)
            if not name:
                continue

            # Extract value text if a value field is configured
            value = ""
            if profile.var_value_field:
                val_node = node.child_by_field_name(profile.var_value_field)
                if val_node:
                    raw = val_node.text.decode("utf-8", errors="ignore").strip()
                    value = raw[:120] if len(raw) > 120 else raw

            # For Java fields, qualify with the class name if inside a class body
            if profile.language_name == "java" and node.parent:
                parent_name = _resolve_name(
                    node.parent, profile.class_name_strategy, code
                )
                if parent_name:
                    name = f"{parent_name}.{name}"

            start_line = node.start_point[0]
            kind = profile.var_kind_map.get(node.type, "var")
            doc = _get_leading_doc(node, code, profile.doc_types)

            variables.append(
                VariableInfo(
                    name=name,
                    line=start_line,
                    kind=kind,
                    value=value,
                    docstring=doc,
                )
            )

    # --- Imports ---
    for node in _find_all(root, profile.import_types):
        # Ruby: only require/require_relative calls
        if profile.language_name == "ruby" and not _is_ruby_require(node):
            continue
        # Lua: only require() calls
        if profile.language_name == "lua" and not _is_lua_require(node):
            continue

        text = node.text.decode("utf-8", errors="ignore").strip()
        # Trim common boilerplate from import text
        if profile.language_name in ("c", "cpp"):
            text = text.lstrip("#").strip()  # remove "#" from preproc_include
        imports.append(text)

    return FileAnalysis(
        file_path=file_path,
        language=profile.language_name,
        functions=functions,
        classes=classes,
        imports=imports,
        variables=variables,
    )