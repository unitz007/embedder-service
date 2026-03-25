# analyzers/js_analyzer.py

import tree_sitter_javascript
from tree_sitter import Language, Parser
from typing import List
from models import FileAnalysis, FunctionInfo, ClassInfo

JS_LANGUAGE_PTR = tree_sitter_javascript.language()
JS_LANGUAGE = Language(JS_LANGUAGE_PTR)


def _get_leading_comment(source_lines: List[str], start_line: int) -> str:
    """
    Extract JSDoc (/** ... */) or // comment block immediately above start_line (0-indexed).
    Skips one blank separator line.
    """
    i = start_line - 1
    if i >= 0 and not source_lines[i].strip():
        i -= 1
    if i < 0:
        return ""

    # JSDoc block: ends with */
    if source_lines[i].strip() == "*/":
        lines = []
        while i >= 0 and "/**" not in source_lines[i]:
            lines.insert(0, source_lines[i].strip().lstrip("*").strip())
            i -= 1
        if i >= 0:
            lines.insert(0, source_lines[i].strip().lstrip("/**").strip())
        return " ".join(l for l in lines if l)

    # Line comment block
    comments = []
    while i >= 0 and source_lines[i].strip().startswith("//"):
        comments.insert(0, source_lines[i].strip()[2:].strip())
        i -= 1
    return " ".join(comments)


def _get_signature(node, source_lines: List[str]) -> str:
    """Extract the first line of a declaration as a signature."""
    line = node.start_point[0]
    if line < len(source_lines):
        return source_lines[line].strip()
    return ""


def _extract_from_node(node, functions, classes, imports, source_lines):
    """
    Recursively extract symbols from a node, covering common JS/TS patterns:
      - function declarations (including generators)
      - class declarations and their methods
      - const/let arrow functions and function expressions
      - export statements (recurse into declaration)
      - import statements
    """
    t = node.type

    if t in ("function_declaration", "generator_function_declaration"):
        name_node = node.child_by_field_name("name")
        if name_node:
            start = node.start_point[0]
            functions.append(FunctionInfo(
                name=name_node.text.decode(),
                line=start,
                signature=_get_signature(node, source_lines),
                docstring=_get_leading_comment(source_lines, start),
            ))

    elif t == "class_declaration":
        name_node = node.child_by_field_name("name")
        if name_node:
            start = node.start_point[0]
            classes.append(ClassInfo(
                name=name_node.text.decode(),
                line=start,
                docstring=_get_leading_comment(source_lines, start),
                kind="class",
            ))
        # Extract methods from class body
        body = node.child_by_field_name("body")
        if body:
            for child in body.children:
                if child.type == "method_definition":
                    method_name = child.child_by_field_name("name")
                    if method_name:
                        m_start = child.start_point[0]
                        functions.append(FunctionInfo(
                            name=method_name.text.decode(),
                            line=m_start,
                            signature=_get_signature(child, source_lines),
                            docstring=_get_leading_comment(source_lines, m_start),
                        ))

    elif t in ("lexical_declaration", "variable_declaration"):
        # const foo = () => {} or const foo = function() {}
        for child in node.children:
            if child.type == "variable_declarator":
                name_node = child.child_by_field_name("name")
                value_node = child.child_by_field_name("value")
                if name_node and value_node and value_node.type in ("arrow_function", "function_expression"):
                    start = child.start_point[0]
                    functions.append(FunctionInfo(
                        name=name_node.text.decode(),
                        line=start,
                        signature=_get_signature(child, source_lines),
                        docstring=_get_leading_comment(source_lines, start),
                    ))

    elif t == "export_statement":
        declaration = node.child_by_field_name("declaration")
        if declaration:
            _extract_from_node(declaration, functions, classes, imports, source_lines)

    elif t == "import_statement":
        source_node = node.child_by_field_name("source")
        if source_node:
            imports.append(source_node.text.decode().strip("\"'"))


def analyze_js(file_path):
    parser = Parser()
    parser.language = JS_LANGUAGE

    with open(file_path, "rb") as f:
        code = f.read()

    source_lines = code.decode("utf-8", errors="ignore").splitlines()
    tree = parser.parse(code)
    root = tree.root_node

    functions: List[FunctionInfo] = []
    classes: List[ClassInfo] = []
    imports: List[str] = []

    for node in root.children:
        _extract_from_node(node, functions, classes, imports, source_lines)

    return FileAnalysis(
        file_path=file_path,
        language="javascript",
        functions=functions,
        classes=classes,
        imports=imports,
    )
