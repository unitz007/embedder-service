# analyzers/python_analyzer.py

import ast
import os
from typing import List, Optional
from models import FileAnalysis, FunctionInfo, ClassInfo


def _get_signature(node, source_lines: List[str]) -> str:
    """Extract the def/class signature line(s) from source, stopping before the body."""
    start = node.lineno - 1  # 0-indexed
    sig_lines = []
    for i in range(start, min(start + 10, len(source_lines))):
        sig_lines.append(source_lines[i].rstrip())
        if source_lines[i].rstrip().endswith(":"):
            break
    return " ".join(l.strip() for l in sig_lines).rstrip(":")


def _get_return_annotation(node) -> str:
    """Return the return type annotation as a string, if present."""
    if node.returns is None:
        return ""
    try:
        return ast.unparse(node.returns)
    except Exception:
        return ""


def _get_decorators(node) -> List[str]:
    """Extract decorator strings from a function/method node, preserving source order."""
    decorators = []
    for dec in node.decorator_list:
        try:
            decorators.append("@" + ast.unparse(dec))
        except Exception:
            continue
    return decorators


def _process_function(
    node, source_lines: List[str], class_name: str = ""
) -> FunctionInfo:
    signature = _get_signature(node, source_lines)
    docstring = ast.get_docstring(node) or ""
    params = [arg.arg for arg in node.args.args]
    return_type = _get_return_annotation(node)
    decorators = _get_decorators(node)
    full_name = f"{class_name}.{node.name}" if class_name else node.name
    return FunctionInfo(
        name=full_name,
        line=node.lineno,
        signature=signature,
        docstring=docstring,
        params=params,
        return_type=return_type,
        decorators=decorators,
    )


def _get_package(file_path: str) -> str:
    """Derive a dotted module name from the file path (best-effort)."""
    parts = []
    path = os.path.splitext(file_path)[0]
    while True:
        head, tail = os.path.split(path)
        if not tail:
            break
        parts.insert(0, tail)
        init = os.path.join(head, "__init__.py")
        if not os.path.exists(init):
            break
        path = head
    return ".".join(parts) if parts else ""


def analyze_python(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        source = f.read()

    source_lines = source.splitlines()

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return FileAnalysis(
            file_path=file_path,
            language="python",
            functions=[],
            classes=[],
            imports=[],
        )

    functions: List[FunctionInfo] = []
    classes: List[ClassInfo] = []
    imports: List[str] = []

    # Traverse only top-level nodes to avoid duplicate chunks from ast.walk.
    # Methods are captured under their parent class with "ClassName.method_name" naming.
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(_process_function(node, source_lines))

        elif isinstance(node, ast.ClassDef):
            docstring = ast.get_docstring(node) or ""
            classes.append(ClassInfo(
                name=node.name,
                line=node.lineno,
                docstring=docstring,
                kind="class",
            ))
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    functions.append(_process_function(item, source_lines, class_name=node.name))

        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)

    return FileAnalysis(
        file_path=file_path,
        language="python",
        functions=functions,
        classes=classes,
        imports=imports,
        package=_get_package(file_path),
    )
