# analyzers/python_analyzer.py

import ast
import os
from typing import List, Optional
from models import FileAnalysis, FunctionInfo, ClassInfo, VariableInfo


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


def _process_function(
    node, source_lines: List[str], class_name: str = ""
) -> FunctionInfo:
    signature = _get_signature(node, source_lines)
    docstring = ast.get_docstring(node) or ""
    params = [arg.arg for arg in node.args.args]
    return_type = _get_return_annotation(node)
    full_name = f"{class_name}.{node.name}" if class_name else node.name
    return FunctionInfo(
        name=full_name,
        line=node.lineno,
        signature=signature,
        docstring=docstring,
        params=params,
        return_type=return_type,
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


def _is_all_uppercase(name: str) -> bool:
    """Check if a name is all uppercase (with optional underscores), heuristic for constants."""
    stripped = name.strip()
    if not stripped:
        return False
    # Must have at least two characters or be a single uppercase letter
    return stripped.replace("_", "").isupper() and len(stripped) > 1


def _extract_module_level_variables(tree, source_lines: List[str]) -> List[VariableInfo]:
    """Extract module-level assignments where LHS is an ALL_CAPS identifier (constant heuristic)."""
    variables: List[VariableInfo] = []

    for node in tree.body:
        if isinstance(node, ast.Assign):
            # Get all target names
            for target in node.targets:
                if isinstance(target, ast.Name) and _is_all_uppercase(target.id):
                    # Try to get the value as a string
                    try:
                        value_str = ast.unparse(node.value)
                    except Exception:
                        value_str = source_lines[node.lineno - 1].strip() if node.lineno <= len(source_lines) else ""

                    # Check for type annotation in a comment (e.g., # type: int)
                    type_annotation = ""
                    if isinstance(node, ast.AnnAssign) and node.annotation:
                        try:
                            type_annotation = ast.unparse(node.annotation)
                        except Exception:
                            pass

                    variables.append(VariableInfo(
                        name=target.id,
                        line=node.lineno,
                        kind="const",
                        value=value_str[:200],
                        type_annotation=type_annotation,
                        docstring="",
                    ))

        elif isinstance(node, ast.AnnAssign):
            # Handle annotated assignments like CONST: int = 42
            if isinstance(node.target, ast.Name) and _is_all_uppercase(node.target.id):
                try:
                    value_str = ast.unparse(node.value) if node.value else ""
                except Exception:
                    value_str = source_lines[node.lineno - 1].strip() if node.lineno <= len(source_lines) else ""

                try:
                    type_annotation = ast.unparse(node.annotation)
                except Exception:
                    type_annotation = ""

                variables.append(VariableInfo(
                    name=node.target.id,
                    line=node.lineno,
                    kind="const",
                    value=value_str[:200],
                    type_annotation=type_annotation,
                    docstring="",
                ))

    return variables


def _extract_class_level_variables(class_node: ast.ClassDef, source_lines: List[str]) -> List[VariableInfo]:
    """Extract class-level assignments (class variables) from a class definition."""
    variables: List[VariableInfo] = []

    for item in class_node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Skip methods — they are not class-level variables
            continue

        if isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name):
                    try:
                        value_str = ast.unparse(item.value)
                    except Exception:
                        value_str = source_lines[item.lineno - 1].strip() if item.lineno <= len(source_lines) else ""

                    kind = "const" if _is_all_uppercase(target.id) else "class_var"

                    variables.append(VariableInfo(
                        name=target.id,
                        line=item.lineno,
                        kind=kind,
                        value=value_str[:200],
                        type_annotation="",
                        docstring="",
                    ))

        elif isinstance(item, ast.AnnAssign):
            if isinstance(item.target, ast.Name):
                try:
                    value_str = ast.unparse(item.value) if item.value else ""
                except Exception:
                    value_str = source_lines[item.lineno - 1].strip() if item.lineno <= len(source_lines) else ""

                try:
                    type_annotation = ast.unparse(item.annotation)
                except Exception:
                    type_annotation = ""

                kind = "const" if _is_all_uppercase(item.target.id) else "class_var"

                variables.append(VariableInfo(
                    name=item.target.id,
                    line=item.lineno,
                    kind=kind,
                    value=value_str[:200],
                    type_annotation=type_annotation,
                    docstring="",
                ))

    return variables


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
            variables=[],
        )

    functions: List[FunctionInfo] = []
    classes: List[ClassInfo] = []
    imports: List[str] = []
    variables: List[VariableInfo] = []

    # Extract module-level variables
    variables.extend(_extract_module_level_variables(tree, source_lines))

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

            # Extract class-level variables
            variables.extend(_extract_class_level_variables(node, source_lines))

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
        variables=variables,
    )
