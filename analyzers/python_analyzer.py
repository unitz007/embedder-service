# analyzers/python_analyzer.py

import ast
import os
import re
from typing import List
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


def _get_decorators(node) -> List[str]:
    """Extract decorator strings from a function/method node.

    Iterates over ``node.decorator_list`` and unparses each decorator AST
    node into its source representation (e.g. ``@staticmethod``,
    ``@app.route("/health")``).  The order matches source appearance
    (top-to-bottom), which is the order in ``decorator_list``.
    """
    decorators: List[str] = []
    for dec in node.decorator_list:
        try:
            decorators.append("@" + ast.unparse(dec))
        except Exception:
            # Fallback: skip decorators that cannot be unparsed
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


# ---------------------------------------------------------------------------
# Regex-based fallback for files with SyntaxError
# ---------------------------------------------------------------------------

# Patterns for top-level definitions (only when not indented)
_RE_DEF = re.compile(r'^(\s*)(async\s+)?def\s+(\w+)\s*\((.*)')
_RE_CLASS = re.compile(r'^(\s*)class\s+(\w+)')
_RE_IMPORT = re.compile(r'^(\s*)import\s+([\w.]+(?:\s*,\s*[\w.]+)*)')
_RE_FROM_IMPORT = re.compile(r'^(\s*)from\s+([\w.]+)\s+import')
_RE_TRIPLE_QUOTE_START = re.compile(r'^\s*("""|\'\'\')(.*?)("""|\'\'\')\s*$', re.DOTALL)
_RE_TRIPLE_QUOTE_OPEN = re.compile(r'^\s*("""|\'\'\')(.*?)(?<!\\)$', re.DOTALL)


def _try_extract_docstring(source_lines: List[str], start_idx: int) -> str:
    """Attempt to extract a triple-quoted docstring starting at or near *start_idx*.

    Looks at lines beginning from *start_idx* (0-based).  Skips blank lines
    and lines that look like decorators.  Returns the docstring content
    (without quotes) or an empty string if none is found within a small window.
    """
    for i in range(start_idx, min(start_idx + 3, len(source_lines))):
        line = source_lines[i]
        stripped = line.lstrip()

        # Skip blank lines between the definition and potential docstring
        if not stripped:
            continue

        # Skip decorator-like lines (shouldn't happen after def/class, but be safe)
        if stripped.startswith('@'):
            continue

        # Single-line triple-quoted string: """docstring"""
        m = _RE_TRIPLE_QUOTE_START.match(line)
        if m:
            return m.group(2).strip()

        # Opening triple-quote (multi-line docstring)
        m = _RE_TRIPLE_QUOTE_OPEN.match(line)
        if m:
            quote_char = m.group(1)
            content = m.group(2)
            # Look for closing quote on subsequent lines
            close_pattern = re.compile(re.escape(quote_char) + r'\s*$')
            for j in range(i + 1, min(i + 20, len(source_lines))):
                if close_pattern.search(source_lines[j]):
                    content += "\n" + source_lines[j][:source_lines[j].index(quote_char)] if quote_char in source_lines[j] else ""
                    break
                else:
                    content += "\n" + source_lines[j]
            return content.strip()

        # Not a docstring at all — stop looking
        break

    return ""


def _collect_signature_lines(source_lines: List[str], start_idx: int) -> str:
    """Collect the definition signature starting from *start_idx* (0-based).

    For multi-line signatures (e.g. with line continuations or open parens),
    we keep consuming lines until we find one ending with ``:``.
    """
    parts: List[str] = []
    for i in range(start_idx, min(start_idx + 10, len(source_lines))):
        parts.append(source_lines[i].rstrip())
        if source_lines[i].rstrip().endswith(":"):
            break
    return " ".join(l.strip() for l in parts).rstrip(":")


def _regex_fallback(source: str, file_path: str, source_lines: List[str]) -> FileAnalysis:
    """Best-effort extraction of top-level definitions using regex.

    This is invoked when ``ast.parse()`` fails with a :class:`SyntaxError`.
    It recovers ``def``/``async def``/``class`` definitions, and
    ``import``/``from ... import`` statements so that a file with a single
    syntax error on line 200 does not lose all symbols from lines 1-199.
    """
    functions: List[FunctionInfo] = []
    classes: List[ClassInfo] = []
    imports: List[str] = []

    for idx, line in enumerate(source_lines):
        stripped = line.lstrip()
        indent = len(line) - len(stripped)

        # Only consider top-level (or very shallow) definitions.
        # We use indent == 0 to match truly top-level items.
        if indent != 0:
            continue

        # --- def / async def ---
        m = _RE_DEF.match(line)
        if m:
            is_async = m.group(2) is not None
            name = m.group(3)
            signature = _collect_signature_lines(source_lines, idx)
            if is_async:
                signature = signature.replace("async def", "async def", 1)  # ensure "async" prefix kept
            docstring = _try_extract_docstring(source_lines, idx + 1)
            functions.append(FunctionInfo(
                name=name,
                line=idx + 1,  # 1-based
                signature=signature,
                docstring=docstring,
                params=[],
                return_type="",
                decorators=[],
            ))
            continue

        # --- class ---
        m = _RE_CLASS.match(line)
        if m:
            name = m.group(2)
            signature = _collect_signature_lines(source_lines, idx)
            docstring = _try_extract_docstring(source_lines, idx + 1)
            classes.append(ClassInfo(
                name=name,
                line=idx + 1,
                docstring=docstring,
                kind="class",
            ))
            continue

        # --- import X [, Y, ...] ---
        m = _RE_IMPORT.match(line)
        if m:
            modules_str = m.group(2)
            for mod in modules_str.split(","):
                mod = mod.strip()
                if mod:
                    imports.append(mod)
            continue

        # --- from X import ... ---
        m = _RE_FROM_IMPORT.match(line)
        if m:
            module = m.group(2)
            if module:
                imports.append(module)
            continue

    return FileAnalysis(
        file_path=file_path,
        language="python",
        functions=functions,
        classes=classes,
        imports=imports,
        package=_get_package(file_path),
        variables=[],
    )


def analyze_python(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        source = f.read()

    source_lines = source.splitlines()

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _regex_fallback(source, file_path, source_lines)

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