# analyzers/go_analyzer.py

import tree_sitter_go
from tree_sitter import Language, Parser
from typing import List
from models import FileAnalysis, FunctionInfo, ClassInfo, VariableInfo, FieldInfo

GO_LANGUAGE_PTR = tree_sitter_go.language()
GO_LANGUAGE = Language(GO_LANGUAGE_PTR)


def _find_nodes_of_type(node, target_type: str):
    """Recursively find all nodes of a given type."""
    results = []
    if node.type == target_type:
        results.append(node)
    for child in node.children:
        results.extend(_find_nodes_of_type(child, target_type))
    return results


def _get_leading_comments(source_lines: List[str], start_line: int) -> str:
    """
    Extract consecutive // comment lines immediately before start_line (0-indexed).
    Skips at most one blank line between the comment block and the declaration.
    """
    comments = []
    i = start_line - 1
    # Allow one blank separator line between comment and declaration
    if i >= 0 and not source_lines[i].strip():
        i -= 1
    while i >= 0 and source_lines[i].strip().startswith("//"):
        comments.insert(0, source_lines[i].strip()[2:].strip())
        i -= 1
    return " ".join(comments)


def _get_signature(node, code: bytes) -> str:
    """Extract the function/method signature (everything up to but not including the body)."""
    body = node.child_by_field_name("body")
    if body:
        sig_bytes = code[node.start_byte:body.start_byte]
        return " ".join(sig_bytes.decode("utf-8", errors="ignore").split())
    # Fallback: first line of the declaration
    return code[node.start_byte:node.end_byte].decode("utf-8", errors="ignore").splitlines()[0].strip()


def _get_package(root) -> str:
    """Extract the package name from the root node."""
    for child in root.children:
        if child.type == "package_clause":
            name_node = child.child_by_field_name("name")
            if name_node:
                return name_node.text.decode()
    return ""


def _extract_struct_fields(type_node) -> List[FieldInfo]:
    """
    Extract fields from a Go struct_type node.

    Each field_definition node may have:
      - A "name" field (field name)
      - A "type" field (type identifier)
      - A "tag" field (struct tag like `json:"id" db:"id"`)
    """
    fields: List[FieldInfo] = []

    if type_node is None or type_node.type != "struct_type":
        return fields

    field_defs = _find_nodes_of_type(type_node, "field_definition")
    for fd in field_defs:
        name_node = fd.child_by_field_name("name")
        type_node_field = fd.child_by_field_name("type")
        tag_node = fd.child_by_field_name("tag")

        if name_node is None:
            continue

        name = name_node.text.decode("utf-8", errors="ignore")
        type_str = type_node_field.text.decode("utf-8", errors="ignore") if type_node_field else ""
        tag = tag_node.text.decode("utf-8", errors="ignore") if tag_node else ""

        fields.append(FieldInfo(name=name, type_str=type_str, tag=tag))

    return fields


def _extract_variables(root, source_lines: List[str], code: bytes) -> List[VariableInfo]:
    """
    Extract const and var declarations from top-level nodes.

    Handles both single declarations (const foo = 1) and parenthesised blocks:
        const (
            APIVersion = "v2"
            DefaultTimeout = 30 * time.Second
        )
    """
    variables: List[VariableInfo] = []

    for child in root.children:
        if child.type not in ("const_declaration", "var_declaration"):
            continue

        kind = "const" if child.type == "const_declaration" else "var"
        doc = _get_leading_comments(source_lines, child.start_point[0])

        # Check for parenthesised block: const ( ... ) or var ( ... )
        # The first child may be a "const_spec" or a "(" block containing multiple specs
        specs = _find_nodes_of_type(child, "const_spec") + _find_nodes_of_type(child, "var_spec")

        for spec in specs:
            # name is the first child, value is field "value"
            name_node = spec.child_by_field_name("name")
            value_node = spec.child_by_field_name("value")
            type_node = spec.child_by_field_name("type")

            if name_node:
                name = name_node.text.decode("utf-8", errors="ignore")
                value = ""
                if value_node:
                    raw_value = value_node.text.decode("utf-8", errors="ignore").strip()
                    value = raw_value[:200]
                type_annotation = ""
                if type_node:
                    type_annotation = type_node.text.decode("utf-8", errors="ignore").strip()
                start_line = spec.start_point[0]

                # For parenthesised blocks, each spec can have its own comment
                spec_doc = doc
                # Check if this individual spec has its own leading comment
                own_doc = _get_leading_comments(source_lines, start_line)
                if own_doc and own_doc != doc:
                    spec_doc = own_doc

                variables.append(VariableInfo(
                    name=name,
                    line=start_line,
                    kind=kind,
                    value=value,
                    type_annotation=type_annotation,
                    docstring=spec_doc,
                ))

    return variables


def analyze_go(file_path):
    parser = Parser()
    parser.language = GO_LANGUAGE

    with open(file_path, "rb") as f:
        code = f.read()

    source_lines = code.decode("utf-8", errors="ignore").splitlines()
    tree = parser.parse(code)
    root = tree.root_node

    functions: List[FunctionInfo] = []
    classes: List[ClassInfo] = []
    imports: List[str] = []
    package = _get_package(root)

    for child in root.children:

        # Top-level functions
        if child.type == "function_declaration":
            name_node = child.child_by_field_name("name")
            if name_node:
                start = child.start_point[0]
                functions.append(FunctionInfo(
                    name=name_node.text.decode(),
                    line=start,
                    signature=_get_signature(child, code),
                    docstring=_get_leading_comments(source_lines, start),
                ))

        # Methods on types: func (r *MyStruct) MethodName()
        elif child.type == "method_declaration":
            name_node = child.child_by_field_name("name")
            receiver = child.child_by_field_name("receiver")
            if name_node:
                receiver_name = ""
                if receiver:
                    for spec in _find_nodes_of_type(receiver, "type_identifier"):
                        receiver_name = spec.text.decode()
                        break
                    if not receiver_name:
                        for spec in _find_nodes_of_type(receiver, "pointer_type"):
                            for ident in _find_nodes_of_type(spec, "type_identifier"):
                                receiver_name = ident.text.decode()
                                break
                            break

                full_name = f"{receiver_name}.{name_node.text.decode()}" if receiver_name else name_node.text.decode()
                start = child.start_point[0]
                functions.append(FunctionInfo(
                    name=full_name,
                    line=start,
                    signature=_get_signature(child, code),
                    docstring=_get_leading_comments(source_lines, start),
                ))

        # Type declarations: structs, interfaces, type aliases
        elif child.type == "type_declaration":
            for spec in _find_nodes_of_type(child, "type_spec"):
                name_node = spec.child_by_field_name("name")
                type_node = spec.child_by_field_name("type")
                if name_node:
                    kind = "type_alias"
                    struct_fields: List[FieldInfo] = []
                    if type_node:
                        if type_node.type == "struct_type":
                            kind = "struct"
                            struct_fields = _extract_struct_fields(type_node)
                        elif type_node.type == "interface_type":
                            kind = "interface"
                    start = spec.start_point[0]
                    classes.append(ClassInfo(
                        name=name_node.text.decode(),
                        line=start,
                        docstring=_get_leading_comments(source_lines, start),
                        kind=kind,
                        fields=struct_fields,
                    ))

        # Import declarations
        elif child.type == "import_declaration":
            for spec in _find_nodes_of_type(child, "import_spec"):
                path_node = spec.child_by_field_name("path")
                if path_node:
                    imports.append(path_node.text.decode().strip('"'))

    # Extract variables
    variables = _extract_variables(root, source_lines, code)

    return FileAnalysis(
        file_path=file_path,
        language="go",
        functions=functions,
        classes=classes,
        imports=imports,
        package=package,
        variables=variables,
    )
