# analyzers/go_analyzer.py

import tree_sitter_go
from tree_sitter import Language, Parser
from typing import List, Optional
from models import FileAnalysis, FunctionInfo, ClassInfo, FieldInfo

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


def _extract_struct_fields(struct_type_node, code: bytes) -> List[FieldInfo]:
    """
    Extract field info from a struct_type node's field_definition children.

    In tree-sitter-go, each field_definition node has:
      - A "field_identifier" child for named fields
      - A "tag" child (optional) for struct tags like `json:"name,omitempty"`
      - A type child (direct child, not nested) for the field type

    Embedded fields (e.g. `http.Handler`) have no field_identifier child.
    Blank identifier fields (e.g. `_ int`) have field_identifier text "_".
    """
    fields: List[FieldInfo] = []

    for child in struct_type_node.children:
        if child.type != "field_definition":
            continue

        field_name = ""
        type_str = ""
        tag_str = ""

        # Look at direct children to extract name, type, and tag
        for fd_child in child.children:
            if fd_child.type == "field_identifier":
                field_name = fd_child.text.decode("utf-8", errors="ignore")
            elif fd_child.type == "tag":
                tag_str = fd_child.text.decode("utf-8", errors="ignore")
            # The type is any direct child that is not field_identifier, tag, or comment
            elif fd_child.type not in ("comment", "field_identifier", "tag"):
                type_str = fd_child.text.decode("utf-8", errors="ignore")

        fields.append(FieldInfo(
            name=field_name,
            type_str=type_str,
            tag=tag_str,
        ))

    return fields


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
                    # Extract generic type parameters if present
                    type_params = ""
                    for c in spec.children:
                        if c.type == "type_parameter_list":
                            type_params = c.text.decode("utf-8", errors="ignore")
                            break

                    kind = "type_alias"
                    struct_fields: List[FieldInfo] = []
                    if type_node:
                        if type_node.type == "struct_type":
                            kind = "struct"
                            struct_fields = _extract_struct_fields(type_node, code)
                        elif type_node.type == "interface_type":
                            kind = "interface"
                    start = spec.start_point[0]
                    classes.append(ClassInfo(
                        name=name_node.text.decode(),
                        line=start,
                        docstring=_get_leading_comments(source_lines, start),
                        kind=kind,
                        type_params=type_params,
                        fields=struct_fields,
                    ))

        # Import declarations
        elif child.type == "import_declaration":
            for spec in _find_nodes_of_type(child, "import_spec"):
                path_node = spec.child_by_field_name("path")
                if path_node:
                    imports.append(path_node.text.decode().strip('"'))

    return FileAnalysis(
        file_path=file_path,
        language="go",
        functions=functions,
        classes=classes,
        imports=imports,
        package=package,
    )
