# analyzers/go_analyzer.py

import tree_sitter_go
from tree_sitter import Language, Parser
from typing import List
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


def _extract_struct_fields(type_node, code: bytes) -> List[FieldInfo]:
    """Extract field definitions from a struct_type node."""
    fields: List[FieldInfo] = []

    for child in type_node.children:
        if child.type == "field_definition":
            # Determine field name
            field_name = ""
            field_identifier_nodes = _find_nodes_of_type(child, "field_identifier")
            if field_identifier_nodes:
                field_name = field_identifier_nodes[0].text.decode()
            elif child.child_by_field_name("name") is not None:
                # Could be a blank identifier (_)
                name_node = child.child_by_field_name("name")
                if name_node:
                    field_name = name_node.text.decode()
            # If neither field_identifier nor name child, it's an embedded field → name=""

            # Determine field type: everything except the name and tag children
            # Use the "type" field directly if available (works for named fields)
            type_node_child = child.child_by_field_name("type")
            if type_node_child:
                type_str = type_node_child.text.decode()
            else:
                # Embedded field: the entire node text (minus tag) is the type
                tag_node = child.child_by_field_name("tag")
                if tag_node:
                    type_str = code[child.start_byte:tag_node.start_byte].decode("utf-8", errors="ignore").strip()
                else:
                    type_str = child.text.decode("utf-8", errors="ignore").strip()

            # Determine struct tag
            tag = ""
            tag_node = child.child_by_field_name("tag")
            if tag_node:
                tag = tag_node.text.decode()

            fields.append(FieldInfo(
                name=field_name,
                type_str=type_str,
                tag=tag,
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
