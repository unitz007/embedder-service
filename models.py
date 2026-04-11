# models.py

from dataclasses import dataclass, field
from typing import List


@dataclass
class FunctionInfo:
    name: str
    line: int
    signature: str = ""         # Full declaration e.g. "func (r *Repo) Find(id int) (*User, error)"
    docstring: str = ""         # Doc comment or Python docstring
    params: List[str] = field(default_factory=list)
    return_type: str = ""


@dataclass
class FieldInfo:
    """A single field within a struct, class, or interface."""
    name: str
    type_str: str = ""          # e.g. "int", "string", "time.Time"
    tag: str = ""               # e.g. `json:"id" db:"id"`


@dataclass
class ClassInfo:
    name: str
    line: int
    docstring: str = ""
    kind: str = ""              # "class", "struct", "interface", "type_alias", "enum"
    fields: List[FieldInfo] = field(default_factory=list)


@dataclass
class VariableInfo:
    name: str
    line: int
    kind: str = ""              # "const", "var", "let", "static", "class_var"
    value: str = ""             # The assigned value (truncated to 200 chars)
    type_annotation: str = ""   # e.g. "time.Duration", "string", "*Config"
    docstring: str = ""


@dataclass
class FileAnalysis:
    file_path: str
    language: str
    functions: List[FunctionInfo]
    classes: List[ClassInfo]
    imports: List[str]
    package: str = ""           # Go package name, Python module path, JS module name
    variables: List[VariableInfo] = field(default_factory=list)
