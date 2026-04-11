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
    """Represents a single field within a Go struct definition."""
    name: str                  # Field name ("" for embedded fields, "_" for blank identifiers)
    type_str: str              # Raw type text, e.g. "string", "*User", "[]int"
    tag: str                   # Raw struct tag string, e.g. `json:"name,omitempty"`, or "" if absent


@dataclass
class ClassInfo:
    name: str
    line: int
    docstring: str = ""
    kind: str = ""              # "class", "struct", "interface", "type_alias", "enum"
    fields: List[FieldInfo] = field(default_factory=list)


@dataclass
class FileAnalysis:
    file_path: str
    language: str
    functions: List[FunctionInfo]
    classes: List[ClassInfo]
    imports: List[str]
    package: str = ""           # Go package name, Python module path, JS module name
