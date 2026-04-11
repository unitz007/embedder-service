# tests/test_go_analyzer.py

import os
import pytest
from analyzers.go_analyzer import analyze_go
from models import FileAnalysis, FunctionInfo, ClassInfo, FieldInfo

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "fixtures", "go")


def _parse(fixture_name: str) -> FileAnalysis:
    return analyze_go(os.path.join(FIXTURES, fixture_name))


# ---------- basic ----------

class TestBasicAnalysis:
    def test_returns_file_analysis(self):
        result = _parse("basic.go")
        assert isinstance(result, FileAnalysis)
        assert result.language == "go"
        assert result.file_path.endswith("basic.go")

    def test_package_name(self):
        assert _parse("basic.go").package == "main"

    def test_imports(self):
        result = _parse("basic.go")
        assert "fmt" in result.imports
        assert "math" in result.imports


# ---------- functions ----------

class TestFunctions:
    def test_basic_function(self):
        result = _parse("basic.go")
        assert len(result.functions) == 3
        assert result.functions[0].name == "add"
        assert result.functions[0].line == 6
        assert result.functions[0].docstring == "add adds two integers and returns the sum"

    def test_function_params(self):
        result = _parse("basic.go")
        assert result.functions[0].params == []
        assert result.functions[1].name == "greet"
        assert result.functions[1].docstring == "greet prints a greeting message"
        assert result.functions[1].signature is not None

    def test_function_signature(self):
        result = _parse("basic.go")
        sig = result.functions[0].signature
        assert "add" in sig
        assert "int" in sig


# ---------- structs ----------

class TestStructs:
    def test_struct_count(self):
        result = _parse("basic.go")
        structs = [c for c in result.classes if c.kind == "struct"]
        assert len(structs) == 2
        names = {s.name for s in structs}
        assert names == {"User", "Config"}

    def test_struct_docstring(self):
        result = _parse("basic.go")
        user = next(c for c in result.classes if c.name == "User")
        assert user.docstring == "User represents a user in the system"

    def test_struct_line(self):
        result = _parse("basic.go")
        user = next(c for c in result.classes if c.name == "User")
        assert user.line == 19

    def test_interface_type(self):
        result = _parse("basic.go")
        iface = next((c for c in result.classes if c.kind == "interface"), None)
        assert iface is not None
        assert iface.name == "Reader"
        assert iface.line == 29


# ---------- struct fields ----------

class TestStructFields:
    def test_user_fields_count(self):
        result = _parse("basic.go")
        user = next(c for c in result.classes if c.name == "User")
        assert len(user.fields) == 4

    def test_simple_field_name_and_type(self):
        result = _parse("basic.go")
        user = next(c for c in result.classes if c.name == "User")
        assert user.fields[0].name == "Name"
        assert user.fields[0].type_str == "string"

    def test_pointer_field_type(self):
        result = _parse("basic.go")
        user = next(c for c in result.classes if c.name == "User")
        assert user.fields[2].name == "Profile"
        assert user.fields[2].type_str == "*Profile"

    def test_embedded_field(self):
        result = _parse("basic.go")
        user = next(c for c in result.classes if c.name == "User")
        assert user.fields[3].name == ""
        assert user.fields[3].type_str == "time.Time"

    def test_field_tags(self):
        result = _parse("basic.go")
        user = next(c for c in result.classes if c.name == "User")
        assert user.fields[0].tag == '`json:"name"`'
        assert user.fields[1].tag == '`json:"age,omitempty" db:"age"`'
        assert user.fields[2].tag == '`json:"profile,omitempty"`'

    def test_no_tag_field(self):
        result = _parse("basic.go")
        user = next(c for c in result.classes if c.name == "User")
        assert user.fields[3].tag == ""

    def test_config_struct_fields(self):
        result = _parse("basic.go")
        config = next(c for c in result.classes if c.name == "Config")
        assert len(config.fields) == 2
        assert config.fields[0].name == "Debug"
        assert config.fields[0].type_str == "bool"
        assert config.fields[0].tag == '`json:"debug" yaml:"debug"`'
        assert config.fields[1].name == "Port"
        assert config.fields[1].type_str == "int"
        assert config.fields[1].tag == '`json:"port"`'


# ---------- edge cases ----------

class TestEdgeCases:
    def test_empty_file(self):
        result = _parse("empty.go")
        assert result.package == "empty"
        assert result.functions == []
        assert result.classes == []
        assert result.imports == []

    def test_methods(self):
        result = _parse("methods.go")
        method_names = [f.name for f in result.functions]
        assert "Repo.Find" in method_names
        assert "Repo.Save" in method_names
        assert "Repo.Delete" in method_names
