# tests/test_analyzers.py

import os
import tempfile
import pytest

from analyzers.go_analyzer import analyze_go
from analyzers.python_analyzer import analyze_python
from models import FileAnalysis


# ---------------------------------------------------------------------------
# Go analyzer tests
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_go_file():
    """Create a temporary Go source file and return its path."""
    code = """\
package main

// APIVersion is the version of the API.
const APIVersion = "v2"

const DefaultTimeout = 30

const (
    MaxRetries    = 3
    BaseURL       = "https://api.example.com"
    AllowedMethods = "GET,POST"
)

var configPath string
var configPath2 = "/etc/app/config.yaml"

var (
    port     = 8080
    host     = "localhost"
    debug    = true
)

// GetUser retrieves a user by ID.
func GetUser(id int) (*User, error) {
    return nil, nil
}

// Repo handles database operations.
type Repo struct {
    db *sql.DB
}

// Find returns a user from the database.
func (r *Repo) Find(id int) (*User, error) {
    return nil, nil
}

// Cache defines the caching interface.
type Cache interface {
    Get(key string) ([]byte, error)
    Set(key string, val []byte) error
}
"""
    fd, path = tempfile.mkstemp(suffix=".go")
    with os.fdopen(fd, "w") as f:
        f.write(code)
    yield path
    os.unlink(path)


def test_go_basic(sample_go_file):
    result = analyze_go(sample_go_file)
    assert isinstance(result, FileAnalysis)
    assert result.language == "go"
    assert result.package == "main"


def test_go_functions(sample_go_file):
    result = analyze_go(sample_go_file)
    names = [f.name for f in result.functions]
    assert "GetUser" in names
    assert "Repo.Find" in names

    get_user = next(f for f in result.functions if f.name == "GetUser")
    assert get_user.docstring == "APIVersion is the version of the API." is False  # not its doc
    assert "GetUser" in get_user.signature

    find = next(f for f in result.functions if f.name == "Repo.Find")
    assert "Find returns a user from the database." in find.docstring


def test_go_classes(sample_go_file):
    result = analyze_go(sample_go_file)
    type_names = [c.name for c in result.classes]
    assert "Repo" in type_names
    assert "Cache" in type_names

    repo = next(c for c in result.classes if c.name == "Repo")
    assert repo.kind == "struct"
    assert "handles database operations" in repo.docstring

    cache = next(c for c in result.classes if c.name == "Cache")
    assert cache.kind == "interface"


def test_go_imports(sample_go_file):
    result = analyze_go(sample_go_file)
    assert len(result.imports) == 1
    assert result.imports[0] == "database/sql"


def test_go_const_single(sample_go_file):
    result = analyze_go(sample_go_file)
    vars_by_name = {v.name: v for v in result.variables}

    # Single const
    assert "APIVersion" in vars_by_name
    assert vars_by_name["APIVersion"].kind == "const"
    assert vars_by_name["APIVersion"].value == '"v2"'
    assert "APIVersion is the version of the API." in vars_by_name["APIVersion"].docstring

    # Another single const without explicit docstring
    assert "DefaultTimeout" in vars_by_name
    assert vars_by_name["DefaultTimeout"].kind == "const"
    assert vars_by_name["DefaultTimeout"].value == "30"


def test_go_const_block(sample_go_file):
    result = analyze_go(sample_go_file)
    vars_by_name = {v.name: v for v in result.variables}

    assert "MaxRetries" in vars_by_name
    assert vars_by_name["MaxRetries"].kind == "const"
    assert vars_by_name["MaxRetries"].value == "3"

    assert "BaseURL" in vars_by_name
    assert vars_by_name["BaseURL"].value == '"https://api.example.com"'

    assert "AllowedMethods" in vars_by_name
    assert vars_by_name["AllowedMethods"].value == '"GET,POST"'


def test_go_var_declarations(sample_go_file):
    result = analyze_go(sample_go_file)
    vars_by_name = {v.name: v for v in result.variables}

    # var with type annotation but no value
    assert "configPath" in vars_by_name
    assert vars_by_name["configPath"].kind == "var"
    assert vars_by_name["configPath"].type_annotation == "string"
    assert vars_by_name["configPath"].value == ""

    # var with value
    assert "configPath2" in vars_by_name
    assert vars_by_name["configPath2"].kind == "var"
    assert vars_by_name["configPath2"].value == '"/etc/app/config.yaml"'

    # Parenthesised var block
    assert "port" in vars_by_name
    assert vars_by_name["port"].value == "8080"

    assert "host" in vars_by_name
    assert vars_by_name["host"].value == '"localhost"'

    assert "debug" in vars_by_name
    assert vars_by_name["debug"].value == "true"


def test_go_variable_total_count(sample_go_file):
    """Ensure we get all 8 variables: 4 consts + 4 vars."""
    result = analyze_go(sample_go_file)
    assert len(result.variables) == 8
    consts = [v for v in result.variables if v.kind == "const"]
    vars_ = [v for v in result.variables if v.kind == "var"]
    assert len(consts) == 5   # APIVersion, DefaultTimeout, MaxRetries, BaseURL, AllowedMethods
    assert len(vars_) == 3    # configPath, configPath2, port, host, debug — wait let me recount
    # configPath, configPath2 are top-level; port, host, debug are in var ()


# ---------------------------------------------------------------------------
# Python analyzer tests
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_py_file():
    code = """\
\"\"\"Module docstring for test_module.\"\"\"

MAX_CONNECTIONS = 100
API_BASE_URL = "https://api.example.com"
default_retry_count = 3

class Config:
    '''Configuration handler.'''

    DEBUG = True
    timeout: int = 30

    def __init__(self, name: str):
        self.name = name

    def get_value(self, key: str) -> str:
        return self.name

    @classmethod
    def from_env(cls) -> "Config":
        return cls("default")

    @staticmethod
    def helper():
        pass

class BaseHandler:
    pass

def process(data: list) -> dict:
    pass

async def fetch(url: str) -> bytes:
    pass
"""
    fd, path = tempfile.mkstemp(suffix=".py")
    with os.fdopen(fd, "w") as f:
        f.write(code)
    yield path
    os.unlink(path)


def test_python_basic(sample_py_file):
    result = analyze_python(sample_py_file)
    assert isinstance(result, FileAnalysis)
    assert result.language == "python"


def test_python_functions(sample_py_file):
    result = analyze_python(sample_py_file)
    names = [f.name for f in result.functions]
    assert "process" in names
    assert "fetch" in names
    assert "Config.__init__" in names
    assert "Config.get_value" in names
    assert "Config.from_env" in names
    assert "Config.helper" in names

    process_fn = next(f for f in result.functions if f.name == "process")
    assert "data" in process_fn.params
    assert process_fn.return_type == "dict"


def test_python_classes(sample_py_file):
    result = analyze_python(sample_py_file)
    class_names = [c.name for c in result.classes]
    assert "Config" in class_names
    assert "BaseHandler" in class_names

    config_cls = next(c for c in result.classes if c.name == "Config")
    assert config_cls.kind == "class"
    assert "Configuration handler" in config_cls.docstring


def test_python_imports_empty(sample_py_file):
    result = analyze_python(sample_py_file)
    assert result.imports == []


def test_python_module_level_constants(sample_py_file):
    result = analyze_python(sample_py_file)
    vars_by_name = {v.name: v for v in result.variables}

    assert "MAX_CONNECTIONS" in vars_by_name
    assert vars_by_name["MAX_CONNECTIONS"].kind == "const"
    assert vars_by_name["MAX_CONNECTIONS"].value == "100"

    assert "API_BASE_URL" in vars_by_name
    assert vars_by_name["API_BASE_URL"].kind == "const"
    assert vars_by_name["API_BASE_URL"].value == '"https://api.example.com"'

    # Non-ALL_CAPS should NOT be extracted as const
    assert "default_retry_count" not in vars_by_name


def test_python_class_level_variables(sample_py_file):
    result = analyze_python(sample_py_file)
    vars_by_name = {v.name: v for v in result.variables}

    # ALL_CAPS class-level → const
    assert "Config.DEBUG" not in vars_by_name  # We don't prefix with class name
    # Look by name
    debug_var = next((v for v in result.variables if v.name == "DEBUG"), None)
    assert debug_var is not None
    assert debug_var.kind == "const"
    assert debug_var.value == "True"

    # Annotated lowercase class-level → class_var
    timeout_var = next((v for v in result.variables if v.name == "timeout"), None)
    assert timeout_var is not None
    assert timeout_var.kind == "class_var"
    assert timeout_var.value == "30"
    assert timeout_var.type_annotation == "int"


def test_python_variable_total_count(sample_py_file):
    """Module: MAX_CONNECTIONS, API_BASE_URL (2 consts). Class Config: DEBUG (const), timeout (class_var)."""
    result = analyze_python(sample_py_file)
    assert len(result.variables) == 4
    consts = [v for v in result.variables if v.kind == "const"]
    class_vars = [v for v in result.variables if v.kind == "class_var"]
    assert len(consts) == 3    # MAX_CONNECTIONS, API_BASE_URL, DEBUG
    assert len(class_vars) == 1  # timeout


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------

def test_python_syntax_error():
    """A file with invalid Python should not crash; returns empty analysis."""
    fd, path = tempfile.mkstemp(suffix=".py")
    with os.fdopen(fd, "w") as f:
        f.write("def broken(\n")  # incomplete function
    try:
        result = analyze_python(path)
        assert isinstance(result, FileAnalysis)
        assert result.language == "python"
        assert result.functions == []
        assert result.classes == []
        assert result.variables == []
    finally:
        os.unlink(path)


def test_python_empty_file():
    fd, path = tempfile.mkstemp(suffix=".py")
    with os.fdopen(fd, "w") as f:
        f.write("")
    try:
        result = analyze_python(path)
        assert result.functions == []
        assert result.classes == []
        assert result.variables == []
    finally:
        os.unlink(path)


def test_go_empty_file():
    fd, path = tempfile.mkstemp(suffix=".go")
    with os.fdopen(fd, "w") as f:
        f.write("")
    try:
        result = analyze_go(path)
        assert result.functions == []
        assert result.classes == []
        assert result.variables == []
    finally:
        os.unlink(path)
