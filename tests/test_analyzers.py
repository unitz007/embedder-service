import os
import tempfile
import pytest

from analyzers.js_analyzer import analyze_js
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
    """Ensure we get all 10 variables: 5 consts + 5 vars."""
    result = analyze_go(sample_go_file)
    assert len(result.variables) == 10
    consts = [v for v in result.variables if v.kind == "const"]
    vars_ = [v for v in result.variables if v.kind == "var"]
    assert len(consts) == 5   # APIVersion, DefaultTimeout, MaxRetries, BaseURL, AllowedMethods
    assert len(vars_) == 5    # configPath, configPath2, port, host, debug


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
# JavaScript analyzer tests
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_js_file():
    code = """\
// API version constant
const API_VERSION = "2.0";

// Default configuration
let config = {
  host: "localhost",
  port: 3000,
};

// Feature flags
var debugMode = true;

// Maximum retries allowed
const MAX_RETRIES = 3;

// This is an arrow function — should be a function, not a variable
const fetchData = async (url) => {
  const response = await fetch(url);
  return response.json();
};

// This is a function expression — should be a function, not a variable
const processData = function(data) {
  return data.map(item => item.value);
};

// Regular function declaration
function calculateTotal(items) {
  return items.reduce((sum, item) => sum + item.price, 0);
}

// Named export with variable
export const APP_NAME = "MyApp";

// Array variable
const ALLOWED_METHODS = ["GET", "POST", "PUT"];
"""
    fd, path = tempfile.mkstemp(suffix=".js")
    with os.fdopen(fd, "w") as f:
        f.write(code)
    yield path
    os.unlink(path)


def test_js_basic(sample_js_file):
    result = analyze_js(sample_js_file)
    assert isinstance(result, FileAnalysis)
    assert result.language == "javascript"


def test_js_functions(sample_js_file):
    result = analyze_js(sample_js_file)
    names = [f.name for f in result.functions]
    # Arrow function
    assert "fetchData" in names
    # Function expression
    assert "processData" in names
    # Regular function declaration
    assert "calculateTotal" in names
    assert len(result.functions) == 3


def test_js_variables(sample_js_file):
    result = analyze_js(sample_js_file)
    vars_by_name = {v.name: v for v in result.variables}

    # const API_VERSION
    assert "API_VERSION" in vars_by_name
    assert vars_by_name["API_VERSION"].kind == "const"
    assert '"2.0"' in vars_by_name["API_VERSION"].value
    assert "API version constant" in vars_by_name["API_VERSION"].docstring

    # let config (object)
    assert "config" in vars_by_name
    assert vars_by_name["config"].kind == "let"
    assert "host" in vars_by_name["config"].value

    # var debugMode
    assert "debugMode" in vars_by_name
    assert vars_by_name["debugMode"].kind == "var"
    assert vars_by_name["debugMode"].value == "true"

    # const MAX_RETRIES
    assert "MAX_RETRIES" in vars_by_name
    assert vars_by_name["MAX_RETRIES"].kind == "const"
    assert vars_by_name["MAX_RETRIES"].value == "3"
    assert "Maximum retries allowed" in vars_by_name["MAX_RETRIES"].docstring

    # exported const APP_NAME
    assert "APP_NAME" in vars_by_name
    assert vars_by_name["APP_NAME"].kind == "const"

    # const ALLOWED_METHODS (array)
    assert "ALLOWED_METHODS" in vars_by_name
    assert "GET" in vars_by_name["ALLOWED_METHODS"].value

    # Arrow functions and function expressions should NOT be in variables
    assert "fetchData" not in vars_by_name
    assert "processData" not in vars_by_name


def test_js_variable_total_count(sample_js_file):
    """6 variables: API_VERSION, config, debugMode, MAX_RETRIES, APP_NAME, ALLOWED_METHODS."""
    result = analyze_js(sample_js_file)
    assert len(result.variables) == 6
    consts = [v for v in result.variables if v.kind == "const"]
    lets = [v for v in result.variables if v.kind == "let"]
    vars_ = [v for v in result.variables if v.kind == "var"]
    assert len(consts) == 4   # API_VERSION, MAX_RETRIES, APP_NAME, ALLOWED_METHODS
    assert len(lets) == 1     # config
    assert len(vars_) == 1    # debugMode


def test_js_empty_file():
    fd, path = tempfile.mkstemp(suffix=".js")
    with os.fdopen(fd, "w") as f:
        f.write("")
    try:
        result = analyze_js(path)
        assert isinstance(result, FileAnalysis)
        assert result.functions == []
        assert result.classes == []
        assert result.variables == []
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Chunker variable tests
# ---------------------------------------------------------------------------

from lib.chunker import chunk_file_analysis


def test_chunker_variable_chunks_go(sample_go_file):
    """Go file with variables should produce variable chunks."""
    result = analyze_go(sample_go_file)
    chunks = chunk_file_analysis(result)

    var_chunks = [c for c in chunks if c["type"] == "variable"]
    assert len(var_chunks) > 0

    # Each variable chunk should have proper metadata
    api_chunk = next(c for c in var_chunks if c["metadata"].get("variable_name") == "APIVersion")
    assert api_chunk["metadata"]["symbol_type"] == "const"
    assert api_chunk["metadata"]["kind"] == "const"
    assert api_chunk["metadata"]["value"] == '"v2"'
    assert "API version" in api_chunk["metadata"]["docstring"]


def test_chunker_variable_chunks_js(sample_js_file):
    """JS file with variables should produce variable chunks."""
    result = analyze_js(sample_js_file)
    chunks = chunk_file_analysis(result)

    var_chunks = [c for c in chunks if c["type"] == "variable"]
    assert len(var_chunks) == 6

    # Check metadata fields
    for vc in var_chunks:
        assert "variable_name" in vc["metadata"]
        assert "kind" in vc["metadata"]
        assert "value" in vc["metadata"]
        assert "content" in vc["metadata"]
        assert vc["content"]  # non-empty content

    # Check specific variable
    config_chunk = next(c for c in var_chunks if c["metadata"]["variable_name"] == "config")
    assert config_chunk["metadata"]["kind"] == "let"


def test_chunker_summary_includes_variables(sample_go_file):
    """File summary chunk should list variable names."""
    result = analyze_go(sample_go_file)
    chunks = chunk_file_analysis(result)

    summary_chunk = next(c for c in chunks if c["type"] == "file_summary")
    assert "Variables:" in summary_chunk["content"]
    assert "APIVersion" in summary_chunk["content"]
    assert summary_chunk["metadata"]["variable_count"] == len(result.variables)


def test_chunker_summary_includes_variables_js(sample_js_file):
    """JS file summary should list variable names."""
    result = analyze_js(sample_js_file)
    chunks = chunk_file_analysis(result)

    summary_chunk = next(c for c in chunks if c["type"] == "file_summary")
    assert "Variables:" in summary_chunk["content"]
    assert "API_VERSION" in summary_chunk["content"]
    assert summary_chunk["metadata"]["variable_count"] == 6


def test_chunker_no_variables_no_var_section():
    """Files without variables should not have a Variables: line in summary."""
    # Use a Go empty file which has no variables
    fd, path = tempfile.mkstemp(suffix=".go")
    with os.fdopen(fd, "w") as f:
        f.write("package empty\n")
    try:
        result = analyze_go(path)
        chunks = chunk_file_analysis(result)
        summary_chunk = next(c for c in chunks if c["type"] == "file_summary")
        assert "Variables:" not in summary_chunk["content"]
        assert summary_chunk["metadata"]["variable_count"] == 0
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------

def test_python_syntax_error():
    """A file with invalid Python triggers the regex fallback and recovers symbols."""
    fd, path = tempfile.mkstemp(suffix=".py")
    with os.fdopen(fd, "w") as f:
        f.write("""\
import os
import sys
from collections import OrderedDict

def hello(name):
    '''Say hello.'''
    print(name)

class MyClass:
    '''A test class.'''

def broken(
""")  # SyntaxError: missing closing paren
    try:
        result = analyze_python(path)
        assert isinstance(result, FileAnalysis)
        assert result.language == "python"

        # The regex fallback should have recovered the definitions above the error
        assert len(result.functions) >= 1
        assert len(result.imports) >= 1
        assert len(result.classes) >= 1

        # Check specific recovered symbols
        fn_names = [f.name for f in result.functions]
        assert "hello" in fn_names
        hello_fn = next(f for f in result.functions if f.name == "hello")
        assert hello_fn.line == 5
        assert "hello" in hello_fn.name
        assert hello_fn.signature != ""

        class_names = [c.name for c in result.classes]
        assert "MyClass" in class_names
        myclass = next(c for c in result.classes if c.name == "MyClass")
        assert "test class" in myclass.docstring

        # Imports recovered
        assert "os" in result.imports
        assert "sys" in result.imports
        assert "collections" in result.imports

        # No variables in fallback (regex fallback doesn't extract variables)
        assert result.variables == []
    finally:
        os.unlink(path)


def test_python_syntax_error_incomplete_def():
    """Minimal incomplete function def still returns a non-empty FileAnalysis."""
    fd, path = tempfile.mkstemp(suffix=".py")
    with os.fdopen(fd, "w") as f:
        f.write("def broken(\n")  # incomplete function — SyntaxError
    try:
        result = analyze_python(path)
        assert isinstance(result, FileAnalysis)
        assert result.language == "python"
        # The regex fallback should detect "def broken(" even though it's invalid
        # and return it as a best-effort function entry.
        assert len(result.functions) >= 1
        assert result.functions[0].name == "broken"
    finally:
        os.unlink(path)


def test_python_syntax_error_empty_file():
    """An empty file has no syntax error and should return empty analysis."""
    fd, path = tempfile.mkstemp(suffix=".py")
    with os.fdopen(fd, "w") as f:
        f.write("")
    try:
        result = analyze_python(path)
        assert isinstance(result, FileAnalysis)
        assert result.functions == []
        assert result.classes == []
        assert result.variables == []
    finally:
        os.unlink(path)


def test_python_syntax_error_async_def():
    """Regex fallback recovers async def definitions."""
    fd, path = tempfile.mkstemp(suffix=".py")
    with os.fdopen(fd, "w") as f:
        f.write("""\
import asyncio

async def fetch_data(url: str) -> bytes:
    '''Fetch data from URL.'''
    pass

def broken(
""")  # SyntaxError on last line
    try:
        result = analyze_python(path)
        fn_names = [f.name for f in result.functions]
        assert "fetch_data" in fn_names
        assert "asyncio" in result.imports
    finally:
        os.unlink(path)


def test_python_syntax_error_import_recovery():
    """Regex fallback recovers both 'import X' and 'from X import Y' statements."""
    fd, path = tempfile.mkstemp(suffix=".py")
    with os.fdopen(fd, "w") as f:
        f.write("""\
import os
import sys, json
from pathlib import Path
from typing import List, Dict
from collections import OrderedDict

def broken(
""")  # SyntaxError
    try:
        result = analyze_python(path)
        assert "os" in result.imports
        assert "sys" in result.imports
        assert "json" in result.imports
        assert "pathlib" in result.imports
        assert "typing" in result.imports
        assert "collections" in result.imports
    finally:
        os.unlink(path)


def test_python_syntax_error_multiline_docstring():
    """Regex fallback recovers multi-line triple-quoted docstrings."""
    fd, path = tempfile.mkstemp(suffix=".py")
    with os.fdopen(fd, "w") as f:
        f.write('''\
class Service:
    """A service that does things.

    It has multiple lines in its docstring.
    """

def broken(
''')  # SyntaxError
    try:
        result = analyze_python(path)
        service_cls = next(c for c in result.classes if c.name == "Service")
        assert "A service that does things" in service_cls.docstring
        assert "multiple lines" in service_cls.docstring
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