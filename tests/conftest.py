"""
Shared pytest fixtures for the test suite.

This conftest provides common test helpers and sample file fixtures
used across test_analyzers, test_file_hashes, test_file_scanner,
and test_go_analyzer.
"""

import os

import pytest


# ---------------------------------------------------------------------------
# sys.path configuration
# ---------------------------------------------------------------------------

# Ensure the project root is on sys.path so imports work regardless of
# the working directory when running `pytest`.
@pytest.fixture(scope="session", autouse=True)
def _add_project_root_to_syspath():
    """Add the project root directory to sys.path for the test session."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in os.environ.get("PYTHONPATH", ""):
        os.environ["PYTHONPATH"] = (
            project_root + os.pathsep + os.environ.get("PYTHONPATH", "")
        )
    import sys
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


# ---------------------------------------------------------------------------
# Sample file fixtures (shared across test modules)
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_python_file(tmp_path):
    """Create a temporary Python source file with common constructs."""
    code = '''\
"""Module docstring for test_module."""

MAX_CONNECTIONS = 100
API_BASE_URL = "https://api.example.com"
default_retry_count = 3


class Config:
    """Configuration handler."""

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
'''
    path = tmp_path / "sample.py"
    path.write_text(code, encoding="utf-8")
    return str(path)


@pytest.fixture
def sample_javascript_file(tmp_path):
    """Create a temporary JavaScript source file with common constructs."""
    code = '''\
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

// Arrow function
const fetchData = async (url) => {
  const response = await fetch(url);
  return response.json();
};

// Function expression
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
'''
    path = tmp_path / "sample.js"
    path.write_text(code, encoding="utf-8")
    return str(path)


@pytest.fixture
def sample_go_file(tmp_path):
    """Create a temporary Go source file with common constructs."""
    code = '''\
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
'''
    path = tmp_path / "sample.go"
    path.write_text(code, encoding="utf-8")
    return str(path)
