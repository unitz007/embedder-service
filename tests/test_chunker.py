# tests/test_chunker.py
"""Unit tests for lib/chunker.py — code chunking logic."""

import os
import tempfile

import pytest

from models import FileAnalysis, FunctionInfo, ClassInfo
from lib.chunker import (
    chunk_file_analysis,
    chunk_repository_analyses,
    extract_imports_from_lines,
    extract_function_body,
    extract_class_body,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_file(path: str, content: str) -> str:
    """Write *content* to *path* (creating intermediate dirs) and return *path*."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    return path


def _make_analysis(file_path: str, language: str = "python", **overrides) -> FileAnalysis:
    """Build a FileAnalysis with sensible defaults, overriding as needed."""
    defaults = dict(
        functions=[],
        classes=[],
        imports=[],
        package="",
    )
    defaults.update(overrides)
    return FileAnalysis(file_path=file_path, language=language, **defaults)


# ---------------------------------------------------------------------------
# extract_imports_from_lines
# ---------------------------------------------------------------------------

class TestExtractImportsFromLines:
    def test_python_import(self):
        lines = ["import os\n", "from sys import path\n", "x = 1\n"]
        result = extract_imports_from_lines(lines, "python")
        assert result == ["import os", "from sys import path"]

    def test_python_skips_non_imports(self):
        lines = ["x = 1\n", "print('hi')\n"]
        result = extract_imports_from_lines(lines, "python")
        assert result == []

    def test_go_import(self):
        lines = ["import \"fmt\"\n", "import (\n", "\t\"os\"\n", ")\n"]
        result = extract_imports_from_lines(lines, "go")
        assert len(result) == 2

    def test_javascript_import(self):
        lines = ["import React from 'react';\n", "const x = 1;\n"]
        result = extract_imports_from_lines(lines, "javascript")
        assert result == ["import React from 'react';"]

    def test_javascript_require(self):
        lines = ["const fs = require('fs');\n"]
        result = extract_imports_from_lines(lines, "javascript")
        assert result == ["const fs = require('fs');"]

    def test_unknown_language(self):
        lines = ["import foo\n"]
        result = extract_imports_from_lines(lines, "ruby")
        assert result == []


# ---------------------------------------------------------------------------
# extract_function_body
# ---------------------------------------------------------------------------

class TestExtractFunctionBody:
    def test_python_function(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def hello():\n    return 42\n\ndef other():\n    pass\n")
            path = f.name
        try:
            lines = open(path).readlines()
            body = extract_function_body(lines, start_line=1, language="python")
            assert "def hello():" in body
            assert "def other():" not in body
        finally:
            os.unlink(path)

    def test_empty_lines(self):
        body = extract_function_body([], start_line=1, language="python")
        assert "Could not extract" in body

    def test_go_function_braces(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".go", delete=False) as f:
            f.write("func main() {\n\tfmt.Println()\n}\n\nfunc other() {}\n")
            path = f.name
        try:
            lines = open(path).readlines()
            body = extract_function_body(lines, start_line=1, language="go")
            assert "func main()" in body
            assert "func other()" not in body
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# extract_class_body
# ---------------------------------------------------------------------------

class TestExtractClassBody:
    def test_python_class(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("class Foo:\n    def bar(self):\n        pass\n\nclass Baz:\n    pass\n")
            path = f.name
        try:
            lines = open(path).readlines()
            body = extract_class_body(lines, start_line=1, language="python")
            assert "class Foo:" in body
            assert "class Baz:" not in body
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# chunk_file_analysis
# ---------------------------------------------------------------------------

class TestChunkFileAnalysis:
    def test_python_with_functions_and_classes(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                "import os\n\n"
                "def foo():\n    pass\n\n"
                "class Bar:\n    pass\n"
            )
            path = f.name
        try:
            analysis = _make_analysis(
                path, language="python",
                imports=["import os"],
                functions=[FunctionInfo(name="foo", line=3)],
                classes=[ClassInfo(name="Bar", line=5)],
            )
            chunks = chunk_file_analysis(analysis)

            # Should have: imports, function, class, file_summary
            types = [c["type"] for c in chunks]
            assert "imports" in types
            assert "function" in types
            assert "class" in types
            assert "file_summary" in types
        finally:
            os.unlink(path)

    def test_empty_file_gets_fallback(self):
        """A file with no symbols should still get a full-file fallback chunk."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("")
            path = f.name
        try:
            analysis = _make_analysis(path, language="python")
            chunks = chunk_file_analysis(analysis)
            types = [c["type"] for c in chunks]
            # file_summary + full file fallback
            assert "file" in types
            assert "file_summary" in types
        finally:
            os.unlink(path)

    def test_file_summary_metadata(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def a(): pass\ndef b(): pass\n")
            path = f.name
        try:
            analysis = _make_analysis(
                path, language="python",
                functions=[
                    FunctionInfo(name="a", line=1),
                    FunctionInfo(name="b", line=2),
                ],
            )
            chunks = chunk_file_analysis(analysis)
            summary = [c for c in chunks if c["type"] == "file_summary"][0]
            assert summary["metadata"]["function_count"] == 2
            assert summary["metadata"]["chunk_type"] == "file_summary"
        finally:
            os.unlink(path)

    def test_no_imports_chunk_when_empty(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def foo(): pass\n")
            path = f.name
        try:
            analysis = _make_analysis(path, language="python", functions=[FunctionInfo(name="foo", line=1)])
            chunks = chunk_file_analysis(analysis)
            types = [c["type"] for c in chunks]
            assert "imports" not in types
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# chunk_repository_analyses
# ---------------------------------------------------------------------------

class TestChunkRepositoryAnalyses:
    def test_multiple_files(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def a(): pass\n")
            path_a = f.name
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def b(): pass\n")
            path_b = f.name
        try:
            analyses = [
                _make_analysis(path_a, functions=[FunctionInfo(name="a", line=1)]),
                _make_analysis(path_b, functions=[FunctionInfo(name="b", line=1)]),
            ]
            chunks = chunk_repository_analyses(analyses)
            assert len(chunks) >= 4  # 2 func + 2 summary at minimum

            # Verify file_path separation
            paths = {c["metadata"]["file_path"] for c in chunks}
            assert path_a in paths
            assert path_b in paths
        finally:
            os.unlink(path_a)
            os.unlink(path_b)

    def test_empty_analyses(self):
        assert chunk_repository_analyses([]) == []
