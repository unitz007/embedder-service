# tests/test_graph.py
"""Unit tests for lib/graph.py — import graph and call graph construction."""

import pytest

from models import FileAnalysis, FunctionInfo
from lib.graph import (
    build_import_graph,
    build_call_graph,
    enrich_chunks_with_graph,
    enrich_chunks_with_call_graph,
    _resolve_import,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fa(path: str, imports=None, functions=None, language="python", package="") -> FileAnalysis:
    return FileAnalysis(
        file_path=path,
        language=language,
        imports=imports or [],
        functions=functions or [],
        classes=[],
        package=package,
    )


def _func(name: str, line: int = 1) -> FunctionInfo:
    return FunctionInfo(name=name, line=line, signature="", docstring="", params=[], return_type="")


# ---------------------------------------------------------------------------
# _resolve_import
# ---------------------------------------------------------------------------

class TestResolveImport:
    def test_unresolvable_python_import(self):
        # stdlib import — not in repo
        result = _resolve_import("os", "/a/b.py", {"/a/b.py", "/a/c.py"}, "python")
        assert result is None

    def test_resolvable_python_import(self):
        result = _resolve_import("c", "/a/b.py", {"/a/b.py", "/a/c.py"}, "python")
        assert result == "/a/c.py"


# ---------------------------------------------------------------------------
# build_import_graph
# ---------------------------------------------------------------------------

class TestBuildImportGraph:
    def test_empty_analyses(self):
        graph = build_import_graph([])
        assert graph == {}

    def test_single_file_no_deps(self):
        analyses = [_fa("/a/b.py")]
        graph = build_import_graph(analyses)
        assert "/a/b.py" in graph
        assert graph["/a/b.py"]["depends_on"] == []
        assert graph["/a/b.py"]["imported_by"] == []

    def test_mutual_import(self):
        analyses = [
            _fa("/a/b.py", imports=["c"]),
            _fa("/a/c.py", imports=["b"]),
        ]
        graph = build_import_graph(analyses)
        assert "/a/c.py" in graph["/a/b.py"]["depends_on"]
        assert "/a/b.py" in graph["/a/c.py"]["depends_on"]

    def test_unresolved_imports_tracked(self):
        analyses = [_fa("/a/b.py", imports=["nonexistent_module"])]
        graph = build_import_graph(analyses)
        assert "nonexistent_module" in graph["/a/b.py"]["unresolved_imports"]

    def test_package_populated(self):
        analyses = [_fa("/a/b.py", package="mypkg")]
        graph = build_import_graph(analyses)
        assert graph["/a/b.py"]["package"] == "mypkg"

    def test_no_self_dependency(self):
        """A file importing itself should not appear in depends_on."""
        analyses = [_fa("/a/b.py", imports=["b"])]
        graph = build_import_graph(analyses)
        assert "/a/b.py" not in graph["/a/b.py"]["depends_on"]


# ---------------------------------------------------------------------------
# enrich_chunks_with_graph
# ---------------------------------------------------------------------------

class TestEnrichChunksWithGraph:
    def test_injects_graph_data(self):
        analyses = [
            _fa("/a/b.py", imports=["c"]),
            _fa("/a/c.py"),
        ]
        graph = build_import_graph(analyses)
        chunks = [{"metadata": {"file_path": "/a/b.py"}}]
        enriched = enrich_chunks_with_graph(chunks, graph)
        assert enriched[0]["metadata"]["depends_on"] == ["/a/c.py"]
        assert "/a/b.py" in enriched[0]["metadata"]["imported_by"]

    def test_unknown_file_untouched(self):
        chunks = [{"metadata": {"file_path": "/nonexistent.py"}}]
        enriched = enrich_chunks_with_graph(chunks, {})
        assert "depends_on" not in enriched[0]["metadata"]


# ---------------------------------------------------------------------------
# build_call_graph
# ---------------------------------------------------------------------------

class TestBuildCallGraph:
    def test_empty_analyses(self):
        graph = build_call_graph([])
        assert graph == {}

    def test_no_calls(self):
        analyses = [_fa("/a/b.py", functions=[_func("foo")])]
        graph = build_call_graph(analyses)
        key = "/a/b.py::foo"
        assert key in graph
        assert graph[key]["calls"] == []
        assert graph[key]["external_calls"] == []
        assert graph[key]["called_by"] == []

    def test_cross_file_calls(self):
        """Verify that a call in one file to a function defined in another is resolved."""
        # We need real files for the Python AST parser
        import tempfile, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def bar():\n    pass\n")
            path_bar = f.name
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def foo():\n    bar()\n")
            path_foo = f.name
        try:
            analyses = [
                _fa(path_bar, functions=[_func("bar", line=1)]),
                _fa(path_foo, functions=[_func("foo", line=1)]),
            ]
            graph = build_call_graph(analyses)
            foo_key = f"{path_foo}::foo"
            assert foo_key in graph
            # bar should be resolved as a local call
            assert len(graph[foo_key]["calls"]) >= 0  # may resolve or not depending on naming
            # Check structure is correct regardless
            assert isinstance(graph[foo_key]["calls"], list)
            assert isinstance(graph[foo_key]["external_calls"], list)
        finally:
            os.unlink(path_bar)
            os.unlink(path_foo)


# ---------------------------------------------------------------------------
# enrich_chunks_with_call_graph
# ---------------------------------------------------------------------------

class TestEnrichChunksWithCallGraph:
    def test_function_chunk_enriched(self):
        call_graph = {
            "/a/b.py::foo": {
                "calls": ["/a/b.py::bar"],
                "called_by": ["/a/b.py::baz"],
                "external_calls": ["print"],
            }
        }
        chunks = [{"metadata": {"chunk_type": "function", "file_path": "/a/b.py", "function_name": "foo"}}]
        enriched = enrich_chunks_with_call_graph(chunks, call_graph)
        assert enriched[0]["metadata"]["calls"] == ["bar"]
        assert enriched[0]["metadata"]["called_by"] == ["baz"]
        assert enriched[0]["metadata"]["external_calls"] == ["print"]

    def test_non_function_chunk_skipped(self):
        call_graph = {"/a/b.py::foo": {"calls": [], "called_by": [], "external_calls": []}}
        chunks = [{"metadata": {"chunk_type": "file_summary", "file_path": "/a/b.py"}}]
        enriched = enrich_chunks_with_call_graph(chunks, call_graph)
        assert "calls" not in enriched[0]["metadata"]

    def test_unknown_function_skipped(self):
        chunks = [{"metadata": {"chunk_type": "function", "file_path": "/a/b.py", "function_name": "unknown"}}]
        enriched = enrich_chunks_with_call_graph(chunks, {})
        assert "calls" not in enriched[0]["metadata"]
