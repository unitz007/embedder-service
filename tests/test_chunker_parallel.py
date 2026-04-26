"""Tests for the optimized chunk_repository_analyses (parallel chunking)."""

from models import FileAnalysis, FunctionInfo
from lib.chunker import chunk_repository_analyses, chunk_file_analysis
import tempfile
import os


def _make_analysis(file_path: str) -> FileAnalysis:
    """Create a minimal FileAnalysis for testing."""
    return FileAnalysis(
        file_path=file_path,
        language="python",
        functions=[FunctionInfo(name="foo", line=1)],
        classes=[],
        imports=["import os"],
        package="",
        variables=[],
    )


class TestChunkRepositoryAnalysesParallel:
    """Verify parallel chunking produces same results as serial."""

    def test_empty_analyses(self):
        assert chunk_repository_analyses([]) == []

    def test_single_analysis(self):
        a = _make_analysis("/tmp/test.py")
        results = chunk_repository_analyses([a])
        # Should produce at least a function chunk and a file_summary chunk
        assert len(results) >= 2

    def test_preserves_order(self):
        """Chunks from earlier files should appear before chunks from later files."""
        with tempfile.TemporaryDirectory() as tmp:
            # Create two real files so chunk_file_analysis can read them
            f1 = os.path.join(tmp, "first.py")
            f2 = os.path.join(tmp, "second.py")
            f1_content = "def alpha(): pass\n"
            f2_content = "def beta(): pass\n"
            os.makedirs(os.path.dirname(f1), exist_ok=True)
            with open(f1, "w") as fh:
                fh.write(f1_content)
            with open(f2, "w") as fh:
                fh.write(f2_content)

            a1 = _make_analysis(f1)
            a2 = _make_analysis(f2)
            a1.functions = [FunctionInfo(name="alpha", line=1)]
            a2.functions = [FunctionInfo(name="beta", line=1)]

            results = chunk_repository_analyses([a1, a2])

            # Check ordering: all chunks from f1 should come before f2 chunks
            first_files = [c["metadata"]["file_path"] for c in results]
            last_f1_idx = max(i for i, fp in enumerate(first_files) if fp == f1)
            first_f2_idx = min(i for i, fp in enumerate(first_files) if fp == f2)
            assert last_f1_idx < first_f2_idx

    def test_same_result_count_as_serial(self):
        """Parallel and serial paths should produce the same number of chunks."""
        with tempfile.TemporaryDirectory() as tmp:
            analyses = []
            for i in range(6):
                fp = os.path.join(tmp, f"f{i}.py")
                with open(fp, "w") as fh:
                    fh.write(f"def func_{i}(): pass\n")
                a = _make_analysis(fp)
                a.functions = [FunctionInfo(name=f"func_{i}", line=1)]
                analyses.append(a)

            parallel_results = chunk_repository_analyses(analyses)

            # Serial fallback
            serial_results = [chunk for a in analyses for chunk in chunk_file_analysis(a)]

            assert len(parallel_results) == len(serial_results)

    def test_handles_analysis_error_gracefully(self):
        """If one analysis fails, the rest should still be chunked."""
        with tempfile.TemporaryDirectory() as tmp:
            f1 = os.path.join(tmp, "ok.py")
            with open(f1, "w") as fh:
                fh.write("def ok(): pass\n")
            a1 = _make_analysis(f1)

            # Create a mock analysis whose file doesn't exist — will error on read
            a_bad = _make_analysis("/nonexistent/path.py")

            results = chunk_repository_analyses([a1, a_bad])
            # Should have chunks from the good file (not 0)
            assert len(results) > 0
