# tests/test_pgvector_store.py
"""Unit tests for store/pgvector_store.py — PgVectorStore helper functions."""

import hashlib

from store.pgvector_store import _chunk_id, _vector_str


# ---------------------------------------------------------------------------
# _chunk_id
# ---------------------------------------------------------------------------

class TestChunkId:
    def test_deterministic(self):
        meta = {
            "file_path": "/a/b.py",
            "chunk_type": "function",
            "function_name": "foo",
            "line_number": 10,
        }
        assert _chunk_id(meta) == _chunk_id(meta)

    def test_different_meta_different_id(self):
        meta_a = {"file_path": "/a/b.py", "chunk_type": "function", "function_name": "foo", "line_number": 1}
        meta_b = {"file_path": "/a/b.py", "chunk_type": "function", "function_name": "bar", "line_number": 1}
        assert _chunk_id(meta_a) != _chunk_id(meta_b)

    def test_uses_class_name_when_no_function(self):
        meta = {
            "file_path": "/a/b.py",
            "chunk_type": "class",
            "class_name": "MyClass",
            "line_number": 5,
        }
        result = _chunk_id(meta)
        assert isinstance(result, str)
        assert len(result) == hashlib.md5().hexdigest().length

    def test_empty_meta(self):
        meta = {}
        result = _chunk_id(meta)
        assert isinstance(result, str)
        assert len(result) == hashlib.md5().hexdigest().length

    def test_symbol_name_fallback(self):
        meta = {
            "file_path": "/a/b.py",
            "chunk_type": "file_summary",
            "symbol_name": "b.py",
            "line_number": 0,
        }
        result = _chunk_id(meta)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _vector_str
# ---------------------------------------------------------------------------

class TestVectorStr:
    def test_basic(self):
        result = _vector_str([1.0, 2.5, 3.0])
        assert result == "[1.0,2.5,3.0]"

    def test_empty(self):
        result = _vector_str([])
        assert result == "[]"

    def test_single(self):
        result = _vector_str([42.0])
        assert result == "[42.0]"

    def test_negative(self):
        result = _vector_str([-1.5, 0.0])
        assert result == "[-1.5,0.0]"
