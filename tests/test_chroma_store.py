# tests/test_chroma_store.py
"""Unit tests for store/chroma_store.py — ChromaStore helper functions."""

import pytest

from store.chroma_store import (
    _serialize,
    _deserialize,
    _chunk_id,
)


# ---------------------------------------------------------------------------
# _serialize / _deserialize
# ---------------------------------------------------------------------------

class TestSerializeDeserialize:
    def test_round_trip_scalar_fields(self):
        meta = {
            "file_path": "/a/b.py",
            "language": "python",
            "line_number": 10,
            "chunk_type": "function",
            "function_name": "foo",
        }
        serialized = _serialize(meta)
        restored = _deserialize(serialized)
        assert restored["file_path"] == "/a/b.py"
        assert restored["language"] == "python"
        assert restored["line_number"] == 10

    def test_list_fields_json_encoded(self):
        meta = {
            "calls": ["bar", "baz"],
            "called_by": ["main"],
            "params": ["x", "y"],
        }
        serialized = _serialize(meta)
        # Lists should be JSON strings
        assert isinstance(serialized["calls"], str)
        assert isinstance(serialized["params"], str)

    def test_deserialize_restores_lists(self):
        meta = {
            "calls": '["bar", "baz"]',
            "called_by": '[]',
            "depends_on": '["/a/c.py"]',
            "file_path": "/a/b.py",
        }
        restored = _deserialize(meta)
        assert restored["calls"] == ["bar", "baz"]
        assert restored["called_by"] == []
        assert restored["depends_on"] == ["/a/c.py"]

    def test_deserialize_invalid_json_falls_back_to_empty_list(self):
        meta = {"calls": "not-valid-json", "file_path": "/a/b.py"}
        restored = _deserialize(meta)
        assert restored["calls"] == []

    def test_none_becomes_empty_string(self):
        meta = {"file_path": None}
        serialized = _serialize(meta)
        assert serialized["file_path"] == ""

    def test_bool_passes_through(self):
        meta = {"is_test": True, "is_prod": False}
        serialized = _serialize(meta)
        assert serialized["is_test"] is True
        assert serialized["is_prod"] is False

    def test_external_calls_round_trip(self):
        meta = {"external_calls": ["print", "len"]}
        serialized = _serialize(meta)
        restored = _deserialize(serialized)
        assert restored["external_calls"] == ["print", "len"]

    def test_imported_by_round_trip(self):
        meta = {"imported_by": ["/a/b.py"]}
        serialized = _serialize(meta)
        restored = _deserialize(serialized)
        assert restored["imported_by"] == ["/a/b.py"]

    def test_empty_lists_become_json_empty_array(self):
        meta = {"calls": [], "params": []}
        serialized = _serialize(meta)
        assert serialized["calls"] == "[]"
        assert serialized["params"] == "[]"


# ---------------------------------------------------------------------------
# _chunk_id
# ---------------------------------------------------------------------------

class TestChromaChunkId:
    def test_deterministic(self):
        meta = {"file_path": "/a/b.py", "chunk_type": "function", "function_name": "foo", "line_number": 10}
        assert _chunk_id(meta) == _chunk_id(meta)

    def test_matches_pgvector_format(self):
        """Both stores should produce the same ID for the same metadata."""
        from store.pgvector_store import _chunk_id as pg_chunk_id
        meta = {"file_path": "/a/b.py", "chunk_type": "function", "function_name": "foo", "line_number": 10}
        assert _chunk_id(meta) == pg_chunk_id(meta)

    def test_different_file_different_id(self):
        meta_a = {"file_path": "/a/b.py", "chunk_type": "function", "function_name": "foo", "line_number": 1}
        meta_b = {"file_path": "/a/c.py", "chunk_type": "function", "function_name": "foo", "line_number": 1}
        assert _chunk_id(meta_a) != _chunk_id(meta_b)
