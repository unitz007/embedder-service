"""Tests for PgVectorStore optimizations — connection context manager,
dimension validation, and search pagination."""

from unittest.mock import MagicMock, patch, PropertyMock
import pytest

from store.pgvector_store import PgVectorStore


class TestPgVectorStoreDimensionValidation:
    """Verify that add_vectors raises ValueError on dimension mismatch."""

    def test_correct_dimension_passes(self):
        store = PgVectorStore("ns", "proj")
        store._dim = 3  # bypass the DB query
        # Should not raise
        store._validate_dimension([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    def test_wrong_dimension_raises(self):
        store = PgVectorStore("ns", "proj")
        store._dim = 3
        with pytest.raises(ValueError, match="dimension mismatch"):
            store._validate_dimension([[1.0, 2.0]])

    def test_empty_vectors_passes(self):
        store = PgVectorStore("ns", "proj")
        store._dim = 768
        store._validate_dimension([])

    def test_dimension_cache(self):
        """After first _get_dimension call, the value is cached."""
        store = PgVectorStore("ns", "proj")
        store._dim = 42
        assert store._get_dimension() == 42


class TestPgVectorStoreConnectionManager:
    """Verify the _connection context manager guarantees putconn."""

    @patch("store.pgvector_store.db")
    def test_connection_released_on_success(self, mock_db):
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_pool.getconn.return_value = mock_conn
        mock_db.get_pool.return_value = mock_pool

        store = PgVectorStore("ns", "proj")
        with store._connection() as conn:
            assert conn is mock_conn

        mock_pool.putconn.assert_called_once_with(mock_conn)

    @patch("store.pgvector_store.db")
    def test_connection_released_on_exception(self, mock_db):
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_pool.getconn.return_value = mock_conn
        mock_db.get_pool.return_value = mock_pool

        store = PgVectorStore("ns", "proj")
        with pytest.raises(RuntimeError):
            with store._connection() as conn:
                raise RuntimeError("boom")

        mock_pool.putconn.assert_called_once_with(mock_conn)


class TestPgVectorStoreSearchPagination:
    """Verify search passes offset and limit to SQL."""

    @patch("store.pgvector_store.db")
    def test_search_default_offset_zero(self, mock_db):
        """Default offset is 0."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pool.getconn.return_value = mock_conn
        mock_db.get_pool.return_value = mock_pool
        mock_cur.fetchall.return_value = []

        store = PgVectorStore("ns", "proj")
        store.search([0.1] * 768, k=5)

        args = mock_cur.execute.call_args[0]
        # Last two positional args should be limit and offset
        assert args[-1] == 0  # offset
        assert args[-2] == 5  # limit

    @patch("store.pgvector_store.db")
    def test_search_custom_offset(self, mock_db):
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pool.getconn.return_value = mock_conn
        mock_db.get_pool.return_value = mock_pool
        mock_cur.fetchall.return_value = []

        store = PgVectorStore("ns", "proj")
        store.search([0.1] * 768, k=5, offset=10)

        args = mock_cur.execute.call_args[0]
        assert args[-1] == 10  # offset

    @patch("store.pgvector_store.db")
    def test_search_custom_limit(self, mock_db):
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pool.getconn.return_value = mock_conn
        mock_db.get_pool.return_value = mock_pool
        mock_cur.fetchall.return_value = []

        store = PgVectorStore("ns", "proj")
        store.search([0.1] * 768, k=5, offset=0, limit=3)

        args = mock_cur.execute.call_args[0]
        assert args[-2] == 3  # limit
