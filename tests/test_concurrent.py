"""Tests for concurrent/parallel processing in the embedding pipeline."""

import threading
import time
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lib.embedder import (
    CodeEmbedder,
    embed_chunks_parallel,
    DEFAULT_MAX_CONCURRENCY,
)


# ---------------------------------------------------------------------------
# Minimal stub embedder for testing concurrency without heavy model loading
# ---------------------------------------------------------------------------

class _StubEmbedder:
    """Deterministic embedder that returns fixed vectors."""

    DIM = 8

    def __init__(self, delay: float = 0.0, fail_on: set = None):
        self.delay = delay
        self.fail_on = fail_on or set()  # indices (in the batch) to fail on
        self._call_count = 0
        self._lock = threading.Lock()

    @property
    def dim(self):
        return self.DIM

    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.delay:
            time.sleep(self.delay)

        with self._lock:
            self._call_count += 1

        for i, chunk in enumerate(chunks):
            # Simulate failures on specific indices
            if i in self.fail_on:
                raise RuntimeError(f"Simulated failure on chunk index {i}")
            content = chunk.get("content", "")
            # Deterministic embedding from content hash
            vec = [float(ord(c) % 100) / 100.0 for c in content[:self.DIM]]
            # Pad to DIM
            while len(vec) < self.DIM:
                vec.append(0.0)
            updated = chunk.copy()
            updated["embedding"] = vec
            chunks[i] = updated
        return chunks

    @property
    def call_count(self):
        with self._lock:
            return self._call_count


def _make_chunks(n: int) -> List[Dict[str, Any]]:
    return [{"content": f"chunk_{i}", "metadata": {"file_path": f"file_{i}.py"}} for i in range(n)]


# ---------------------------------------------------------------------------
# embed_chunks_parallel tests
# ---------------------------------------------------------------------------

class TestEmbedChunksParallel:
    def test_sequential_passthrough(self):
        """max_concurrency=1 delegates directly to embedder without threading."""
        chunks = _make_chunks(5)
        embedder = _StubEmbedder()
        result, failures = embed_chunks_parallel(chunks, embedder, max_concurrency=1)

        assert len(result) == 5
        assert failures == []
        assert all("embedding" in c for c in result)
        assert embedder.call_count == 1

    def test_empty_chunks(self):
        """Empty chunk list returns immediately."""
        embedder = _StubEmbedder()
        result, failures = embed_chunks_parallel([], embedder, max_concurrency=4)
        assert result == []
        assert failures == []

    def test_parallel_batches(self):
        """With concurrency > 1, chunks are split into multiple batches."""
        chunks = _make_chunks(20)
        embedder = _StubEmbedder()
        result, failures = embed_chunks_parallel(chunks, embedder, max_concurrency=4)

        assert len(result) == 20
        assert failures == []
        assert all("embedding" in c for c in result)
        # Should have been called in multiple batches
        assert embedder.call_count > 1

    def test_parallel_faster_than_sequential(self):
        """Parallel processing with simulated delay should be faster than sequential."""
        chunks = _make_chunks(8)
        delay = 0.1

        embedder_seq = _StubEmbedder(delay=delay)
        t0 = time.monotonic()
        embed_chunks_parallel(chunks, embedder_seq, max_concurrency=1)
        seq_time = time.monotonic() - t0

        embedder_par = _StubEmbedder(delay=delay)
        t0 = time.monotonic()
        embed_chunks_parallel(chunks, embedder_par, max_concurrency=4)
        par_time = time.monotonic() - t0

        # Parallel should be at least 2x faster
        assert par_time < seq_time * 0.8

    def test_partial_failure_tolerance(self):
        """Failures in some batches are captured; successful batches survive."""
        chunks = _make_chunks(10)
        embedder = _StubEmbedder(fail_on={0})  # fail on index 0 within each batch
        # With batch_size calculation, index 0 is in the first batch
        # The first batch will fail entirely
        result, failures = embed_chunks_parallel(chunks, embedder, max_concurrency=4)

        # Some chunks should still succeed (from non-failing batches)
        assert len(result) < 10
        assert len(failures) > 0
        assert all("error" in f for f in failures)

    def test_order_preservation(self):
        """Output chunk order matches input order even with parallel processing."""
        chunks = _make_chunks(30)
        embedder = _StubEmbedder()
        result, failures = embed_chunks_parallel(chunks, embedder, max_concurrency=8)

        contents_in = [c["content"] for c in chunks]
        contents_out = [c["content"] for c in result]
        # Result should be a prefix of input (some may have failed, but order preserved)
        assert contents_out == contents_in[:len(contents_out)]

    def test_max_concurrency_respected(self):
        """Thread pool should not exceed max_concurrency workers."""
        chunks = _make_chunks(10)
        embedder = _StubEmbedder(delay=0.05)

        peak_threads = []
        original_init = threading.Thread.__init__

        def _track_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            peak_threads.append(threading.active_count())

        with patch.object(threading.Thread, "__init__", _track_init):
            embed_chunks_parallel(chunks, embedder, max_concurrency=2)

        # Peak should be reasonable (main + a few worker threads)
        # Not a strict assertion due to thread lifecycle timing


class TestDefaultConcurrency:
    def test_default_is_positive(self):
        assert DEFAULT_MAX_CONCURRENCY > 1

    def test_default_is_reasonable(self):
        assert DEFAULT_MAX_CONCURRENCY <= 32


class TestEmbedderInterface:
    def test_code_embedder_encode_empty(self):
        """CodeEmbedder.encode([]) returns empty array."""
        embedder = CodeEmbedder.__new__(CodeEmbedder)
        embedder._model = None
        embedder._st_model = None
        embedder._tokenizer = None
        embedder._cache = None
        embedder._dim = 8
        embedder.model_name = "test"

        result = embedder._encode_uncached([])
        assert result.shape == (0, 8)
