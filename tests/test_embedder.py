# tests/test_embedder.py
"""Unit tests for lib/embedder.py — constants, helpers, and embedder interface."""

import os
import pytest

from lib.embedder import (
    DEFAULT_MODEL,
    EMBEDDING_DIM,
    LARGE_CODEBASE_THRESHOLD,
    VOYAGE_MODEL,
    VOYAGE_EMBEDDING_DIM,
    VOYAGE_BATCH_SIZE,
    _is_codebert_family,
    _mean_pool,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_default_model(self):
        assert DEFAULT_MODEL == "microsoft/codebert-base"

    def test_embedding_dim(self):
        assert EMBEDDING_DIM == 768

    def test_voyage_model(self):
        assert VOYAGE_MODEL == "voyage-code-3"

    def test_voyage_dim(self):
        assert VOYAGE_EMBEDDING_DIM == 1024

    def test_voyage_batch_size(self):
        assert VOYAGE_BATCH_SIZE == 128


# ---------------------------------------------------------------------------
# _is_codebert_family
# ---------------------------------------------------------------------------

class TestIsCodebertFamily:
    def test_codebert(self):
        assert _is_codebert_family("microsoft/codebert-base")

    def test_graphcodebert(self):
        assert _is_codebert_family("microsoft/graphcodebert-base")

    def test_unixcoder(self):
        assert _is_codebert_family("microsoft/unixcoder-base")

    def test_codet5(self):
        assert _is_codebert_family("Salesforce/codet5-base")

    def test_plbart(self):
        assert _is_codebert_family("uclanlp/plbart-base")

    def test_not_codebert(self):
        assert not _is_codebert_family("sentence-transformers/all-MiniLM-L6-v2")

    def test_case_insensitive(self):
        assert _is_codebert_family("Microsoft/CodeBERT-Base")


# ---------------------------------------------------------------------------
# _mean_pool
# ---------------------------------------------------------------------------

class TestMeanPool:
    def test_basic_shape(self):
        import torch
        batch_size, seq_len, hidden = 2, 10, 768
        hidden_states = torch.randn(batch_size, seq_len, hidden)
        attention_mask = torch.ones(batch_size, seq_len)
        pooled = _mean_pool(hidden_states, attention_mask)
        assert pooled.shape == (batch_size, hidden)

    def test_masked_positions_ignored(self):
        import torch
        hidden_states = torch.tensor([[[1.0], [2.0], [3.0]]])  # (1, 3, 1)
        attention_mask = torch.tensor([[1, 1, 0]])  # last token masked
        pooled = _mean_pool(hidden_states, attention_mask)
        # Should be mean of [1.0, 2.0] = 1.5
        assert abs(pooled.item() - 1.5) < 1e-5

    def test_full_mask_still_works(self):
        import torch
        hidden_states = torch.tensor([[[1.0], [2.0]]])
        attention_mask = torch.tensor([[0, 0]])
        pooled = _mean_pool(hidden_states, attention_mask)
        # With clamped counts (min=1e-9), this should produce very large values but not NaN/Inf
        assert not torch.isnan(pooled).any()
        assert not torch.isinf(pooled).any()
