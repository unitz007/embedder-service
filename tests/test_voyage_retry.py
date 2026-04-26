"""Tests for VoyageEmbedder retry with exponential backoff."""

from unittest.mock import MagicMock, patch, call
import pytest

from lib.embedder import (
    VoyageEmbedder,
    VOYAGE_MAX_RETRIES,
    VOYAGE_RETRY_BASE_DELAY,
    VOYAGE_RETRY_STATUS_CODES,
)


class TestVoyageRetryConfig:
    """Verify retry configuration constants."""

    def test_max_retries_is_positive(self):
        assert VOYAGE_MAX_RETRIES >= 1

    def test_base_delay_is_positive(self):
        assert VOYAGE_RETRY_BASE_DELAY > 0

    def test_retry_status_codes_include_429_and_5xx(self):
        assert 429 in VOYAGE_RETRY_STATUS_CODES
        assert 500 in VOYAGE_RETRY_STATUS_CODES
        assert 502 in VOYAGE_RETRY_STATUS_CODES
        assert 503 in VOYAGE_RETRY_STATUS_CODES
        assert 504 in VOYAGE_RETRY_STATUS_CODES


class TestVoyageRetryBehaviour:
    """Verify _call_voyage_api retries on retriable errors and raises after exhausting retries."""

    @patch("lib.embedder.time.sleep")
    @patch("lib.embedder.requests.post")
    def test_immediate_success(self, mock_post, mock_sleep):
        """A 200 response returns embeddings without any retry delay."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]
        }
        mock_post.return_value = mock_response

        embedder = VoyageEmbedder.__new__(VoyageEmbedder)
        embedder.api_key = "test-key"
        embedder.input_type = "document"
        embedder._dim = 3

        result = embedder._call_voyage_api(["hello"], batch_offset=0)
        assert result.shape == (1, 3)
        mock_sleep.assert_not_called()

    @patch("lib.embedder.time.sleep")
    @patch("lib.embedder.requests.post")
    def test_retry_on_429(self, mock_post, mock_sleep):
        """A 429 followed by 200 succeeds after one retry."""
        responses = [
            MagicMock(status_code=429, text="rate limited", headers={}),
            MagicMock(status_code=200, json={"data": [{"index": 0, "embedding": [0.1, 0.2]}]}),
        ]
        mock_post.side_effect = responses

        embedder = VoyageEmbedder.__new__(VoyageEmbedder)
        embedder.api_key = "test-key"
        embedder.input_type = "document"
        embedder._dim = 2

        result = embedder._call_voyage_api(["hello"], batch_offset=0)
        assert result.shape == (1, 2)
        assert mock_post.call_count == 2
        mock_sleep.assert_called_once()

    @patch("lib.embedder.time.sleep")
    @patch("lib.embedder.requests.post")
    def test_retry_on_502(self, mock_post, mock_sleep):
        """A 502 followed by 200 succeeds after one retry."""
        responses = [
            MagicMock(status_code=502, text="bad gateway", headers={}),
            MagicMock(status_code=200, json={"data": [{"index": 0, "embedding": [0.1]}]}),
        ]
        mock_post.side_effect = responses

        embedder = VoyageEmbedder.__new__(VoyageEmbedder)
        embedder.api_key = "test-key"
        embedder.input_type = "document"
        embedder._dim = 1

        result = embedder._call_voyage_api(["hello"], batch_offset=0)
        assert result.shape == (1, 1)
        assert mock_post.call_count == 2

    @patch("lib.embedder.time.sleep")
    @patch("lib.embedder.requests.post")
    def test_no_retry_on_400(self, mock_post, mock_sleep):
        """A 400 is not retried — raises immediately."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "bad request"
        mock_post.return_value = mock_response

        embedder = VoyageEmbedder.__new__(VoyageEmbedder)
        embedder.api_key = "test-key"
        embedder.input_type = "document"
        embedder._dim = 3

        with pytest.raises(RuntimeError, match="Voyage AI API error 400"):
            embedder._call_voyage_api(["hello"], batch_offset=0)

        mock_sleep.assert_not_called()
        mock_post.assert_called_once()

    @patch("lib.embedder.time.sleep")
    @patch("lib.embedder.requests.post")
    def test_raises_after_max_retries(self, mock_post, mock_sleep):
        """After VOYAGE_MAX_RETRIES + 1 attempts, a RuntimeError is raised."""
        mock_post.return_value = MagicMock(
            status_code=500, text="internal server error", headers={}
        )

        embedder = VoyageEmbedder.__new__(VoyageEmbedder)
        embedder.api_key = "test-key"
        embedder.input_type = "document"
        embedder._dim = 3

        with pytest.raises(RuntimeError, match="failed after"):
            embedder._call_voyage_api(["hello"], batch_offset=42)

        total_attempts = VOYAGE_MAX_RETRIES + 1
        assert mock_post.call_count == total_attempts
        assert mock_sleep.call_count == VOYAGE_MAX_RETRIES

    @patch("lib.embedder.time.sleep")
    @patch("lib.embedder.requests.post")
    def test_retry_after_header_respected(self, mock_post, mock_sleep):
        """When the API returns Retry-After, that value is used for the delay."""
        responses = [
            MagicMock(
                status_code=429,
                text="rate limited",
                headers={"Retry-After": "5"},
            ),
            MagicMock(status_code=200, json={"data": [{"index": 0, "embedding": [0.1]}]}),
        ]
        mock_post.side_effect = responses

        embedder = VoyageEmbedder.__new__(VoyageEmbedder)
        embedder.api_key = "test-key"
        embedder.input_type = "document"
        embedder._dim = 1

        embedder._call_voyage_api(["hello"], batch_offset=0)
        mock_sleep.assert_called_once_with(5.0)

    @patch("lib.embedder.time.sleep")
    @patch("lib.embedder.requests.post")
    def test_retry_on_connection_error(self, mock_post, mock_sleep):
        """Network errors (RequestException) are retried."""
        import requests

        mock_post.side_effect = [
            requests.exceptions.ConnectionError("refused"),
            MagicMock(status_code=200, json={"data": [{"index": 0, "embedding": [0.1]}]}),
        ]

        embedder = VoyageEmbedder.__new__(VoyageEmbedder)
        embedder.api_key = "test-key"
        embedder.input_type = "document"
        embedder._dim = 1

        result = embedder._call_voyage_api(["hello"], batch_offset=0)
        assert result.shape == (1, 1)
        assert mock_post.call_count == 2
