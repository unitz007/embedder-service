"""
Fixtures and configuration for tests.
"""

import pytest


@pytest.fixture
def sample_text():
    """Sample text fixture for testing."""
    return "This is a sample text for testing embeddings."


@pytest.fixture
def sample_embedding():
    """Sample embedding fixture for testing."""
    return [0.1, 0.2, 0.3, 0.4, 0.5]