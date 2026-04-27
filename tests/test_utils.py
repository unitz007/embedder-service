"""
Test cases for utility functions.
"""

import pytest


def test_example():
    """A simple test to verify the testing setup works."""
    assert 1 + 1 == 2


def test_string_length(sample_text):
    """Test using a fixture."""
    assert len(sample_text) > 0


class TestExampleClass:
    """Example test class."""
    
    def test_example_method(self):
        """Example test method."""
        assert True