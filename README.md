# Embedder Service

This service provides text embedding capabilities using transformer models.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running Tests

To run the test suite, use pytest:

```bash
pytest
```

To run tests with coverage:
```bash
pytest --cov=.
```

## Directory Structure

- `tests/` - Unit tests
- `src/` - Source code
- `models/` - Data models
- `utils/` - Utility functions

## Adding New Tests

Add new test files in the `tests/` directory with the naming pattern `test_*.py`.
