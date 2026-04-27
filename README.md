# Embedder Service

A service for embedding code snippets and managing similarity searches.

## Development Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the service:
   ```bash
   ./deploy.sh
   ```

## Testing

To run the test suite:
```bash
pytest
```

To run tests with coverage:
```bash
pytest --cov=.
```

Tests are organized in the `tests/` directory and use pytest as the testing framework.