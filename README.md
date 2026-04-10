# Embedder Service

<!-- TODO: add project badges once CI/CD and licensing are configured -->
<!-- [![Build Status](https://img.shields.io/github/actions/workflow/status/unitz007/embedder-service/ci.yml?branch=main)](https://github.com/unitz007/embedder-service/actions) -->
<!-- [![Go Report Card](https://goreportcard.com/badge/github.com/unitz007/embedder-service)](https://goreportcard.com/report/github.com/unitz007/embedder-service) -->
<!-- [![License](https://img.shields.io/github/license/unitz007/embedder-service)](./LICENSE) -->

A lightweight HTTP service that accepts text input and returns high-dimensional vector embeddings. Designed to plug into RAG pipelines, semantic search systems, and any application that needs to turn natural language or structured data into machine-readable vectors. The service abstracts over multiple embedding model providers (OpenAI, HuggingFace, local models, etc.) behind a simple RESTful API, making it easy to swap or compare models without changing downstream consumers.

## Features

<!-- TODO: update this list once the implementation takes shape -->

- **Simple REST API** — Send text, receive vectors. A single `POST /v1/embed` endpoint does the heavy lifting.
- **Model provider abstraction** — Swap between OpenAI, HuggingFace, or locally-hosted models by changing configuration, not code.
- **Batch support** — Embed multiple texts in a single request to reduce latency and network overhead.
- **Configurable dimensions** — Control output vector dimensions via request parameters or server-side defaults.
- **Health and readiness checks** — Built-in `/healthz` and `/readyz` endpoints for orchestration platforms.

## Quick Start

<!-- TODO: fill in with actual build/run instructions once the project is bootstrapped -->

### Prerequisites

- **Go 1.22+** (proposed default — see [Architecture](./ARCHITECTURE.md) for rationale)
- <!-- TODO: add other prerequisites (Docker, model runtime, etc.) once confirmed -->

### Build

```bash
git clone https://github.com/unitz007/embedder-service.git
cd embedder-service

# Build the binary
go build -o bin/embedder-service ./cmd/server

# Or use Docker (once a Dockerfile is added)
# docker build -t embedder-service .
```

### Run

```bash
# Start the server with default settings
./bin/embedder-service

# Or with environment variable overrides
EMBEDDER_PORT=8080 EMBEDDER_MODEL=text-embedding-3-small ./bin/embedder-service
```

The server starts listening on `http://localhost:8080` by default.

### Docker

<!-- TODO: uncomment and adjust once a Dockerfile exists -->

<!-- ```bash
docker run -d \
  -p 8080:8080 \
  -e EMBEDDER_MODEL=text-embedding-3-small \
  -e OPENAI_API_KEY=sk-... \
  unitz007/embedder-service:latest
``` -->

## API Overview

<!-- TODO: finalise the OpenAPI schema and link it here -->

All API endpoints are prefixed with `/v1`. The service accepts and returns JSON.

### `POST /v1/embed`

Generate embedding vectors for one or more text inputs.

**Request body:**

```json
{
  "texts": ["Hello, world!", "How does embedding work?"],
  "model": "text-embedding-3-small",
  "dimensions": 1536
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `texts` | `string[]` | Yes | One or more text strings to embed (max batch size TBD) |
| `model` | `string` | No | Model identifier to use. Overrides the server default. |
| `dimensions` | `integer` | No | Desired output dimensions. Model-dependent; may be ignored. |

**Response (200 OK):**

```json
{
  "model": "text-embedding-3-small",
  "dimensions": 1536,
  "embeddings": [
    [0.0023, -0.0141, 0.0387, "..."],
    [-0.0082, 0.0255, -0.0019, "..."]
  ],
  "usage": {
    "prompt_tokens": 12,
    "total_tokens": 12
  }
}
```

| Field | Type | Description |
|---|---|---|
| `model` | `string` | The model that was used to generate embeddings |
| `dimensions` | `integer` | Dimensionality of each embedding vector |
| `embeddings` | `float[][]` | Array of embedding vectors, one per input text |
| `usage` | `object` | Token usage information (provider-specific) |

**Error responses:**

| Status | Code | Description |
|---|---|---|
| 400 | `INVALID_REQUEST` | Request body is malformed or missing required fields |
| 422 | `MODEL_NOT_FOUND` | The requested model is not configured or available |
| 429 | `RATE_LIMITED` | Too many requests — retry after the `Retry-After` header |
| 500 | `INTERNAL_ERROR` | An unexpected error occurred during embedding |

### `GET /healthz`

Liveness probe. Returns `200 OK` if the server process is running.

### `GET /readyz`

Readiness probe. Returns `200 OK` if the server is ready to accept requests (model loaded, dependencies available).

### `GET /v1/models`

<!-- TODO: implement once model registry is built -->

Returns a list of available embedding models and their metadata.

```json
{
  "models": [
    {
      "id": "text-embedding-3-small",
      "provider": "openai",
      "dimensions": 1536,
      "max_input_tokens": 8191
    }
  ]
}
```

## Configuration

<!-- TODO: confirm the final configuration schema once implementation begins -->

The service is configured via environment variables. All variables have sensible defaults for local development.

| Variable | Description | Default |
|---|---|---|
| `EMBEDDER_PORT` | HTTP listen port | `8080` |
| `EMBEDDER_HOST` | HTTP listen address | `0.0.0.0` |
| `EMBEDDER_MODEL` | Default embedding model identifier | `text-embedding-3-small` |
| `EMBEDDER_DIMENSIONS` | Default output dimensions | `1536` |
| `EMBEDDER_LOG_LEVEL` | Log verbosity (`debug`, `info`, `warn`, `error`) | `info` |
| `EMBEDDER_MAX_BATCH_SIZE` | Maximum number of texts per request | `64` |
| `OPENAI_API_KEY` | API key for OpenAI provider | — |
| `HUGGINGFACE_API_KEY` | API key for HuggingFace Inference API | — |
| `OPENAI_BASE_URL` | Override OpenAI API base URL (for proxies) | — |
| `CACHE_ENABLED` | Enable response caching | `false` |
| `CACHE_TTL` | Cache time-to-live in seconds | `3600` |
| `CACHE_BACKEND` | Cache backend (`memory`, `redis`) | `memory` |
| `REDIS_URL` | Redis connection URL (when `CACHE_BACKEND=redis`) | `localhost:6379` |

### Configuration File

<!-- TODO: decide on config file format (YAML, TOML, env) and document -->

A configuration file can be used as an alternative to environment variables. The service looks for `config.yaml` in the working directory by default. Environment variables take precedence over file values.

## Development

<!-- TODO: fill in with concrete instructions once the project structure is bootstrapped -->

### Project Layout

See [ARCHITECTURE.md](./ARCHITECTURE.md) for the full directory structure proposal.

### Building

```bash
# Build the server binary
go build -o bin/embedder-service ./cmd/server

# Build with race detector
go build -race -o bin/embedder-service ./cmd/server
```

### Testing

```bash
# Run all tests
go test ./...

# Run tests with verbose output and coverage
go test -v -coverprofile=coverage.out ./...
go tool cover -html=coverage.out
```

### Linting

```bash
# Lint the codebase
golangci-lint run ./...

# Format code
gofmt -w .
```

### Running Locally

```bash
# Run directly with go run
go run ./cmd/server

# Or with hot-reload using air (once configured)
air
```

## Links

<!-- TODO: add real links as the project matures -->

- [Architecture Documentation](./ARCHITECTURE.md)
- [GitHub Repository](https://github.com/unitz007/embedder-service)
- [Issues](https://github.com/unitz007/embedder-service/issues)
<!-- - [API Reference (Swagger/OpenAPI)](./docs/api.yaml) -->
<!-- - [Contributing Guide](./CONTRIBUTING.md) -->
<!-- - [Changelog](./CHANGELOG.md) -->

## License

<!-- TODO: add a LICENSE file and update this section -->

This project is licensed under the [MIT License](./LICENSE).
