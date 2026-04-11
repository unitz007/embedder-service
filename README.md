# Code Indexer API

FastAPI service that indexes code repositories into **pgvector** for semantic search.

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt

# Set database URL (defaults to postgresql://postgres:postgres@localhost:5432/indexer)
export DATABASE_URL=postgresql://...

# Optional: Voyage AI cloud embedder for large codebases
export VOYAGE_API_KEY=sk-...

# Run the server
uvicorn main:app --reload --port 8000
```

## API

### Health

```
GET /health
```

Returns `{"status": "ok"}`.

### Start indexing

```
POST /index
```

**Request body:**

| Field             | Type    | Required | Description                                |
|-------------------|---------|----------|--------------------------------------------|
| `namespace`       | string  | ✅       | Tenant namespace                           |
| `project_id`      | string  | ✅       | Project identifier                         |
| `source`          | string  |          | Source of the repo (e.g. "github")         |
| `owner`           | string  |          | Repository owner                           |
| `repo`            | string  |          | Repository name                            |
| `ref`             | string  |          | Git ref (branch / tag / SHA)               |
| `webhook_url`     | string  |          | URL to receive completion callback         |
| `include_dotfiles`| boolean |          | Include hidden files (for dotfile repos)   |

**Response (202):** job record.

```json
{
  "job_id": "a1b2c3d4e5f6",
  "namespace": "acme",
  "project_id": "web-app",
  "status": "completed",
  "total_vectors": 42,
  "embedder": "CodeEmbedder"
}
```

### Get job status

```
GET /jobs/{job_id}
```

### Delete project embeddings

```
DELETE /embeddings/{namespace}/{project_id}
```

Deletes all embeddings, graph data, and the cloned repository.

### Embedder configuration

Allows tenants to control which embedding model is used for their project, overriding the default size-based heuristic.

#### Get current config

```
GET /embeddings/{namespace}/{project_id}/config
```

**Response:**

```json
{
  "namespace": "acme",
  "project_id": "web-app",
  "preferred_embedder": "voyage-code-3",
  "embedder": "VoyageEmbedder",
  "model": "voyage-code-3",
  "embedding_dim": 1024,
  "use_cloud": true,
  "last_indexed_at": "2025-01-15T10:30:00Z"
}
```

| Field                | Type    | Description                                              |
|----------------------|---------|----------------------------------------------------------|
| `preferred_embedder` | string  | Tenant's configured model preference (null = automatic)  |
| `embedder`           | string  | Class name actually used in the last indexing run        |
| `model`              | string  | Model identifier used                                    |
| `embedding_dim`      | integer | Dimension of the embeddings                              |
| `use_cloud`          | boolean | Whether a cloud embedder was used                        |
| `last_indexed_at`    | string  | Timestamp of the most recent indexing run                |

#### Set embedder preference

```
PUT /embeddings/{namespace}/{project_id}/config
Content-Type: application/json
```

**Request body:**

```json
{
  "preferred_embedder": "voyage-code-3"
}
```

**Supported values:**

| Value                   | Effect                                            |
|-------------------------|---------------------------------------------------|
| `"voyage-code-3"`       | Uses Voyage AI cloud embedder (requires `VOYAGE_API_KEY`) |
| Any string with "voyage" | Same as above (case-insensitive match)           |
| `"microsoft/codebert-base"` | Uses local CodeBERT model via sentence-transformers |
| `"BAAI/bge-small-en-v1.5"` | Uses any HuggingFace sentence-transformers model |
| `null`                  | Clears the preference; reverts to automatic heuristic |

**Clear preference (revert to automatic):**

```json
{
  "preferred_embedder": null
}
```

The preference takes effect on the **next** indexing run; it does not re-index existing data.

## Embedder selection

By default the service automatically selects an embedder based on the number of chunks:

| Chunks             | Embedder           | Model                | Dim  |
|--------------------|--------------------|----------------------|------|
| ≤ 99 999           | `CodeEmbedder`     | `microsoft/codebert-base` | 768 |
| \> 99 999          | `VoyageEmbedder`   | `voyage-code-3`      | 1024 |

When a `preferred_embedder` is configured for a tenant, it takes priority over this heuristic.

## Environment variables

| Variable            | Default | Description                         |
|---------------------|---------|-------------------------------------|
| `DATABASE_URL`      |         | PostgreSQL connection string        |
| `LOCAL_DATABASE_URL`|         | Fallback for non-production envs    |
| `VOYAGE_API_KEY`    |         | Voyage AI API key (large codebases) |
| `APP_ENV`           | `dev`   | `prod` appends `sslmode=require`    |
| `DB_POOL_MAX`       | `20`    | Max database connections in pool    |
| `PERSIST_ROOT`      | `/tmp/indexer` | Root directory for cloned repos |
