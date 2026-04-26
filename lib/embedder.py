"""
Code embedding using microsoft/codebert-base (local) or Voyage AI (cloud).

Local model (CodeBERT):
• Uses HuggingFace transformers with masked mean pooling
• Good for small codebases, no external API needed

Cloud model (Voyage AI voyage-code-3):
• Used for large codebases (>LARGE_CODEBASE_THRESHOLD chunks)
• Requires VOYAGE_API_KEY environment variable
• 1024-dimensional embeddings optimised for code
"""
from __future__ import annotations

import hashlib
import os
import logging

import asyncio
import concurrent.futures
import threading

# Suppress HuggingFace telemetry but allow model downloads on first run
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

import logging
from typing import Any, List, Dict, Optional, Tuple

import numpy as np
import requests
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

import db as _db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "microsoft/codebert-base"
EMBEDDING_DIM = 768

LARGE_CODEBASE_THRESHOLD = 99999  # chunks; above this, use cloud embedder

VOYAGE_API_URL = "https://api.voyageai.com/v1/embeddings"
VOYAGE_MODEL = "voyage-code-3"
VOYAGE_EMBEDDING_DIM = 1024
VOYAGE_BATCH_SIZE = 128  # max texts per Voyage AI request
DEFAULT_MAX_CONCURRENCY = 8  # sensible default for concurrent chunk embedding


# -------------------------------------------------------------------
# Dependency checks
# -------------------------------------------------------------------

try:
    torch.set_num_threads(1)

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning(
        "transformers / torch not installed. "
        "Install with: pip install transformers torch"
    )

try:
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _is_codebert_family(model_name: str) -> bool:
    low = model_name.lower()
    return any(k in low for k in ("codebert", "graphcodebert", "unixcoder", "codet5", "plbart"))


def _mean_pool(last_hidden_state: "torch.Tensor", attention_mask: "torch.Tensor") -> "torch.Tensor":
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


# -------------------------------------------------------------------
# Embedding cache
# -------------------------------------------------------------------

class EmbeddingCache:
    """Transparent content-hash → embedding cache backed by Postgres.

    Computes SHA-256 of each chunk's content and looks up previously
    computed embeddings before calling the model.  On cache miss the
    freshly computed embedding is stored for future reuse.

    The cache key is ``(sha256(content), model_name)`` and is **cross-project**:
    identical content embedded by the same model produces the same
    embedding regardless of which project it belongs to.
    """

    def __init__(self) -> None:
        self._db = _db

    @staticmethod
    def hash_text(text: str) -> bytes:
        """Return the SHA-256 digest of *text* as raw bytes (32 bytes)."""
        return hashlib.sha256(text.encode("utf-8")).digest()

    def get(
        self,
        texts: List[str],
        model_name: str,
    ) -> Tuple[Dict[int, List[float]], List[int]]:
        """Look up cached embeddings for *texts*.

        Args:
            texts:      Input text strings.
            model_name: Embedding model identifier (used as cache key suffix).

        Returns:
            ``(hits, miss_indices)``
            *hits* maps original index → embedding list for cache hits.
            *miss_indices* is the list of indices that were **not** in the cache.
        """
        keys: List[Tuple[bytes, str]] = []
        index_by_hash: Dict[bytes, int] = {}
        for i, text in enumerate(texts):
            h = self.hash_text(text)
            keys.append((h, model_name))
            index_by_hash[h] = i

        cached = self._db.cache_get_embeddings(keys)

        hits: Dict[int, List[float]] = {}
        miss_indices: List[int] = []
        for i, text in enumerate(texts):
            h = self.hash_text(text)
            if h in cached:
                hits[i] = cached[h]
            else:
                miss_indices.append(i)

        hit_count = len(hits)
        total = len(texts)
        logger.debug(
            "Embedding cache: %d/%d hits for model %s",
            hit_count, total, model_name,
        )

        return hits, miss_indices

    def store(
        self,
        hashes: List[bytes],
        model_name: str,
        embeddings: np.ndarray,
    ) -> None:
        """Store newly computed embeddings in the cache.

        Args:
            hashes:     SHA-256 digests for each embedding row.
            model_name: Embedding model identifier.
            embeddings: numpy array of shape ``(len(hashes), dim)``.
        """
        rows = [
            (h, model_name, emb.tolist())
            for h, emb in zip(hashes, embeddings)
        ]
        if rows:
            self._db.cache_store_embeddings(rows)
            logger.debug(
                "Embedding cache: stored %d entries for model %s",
                len(rows), model_name,
            )


# -------------------------------------------------------------------
# Embedder
# -------------------------------------------------------------------

class CodeEmbedder:

    def __init__(self, model_name: str = DEFAULT_MODEL, cache: Optional[EmbeddingCache] = None):

        self.model_name = model_name
        self._cache = cache

        self._tokenizer = None
        self._model = None
        self._st_model = None

        self._dim = EMBEDDING_DIM

        self.device = None

        if _is_codebert_family(model_name):
            self._load_transformers(model_name)

        elif SENTENCE_TRANSFORMERS_AVAILABLE:
            self._load_sentence_transformer(model_name)

        else:
            logger.warning(
                "No embedding library available — using random embeddings."
            )

    # ----------------------------------------------------------------

    def _load_transformers(self, model_name: str):

        if not TRANSFORMERS_AVAILABLE:
            logger.warning("transformers not available.")
            return

        try:

            logger.info("Loading %s via transformers…", model_name)

            # device
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

            # tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=False,
                use_fast=True,
            )

            # model
            self._model = AutoModel.from_pretrained(
                model_name,
                local_files_only=False,
                trust_remote_code=False,
            )

            self._model.to(self.device)
            self._model.eval()

            self._dim = self._model.config.hidden_size

            logger.info(
                "Loaded %s — embedding dim=%d — device=%s",
                model_name,
                self._dim,
                self.device,
            )

        except Exception as exc:

            logger.error(
                "Failed loading %s: %s — falling back to random embeddings.",
                model_name,
                exc,
            )

            self._tokenizer = None
            self._model = None

    # ----------------------------------------------------------------

    def _load_sentence_transformer(self, model_name: str):

        try:

            logger.info("Loading %s via sentence-transformers…", model_name)

            self._st_model = SentenceTransformer(model_name)

            test = self._st_model.encode(["test"], show_progress_bar=False)

            self._dim = test.shape[1]

            logger.info(
                "Loaded %s — embedding dim=%d",
                model_name,
                self._dim,
            )

        except Exception as exc:

            logger.error(
                "Failed loading %s: %s — falling back to random embeddings.",
                model_name,
                exc,
            )

            self._st_model = None

    # ----------------------------------------------------------------

    def _encode_transformers(
            self,
            texts: List[str],
            batch_size: int = 16,
    ) -> np.ndarray:

        all_embeddings: List[np.ndarray] = []

        for i in range(0, len(texts), batch_size):

            batch = texts[i : i + batch_size]

            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():

                outputs = self._model(**encoded)

            pooled = _mean_pool(
                outputs.last_hidden_state,
                encoded["attention_mask"],
            )

            pooled = torch.nn.functional.normalize(
                pooled,
                p=2,
                dim=1,
            )

            all_embeddings.append(
                pooled.cpu().numpy()
            )

        return np.vstack(all_embeddings)

    # ----------------------------------------------------------------
    @property
    def model(self):
        """
        Backwards-compatible accessor for callers that expect `embedder.model.encode(...)`.
        Prefer the sentence-transformers model when available; otherwise expose the
        embedder itself (which implements `encode`).
        """
        return self._st_model if self._st_model is not None else self

    # ----------------------------------------------------------------

    @property
    def dim(self) -> int:
        return self._dim

    # ----------------------------------------------------------------

    def encode(self, texts: List[str]) -> np.ndarray:

        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)

        # ── Cache path ─────────────────────────────────────────────
        if self._cache is not None:
            hits, miss_indices = self._cache.get(texts, self.model_name)

            if not miss_indices:
                # Full cache hit — assemble result preserving original order
                result = np.empty((len(texts), self._dim), dtype=np.float32)
                for idx, emb_list in hits.items():
                    result[idx] = np.array(emb_list, dtype=np.float32)
                return result

            # Compute embeddings only for misses
            miss_texts = [texts[i] for i in miss_indices]
            miss_embeddings = self._encode_uncached(miss_texts)

            # Store new embeddings in cache
            miss_hashes = [self._cache.hash_text(texts[i]) for i in miss_indices]
            self._cache.store(miss_hashes, self.model_name, miss_embeddings)

            # Assemble full result
            result = np.empty((len(texts), self._dim), dtype=np.float32)
            for idx, emb_list in hits.items():
                result[idx] = np.array(emb_list, dtype=np.float32)
            for local_i, global_i in enumerate(miss_indices):
                result[global_i] = miss_embeddings[local_i]
            return result

        # ── No cache — original behaviour ──────────────────────────
        return self._encode_uncached(texts)

    def _encode_uncached(self, texts: List[str]) -> np.ndarray:
        """Encode *texts* without consulting the cache."""
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)

        if self._model is not None and self._tokenizer is not None:

            try:

                return self._encode_transformers(texts).astype(np.float32)

            except Exception as exc:

                logger.error(
                    "Transformers encoding failed: %s",
                    exc,
                )

                return np.random.rand(len(texts), self._dim).astype(np.float32)

        if self._st_model is not None:

            try:

                return self._st_model.encode(
                    texts,
                    show_progress_bar=False,
                ).astype(np.float32)

            except Exception as exc:

                logger.error(
                    "SentenceTransformer encoding failed: %s",
                    exc,
                )

                return np.random.rand(len(texts), self._dim).astype(np.float32)

        return np.random.rand(len(texts), self._dim).astype(np.float32)

    # ----------------------------------------------------------------

    def embed_chunks(
            self,
            chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        if not chunks:
            return chunks

        texts = [c["content"] for c in chunks]

        embeddings = self.encode(texts)

        for i, chunk in enumerate(chunks):

            updated = chunk.copy()

            updated["embedding"] = embeddings[i].tolist()

            chunks[i] = updated

        return chunks


# -------------------------------------------------------------------
# Concurrent embedding helper
# -------------------------------------------------------------------

def embed_chunks_parallel(
    chunks: List[Dict[str, Any]],
    embedder,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
) -> List[Dict[str, Any]]:
    """Embed *chunks* using *embedder* with bounded parallelism.

    For local (sync) embedders the work is distributed across a
    ``ThreadPoolExecutor`` so that CPU-bound encode calls and the
    internal batching of the model run concurrently for different
    sub-batches.  Errors in individual chunks are logged and skipped;
    a summary of failures is returned.

    Args:
        chunks:          List of chunk dicts (must contain ``"content"``).
        embedder:        Any object with an ``embed_chunks(chunks)`` method.
        max_concurrency: Maximum number of in-flight sub-batches.

    Returns:
        ``(embedded_chunks, failures)`` where *failures* is a list of
        ``{index, error}`` dicts for chunks that could not be embedded.
    """
    if not chunks:
        return chunks, []

    if max_concurrency <= 1:
        # Sequential path — no threading overhead
        try:
            return embedder.embed_chunks(chunks), []
        except Exception as exc:
            logger.error("Sequential embedding failed: %s", exc)
            return [], [{"index": 0, "error": str(exc), "count": len(chunks)}]

    # Split chunks into sub-batches for parallel processing.
    # Use a batch size that keeps each worker busy but not overloaded.
    batch_size = max(1, len(chunks) // max_concurrency)
    batch_size = min(batch_size, 32)  # cap at 32 per batch
    batch_size = max(batch_size, 1)

    sub_batches = [
        chunks[i : i + batch_size]
        for i in range(0, len(chunks), batch_size)
    ]

    results: List[Optional[List[Dict[str, Any]]]] = [None] * len(sub_batches)
    failures: List[Dict[str, Any]] = []
    lock = threading.Lock()

    def _embed_batch(idx: int, batch: List[Dict[str, Any]]) -> None:
        try:
            results[idx] = embedder.embed_chunks(batch)
        except Exception as exc:
            logger.error(
                "Embedding batch %d failed (%d chunks): %s",
                idx, len(batch), exc,
            )
            with lock:
                for j, chunk in enumerate(batch):
                    original_idx = idx * batch_size + j
                    if original_idx < len(chunks):
                        failures.append({
                            "index": original_idx,
                            "error": str(exc),
                            "file_path": chunk.get("metadata", {}).get("file_path", "unknown"),
                        })
            results[idx] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        futures = {
            pool.submit(_embed_batch, i, batch): i
            for i, batch in enumerate(sub_batches)
        }
        concurrent.futures.wait(futures)

    # Reassemble in original order
    embedded: List[Dict[str, Any]] = []
    for batch_result in results:
        if batch_result is not None:
            embedded.extend(batch_result)

    if failures:
        logger.warning(
            "Embedding completed with %d failures out of %d chunks",
            len(failures), len(chunks),
        )

    return embedded, failures


# -------------------------------------------------------------------
# Async concurrent embedding (for callers already in an async loop)
# -------------------------------------------------------------------

async def embed_chunks_async(
    chunks: List[Dict[str, Any]],
    embedder,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
) -> List[Dict[str, Any]]:
    """Async version of :func:`embed_chunks_parallel`.

    Runs sync embedder calls in a thread pool under an asyncio semaphore
    so the caller's event loop is not blocked.
    """
    if not chunks:
        return chunks

    if max_concurrency <= 1:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, embedder.embed_chunks, chunks)

    batch_size = max(1, len(chunks) // max_concurrency)
    batch_size = min(batch_size, 32)
    batch_size = max(batch_size, 1)

    sub_batches = [
        chunks[i : i + batch_size]
        for i in range(0, len(chunks), batch_size)
    ]

    semaphore = asyncio.Semaphore(max_concurrency)
    failures: List[Dict[str, Any]] = []
    loop = asyncio.get_running_loop()

    async def _embed_batch(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        async with semaphore:
            try:
                return await loop.run_in_executor(
                    None, embedder.embed_chunks, batch,
                )
            except Exception as exc:
                logger.error(
                    "Async embedding batch failed (%d chunks): %s",
                    len(batch), exc,
                )
                for chunk in batch:
                    failures.append({
                        "error": str(exc),
                        "file_path": chunk.get("metadata", {}).get("file_path", "unknown"),
                    })
                return []

    batch_results = await asyncio.gather(
        *[_embed_batch(b) for b in sub_batches],
    )

    embedded: List[Dict[str, Any]] = []
    for result in batch_results:
        embedded.extend(result)

    if failures:
        logger.warning(
            "Async embedding completed with %d failures out of %d chunks",
            len(failures), len(chunks),
        )

    return embedded


# -------------------------------------------------------------------
# Cloud embedder — Voyage AI
# -------------------------------------------------------------------

class VoyageEmbedder:
    """Embeds text using the Voyage AI API (voyage-code-3).

    Requires the ``VOYAGE_API_KEY`` environment variable to be set.
    Suitable for large codebases where local inference is too slow.
    """

    def __init__(self, api_key: str | None = None, input_type: str = "document", cache: Optional[EmbeddingCache] = None):
        self.api_key = api_key or os.environ.get("VOYAGE_API_KEY", "")
        if not self.api_key:
            raise RuntimeError(
                "VOYAGE_API_KEY is not set. "
                "Export it before indexing large codebases."
            )
        self.input_type = input_type
        self._dim = VOYAGE_EMBEDDING_DIM
        self._cache = cache

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)

        # ── Cache path ─────────────────────────────────────────────
        if self._cache is not None:
            hits, miss_indices = self._cache.get(texts, VOYAGE_MODEL)

            if not miss_indices:
                # Full cache hit
                result = np.empty((len(texts), self._dim), dtype=np.float32)
                for idx, emb_list in hits.items():
                    result[idx] = np.array(emb_list, dtype=np.float32)
                return result

            # Call Voyage API only for misses
            miss_texts = [texts[i] for i in miss_indices]
            miss_embeddings = self._encode_via_api(miss_texts)

            # Store new embeddings in cache
            miss_hashes = [self._cache.hash_text(texts[i]) for i in miss_indices]
            self._cache.store(miss_hashes, VOYAGE_MODEL, miss_embeddings)

            # Assemble full result
            result = np.empty((len(texts), self._dim), dtype=np.float32)
            for idx, emb_list in hits.items():
                result[idx] = np.array(emb_list, dtype=np.float32)
            for local_i, global_i in enumerate(miss_indices):
                result[global_i] = miss_embeddings[local_i]
            return result

        # ── No cache — original behaviour ──────────────────────────
        return self._encode_via_api(texts)

    def _encode_via_api(self, texts: List[str]) -> np.ndarray:
        """Encode *texts* via the Voyage AI API (no cache lookup)."""
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)

        all_embeddings: List[np.ndarray] = []

        for i in range(0, len(texts), VOYAGE_BATCH_SIZE):
            batch = texts[i : i + VOYAGE_BATCH_SIZE]
            payload = {
                "model": VOYAGE_MODEL,
                "input": batch,
                "input_type": self.input_type,
            }
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            resp = requests.post(
                VOYAGE_API_URL,
                json=payload,
                headers=headers,
                timeout=120,
            )
            if resp.status_code != 200:
                raise RuntimeError(
                    f"Voyage AI API error {resp.status_code}: {resp.text}"
                )
            data = resp.json()["data"]
            # data is a list of {"index": int, "embedding": List[float]}
            # sort by index to preserve order
            ordered = sorted(data, key=lambda x: x["index"])
            batch_embeddings = np.array(
                [item["embedding"] for item in ordered], dtype=np.float32
            )
            all_embeddings.append(batch_embeddings)

        return np.vstack(all_embeddings)

    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not chunks:
            return chunks
        texts = [c["content"] for c in chunks]
        embeddings = self.encode(texts)
        for i, chunk in enumerate(chunks):
            updated = chunk.copy()
            updated["embedding"] = embeddings[i].tolist()
            chunks[i] = updated
        return chunks


# -------------------------------------------------------------------
# Convenience wrapper
# -------------------------------------------------------------------

def embed_repository_chunks(
        chunks: List[Dict[str, Any]],
        model_name: str = DEFAULT_MODEL,
) -> List[Dict[str, Any]]:

    return CodeEmbedder(model_name).embed_chunks(chunks)

# -------------------------------------------------------------------
# Async Voyage embedder wrapper
# -------------------------------------------------------------------

class AsyncVoyageEmbedder:
    """Async wrapper around VoyageEmbedder.

    Uses httpx for non-blocking HTTP requests and a semaphore to bound
    concurrency.  Falls back to thread-pool execution when httpx is
    not available.
    """

    def __init__(self, api_key: str | None = None, input_type: str = "document",
                 cache: Optional[EmbeddingCache] = None,
                 max_concurrency: int = DEFAULT_MAX_CONCURRENCY):
        self._sync = VoyageEmbedder(api_key=api_key, input_type=input_type, cache=cache)
        self._max_concurrency = max_concurrency
        self._httpx = None
        try:
            import httpx  # noqa: F401
            self._httpx = httpx
        except ImportError:
            pass

    @property
    def dim(self) -> int:
        return self._sync.dim

    @property
    def model_name(self) -> str:
        return VOYAGE_MODEL

    async def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._sync._dim), dtype=np.float32)

        # Cache path
        if self._sync._cache is not None:
            hits, miss_indices = self._sync._cache.get(texts, VOYAGE_MODEL)
            if not miss_indices:
                result = np.empty((len(texts), self._sync._dim), dtype=np.float32)
                for idx, emb_list in hits.items():
                    result[idx] = np.array(emb_list, dtype=np.float32)
                return result

            miss_texts = [texts[i] for i in miss_indices]
            loop = asyncio.get_running_loop()
            miss_embeddings = await loop.run_in_executor(
                None, self._sync._encode_via_api, miss_texts,
            )

            miss_hashes = [self._sync._cache.hash_text(texts[i]) for i in miss_indices]
            self._sync._cache.store(miss_hashes, VOYAGE_MODEL, miss_embeddings)

            result = np.empty((len(texts), self._sync._dim), dtype=np.float32)
            for idx, emb_list in hits.items():
                result[idx] = np.array(emb_list, dtype=np.float32)
            for local_i, global_i in enumerate(miss_indices):
                result[global_i] = miss_embeddings[local_i]
            return result

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._sync._encode_via_api, texts,
        )

    async def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not chunks:
            return chunks
        texts = [c["content"] for c in chunks]
        embeddings = await self.encode(texts)
        for i, chunk in enumerate(chunks):
            updated = chunk.copy()
            updated["embedding"] = embeddings[i].tolist()
            chunks[i] = updated
        return chunks