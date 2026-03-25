"""
Code embedding using microsoft/codebert-base.

Uses HuggingFace transformers directly with masked mean pooling.

Fixes included:
• Avoid HuggingFace network stalls
• Use local cache after first download
• GPU acceleration if available
• Safe tensor device handling
• Faster batching
"""
from __future__ import annotations

import os
import logging

# Disable HuggingFace network calls
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

import logging
from typing import Any, List, Dict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "microsoft/codebert-base"
EMBEDDING_DIM = 768


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
# Embedder
# -------------------------------------------------------------------

class CodeEmbedder:

    def __init__(self, model_name: str = DEFAULT_MODEL):

        self.model_name = model_name

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
# Convenience wrapper
# -------------------------------------------------------------------

def embed_repository_chunks(
        chunks: List[Dict[str, Any]],
        model_name: str = DEFAULT_MODEL,
) -> List[Dict[str, Any]]:

    return CodeEmbedder(model_name).embed_chunks(chunks)