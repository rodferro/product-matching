"""Sentence embeddings with an on-disk cache for the catalogue."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import ARTIFACTS_DIR, EMBEDDING_MODEL, set_seeds


@lru_cache(maxsize=4)
def _load_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    set_seeds()
    return SentenceTransformer(model_name, device="cpu")


class Encoder:
    """Wraps a sentence-transformers model. e5 models get their query/passage prefixes."""

    def __init__(self, model_name: str = EMBEDDING_MODEL, cache_dir: Path | None = None):
        self.model_name = model_name
        self.cache_dir = cache_dir or ARTIFACTS_DIR / "embeddings"

    @property
    def model(self):
        return _load_model(self.model_name)

    def _prefix(self, kind: str) -> str:
        if "e5" in self.model_name.lower():
            return "query: " if kind == "query" else "passage: "
        return ""

    def encode(
        self, texts: Sequence[str], kind: str = "passage", batch_size: int = 128
    ) -> np.ndarray:
        prefix = self._prefix(kind)
        vectors = self.model.encode(
            [prefix + t for t in texts],
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32)

    def encode_catalog(self, texts: Sequence[str]) -> np.ndarray:
        """Encode listing names, reusing a cached copy when the texts are unchanged."""
        digest = hashlib.sha1(
            (self.model_name + "\n" + "\n".join(texts)).encode("utf-8")
        ).hexdigest()[:16]
        path = self.cache_dir / f"{self.model_name.replace('/', '__')}_{digest}.npy"
        if path.exists():
            return np.load(path)
        vectors = self.encode(texts, kind="passage")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, vectors)
        return vectors
