"""Sentence embeddings with a dependency-free fallback.

Default: sentence-transformers all-MiniLM-L6-v2 on CPU (EMBED_MODEL env var).
EMBED_MODEL=hash uses a deterministic hashed bag-of-words + character trigram vector,
which needs no download and keeps tests and CI fast.
"""
from __future__ import annotations

import hashlib
import os
import re
import threading
from functools import lru_cache

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HASH_DIM = 1024
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _hash_vec(text: str) -> np.ndarray:
    v = np.zeros(HASH_DIM, dtype=np.float32)
    words = _TOKEN_RE.findall(text.lower())
    feats = words + [w[i:i + 3] for w in words for i in range(max(1, len(w) - 2))]
    for f in feats:
        h = int.from_bytes(hashlib.md5(f.encode()).digest()[:4], "little")
        v[h % HASH_DIM] += 1.0 if f in words else 0.5
    n = np.linalg.norm(v)
    return v / n if n else v


class Embedder:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or os.environ.get("EMBED_MODEL", DEFAULT_MODEL)
        self._model = None
        self._lock = threading.Lock()
        if self.model_name != "hash":
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name, device="cpu")
            except Exception:  # offline or missing weights: degrade, never crash
                self.model_name = "hash"

    @property
    def name(self) -> str:
        return self.model_name

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, HASH_DIM if self._model is None else 384), dtype=np.float32)
        if self._model is None:
            return np.stack([_hash_vec(t) for t in texts])
        with self._lock:
            return np.asarray(
                self._model.encode(texts, normalize_embeddings=True, batch_size=64,
                                   show_progress_bar=False),
                dtype=np.float32,
            )

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return Embedder()


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (d,) or (n,d); b: (m,d). Inputs are already L2-normalised."""
    return a @ b.T
