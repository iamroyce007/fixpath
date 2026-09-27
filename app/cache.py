"""Three-tier semantic cache, all in memory.

tier 1  exact: sha256(normalised query + siis hash)
tier 2  fingerprint: (component|symptom, siis hash)
tier 3  embedding: cosine(normalised query) >= threshold, same siis hash when one is given
"""
from __future__ import annotations

import hashlib
import os
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np

ANY = "*"
SIM_WITH_SIIS = float(os.environ.get("CACHE_SIM_THRESHOLD", "0.62"))
SIM_NO_SIIS = float(os.environ.get("CACHE_SIM_THRESHOLD_NOSIIS", "0.72"))


def exact_key(norm_query: str, siis_hash: str) -> str:
    return hashlib.sha256(f"{norm_query}\x00{siis_hash}".encode()).hexdigest()


@dataclass
class Entry:
    query: str
    norm_query: str
    siis_hash: str
    fp_key: str
    response: dict
    info: dict
    vector: np.ndarray


@dataclass
class Hit:
    entry: Entry
    tier: str
    similarity: float


class SemanticCache:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.entries: list[Entry] = []
        self.exact: dict[str, int] = {}
        self.by_fp: dict[tuple[str, str], int] = {}
        self._matrix: Optional[np.ndarray] = None
        self.stats = {"lookups": 0, "exact": 0, "fingerprint": 0, "embedding": 0, "miss": 0}

    def __len__(self) -> int:
        return len(self.entries)

    def put(self, entry: Entry, generic_fp: bool = False) -> None:
        with self._lock:
            idx = len(self.entries)
            self.entries.append(entry)
            self.exact.setdefault(exact_key(entry.norm_query, entry.siis_hash), idx)
            self.exact.setdefault(exact_key(entry.norm_query, ANY), idx)
            if not generic_fp:
                self.by_fp.setdefault((entry.fp_key, entry.siis_hash), idx)
            self._matrix = None

    def _mat(self) -> np.ndarray:
        if self._matrix is None:
            self._matrix = np.stack([e.vector for e in self.entries]) if self.entries else np.zeros((0, 1))
        return self._matrix

    def get(self, norm_query: str, siis_hash: Optional[str], fp_key: str, vector_fn,
            generic_fp: bool = False) -> Optional[Hit]:
        """vector_fn is only called when tiers 1 and 2 miss, so exact hits stay sub-millisecond."""
        self.stats["lookups"] += 1
        sh = siis_hash or ANY
        idx = self.exact.get(exact_key(norm_query, sh))
        if idx is not None:
            self.stats["exact"] += 1
            return Hit(self.entries[idx], "exact", 1.0)
        if siis_hash and not generic_fp:
            idx = self.by_fp.get((fp_key, siis_hash))
            if idx is not None:
                self.stats["fingerprint"] += 1
                return Hit(self.entries[idx], "fingerprint", 1.0)
        if not self.entries:
            self.stats["miss"] += 1
            return None
        vec = vector_fn()
        sims = self._mat() @ vec
        if siis_hash:
            mask = np.array([e.siis_hash == siis_hash for e in self.entries])
            sims = np.where(mask, sims, -1)
            threshold = SIM_WITH_SIIS
        else:
            threshold = SIM_NO_SIIS
        best = int(np.argmax(sims))
        if sims[best] >= threshold:
            self.stats["embedding"] += 1
            return Hit(self.entries[best], "embedding", float(sims[best]))
        self.stats["miss"] += 1
        return None
