"""Deeplink catalog retrieval: BM25 keyword + dense embeddings (hybrid).

Matching uses only descriptive metadata (message, description, qna_description,
validation key, originalType). The masked URI string is never used for matching.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz

from app.embed import Embedder, get_embedder
from app.kit import DUMMY_DEEPLINK, Catalog, load_catalog

STOP = {
    "the", "a", "an", "to", "of", "on", "in", "for", "and", "or", "via", "device", "settings",
    "page", "your", "you", "is", "it", "with", "by", "from", "this", "that", "be", "can", "opens",
    "open", "tap", "go", "navigate", "select", "then", "screen", "phone",
}
TV_RE = re.compile(r"\b(tv|television|smartthings)\b", re.I)
APPLIANCE_RE = re.compile(r"\b(refrigerator|fridge|air conditioner|washer|dryer|oven|vacuum|dishwasher)\b", re.I)
OFF_RE = re.compile(r"\b(disable|disabling|turn(?:ing)? off|switch(?:ing)? off|deactivate|to disable it)\b", re.I)
ON_RE = re.compile(r"\b(enable|enabling|turn(?:ing)? on|switch(?:ing)? on|activate)\b", re.I)

W_BM25, W_EMB = 0.45, 0.55
POLARITY_BONUS = 0.12
KEY_MATCH_MIN = 0.85   # fuzzy match between a named target screen and the entry key
KEY_BONUS = 0.35
ACCEPT_HYBRID = 0.86   # without a key match, only accept very strong hybrid matches
VERB_PREFIX_RE = re.compile(r"^(enable|disable|view|adjust|switch|set|open)\s+", re.I)
TARGET_RES = [
    re.compile(r"switch(?:es)? next to ([A-Z][\w' +&/-]{1,40}?)(?=\s+(?:to|and|or)\b|[.,]|$)"),
    re.compile(r"\b(?i:enabl\w*|disabl\w*|turn(?:ing)? (?:on|off))\s+(?:the\s+)?[\"']?([A-Z][\w' +&/-]{1,40}?)[\"']?(?=\s+(?:option|feature|setting|mode\b)|[.,]|$)"),
    re.compile(r"\b(?i:select|tap)\s+(?:on\s+)?([A-Z][\w' +&/-]{1,40}?)(?=[.,]|$|\s+(?:and|then|to|again)\b)"),
]
GENERIC_TARGETS = {"settings", "ok", "restart", "reset", "power", "apps", "display", "connections",
                   "delete all", "start now", "yes", "allow", "home"}


def tokenize(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return [t[:-1] if t.endswith("s") and len(t) > 4 else t for t in toks if t not in STOP]


def family(entry: dict) -> str:
    text = f"{entry.get('description', '')} {entry.get('message', '')}"
    if APPLIANCE_RE.search(text):
        return "appliance"
    if re.search(r"\bvia TV\b|\bTV (Settings|VoiceAssist)\b", text):
        return "tv"
    return "mobile"


def index_text(entry: dict) -> str:
    val = (entry.get("validation") or {}).get("key", "")
    return " ".join(
        str(x) for x in (entry.get("message"), entry.get("description"),
                         entry.get("qna_description"), val, entry.get("originalType")) if x
    )


def norm_label(text: str) -> str:
    text = VERB_PREFIX_RE.sub("", text or "")
    return " ".join(re.findall(r"[a-z0-9]+", text.lower().replace("wi-fi", "wifi")))


def extract_targets(text: str, path: Optional[list[str]] = None) -> list[str]:
    """Named screens or settings a step group points at, most specific first."""
    out: list[str] = []
    if path:
        out.extend(reversed([h for h in path[1:]]))
    for rx in TARGET_RES:
        out.extend(m.strip() for m in rx.findall(text))
    seen, result = set(), []
    for t in out:
        k = norm_label(t)
        if k and k not in GENERIC_TARGETS and k not in seen:
            seen.add(k)
            result.append(t)
    return result


@dataclass(frozen=True)
class Candidate:
    id: str
    score: float
    message: str
    description: str
    originalType: str
    key_match: float = 0.0
    hybrid: float = 0.0  # BM25 + embedding (+ polarity), before the key bonus

    @property
    def accepted(self) -> bool:
        return self.key_match >= KEY_MATCH_MIN or self.hybrid >= ACCEPT_HYBRID

    def public(self) -> dict:
        """What the LLM is allowed to see: never the URI."""
        return {"id": self.id, "message": self.message, "description": self.description,
                "type": self.originalType}


class CatalogIndex:
    def __init__(self, catalog: Catalog, embedder: Optional[Embedder] = None) -> None:
        self.catalog = catalog
        self.embedder = embedder or get_embedder()
        self.entries = [
            e for e in catalog.entries
            if e["deeplink"] != DUMMY_DEEPLINK and e.get("originalType") not in (None, "placeholder")
        ]
        self.families = [family(e) for e in self.entries]
        self.bm25 = BM25Okapi([tokenize(index_text(e)) for e in self.entries])
        self.vectors = self.embedder.encode([index_text(e) for e in self.entries])
        self.labels = [
            {norm_label((e.get("validation") or {}).get("key", "")), norm_label(e.get("message", ""))}
            - {""}
            for e in self.entries
        ]

    def _key_match(self, i: int, targets: list[str]) -> float:
        best = 0.0
        for t in targets:
            nt = norm_label(t)
            for lab in self.labels[i]:
                r = fuzz.ratio(nt, lab) / 100
                if r > best:
                    best = r
        return best

    def __len__(self) -> int:
        return len(self.entries)

    def candidates(self, text: str, k: int = 5, context: str = "",
                   vector: Optional[np.ndarray] = None,
                   targets: Optional[list[str]] = None) -> list[Candidate]:
        """Top-k catalog entries for a step or step group.

        context adds on/off polarity cues; targets are named screens (see extract_targets).
        """
        full = f"{text} {context}".strip()
        targets = targets if targets is not None else extract_targets(text)
        allow_tv = bool(TV_RE.search(full))
        bm = np.asarray(self.bm25.get_scores(tokenize(text)), dtype=np.float32)
        bm = bm / bm.max() if bm.max() > 0 else bm
        q = vector if vector is not None else self.embedder.encode_one(text)
        emb = np.clip(self.vectors @ q, 0, 1)
        score = W_BM25 * bm + W_EMB * emb

        want_off, want_on = bool(OFF_RE.search(full)), bool(ON_RE.search(full))
        key_match = np.zeros(len(self.entries), dtype=np.float32)
        for i, e in enumerate(self.entries):
            if targets:
                key_match[i] = self._key_match(i, targets)
            fam = self.families[i]
            if fam == "appliance" or (fam == "tv" and not allow_tv):
                score[i] = -1
                continue
            t = e.get("originalType")
            if want_off and not want_on:
                score[i] += POLARITY_BONUS if t == "offURL" else (-POLARITY_BONUS if t == "onURL" else 0)
            elif want_on and not want_off:
                score[i] += POLARITY_BONUS if t == "onURL" else (-POLARITY_BONUS if t == "offURL" else 0)

        hybrid = score.copy()
        score = score + np.where(key_match >= KEY_MATCH_MIN, KEY_BONUS * key_match, 0)
        order = sorted(range(len(self.entries)), key=lambda i: (-round(float(score[i]), 6),
                                                                 self.entries[i]["id"]))
        out = []
        for i in order[:k]:
            if score[i] <= 0:
                break
            e = self.entries[i]
            out.append(Candidate(e["id"], round(float(min(score[i] / (1 + KEY_BONUS), 1.0)), 4),
                                 e.get("message") or "", e.get("description") or "",
                                 e.get("originalType") or "", round(float(key_match[i]), 4),
                                 round(float(hybrid[i]), 4)))
        return out


@lru_cache(maxsize=1)
def get_index() -> CatalogIndex:
    return CatalogIndex(load_catalog())
