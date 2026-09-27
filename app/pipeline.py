"""The engine: cache -> (LLM | rules) -> validator -> score -> diagnosis.

One instance per process, warmed at boot with every kit article and every
pre-computed query variation from results.jsonl.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import threading
import time
from collections import defaultdict, deque
from typing import Any, Optional

import numpy as np

from app.cache import Entry, SemanticCache
from app.catalog import get_index
from app.config import ROOT, settings
from app.diagnose import diagnose
from app.extract import llm_plan
from app.fallback import build_plan
from app.fingerprint import fingerprint, normalize
from app.kit import load_catalog, load_siis_rows
from app.llm import LLMError, make_provider
from app.segment import Article, segment, siis_hash
from app.validate import validate_and_repair

log = logging.getLogger("fixpath")
RESULTS_PATH = ROOT / "results.jsonl"
NO_SIIS_ARTICLE_MIN = float(os.environ.get("NO_SIIS_ARTICLE_MIN", "0.30"))
LLM_RUNTIME = os.environ.get("LLM_RUNTIME", "1") != "0"


def _pct(values, p: float) -> Optional[float]:
    if not values:
        return None
    return round(float(np.percentile(np.asarray(values), p)), 2)


class Engine:
    def __init__(self) -> None:
        self.ready = False
        self.catalog = load_catalog()
        self.index = None
        self.cache = SemanticCache()
        self.provider = None
        self.articles: dict[str, Article] = {}
        self._article_keys: list[str] = []
        self._article_vecs: Optional[np.ndarray] = None
        self.latency: dict[str, deque] = defaultdict(lambda: deque(maxlen=2000))
        self.counters: dict[str, float] = defaultdict(float)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ boot
    def warm(self) -> "Engine":
        t0 = time.perf_counter()
        self.index = get_index()
        self.provider = make_provider("runtime") if LLM_RUNTIME else None
        rows = load_siis_rows()
        for r in rows:
            art = segment(r.title, r.content)
            self.articles.setdefault(art.hash, art)
        self._article_keys = list(self.articles)
        emb = self.index.embedder
        self._article_vecs = emb.encode([
            f"{a.title}. " + " ".join(s.heading for s in a.sections) + " " + a.body[:600]
            for a in (self.articles[k] for k in self._article_keys)
        ])

        precomputed = self._load_results()
        for r in rows:
            art = segment(r.title, r.content)
            pre = precomputed.get(r.original_query.strip())
            if pre:
                response, info = pre["response"], {"path": "precompiled", "model": pre.get("meta", {}).get("model")}
            else:
                response, info = self._build(r.original_query, art)
            variations = (pre or {}).get("query_variations", [])
            self._prime(r.original_query, art.hash, response, info, variations)
        self.ready = True
        log.info(json.dumps({"event": "warm", "ms": round((time.perf_counter() - t0) * 1000),
                             "cache_entries": len(self.cache), "embedder": emb.name,
                             "llm": getattr(self.provider, "name", None)}))
        return self

    def _load_results(self) -> dict[str, dict]:
        if not RESULTS_PATH.exists():
            return {}
        out = {}
        for line in RESULTS_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                out[row["query"].strip()] = row
        return out

    def _prime(self, query: str, sh: str, response: dict, info: dict, variations: list[str]) -> None:
        texts = [query] + [v for v in variations if v and v.strip()]
        norms = [normalize(t) for t in texts]
        vecs = self.index.embedder.encode(norms)
        for text, norm, vec in zip(texts, norms, vecs):
            fp = fingerprint(text)
            self.cache.put(Entry(text, norm, sh, fp.key, response, info, vec),
                           generic_fp=fp.symptom == "general")

    # ------------------------------------------------------------------ build
    def _build(self, query: str, article: Article, content: Optional[str] = None) -> tuple[dict, dict]:
        content = content if content is not None else article.body
        info: dict[str, Any] = {"path": "rules", "model": "rules", "input_tokens": 0,
                                "output_tokens": 0, "cost_usd": 0.0}
        plan, rinfo = None, {}
        if self.provider is not None:
            try:
                plan, res = llm_plan(query, article, self.index, self.provider, settings.llm_timeout_s)
                info.update(path="llm", model=res.model, input_tokens=res.input_tokens,
                            output_tokens=res.output_tokens, cost_usd=res.cost_usd,
                            llm_ms=round(res.latency_ms))
            except LLMError as exc:
                info["llm_error"] = str(exc)[:200]
                plan = None
        if plan is None:
            plan, rinfo = build_plan(query, article, self.index)
        conf = rinfo.get("retrieval_confidence")
        if conf is None:  # LLM path: confidence from query-article similarity
            conf = self._query_article_conf(query, article)
        fixed, report = validate_and_repair(plan, content, self.catalog, retrieval_confidence=conf)
        if not fixed["contexts"] and info["path"] == "llm":  # never return empty when rules can help
            plan, rinfo = build_plan(query, article, self.index)
            fixed, report = validate_and_repair(plan, content, self.catalog,
                                                retrieval_confidence=rinfo.get("retrieval_confidence", 0.5))
            info["path"] = "rules_after_llm"
        info.update(validator_fixes=report["fix_count"], provenance=report["provenance"],
                    retrieval_confidence=conf, dropped_steps=len(report["dropped_steps"]))
        if not fixed["contexts"]:
            info["reason"] = rinfo.get("reason", "no steps in the article match the complaint")
        return fixed, info

    def _query_article_conf(self, query: str, article: Article) -> float:
        emb = self.index.embedder
        q = emb.encode_one(query)
        a = emb.encode_one(f"{article.title}. {article.body[:600]}")
        return round(max(0.0, min(1.0, 0.45 + 0.9 * float(a @ q))), 4)

    # ------------------------------------------------------------------ request
    def troubleshoot(self, query: str, siis: Any = None, device_state: Optional[dict] = None) -> tuple[dict, dict]:
        t0 = time.perf_counter()
        query = (query or "").strip()
        title, content = "", ""
        if isinstance(siis, dict):
            title, content = str(siis.get("title") or ""), str(siis.get("content") or "")
        elif isinstance(siis, str):
            content = siis
        has_siis = bool(content.strip())
        sh = siis_hash(title, content) if has_siis else None

        norm = normalize(query)
        fp = fingerprint(query)
        vec_holder: dict[str, np.ndarray] = {}

        def vector():
            if "v" not in vec_holder:
                vec_holder["v"] = self.index.embedder.encode_one(norm)
            return vec_holder["v"]

        hit = self.cache.get(norm, sh, fp.key, vector, generic_fp=fp.symptom == "general")
        meta: dict[str, Any] = {"fingerprint": fp.to_dict()}
        if hit:
            body = copy.deepcopy(hit.entry.response)
            meta.update(cache_hit=True, cache_tier=hit.tier, similarity=round(hit.similarity, 4),
                        path="cache", model=hit.entry.info.get("model", "rules"),
                        input_tokens=0, output_tokens=0, cost_usd=0.0, matched_query=hit.entry.query[:120])
        else:
            article = None
            if has_siis:
                article = segment(title, content)
            else:
                article, sim = self._nearest_article(vector())
                meta["article_lookup_similarity"] = round(sim, 4)
            if article is None:
                body = {"contexts": [], "fallback": "no_siis_context"}
                meta.update(cache_hit=False, cache_tier=None, path="no_siis_context", model=None,
                            cost_usd=0.0)
            else:
                body, info = self._build(query, article, content if has_siis else None)
                if not body["contexts"]:
                    body = {"contexts": [], "fallback": "no_match"}
                meta.update(cache_hit=False, cache_tier=None, **info)
                self.cache.put(Entry(query, norm, article.hash if not has_siis else sh, fp.key,
                                     copy.deepcopy(body), info, vector()),
                               generic_fp=fp.symptom == "general")

        body, findings = diagnose(body, device_state)
        if device_state is not None:
            meta["diagnosis"] = findings
        latency = round((time.perf_counter() - t0) * 1000, 2)
        meta["latency_ms"] = latency
        self._record(meta, latency)
        return body, meta

    def _nearest_article(self, qvec: np.ndarray) -> tuple[Optional[Article], float]:
        if self._article_vecs is None or not len(self._article_keys):
            return None, 0.0
        sims = self._article_vecs @ qvec
        i = int(np.argmax(sims))
        if sims[i] < NO_SIIS_ARTICLE_MIN:
            return None, float(sims[i])
        return self.articles[self._article_keys[i]], float(sims[i])

    # ------------------------------------------------------------------ metrics
    def _record(self, meta: dict, latency: float) -> None:
        with self._lock:
            key = f"cache_{meta.get('cache_tier')}" if meta.get("cache_hit") else meta.get("path", "other")
            self.latency[key].append(latency)
            self.latency["all"].append(latency)
            self.counters["requests"] += 1
            self.counters["cache_hits"] += 1 if meta.get("cache_hit") else 0
            self.counters["cost_usd"] += float(meta.get("cost_usd") or 0)
            self.counters["tokens"] += float((meta.get("input_tokens") or 0) + (meta.get("output_tokens") or 0))

    def metrics(self) -> dict:
        with self._lock:
            req = self.counters["requests"] or 1
            return {
                "requests": int(self.counters["requests"]),
                "cache_hit_rate": round(self.counters["cache_hits"] / req, 4),
                "total_cost_usd": round(self.counters["cost_usd"], 6),
                "total_tokens": int(self.counters["tokens"]),
                "cache_entries": len(self.cache),
                "cache_tiers": dict(self.cache.stats),
                "latency_ms": {k: {"n": len(v), "p50": _pct(v, 50), "p95": _pct(v, 95)}
                               for k, v in self.latency.items()},
                "embedder": self.index.embedder.name if self.index else None,
                "llm": getattr(self.provider, "name", None),
            }


_ENGINE: Optional[Engine] = None


def get_engine() -> Engine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = Engine()
    return _ENGINE
