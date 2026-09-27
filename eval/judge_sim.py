"""Replays the judges' probes (FAQ Q21) and checks every automated gate.

python -m eval.judge_sim                       # in-process (no server needed)
python -m eval.judge_sim --url http://localhost:8080
Probes: /health, canonical kit queries (cold), exact repeats, held-out paraphrases
(never pre-cached), queries without siis_response, unseen SIIS articles.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from typing import Any

import numpy as np

from app.config import ROOT
from app.kit import DUMMY_DEEPLINK, load_catalog, load_schema, load_siis_rows

GOAL_RE = re.compile(r"^Follow these steps to perform this .+ (Troubleshooting|Configuration)\.?$")
URL_LEAK_RE = re.compile(r"https?://|www\.|\.com\b|\.html?\b|\]\(|<a\s", re.I)


class Client:
    def __init__(self, url: str | None) -> None:
        self.url = url.rstrip("/") if url else None
        if self.url:
            import httpx

            self.http = httpx.Client(timeout=30)
        else:
            import os

            os.environ.setdefault("WARM_SYNC", "1")
            from fastapi.testclient import TestClient

            from app.main import app

            self.tc = TestClient(app)
            self.tc.__enter__()

    def get(self, path: str):
        return self.http.get(self.url + path) if self.url else self.tc.get(path)

    def post(self, path: str, payload: dict):
        return self.http.post(self.url + path, json=payload) if self.url else self.tc.post(path, json=payload)


def text_fields(body: dict) -> list[str]:
    out = []
    for g in body.get("contexts", []):
        out += [g.get("goal", ""), g.get("title", "")]
        for a in g.get("actions", []):
            out += [a.get("actionName", ""), a.get("description", "")]
            for grp in a.get("stepGroups", []):
                out += grp.get("steps", [])
                dl = grp.get("actionableDeeplink") or {}
                out += [dl.get("description") or "", dl.get("message") or ""]
    return out


def check(body: dict, catalog) -> dict[str, Any]:
    schema = load_schema()
    r: dict[str, Any] = {"schema_valid": True, "non_empty": bool(body.get("contexts"))}
    try:
        schema.ContextDeeplinkResponse.model_validate(body)
    except Exception:
        r["schema_valid"] = False
    scored = {k: v for k, v in body.items() if k != "meta"}
    r["url_leak"] = bool(URL_LEAK_RE.search(json.dumps(scored)))
    counts = defaultdict(int)
    for g in body.get("contexts", []):
        counts["goals"] += 1
        counts["goal_ok"] += bool(GOAL_RE.match(g.get("goal", "")))
        counts["title_ok"] += 2 <= len(g.get("title", "").split()) <= 3
        counts["score_ok"] += isinstance(g.get("score"), (int, float)) and 0 <= g["score"] <= 1
        cats = [a.get("category") for a in g.get("actions", [])]
        if "critical" in cats:
            counts["critical_last_ok"] += all(c == "critical" for c in cats[cats.index("critical"):])
            counts["has_critical"] += 1
        for a in g.get("actions", []):
            counts["actions"] += 1
            d = a.get("description", "")
            counts["desc_ok"] += d.startswith("It will") and 5 <= len(d.split()) <= 7
            links = [grp.get("actionableDeeplink") for grp in a.get("stepGroups", [])]
            has = any(links)
            if a.get("category") == "auto":
                counts["auto"] += 1
                counts["auto_with_link"] += has
            if a.get("category") == "manual" and has:
                counts["manual_with_link"] += 1
            for dl in filter(None, links):
                counts["links"] += 1
                uri = dl.get("deeplink")
                ok = uri == DUMMY_DEEPLINK or uri in catalog.by_deeplink
                counts["links_in_catalog"] += ok
                counts["dummy_links"] += uri == DUMMY_DEEPLINK
            for grp in a.get("stepGroups", []):
                counts["empty_steps"] += any(not s.strip() for s in grp.get("steps", [])) or not grp.get("steps")
    r.update(counts)
    return r


def pct(xs, p):
    return round(float(np.percentile(xs, p)), 2) if xs else None


def run(url: str | None = None, repeats: int = 2) -> dict:
    catalog = load_catalog()
    rows = load_siis_rows()
    heldout = json.loads((ROOT / "eval" / "heldout_paraphrases.json").read_text())
    unseen_path = ROOT / "eval" / "unseen.json"
    if not unseen_path.exists():
        from compiler.unseen import main as build_unseen

        build_unseen()
    unseen = json.loads(unseen_path.read_text())

    c = Client(url)
    t0 = time.perf_counter()
    while True:
        h = c.get("/health")
        if h.status_code == 200 or time.perf_counter() - t0 > 180:
            break
        time.sleep(0.5)
    health_ok = h.status_code == 200 and h.json() == {"status": "ok"}

    results: dict[str, list] = defaultdict(list)

    def probe(path: str, payload: dict, extra: dict | None = None):
        t = time.perf_counter()
        resp = c.post("/v1/troubleshoot", payload)
        client_ms = (time.perf_counter() - t) * 1000
        body = resp.json()
        meta = body.get("meta", {})
        rec = {"status": resp.status_code, "client_ms": client_ms, "cache_hit": bool(meta.get("cache_hit")),
               "tier": meta.get("cache_tier"), **check(body, catalog), **(extra or {})}
        results[path].append(rec)
        return body

    for r in rows:
        probe("canonical_cold", {"query": r.original_query,
                                 "siis_response": {"title": r.title, "content": r.content}})
    for _ in range(repeats):
        for r in rows:
            probe("repeat_exact", {"query": r.original_query,
                                   "siis_response": {"title": r.title, "content": r.content}})
    for r in rows:
        for q in heldout.get(r.id, []):
            probe("paraphrase_heldout", {"query": q, "siis_response": {"title": r.title, "content": r.content}})
            probe("paraphrase_no_siis", {"query": q})
    for u in unseen:
        body = probe("unseen_siis", {"query": u["query"], "siis_response": u["siis_response"]})
        if u.get("expected_deeplink"):
            want = catalog.by_id[u["expected_deeplink"]]
            got = [grp["actionableDeeplink"]["deeplink"] for g in body.get("contexts", [])
                   for a in g["actions"] for grp in a["stepGroups"] if grp.get("actionableDeeplink")]
            results["unseen_deeplink"].append({"exact": want["deeplink"] in got,
                                               "same_key": any(catalog.by_deeplink.get(x, {}).get("validation", {}) and
                                                               catalog.by_deeplink[x]["validation"]["key"] == want["validation"]["key"]
                                                               for x in got)})
    for q in ["how do I bake sourdough bread", "what is the capital of France"]:
        probe("off_topic_no_siis", {"query": q})

    everything = [x for k, v in results.items() if k != "unseen_deeplink" for x in v]
    tot = lambda key: sum(x.get(key, 0) for x in everything)  # noqa: E731
    lat = {k: {"n": len(v), "p50": pct([x["client_ms"] for x in v], 50), "p95": pct([x["client_ms"] for x in v], 95),
               "cache_hit_rate": round(sum(x["cache_hit"] for x in v) / len(v), 4) if v else None}
           for k, v in results.items() if k != "unseen_deeplink"}
    scored = [x for k, v in results.items() if k not in ("unseen_deeplink", "off_topic_no_siis") for x in v]
    ud = results["unseen_deeplink"]
    report = {
        "target": url or "in-process",
        "health_ok": health_ok,
        "requests": len(everything),
        "schema_valid_pct": round(100 * sum(x["schema_valid"] for x in everything) / len(everything), 2),
        "url_leaks": sum(x["url_leak"] for x in everything),
        "non_empty_pct_scored": round(100 * sum(x["non_empty"] for x in scored) / len(scored), 2),
        "unseen_non_empty_pct": round(100 * sum(x["non_empty"] for x in results["unseen_siis"]) / max(1, len(results["unseen_siis"])), 2),
        "goal_regex_pct": round(100 * tot("goal_ok") / max(1, tot("goals")), 2),
        "title_rule_pct": round(100 * tot("title_ok") / max(1, tot("goals")), 2),
        "description_rule_pct": round(100 * tot("desc_ok") / max(1, tot("actions")), 2),
        "score_range_pct": round(100 * tot("score_ok") / max(1, tot("goals")), 2),
        "deeplinks_in_catalog_pct": round(100 * tot("links_in_catalog") / max(1, tot("links")), 2),
        "auto_with_deeplink_pct": round(100 * tot("auto_with_link") / max(1, tot("auto")), 2),
        "manual_with_deeplink": tot("manual_with_link"),
        "critical_last_pct": round(100 * tot("critical_last_ok") / max(1, tot("has_critical")), 2),
        "empty_step_groups": tot("empty_steps"),
        "unseen_deeplink_exact_pct": round(100 * sum(x["exact"] for x in ud) / max(1, len(ud)), 2),
        "unseen_deeplink_same_setting_pct": round(100 * sum(x["same_key"] for x in ud) / max(1, len(ud)), 2),
        "latency": lat,
        "off_topic_fallbacks": [x["non_empty"] for x in results["off_topic_no_siis"]],
    }
    rep = lat.get("repeat_exact", {})
    para = lat.get("paraphrase_heldout", {})
    unseen_lat = lat.get("unseen_siis", {})
    report["gates"] = {
        "G2_health": health_ok,
        "G4_schema_valid_ge_90": report["schema_valid_pct"] >= 90,
        "G5_zero_url_leaks": report["url_leaks"] == 0,
        "A3_repeat_p95_le_300ms": (rep.get("p95") or 1e9) <= 300,
        "A3_repeat_hit_rate_ge_90": (rep.get("cache_hit_rate") or 0) >= 0.9,
        "A3_paraphrase_hit_rate_ge_80": (para.get("cache_hit_rate") or 0) >= 0.8,
        "A3_cold_p95_le_8s": (unseen_lat.get("p95") or 1e9) <= 8000,
        "A4_unseen_non_empty": report["unseen_non_empty_pct"] == 100,
        "A2_auto_have_deeplinks": report["auto_with_deeplink_pct"] == 100,
    }
    report["all_gates_pass"] = all(report["gates"].values())
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url")
    ap.add_argument("--out", default=str(ROOT / "eval" / "judge_report.json"))
    args = ap.parse_args()
    report = run(args.url)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "latency"}, indent=1))
    for k, v in report["latency"].items():
        print(f"  {k:22} n={v['n']:3} p50={v['p50']} ms p95={v['p95']} ms hit={v['cache_hit_rate']}")
    sys.exit(0 if report["all_gates_pass"] else 1)


if __name__ == "__main__":
    main()
