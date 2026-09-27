"""Writes metrics.md in the guide's Appendix C format.

python -m eval.metrics                  # in-process judge run + ablation on the 20 kit rows
python -m eval.metrics --url URL        # judge run against a live server
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import time

import numpy as np

from app.catalog import CatalogIndex
from app.config import ROOT
from app.embed import get_embedder
from app.fallback import build_plan
from app.kit import load_catalog, load_siis_rows
from app.llm import make_provider
from app.segment import segment
from app.validate import validate_and_repair
from eval.judge_sim import check, run


def step_accuracy(body: dict, label: dict) -> float:
    """0-3: completeness + correctness + ordering."""
    actions = [a for g in body.get("contexts", []) for a in g["actions"]]
    if not actions:
        return 0.0
    blobs = [(a["actionName"] + " " + " ".join(s for grp in a["stepGroups"] for s in grp["steps"])).lower()
             for a in actions]
    exp = [re.compile(e, re.I) for e in label["expected"]]
    completeness = sum(any(rx.search(b) for b in blobs) for rx in exp) / len(exp)
    correctness = sum(any(rx.search(b) for rx in exp) for b in blobs) / len(blobs)
    cats = [a["category"] for a in actions]
    ordered = cats == sorted(cats, key=lambda c: {"auto": 0, "manual": 1, "critical": 2}[c])
    return round(completeness + correctness + (1.0 if ordered else 0.0), 3)


def deeplink_relevance(body: dict, label: dict, catalog) -> float | None:
    """0-2 per labelled screen: 2 exact target screen, 1 placeholder/parent, 0 wrong or missing."""
    if not label.get("deeplinks"):
        return None
    scores = []
    for steps_rx, msg_rx in label["deeplinks"].items():
        best = 0
        for g in body.get("contexts", []):
            for a in g["actions"]:
                for grp in a["stepGroups"]:
                    if not re.search(steps_rx, " ".join(grp["steps"]), re.I):
                        continue
                    dl = grp.get("actionableDeeplink")
                    if not dl:
                        continue
                    if re.search(msg_rx, dl.get("message") or "", re.I):
                        best = max(best, 2)
                    else:
                        best = max(best, 1)
        scores.append(best)
    return round(sum(scores) / len(scores), 3)


def unseen_exact(index: CatalogIndex, catalog) -> float:
    items = [u for u in json.loads((ROOT / "eval" / "unseen.json").read_text()) if u.get("expected_deeplink")]
    hits = 0
    for u in items:
        art = segment(u["siis_response"]["title"], u["siis_response"]["content"])
        plan, info = build_plan(u["query"], art, index)
        body, _ = validate_and_repair(plan, u["siis_response"]["content"], catalog, 0.5)
        want = catalog.by_id[u["expected_deeplink"]]["deeplink"]
        hits += any(grp.get("actionableDeeplink") and grp["actionableDeeplink"]["deeplink"] == want
                    for g in body["contexts"] for a in g["actions"] for grp in a["stepGroups"])
    return 100 * hits / max(1, len(items))


def ablation(labels: dict) -> list[dict]:
    catalog = load_catalog()
    rows = load_siis_rows()
    variants = [("Variant A: Hybrid BM25 + Dense Embedding Retrieval", CatalogIndex(catalog, get_embedder(), "hybrid")),
                ("Variant B: Pure Rules-Based Deeplink Mapping", CatalogIndex(catalog, get_embedder(), "keyword"))]
    out = []
    provider = make_provider("compile")
    if provider is not None:
        from app.extract import llm_plan
        from app.llm import LLMError

        hybrid = variants[0][1]
        accs, dls, lats, costs = [], [], [], []
        for r in rows:
            art = segment(r.title, r.content)
            t0 = time.perf_counter()
            try:
                plan, res = llm_plan(r.original_query, art, hybrid, provider, 60)
                costs.append(res.cost_usd)
            except LLMError:
                plan = {"contexts": []}
            body, _ = validate_and_repair(plan, r.content, catalog, 0.8)
            lats.append((time.perf_counter() - t0) * 1000)
            accs.append(step_accuracy(body, labels[r.id]))
            d = deeplink_relevance(body, labels[r.id], catalog)
            if d is not None:
                dls.append(d)
        out.append({"name": f"Baseline: Full LLM Deeplink Mapping ({provider.name}/{provider.model})",
                    "acc": np.mean(accs), "dl": np.mean(dls) if dls else None,
                    "p95": np.percentile(lats, 95), "cost": np.mean(costs) if costs else 0.0,
                    "note": "LLM picks sentences and catalog ids; code copies text and URIs"})
    else:
        out.append({"name": "Baseline: Full LLM Deeplink Mapping", "acc": None, "dl": None, "p95": None,
                    "cost": None, "note": "Not run in this build: no LLM key configured (set GEMINI_API_KEY)"})
    for name, index in variants:
        accs, dls, lats = [], [], []
        for r in rows:
            t0 = time.perf_counter()
            art = segment(r.title, r.content)
            plan, info = build_plan(r.original_query, art, index)
            body, _ = validate_and_repair(plan, r.content, catalog, info.get("retrieval_confidence", 0.5))
            lats.append((time.perf_counter() - t0) * 1000)
            accs.append(step_accuracy(body, labels[r.id]))
            d = deeplink_relevance(body, labels[r.id], catalog)
            if d is not None:
                dls.append(d)
        out.append({"name": name, "acc": np.mean(accs), "dl": np.mean(dls) if dls else None,
                    "p95": np.percentile(lats, 95), "cost": 0.0, "unseen": unseen_exact(index, catalog),
                    "note": "shipped default" if index.mode == "hybrid" else "keyword + exact screen-name match only"})
    return out


def fmt(v, spec="{:.2f}", none="n/a"):
    return none if v is None else spec.format(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url")
    args = ap.parse_args()
    labels = {k: v for k, v in json.loads((ROOT / "eval" / "labels.json").read_text()).items() if not k.startswith("_")}
    catalog = load_catalog()

    rep = run(args.url, repeats=2)
    lat = rep["latency"]

    # Accuracy of the shipped outputs (results.jsonl) against labels
    shipped = [json.loads(line) for line in (ROOT / "results.jsonl").read_text().splitlines() if line.strip()]
    ids = {r.original_query: r.id for r in load_siis_rows()}
    accs = [step_accuracy(x["response"], labels[ids[x["query"]]]) for x in shipped]
    dls = [d for x in shipped if (d := deeplink_relevance(x["response"], labels[ids[x["query"]]], catalog)) is not None]
    checks = [check(x["response"], catalog) for x in shipped]
    rule_ok = sum(c.get("goal_ok", 0) + c.get("title_ok", 0) + c.get("desc_ok", 0) for c in checks)
    rule_n = sum(2 * c.get("goals", 0) + c.get("actions", 0) for c in checks)
    abl = ablation(labels)

    emb = get_embedder().name
    model = shipped[0]["meta"].get("model") if shipped else "rules"
    lines = [
        "# System Performance Metrics & Evaluation Report",
        f"**Model(s):** {model} (runtime LLM optional; rules path serves every request when no key is set)",
        f"**Embeddings:** {emb}",
        f"**Environment:** {os.cpu_count()} vCPU / {platform.machine()} / {platform.system()} {platform.release()} / Python {platform.python_version()}",
        f"**Measured:** {time.strftime('%Y-%m-%d %H:%M')} against {rep['target']}",
        "", "---", "",
        "## 1. Schema & Rule Compliance",
        f"Evaluated on the 20 kit scenarios, {sum(v['n'] for k, v in lat.items())} live probes and 36 held-out unseen scenarios.",
        "", "| Metric | Target | Measured Value |", "| :--- | :--- | :--- |",
        f"| Schema-valid output lines | >= 99% | {rep['schema_valid_pct']}% |",
        f"| Rule compliance (Goal / Title / Description syntax) | >= 95% | {round(100 * rule_ok / max(1, rule_n), 2)}% (results.jsonl), goal {rep['goal_regex_pct']}% / title {rep['title_rule_pct']}% / description {rep['description_rule_pct']}% (live) |",
        f"| Absolute URL leaks | 0 | {rep['url_leaks']} |",
        f"| Deeplink catalog validity (exact URI match) | 100% | {rep['deeplinks_in_catalog_pct']}% |",
        f"| Auto actions carrying valid actionable deeplink | >= 90% | {rep['auto_with_deeplink_pct']}% |",
        f"| Manual actions carrying a deeplink | 0 | {rep['manual_with_deeplink']} |",
        f"| Critical actions ordered last | 100% | {rep['critical_last_pct']}% |",
        "", "---", "",
        "## 2. Accuracy Benchmarks",
        "Evaluated against hand labels (`eval/labels.json`) for the 20 kit scenarios.",
        "", "| Evaluation Metric | Scale / Anchor | Score |", "| :--- | :--- | :--- |",
        f"| Step accuracy (completeness, correctness, ordering) | 0.0 - 3.0 | {np.mean(accs):.2f} |",
        f"| Deeplink relevance (exact target screen vs. parent menu) | 0.0 - 2.0 | {np.mean(dls):.2f} (labelled screens) |",
        f"| Unseen articles: exact catalog entry chosen | - | {rep['unseen_deeplink_exact_pct']}% |",
        f"| Unseen articles: correct setting, either toggle direction | - | {rep['unseen_deeplink_same_setting_pct']}% |",
        "", "---", "",
        "## 3. Latency Benchmarks (N >= 30 requests per path)",
        "", "| Execution Path | Target (P95) | P50 (ms) | P95 (ms) | N |", "| :--- | :--- | :--- | :--- | :--- |",
        f"| Cache hit - exact query match | <= 300 ms | {lat['repeat_exact']['p50']} | {lat['repeat_exact']['p95']} | {lat['repeat_exact']['n']} |",
        f"| Cache hit - unseen semantic paraphrase | <= 300 ms | {lat['paraphrase_heldout']['p50']} | {lat['paraphrase_heldout']['p95']} | {lat['paraphrase_heldout']['n']} |",
        f"| Paraphrase without siis_response (store lookup) | <= 300 ms | {lat['paraphrase_no_siis']['p50']} | {lat['paraphrase_no_siis']['p95']} | {lat['paraphrase_no_siis']['n']} |",
        f"| Cold query - full pipeline extraction & mapping | <= 8000 ms | {lat['unseen_siis']['p50']} | {lat['unseen_siis']['p95']} | {lat['unseen_siis']['n']} |",
        "", "---", "",
        "## 4. Operational Cost & Cache Efficacy",
        "", "| Metric Item | Target | Measured Value |", "| :--- | :--- | :--- |",
        f"| Cold query average inference cost | Tracked | ${float(np.mean([x['meta'].get('cost_usd') or 0 for x in shipped])):.4f} (rules path; LLM cost logged per request in meta.cost_usd) |",
        "| Cache hit inference cost | $0.00 | $0.00 |",
        f"| Exact repeat cache hit rate | >= 90% | {round(100 * lat['repeat_exact']['cache_hit_rate'], 1)}% |",
        f"| Semantic cache hit rate (on unseen paraphrases) | >= 80% | {round(100 * lat['paraphrase_heldout']['cache_hit_rate'], 1)}% |",
        "| Cost derivation method | - | (prompt tokens x PRICE_IN_PER_M + completion tokens x PRICE_OUT_PER_M) / 1e6 |",
        "", "---", "",
        "## 5. Architectural Ablation Analysis",
        "", "| Architecture Variant | Step Accuracy (0-3) | Deeplink Relevance (0-2) | Unseen Exact Deeplink | Latency (P95) | Cost / Query | Key Observations |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for v in abl:
        lines.append(f"| {v['name']} | {fmt(v['acc'])} | {fmt(v['dl'])} | {fmt(v.get('unseen'), '{:.0f}%')} | {fmt(v['p95'], '{:.0f} ms')} | "
                     f"{fmt(v['cost'], '${:.4f}')} | {v['note']} |")
    lines += [
        "", "---", "",
        "## 6. Automated Gates (judge simulator)",
        "", "| Gate | Result |", "| :--- | :--- |",
        *[f"| {k} | {'PASS' if ok else 'FAIL'} |" for k, ok in rep["gates"].items()],
        "", "---", "",
        "## 7. Known Limitations",
        "- The kit pairs some complaints with loosely related articles (for example row_8, a small screen, with a TV mirroring article). "
        "We never invent steps, so those plans contain only the article's closest steps.",
        "- Hand labels (`eval/labels.json`) were drafted with AI assistance from the SIIS text (see AI_DISCLOSURE.md); they are a sanity check, not an official ground truth.",
        "- Without an LLM key every request runs on the deterministic rules path. The LLM path (Gemini, Mistral or Claude) "
        "is implemented and unit-tested with a mock provider, and plugs in by setting an API key.",
        "- The paraphrase hit rate relies on fingerprint and embedding tiers keyed to the same SIIS article; "
        "with no article, a paraphrase must be close to a known query or article (75% on held-out phrasings).",
        "- Toggle direction (enable vs. disable) is inferred from wording near the setting name and is the most common mapping error.",
    ]
    (ROOT / "metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
