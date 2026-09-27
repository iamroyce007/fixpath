"""Offline compile: every kit row -> validated plan + 8-10 query variations -> results.jsonl.

python -m compiler.compile            # LLM when configured, else rules
python -m compiler.compile --rules    # force the rules path
Each line: {"query", "query_variations", "response", "meta"} (guide Appendix B).
"""
from __future__ import annotations

import argparse
import json
import os
import time

from app.config import ROOT
from app.kit import load_siis_rows
from app.llm import make_provider
from app.segment import segment
from compiler.paraphrase import variations


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", action="store_true", help="never call an LLM")
    ap.add_argument("--out", default=str(ROOT / "results.jsonl"))
    args = ap.parse_args()
    if args.rules:
        os.environ["LLM_PROVIDER"] = "none"

    from app.pipeline import Engine  # after env is final

    engine = Engine()
    engine.index = __import__("app.catalog", fromlist=["get_index"]).get_index()
    engine.provider = None if args.rules else make_provider("compile")
    para_provider = engine.provider

    lines = []
    for r in load_siis_rows():
        art = segment(r.title, r.content)
        t0 = time.perf_counter()
        response, info = engine._build(r.original_query, art, r.content)
        ms = round((time.perf_counter() - t0) * 1000, 2)
        if not response["contexts"]:
            response = {"contexts": [], "fallback": "no_match"}
        vars_, source = variations(r.original_query, para_provider)
        lines.append({
            "query": r.original_query,
            "query_variations": vars_,
            "response": response,
            "meta": {
                "latency_ms": ms,
                "cache_hit": False,
                "model": info.get("model"),
                "cost_usd": info.get("cost_usd", 0.0),
                "path": info.get("path"),
                "validator_fixes": info.get("validator_fixes"),
                "provenance_coverage": info.get("provenance", {}).get("coverage"),
                "variations_source": source,
                "id": r.id,
            },
        })
        print(f"{r.id}: {info.get('path'):6} goals={len(response['contexts'])} "
              f"actions={sum(len(g['actions']) for g in response['contexts'])} vars={len(vars_)} {ms} ms")
    with open(args.out, "w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
    print(f"wrote {len(lines)} lines to {args.out}")


if __name__ == "__main__":
    main()
