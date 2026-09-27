"""Print the rules-path plan for every kit row (developer review aid)."""
import sys
import time

from app.catalog import get_index
from app.fallback import build_plan
from app.kit import load_catalog, load_siis_rows
from app.segment import segment
from app.validate import validate_and_repair

ix, cat = get_index(), load_catalog()
only = set(sys.argv[1:])
for r in load_siis_rows():
    if only and r.id not in only:
        continue
    t0 = time.perf_counter()
    art = segment(r.title, r.content)
    plan, info = build_plan(r.original_query, art, ix)
    fixed, rep = validate_and_repair(plan, r.content, cat, info["retrieval_confidence"])
    ms = (time.perf_counter() - t0) * 1000
    print(f"\n=== {r.id} [{ms:.0f} ms] fixes={rep['fix_count']} dropped={len(rep['dropped_steps'])} | {r.original_query[:90]}")
    print(f"    article: {r.title}")
    for g in fixed["contexts"]:
        print(f"  GOAL {g['goal']} | {g['title']} | score={g['score']}")
        for a in g["actions"]:
            grp = a["stepGroups"][0]
            dl = grp["actionableDeeplink"]
            print(f"    [{a['category']:8}] {a['actionName']} — {a['description']}"
                  f"  {'-> ' + dl['message'] if dl else ''}")
            for s in grp["steps"]:
                print(f"        · {s}")
    if not fixed["contexts"]:
        print("  NO_MATCH", info.get("reason"))
