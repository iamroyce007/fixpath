"""Synthesise unseen SIIS scenarios for the generalisation probe (never put in the cache).

Two families, deterministic (seeded):
  1. mutated kit articles: sections shuffled/dropped, new headings, reworded complaint
  2. articles written around catalog entries: a Settings path to a real toggle, plus a
     restart and a service-centre fallback, in SIIS style
python -m compiler.unseen  -> eval/unseen.json
"""
from __future__ import annotations

import json
import random
import re

from app.catalog import family
from app.config import ROOT
from app.kit import load_catalog, load_siis_rows
from app.segment import strip_prefix
from app.twin import screen_for

OUT = ROOT / "eval" / "unseen.json"
PREFIX = "Smartphone,Tablet,Others Mobile,Mobile Accessories {title} ( Smartphone,Tablet,Others Mobile,Mobile Accessories): "

COMPLAINTS = {
    "on": ["My phone's {k} is off and I need it on, how do I fix this?",
           "{k} keeps getting disabled on my phone, I want it turned on.",
           "I can't find where to enable {k} on my new phone."],
    "off": ["{k} is on and it's annoying me, how do I turn it off?",
            "My phone keeps using {k} and I don't want it anymore.",
            "Please help me disable {k}, it started after the last update."],
}


def mutated(rows, rnd: random.Random) -> list[dict]:
    out, seen = [], set()
    for r in rows:
        if r.title in seen:
            continue
        seen.add(r.title)
        body = strip_prefix(r.content)
        parts = re.split(r"(?m)^(?=#{1,4} )", body)
        head, secs = parts[0], [p for p in parts[1:] if p.strip()]
        if len(secs) >= 3:
            keep = sorted(rnd.sample(range(len(secs)), max(2, len(secs) - 1)))
            secs = [secs[i] for i in keep]
        new_title = f"{r.title} (revised guide)"
        content = PREFIX.format(title=new_title) + head + "".join(secs)
        q = re.sub(r"\b(TechCorp|Nexa|Fold|X1|Ultra|A1[45]G?)\b", "", r.original_query)
        q = re.sub(r"^\s*\d+\.\s*", "", q).strip(' "')
        q = "Customer says: " + re.sub(r"\s+", " ", q)
        out.append({"id": f"unseen_mut_{len(out) + 1}", "family": "mutated_kit", "query": q,
                    "siis_response": {"title": new_title, "content": content}})
    return out


def from_catalog(rnd: random.Random, n: int) -> list[dict]:
    cat = load_catalog()
    pool = [e for e in cat.entries if e.get("originalType") in ("onURL", "offURL")
            and family(e) == "mobile" and e.get("validation")]
    rnd.shuffle(pool)
    out, used = [], set()
    for e in pool:
        key = e["validation"]["key"]
        if key in used or len(key) > 40:
            continue
        used.add(key)
        on = e["originalType"] == "onURL"
        screen = screen_for(e)
        title = f"{'Turn on' if on else 'Turn off'} {key} on your smartphone"
        content = PREFIX.format(title=title) + "\n".join([
            f"# {title}",
            f"{e.get('qna_description') or e['description']}",
            f"## Change the {key} setting",
            f"Navigate to Settings, tap {screen}, and then tap the switch next to {key} to "
            f"{'turn it on' if on else 'turn it off'}.",
            "## Restart your phone",
            "If the setting does not stay changed, restart your phone. Press and hold the Power button, "
            "and then tap Restart.",
            "## Contact support",
            "If the issue continues, contact the Customer Support Center for further assistance.",
        ])
        q = rnd.choice(COMPLAINTS["on" if on else "off"]).format(k=key)
        out.append({"id": f"unseen_cat_{len(out) + 1}", "family": "catalog_article", "query": q,
                    "siis_response": {"title": title, "content": content},
                    "expected_deeplink": e["id"]})
        if len(out) >= n:
            break
    return out


def build(seed: int = 2026, n_catalog: int = 25) -> list[dict]:
    rnd = random.Random(seed)
    return mutated(load_siis_rows(), rnd) + from_catalog(rnd, n_catalog)


def main() -> None:
    items = build()
    OUT.write_text(json.dumps(items, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(items)} unseen scenarios to {OUT}")


if __name__ == "__main__":
    main()
