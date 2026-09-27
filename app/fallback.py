"""Rules-only plan builder. No LLM, deterministic, typically well under 100 ms.

article sections -> action units (one per screen/section) -> relevance filter against the
query -> catalog deeplink per unit -> categories -> goals. The result then goes through
validate_and_repair like every other path.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from app.catalog import CatalogIndex, Candidate, extract_targets, get_index
from app.fingerprint import fingerprint, split_complaints, topic_for
from app.kit import DUMMY_DEEPLINK
from app.segment import Article

MAX_ACTIONS = 6
MAX_STEPS = 8
REL_WINDOW = 0.30       # keep units within this similarity of the best one
REL_FLOOR = 0.12        # and never below this
HEADING_PREFIX_RE = re.compile(r"^\s*(?:step\s*\d+\s*[:.)-]\s*|\d+\.\s*)", re.I)
GENERIC_HEADINGS = re.compile(r"^(note|important notes?|overview|introduction|some things)\b", re.I)
TROUBLESHOOT_PREFIX_RE = re.compile(r"^troubleshoot(?:ing)?\s+(?:steps\s+for\s+)?", re.I)
CRITICAL_NAMES = [
    (re.compile(r"factory (data )?reset", re.I), "Perform a Factory Data Reset"),
    (re.compile(r"safe mode", re.I), "Restart in Safe Mode"),
    (re.compile(r"force|volume down", re.I), "Force a Restart"),
    (re.compile(r"update", re.I), "Update Device Software"),
]
ACTION_VERBS = {
    "access", "add", "adjust", "back", "change", "charge", "check", "choose", "clean", "clear",
    "close", "connect", "contact", "create", "customize", "disable", "disconnect", "enable",
    "exit", "fix", "force", "increase", "inspect", "mirror", "open", "perform", "reboot", "remove",
    "reset", "restart", "review", "schedule", "select", "set", "switch", "test", "turn", "update",
    "use", "verify", "visit", "attempt", "back up", "try",
}


@dataclass
class Unit:
    heading: str
    steps: list[str]
    text: str
    path: list[str] = field(default_factory=list)
    critical: bool = False
    manual: bool = False
    sim: float = 0.0
    candidate: Optional[Candidate] = None


def clean_heading(h: str) -> str:
    return HEADING_PREFIX_RE.sub("", h).strip(" :.-")


def build_units(article: Article) -> list[Unit]:
    """One unit per section; a section mixing safe and critical steps is split in two."""
    by_id = article.by_id()
    units: list[Unit] = []
    for sec in article.sections:
        sents = [by_id[i] for i in sec.sentence_ids]
        step_sents = [s for s in sents if s.steps]
        if not step_sents:
            continue
        heading_critical = any(s.critical for s in step_sents) and all(
            s.critical for s in step_sents)
        runs: list[list] = []
        for s in step_sents:
            flag = heading_critical or s.critical
            if runs and runs[-1][0] == flag:
                runs[-1][1].append(s)
            else:
                runs.append([flag, [s]])
        for flag, run in runs:
            steps: list[str] = []
            for s in run:
                for st in s.steps:
                    if st not in steps:
                        steps.append(st)
            heading = clean_heading(sec.heading)
            if flag and len(runs) > 1:
                text = " ".join(s.text for s in run)
                heading = next((n for rx, n in CRITICAL_NAMES if rx.search(text)), "Restart the Device")
            units.append(Unit(
                heading=heading,
                steps=steps[:MAX_STEPS],
                text=" ".join(s.text for s in sents),
                path=next((s.path for s in run if len(s.path) > 1), []),
                critical=flag,
                manual=sum(s.manual for s in run) * 2 >= len(run),
            ))
    return units


def _trim_words(words: list[str], limit: int) -> list[str]:
    from app.validate import TRAILING_STOP

    words = words[:limit]
    while len(words) > 2 and words[-1].lower().strip(".,") in TRAILING_STOP:
        words.pop()
    return words


def _action_name(unit: Unit) -> str:
    h = TROUBLESHOOT_PREFIX_RE.sub("Fix ", unit.heading) if TROUBLESHOOT_PREFIX_RE.match(unit.heading) else unit.heading
    if not h or GENERIC_HEADINGS.match(h) or len(h.split()) > 6:
        if unit.path and len(unit.path) > 1:
            h = f"Open {unit.path[-1]} Settings"
        else:
            pick = next((st for st in unit.steps[:3] if not re.search(r"\b(it|them)\b", st, re.I)),
                        unit.steps[0])
            h = " ".join(_trim_words(pick.rstrip(".").split(), 4))
    return h


def _description(unit: Unit, name: str) -> str:
    """'It will ...' in plain language; the validator enforces 5-7 words."""
    c = unit.candidate if unit.candidate and unit.candidate.accepted else None
    if c:
        entry_key = c.message
        for prefix in ("Enable ", "Disable ", "View ", "Adjust "):
            entry_key = entry_key.removeprefix(prefix)
        if c.originalType == "onURL":
            return f"It will turn on {entry_key.lower()} setting"
        if c.originalType == "offURL":
            return f"It will turn off {entry_key.lower()} setting"
        return f"It will open {entry_key.lower()} settings page"
    words = name.split()
    first = words[0].lower() if words else ""
    if first in ACTION_VERBS or " ".join(words[:2]).lower() in ACTION_VERBS:
        return "It will " + " ".join([first] + [w.lower() for w in words[1:]])
    return "It will help with " + " ".join(_trim_words([w.lower() for w in words], 3))


def _dummy_deeplink(unit: Unit) -> dict:
    screen = unit.path[-1] if unit.path else "relevant Settings"
    return {
        "deeplink": DUMMY_DEEPLINK,
        "description": f"Opens the {screen} screen in Settings",
        "message": f"Open {screen} settings",
    }


def _category_and_links(unit: Unit, index: CatalogIndex) -> tuple[str, Optional[dict]]:
    steps_text = " ".join(unit.steps)
    targets = extract_targets(steps_text, unit.path)
    from app.segment import split_sentences

    target_keys = [t.lower() for t in targets]
    ctx = " ".join(x for x in split_sentences(unit.text)
                   if any(k in x.lower() for k in target_keys)) or steps_text
    cands = index.candidates(steps_text, k=3, context=ctx, targets=targets)
    unit.candidate = cands[0] if cands else None
    accepted = unit.candidate if unit.candidate and unit.candidate.accepted else None

    if unit.critical:
        return "critical", ({"id": accepted.id} if accepted and accepted.key_match >= 0.85 else None)
    if accepted:
        return "auto", {"id": accepted.id}
    if len(unit.path) > 1 and not unit.manual:
        return "auto", _dummy_deeplink(unit)
    return "manual", None


def _relevance(query: str, units: list[Unit], index: CatalogIndex) -> None:
    emb = index.embedder
    q = emb.encode_one(query)
    vecs = emb.encode([f"{u.heading}. {' '.join(u.steps)} {u.text[:400]}" for u in units])
    sims = vecs @ q
    for u, s in zip(units, sims):
        u.sim = float(s)


def build_plan(query: str, article: Article, index: Optional[CatalogIndex] = None) -> tuple[dict, dict]:
    """-> (response_dict, info). info carries retrieval confidence and per-unit trace."""
    index = index or get_index()
    all_units = build_units(article)
    if not all_units:
        return {"contexts": []}, {"retrieval_confidence": 0.0, "reason": "no imperative steps in SIIS"}

    subqueries = split_complaints(query)
    goals, trace, confidences = [], [], []
    used: set[int] = set()
    for sq in subqueries:
        units = [u for u in all_units]
        _relevance(sq, units, index)
        best = max(u.sim for u in units)
        keep = [i for i, u in enumerate(units)
                if u.sim >= max(REL_FLOOR, best - REL_WINDOW) and (len(subqueries) == 1 or i not in used)]
        keep = sorted(keep, key=lambda i: (-units[i].sim, i))[:MAX_ACTIONS]
        keep.sort()  # article order; the validator then sorts by category (stable)
        if not keep:
            continue
        used.update(keep)

        actions = []
        for i in keep:
            u = units[i]
            category, link = _category_and_links(u, index)
            name = _action_name(u)
            actions.append({
                "actionName": name,
                "description": _description(u, name),
                "category": category,
                "stepGroups": [{"steps": u.steps, "actionableDeeplink": link, "validationDeeplink": None}],
            })
            trace.append({"subquery": sq[:80], "unit": u.heading, "sim": round(u.sim, 3),
                          "category": category,
                          "deeplink": (link or {}).get("id") or (link or {}).get("deeplink"),
                          "candidate": u.candidate.id if u.candidate else None,
                          "cand_score": u.candidate.score if u.candidate else None})
        fp = fingerprint(sq)
        topic, title, kind = topic_for(fp, article.title)
        confidences.append(float(np.mean([units[i].sim for i in keep])))
        goals.append({
            "goal": f"Follow these steps to perform this {topic} {kind}",
            "title": title,
            "score": 0.0,
            "actions": actions,
        })

    conf = max(0.0, min(1.0, 0.45 + 0.9 * (sum(confidences) / len(confidences)))) if confidences else 0.0
    return {"contexts": goals}, {"retrieval_confidence": round(conf, 4), "units": trace}
