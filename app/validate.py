"""Enforces every hard rule on a response and repairs what can be repaired deterministically.

validate_and_repair(response_dict, siis_text, catalog) -> (fixed_dict, report)

Nothing here trusts the LLM. Each repair is recorded in report["fixes"] so the score
and the logs can see how much the validator had to change.
"""
from __future__ import annotations

import copy
import re
from typing import Any, Optional

from rapidfuzz import fuzz, utils

from app.config import settings
from app.kit import DUMMY_DEEPLINK, Catalog, load_schema
from app.scoring import compute_score

GOAL_RE = re.compile(
    r"^\s*follow these steps to perform this\s+(?P<topic>.+?)\s+"
    r"(?P<kind>troubleshooting|configuration)\s*\.?\s*$",
    re.IGNORECASE,
)
CONFIG_HINT_RE = re.compile(r"\b(configur\w*|set ?up|customi[sz]\w*|enable|turn on)\b", re.I)

CRITICAL_RE = re.compile(
    r"\b(factory (data )?reset|reset to factory|restart|reboot|safe mode|firmware|"
    r"software update|system update|wipe|master reset|hard reset)\b",
    re.IGNORECASE,
)

# URL scrubber: text fields only, never deeplink fields.
MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
URL_RE = re.compile(
    r"(?:https?://\S+|www\.\S+|\b[\w.-]+\.(?:com|html?|org|net|io|co)\b\S*)",
    re.IGNORECASE,
)

SMALL_WORDS = {
    "a", "an", "and", "as", "at", "but", "by", "for", "in", "nor", "of",
    "on", "or", "the", "to", "via", "with", "your",
}
TRAILING_STOP = SMALL_WORDS | {
    "up", "from", "into", "then", "so", "is", "are", "this", "that", "between", "across",
    "about", "over", "after", "before", "through", "when", "while", "if", "using", "where",
    "which", "who", "like", "nearest", "any", "all", "some",
}
# Dropped (after "It will") before truncating, so long descriptions keep their meaning.
DESC_FILLER = {"you", "your", "the", "a", "an", "please", "just", "that", "simply", "any"}
DESC_MIN, DESC_MAX = 5, 7
DESC_PAD = ["on", "your", "device"]
ES_VERB_ENDINGS = ("ches", "shes", "sses", "xes", "zes")

CATEGORY_RANK = {"auto": 0, "manual": 1, "critical": 2}
VALID_CATEGORIES = set(CATEGORY_RANK)


# --------------------------------------------------------------------------- text helpers


def scrub_urls(text: str) -> str:
    text = MD_LINK_RE.sub(r"\1", text)
    text = URL_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def _keep_case(word: str) -> bool:
    """Acronyms and mixed-case brand words (USB, TechCorp, Wi-Fi) keep their casing."""
    letters = [c for c in word[1:] if c.isalpha()]
    return any(c.isupper() for c in letters)


def _uncap(text: str) -> str:
    """ALL-CAPS input carries no casing signal, so treat it as lower case."""
    return text.lower() if text.isupper() else text


def title_case(text: str) -> str:
    words = _uncap(text).split()
    out = []
    for i, w in enumerate(words):
        if _keep_case(w):
            out.append(w)
        elif 0 < i < len(words) - 1 and w.lower() in SMALL_WORDS:
            out.append(w.lower())
        else:
            out.append("-".join(p[:1].upper() + p[1:].lower() for p in w.split("-")))
    return " ".join(out)


def sentence_case(text: str) -> str:
    words = _uncap(text).split()
    out = []
    for i, w in enumerate(words):
        if _keep_case(w):
            out.append(w)
        elif i == 0:
            out.append(w[:1].upper() + w[1:].lower())
        else:
            out.append(w.lower())
    return " ".join(out)


def _strip_punct(text: str) -> str:
    return text.strip().strip(".!?,;:").strip()


def _base_verb(word: str) -> str:
    """'Opens' -> 'open', 'Fixes' -> 'fix' so 'It will' reads correctly."""
    w = word.lower()
    if w.endswith(ES_VERB_ENDINGS):
        return w[:-2]
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


# --------------------------------------------------------------------------- field rules


def fix_goal(goal: Any, title: str) -> str:
    period = "." if settings.goal_trailing_period else ""
    text = scrub_urls(goal) if isinstance(goal, str) else ""
    m = GOAL_RE.match(text)
    if m:
        topic, kind = m.group("topic"), m.group("kind").title()
    else:
        topic = title or text or "Device"
        kind = "Configuration" if CONFIG_HINT_RE.search(text) else "Troubleshooting"
    topic = title_case(_strip_punct(topic))
    return f"Follow these steps to perform this {topic} {kind}{period}"


def fix_title(title: Any, fallback_topic: str = "") -> str:
    text = scrub_urls(title) if isinstance(title, str) else ""
    words = _strip_punct(text).split() or _strip_punct(fallback_topic).split()
    words = words[:3]
    while len(words) > 2 and words[-1].lower() in TRAILING_STOP:
        words.pop()
    if not words:
        words = ["Device", "troubleshooting"]
    elif len(words) == 1:
        words.append("issues")
    return sentence_case(" ".join(words))


def fix_action_name(name: Any) -> str:
    text = _strip_punct(scrub_urls(name)) if isinstance(name, str) else ""
    return title_case(text) if text else "Follow Steps"


def fix_description(desc: Any, action_name: str) -> str:
    text = scrub_urls(desc) if isinstance(desc, str) else ""
    text = re.sub(r"(?<=\w)-(?=\w)", " ", text)  # hyphens: same count for any tokenizer
    text = re.sub(r"[,;:()\"]", " ", text)
    words = _strip_punct(text).split()

    if len(words) >= 2 and words[0].lower() == "it" and words[1].lower() == "will":
        words = ["It", "will"] + words[2:]
    else:
        if words and words[0].lower() in {"it", "this", "will"}:
            words = words[1:]
        if words and words[0].lower() == "will":
            words = words[1:]
        if words:
            words[0] = _base_verb(words[0])
        words = ["It", "will"] + words

    if len(words) > DESC_MAX:
        head, tail = words[:3], words[3:]
        while len(head) + len(tail) > DESC_MAX and any(w.lower() in DESC_FILLER for w in tail):
            tail.remove(next(w for w in tail if w.lower() in DESC_FILLER))
        words = (head + tail)[:DESC_MAX]
        while len(words) > DESC_MIN and words[-1].lower() in TRAILING_STOP:
            words.pop()
    if len(words) < DESC_MIN:
        present = {w.lower() for w in words}
        pool = ["help"] if len(words) == 2 else []
        pool += [w.lower() for w in action_name.split() if w.lower() not in present]
        for w in pool + DESC_PAD:
            if len(words) >= DESC_MIN:
                break
            words.append(w)
    return " ".join(words)


def is_critical(action_name: str, steps: list[str]) -> bool:
    return bool(CRITICAL_RE.search(action_name)) or any(CRITICAL_RE.search(s) for s in steps)


# --------------------------------------------------------------------------- provenance


def siis_sentences(siis_text: str) -> list[str]:
    out: list[str] = []
    for line in siis_text.splitlines():
        line = re.sub(r"^\s*#+\s*", "", line).strip()
        if not line:
            continue
        out.append(line)
        parts = re.split(r"(?<=[.!?])\s+", line)
        if len(parts) > 1:
            out.extend(p.strip() for p in parts if p.strip())
    return out


def provenance_ratio(step: str, sentences: list[str]) -> float:
    if not sentences:
        return 0.0
    return max(
        fuzz.token_set_ratio(step, s, processor=utils.default_process) for s in sentences
    )


# --------------------------------------------------------------------------- deeplinks


def _compact(d: Optional[dict]) -> Optional[dict]:
    if d is None:
        return None
    return {k: v for k, v in d.items() if v is not None}


def _expected_validation(entry: dict) -> Optional[dict]:
    val = Catalog.validation(entry)
    if val is None:
        return None
    if settings.validation_expected_values and entry.get("originalType") in ("onURL", "offURL"):
        val.update(
            resultType="boolean",
            condition="equal",
            value="True" if entry["originalType"] == "onURL" else "False",
        )
    return val


def _canonical_actionable(given: dict, entry: dict) -> dict:
    canon = Catalog.actionable(entry)
    if entry["deeplink"] == DUMMY_DEEPLINK:
        # The catalog asks us to write description and message for the placeholder.
        for f in ("description", "message"):
            v = given.get(f)
            if isinstance(v, str) and scrub_urls(v):
                canon[f] = scrub_urls(v)
    return canon


def _ref(obj: Any) -> Optional[str]:
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return obj.get("deeplink") or obj.get("id")
    return None


# --------------------------------------------------------------------------- main entry


class _Report:
    def __init__(self) -> None:
        self.fixes: list[dict] = []
        self.dropped_steps: list[dict] = []
        self.errors: list[str] = []
        self.steps_total = 0
        self.steps_kept = 0

    def fix(self, path: str, rule: str, before: Any = None, after: Any = None) -> None:
        self.fixes.append({"path": path, "rule": rule, "before": before, "after": after})


def _fix_group(
    group: Any, path: str, catalog: Catalog, sentences: Optional[list[str]], rep: _Report
) -> Optional[dict]:
    if not isinstance(group, dict):
        rep.fix(path, "group_not_object", before=group)
        return None

    steps: list[str] = []
    for j, raw in enumerate(group.get("steps") or []):
        spath = f"{path}.steps[{j}]"
        if not isinstance(raw, str):
            rep.fix(spath, "step_not_string", before=raw)
            continue
        step = scrub_urls(raw)
        if step != raw.strip():
            rep.fix(spath, "url_scrubbed", before=raw, after=step)
        if not step:
            continue
        if step in steps:
            rep.fix(spath, "duplicate_step", before=step)
            continue
        rep.steps_total += 1
        if sentences is not None:
            ratio = provenance_ratio(step, sentences)
            if ratio < settings.provenance_min_ratio:
                rep.dropped_steps.append({"path": spath, "step": step, "best_ratio": ratio})
                rep.fix(spath, "provenance_failed", before=step)
                continue
        rep.steps_kept += 1
        steps.append(step)

    if not steps:
        rep.fix(path, "empty_group_removed")
        return None

    actionable, validation = None, None
    act_ref = _ref(group.get("actionableDeeplink"))
    if act_ref:
        entry = catalog.get(act_ref)
        if entry is None:
            rep.fix(f"{path}.actionableDeeplink", "unknown_deeplink_removed", before=act_ref)
        else:
            given = group["actionableDeeplink"] if isinstance(group["actionableDeeplink"], dict) else {}
            actionable = _canonical_actionable(given, entry)
            if _compact(given) != _compact(actionable):
                rep.fix(f"{path}.actionableDeeplink", "deeplink_canonicalised",
                        before=given or act_ref, after=actionable)
            validation = _expected_validation(entry)
    else:
        val_ref = _ref(group.get("validationDeeplink"))
        entry = catalog.by_validation.get(val_ref) if val_ref else None
        if entry is not None:
            validation = _expected_validation(entry)

    given_val = group.get("validationDeeplink")
    if _compact(given_val if isinstance(given_val, dict) else None) != _compact(validation):
        rep.fix(f"{path}.validationDeeplink", "validation_canonicalised",
                before=given_val, after=validation)

    return {"steps": steps, "actionableDeeplink": actionable, "validationDeeplink": validation}


def _fix_action(
    action: Any, path: str, catalog: Catalog, sentences: Optional[list[str]], rep: _Report
) -> Optional[dict]:
    if not isinstance(action, dict):
        rep.fix(path, "action_not_object", before=action)
        return None

    groups = []
    for k, g in enumerate(action.get("stepGroups") or []):
        fixed = _fix_group(g, f"{path}.stepGroups[{k}]", catalog, sentences, rep)
        if fixed:
            groups.append(fixed)
    if not groups:
        rep.fix(path, "empty_action_removed", before=action.get("actionName"))
        return None

    name = fix_action_name(action.get("actionName"))
    if name != action.get("actionName"):
        rep.fix(f"{path}.actionName", "action_name_title_case", action.get("actionName"), name)

    desc = fix_description(action.get("description"), name)
    if desc != action.get("description"):
        rep.fix(f"{path}.description", "description_rule", action.get("description"), desc)

    category = _fix_category(action.get("category"), name, groups, path, rep)
    return {"actionName": name, "description": desc, "stepGroups": groups, "category": category}


def _fix_category(given: Any, name: str, groups: list[dict], path: str, rep: _Report) -> str:
    cat = given.value if hasattr(given, "value") else given
    cat = cat.lower().strip() if isinstance(cat, str) else None
    has_link = any(g["actionableDeeplink"] for g in groups)
    all_steps = [s for g in groups for s in g["steps"]]

    if cat not in VALID_CATEGORIES:
        new = "critical" if is_critical(name, all_steps) else ("auto" if has_link else "manual")
        rep.fix(f"{path}.category", "category_inferred", given, new)
        cat = new
    if cat != "critical" and is_critical(name, all_steps):
        rep.fix(f"{path}.category", "critical_promoted", cat, "critical")
        cat = "critical"
    if cat == "auto" and not has_link:
        rep.fix(f"{path}.category", "auto_without_deeplink_downgraded", "auto", "manual")
        cat = "manual"
    if cat == "manual" and has_link:
        for k, g in enumerate(groups):
            if g["actionableDeeplink"] or g["validationDeeplink"]:
                rep.fix(f"{path}.stepGroups[{k}]", "manual_deeplink_stripped",
                        before=g["actionableDeeplink"])
                g["actionableDeeplink"] = None
                g["validationDeeplink"] = None
    return cat


def _merge_same_screen(actions: list[dict], path: str, rep: _Report) -> list[dict]:
    merged: dict[str, dict] = {}
    for a in actions:
        key = a["actionName"].lower()
        if key not in merged:
            merged[key] = a
            continue
        base = merged[key]
        base["stepGroups"].extend(a["stepGroups"])
        cats = {base["category"], a["category"]}
        if "critical" in cats:
            base["category"] = "critical"
        elif any(g["actionableDeeplink"] for g in base["stepGroups"]):
            base["category"] = "auto"
        rep.fix(path, "same_screen_actions_merged", before=a["actionName"])
    return list(merged.values())


def _order_actions(actions: list[dict], path: str, rep: _Report) -> list[dict]:
    ordered = sorted(actions, key=lambda a: CATEGORY_RANK[a["category"]])  # stable
    if [a["actionName"] for a in ordered] != [a["actionName"] for a in actions]:
        rep.fix(path, "actions_reordered",
                before=[a["actionName"] for a in actions],
                after=[a["actionName"] for a in ordered])
    return ordered


def validate_and_repair(
    response_dict: Any,
    siis_text: Optional[str],
    catalog: Catalog,
    retrieval_confidence: Optional[float] = None,
) -> tuple[dict, dict]:
    """Return (fixed_dict, report). fixed_dict always validates against the kit schema.

    siis_text=None skips the provenance check (recorded in the report).
    retrieval_confidence, when given, replaces each goal score with the deterministic score.
    """
    rep = _Report()
    src = copy.deepcopy(response_dict) if isinstance(response_dict, dict) else {}
    if not isinstance(response_dict, dict):
        rep.errors.append("response is not a JSON object")
    sentences = siis_sentences(siis_text) if siis_text else None

    goals_out = []
    for i, goal in enumerate(src.get("contexts") or []):
        gpath = f"contexts[{i}]"
        if not isinstance(goal, dict):
            rep.fix(gpath, "goal_not_object", before=goal)
            continue
        fixes_before = len(rep.fixes)
        kept_before, total_before = rep.steps_kept, rep.steps_total

        actions = []
        for a_i, action in enumerate(goal.get("actions") or []):
            fixed = _fix_action(action, f"{gpath}.actions[{a_i}]", catalog, sentences, rep)
            if fixed:
                actions.append(fixed)
        if not actions:
            rep.fix(gpath, "empty_goal_removed", before=goal.get("goal"))
            continue
        actions = _merge_same_screen(actions, f"{gpath}.actions", rep)
        actions = _order_actions(actions, f"{gpath}.actions", rep)

        m = GOAL_RE.match(goal.get("goal") or "") if isinstance(goal.get("goal"), str) else None
        title = fix_title(goal.get("title"), fallback_topic=m.group("topic") if m else "")
        if title != goal.get("title"):
            rep.fix(f"{gpath}.title", "title_rule", goal.get("title"), title)
        goal_text = fix_goal(goal.get("goal"), title)
        if goal_text != goal.get("goal"):
            rep.fix(f"{gpath}.goal", "goal_template", goal.get("goal"), goal_text)

        total = rep.steps_total - total_before
        coverage = (rep.steps_kept - kept_before) / total if total else 0.0
        if retrieval_confidence is not None:
            score = compute_score(retrieval_confidence, coverage, len(rep.fixes) - fixes_before)
        else:
            raw = goal.get("score")
            score = float(raw) if isinstance(raw, (int, float)) and not isinstance(raw, bool) else 0.0
            clamped = max(0.0, min(1.0, score))
            if clamped != raw:
                rep.fix(f"{gpath}.score", "score_clamped", raw, clamped)
            score = clamped

        goals_out.append({"goal": goal_text, "title": title, "score": score, "actions": actions})

    fixed: dict = {"contexts": goals_out}
    if not goals_out:
        fixed["fallback"] = "no_match"

    schema = load_schema()
    try:
        schema.ContextDeeplinkResponse.model_validate(fixed)
        valid = True
    except Exception as exc:  # pragma: no cover - should be unreachable after repair
        rep.errors.append(f"schema: {exc}")
        valid = False

    report = {
        "valid": valid,
        "fixes": rep.fixes,
        "fix_count": len(rep.fixes),
        "dropped_steps": rep.dropped_steps,
        "provenance": {
            "checked": sentences is not None,
            "steps_total": rep.steps_total,
            "steps_kept": rep.steps_kept,
            "coverage": round(rep.steps_kept / rep.steps_total, 4) if rep.steps_total else 0.0,
        },
        "fallback": fixed.get("fallback"),
        "errors": rep.errors,
    }
    return fixed, report
