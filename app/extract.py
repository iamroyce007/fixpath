"""One structured-output LLM call that chooses and groups steps, never writes them.

The model sees sentence ids with their text and, per unit, catalog candidates as
{id, message, description}. It returns ids only. Code maps ids back to verbatim SIIS
steps and catalog entries, so the LLM cannot invent steps or URIs.
"""
from __future__ import annotations

import json
from typing import Optional

from app.catalog import CatalogIndex, extract_targets
from app.fallback import build_units
from app.llm import LLMResult, Provider, call_with_timeout
from app.segment import Article

SYSTEM = (
    "You turn a device complaint and a knowledge article into a troubleshooting plan. "
    "Use ONLY sentence ids from the article; never write new steps. Keep only sentences that help "
    "with the complaint. One action = one physical screen. Order: settings toggles first, then "
    "optimisations, then restarts/resets (category critical). category auto only when you pick a "
    "deeplink_id from that unit's candidates; manual for physical work (no deeplink). "
    "description: 5 to 7 words starting with 'It will'. title: 2 to 3 words, sentence case. "
    "topic: 1 to 3 words in Title Case. Output JSON only."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "goals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "kind": {"type": "string", "enum": ["Troubleshooting", "Configuration"]},
                    "title": {"type": "string"},
                    "actions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "description": {"type": "string"},
                                "category": {"type": "string", "enum": ["auto", "manual", "critical"]},
                                "sentence_ids": {"type": "array", "items": {"type": "string"}},
                                "deeplink_id": {"type": ["string", "null"]},
                            },
                            "required": ["name", "description", "category", "sentence_ids", "deeplink_id"],
                        },
                    },
                },
                "required": ["topic", "kind", "title", "actions"],
            },
        }
    },
    "required": ["goals"],
}


def build_prompt(query: str, article: Article, index: CatalogIndex) -> tuple[str, set[str]]:
    by_id = article.by_id()
    allowed_ids: set[str] = set()
    lines = [f"COMPLAINT: {query}", f"ARTICLE: {article.title}", "UNITS:"]
    for n, unit in enumerate(build_units(article), 1):
        sids = [s.id for s in article.sentences
                if s.steps and s.text in unit.text and any(st in unit.steps for st in s.steps)]
        if not sids:
            continue
        text = " ".join(unit.steps)
        cands = index.candidates(text, k=5, context=unit.text, targets=extract_targets(text, unit.path))
        allowed_ids.update(c.id for c in cands)
        lines.append(f"- unit {n}: {unit.heading}")
        for sid in sids:
            lines.append(f"    {sid}: {by_id[sid].text}")
        lines.append("    deeplink candidates: " + json.dumps([c.public() for c in cands]))
    return "\n".join(lines), allowed_ids


def to_response(data: dict, article: Article, allowed_ids: set[str]) -> dict:
    by_id = article.by_id()
    goals = []
    for g in data.get("goals") or []:
        actions = []
        for a in g.get("actions") or []:
            steps: list[str] = []
            for sid in a.get("sentence_ids") or []:
                s = by_id.get(str(sid).strip())
                if s:
                    for st in s.steps or [s.text]:
                        if st not in steps:
                            steps.append(st)
            if not steps:
                continue
            dl = a.get("deeplink_id")
            link = {"id": dl} if dl in allowed_ids else None
            actions.append({
                "actionName": a.get("name", ""),
                "description": a.get("description", ""),
                "category": a.get("category", "manual"),
                "stepGroups": [{"steps": steps, "actionableDeeplink": link, "validationDeeplink": None}],
            })
        if actions:
            goals.append({
                "goal": f"Follow these steps to perform this {g.get('topic', 'Device')} {g.get('kind', 'Troubleshooting')}",
                "title": g.get("title", ""),
                "score": 0.0,
                "actions": actions,
            })
    return {"contexts": goals}


def llm_plan(query: str, article: Article, index: CatalogIndex, provider: Provider,
             timeout_s: float) -> tuple[dict, LLMResult]:
    prompt, allowed = build_prompt(query, article, index)
    result = call_with_timeout(provider, SYSTEM, prompt, SCHEMA, timeout_s)
    return to_response(result.data, article, allowed), result
