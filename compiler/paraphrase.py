"""Query variations: generate ~15 candidates across registers, keep the 10 most diverse.

Registers: formal, casual, keyword-only, frustrated, typo-inclusive, question, how-to,
third person, support ticket, Hinglish, synonym swap. With an LLM configured the candidates
come from the model; otherwise a deterministic template generator is used. Selection is
maximal marginal relevance (MMR) over sentence embeddings: stay on-meaning, maximise spread.
"""
from __future__ import annotations

import random
import re
from typing import Optional

import numpy as np

from app.embed import get_embedder
from app.fingerprint import fingerprint, split_complaints, strip_devices
from app.llm import LLMError, Provider, call_with_timeout

N_CANDIDATES, N_KEEP, N_MIN = 15, 10, 8

PHRASES = {
    # symptom: (casual, question clause, noun phrase, keywords, hinglish)
    "blank": ("screen just goes black", "screen stay black", "a black screen", "black screen display", "screen bilkul black ho gaya hai"),
    "half blank": ("half the screen is dark", "half the display go dark", "a half black display", "half screen black", "aadhi screen black hai"),
    "flicker": ("screen keeps flickering", "screen keep flashing", "a flickering display", "screen flicker flash", "screen flicker kar rahi hai"),
    "crack": ("screen is totally cracked", "screen keep cracking", "a cracked screen", "cracked screen repair", "screen toot gayi hai"),
    "lag": ("touch is super laggy", "touch respond so slowly", "laggy touch input", "touch lag delay", "touch bahut slow hai"),
    "not responding": ("touchscreen stopped responding", "touchscreen not respond", "an unresponsive touchscreen", "touch not working", "touch kaam nahi kar raha"),
    "distorted": ("display looks all distorted", "display look distorted", "a distorted display", "distorted display test", "display distorted dikh raha hai"),
    "small screen": ("screen won't fill the whole display", "screen not fill the display", "a shrunken screen view", "screen size small full", "screen poori nahi bhar rahi"),
    "unwanted overlay": ("has this floating circle stuck on screen", "floating circle keep showing", "a floating shortcut circle", "floating circle remove", "floating circle hatana hai"),
    "no power": ("won't turn on at all", "phone not turn on", "a phone that will not power on", "phone not turning on", "phone on nahi ho raha"),
    "poor visibility": ("screen is barely visible", "screen look so faint", "a barely visible screen", "screen hard to see", "screen pe kuch dikh nahi raha"),
}
TYPO_MAP = {"screen": "scren", "phone": "phoen", "display": "dispaly", "black": "blak", "turn": "trun",
            "tablet": "tablte", "working": "wroking", "because": "becuase", "completely": "completly",
            "flickers": "flikers", "touch": "tuoch", "charger": "chager", "cracked": "craked"}
SYN_MAP = {"screen": "display", "phone": "handset", "black": "dark", "blank": "empty", "completely": "totally",
           "whenever": "every time", "stays": "remains", "turn it on": "power it up", "doesn't": "does not",
           "can't": "cannot", "shows": "displays", "went": "turned"}


def _device(q: str) -> str:
    return "tablet" if re.search(r"\btablet\b", q, re.I) else "phone"


def _clean_base(q: str) -> str:
    q = re.sub(r"^\s*\d+\.\s*", "", q.strip()).strip(' "“”')
    q = strip_devices(q)
    q = re.sub(r"\bMy\s+(?=screen|inner|has|only)", "My phone ", q)
    return re.sub(r"\s+([,.;])", r"\1", re.sub(r"\s+", " ", q)).strip()


def _typo(text: str) -> str:
    out, n = text, 0
    for w, t in TYPO_MAP.items():
        if n < 2 and re.search(rf"\b{w}\b", out, re.I):
            out = re.sub(rf"\b{w}\b", t, out, count=1, flags=re.I)
            n += 1
    return out.lower().rstrip(".") + (" pls help" if n else " plz")


def _synonyms(text: str) -> str:
    out = text
    for a, b in SYN_MAP.items():
        out = re.sub(rf"\b{re.escape(a)}\b", b, out, flags=re.I)
    return out


def template_candidates(query: str) -> list[str]:
    parts = split_complaints(query)
    base = _clean_base("; ".join(p.rstrip(". ") for p in parts) if len(parts) > 1 else parts[0])
    fp = fingerprint(query)
    dev = _device(query)
    casual, qclause, np_, kw, hi = PHRASES.get(fp.symptom, (
        "is acting up", "act up like this", "a device problem", f"{fp.component} problem fix", "problem aa rahi hai"))
    # Only reuse a trigger that reads as a clause ("i plug in the charger"), never fragments.
    trig = f" when {fp.trigger}" if re.match(r"^(i|it|the|my)\b", fp.trigger or "") else ""
    first = re.split(r"(?<=[.;!?])\s+|,\s+(?:so|and|while|but)\s+", base)[0].rstrip(".;")
    lower = first[0].lower() + first[1:] if first else base
    return [
        f"I am experiencing the following issue. {first}.",
        f"hey so my {dev} {casual}{trig}, any idea how to fix it?",
        f"{kw} {dev} {fp.component} fix",
        f"Seriously, my {dev} {casual} AGAIN. This is so annoying, please help!",
        _typo(f"my {dev} {casual}{trig}"),
        f"Why does my {dev}'s {qclause}{trig}?",
        f"How do I fix {np_} on my {dev}?",
        f"My friend's {dev} has the same problem, {re.sub(r'^my ', 'their ', lower)}.",
        f"Support ticket: {dev} shows {np_}{trig}. Need troubleshooting steps.",
        f"Mera {dev} ka {hi}, kya karun?",
        _synonyms(base),
        f"{np_[0].upper() + np_[1:]}{trig} — what should I check first?",
        f"Help! {first}.",
        f"{dev} {fp.symptom} {fp.component} {fp.trigger}".strip(),
        f"Is there a setting to fix {np_}? My {dev} {casual}.",
    ]


def llm_candidates(query: str, provider: Provider, timeout_s: float = 30) -> list[str]:
    schema = {"type": "object", "properties": {"variations": {"type": "array", "items": {"type": "string"}}},
              "required": ["variations"]}
    system = ("Write 15 paraphrases of a device support complaint. Keep the meaning and every symptom. "
              "Vary register: formal, casual, keyword-only, frustrated, typo-inclusive, question, Hinglish. "
              "Remove brand and model names. No URLs. JSON only.")
    res = call_with_timeout(provider, system, query, schema, timeout_s)
    return [v.strip() for v in res.data.get("variations", []) if isinstance(v, str) and v.strip()]


def mmr_select(original: str, candidates: list[str], k: int = N_KEEP, lam: float = 0.55) -> list[str]:
    seen, uniq = {original.strip().lower()}, []
    for c in candidates:
        key = re.sub(r"\s+", " ", c.strip().lower())
        if key and key not in seen and "http" not in key and "www." not in key:
            seen.add(key)
            uniq.append(c.strip())
    if len(uniq) <= k:
        return uniq
    emb = get_embedder()
    vecs = emb.encode(uniq)
    q = emb.encode_one(original)
    rel = vecs @ q
    chosen: list[int] = []
    while len(chosen) < k:
        best, best_i = -1e9, -1
        for i in range(len(uniq)):
            if i in chosen:
                continue
            div = max((float(vecs[i] @ vecs[j]) for j in chosen), default=0.0)
            score = lam * float(rel[i]) - (1 - lam) * div
            if score > best:
                best, best_i = score, i
        chosen.append(best_i)
    return [uniq[i] for i in sorted(chosen)]


def variations(query: str, provider: Optional[Provider] = None) -> tuple[list[str], str]:
    """-> (8-10 variations, source)."""
    source = "templates"
    cands: list[str] = []
    if provider is not None:
        try:
            cands = llm_candidates(query, provider)
            source = f"llm:{provider.name}"
        except LLMError:
            cands = []
    if len(cands) < N_MIN:
        cands = cands + template_candidates(query)
    out = mmr_select(query, cands)
    rnd = random.Random(query)
    pad = template_candidates(query)
    while len(out) < N_MIN and pad:
        c = pad.pop(rnd.randrange(len(pad)))
        if c not in out and c.strip().lower() != query.strip().lower():
            out.append(c)
    return out[:N_KEEP], source
