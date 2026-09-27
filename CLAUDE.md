# Project: FixPath, Smart Guided Troubleshooting Engine (PRISM Theme 2)

## What we are building
A REST API that turns a vague device complaint plus a SIIS knowledge article into a
structured, deeplinked troubleshooting plan (ContextDeeplinkResponse, see kit/schema.py).
Core idea: compile each article once into a validated guide, serve it from cache, and
call the LLM only for genuinely new input. On top of that sit symptom fingerprints,
sensor-first diagnosis using validation deeplinks, and a digital twin phone for the demo.

## The kit (source of truth, never edit)
- kit/schema.py: Pydantic response models. Import them, never redefine them.
- kit/siis_responses.json: {"_readme","count","responses":[{id, original_query, siis_response:{title, content}}]}
  20 rows (ids row_1..row_22, no row_6 or row_18), 11 distinct articles, mostly display-domain. Some rows pair a query with a
  loosely related article (row_1 email server, row_8 screen mirroring, row_20 rotation).
- kit/deeplinks.json: {"_readme","count","deeplinks":[{id, deeplink, description, message,
  originalType, control_type, qna_description, validation:{deeplink, key}}]}
  578 entries, 570 with validation. URIs are masked: voiceassist://masked/act/<hex> and
  voiceassist://masked/val/<hex>. The only generic placeholder is voiceassist://dummy_positive.
- kit/sample_output.json: one reference output. kit/input.txt: the 20 queries.

## Hard rules (every one is scored; enforce in code, never trust the prompt)
1. Output is pure JSON matching kit/schema.py. No markdown, no prose.
2. goal: "Follow these steps to perform this <Topic> Troubleshooting" or "... Configuration".
   Trailing full stop is configurable (GOAL_TRAILING_PERIOD, default false, matches sample_output.json).
3. title: 2 to 3 words, sentence case.
4. actionName: Title Case. One action = one physical screen. Steps on the same screen share one action.
5. description: exactly 5 to 7 words (split on whitespace), must start with "It will".
6. steps: imperative, one physical interaction each, non-empty, no URLs.
7. category: auto = has actionableDeeplink (mandatory). manual = physical work, MUST NOT carry
   an actionableDeeplink. critical = factory reset, restart, firmware update, safe mode, MUST be last.
   Order: settings toggles, then system optimisations, then reboots/critical.
8. actionableDeeplink: copied verbatim from deeplinks.json (deeplink, description, message, originalType only).
   validationDeeplink: copied from the same entry's validation (deeplink, key). Never invent or edit a URI.
   The LLM never sees or writes URIs, only catalog IDs like DL-0042.
9. Zero URL leaks in human text fields: no http, https, www., .com, .html, markdown links.
   The URL scrubber must NOT touch deeplink fields (they legitimately contain voiceassist://).
10. No invented steps. Every step must trace back to a sentence in the SIIS text.
    If nothing in the article applies, return {"contexts": [], "fallback": "no_match"}.
11. score: float in [0, 1], computed deterministically (see app/scoring.py), not guessed by the LLM.
12. Same input must give the same output (temperature 0, cache, sorted tie-breaks).

## Performance targets
- Repeat query p95 <= 300 ms, cache hit rate >= 90%.
- Paraphrase cache hit rate >= 80%.
- Cold (new) query p95 <= 8 s. Hard LLM timeout 6 s, then deterministic fallback.
- GET /health returns {"status":"ok"} only when cache, indexes and model client are ready.

## Stack
Python 3.11, FastAPI, Uvicorn, Pydantic v2, google-genai SDK (Gemini), rank_bm25,
sentence-transformers (small CPU model, e.g. all-MiniLM-L6-v2 or bge-small), numpy,
pytest, httpx. No database in the MVP: JSON guide store on disk + in-memory cache.
Model names come from env vars (GEMINI_MODEL_RUNTIME, GEMINI_MODEL_COMPILE); check current
names in the official Gemini docs before hardcoding anything.

## Conventions
- Every module has unit tests. Run `pytest -q` before saying a task is done.
- Never commit secrets. Keys only via .env (in .gitignore) and .env.example.
- Log one JSON line per request: path, cache_tier, latency_ms, model, tokens, cost_usd, validator_fixes.
- Extras (fingerprint, diagnosis, provenance, twin) live in `meta` or separate endpoints.
  The scored response body must stay schema-pure.
- When a rule in the organiser material conflicts, the kit files win, then the written
  rules in the Theme 2 guide, then the FAQ. Record every such decision in docs/DECISIONS.md.
