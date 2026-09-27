# FixPath: Claude Code build plan (Samsung PRISM Gen AI Hackathon, Theme 2)

How to use this file:

1. Create an empty repo, copy the `Theme 2/` kit folder into it as `kit/`, and put this file in the repo root.
2. Save **Part A** as `CLAUDE.md` in the repo root. Claude Code reads it automatically on every session.
3. Paste **Part B** as your first message to Claude Code.
4. Then run the phases in **Part C** one at a time. Paste each phase prompt, let it finish, and check its acceptance tests before moving on.
5. **Part D** lists what will go wrong later and how the plan already handles it.

Deadline: **20 October 2026**. Freeze features on 15 October and spend the rest on the video, deck and hardening.

---

## Part A: `CLAUDE.md` (save this in the repo root)

```markdown
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
  20 rows, only 9 distinct articles, all display-domain. Some rows pair a query with a
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
```

---

## Part B: kickoff prompt (paste as your first message)

```text
Read CLAUDE.md and everything in kit/. Do not write feature code yet.

1. Summarise the kit back to me: schema fields, deeplink catalog fields and counts,
   the 20 rows, how many distinct SIIS articles, and any conflicts between
   sample_output.json and the rules in CLAUDE.md.
2. Propose the repo layout below, adjusted if you see a better one, and list every
   file you will create.
3. Write docs/DECISIONS.md with the conflicts you found and the choice for each.
4. Set up the skeleton: pyproject or requirements.txt, app/ package, tests/, .env.example,
   .gitignore, Makefile targets (install, test, run, compile, score, bench).
5. Stop and show me the plan before Phase 1.

Target layout:
app/
  main.py          FastAPI app, /health, /v1/troubleshoot, /v1/verify, /v1/twin/*, /metrics
  kit.py           loads schema, catalog and SIIS rows once
  catalog.py       deeplink index: BM25 + embeddings, candidates(text, k)
  segment.py       SIIS text -> sections, sentences, imperative step candidates, Settings paths
  fingerprint.py   complaint -> symptom fingerprint (component, symptom, trigger, condition, onset)
  extract.py       single Gemini call with structured output: goals/actions/steps + chosen catalog IDs
  fallback.py      rules-only plan builder (no LLM)
  validate.py      every hard rule, plus auto-repair, plus provenance check
  scoring.py       deterministic score
  cache.py         tier 1 exact key, tier 2 fingerprint key, tier 3 embedding similarity
  store.py         compiled guide store (JSON on disk), warm-up at boot
  diagnose.py      sensor-first diagnosis using validation keys (optional device_state)
  settings_map.py  Settings menu graph from SIIS paths + catalog (stretch)
  llm.py           thin provider interface (Gemini now, Mistral later), timeout, retries, cost
compiler/
  compile.py       offline: compile all articles, paraphrases, write store + results.jsonl
  paraphrase.py    generate 15, keep 10 most diverse (max marginal relevance)
  harvest.py       community wording by symptom (optional, offline)
  unseen.py        synthesise unseen SIIS articles for testing
eval/
  judge_sim.py     replays the judges' probes against a running server
  metrics.py       writes metrics.md (Appendix C format) incl. ablation table
twin/
  index.html       digital twin phone, generated from deeplinks.json
tests/
docs/DECISIONS.md, README.md, Dockerfile, results.jsonl, metrics.md
```

---

## Part C: phases (paste one at a time)

Each phase ends with acceptance tests. Do not move on until they pass.

### Phase 1: Contract and validator first (day 1 to 2)

```text
Build app/kit.py, app/validate.py, app/scoring.py and tests.

validate.py must expose validate_and_repair(response_dict, siis_text, catalog) -> (fixed_dict, report).
Checks and repairs, each with a unit test using a deliberately broken input:
- schema validation via kit/schema.py ContextDeeplinkResponse
- goal template (configurable trailing period), title 2 to 3 words sentence case,
  actionName Title Case, description 5 to 7 words starting "It will" (rewrite or trim deterministically)
- auto must have actionableDeeplink, else downgrade to manual; manual must not have one, strip it
- critical actions moved to the end; stable ordering toggles -> optimisations -> critical
- every deeplink exists verbatim in the catalog; copy only schema fields from the entry
- URL scrubber on text fields only (never on deeplink fields)
- provenance: each step must fuzzy-match (rapidfuzz token_set_ratio >= 80, tune later)
  a sentence in siis_text; drop steps that fail and record them in the report
- empty actions/goals removed; if nothing survives return {"contexts": [], "fallback": "no_match"}
scoring.py: score = weighted mix of retrieval confidence, provenance coverage, validator fixes (fewer fixes = higher).
Acceptance: pytest passes; kit/sample_output.json passes validation after repair
(report lists the fixes, e.g. its 12-word description gets trimmed).
```

### Phase 2: Understand the article and the catalog (day 2 to 4)

```text
Build app/segment.py and app/catalog.py.

segment.py:
- strip the category prefix at the start of content ("Smartphone,Others Mobile,... :")
- split on markdown headings (#, ##, "Step N") into sections
- split sections into sentences; tag imperative ones (Tap, Go to, Navigate, Select, Open,
  Turn on/off, Swipe, Press, Clear, Restart, Contact...)
- extract Settings paths ("Settings > Display > Screen mode", "go to Settings, tap Apps, then...")
  into ordered lists of screen names
- tag critical sentences (factory reset, restart, safe mode, firmware/software update, wipe)
- tag manual sentences (service centre, clean the port, replace, repair, contact support)

catalog.py:
- index text = description + message + qna_description (+ originalType as a token)
- BM25 + embeddings, hybrid score (tune weights later), candidates(text, k=5) -> [(id, score)]
- ignore obviously irrelevant device families at query time (e.g. air conditioner entries)
Acceptance: for each of the 9 distinct articles print sections, step candidates and the
top 3 catalog candidates per step. I review the printout by eye.
```

### Phase 3: Plan builder: LLM path plus rules fallback (day 4 to 7)

```text
Build app/llm.py, app/extract.py, app/fallback.py.

extract.py makes ONE structured-output Gemini call (temperature 0, JSON schema enforced):
input = query, cleaned sections with sentence IDs, and for each step candidate its top-5
catalog candidates shown as {id, message, description} (never URIs).
output = goals -> actions -> step groups where every step is a sentence ID (plus an optional
light rewrite) and every deeplink is a catalog ID or null. The model also marks category.
Map IDs back to text and catalog entries in code, then run validate_and_repair.

fallback.py builds a plan with no LLM: group imperative sentences by the screen named in
their path, one action per screen, best catalog candidate above DEEPLINK_MIN_SCORE or
dummy_positive, categories from the segment tags.

llm.py: timeout LLM_TIMEOUT_S (default 6), one retry on 5xx, token counting and cost_usd.
Acceptance: all 20 kit rows produce schema-valid output through both paths; print a
side-by-side comparison; rows 1, 8 and 20 either keep only relevant steps or return no_match
with a reason in meta.
```

### Phase 4: API, cache and compile-ahead (day 7 to 9)

```text
Build app/fingerprint.py, app/cache.py, app/store.py, app/main.py, compiler/compile.py.

fingerprint.py: extract {component, symptom, trigger, condition, onset} with a tiny LLM call
or rules, normalise to a canonical string. Strip anonymised device names (TechCorp, Nexa X1,
Fold X1, A14/A15, Ultra) before fingerprinting. Split multi-complaint queries (row_19) into
several fingerprints -> several goals.

cache.py tiers: (1) sha256 of normalised query + siis hash, (2) fingerprint + siis hash,
(3) embedding cosine >= CACHE_SIM_THRESHOLD on the normalised query when siis matches or is
absent. Everything in memory, rebuilt from the store at boot.

main.py:
- POST /v1/troubleshoot {query, siis_response?}; if siis_response missing, look up the
  store by fingerprint/embedding (the guide requires this)
- response body = ContextDeeplinkResponse JSON (+ "fallback" when no_match)
- latency, cache_hit, model, cost_usd, fingerprint, provenance go into a `meta` field
  (check Appendix B of the guide: meta sits beside response in results.jsonl; for the live
  API put meta in a response header X-FixPath-Meta AND an optional ?debug=1 body field,
  so the scored body stays clean. Record this in DECISIONS.md.)
- GET /health only ok after warm-up
compile.py: compile all 20 rows, write the store and warm the cache at boot.
Acceptance: make run; curl each kit query twice; second call < 50 ms locally; /health ok.
```

### Phase 5: Paraphrases, results.jsonl and the judge simulator (day 9 to 11)

```text
Build compiler/paraphrase.py, eval/judge_sim.py, compiler/unseen.py.

paraphrase.py: for each query generate 15 variants across registers (formal, casual,
keyword-only, frustrated, typo-inclusive, Hinglish optional), keep 10 by maximal marginal
relevance, dedupe, enforce 8 to 10. Also pre-cache ~50 extra paraphrases per article.
compile.py writes results.jsonl: {query, query_variations, response, meta} per line,
all 20 rows (G3 needs >= 95% coverage).

judge_sim.py replays the judges' probes against a URL: /health, cold canonical queries,
exact repeats, held-out paraphrases (never pre-cached), unseen SIIS from unseen.py.
Reports p50/p95 per path, cache hit rates, schema-valid %, URL leaks, auto-without-deeplink,
manual-with-deeplink, description rule failures.
unseen.py: create 30+ unseen articles by mutating kit articles and by writing articles
around catalog entries; keep them out of the store.
Acceptance: judge_sim on localhost shows every gate passing and targets met. Add it to CI.
```

### Phase 6: Diagnosis, verify loop and the digital twin (day 11 to 14)

```text
Build app/diagnose.py, POST /v1/verify, twin/index.html and /v1/twin endpoints.

diagnose.py: optional device_state {validation_key: value} in the request. Keep a
healthy-value table per validation key (docs/healthy_values.json, our documented
assumption since the kit gives no expected values). Actions whose setting is already
healthy are moved down or skipped (recorded in meta.diagnosis). Scored body stays valid.

/v1/verify {action, device_state_after} -> resolved | next_action | escalate.

twin/index.html: one static page generated from deeplinks.json. Shows a phone frame,
a searchable Settings tree, current values per key. Buttons: speak or type a complaint,
read sensors, show plan, Fix it (animates to the screen, flips the value), verify (turns
the validation check green). Talks to the API. No real Samsung branding.
Acceptance: the full loop sense -> diagnose -> act -> verify works in the browser for 3 kit rows.
```

### Phase 7: Settings map (stretch, day 14 to 15)

```text
Build app/settings_map.py: a graph of Settings screens from SIIS paths plus catalog
descriptions ("Opens the X settings page"). When no exact deeplink fits a step, return
the deepest reachable catalog screen plus the remaining taps as steps. Show the route
highlighted in the twin. Skip this phase if Phases 1 to 6 are not solid.
```

### Phase 8: Metrics, packaging and deploy (day 15 to 17)

```text
eval/metrics.py writes metrics.md in the guide's Appendix C format: schema and rule
compliance, step accuracy on hand labels for the 20 rows (label them in eval/labels.json),
latency p50/p95 per path (N >= 30 per path), cost per query, cache efficacy, and the
ablation table: full LLM mapping vs hybrid BM25+embeddings vs pure rules.
Also a known limitations section.

Dockerfile (slim, model weights baked in, non-root), README with one-command setup,
.env.example, deploy notes for Cloud Run with min instances 1 during evaluation.
Acceptance: docker build and run locally, judge_sim passes against the container and
against the deployed URL.
```

### Phase 9: Submission (day 17 to 20)

```text
Checklist script scripts/check_submission.py: README, Dockerfile, results.jsonl (20 lines,
8 to 10 variations each), metrics.md, deck PDF/PPT, video link in README, no secrets in git
history, URL-leak grep clean. Then:
git tag -a PRISM_GENAI_HACKATHON_Y2026 -m "PRISM Gen AI Hackathon Y2026 Final Submission"
git push origin PRISM_GENAI_HACKATHON_Y2026
```

---

## Part D: ten steps ahead (what goes wrong later, already handled)

| # | What will happen | Handled by |
|---|---|---|
| 1 | Judges send paraphrases you never cached | Fingerprint tier + embedding tier, tested with held-out paraphrases in judge_sim |
| 2 | Judges send unseen, long SIIS articles (the kit has one of 9,269 chars) | Segmenter trims to relevant sections before the LLM; 6 s timeout then rules fallback |
| 3 | Judges call with no siis_response | Store lookup by fingerprint/embedding (a stated requirement) |
| 4 | Free-tier rate limits or an outage during the evaluation window | Everything known is cached; fallback path needs no LLM; paid key on standby |
| 5 | Cloud Run cold start breaks /health or the 8 s limit | Min instances 1, weights baked into the image, warm-up before /health says ok |
| 6 | The scorer counts description words differently (punctuation, hyphens) | Validator counts on whitespace and avoids hyphenated words and trailing punctuation |
| 7 | URL scrubber deletes real deeplinks and fails the deeplink score | Scrubber only runs on text fields; unit test proves deeplinks survive |
| 8 | Mismatched article rows get forced, invented-looking steps | Provenance check drops untraceable steps; no_match when nothing is left |
| 9 | Same query gives slightly different plans on two calls | Temperature 0, cache, sorted tie-breaks, deterministic score |
| 10 | Rules turn out to differ from our reading (goal full stop, meta placement) | Config flags + docs/DECISIONS.md; one-line switch, no refactor |
| 11 | API key leaks into the public repo | .env ignored, secret scan in check_submission.py |
| 12 | Demo breaks live in the final round | Record a backup video; twin works offline against a local server |

## Part E: priorities if time runs short

1. Phases 1 to 5 are the score. Never cut them.
2. Phase 6 (diagnosis + twin) is the jury moment. Cut the Settings map before this.
3. Phase 7 is optional.
4. Community harvest (compiler/harvest.py) only if everything above is green.
