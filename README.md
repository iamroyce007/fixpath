# FixPath: Smart Guided Troubleshooting Engine

Samsung PRISM Gen AI Hackathon 3.0, Theme 2.

FixPath is a REST API. It turns a vague device complaint plus a SIIS knowledge article into a
structured troubleshooting plan with deeplinks (`ContextDeeplinkResponse`, see `kit/schema.py`).
Each article is compiled once into a validated guide and served from cache. The LLM is called only for genuinely new input.
Every output rule is enforced in code, never trusted to the prompt.

## Status

| Phase | What | State |
|---|---|---|
| 0 | Repo skeleton, kit, decisions | done |
| 1 | Contract: `kit.py`, `validate.py`, `scoring.py` + tests | done |
| 2 | Article segmenter + deeplink catalog retrieval | next |
| 3 | LLM plan builder + rules fallback | |
| 4 | API, cache, compile-ahead | |
| 5 | Paraphrases, results.jsonl, judge simulator | |
| 6 | Diagnosis, verify loop, digital twin | |
| 7 | Settings map (stretch) | |
| 8 | Metrics, Docker, deploy | |
| 9 | Submission | |

See `CLAUDE_CODE_BUILD_PLAN.md` for the full plan and `docs/DECISIONS.md` for rule interpretations.

## Setup

```bash
make install   # creates .venv and installs requirements-dev.txt
cp .env.example .env   # add GEMINI_API_KEY and model names
make test
make run       # http://localhost:8080/health
```

## Layout

```
app/        API and pipeline (kit, validate, scoring, ...)
compiler/   offline compile, paraphrases, unseen articles
eval/       judge simulator, metrics
twin/       digital twin phone (demo)
kit/        organiser kit, never edited
tests/
docs/DECISIONS.md
```
