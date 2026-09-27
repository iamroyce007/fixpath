# System Performance Metrics & Evaluation Report
**Model(s):** rules (runtime LLM optional; rules path serves every request when no key is set)
**Embeddings:** sentence-transformers/all-MiniLM-L6-v2
**Environment:** 8 vCPU / arm64 / Darwin 25.5.0 / Python 3.14.3
**Measured:** 2026-09-27 22:47 against in-process

---

## 1. Schema & Rule Compliance
Evaluated on the 20 kit scenarios, 178 live probes and 36 held-out unseen scenarios.

| Metric | Target | Measured Value |
| :--- | :--- | :--- |
| Schema-valid output lines | >= 99% | 100.0% |
| Rule compliance (Goal / Title / Description syntax) | >= 95% | 100.0% (results.jsonl), goal 100.0% / title 100.0% / description 100.0% (live) |
| Absolute URL leaks | 0 | 0 |
| Deeplink catalog validity (exact URI match) | 100% | 100.0% |
| Auto actions carrying valid actionable deeplink | >= 90% | 100.0% |
| Manual actions carrying a deeplink | 0 | 0 |
| Critical actions ordered last | 100% | 100.0% |

---

## 2. Accuracy Benchmarks
Evaluated against hand labels (`eval/labels.json`) for the 20 kit scenarios.

| Evaluation Metric | Scale / Anchor | Score |
| :--- | :--- | :--- |
| Step accuracy (completeness, correctness, ordering) | 0.0 - 3.0 | 2.82 |
| Deeplink relevance (exact target screen vs. parent menu) | 0.0 - 2.0 | 2.00 (labelled screens) |
| Unseen articles: exact catalog entry chosen | - | 92.0% |
| Unseen articles: correct setting, either toggle direction | - | 96.0% |

---

## 3. Latency Benchmarks (N >= 30 requests per path)

| Execution Path | Target (P95) | P50 (ms) | P95 (ms) | N |
| :--- | :--- | :--- | :--- | :--- |
| Cache hit - exact query match | <= 300 ms | 0.72 | 0.86 | 40 |
| Cache hit - unseen semantic paraphrase | <= 300 ms | 0.92 | 7.62 | 40 |
| Paraphrase without siis_response (store lookup) | <= 300 ms | 6.94 | 151.94 | 40 |
| Cold query - full pipeline extraction & mapping | <= 8000 ms | 37.74 | 147.53 | 36 |

---

## 4. Operational Cost & Cache Efficacy

| Metric Item | Target | Measured Value |
| :--- | :--- | :--- |
| Cold query average inference cost | Tracked | $0.0000 (rules path; LLM cost logged per request in meta.cost_usd) |
| Cache hit inference cost | $0.00 | $0.00 |
| Exact repeat cache hit rate | >= 90% | 100.0% |
| Semantic cache hit rate (on unseen paraphrases) | >= 80% | 100.0% |
| Cost derivation method | - | (prompt tokens x PRICE_IN_PER_M + completion tokens x PRICE_OUT_PER_M) / 1e6 |

---

## 5. Architectural Ablation Analysis

| Architecture Variant | Step Accuracy (0-3) | Deeplink Relevance (0-2) | Unseen Exact Deeplink | Latency (P95) | Cost / Query | Key Observations |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Baseline: Full LLM Deeplink Mapping | n/a | n/a | n/a | n/a | n/a | Not run in this build: no LLM key configured (set GEMINI_API_KEY) |
| Variant A: Hybrid BM25 + Dense Embedding Retrieval | 2.82 | 2.00 | 92% | 176 ms | $0.0000 | shipped default |
| Variant B: Pure Rules-Based Deeplink Mapping | 2.82 | 2.00 | 88% | 179 ms | $0.0000 | keyword + exact screen-name match only |

---

## 6. Automated Gates (judge simulator)

| Gate | Result |
| :--- | :--- |
| G2_health | PASS |
| G4_schema_valid_ge_90 | PASS |
| G5_zero_url_leaks | PASS |
| A3_repeat_p95_le_300ms | PASS |
| A3_repeat_hit_rate_ge_90 | PASS |
| A3_paraphrase_hit_rate_ge_80 | PASS |
| A3_cold_p95_le_8s | PASS |
| A4_unseen_non_empty | PASS |
| A2_auto_have_deeplinks | PASS |

---

## 7. Known Limitations
- The kit pairs some complaints with loosely related articles (for example row_8, a small screen, with a TV mirroring article). We never invent steps, so those plans contain only the article's closest steps.
- Hand labels (`eval/labels.json`) were drafted with AI assistance from the SIIS text (see AI_DISCLOSURE.md); they are a sanity check, not an official ground truth.
- Without an LLM key every request runs on the deterministic rules path. The LLM path (Gemini, Mistral or Claude) is implemented and unit-tested with a mock provider, and plugs in by setting an API key.
- The paraphrase hit rate relies on fingerprint and embedding tiers keyed to the same SIIS article; with no article, a paraphrase must be close to a known query or article (75% on held-out phrasings).
- Toggle direction (enable vs. disable) is inferred from wording near the setting name and is the most common mapping error.
