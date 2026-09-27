# Decisions

Precedence when sources disagree: kit files, then the written rules in the Theme 2 guide,
then the FAQ. Every call below can be changed with a flag or a small edit.

| # | Conflict or gap | Decision | Where |
|---|---|---|---|
| D1 | The build plan says 9 distinct SIIS articles. The kit has **11** (20 rows, ids `row_1` to `row_22`, with no `row_6` or `row_18`). | Trust the kit. Corrected CLAUDE.md. | `tests/test_kit.py` |
| D2 | `sample_output.json` descriptions have 9 and 12 words. The written rule says 5 to 7. | Enforce 5 to 7. The sample only shows the format, so the rule wins here. The validator drops filler words first, then trims, then removes a trailing preposition. | `app/validate.py::fix_description` |
| D3 | The sample `validationDeeplink` has `resultType`, `condition` and `value`. The catalog only has `deeplink` and `key`. | For `onURL`/`offURL` entries, output `boolean`/`equal`/`"True"` or `"False"`, which reproduces the sample exactly. Other entries get only `deeplink` and `key`. Flag: `VALIDATION_EXPECTED_VALUES`. | `_expected_validation` |
| D4 | Trailing full stop on `goal`. | The sample has none. Default is off; flag `GOAL_TRAILING_PERIOD`. | `fix_goal` |
| D5 | None of the sample's steps can be traced to a kit article (its query is not in the kit, and its backup steps are not in the cracked-screen article). | Use the sample as a format reference only. The provenance rule still applies to our own outputs. The acceptance test runs the sample with provenance off. | `tests/test_validate.py` |
| D6 | Title Case for particles: the sample writes "Back **Up** Phone Data". | Capitalise particles (Up, Off). Lowercase only articles, conjunctions and short prepositions in the middle of a name. Keep mixed-case words (TechCorp, Wi-Fi) as they are. | `title_case` |
| D7 | `voiceassist://dummy_positive`: its catalog entry says to write `description` and `message` yourself. | This is the only exception to verbatim copying. The URI and `originalType: placeholder` still come from the catalog, the text is URL-scrubbed, and there is no validation deeplink. | `_canonical_actionable` |
| D8 | A `manual` action that carries a deeplink. | Strip the deeplink and its validation (plan rule 7). Never upgrade the action to `auto`. | `_fix_category` |
| D9 | When an action counts as critical. | Keyword match on the action name or any step: factory reset, restart, reboot, safe mode, firmware, software/system update, wipe. Critical actions are promoted and moved last. | `CRITICAL_RE` |
| D10 | Action order. | A stable sort: `auto`, then `manual`, then `critical`. | `_order_actions` |
| D11 | Scorers may count hyphenated words differently. | Descriptions replace hyphens with spaces, so whitespace and hyphen-splitting tokenizers give the same count. | `fix_description` |
| D12 | The body for no match. | `{"contexts": [], "fallback": "no_match"}`. The extra field is ignored by the schema model. | `validate_and_repair` |
| D13 | Where `meta` goes in the live API. | To be decided in Phase 4. The plan is an `X-FixPath-Meta` header plus a `?debug=1` body field, so the scored body stays clean. | — |
| D14 | The Theme 2 guide PDF is DRM-wrapped (NASCA), so tools cannot read it. | Rules are taken from the build plan. **Check them by hand against the guide**, especially Appendix B/C and the rules for description and goal. | — |
| D15 | The plan targets Python 3.11, but only 3.14 is installed locally. | `requires-python >= 3.11`. Develop on 3.14 locally; the Docker image will pin 3.11. | `Makefile`, Dockerfile (Phase 8) |
