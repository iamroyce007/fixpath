import json

import pytest
from fastapi.testclient import TestClient

from app.catalog import extract_targets, get_index
from app.extract import build_prompt, to_response
from app.fallback import build_plan
from app.fingerprint import fingerprint, normalize, split_complaints, topic_for
from app.kit import load_schema
from app.llm import LLMResult
from app.main import app
from app.segment import atomic_steps, segment, settings_path, strip_prefix
from app.validate import validate_and_repair
from compiler.paraphrase import variations
from compiler.unseen import build as build_unseen


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _payload(row):
    return {"query": row.original_query, "siis_response": {"title": row.title, "content": row.content}}


# ------------------------------------------------------------------ segment


def test_prefix_stripped(rows):
    body = strip_prefix(rows["row_21"].content)
    assert body.startswith("# Troubleshooting Touchscreen Issues")


@pytest.mark.parametrize("sentence,expected", [
    ("Go to Settings, tap Display, and then tap Brightness.", ["Go to Settings.", "Tap Display.", "Tap Brightness."]),
    ("Navigate to Settings, search for and select Factory data reset, and then tap Factory data reset again.",
     ["Navigate to Settings.", "Search for and select Factory data reset.", "Tap Factory data reset again."]),
    ("If you need to change it, you can switch between Portrait or Landscape.", ["Switch between Portrait or Landscape."]),
    ("Obvious cracks on the screen usually result from an impact.", []),
])
def test_atomic_steps(sentence, expected):
    assert atomic_steps(sentence) == expected


def test_settings_path():
    assert settings_path("Go to Settings > Display > Navigation bar") == ["Settings", "Display", "Navigation bar"]
    assert settings_path("Go to Settings, tap Display, and then tap Navigation bar.") == ["Settings", "Display", "Navigation bar"]


def test_every_kit_article_has_steps(rows):
    for r in rows.values():
        art = segment(r.title, r.content)
        assert art.step_sentences(), r.id
        ids = [s.id for s in art.sentences]
        assert len(ids) == len(set(ids))


# ------------------------------------------------------------------ fingerprint


def test_fingerprint_strips_devices_and_is_stable():
    a = fingerprint("My TechCorp Nexa X1 Ultra screen is completely black")
    b = fingerprint("my phone's display went totally black")
    assert a.key == b.key == "screen|blank"
    assert "nexa" not in normalize("My Nexa Fold X1 screen")


def test_split_multi_complaint(rows):
    parts = split_complaints(rows["row_19"].original_query)
    assert len(parts) == 3


def test_topic_and_kind():
    topic, title, kind = topic_for(fingerprint("I want to remove the floating circle on my screen"))
    assert kind == "Configuration" and 2 <= len(title.split()) <= 3


# ------------------------------------------------------------------ catalog


def test_catalog_resolves_named_screen():
    ix = get_index()
    text = "Go to Settings. Tap Display. Tap the switch next to Touch sensitivity to disable it."
    top = ix.candidates(text, 3, targets=extract_targets(text, ["Settings", "Display", "Touch sensitivity"]))[0]
    assert top.message == "Disable Touch sensitivity" and top.accepted


def test_catalog_rejects_weak_matches():
    ix = get_index()
    top = ix.candidates("Press and hold the Power button. Tap Safe mode.", 3)
    assert not top or not top[0].accepted


def test_catalog_never_returns_tv_or_appliance_for_phone_steps():
    ix = get_index()
    for c in ix.candidates("Adjust the brightness of the screen", 10):
        assert "TV" not in c.description and "refrigerator" not in c.description.lower()


# ------------------------------------------------------------------ rules planner


def test_all_kit_rows_valid_and_non_empty(rows, catalog):
    schema = load_schema()
    for r in rows.values():
        plan, info = build_plan(r.original_query, segment(r.title, r.content))
        fixed, rep = validate_and_repair(plan, r.content, catalog, info["retrieval_confidence"])
        schema.ContextDeeplinkResponse.model_validate(fixed)
        assert fixed["contexts"], r.id
        assert rep["provenance"]["coverage"] == 1.0, r.id  # rules path copies SIIS text verbatim


def test_touchscreen_row_gets_deeplinks(rows, catalog):
    r = rows["row_21"]
    plan, info = build_plan(r.original_query, segment(r.title, r.content))
    fixed, _ = validate_and_repair(plan, r.content, catalog, info["retrieval_confidence"])
    msgs = {grp["actionableDeeplink"]["message"] for g in fixed["contexts"] for a in g["actions"]
            for grp in a["stepGroups"] if grp["actionableDeeplink"]}
    assert msgs & {"Enable Touch sensitivity", "Disable Touch sensitivity"}


def test_off_topic_is_no_match(rows, catalog):
    r = rows["row_2"]
    plan, info = build_plan("how do I bake sourdough bread", segment(r.title, r.content))
    fixed, _ = validate_and_repair(plan, r.content, catalog, info["retrieval_confidence"])
    assert fixed == {"contexts": [], "fallback": "no_match"}


# ------------------------------------------------------------------ LLM extract (mock provider)


class FakeProvider:
    name, model = "fake", "fake-1"

    def __init__(self, data):
        self.data = data

    def generate_json(self, system, prompt, schema):
        return LLMResult(data=self.data, model="fake/fake-1", input_tokens=100, output_tokens=50)


def test_llm_ids_map_to_verbatim_text_and_invented_ids_are_dropped(rows, catalog):
    r = rows["row_21"]
    art = segment(r.title, r.content)
    prompt, allowed = build_prompt(r.original_query, art, get_index())
    assert "voiceassist://" not in prompt  # the model never sees URIs
    sid = next(s.id for s in art.sentences if "Touch sensitivity to disable" in s.text)
    data = {"goals": [{"topic": "Touchscreen Lag", "kind": "Troubleshooting", "title": "Touchscreen lag fix",
                       "actions": [
                           {"name": "Turn off touch sensitivity", "description": "It will reduce accidental touches",
                            "category": "auto", "sentence_ids": [sid, "S9999"], "deeplink_id": "DL-0125"},
                           {"name": "Magic fix", "description": "It will fix everything instantly",
                            "category": "auto", "sentence_ids": ["S9999"], "deeplink_id": "DL-0001"}]}]}
    plan = to_response(data, art, allowed)
    fixed, rep = validate_and_repair(plan, r.content, catalog, 0.8)
    actions = fixed["contexts"][0]["actions"]
    assert len(actions) == 1  # the action with only invented ids disappears
    assert rep["provenance"]["coverage"] == 1.0 and not rep["dropped_steps"]
    assert actions[0]["stepGroups"][0]["actionableDeeplink"]["deeplink"] == catalog.by_id["DL-0125"]["deeplink"]


def test_llm_deeplink_outside_candidates_is_rejected(rows):
    r = rows["row_21"]
    art = segment(r.title, r.content)
    sid = art.step_sentences()[0].id
    plan = to_response({"goals": [{"topic": "X", "kind": "Troubleshooting", "title": "X y",
                                   "actions": [{"name": "A", "description": "It will do a thing",
                                                "category": "auto", "sentence_ids": [sid],
                                                "deeplink_id": "DL-0468"}]}]}, art, {"DL-0125"})
    assert plan["contexts"][0]["actions"][0]["stepGroups"][0]["actionableDeeplink"] is None


def test_engine_uses_llm_then_validates(rows, monkeypatch):
    from app.pipeline import Engine

    r = rows["row_21"]
    art = segment(r.title, r.content)
    sid = next(s.id for s in art.sentences if "Navigation bar" in s.text and s.steps)
    eng = Engine()
    eng.index = get_index()
    eng.provider = FakeProvider({"goals": [{"topic": "Touchscreen Lag", "kind": "Troubleshooting",
                                            "title": "Touchscreen lag fix", "actions": [
                                                {"name": "Open Navigation Bar", "description": "Opens nav bar",
                                                 "category": "auto", "sentence_ids": [sid], "deeplink_id": "DL-0169"}]}]})
    body, info = eng._build(r.original_query, art, r.content)
    assert info["path"] == "llm" and info["model"] == "fake/fake-1"
    assert body["contexts"][0]["actions"][0]["description"].startswith("It will")


# ------------------------------------------------------------------ paraphrases and unseen


def test_variations_count_and_uniqueness(rows):
    for r in list(rows.values())[:5]:
        v, source = variations(r.original_query)
        assert 8 <= len(v) <= 10 and len(set(map(str.lower, v))) == len(v)
        assert all("http" not in x for x in v)


def test_unseen_builder_is_deterministic():
    a, b = build_unseen(), build_unseen()
    assert a == b and len(a) >= 30


# ------------------------------------------------------------------ API


def test_troubleshoot_repeat_is_cached_and_identical(client, rows):
    p = _payload(rows["row_4"])
    a = client.post("/v1/troubleshoot", json=p).json()
    b = client.post("/v1/troubleshoot", json=p).json()
    assert b["meta"]["cache_hit"] and a["contexts"] == b["contexts"]
    load_schema().ContextDeeplinkResponse.model_validate(b)
    assert "X-FixPath-Meta" in client.post("/v1/troubleshoot", json=p).headers


def test_paraphrase_hits_cache(client, rows):
    r = rows["row_22"]
    body = client.post("/v1/troubleshoot", json={"query": "phone display stays black but calls still ring",
                                                 "siis_response": {"title": r.title, "content": r.content}}).json()
    assert body["meta"]["cache_hit"]


def test_no_siis_lookup_and_no_context(client):
    ok = client.post("/v1/troubleshoot", json={"query": "touchscreen is laggy and slow to respond"}).json()
    assert ok["contexts"]
    none = client.post("/v1/troubleshoot", json={"query": "what is the capital of France"}).json()
    assert none == {"contexts": [], "fallback": "no_siis_context", "meta": none["meta"]}


def test_siis_as_plain_string(client, rows):
    body = client.post("/v1/troubleshoot", json={"query": "touch lag", "siis_response": rows["row_21"].content}).json()
    assert body["contexts"]


def test_unseen_article_non_empty_and_clean(client):
    u = build_unseen()[-1]
    body = client.post("/v1/troubleshoot", json={"query": u["query"], "siis_response": u["siis_response"]}).json()
    assert body["contexts"] and not body["meta"]["cache_hit"]
    scored = json.dumps({k: v for k, v in body.items() if k != "meta"})
    assert "http" not in scored and "www." not in scored


def test_device_state_diagnosis_and_verify(client, rows):
    p = _payload(rows["row_21"]) | {"device_state": {"Touch sensitivity": "False"}}
    body = client.post("/v1/troubleshoot", json=p).json()
    diag = body["meta"]["diagnosis"]
    assert any(d["status"] == "healthy" for d in diag) and any(d["status"] == "needs_fix" for d in diag)
    val = next(grp["validationDeeplink"] for g in body["contexts"] for a in g["actions"] for grp in a["stepGroups"]
               if grp["validationDeeplink"] and grp["validationDeeplink"].get("value") == "True")
    ok = client.post("/v1/verify", json={"validationDeeplink": val, "device_state_after": {val["key"]: "True"}}).json()
    assert ok["status"] == "resolved"
    again = client.post("/v1/verify", json={"validationDeeplink": val, "device_state_after": {val["key"]: "False"},
                                            "attempt": 2}).json()
    assert again["status"] == "escalate"


def test_metrics_and_twin(client):
    m = client.get("/metrics").json()
    assert m["requests"] >= 1 and "latency_ms" in m
    tw = client.get("/v1/twin/catalog").json()
    assert any(s["name"] == "Display" for s in tw["screens"])
    assert client.get("/twin").status_code == 200
