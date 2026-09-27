import copy

import pytest

from app import validate as v
from app.kit import DUMMY_DEEPLINK, load_sample_output, load_schema
from app.validate import validate_and_repair


SIIS = """# Fix the display
## Step 1: Adjust brightness
Swipe down from the top of the screen to open the Quick settings panel.
Go to Settings, tap Display, and then tap Brightness.
## Step 2: Restart the phone
Press and hold the Power button and the Volume down button for 20 seconds.
## Step 3: Get help
Contact the Customer Support Center for further assistance.
"""


@pytest.fixture(scope="module")
def on_entry(catalog):
    return next(e for e in catalog.entries if e["originalType"] == "onURL" and e.get("validation"))


def _goal(actions, **kw):
    g = {
        "goal": "Follow these steps to perform this Display Troubleshooting",
        "title": "Display issues",
        "score": 0.9,
        "actions": actions,
    }
    g.update(kw)
    return {"contexts": [g]}


def _action(name="Open Display Settings", desc="It will open display settings now",
            steps=None, cat="manual", act=None, val=None):
    return {
        "actionName": name,
        "description": desc,
        "category": cat,
        "stepGroups": [{
            "steps": steps or ["Go to Settings, tap Display, and then tap Brightness."],
            "actionableDeeplink": act,
            "validationDeeplink": val,
        }],
    }


def _rules(report):
    return {f["rule"] for f in report["fixes"]}


# ---------------------------------------------------------------- clean input stays untouched


def test_clean_response_has_no_fixes(catalog):
    fixed, rep = validate_and_repair(_goal([_action()]), SIIS, catalog)
    assert rep["valid"] and rep["fix_count"] == 0
    load_schema().ContextDeeplinkResponse.model_validate(fixed)


# ---------------------------------------------------------------- text rules


@pytest.mark.parametrize("period", [False, True])
def test_goal_template_and_period(catalog, monkeypatch, period):
    monkeypatch.setattr(v, "settings", v.settings.__class__(**{**v.settings.__dict__,
                                                               "goal_trailing_period": period}))
    data = _goal([_action()], goal="follow these steps to perform this display troubleshooting.")
    fixed, _ = validate_and_repair(data, SIIS, catalog)
    expected = "Follow these steps to perform this Display Troubleshooting" + ("." if period else "")
    assert fixed["contexts"][0]["goal"] == expected


def test_goal_rebuilt_from_title_when_off_template(catalog):
    fixed, rep = validate_and_repair(_goal([_action()], goal="Fix your screen"), SIIS, catalog)
    assert fixed["contexts"][0]["goal"] == "Follow these steps to perform this Display Issues Troubleshooting"
    assert "goal_template" in _rules(rep)


def test_goal_configuration_kind(catalog):
    data = _goal([_action()], goal="Follow these steps to perform this Rotation Configuration")
    fixed, _ = validate_and_repair(data, SIIS, catalog)
    assert fixed["contexts"][0]["goal"].endswith("Rotation Configuration")


@pytest.mark.parametrize("given,expected", [
    ("SCREEN DISPLAY DAMAGE ISSUES", "Screen display damage"),
    ("Display", "Display issues"),
    ("Wi-Fi connection problems for you", "Wi-Fi connection problems"),
    ("screen of the", "Screen of"),
])
def test_title_rule(given, expected):
    assert v.fix_title(given) == expected


@pytest.mark.parametrize("given,expected", [
    ("open display settings", "Open Display Settings"),
    ("turn off the auto-rotate", "Turn Off the Auto-Rotate"),
    ("Back up data to TechCorp cloud.", "Back Up Data to TechCorp Cloud"),
])
def test_action_name_title_case(given, expected):
    assert v.fix_action_name(given) == expected


@pytest.mark.parametrize("given", [
    "It will help you locate the nearest TechCorp service center and schedule",  # 12 words
    "Opens the display brightness page",  # missing "It will", 3rd-person verb
    "It will reset",  # too short
    "",
    None,
    "It will turn on auto-rotate for screen.",  # hyphen and trailing period
])
def test_description_rule(given):
    out = v.fix_description(given, "Reset Network Settings")
    words = out.split()
    assert 5 <= len(words) <= 7, out
    assert out.startswith("It will ")
    assert not out.endswith((".", "!", "?"))
    assert "-" not in out


def test_description_verb_is_base_form():
    assert v.fix_description("Opens the display brightness page", "X").startswith("It will open the")


# ---------------------------------------------------------------- categories and ordering


def test_auto_without_deeplink_downgraded(catalog):
    fixed, rep = validate_and_repair(_goal([_action(cat="auto")]), SIIS, catalog)
    assert fixed["contexts"][0]["actions"][0]["category"] == "manual"
    assert "auto_without_deeplink_downgraded" in _rules(rep)


def test_manual_deeplink_stripped(catalog, on_entry):
    fixed, rep = validate_and_repair(
        _goal([_action(cat="manual", act={"deeplink": on_entry["deeplink"]})]), SIIS, catalog)
    grp = fixed["contexts"][0]["actions"][0]["stepGroups"][0]
    assert grp["actionableDeeplink"] is None and grp["validationDeeplink"] is None
    assert "manual_deeplink_stripped" in _rules(rep)


def test_critical_promoted_and_moved_last(catalog, on_entry):
    restart = _action(name="Force Restart", desc="It will restart your phone safely",
                      steps=["Press and hold the Power button and the Volume down button for 20 seconds."],
                      cat="manual")
    support = _action(name="Contact Support", desc="It will connect you with support",
                      steps=["Contact the Customer Support Center for further assistance."])
    auto = _action(cat="auto", act={"deeplink": on_entry["deeplink"]})
    fixed, rep = validate_and_repair(_goal([restart, support, auto]), SIIS, catalog)
    cats = [a["category"] for a in fixed["contexts"][0]["actions"]]
    assert cats == ["auto", "manual", "critical"]
    assert {"critical_promoted", "actions_reordered"} <= _rules(rep)


def test_invalid_category_inferred(catalog, on_entry):
    fixed, rep = validate_and_repair(
        _goal([_action(cat="bogus", act={"deeplink": on_entry["deeplink"]})]), SIIS, catalog)
    assert fixed["contexts"][0]["actions"][0]["category"] == "auto"
    assert "category_inferred" in _rules(rep)


def test_same_screen_actions_merged(catalog):
    a1 = _action(steps=["Go to Settings, tap Display, and then tap Brightness."])
    a2 = _action(name="open display settings",
                 steps=["Swipe down from the top of the screen to open the Quick settings panel."])
    fixed, rep = validate_and_repair(_goal([a1, a2]), SIIS, catalog)
    actions = fixed["contexts"][0]["actions"]
    assert len(actions) == 1 and len(actions[0]["stepGroups"]) == 2
    assert "same_screen_actions_merged" in _rules(rep)


# ---------------------------------------------------------------- deeplinks


def test_deeplink_copied_verbatim_by_id(catalog, on_entry):
    act = {"id": on_entry["id"], "description": "made up", "message": "made up", "classes": {"x": "y"}}
    fixed, rep = validate_and_repair(_goal([_action(cat="auto", act=act)]), SIIS, catalog)
    grp = fixed["contexts"][0]["actions"][0]["stepGroups"][0]
    assert grp["actionableDeeplink"] == {k: on_entry[k] for k in
                                         ("deeplink", "description", "message", "originalType")}
    assert grp["validationDeeplink"]["deeplink"] == on_entry["validation"]["deeplink"]
    assert grp["validationDeeplink"]["key"] == on_entry["validation"]["key"]
    assert grp["validationDeeplink"]["value"] == "True"
    assert "deeplink_canonicalised" in _rules(rep)


def test_unknown_deeplink_removed_and_downgraded(catalog):
    act = {"deeplink": "voiceassist://masked/act/0000000000", "description": "x"}
    fixed, rep = validate_and_repair(_goal([_action(cat="auto", act=act)]), SIIS, catalog)
    action = fixed["contexts"][0]["actions"][0]
    assert action["stepGroups"][0]["actionableDeeplink"] is None
    assert action["category"] == "manual"
    assert "unknown_deeplink_removed" in _rules(rep)


def test_wrong_validation_replaced(catalog, on_entry):
    other = next(e for e in catalog.entries if e.get("validation") and e["id"] != on_entry["id"])
    val = {"deeplink": other["validation"]["deeplink"], "key": other["validation"]["key"]}
    fixed, rep = validate_and_repair(
        _goal([_action(cat="auto", act={"deeplink": on_entry["deeplink"]}, val=val)]), SIIS, catalog)
    got = fixed["contexts"][0]["actions"][0]["stepGroups"][0]["validationDeeplink"]
    assert got["deeplink"] == on_entry["validation"]["deeplink"]
    assert "validation_canonicalised" in _rules(rep)


def test_dummy_keeps_written_text_and_has_no_validation(catalog):
    act = {"deeplink": DUMMY_DEEPLINK, "description": "Opens the brightness screen www.x.com",
           "message": "Open Brightness"}
    fixed, _ = validate_and_repair(_goal([_action(cat="auto", act=act)]), SIIS, catalog)
    grp = fixed["contexts"][0]["actions"][0]["stepGroups"][0]
    assert grp["actionableDeeplink"]["deeplink"] == DUMMY_DEEPLINK
    assert grp["actionableDeeplink"]["description"] == "Opens the brightness screen"
    assert grp["actionableDeeplink"]["originalType"] == "placeholder"
    assert grp["validationDeeplink"] is None


# ---------------------------------------------------------------- URL scrubber


def test_url_scrubber_text_only(catalog, on_entry):
    steps = ["Go to Settings, tap Display, and then tap Brightness. See https://help.example.com/x"]
    data = _goal([_action(name="Open [Display](http://x.io) Settings", steps=steps, cat="auto",
                          desc="It will open www.techcorp.com display settings",
                          act={"deeplink": on_entry["deeplink"]})])
    fixed, rep = validate_and_repair(data, SIIS, catalog)
    action = fixed["contexts"][0]["actions"][0]
    text = " ".join([action["actionName"], action["description"], *action["stepGroups"][0]["steps"]])
    for bad in ("http", "www.", ".com", ".io", "]("):
        assert bad not in text
    # deeplink fields are untouched
    assert action["stepGroups"][0]["actionableDeeplink"]["deeplink"].startswith("voiceassist://")
    assert action["stepGroups"][0]["validationDeeplink"]["deeplink"].startswith("voiceassist://")
    assert "url_scrubbed" in _rules(rep)


@pytest.mark.parametrize("text,clean", [
    ("Visit https://a.b/c now", "Visit now"),
    ("Read [the guide](https://x.y) first", "Read the guide first"),
    ("Go to www.techcorp.com/support", "Go to"),
    ("Open help.html please", "Open please"),
    ("voiceassist free text", "voiceassist free text"),
])
def test_scrub_urls(text, clean):
    assert v.scrub_urls(text) == clean


# ---------------------------------------------------------------- provenance and no_match


def test_invented_step_dropped(catalog):
    steps = ["Go to Settings, tap Display, and then tap Brightness.",
             "Download the SuperFix app and run a quantum recalibration."]
    fixed, rep = validate_and_repair(_goal([_action(steps=steps)]), SIIS, catalog)
    assert fixed["contexts"][0]["actions"][0]["stepGroups"][0]["steps"] == steps[:1]
    assert rep["dropped_steps"][0]["step"] == steps[1]
    assert rep["provenance"]["coverage"] == 0.5


def test_nothing_survives_gives_no_match(catalog):
    data = _goal([_action(steps=["Download the SuperFix app and run a quantum recalibration."])])
    fixed, rep = validate_and_repair(data, SIIS, catalog)
    assert fixed == {"contexts": [], "fallback": "no_match"}
    assert rep["fallback"] == "no_match" and rep["valid"]


@pytest.mark.parametrize("bad", [None, [], "text", {"contexts": None}, {"contexts": [None, 3]}])
def test_garbage_input_is_no_match(catalog, bad):
    fixed, rep = validate_and_repair(bad, SIIS, catalog)
    assert fixed == {"contexts": [], "fallback": "no_match"}


def test_empty_steps_and_groups_removed(catalog):
    a = _action()
    a["stepGroups"].append({"steps": ["", "   "], "actionableDeeplink": None})
    a["stepGroups"].append({"steps": []})
    fixed, rep = validate_and_repair(_goal([a]), SIIS, catalog)
    assert len(fixed["contexts"][0]["actions"][0]["stepGroups"]) == 1
    assert "empty_group_removed" in _rules(rep)


# ---------------------------------------------------------------- score


def test_score_clamped(catalog):
    fixed, rep = validate_and_repair(_goal([_action()], score=1.7), SIIS, catalog)
    assert fixed["contexts"][0]["score"] == 1.0
    assert "score_clamped" in _rules(rep)


def test_score_deterministic_with_retrieval(catalog):
    data = _goal([_action()], score=0.123)
    a, _ = validate_and_repair(copy.deepcopy(data), SIIS, catalog, retrieval_confidence=0.8)
    b, _ = validate_and_repair(copy.deepcopy(data), SIIS, catalog, retrieval_confidence=0.8)
    assert a == b
    assert a["contexts"][0]["score"] != 0.123
    assert 0 <= a["contexts"][0]["score"] <= 1


def test_input_not_mutated(catalog):
    data = _goal([_action(cat="auto", desc="bad")])
    before = copy.deepcopy(data)
    validate_and_repair(data, SIIS, catalog)
    assert data == before


# ---------------------------------------------------------------- acceptance: kit sample


def test_sample_output_passes_after_repair(catalog):
    sample = load_sample_output()["response"]
    fixed, rep = validate_and_repair(sample, None, catalog)
    assert rep["valid"]
    assert not rep["provenance"]["checked"]
    load_schema().ContextDeeplinkResponse.model_validate(fixed)
    descs = [a["description"] for a in fixed["contexts"][0]["actions"]]
    assert all(5 <= len(d.split()) <= 7 and d.startswith("It will") for d in descs)
    fixed_paths = {f["path"] for f in rep["fixes"] if f["rule"] == "description_rule"}
    assert "contexts[0].actions[1].description" in fixed_paths  # the 12-word one
    assert fixed["contexts"][0]["goal"] == sample["contexts"][0]["goal"]
    assert fixed["contexts"][0]["title"] == sample["contexts"][0]["title"]
    # Deeplinks in the sample are real catalog entries and survive unchanged.
    grp = fixed["contexts"][0]["actions"][0]["stepGroups"][0]
    assert grp["actionableDeeplink"] == sample["contexts"][0]["actions"][0]["stepGroups"][0]["actionableDeeplink"]
    assert grp["validationDeeplink"] == sample["contexts"][0]["actions"][0]["stepGroups"][0]["validationDeeplink"]


def test_sample_against_cracked_article_drops_untraceable_steps(catalog, rows):
    sample = load_sample_output()["response"]
    fixed, rep = validate_and_repair(sample, rows["row_14"].content, catalog)
    assert rep["provenance"]["checked"]
    assert rep["dropped_steps"]  # backup steps are not in the cracked-screen article
