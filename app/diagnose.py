"""Sensor-first diagnosis using validation deeplinks.

The client may send device_state {validation_key: current_value}. For each action whose
validationDeeplink key is known, we compare the current value with the healthy value:
the value the plan wants (validationDeeplink.value), else docs/healthy_values.json.
Healthy actions move to the end of their category block, so the user fixes what is
actually wrong first. The response body stays schema-valid; findings go to meta.
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Optional

from app.config import ROOT

CATEGORY_RANK = {"auto": 0, "manual": 1, "critical": 2}
UNKNOWN_READINGS = {"", "—", "-", "none", "null", "unknown", "?"}


@lru_cache(maxsize=1)
def healthy_table() -> dict[str, str]:
    path = ROOT / "docs" / "healthy_values.json"
    if not path.exists():
        return {}
    return {k: v for k, v in json.loads(path.read_text()).items() if not k.startswith("_")}


def _norm(v: Any) -> str:
    return str(v).strip().lower()


def expected_for(validation: Optional[dict]) -> Optional[str]:
    if not validation:
        return None
    if validation.get("value") is not None:
        return str(validation["value"])
    return healthy_table().get(validation.get("key", ""))


def diagnose(response: dict, device_state: Optional[dict]) -> tuple[dict, list[dict]]:
    if not device_state:
        return response, []
    findings: list[dict] = []
    for g in response.get("contexts", []):
        healthy_flags = []
        for a in g["actions"]:
            status = "unknown"
            for grp in a["stepGroups"]:
                val = grp.get("validationDeeplink")
                if not val or val.get("key") not in device_state:
                    continue
                current, expected = device_state[val["key"]], expected_for(val)
                if expected is None or _norm(current) in UNKNOWN_READINGS:
                    status = "unknown"
                else:
                    status = "healthy" if _norm(current) == _norm(expected) else "needs_fix"
                findings.append({"action": a["actionName"], "key": val["key"], "current": current,
                                 "expected": expected, "status": status})
            healthy_flags.append(status == "healthy")
        order = sorted(range(len(g["actions"])),
                       key=lambda i: (CATEGORY_RANK.get(g["actions"][i]["category"], 1), healthy_flags[i], i))
        g["actions"] = [g["actions"][i] for i in order]
    return response, findings


def verify(validation: dict, device_state_after: dict, attempt: int = 1) -> dict:
    """-> {"status": resolved | next_action | escalate, ...}"""
    key = validation.get("key")
    expected = expected_for(validation)
    current = device_state_after.get(key)
    if current is not None and expected is not None and _norm(current) == _norm(expected):
        return {"status": "resolved", "key": key, "current": current, "expected": expected}
    if attempt >= 2:
        return {"status": "escalate", "key": key, "current": current, "expected": expected,
                "reason": "setting did not reach the expected value after a retry"}
    return {"status": "next_action", "key": key, "current": current, "expected": expected}
