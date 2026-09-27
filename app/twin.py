"""Digital twin data: the catalog reshaped into a Settings tree with simulated sensor values."""
from __future__ import annotations

import hashlib
import re

from app.catalog import family
from app.kit import DUMMY_DEEPLINK, Catalog

SCREENS = [
    ("Display", r"display|brightness|screen|touch sensitiv|navigation bar|font|dark mode|refresh|edge|dim|rotate|eye comfort"),
    ("Connections", r"wi-?fi|bluetooth|nfc|airplane|mobile data|hotspot|network|sim"),
    ("Sounds and vibration", r"sound|vibrat|volume|ringtone|audio|haptic"),
    ("Notifications", r"notif|do not disturb|alert"),
    ("Battery", r"battery|power saving|charging|adaptive power"),
    ("Security and privacy", r"lock|security|privacy|fingerprint|biometric|password|pin\b|location"),
    ("Accessibility", r"talkback|accessib|magnif|auto-click|cursor|hearing|colour|color|voice assistant|interaction"),
    ("Advanced features", r"multi window|split screen|pop-up|gesture|motion|one-handed|side button|quick access|panel"),
    ("Accounts and backup", r"backup|back up|sync|cloud|account"),
    ("General management", r"time|date|keyboard|language|reset|input"),
    ("Apps", r"app"),
]


def screen_for(entry: dict) -> str:
    text = f"{entry.get('message', '')} {entry.get('description', '')} {(entry.get('validation') or {}).get('key', '')}".lower()
    for name, rx in SCREENS:
        if re.search(rx, text):
            return name
    return "Other settings"


def initial_value(entry: dict) -> str:
    """Deterministic simulated reading so every demo run starts the same."""
    key = (entry.get("validation") or {}).get("key", entry["id"])
    t = entry.get("originalType")
    h = int(hashlib.md5(key.encode()).hexdigest(), 16)
    if t in ("onURL", "offURL"):
        return "True" if h % 3 == 0 else "False"
    if t == "updateURL":
        return str(h % 100)
    return "—"


_CACHE: dict[int, dict] = {}


def _build(catalog: Catalog) -> dict:
    screens: dict[str, dict] = {}
    for e in catalog.entries:
        if e["deeplink"] == DUMMY_DEEPLINK or family(e) != "mobile" or not e.get("originalType"):
            continue
        val = e.get("validation") or {}
        key = val.get("key") or e.get("message")
        s = screens.setdefault(screen_for(e), {})
        item = s.setdefault(key, {"key": key, "entries": [], "value": initial_value(e),
                                  "kind": "toggle" if e["originalType"] in ("onURL", "offURL") else
                                  "slider" if e["originalType"] == "updateURL" else "page"})
        item["entries"].append({"id": e["id"], "type": e["originalType"], "message": e.get("message"),
                                "deeplink": e["deeplink"]})
    return {
        "screens": [{"name": n, "items": sorted(items.values(), key=lambda x: x["key"].lower())}
                    for n, items in sorted(screens.items())],
    }


def twin_catalog(catalog: Catalog) -> dict:
    if id(catalog) not in _CACHE:
        _CACHE[id(catalog)] = _build(catalog)
    return _CACHE[id(catalog)]
