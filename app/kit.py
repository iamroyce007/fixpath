"""Loads the organiser kit (schema, deeplink catalog, SIIS rows) once per process.

The kit is the source of truth and is never edited. schema.py is imported from the
kit folder so the response models are never redefined here.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Optional

from app.config import settings

DUMMY_DEEPLINK = "voiceassist://dummy_positive"
# Fields the response schema allows on an actionableDeeplink, copied from the catalog.
ACTIONABLE_FIELDS = ("deeplink", "description", "message", "originalType")


@lru_cache(maxsize=None)
def load_schema(kit_dir: Optional[Path] = None) -> ModuleType:
    kit_dir = kit_dir or settings.kit_dir
    path = kit_dir / "schema.py"
    spec = importlib.util.spec_from_file_location("kit_schema", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["kit_schema"] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class Catalog:
    entries: list[dict]
    by_id: dict[str, dict] = field(default_factory=dict)
    by_deeplink: dict[str, dict] = field(default_factory=dict)
    by_validation: dict[str, dict] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for e in self.entries:
            self.by_id[e["id"]] = e
            self.by_deeplink[e["deeplink"]] = e
            val = e.get("validation")
            if val and val.get("deeplink"):
                self.by_validation[val["deeplink"]] = e

    def __len__(self) -> int:
        return len(self.entries)

    def get(self, ref: str) -> Optional[dict]:
        """Look up an entry by catalog id (DL-0042) or by its exact action URI."""
        return self.by_id.get(ref) or self.by_deeplink.get(ref)

    @staticmethod
    def actionable(entry: dict) -> dict:
        return {k: entry.get(k) for k in ACTIONABLE_FIELDS}

    @staticmethod
    def validation(entry: dict) -> Optional[dict]:
        val = entry.get("validation")
        if not val:
            return None
        return {"deeplink": val["deeplink"], "key": val["key"]}


@dataclass(frozen=True)
class SiisRow:
    id: str
    original_query: str
    title: str
    content: str


@lru_cache(maxsize=None)
def load_catalog(kit_dir: Optional[Path] = None) -> Catalog:
    kit_dir = kit_dir or settings.kit_dir
    data = json.loads((kit_dir / "deeplinks.json").read_text(encoding="utf-8"))
    return Catalog(entries=data["deeplinks"])


@lru_cache(maxsize=None)
def load_siis_rows(kit_dir: Optional[Path] = None) -> tuple[SiisRow, ...]:
    kit_dir = kit_dir or settings.kit_dir
    data = json.loads((kit_dir / "siis_responses.json").read_text(encoding="utf-8"))
    return tuple(
        SiisRow(
            id=r["id"],
            original_query=r["original_query"],
            title=r["siis_response"]["title"],
            content=r["siis_response"]["content"],
        )
        for r in data["responses"]
    )


def load_sample_output(kit_dir: Optional[Path] = None) -> dict:
    kit_dir = kit_dir or settings.kit_dir
    return json.loads((kit_dir / "sample_output.json").read_text(encoding="utf-8"))


def load_queries(kit_dir: Optional[Path] = None) -> list[str]:
    kit_dir = kit_dir or settings.kit_dir
    lines = (kit_dir / "input.txt").read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip()]
