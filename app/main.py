"""FastAPI service.

POST /v1/troubleshoot   complaint (+ optional SIIS payload, + optional device_state) -> plan
POST /v1/verify         after a fix: resolved | next_action | escalate
GET  /health            {"status":"ok"} only after cache, indexes and model client are warm
GET  /metrics           latency percentiles per path, cache efficacy, cost
GET  /twin              digital twin demo page
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel

from app.config import ROOT
from app.diagnose import verify
from app.kit import load_catalog, load_siis_rows
from app.pipeline import get_engine
from app.twin import twin_catalog

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger("fixpath")
META_IN_BODY = os.environ.get("META_IN_BODY", "1") != "0"
STATE: dict[str, Any] = {"error": None}


def _warm() -> None:
    try:
        get_engine().warm()
    except Exception as exc:  # surfaced by /health
        STATE["error"] = repr(exc)
        log.exception("warm-up failed")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if os.environ.get("WARM_SYNC") == "1":
        _warm()
    else:
        threading.Thread(target=_warm, daemon=True).start()
    yield


app = FastAPI(title="FixPath", version="1.0.0", lifespan=lifespan,
              description="Smart Guided Troubleshooting Engine (Samsung PRISM Gen AI Hackathon 3.0, Theme 2)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class TroubleshootRequest(BaseModel):
    query: str
    siis_response: Optional[Any] = None
    device_state: Optional[dict[str, Any]] = None


class VerifyRequest(BaseModel):
    validationDeeplink: dict[str, Any]
    device_state_after: dict[str, Any]
    attempt: int = 1


@app.get("/health")
def health():
    engine = get_engine()
    if engine.ready:
        return {"status": "ok"}
    return JSONResponse({"status": "error" if STATE["error"] else "warming", "detail": STATE["error"]},
                        status_code=503)


@app.post("/v1/troubleshoot")
def troubleshoot(req: TroubleshootRequest, debug: bool = Query(False)):
    engine = get_engine()
    if not engine.ready:
        _wait_ready(engine)
    t0 = time.perf_counter()
    try:
        body, meta = engine.troubleshoot(req.query, req.siis_response, req.device_state)
    except Exception as exc:  # never leak a stack trace or break the schema
        log.exception("troubleshoot failed")
        body, meta = {"contexts": [], "fallback": "no_match"}, {"error": type(exc).__name__}
    meta.setdefault("latency_ms", round((time.perf_counter() - t0) * 1000, 2))
    log.info(json.dumps({"path": "/v1/troubleshoot", "cache_tier": meta.get("cache_tier"),
                         "latency_ms": meta.get("latency_ms"), "model": meta.get("model"),
                         "tokens": (meta.get("input_tokens") or 0) + (meta.get("output_tokens") or 0),
                         "cost_usd": meta.get("cost_usd"), "validator_fixes": meta.get("validator_fixes")}))
    out = dict(body)
    if META_IN_BODY or debug:
        out["meta"] = meta
    header = json.dumps({k: meta.get(k) for k in ("latency_ms", "cache_hit", "cache_tier", "model", "cost_usd")})
    return JSONResponse(out, headers={"X-FixPath-Meta": header})


def _wait_ready(engine, timeout_s: float = 60) -> None:
    deadline = time.monotonic() + timeout_s
    while not engine.ready and time.monotonic() < deadline and not STATE["error"]:
        time.sleep(0.05)


@app.post("/v1/verify")
def verify_fix(req: VerifyRequest):
    return verify(req.validationDeeplink, req.device_state_after, req.attempt)


@app.get("/metrics")
def metrics():
    return get_engine().metrics()


@app.get("/v1/scenarios")
def scenarios():
    return [{"id": r.id, "query": r.original_query, "siis_response": {"title": r.title, "content": r.content}}
            for r in load_siis_rows()]


@app.get("/v1/twin/catalog")
def twin():
    return twin_catalog(load_catalog())


@app.get("/twin")
def twin_page():
    return FileResponse(ROOT / "twin" / "index.html")


@app.get("/")
def root():
    return RedirectResponse("/twin")
