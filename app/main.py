"""FastAPI entry point. Phase 0 stub: only /health. Routes arrive in Phase 4."""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.kit import load_catalog, load_schema, load_siis_rows

STATE = {"ready": False}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_schema()
    load_catalog()
    load_siis_rows()
    STATE["ready"] = True
    yield


app = FastAPI(title="FixPath", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    if not STATE["ready"]:
        return JSONResponse({"status": "starting"}, status_code=503)
    return {"status": "ok"}
