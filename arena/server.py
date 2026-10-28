"""Live demo server: runs the pipeline on posted CSV rows and serves the static page in web/."""
from __future__ import annotations

import csv
import io
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles

from arena.data import _rows_from, load_practice
from arena.lexicon import Lexicon
from arena.pipeline import load_artifacts, run

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(ROOT, "web")
PRACTICE = os.path.join(ROOT, "data", "practice.csv")
LEXICON = os.path.join(ROOT, "data", "lexicon.json")
SAMPLE = os.path.join(ROOT, "sample", "sample.csv")
COLUMNS = ["id", "text", "date", "label"]
N_CLAIMS_SHOWN = 20
MAX_ROWS = 2000

state: dict = {}


def artifacts_dir() -> str:
    d = os.environ.get("ARENA_ARTIFACTS", "artifacts")
    return d if os.path.isabs(d) else os.path.join(ROOT, d)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    art_dir = artifacts_dir()
    state["lexicon"] = Lexicon.load(LEXICON)
    state["art"] = load_artifacts(art_dir, PRACTICE)
    state["background"], _ = load_practice(PRACTICE)
    yield
    state.clear()


app = FastAPI(title="Hype Arena", lifespan=lifespan)


def _to_csv(rows: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def _parse(text) -> list:
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(400, "Missing CSV text.")
    reader = csv.DictReader(io.StringIO(text.lstrip("﻿")))
    missing = [c for c in COLUMNS if c not in (reader.fieldnames or [])]
    if missing:
        raise HTTPException(400, "CSV needs columns id,text,date,label; missing: " + ", ".join(missing))
    try:
        records = [r for r in reader if any((v or "").strip() for v in r.values() if isinstance(v, str))]
    except csv.Error as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")
    if not records:
        raise HTTPException(400, "CSV has no data rows.")
    if len(records) > MAX_ROWS:
        raise HTTPException(400, f"Too many rows (max {MAX_ROWS}).")
    for r in records:
        if None in r or any(r[c] is None for c in COLUMNS) or not r["id"].strip():
            raise HTTPException(400, "Every row needs id, text, date and label cells.")
        r["id"] = r["id"].strip()
    return _rows_from(records)


@app.get("/api/health")
def health():
    return {"ok": True, "rows_background": len(state["background"])}


@app.get("/api/sample")
def sample():
    if os.path.exists(SAMPLE):
        with open(SAMPLE, encoding="utf-8-sig", newline="") as f:
            return {"csv": f.read()}
    rows = [{"id": r.id, "text": r.text, "date": r.date, "label": ""} for r in state["background"][:20]]
    return {"csv": _to_csv(rows)}


@app.post("/api/run")
async def api_run(request: Request):
    try:
        payload = await request.json()
    except ValueError:
        raise HTTPException(400, "Request body must be JSON like {\"csv\": \"...\"}.")
    if not isinstance(payload, dict):
        raise HTTPException(400, "Request body must be a JSON object with a \"csv\" field.")
    return await run_in_threadpool(_run_payload, payload)


def _run_payload(payload: dict):
    rows = _parse(payload.get("csv"))
    posted = {r.id for r in rows}
    background = [r for r in state["background"] if r.id not in posted]  # posted rows win
    res = run(rows, state["lexicon"], state["art"], background=background, generate=True,
              n_claims=N_CLAIMS_SHOWN)
    labels = {rid: (label, conf) for rid, label, conf in res.labels}
    out = []
    for r in rows:
        if r.label != "":
            label, conf = r.label, None
        else:
            label, conf = labels[r.id]
        tr = res.traces.get(r.id, {})
        out.append({"id": r.id, "text": r.text, "date": r.date, "label": label, "confidence": conf,
                    "facts": tr.get("facts", []), "checks": tr.get("checks", [])})
    return JSONResponse({"rows": out, "claims": res.claims[:N_CLAIMS_SHOWN], "timings": res.timings})


@app.get("/")
def index():
    return FileResponse(os.path.join(WEB, "index.html"))


app.mount("/static", StaticFiles(directory=WEB), name="static")


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))


if __name__ == "__main__":
    main()
