import csv
import io
import os
import warnings

import pytest

with warnings.catch_warnings():  # starlette's TestClient touches a deprecated anyio alias on import
    warnings.simplefilter("ignore", DeprecationWarning)
    from fastapi.testclient import TestClient

from arena.train import build_artifacts


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    art = os.environ.get("ARENA_ARTIFACTS")
    if not art or not os.path.exists(os.path.join(art, "decider.pkl")):
        art = str(tmp_path_factory.mktemp("art"))
        build_artifacts("data/practice.csv", "data/lexicon.json", art)
    os.environ["ARENA_ARTIFACTS"] = art
    from arena.server import app
    with TestClient(app) as c:
        yield c


def _csv(rows):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["id", "text", "date", "label"])
    w.writerows(rows)
    return buf.getvalue()


def test_health_and_sample(client):
    h = client.get("/api/health").json()
    assert h["ok"] is True and h["rows_background"] > 0
    assert client.get("/api/sample").json()["csv"].startswith("id,text,date,label")


def test_sample_is_unlabelled_and_has_20_rows(client):
    rows = list(csv.DictReader(io.StringIO(client.get("/api/sample").json()["csv"])))
    assert len(rows) == 20 and all(r["label"] == "" for r in rows)


def test_run_returns_labels(client):
    body = client.post("/api/run", json={"csv": _csv([["s-1", "Hooly AI Pro costs €59 per month.", "2026-05-10", ""]])}).json()
    row = body["rows"][0]
    assert row["id"] == "s-1" and row["label"] in ("true", "false", "unsure")
    assert row["facts"][0]["attribute"] == "price"
    assert set(body["timings"]) == {"extract_ms", "evidence_ms", "decide_ms", "generate_ms"}
    assert isinstance(body["claims"], list) and len(body["claims"]) <= 20


def test_labelled_rows_are_evidence_only(client):
    text = _csv([["x-1", "Hooly AI Pro costs €59 per month.", "2026-05-10", "true"],
                 ["x-2", "Hooly AI Pro costs €59 per month.", "2026-05-10", ""]])
    rows = client.post("/api/run", json={"csv": text}).json()["rows"]
    assert [r["id"] for r in rows] == ["x-1", "x-2"]
    assert rows[0]["label"] == "true" and rows[0]["confidence"] is None
    assert rows[1]["label"] in ("true", "false", "unsure")


def test_posted_row_may_reuse_practice_id(client):
    r = client.post("/api/run", json={"csv": _csv([["p-1", "Hooly AI Pro costs €59 per month.", "2026-05-10", ""]])})
    assert r.status_code == 200 and r.json()["rows"][0]["id"] == "p-1"
    assert r.json()["rows"][0]["facts"][0]["attribute"] == "price"


@pytest.mark.parametrize("payload", [{"csv": "nope"}, {"csv": ""}, {}, {"csv": "id,text\na,b\n"}])
def test_bad_csv(client, payload):
    r = client.post("/api/run", json=payload)
    assert r.status_code == 400 and isinstance(r.json()["detail"], str)


def test_index_and_static_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "<svg" in r.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/styles.css").status_code == 200
