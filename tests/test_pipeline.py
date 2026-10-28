import csv
import os
from collections import Counter
import shutil
import signal
import subprocess
import sys
import time

import pytest

from arena.cli import main
from arena.lexicon import Lexicon
from arena.model import Row
from arena.pipeline import load_artifacts, run
from arena.train import build_artifacts

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRACTICE = os.path.join(ROOT, "data", "practice.csv")
LEXICON = os.path.join(ROOT, "data", "lexicon.json")


@pytest.fixture(scope="session")
def ART(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("artifacts"))
    build_artifacts(PRACTICE, LEXICON, d)
    return d


@pytest.fixture(scope="session")
def art(ART):
    return load_artifacts(ART, PRACTICE)


@pytest.fixture(scope="session")
def L():
    return Lexicon.load(LEXICON)


def _universe(tmp_path, src=PRACTICE):
    uni, out = tmp_path / "universe", tmp_path / "output"
    uni.mkdir()
    out.mkdir()
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "input.py"), src, str(uni / "universe.csv")], check=True)
    shutil.copy(LEXICON, uni / "lexicon.json")
    return uni, out


def test_unparseable_row_gets_label(art, L):
    r = run([Row("z", "Nothing to see here", "2026-01-01", "", 0)], L, art, generate=False)
    assert r.labels[0][0] == "z" and r.labels[0][1] in ("true", "false", "unsure")
    if r.labels[0][1] == "unsure":
        assert r.labels[0][2] is None
    else:
        assert 0.5 <= r.labels[0][2] <= 1


def test_labelled_rows_are_evidence_not_output(art, L):
    rows = [Row("a", "Nothing to see here", "2026-01-01", "true", 0), Row("b", "Nothing else", "2026-01-01", "", 0)]
    r = run(rows, L, art, generate=False)
    assert [x[0] for x in r.labels] == ["b"] and "b" in r.traces
    assert set(r.traces["b"]) == {"facts", "checks", "features"}


def test_deadline_passed_leaves_safe_output(tmp_path, monkeypatch):
    uni, out = _universe(tmp_path)
    with open(uni / "universe.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r["label"] == ""][:3]
    with open(uni / "universe.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "text", "date", "label"])
        w.writeheader()
        w.writerows(rows)
    monkeypatch.setenv("ARENA_DEADLINE", str(int(time.time()) - 1))
    assert main(["--universe", str(uni), "--output", str(out), "--artifacts", "/nonexistent", "--practice", PRACTICE]) == 0
    got = list(csv.DictReader(open(out / "output.csv")))
    assert len(got) == 3 and all(r["label"] == "true" and r["confidence"] == "0.6" for r in got)


def test_failure_keeps_safe_output(tmp_path, monkeypatch):
    uni, out = _universe(tmp_path)
    monkeypatch.setenv("ARENA_DEADLINE", str(int(time.time()) + 3600))
    assert main(["--universe", str(uni), "--output", str(out), "--artifacts", "/nonexistent", "--practice", PRACTICE]) == 0
    assert (out / "output.csv").exists()


def test_sigterm_handler_exits_zero():
    from arena.cli import _on_term
    with pytest.raises(SystemExit) as e:
        _on_term(signal.SIGTERM, None)
    assert e.value.code == 0


@pytest.mark.slow
def test_end_to_end_practice_shape(tmp_path, ART, monkeypatch, capsys):
    uni, out = _universe(tmp_path)
    monkeypatch.setenv("ARENA_DEADLINE", str(int(time.time()) + 1800))
    t0 = time.time()
    assert main(["--universe", str(uni), "--output", str(out), "--artifacts", ART, "--practice", PRACTICE]) == 0
    wall = time.time() - t0
    res = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "check_output.py"), str(out), PRACTICE,
                          str(uni / "universe.csv")], capture_output=True, text=True)
    by_type = Counter(r["type"] for r in csv.DictReader(open(out / "output.csv")) if r["type"])
    with capsys.disabled():
        print(f"\nWALL {wall:.1f}s {dict(by_type)}\n{res.stdout}")
    problems = [l for l in res.stdout.splitlines() if l.startswith("  ")]
    assert all("new claims, not exactly" in p for p in problems), res.stdout
    assert "380 of the 380 hidden rows labelled" in res.stdout


def _write_universe(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "text", "date", "label"])
        w.writeheader()
        w.writerows(rows)


def pipeline_load():
    from arena.pipeline import load_practice
    return load_practice(PRACTICE)


def test_sigterm_before_first_safe_write(tmp_path):
    uni, out = _universe(tmp_path)
    script = tmp_path / "wrap.py"
    script.write_text(
        "import sys, time\n"
        f"sys.path.insert(0, {ROOT!r})\n"
        "import arena.cli as c\n"
        "orig = c.read_universe\n"
        "def slow(p):\n"
        "    print('READY', flush=True); time.sleep(2); return orig(p)\n"
        "c.read_universe = slow\n"
        f"sys.exit(c.main(['--universe', {str(uni)!r}, '--output', {str(out)!r}, '--artifacts', '/nonexistent', '--practice', {PRACTICE!r}]))\n")
    p = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == "READY"
    p.send_signal(signal.SIGTERM)
    assert p.wait(timeout=60) == 0
    with open(uni / "universe.csv", encoding="utf-8", newline="") as f:
        want = {r["id"] for r in csv.DictReader(f) if r["label"] == ""}
    got = {r["id"] for r in csv.DictReader(open(out / "output.csv")) if r["label"]}
    assert got == want and want


def test_failing_chunk_keeps_other_labels(art, L, monkeypatch, capsys):
    from arena import pipeline
    orig = pipeline.Extractor.extract

    def flaky(self, chunk):
        if any(r.id == "bad" for r in chunk):
            raise RuntimeError("boom")
        return orig(self, chunk)

    monkeypatch.setattr(pipeline.Extractor, "extract", flaky)
    rows = [Row("bad", "Nothing here", "2026-01-01", "", 0), Row("ok", "Nothing else", "2026-02-01", "", 1)]
    r = run(rows, L, art, generate=False)
    got = {x[0]: x for x in r.labels}
    assert got["bad"][1:] == ("true", 0.6) and "ok" in got and "ok" in r.traces
    assert "boom" in capsys.readouterr().err


def test_failing_row_keeps_other_labels(art, L, monkeypatch, capsys):
    from arena import pipeline
    calls = []
    real = pipeline.row_features

    def flaky(facts, checks, rel):
        calls.append(1)
        if len(calls) == 1:
            raise ValueError("rowfail")
        return real(facts, checks, rel)

    monkeypatch.setattr(pipeline, "row_features", flaky)
    rows = [Row("a", "Nothing here", "2026-01-01", "", 0), Row("b", "Nothing else", "2026-01-01", "", 0)]
    r = run(rows, L, art, generate=False)
    assert len(r.labels) == 2 and r.labels[0][1:] == ("true", 0.6) and "b" in r.traces
    assert "rowfail" in capsys.readouterr().err


def test_deadline_expiring_during_generation_writes_partial(art, L, monkeypatch):
    from arena import generate, pipeline
    from arena.evidence import EvidenceStore
    from arena.extract import Extractor
    rows, _ = pipeline_load()
    ex = Extractor(L, art.names, art.kinds, art.model, art.gold, art.gold_rows)
    facts = ex.extract(rows)
    store = EvidenceStore()
    for r in rows:
        store.add_row(r, facts[r.id])
    store.build()
    cands = pipeline.enumerate_candidates(store, L, art.kinds, art.names)
    assert len(generate.select_claims(cands, L, set(), n=30, kinds=art.kinds)) == 30
    clock = iter(range(10**6))
    monkeypatch.setattr(generate.time, "time", lambda: next(clock))
    partial = generate.select_claims(cands, L, set(), n=30, kinds=art.kinds, stop_at=5)
    assert 0 < len(partial) < 30
    monkeypatch.undo()
    expired = run([], L, art, background=rows, n_claims=30, stop_at=time.time() - 1)
    assert expired.claims == []


def test_claims_not_started_without_time(art, L):
    rows, _ = pipeline_load()
    assert run([], L, art, background=rows, claims_after=time.time() - 1).claims == []


def test_claim_ids_avoid_input_ids(art, L):
    rows, _ = pipeline_load()
    rows = [Row("c-0001", r.text, r.date, r.label, r.page) if i == 0 else r for i, r in enumerate(rows)]
    r = run([], L, art, background=rows, n_claims=5)
    ids = [c["id"] for c in r.claims]
    assert "c-0001" not in ids and len(set(ids)) == len(ids) == 5


@pytest.mark.slow
def test_scale_20k_rows(tmp_path, ART, monkeypatch, capfd):
    from datetime import date, timedelta
    uni, out = tmp_path / "u", tmp_path / "o"
    uni.mkdir()
    out.mkdir()
    shutil.copy(LEXICON, uni / "lexicon.json")
    with open(PRACTICE, encoding="utf-8-sig", newline="") as f:
        src = list(csv.DictReader(f))
    rows = []
    for k in range(-(-20000 // len(src))):
        for r in src:
            d = date.fromisoformat(r["date"]) - timedelta(days=3 * k)
            rows.append({"id": r["id"] if k == 0 else f"r{k}-{r['id']}", "text": r["text"], "date": d.isoformat(),
                         "label": r["label"] if k == 0 else ""})
    rows = rows[:20000]
    _write_universe(uni / "universe.csv", rows)
    monkeypatch.setenv("ARENA_DEADLINE", str(int(time.time()) + 1800))
    t0 = time.time()
    assert main(["--universe", str(uni), "--output", str(out), "--artifacts", ART, "--practice", PRACTICE]) == 0
    wall = time.time() - t0
    err = capfd.readouterr().err
    with capfd.disabled():
        print(f"\nSCALE rows={len(rows)} wall={wall:.1f}s\n{err}")
    assert wall < 600
    got = list(csv.DictReader(open(out / "output.csv")))
    assert len([r for r in got if r["label"] and not r["type"]]) == sum(r["label"] == "" for r in rows)


def test_predict_many_matches_predict(art):
    from arena.decide import FEATURES
    xs = [{k: float((i * 7 + j) % 3) for j, k in enumerate(FEATURES)} for i in range(12)]
    assert art.decider.predict_many(xs) == [art.decider.predict(x) for x in xs]
