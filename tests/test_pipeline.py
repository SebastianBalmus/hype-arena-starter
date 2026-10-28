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
