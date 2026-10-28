"""Pipeline: rows -> facts -> evidence -> labels (+ claims)."""
from __future__ import annotations

import dataclasses
import os
import pickle
import sys
import time
import traceback
from dataclasses import dataclass, field

from arena.data import load_practice
from arena.decide import Decider, row_features
from arena.evidence import EvidenceStore
from arena.extract import Extractor
from arena.generate import _normalise_text, enumerate_candidates, select_claims
from arena.kinds import KindIndex
from arena.lexicon import Lexicon, NameIndex
from arena.model import Fact, Row

SAFE_LABEL = ("true", 0.6)


@dataclass
class Artifacts:
    kinds: KindIndex
    names: NameIndex
    model: object
    decider: Decider
    gold: dict[str, list[Fact]]
    gold_rows: dict[str, Row]


@dataclass
class RunResult:
    labels: list[tuple[str, str, float | None]] = field(default_factory=list)
    claims: list[dict] = field(default_factory=list)
    traces: dict[str, dict] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)  # stage -> milliseconds


def load_artifacts(art_dir: str, practice_csv: str) -> Artifacts:
    def get(name):
        with open(os.path.join(art_dir, name), "rb") as f:
            return pickle.load(f)

    rows, gold = load_practice(practice_csv)
    return Artifacts(get("kinds.pkl"), get("names.pkl"), get("attribute_model.pkl"), get("decider.pkl"),
                     gold, {r.id: r for r in rows})


def _chunks(rows: list[Row]):
    """Runs of rows that share a page, so page context stays intact."""
    chunk: list[Row] = []
    for r in rows:
        if chunk and (r.date != chunk[-1].date or r.page != chunk[-1].page):
            yield chunk
            chunk = []
        chunk.append(r)
    if chunk:
        yield chunk


def _conf(label: str, conf: float | None) -> float | None:
    if label == "unsure":
        return None
    return min(1.0, max(0.5, 0.6 if conf is None else conf))


def _log(stage: str, t0: float, tm: dict | None = None, key: str = "") -> float:
    now = time.time()
    if tm is not None:
        tm[key] = tm.get(key, 0.0) + (now - t0) * 1000
    print(f"[arena] {stage}: {now - t0:.1f}s", file=sys.stderr, flush=True)
    return now


def _unique_ids(claims: list[dict], taken: set[str]) -> None:
    for c in claims:
        while c["id"] in taken:
            c["id"] = "x" + c["id"]
        taken.add(c["id"])


def _decide(decider, xs: dict[str, dict], stop_at: float | None) -> dict[str, tuple[str, float | None]]:
    """Predict every row at once when the decider can (one forest pass), else row by row."""
    if not xs:
        return {}
    ids = list(xs)
    many = getattr(decider, "predict_many", None)
    if many is not None:
        try:
            return dict(zip(ids, many([xs[i] for i in ids])))
        except Exception:
            traceback.print_exc()
    out = {}
    for i in ids:
        if stop_at is not None and time.time() > stop_at:
            break
        try:
            out[i] = decider.predict(xs[i])
        except Exception:
            traceback.print_exc()
    return out


def run(rows: list[Row], lexicon: Lexicon, art: Artifacts, background: list[Row] | None = None,
        generate: bool = True, n_claims: int = 1000, stop_at: float | None = None,
        on_labels=None, claims_after: float | None = None) -> RunResult:
    """Label the unlabelled ``rows``; ``background`` rows are evidence only.

    ``stop_at`` (Unix seconds) is a hard stop for extraction, labelling and claim selection: rows not
    reached keep the safe label, claims selected so far are returned. ``claims_after`` is the latest time
    at which claim generation may still start. ``on_labels(result)`` is called once the labels are
    complete, before claim generation. A failure in one page chunk or one row only costs that row its label.
    """
    background = background or []
    everything = background + rows
    tm: dict[str, float] = {"extract_ms": 0.0, "evidence_ms": 0.0, "decide_ms": 0.0, "generate_ms": 0.0}
    t0 = time.time()
    names = NameIndex.from_values(list(art.names.names) + list(lexicon.subjects), lexicon)
    extractor = Extractor(lexicon, names, art.kinds, art.model, art.gold, art.gold_rows)
    facts: dict[str, list[Fact]] = {}
    for chunk in _chunks(everything):
        if stop_at is not None and time.time() > stop_at:
            break
        try:
            facts.update(extractor.extract(chunk))
        except Exception:
            traceback.print_exc()
    t0 = _log("extract", t0, tm, "extract_ms")
    store = EvidenceStore()
    built = True
    try:
        for r in everything:
            if r.id in facts:
                store.add_row(r, facts[r.id])
        store.build()
    except Exception:
        traceback.print_exc()
        built = False
    t0 = _log("evidence", t0, tm, "evidence_ms")

    result = RunResult(timings=tm)
    feats: dict[str, tuple[dict, dict]] = {}  # id -> (features, trace)
    order: list[str] = []
    for r in rows:
        if r.label != "":
            continue
        order.append(r.id)
        if not built or r.id not in facts or (stop_at is not None and time.time() > stop_at):
            continue
        try:
            fs = facts[r.id]
            checks = [store.check(f, r.date) for f in fs]
            x = row_features(fs, checks, store.page_reliability.get(r.page, 0.8))
            feats[r.id] = (x, {
                "facts": [dataclasses.asdict(f) for f in fs],
                "checks": [dataclasses.asdict(c) for c in checks],
                "features": x,
            })
        except Exception:
            traceback.print_exc()
    decided = _decide(art.decider, {i: v[0] for i, v in feats.items()}, stop_at)
    for rid in order:
        if rid in decided:
            label, conf = decided[rid]
            result.labels.append((rid, label, _conf(label, conf)))
            result.traces[rid] = feats[rid][1]
        else:
            result.labels.append((rid, *SAFE_LABEL))
    t0 = _log("label", t0, tm, "decide_ms")
    if on_labels is not None:
        on_labels(result)
    now = time.time()
    if (generate and built and (stop_at is None or now < stop_at)
            and (claims_after is None or now <= claims_after)):
        try:
            texts = {_normalise_text(r.text) for r in everything}
            cands = enumerate_candidates(store, lexicon, art.kinds, art.names)
            t0 = _log(f"enumerate ({len(cands)} candidates)", t0, tm, "generate_ms")
            result.claims = select_claims(cands, lexicon, texts, n=n_claims, kinds=art.kinds, stop_at=stop_at)
            _unique_ids(result.claims, {r.id for r in everything})
            _log(f"select ({len(result.claims)} claims)", t0, tm, "generate_ms")
        except Exception:
            traceback.print_exc()
    return result
