"""Pipeline: rows -> facts -> evidence -> labels (+ claims)."""
from __future__ import annotations

import dataclasses
import os
import pickle
import time
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


def run(rows: list[Row], lexicon: Lexicon, art: Artifacts, background: list[Row] | None = None,
        generate: bool = True, n_claims: int = 1000, stop_at: float | None = None,
        on_labels=None) -> RunResult:
    """Label the unlabelled ``rows``; ``background`` rows are evidence only.

    ``stop_at`` (Unix seconds) cuts extraction short (rows not reached keep the safe label) and skips claims.
    ``on_labels(result)`` is called as soon as the labels are complete, before claim generation.
    """
    background = background or []
    everything = background + rows
    extractor = Extractor(lexicon, art.names, art.kinds, art.model, art.gold, art.gold_rows)
    facts: dict[str, list[Fact]] = {}
    for chunk in _chunks(everything):
        if stop_at is not None and time.time() > stop_at:
            break
        facts.update(extractor.extract(chunk))
    store = EvidenceStore()
    for r in everything:
        if r.id in facts:
            store.add_row(r, facts[r.id])
    store.build()

    result = RunResult()
    for r in rows:
        if r.label != "":
            continue
        if r.id not in facts:
            result.labels.append((r.id, *SAFE_LABEL))
            continue
        fs = facts[r.id]
        checks = [store.check(f, r.date) for f in fs]
        x = row_features(fs, checks, store.page_reliability.get(r.page, 0.8))
        label, conf = art.decider.predict(x)
        result.labels.append((r.id, label, _conf(label, conf)))
        result.traces[r.id] = {
            "facts": [dataclasses.asdict(f) for f in fs],
            "checks": [dataclasses.asdict(c) for c in checks],
            "features": x,
        }
    if on_labels is not None:
        on_labels(result)
    if generate and (stop_at is None or time.time() < stop_at):
        texts = {_normalise_text(r.text) for r in everything}
        cands = enumerate_candidates(store, lexicon, art.kinds, art.names)
        result.claims = select_claims(cands, lexicon, texts, n=n_claims, kinds=art.kinds)
    return result
