"""End to end: the Reader parses rows that have no atoms, then a truth model labels them.

Labelled practice rows keep their gold atoms (at runtime they are recovered from the
packaged practice data by id + text + date); every other row is parsed from its text.

Parse confidence (#4): inside the training data, a second Reader trained without a slice of
pages parses that slice, which shows how often a parse with score s is actually right
(isotonic calibration). Each parsed atom carries that probability; the truth model pulls an
unsure parse towards the prior instead of trusting it fully.
"""
from __future__ import annotations

import random
from dataclasses import replace

from .calibrate import Isotonic
from .folds import page_groups
from .models import Model
from .reader import Reader, atom_match


class Pipeline(Model):
    def __init__(self, truth_cls, threshold: float = 0.3, bool_threshold: float = 0.5, seed: int = 0,
                 parse_aware: bool = True):
        self.truth = truth_cls()
        self.name = self.truth.name
        self.threshold, self.bool_threshold, self.seed = threshold, bool_threshold, seed
        self.parse_aware = parse_aware

    def _reader(self):
        return Reader(self.threshold, self.bool_threshold, seed=self.seed)

    def _parse_calibration(self, train_rows, lexicon, names):
        groups = page_groups(train_rows)
        random.Random(self.seed).shuffle(groups)
        held, n = set(), 0
        for g in groups:
            if n >= len(train_rows) // 5:
                break
            held.update(r.id for r in g)
            n += len(g)
        inner = self._reader().fit([r for r in train_rows if r.id not in held], lexicon, names)
        copies = [replace(r, atoms=None, meta={}) if r.id in held else r for r in train_rows]
        inner.parse_all(copies)
        gold = {r.id: r.atoms for r in train_rows}
        scores, ok = [], []
        for c in copies:
            if c.id in held:
                for a in c.atoms:
                    scores.append(c.meta["parse_scores"].get(a.key, 0.0))
                    ok.append(int(any(atom_match(a, g, names) for g in gold[c.id])))
        return Isotonic().fit(scores, ok)

    def fit(self, rows, lexicon, names):
        train_rows = [r for r in rows if r.atoms]
        self.parse_cal = self._parse_calibration(train_rows, lexicon, names) if self.parse_aware else None
        self.reader = self._reader().fit(train_rows, lexicon, names)
        self.reader.parse_all(rows)  # fills .atoms of rows that had none
        if self.parse_cal:
            for r in rows:
                if r.meta.get("parsed"):
                    r.meta["atom_q"] = {k: self.parse_cal(p) for k, p in r.meta["parse_scores"].items()}
        self.truth.fit(rows, lexicon, names)
        return self

    def predict(self, row):
        p = self.truth.predict(row)
        p.trace["parsed"] = bool(row.meta.get("parsed"))
        p.trace["parse_probs"] = row.meta.get("parse_probs", {})
        return p
