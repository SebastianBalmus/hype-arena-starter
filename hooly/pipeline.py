"""End to end: the Reader parses rows that have no atoms, then a truth model labels them.

Labelled practice rows keep their gold atoms (at runtime they are recovered from the
packaged practice data by id + text + date); every other row is parsed from its text.
"""
from __future__ import annotations

from .models import Model
from .reader import Reader


class Pipeline(Model):
    def __init__(self, truth_cls, threshold: float = 0.02, bool_threshold: float = 0.5):
        self.truth = truth_cls()
        self.name = self.truth.name
        self.threshold, self.bool_threshold = threshold, bool_threshold

    def fit(self, rows, lexicon, names):
        self.reader = Reader(self.threshold, self.bool_threshold).fit([r for r in rows if r.atoms], lexicon, names)
        self.reader.parse_all(rows)  # fills .atoms of rows that had none
        self.truth.fit(rows, lexicon, names)
        return self

    def predict(self, row):
        p = self.truth.predict(row)
        p.trace["parsed"] = bool(row.meta.get("parsed"))
        p.trace["parse_probs"] = row.meta.get("parse_probs", {})
        return p
