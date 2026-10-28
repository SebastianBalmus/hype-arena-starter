"""Naive baselines: the floor every paper variant has to beat.

- AlwaysTrue: the majority label (~83% on practice).
- LatestBefore / Nearest: look up the labelled-true value of the same fact (subject,
  attribute) closest to the row's date and check the row's value against it.

Their confidence is the empirical label distribution of the decision branch they took,
measured on the training rows with each row's own anchor left out.
"""
from __future__ import annotations

import bisect
from collections import Counter, defaultdict
from datetime import date

from .data import Atom, Row
from .models import Model, Prediction, smoothed
from .values import NameNormalizer, satisfies


def _d(s: str) -> date:
    return date.fromisoformat(s)


class AlwaysTrue(Model):
    name = "always_true"

    def fit(self, rows, lexicon, names):
        self.probs = smoothed(Counter(r.label for r in rows if r.label))
        return self

    def predict(self, row):
        return Prediction("true", dict(self.probs), {"rule": "majority label"})


class AnchorIndex:
    """Labelled-true exact observations per fact, sorted by date."""

    def __init__(self, rows: list[Row]):
        self.by_key: dict[tuple, list[tuple[date, dict, str]]] = defaultdict(list)
        for r in rows:
            if r.label != "true" or not r.atoms:
                continue
            for a in r.atoms:
                if "op" not in a.value:
                    self.by_key[a.key].append((_d(r.date), a.value, r.id))
        for obs in self.by_key.values():
            obs.sort(key=lambda o: o[0])

    def candidates(self, key, exclude: str | None = None):
        return [o for o in self.by_key.get(key, []) if o[2] != exclude]


class _LookupBaseline(Model):
    """Shared logic: pick anchor observation(s) for each atom, compare, combine per row."""

    def pick(self, obs, d: date):
        raise NotImplementedError

    def fit(self, rows, lexicon, names: NameNormalizer):
        self.names = names
        self.index = AnchorIndex(rows)
        branch_labels: dict[str, Counter] = defaultdict(Counter)
        for r in rows:
            if r.label:
                branch_labels[self._decide(r, exclude=r.id)[0]][r.label] += 1
        self.branch_probs = {b: smoothed(c) for b, c in branch_labels.items()}
        self.prior = smoothed(Counter(r.label for r in rows if r.label))
        return self

    def _atom(self, a: Atom, d: date, exclude):
        obs = self.index.candidates(a.key, exclude)
        if not obs:
            return "no_evidence", None
        chosen = self.pick(obs, d)
        verdicts = [satisfies(a.value, o[1], self.names) for o in chosen]
        anchor = [(o[0].isoformat(), o[1], o[2]) for o in chosen]
        if any(v is True for v in verdicts):
            return "match", anchor
        if any(v is False for v in verdicts):
            return "mismatch", anchor
        return "no_evidence", anchor

    def _decide(self, row: Row, exclude=None):
        if not row.atoms:
            return "unparsed", []
        d = _d(row.date)
        results = [(a, *self._atom(a, d, exclude)) for a in row.atoms]
        kinds = [k for _, k, _ in results]
        branch = ("any_mismatch" if "mismatch" in kinds
                  else "all_match" if all(k == "match" for k in kinds)
                  else "no_evidence")
        return branch, results

    def predict(self, row):
        branch, results = self._decide(row)
        label = "false" if branch == "any_mismatch" else "true"
        probs = self.branch_probs.get(branch, self.prior)
        trace = {"rule": self.name, "branch": branch,
                 "atoms": [{"subject": a.subject, "attribute": a.attribute, "value": a.value,
                            "verdict": k, "anchors": anc} for a, k, anc in results]}
        return Prediction(label, dict(probs), trace)


class LatestBefore(_LookupBaseline):
    """The most recent labelled-true value on or before the row's date (else the earliest after)."""
    name = "latest_before"

    def pick(self, obs, d):
        dates = [o[0] for o in obs]
        i = bisect.bisect_right(dates, d)
        target = dates[i - 1] if i > 0 else dates[0]
        return [o for o in obs if o[0] == target]


class Nearest(_LookupBaseline):
    """The labelled-true value(s) closest in time to the row's date, either side."""
    name = "nearest"

    def pick(self, obs, d):
        best = min(abs((o[0] - d).days) for o in obs)
        return [o for o in obs if abs((o[0] - d).days) == best]
