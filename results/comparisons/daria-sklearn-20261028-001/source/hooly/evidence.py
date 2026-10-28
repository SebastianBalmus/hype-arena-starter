"""E3: unlabelled rows as votes (truth-discovery style, without source ids).

For an atom, rows of *other pages* that state the same fact near the same date vote for or
against its value. Single-fact rows labelled false that state the same value near the same
date mark it as a known lie (stale values get repeated). Both become modifiers of the E1b
state, so their weight is measured on the training rows like everything else.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date

from .temporal import TemporalWindowsPlus, _coarse, _d
from .values import DIFFERENT, compare, matches


class WindowsEvidence(TemporalWindowsPlus):
    name = "windows_evidence"
    WINDOW = 45      # days: how close an unlabelled vote must be
    NEG_WINDOW = 60  # days: how close a known lie must be

    def fit(self, rows, lexicon, names):
        self.pool = defaultdict(list)   # key -> [(date, value, page)] from unlabelled rows
        self.lies = defaultdict(list)   # key -> [(date, value, page)] from single-fact false rows
        for r in rows:
            if not r.atoms:
                continue
            if r.label is None:
                for a in r.atoms:
                    if "op" not in a.value:
                        self.pool[a.key].append((_d(r.date), a.value, r.page))
            elif r.label == "false" and len(r.atoms) == 1 and "op" not in r.atoms[0].value:
                a = r.atoms[0]
                self.lies[a.key].append((_d(r.date), a.value, r.page))
        return super().fit(rows, lexicon, names)

    def votes(self, atom, d: date, page):
        agree, disagree = set(), set()
        for od, v, pg in self.pool.get(atom.key, []):
            if pg == page or abs((od - d).days) > self.WINDOW:
                continue
            if "op" in atom.value:
                continue
            rel = compare(atom.value, v, self.names)
            if matches(atom.value, v, self.names):
                agree.add(pg)
            elif rel == DIFFERENT:
                disagree.add(pg)
        known_lie = any(pg != page and abs((od - d).days) <= self.NEG_WINDOW and matches(atom.value, v, self.names)
                        for od, v, pg in self.lies.get(atom.key, []))
        return len(agree), len(disagree), known_lie

    def _state(self, atom, d, exclude_page=None):
        st, trace = super()._state(atom, d, exclude_page)
        page = exclude_page if exclude_page is not None else self._current_page
        a, n, lie = self.votes(atom, d, page)
        u = "u0" if a == n == 0 else "u+" if a > n else "u-" if n > a else "u="
        neg = "lie" if lie else "-"
        parts = st.split("|")
        base, rest = parts[0], parts[1:]
        trace = {**trace, "votes_agree": a, "votes_disagree": n, "repeats_known_lie": lie}
        return "|".join([base, u, neg] + rest), trace

    def chain(self, state):
        parts = state.split("|")
        out = ["|".join(parts[:k]) for k in range(len(parts), 1, -1)]
        base = parts[0]
        coarse = _coarse(base)
        return out + [base] + ([coarse] if coarse != base else [])

    def row_probs(self, row, exclude_page=None):
        self._current_page = row.page
        return super().row_probs(row, exclude_page)


class WindowsVotesStack(TemporalWindowsPlus):
    """E3b: the same votes, weighed by a small stacker against the E1b verdict instead of
    splitting the states further (which spreads the training rows too thin)."""
    name = "windows_votes"

    def fit(self, rows, lexicon, names):
        self._ev = WindowsEvidence()
        self._ev.pool, self._ev.lies = defaultdict(list), defaultdict(list)
        WindowsEvidence.fit(self._ev, rows, lexicon, names)  # builds pools (+ its own fit, unused)
        super().fit(rows, lexicon, names)
        from .learn import LogReg
        X, y = [], []
        for r in rows:
            if r.label:
                base, _ = self.row_probs(r, exclude_page=r.page)
                X.append(self._features(r, base))
                y.append(int(r.label == "false"))
        self.stacker = LogReg(l2=1e-3, epochs=60).fit(X, y)
        return self

    def _features(self, row, base):
        from math import log1p
        from .learn import logit
        agree = disagree = lies = 0
        if row.atoms:
            d = _d(row.date)
            for a in row.atoms:
                ag, dis, lie = self._ev.votes(a, d, row.page)
                agree, disagree, lies = agree + ag, disagree + dis, lies + lie
        b = logit(base["false"] / max(1e-9, base["false"] + base["true"]))
        return {"timeline": b, "agree": log1p(agree), "disagree": log1p(disagree),
                "known_lie": float(lies > 0), "agree_x_timeline": log1p(agree) * b / 4,
                "disagree_x_timeline": log1p(disagree) * b / 4}

    def predict(self, row):
        from .models import Prediction
        base, atoms = self.row_probs(row)
        x = self._features(row, base)
        pf = self.stacker.predict_proba(x)
        pu = min(base["unsure"], 1 - pf)
        probs = {"false": pf, "unsure": pu, "true": max(0.0, 1 - pf - pu)}
        return Prediction(max(probs, key=probs.get), probs,
                          {"rule": self.name, "atoms": atoms, "votes": {k: round(v, 2) for k, v in x.items()}})
