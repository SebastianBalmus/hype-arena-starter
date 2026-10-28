"""Dated evidence store: observations -> value intervals, page reliability, fact checks."""
from dataclasses import dataclass
from datetime import date as _date

from arena.model import AS_OF, Fact, Row
from arena.values import satisfies, values_equal


@dataclass
class Interval:
    value: dict
    start: str
    end: str
    pages: set
    trusted: bool
    weight: float


@dataclass
class AtomCheck:
    state: str
    status: str
    support: int
    trusted: bool
    days: int


def _days(a: str, b: str) -> int:
    try:
        return abs((_date.fromisoformat(a) - _date.fromisoformat(b)).days)
    except ValueError:
        return 0


def _valid_date(d: str) -> bool:
    if not isinstance(d, str) or len(d) != 10:
        return False
    try:
        return _date.fromisoformat(d).isoformat() == d
    except ValueError:
        return False


def _is_bound(v: dict) -> bool:
    return v.get("op") not in (None, "eq")


class EvidenceStore:
    def __init__(self):
        self._rows: list[tuple[Row, list[Fact]]] = []
        self._obs: dict = {}      # key -> [(date, page, value, weight_kind, conf)]
        self._refs: dict = {}     # key -> [(date, value)]
        self._ivs: dict = {}      # key -> [Interval]
        self.page_reliability: dict[int, float] = {}

    def add_row(self, row: Row, facts: list[Fact]) -> None:
        # a row without a valid ISO date keeps its label for page reliability but contributes no facts
        self._rows.append((row, list(facts) if _valid_date(row.date) else []))

    # ---- build -----------------------------------------------------
    def build(self, iterations: int = 3) -> None:
        self._obs, self._refs = {}, {}
        counts: dict[int, list[int]] = {}   # page -> [n_true, n_labelled]
        pages = set()
        for row, facts in self._rows:
            pages.add(row.page)
            if row.label in ("true", "false"):
                c = counts.setdefault(row.page, [0, 0])
                c[1] += 1
                c[0] += row.label == "true"
            for f in facts if _valid_date(row.date) else ():
                key = (f.subject, f.attribute)
                if row.label == "false":
                    if len(facts) == 1:
                        self._refs.setdefault(key, []).append((row.date, f.value))
                    continue
                if _is_bound(f.value):
                    continue
                self._obs.setdefault(key, []).append((row.date, row.page, f.value, row.label, f.confidence))
        rel = {p: 0.8 for p in pages}
        labelled = set()
        for p, (t, n) in counts.items():
            rel[p] = (t + 1) / (n + 2)
            labelled.add(p)
        self.page_reliability = rel
        self._compute()
        for _ in range(iterations):
            agree: dict[int, int] = {}
            total: dict[int, int] = {}
            for row, facts in self._rows:
                if row.page in labelled or row.label in ("true", "false"):
                    continue
                for f in facts:
                    total[row.page] = total.get(row.page, 0) + 1
                    if self.check(f, row.date).state == "true":
                        agree[row.page] = agree.get(row.page, 0) + 1
            for p, n in total.items():
                rel[p] = (agree.get(p, 0) + 1.6) / (n + 2)
            self._compute()

    def _compute(self) -> None:
        rel = self.page_reliability
        self._ivs = {}
        for key, obs in self._obs.items():
            by_date: dict[str, list] = {}
            for d, page, val, label, conf in obs:
                if label == "true":
                    w, tr = 3.0, True
                elif label == "unsure":
                    w, tr = 0.3 * rel.get(page, 0.8), False
                else:
                    w, tr = rel.get(page, 0.8) * conf, False
                by_date.setdefault(d, []).append((page, val, w, tr))
            winners = []   # (date, value, weight, trusted, pages)
            for d in sorted(by_date):
                groups: list[list] = []
                for ob in by_date[d]:
                    for g in groups:
                        if values_equal(g[0][1], ob[1]):
                            g.append(ob)
                            break
                    else:
                        groups.append([ob])
                if any(ob[3] for g in groups for ob in g):
                    best = None
                    for g in groups:
                        tw = sum(ob[2] for ob in g if ob[3])
                        if tw and (best is None or tw > best[0]):
                            best = (tw, g)
                    g = best[1]
                    winners.append((d, g[0][1], best[0], True, {ob[0] for ob in g}))
                else:
                    g = max(groups, key=lambda g: sum(ob[2] for ob in g))
                    winners.append((d, g[0][1], sum(ob[2] for ob in g), False, {ob[0] for ob in g}))
            kept = []
            for i, w in enumerate(winners):
                if (not w[3] and len(w[4]) == 1 and 0 < i < len(winners) - 1):
                    p, n = winners[i - 1][1], winners[i + 1][1]
                    if values_equal(p, n) and not values_equal(p, w[1]):
                        continue
                kept.append(w)
            ivs: list[Interval] = []
            for d, val, wt, tr, pg in kept:
                if ivs and values_equal(ivs[-1].value, val):
                    iv = ivs[-1]
                    iv.end = d
                    iv.pages |= pg
                    iv.trusted = iv.trusted or tr
                    iv.weight += wt
                else:
                    ivs.append(Interval(val, d, d, set(pg), tr, wt))
            if ivs:
                self._ivs[key] = ivs

    # ---- queries ---------------------------------------------------
    def intervals(self, subject: str, attribute: str) -> list[Interval]:
        return self._ivs.get((subject, attribute), [])

    def current(self, subject: str, attribute: str):
        ivs = self.intervals(subject, attribute)
        return ivs[-1] if ivs else None

    def keys(self) -> list[tuple[str, str]]:
        return sorted(set(self._ivs) | set(self._refs))

    def check(self, fact: Fact, date: str) -> AtomCheck:
        key = (fact.subject, fact.attribute)
        for d, v in self._refs.get(key, ()):
            if d == date and values_equal(v, fact.value):
                return AtomCheck("false", "in", 1, True, 0)
        ivs = self._ivs.get(key)
        if not ivs:
            return AtomCheck("unknown", "none", 0, False, 0)
        iv, status, days = None, "", 0
        if date < ivs[0].start:
            iv, status, days = ivs[0], "before", _days(ivs[0].start, date)
        elif date > ivs[-1].end:
            iv, status, days = ivs[-1], "after", _days(date, ivs[-1].end)
        else:
            for i, c in enumerate(ivs):
                if c.start <= date <= c.end:
                    iv, status = c, "in"
                    break
                if i + 1 < len(ivs) and c.end < date < ivs[i + 1].start:
                    nxt = ivs[i + 1]
                    status = "gap"
                    if values_equal(c.value, nxt.value):
                        iv, days = c, _days(date, c.end)
                    else:
                        days = min(_days(date, c.end), _days(nxt.start, date))
                    break
        if iv is None:
            return AtomCheck("unknown", status or "gap", 0, False, days)
        state = "true" if satisfies(iv.value, fact.value) else "false"
        return AtomCheck(state, status, len(iv.pages), iv.trusted, days)
