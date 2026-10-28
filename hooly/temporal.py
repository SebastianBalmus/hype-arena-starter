"""E1: four-date temporal windows (after TAC KBP temporal slot filling).

Per fact (subject, attribute), the labelled-true observations are sorted by date and
grouped into runs of one value. A run's first and last sighting are its certain core
[T2, T3]; the gap to the neighbouring run is where the change happened [T3, T4].

Each atom of a row gets a *state* from where its date falls and whether its value
matches. The probability of each label per state is measured on the training rows, each
with its own page's anchors left out, so no number here is hand-tuned.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date

from .data import LABELS, Atom, Row
from .models import Model, Prediction, smoothed
from .values import NameNormalizer, matches, normalize, satisfies

NEAR, MID = 14, 60  # days: distance buckets around the observed range


def _d(s: str) -> date:
    return date.fromisoformat(s)


def _bucket(days: int) -> str:
    return "near" if days <= NEAR else "mid" if days <= MID else "far"


@dataclass
class Run:
    values: list[dict]
    start: date
    end: date
    rows: list[str] = field(default_factory=list)

    def verdict(self, claimed: dict, names) -> bool | None:
        vs = [satisfies(claimed, v, names) for v in self.values]
        if any(v is True for v in vs):
            return True
        if vs and all(v is False for v in vs):
            return False
        return None

    def describe(self):
        return {"value": self.values[0], "from": self.start.isoformat(), "to": self.end.isoformat(),
                "rows": self.rows[:5]}


class Timeline:
    """Runs of one fact's labelled-true values, oldest first."""

    def __init__(self, obs, names: NameNormalizer):
        self.names = names
        self.runs: list[Run] = []
        self.multi = False
        by_date = defaultdict(list)
        for d, v, rid in obs:
            by_date[d].append((v, rid))
        for d in sorted(by_date):
            vals = by_date[d]
            # Different values confirmed true on the same day: a set-valued attribute
            # (several founders, several co-investors), not a timeline.
            if any(not matches(a[0], b[0], names) for a in vals for b in vals):
                self.multi = True
            for v, rid in vals:
                cur = self.runs[-1] if self.runs else None
                if cur and any(matches(v, x, names) for x in cur.values):
                    cur.end = d
                    cur.values.append(v)
                    cur.rows.append(rid)
                else:
                    self.runs.append(Run([v], d, d, [rid]))

    def state(self, claimed: dict, d: date) -> tuple[str, dict]:
        runs, names = self.runs, self.names
        if not runs:
            return "no_anchor", {}
        if self.multi:
            seen = [r.verdict(claimed, names) for r in runs]
            st = "multi_match" if True in seen else "multi_mismatch" if False in seen else "incomparable"
            return st, {"values": [r.describe() for r in runs]}
        for i, r in enumerate(runs):
            if r.start <= d <= r.end:
                v = r.verdict(claimed, names)
                st = "core_match" if v else "core_mismatch" if v is False else "incomparable"
                return st, {"core": r.describe()}
        if d < runs[0].start:
            first = runs[0]
            v = first.verdict(claimed, names)
            b = _bucket((first.start - d).days)
            if v is None:
                return "incomparable", {}
            return f"before_{'match' if v else 'mismatch'}_{b}", {"next": first.describe()}
        if d > runs[-1].end:
            last = runs[-1]
            v = last.verdict(claimed, names)
            b = _bucket((d - last.end).days)
            if v is None:
                return "incomparable", {}
            return f"after_{'match' if v else 'mismatch'}_{b}", {"previous": last.describe()}
        # In a gap between two runs: the value changed somewhere in (prev.end, nxt.start).
        i = max(k for k, r in enumerate(runs) if r.end < d)
        prev, nxt = runs[i], runs[i + 1]
        t = (d - prev.end).days / max(1, (nxt.start - prev.end).days)
        pos = "early" if t < 1 / 3 else "mid" if t < 2 / 3 else "late"
        trace = {"previous": prev.describe(), "next": nxt.describe(), "position": round(t, 2)}
        vp, vn = prev.verdict(claimed, names), nxt.verdict(claimed, names)
        if vp:
            return f"gap_prev_{pos}", trace
        if vn:
            return f"gap_next_{pos}", trace
        if vp is None and vn is None:
            return "incomparable", trace
        return ("gap_neither_between" if self._between(claimed, prev, nxt) else "gap_neither_outside"), trace

    def _between(self, claimed, prev: Run, nxt: Run) -> bool:
        """Is the claimed amount strictly between the two neighbours (an unseen middle value)?"""
        try:
            c, a, b = (normalize(x, self.names) for x in (claimed, prev.values[0], nxt.values[0]))
        except ValueError:
            return False
        if not (c.dim == a.dim == b.dim) or c.dim[0] not in ("quantity", "money", "percent", "duration"):
            return False
        lo, hi = sorted([a.value, b.value])
        return lo < c.value < hi


def _coarse(state: str) -> str:
    parts = state.split("_")
    return "_".join(parts[:2]) if len(parts) > 2 else state


class TemporalWindows(Model):
    name = "windows"

    def __init__(self, m: float = 5.0):
        self.m = m  # strength of the back-off from fine to coarse state statistics

    def fit(self, rows, lexicon, names):
        self.names = names
        self.anchors = defaultdict(list)  # key -> [(date, value, row_id, page)]
        for r in rows:
            if r.label == "true" and r.atoms:
                for a in r.atoms:
                    if "op" not in a.value:
                        self.anchors[a.key].append((_d(r.date), a.value, r.id, r.page))
        self._cache = {}
        prior_counts = Counter(r.label for r in rows if r.label)
        self.prior = smoothed(prior_counts)
        self._prepare(rows)
        self.counts = defaultdict(Counter)
        for r in rows:  # single-atom rows: the row label is the atom's label
            if r.label and r.atoms and len(r.atoms) == 1:
                st, _ = self._state(r.atoms[0], _d(r.date), exclude_page=r.page)
                for level in self.chain(st):
                    self.counts[level][r.label] += 1
        self.state_counts = {s: dict(c) for s, c in self.counts.items()}
        return self

    def _prepare(self, rows):
        """Hook for subclasses: statistics needed before states can be computed."""

    def chain(self, state: str) -> list[str]:
        """From the most specific to the most general description of a state."""
        coarse = _coarse(state)
        return [state] if coarse == state else [state, coarse]

    def probs_for(self, state: str) -> dict:
        probs = self.prior
        for level in reversed(self.chain(state)):
            if level in self.counts:
                probs = self._mix(self.counts[level], probs)
        return probs

    def _mix(self, counts: Counter, parent: dict) -> dict:
        n = sum(counts.values())
        return {l: (counts.get(l, 0) + self.m * parent[l]) / (n + self.m) for l in LABELS}

    def _timeline(self, key, exclude_page):
        ck = (key, exclude_page)
        if ck not in self._cache:
            obs = [(d, v, rid) for d, v, rid, pg in self.anchors.get(key, []) if pg != exclude_page]
            self._cache[ck] = Timeline(obs, self.names)
        return self._cache[ck]

    def _state(self, atom: Atom, d: date, exclude_page=None):
        return self._timeline(atom.key, exclude_page).state(atom.value, d)

    def atom_probs(self, atom: Atom, d: date, exclude_page=None):
        st, trace = self._state(atom, d, exclude_page)
        return st, self.probs_for(st), trace

    def predict(self, row: Row) -> Prediction:
        probs, atoms = self.row_probs(row)
        label = max(probs, key=probs.get)
        return Prediction(label, probs, {"rule": self.name, "atoms": atoms})

    def row_probs(self, row: Row, exclude_page=None):
        """Label probabilities for a row from its atoms' states (atoms assumed independent)."""
        if not row.atoms:
            return dict(self.prior), [{"state": "unparsed"}]
        d = _d(row.date)
        p_true, p_not_unsure, atoms = 1.0, 1.0, []
        atom_q = row.meta.get("atom_q") or {}
        for a in row.atoms:
            st, pr, tr = self.atom_probs(a, d, exclude_page)
            q = atom_q.get(a.key, 1.0)
            if q < 1.0:  # an unsure parse only partly counts (#4)
                pr = {l: q * pr[l] + (1 - q) * self.prior[l] for l in LABELS}
                tr = {**tr, "parse_confidence": round(q, 3)}
            p_true *= pr["true"]
            p_not_unsure *= 1 - pr["unsure"]
            atoms.append({"subject": a.subject, "attribute": a.attribute, "value": a.value,
                          "state": st, "p_true": round(pr["true"], 3), **tr})
        p_unsure = 1 - p_not_unsure
        p_false = max(0.0, 1 - p_true - p_unsure)
        z = p_true + p_false + p_unsure
        probs = {"true": p_true / z, "false": p_false / z, "unsure": p_unsure / z}
        return probs, atoms


_AMOUNT_KINDS = ("quantity", "money", "percent", "duration")


def _version_tuple(v: str):
    try:
        return tuple(int(x) for x in str(v).split("."))
    except ValueError:
        return None


class TemporalWindowsPlus(TemporalWindows):
    """E1b: windows + how stable the fact is + whether the value fits the attribute's trend.

    stability  stable    one value seen over >= 60 days (specs, launch dates, ...)
               volatile  the fact has changed at least once
               thin      one value, seen over a short span: not enough to tell
    direction  fits / against the attribute's usual direction of change (learned per
               attribute name from the anchors: e.g. users go up, latency goes down), for
               rows before the first or after the last sighting, or outside both neighbours.
    """
    name = "windows_plus"
    STABLE_SPAN = 60

    def _prepare(self, rows):
        ups, downs = Counter(), Counter()
        for key, obs in self.anchors.items():
            tl = Timeline([(d, v, rid) for d, v, rid, _ in obs], self.names)
            if tl.multi:
                continue
            for a, b in zip(tl.runs, tl.runs[1:]):
                step = self._order(b.values[0], a.values[0])
                if step == 1:
                    ups[key[1]] += 1
                elif step == -1:
                    downs[key[1]] += 1
        self.trend = {}
        for attr in set(ups) | set(downs):
            n, frac = ups[attr] + downs[attr], ups[attr] / (ups[attr] + downs[attr])
            self.trend[attr] = "up" if frac >= 0.7 else "down" if frac <= 0.3 else "mixed"

    def _order(self, x: dict, y: dict) -> int | None:
        """+1 if x > y, -1 if x < y, 0 if equal, None if they have no order."""
        if x["kind"] == "version" and y["kind"] == "version":
            a, b = _version_tuple(x["value"]), _version_tuple(y["value"])
            if a is None or b is None:
                return None
            return (a > b) - (a < b)
        if x["kind"] not in _AMOUNT_KINDS:
            return None
        nx, ny = normalize(x, self.names), normalize(y, self.names)
        if nx.dim != ny.dim:
            return None
        return (nx.value > ny.value) - (nx.value < ny.value)

    def _state(self, atom, d, exclude_page=None):
        tl = self._timeline(atom.key, exclude_page)
        base, trace = tl.state(atom.value, d)
        if not tl.runs or tl.multi:
            return base, trace
        span = (tl.runs[-1].end - tl.runs[0].start).days
        stability = ("volatile" if len(tl.runs) > 1
                     else "stable" if span >= self.STABLE_SPAN else "thin")
        trend = "up" if atom.value["kind"] == "version" else self.trend.get(atom.attribute)
        direction = "na"
        ref, later = None, None
        if base.startswith("before_mismatch"):
            ref, later = tl.runs[0].values[0], True     # the claimed value is earlier than ref
        elif base.startswith("after_mismatch"):
            ref, later = tl.runs[-1].values[0], False   # the claimed value is later than ref
        elif base == "gap_neither_outside":
            ref, later = tl.runs[-1].values[0], None
        if ref is not None and trend in ("up", "down"):
            o = self._order(atom.value, ref)
            if o:
                if later is None:
                    direction = "outside"
                else:
                    # going up over time: earlier values are lower, later values higher
                    expected = (-1 if later else 1) * (1 if trend == "up" else -1)
                    direction = "fits" if o == expected else "against"
        elif ref is not None:
            direction = "no_trend"
        trace = {**trace, "stability": stability, "direction": direction, "trend": trend}
        return f"{base}|{stability}|{direction}", trace

    def chain(self, state):
        if "|" not in state:
            return super().chain(state)
        base, stability, direction = state.split("|")
        return [state, f"{base}|{stability}"] + super().chain(base)


class ConservativeWindows(TemporalWindows):
    """E7: the evidence policy's verdict rule on top of E1's timelines.

    An atom is proved true inside a core of its value and proved false inside a core of
    another value; a row is false if any atom is proved false, true if all are proved true,
    otherwise unsure."""
    name = "conservative"

    def predict(self, row):
        probs, atoms = self.row_probs(row)
        states = [a.get("state", "unparsed") for a in atoms]
        if any(s.startswith("core_mismatch") for s in states):
            label = "false"
        elif states and all(s.startswith("core_match") for s in states):
            label = "true"
        else:
            label = "unsure"
        return Prediction(label, probs, {"rule": self.name, "atoms": atoms})


class TemporalWindowsShape(TemporalWindowsPlus):
    """#3 lie shapes: for a value that disagrees with the timeline, how does it relate to the
    values the fact actually took?

    seen        it equals a value the fact held at some *other* time (stale / premature)
    up / down   amounts: ratio to the reference value: ~ (within 10%), small (<1.5x), big
    round       the claimed number is rounder than the reference ("5,000" vs "2,600")
    """
    name = "windows_shape"

    def _state(self, atom, d, exclude_page=None):
        st, trace = super()._state(atom, d, exclude_page)
        base = st.split("|")[0]
        if "mismatch" not in base and "neither" not in base and not base.startswith(("gap_prev", "gap_next")):
            return st, trace
        tl = self._timeline(atom.key, exclude_page)
        if not tl.runs or tl.multi:
            return st, trace
        ref = (trace.get("core") or trace.get("previous") or trace.get("next") or {}).get("value")
        others = [r for r in tl.runs if ref is None or not any(v == ref for v in r.values)]
        seen = any(r.verdict(atom.value, self.names) for r in others)
        shape = "seen" if seen else "new"
        if ref is not None and not seen and atom.value["kind"] in ("quantity", "money", "percent", "duration"):
            try:
                c, r0 = normalize(atom.value, self.names), normalize(ref, self.names)
                if c.dim == r0.dim and r0.value > 0 and c.value > 0:
                    ratio = float(c.value / r0.value)
                    if 0.9 <= ratio <= 1.1:
                        shape = "near"
                    elif ratio > 1:
                        shape = "up_small" if ratio < 1.5 else "up_big"
                    else:
                        shape = "down_small" if ratio > 1 / 1.5 else "down_big"
                    if _roundness(atom.value["value"]) > _roundness(ref["value"]):
                        shape += "_round"
            except (ValueError, KeyError, TypeError):
                pass
        trace = {**trace, "shape": shape}
        return f"{st}|{shape}" if "|" in st else f"{st}|na|na|{shape}", trace

    def chain(self, state):
        parts = state.split("|")
        if len(parts) == 4:
            return [state] + super().chain("|".join(parts[:3]))
        return super().chain(state)


def _roundness(x) -> int:
    """Trailing zeros of the integer part (5000 -> 3, 2600 -> 2, 4970 -> 1)."""
    try:
        n = int(abs(float(x)))
    except (TypeError, ValueError):
        return 0
    if n == 0 or float(x) != int(float(x)):
        return 0
    k = 0
    while n % 10 == 0:
        n //= 10
        k += 1
    return k
