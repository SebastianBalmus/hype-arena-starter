"""#1 Probabilistic timelines: a hidden Markov model per fact, fitted with EM.

Hidden state: which value of a fact is true at each observed date (the fact's distinct
values, grouped so 14 days == 2 weeks, 2026-01 ~ 2026-01-12). Between consecutive dates the
value stays with probability exp(-gap / tau), otherwise it moves to another value.

Emissions per observation:
  labelled true        pins the state to its value
  labelled false (one fact)  rules its value out
  everything else      states the true value with probability alpha, otherwise another one

EM learns alpha (how often an unlabelled row tells the truth) and tau (typical time between
changes) from all rows at once; forward-backward then gives, for each row, the probability
that the value it states was the true one on its date. Unlabelled rows are evidence here,
not just anchors: the more rows the runtime has, the more this model sees.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import date

from .data import LABELS, Row
from .models import Model, Prediction, smoothed
from .values import matches, satisfies

HARD = 1e-6


def _d(s: str) -> date:
    return date.fromisoformat(s)


class FactChain:
    def __init__(self, obs, names):
        """obs: [(date, value, kind, row_id)] with kind in {"anchor", "lie", "free"}."""
        self.names = names
        vals = sorted({repr(sorted(v.items())): v for _, v, _, _ in obs}.values(),
                      key=lambda v: -len(str(v.get("value"))))  # precise dates first
        self.classes: list[dict] = []
        for v in vals:
            if not any(matches(v, c, names) for c in self.classes):
                self.classes.append(v)
        self.K = len(self.classes)
        self.dates = sorted({d for d, _, _, _ in obs})
        self.t_of = {d: i for i, d in enumerate(self.dates)}
        self.obs_at = defaultdict(list)  # t -> [(class index, kind, row_id)]
        self.cls_of_row = {}
        for d, v, kind, rid in obs:
            j = self.cls(v)
            self.obs_at[self.t_of[d]].append((j, kind, rid))
            self.cls_of_row[rid] = j
        anchors_by_t = defaultdict(set)
        for t, lst in self.obs_at.items():
            for j, kind, _ in lst:
                if kind == "anchor":
                    anchors_by_t[t].add(j)
        self.multi = any(len(s) > 1 for s in anchors_by_t.values())  # set-valued fact
        self.gamma = None

    def cls(self, v):
        for j, c in enumerate(self.classes):
            if matches(v, c, self.names):
                return j
        return None

    def emissions(self, t, alpha):
        e = [1.0] * self.K
        lie = (1 - alpha) / max(1, self.K - 1)
        for j, kind, _ in self.obs_at.get(t, []):
            for k in range(self.K):
                if kind == "anchor":
                    e[k] *= 1.0 if k == j else HARD
                elif kind == "lie":
                    e[k] *= HARD if k == j else 1.0
                else:
                    e[k] *= alpha if k == j else (lie if self.K > 1 else 1 - alpha)
        return e

    def run(self, alpha, tau):
        """Forward-backward; returns (expected changes, total gap days, free-obs truth mass, n free)."""
        K, T = self.K, len(self.dates)
        E = [self.emissions(t, alpha) for t in range(T)]
        stay = [1.0] + [math.exp(-max(1, (self.dates[t] - self.dates[t - 1]).days) / tau) for t in range(1, T)]
        fw, scale = [], []
        for t in range(T):
            if t == 0:
                a = [E[0][k] / K for k in range(K)]
            else:
                prev, s = fw[-1], stay[t]
                tot = sum(prev)
                move = (1 - s) / max(1, K - 1)
                a = [(prev[k] * s + (tot - prev[k]) * move if K > 1 else prev[k]) * E[t][k] for k in range(K)]
            z = sum(a) or 1e-300
            fw.append([x / z for x in a])
            scale.append(z)
        bw = [[1.0] * K for _ in range(T)]
        for t in range(T - 2, -1, -1):
            s = stay[t + 1]
            move = (1 - s) / max(1, K - 1)
            nb = [E[t + 1][k] * bw[t + 1][k] for k in range(K)]
            tot = sum(nb)
            b = [(nb[k] * s + (tot - nb[k]) * move if K > 1 else nb[k]) for k in range(K)]
            z = sum(b) or 1e-300
            bw[t] = [x / z for x in b]
        self.gamma = []
        for t in range(T):
            g = [fw[t][k] * bw[t][k] for k in range(K)]
            z = sum(g) or 1e-300
            self.gamma.append([x / z for x in g])
        changes, days = 0.0, 0.0
        for t in range(1, T):
            s = stay[t]
            days += max(1, (self.dates[t] - self.dates[t - 1]).days)
            if K == 1:
                continue
            stay_mass = sum(fw[t - 1][k] * s * E[t][k] * bw[t][k] for k in range(K))
            move = (1 - s) / (K - 1)
            all_mass = sum(fw[t - 1][i] * (s if i == k else move) * E[t][k] * bw[t][k] for i in range(K) for k in range(K))
            changes += 1 - stay_mass / (all_mass or 1e-300)
        truth, n = 0.0, 0
        for t, lst in self.obs_at.items():
            for j, kind, _ in lst:
                if kind == "free":
                    truth += self.gamma[t][j]
                    n += 1
        return changes, days, truth, n

    def p_true(self, value, d: date) -> float | None:
        t = self.t_of.get(d)
        if t is None or self.gamma is None:
            return None
        p, known = 0.0, False
        for k, c in enumerate(self.classes):
            v = satisfies(value, c, self.names)
            if v is not None:
                known = True
                if v:
                    p += self.gamma[t][k]
        return p if known else None


class TimelineHMM(Model):
    name = "hmm"

    def __init__(self, iters: int = 8, alpha: float = 0.8, tau: float = 90.0):
        self.iters, self.alpha, self.tau = iters, alpha, tau

    def fit(self, rows, lexicon, names):
        self.names = names
        self.prior = smoothed(Counter(r.label for r in rows if r.label))
        obs = defaultdict(list)
        for r in rows:
            if not r.atoms:
                continue
            kind = ("anchor" if r.label == "true" else
                    "lie" if r.label == "false" and len(r.atoms) == 1 else "free")
            for a in r.atoms:
                if "op" in a.value:
                    continue
                obs[a.key].append((_d(r.date), a.value, kind, r.id))
        self.chains = {k: FactChain(o, names) for k, o in obs.items()}
        self.history = []
        for _ in range(self.iters):
            ch = dy = tr = n = 0.0
            for c in self.chains.values():
                if c.multi:
                    continue
                a, b, x, y = c.run(self.alpha, self.tau)
                ch, dy, tr, n = ch + a, dy + b, tr + x, n + y
            self.alpha = min(0.99, max(0.5, tr / n)) if n else self.alpha
            self.tau = min(720.0, max(15.0, dy / ch)) if ch else self.tau
            self.history.append((round(self.alpha, 3), round(self.tau, 1)))
        for c in self.chains.values():
            if not c.multi:
                c.run(self.alpha, self.tau)
        return self

    def atom_p(self, atom, d):
        c = self.chains.get(atom.key)
        if c is None:
            return None, "no_chain"
        if c.multi:
            anchors = [c.classes[j] for lst in c.obs_at.values() for j, kind, _ in lst if kind == "anchor"]
            hit = any(satisfies(atom.value, v, self.names) for v in anchors)
            return (0.95 if hit else None), "set_valued"
        p = c.p_true(atom.value, d)
        return p, "posterior" if p is not None else "incomparable"

    def predict(self, row: Row):
        if not row.atoms:
            return Prediction("true", dict(self.prior), {"rule": self.name, "state": "unparsed"})
        d = _d(row.date)
        p_true, atoms = 1.0, []
        prior_t = self.prior["true"] / (self.prior["true"] + self.prior["false"])
        q = row.meta.get("atom_q") or {}
        for a in row.atoms:
            p, how = self.atom_p(a, d)
            if p is None:
                p = prior_t
            w = q.get(a.key, 1.0)
            p = w * p + (1 - w) * prior_t
            p_true *= p
            atoms.append({"subject": a.subject, "attribute": a.attribute, "value": a.value,
                          "p_true": round(p, 3), "how": how})
        pu = self.prior["unsure"]
        probs = {"true": p_true * (1 - pu), "false": (1 - p_true) * (1 - pu), "unsure": pu}
        return Prediction(max(probs, key=probs.get), probs,
                          {"rule": self.name, "atoms": atoms, "alpha": round(self.alpha, 3), "tau_days": round(self.tau, 1)})


class HybridStack(Model):
    """windows_shape (hand-made timeline states + lie shapes) and the HMM (global probabilistic
    consistency), weighed by a small stacker. The HMM scores the stacker learns from are
    out-of-fold: each training row's own label is hidden from the HMM that scores it."""
    name = "hybrid"

    def fit(self, rows, lexicon, names):
        from .folds import grouped_folds
        from .learn import LogReg
        from .temporal import TemporalWindowsShape
        from dataclasses import replace
        self.shape = TemporalWindowsShape().fit(rows, lexicon, names)
        labelled = [r for r in rows if r.label]
        oof = {}
        for test_ids in grouped_folds(labelled, k=5, seed=1):
            view = [replace(r, label=None) if r.id in test_ids else r for r in rows]
            h = TimelineHMM().fit(view, lexicon, names)
            for r in view:
                if r.id in test_ids:
                    oof[r.id] = h.predict(r).probs
        self.hmm = TimelineHMM().fit(rows, lexicon, names)
        X, y = [], []
        for r in labelled:
            base, _ = self.shape.row_probs(r, exclude_page=r.page)
            X.append(self._features(base, oof[r.id]))
            y.append(int(r.label == "false"))
        self.stacker = LogReg(l2=1e-3, epochs=60).fit(X, y)
        return self

    @staticmethod
    def _features(base, hp):
        from .learn import logit
        a = logit(base["false"] / max(1e-9, base["false"] + base["true"]))
        b = logit(hp["false"] / max(1e-9, hp["false"] + hp["true"]))
        return {"windows": a, "hmm": b, "agree": a * b / 4}

    def predict(self, row):
        base, atoms = self.shape.row_probs(row)
        hp = self.hmm.predict(row)
        pf = self.stacker.predict_proba(self._features(base, hp.probs))
        pu = min(base["unsure"], 1 - pf)
        probs = {"false": pf, "unsure": pu, "true": max(0.0, 1 - pf - pu)}
        return Prediction(max(probs, key=probs.get), probs,
                          {"rule": self.name, "atoms": atoms, "p_false_windows": round(base["false"], 3),
                           "p_false_hmm": round(hp.probs["false"], 3), "hmm_atoms": hp.trace.get("atoms")})


class GatedHybrid(HybridStack):
    """Hybrid, except where no labelled anchor backs any of the row's facts and the parse is
    confident: there the HMM (which reasons over unlabelled rows) decides on its own."""
    name = "gated"
    MIN_PARSE_Q = 0.8
    MIN_FALSE = 0.8    # only overrule the hybrid with a confident "false"...
    MIN_SUPPORT = 4    # ...backed by observations on at least 4 dates of that fact

    def fit(self, rows, lexicon, names):
        super().fit(rows, lexicon, names)
        self.anchored = {a.key for r in rows if r.label == "true" and r.atoms for a in r.atoms}
        return self

    def predict(self, row):
        p = super().predict(row)
        if not row.atoms:
            return p
        q = row.meta.get("atom_q") or {}
        unanchored = all(a.key not in self.anchored for a in row.atoms)
        confident = all(q.get(a.key, 1.0) >= self.MIN_PARSE_Q for a in row.atoms)
        if unanchored and confident:
            h = self.hmm.predict(row)
            support = min(len(self.hmm.chains[a.key].obs_at) if a.key in self.hmm.chains else 0 for a in row.atoms)
            if h.label == "false" and h.probs["false"] >= self.MIN_FALSE and support >= self.MIN_SUPPORT:
                h.trace["gate"] = "no anchors, strong cross-row evidence: HMM decides"
                return h
        return p
