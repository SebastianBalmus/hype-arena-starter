"""Regime-aware ensemble of independent evidence views, greedily selected.

Views (each fails differently):
  A timeline   states of hooly.temporal (needs labelled anchors)
  B hmm        probabilistic timelines over all rows (works without anchors)
  C votes      other pages stating the same fact near the same date
  D same_text  the identical sentence elsewhere near the same date
  E outlier    how unusual the value is for its fact (robust z on a log scale)
Regime inputs (always on): anchor coverage of the row's facts, anchor counts, parse
confidence; each view also gets a copy gated by the regime (e.g. hmm x no-anchors).

Anchor dropout: training rows almost always have anchors, so each labelled row also yields a
twin example as if its subject were unseen (its fact's labels hidden: HMM re-run on that fact
without anchors, the labelled rows counted as plain votes), so the meta-model learns when to
switch views. Views are kept by greedy forward selection on inner page-grouped CV, and the
output is recalibrated afterwards (isotonic, see hooly.calibrate).
"""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import date

from .folds import grouped_folds
from .hmm import FactChain, TimelineHMM
from .learn import LogReg, logit
from .models import Model, Prediction, smoothed
from .temporal import TemporalWindowsShape
from .values import DIFFERENT, compare, matches, normalize

VIEWS = ["timeline", "hmm", "votes", "same_text", "outlier"]
WINDOW, TEXT_WINDOW = 45, 60


def _d(s):
    return date.fromisoformat(s)


def _norm_text(t):
    return " ".join(t.lower().split()).rstrip(".!? ")


class EnsembleViews(Model):
    name = "ensemble"

    def __init__(self, seed: int = 0, epochs: int = 25):
        self.seed, self.epochs = seed, epochs

    # ------------------------------------------------------------------ fit
    def fit(self, rows, lexicon, names):
        self.names = names
        self.shape = TemporalWindowsShape().fit(rows, lexicon, names)
        self.prior = smoothed(Counter(r.label for r in rows if r.label))
        labelled = [r for r in rows if r.label]
        # observation pools
        self.unlab = defaultdict(list)   # key -> [(date, value, page)] unlabelled rows
        self.lab = defaultdict(list)     # key -> [(date, value, page, label)] labelled rows (single-fact label known)
        self.texts = defaultdict(list)   # normalised text -> [(date, label, page)]
        for r in rows:
            self.texts[_norm_text(r.text)].append((_d(r.date), r.label, r.page))
            for a in (r.atoms or []):
                if "op" in a.value:
                    continue
                if r.label is None:
                    self.unlab[a.key].append((_d(r.date), a.value, r.page))
                else:
                    self.lab[a.key].append((_d(r.date), a.value, r.page, r.label if len(r.atoms) == 1 else "mixed"))
        # HMM: full fit, out-of-fold scores for labelled rows, and per-fact "no anchor" chains
        self.hmm = TimelineHMM().fit(rows, lexicon, names)
        self.hmm_oof = {}
        for test_ids in grouped_folds(labelled, k=5, seed=self.seed + 1):
            view = [replace(r, label=None) if r.id in test_ids else r for r in rows]
            h = TimelineHMM().fit(view, lexicon, names)
            for r in view:
                if r.id in test_ids:
                    self.hmm_oof[r.id] = h.predict(r).probs["false"]
        self.free_chains = {}
        obs = defaultdict(list)
        for r in rows:
            for a in (r.atoms or []):
                if "op" not in a.value:
                    obs[a.key].append((_d(r.date), a.value, "free", r.id))
        for k, o in obs.items():
            c = FactChain(o, names)
            if not c.multi:
                c.run(self.hmm.alpha, self.hmm.tau)
            self.free_chains[k] = c
        # training examples: real (as unseen pages) + anchor-dropout twins
        X, y, real = [], [], []
        for r in labelled:
            X.append(self.features(r, exclude_page=r.page, hmm_false=self.hmm_oof.get(r.id)))
            y.append(int(r.label == "false")); real.append(True)
            if r.atoms:
                X.append(self.features(r, exclude_page=r.page, dropout=True))
                y.append(int(r.label == "false")); real.append(False)
        self.selected = self._select(X, y, real, labelled)
        Xs = [self._subset(x, self.selected) for x in X]
        self.meta = LogReg(l2=2e-3, epochs=self.epochs).fit(Xs, y)
        return self

    # ------------------------------------------------------------------ features
    def features(self, row, exclude_page=None, hmm_false=None, dropout=False):
        f = {"bias": 1.0}
        if not row.atoms:
            f["regime:unparsed"] = 1.0
            return f
        d = _d(row.date)
        q = row.meta.get("atom_q") or {}
        keys = [a.key for a in row.atoms]
        cov, n_anchor = [], []
        for k in keys:
            n = 0 if dropout else sum(1 for od, v, pg, lab in self.lab.get(k, []) if pg != exclude_page and lab == "true")
            cov.append(n > 0)
            n_anchor.append(n)
        coverage = sum(cov) / len(cov)
        nocov = 1 - coverage
        f["regime:coverage"] = coverage
        f["regime:anchors"] = math.log1p(min(n_anchor))
        f["regime:min_parse_q"] = min(q.get(k, 1.0) for k in keys)
        f["regime:n_atoms"] = min(3, len(keys)) / 3
        # A timeline
        if dropout:
            pa = self.prior["false"] / (self.prior["false"] + self.prior["true"])
        else:
            base, _ = self.shape.row_probs(row, exclude_page=exclude_page)
            pa = base["false"] / max(1e-9, base["false"] + base["true"])
        a = logit(pa)
        f["timeline"], f["timeline:cov"] = a, a * coverage
        # B hmm
        if dropout:
            pt = 1.0
            for at in row.atoms:
                c = self.free_chains.get(at.key)
                p = c.p_true(at.value, d) if c is not None and not c.multi else None
                pt *= p if p is not None else 0.85
            pb = 1 - pt
        elif hmm_false is not None:
            pb = hmm_false
        else:
            pb = self.hmm.predict(row).probs["false"]
        b = logit(pb)
        f["hmm"], f["hmm:nocov"] = b, b * nocov
        # C votes (in dropout the labelled rows of the fact count as plain votes)
        agree = disagree = 0
        lie = False
        for at in row.atoms:
            if "op" in at.value:
                continue
            ag, dis = set(), set()
            pool = list(self.unlab.get(at.key, []))
            if dropout:
                pool += [(od, v, pg) for od, v, pg, lab in self.lab.get(at.key, [])]
            for od, v, pg in pool:
                if pg == exclude_page or abs((od - d).days) > WINDOW:
                    continue
                if matches(at.value, v, self.names):
                    ag.add(pg)
                elif compare(at.value, v, self.names) == DIFFERENT:
                    dis.add(pg)
            agree, disagree = agree + len(ag), disagree + len(dis)
            if not dropout:
                lie = lie or any(pg != exclude_page and lab == "false" and abs((od - d).days) <= 60 and matches(at.value, v, self.names)
                                 for od, v, pg, lab in self.lab.get(at.key, []))
        f["votes:agree"], f["votes:disagree"] = math.log1p(agree), math.log1p(disagree)
        f["votes:known_lie"] = float(lie)
        f["votes:disagree:nocov"] = math.log1p(disagree) * nocov
        f["votes:agree:nocov"] = math.log1p(agree) * nocov
        # D same text elsewhere near the same date
        st, sf, su = 0, 0, 0
        for od, lab, pg in self.texts.get(_norm_text(row.text), []):
            if pg == exclude_page or abs((od - d).days) > TEXT_WINDOW:
                continue
            if lab is None or dropout:
                su += 1
            elif lab == "true":
                st += 1
            elif lab == "false":
                sf += 1
        f["same_text:true"], f["same_text:false"], f["same_text:unlabelled"] = float(st > 0), float(sf > 0), math.log1p(su)
        # E outlier: robust z of the value against the fact's other values (log scale)
        z_max, above = 0.0, 0.0
        for at in row.atoms:
            if at.value["kind"] not in ("quantity", "money", "percent", "duration"):
                continue
            ref = [v for od, v, pg, lab in self.lab.get(at.key, []) if pg != exclude_page and (lab == "true" or dropout)]
            if dropout or len(ref) < 2:
                ref += [v for od, v, pg in self.unlab.get(at.key, []) if pg != exclude_page]
            try:
                nv = normalize({k: x for k, x in at.value.items() if k != "op"}, self.names)
                vals = [normalize(v, self.names) for v in ref]
            except ValueError:
                continue
            xs = [float(x.value) for x in vals if x.dim == nv.dim and float(x.value) > 0]
            if len(xs) < 2 or float(nv.value) <= 0:
                continue
            lx = sorted(math.log10(x) for x in xs)
            med = lx[len(lx) // 2]
            mad = sorted(abs(x - med) for x in lx)[len(lx) // 2]
            z = abs(math.log10(float(nv.value)) - med) / (1.4826 * mad + 0.05)
            z_max = max(z_max, min(8.0, z))
            if float(nv.value) > 1.5 * max(xs):
                above = 1.0
        f["outlier:z"], f["outlier:above_max"] = z_max / 4, above
        f["outlier:z:nocov"] = z_max / 4 * nocov
        return f

    @staticmethod
    def _subset(x, views):
        return {k: v for k, v in x.items() if k == "bias" or k.startswith("regime:") or k.split(":")[0] in views}

    def _select(self, X, y, real, labelled):
        """Greedy forward selection of views by inner page-grouped CV accuracy on real rows."""
        idx_of = {}
        i = 0
        for r in labelled:
            idx_of.setdefault(r.id, []).append(i)
            i += 2 if r.atoms else 1
        folds = grouped_folds(labelled, k=3, seed=self.seed + 7)
        fold_idx = []
        for test_ids in folds:
            te = [j for rid in test_ids for j in idx_of.get(rid, [])]
            fold_idx.append(set(te))

        def score(views):
            correct = n = 0
            ll = 0.0
            for te in fold_idx:
                tr = [j for j in range(len(X)) if j not in te]
                m = LogReg(l2=2e-3, epochs=max(8, self.epochs // 2)).fit([self._subset(X[j], views) for j in tr], [y[j] for j in tr])
                for j in te:
                    if not real[j]:
                        continue
                    p = m.predict_proba(self._subset(X[j], views))
                    correct += int((p >= 0.5) == bool(y[j]))
                    ll -= math.log(max(1e-6, p if y[j] else 1 - p))
                    n += 1
            return correct / n, -ll / n

        chosen, best = [], score([])
        self.selection_log = [("(regime only)", round(best[0], 4))]
        while True:
            trials = [(score(chosen + [v]), v) for v in VIEWS if v not in chosen]
            if not trials:
                break
            (acc, nll), v = max(trials, key=lambda t: (t[0][0], t[0][1]))
            if (acc, nll) <= best:
                break
            chosen.append(v)
            best = (acc, nll)
            self.selection_log.append((v, round(acc, 4)))
        return chosen

    # ------------------------------------------------------------------ predict
    def predict(self, row):
        x = self.features(row)
        pf = self.meta.predict_proba(self._subset(x, self.selected))
        base, atoms = self.shape.row_probs(row)
        pu = min(base["unsure"], 1 - pf)
        probs = {"false": pf, "unsure": pu, "true": max(0.0, 1 - pf - pu)}
        why = sorted(((k, round(self.meta.w.get(k, 0.0) * v, 3)) for k, v in self._subset(x, self.selected).items()
                      if k != "bias"), key=lambda kv: -abs(kv[1]))[:6]
        return Prediction(max(probs, key=probs.get), probs,
                          {"rule": self.name, "atoms": atoms, "views": self.selected,
                           "contributions": why, "coverage": x.get("regime:coverage")})
