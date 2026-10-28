"""E5: timeline verdict (E1b) + credibility from text, merged by a small stacker.

Credibility scores for training rows are out-of-fold (inner page-grouped folds), and the
timeline probabilities for training rows leave their own page out, so the stacker learns
from scores that look like the ones it will see on unseen rows.
"""
from __future__ import annotations

from collections import Counter

from .credibility import Credibility, featurize
from .folds import grouped_folds
from .learn import LogReg, logit
from .models import Model, Prediction, smoothed
from .temporal import TemporalWindowsPlus


class CredibilityOnly(Model):
    """Reference point: the text-style model alone, no facts, no timelines."""
    name = "credibility_only"

    def fit(self, rows, lexicon, names):
        self.cred = Credibility().fit([r for r in rows if r.label])
        self.prior = smoothed(Counter(r.label for r in rows if r.label))
        return self

    def predict(self, row):
        pf = self.cred.p_false(row.text)
        pu = self.prior["unsure"]
        probs = {"false": pf * (1 - pu), "unsure": pu, "true": (1 - pf) * (1 - pu)}
        return Prediction(max(probs, key=probs.get), probs, {"rule": self.name, "p_false_text": round(pf, 3)})


class WindowsCredibility(TemporalWindowsPlus):
    name = "windows_cred"

    def fit(self, rows, lexicon, names):
        super().fit(rows, lexicon, names)
        labelled = [r for r in rows if r.label]
        self.p_false_prior = self.prior["false"]
        oof = {}
        for test_ids in grouped_folds(labelled, k=5, seed=1):
            m = Credibility().fit([r for r in labelled if r.id not in test_ids])
            for r in labelled:
                if r.id in test_ids:
                    oof[r.id] = m.p_false(r.text)
        self.cred = Credibility().fit(labelled)
        X, y = [], []
        for r in labelled:
            base, _ = self.row_probs(r, exclude_page=r.page)
            X.append(self._features(base, oof[r.id], bool(r.atoms)))
            y.append(int(r.label == "false"))
        self.stacker = LogReg(l2=1e-3, epochs=60).fit(X, y)
        return self

    def _features(self, base: dict, p_text: float, parsed: bool) -> dict:
        b = logit(base["false"] / max(1e-9, base["false"] + base["true"]))
        c = logit(p_text) - logit(self.p_false_prior)
        return {"timeline": b, "text": c, "timeline_x_text": b * c / 4,
                **({} if parsed else {"unparsed": 1.0})}

    def predict(self, row):
        base, atoms = self.row_probs(row)
        p_text = self.cred.p_false(row.text)
        pf = self.stacker.predict_proba(self._features(base, p_text, bool(row.atoms)))
        pu = min(base["unsure"], 1 - pf)
        probs = {"false": pf, "unsure": pu, "true": max(0.0, 1 - pf - pu)}
        x = featurize(row.text)
        why = sorted(((f, self.cred.model.w.get(f, 0.0)) for f in x), key=lambda kv: -abs(kv[1]))[:5]
        trace = {"rule": self.name, "atoms": atoms, "p_false_timeline": round(base["false"], 3),
                 "p_false_text": round(p_text, 3),
                 "text_signals": [(f, round(w, 2)) for f, w in why]}
        return Prediction(max(probs, key=probs.get), probs, trace)
