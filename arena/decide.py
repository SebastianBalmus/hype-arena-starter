"""Row decider: per-fact evidence checks -> row label with calibrated confidence."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

from arena.evidence import AtomCheck
from arena.model import Fact

FEATURES = [
    "n_facts", "n_true", "n_false", "n_unknown", "verdict_true", "verdict_false", "verdict_unknown",
    "any_trusted_false", "all_trusted_true", "min_support", "max_days", "any_before", "any_after",
    "any_gap", "any_none", "min_fact_conf", "page_rel", "has_op", "n_bool", "n_name",
]


def row_features(facts: list[Fact], checks: list[AtomCheck], page_rel: float) -> dict[str, float]:
    n = len(checks)
    n_true = sum(c.state == "true" for c in checks)
    n_false = sum(c.state == "false" for c in checks)
    verdict = "false" if n_false else ("true" if n and n_true == n else "unknown")
    return {
        "n_facts": float(len(facts)),
        "n_true": float(n_true),
        "n_false": float(n_false),
        "n_unknown": float(n - n_true - n_false),
        "verdict_true": float(verdict == "true"),
        "verdict_false": float(verdict == "false"),
        "verdict_unknown": float(verdict == "unknown"),
        "any_trusted_false": float(any(c.state == "false" and c.trusted for c in checks)),
        "all_trusted_true": float(bool(n) and all(c.state == "true" and c.trusted for c in checks)),
        "min_support": float(min((c.support for c in checks), default=0)),
        "max_days": float(max((c.days for c in checks), default=0)),
        "any_before": float(any(c.status == "before" for c in checks)),
        "any_after": float(any(c.status == "after" for c in checks)),
        "any_gap": float(any(c.status == "gap" for c in checks)),
        "any_none": float(any(c.status == "none" for c in checks)),
        "min_fact_conf": float(min((f.confidence for f in facts), default=1.0)),
        "page_rel": float(page_rel),
        "has_op": float(any(f.value.get("op") not in (None, "eq") for f in facts)),
        "n_bool": float(sum(f.value.get("kind") == "boolean" for f in facts)),
        "n_name": float(sum(f.value.get("kind") == "name" for f in facts)),
    }


def _matrix(X: list[dict]) -> np.ndarray:
    return np.array([[x[k] for k in FEATURES] for x in X], dtype=float)


def _model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0)


class Decider:
    def __init__(self):
        self.model = None
        self.iso: IsotonicRegression | None = None
        self.cv_report: dict = {}

    def fit(self, X: list[dict], y: list[str], groups: list[int]) -> "Decider":
        M, y_arr, g = _matrix(X), np.array(y), np.array(groups)
        oof = np.empty(len(y), dtype=object)
        conf = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=min(5, len(set(groups)))).split(M, y_arr, g):
            if len(set(y_arr[tr])) < 2:
                oof[te], conf[te] = y_arr[tr][0], 1.0
                continue
            m = _model().fit(M[tr], y_arr[tr])
            p = m.predict_proba(M[te])
            oof[te] = m.classes_[p.argmax(1)]
            conf[te] = p.max(1)
        correct = (oof == y_arr).astype(float)
        self.iso = IsotonicRegression(out_of_bounds="clip").fit(conf, correct)
        self.model = _model().fit(M, y_arr)
        self.cv_report = {"accuracy": float(correct.mean()), "oof_pred": [str(v) for v in oof]}
        return self

    def predict(self, x: dict) -> tuple[str, float | None]:
        p = self.model.predict_proba(_matrix([x]))[0]
        label = str(self.model.classes_[p.argmax()])
        if label == "unsure":
            return label, None
        c = float(self.iso.predict([p.max()])[0])
        return label, min(1.0, max(0.5, c))
