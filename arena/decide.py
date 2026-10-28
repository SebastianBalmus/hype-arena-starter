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


CLASSES = ["false", "true", "unsure"]
_GRID = [round(0.05 + 0.025 * i, 3) for i in range(39)] + [1.01]


def _policy(P: np.ndarray, tf: float, tu: float) -> np.ndarray:
    """Columns follow CLASSES. false if P(false)>=tf, unsure if P(unsure)>=tu, else true."""
    out = np.full(len(P), "true", dtype=object)
    uns, fal = P[:, 2] >= tu, P[:, 0] >= tf
    out[uns] = "unsure"
    out[fal & ~(uns & (P[:, 2] > P[:, 0]))] = "false"
    return out


def macro_f1(y, pred) -> float:
    y, pred = np.asarray(y), np.asarray(pred)
    f = []
    for c in CLASSES:
        tp = float(((y == c) & (pred == c)).sum())
        d = float((y == c).sum() + (pred == c).sum())
        f.append(2 * tp / d if d else 0.0)
    return float(np.mean(f))


def _tune(P: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    best, arg = (-1.0, -1.0), (0.5, 1.01)
    for tf in _GRID:
        for tu in _GRID:
            pred = _policy(P, tf, tu)
            score = (float((pred == y).mean()), macro_f1(y, pred))
            if score > best:
                best, arg = score, (tf, tu)
    return arg


def _chosen_prob(P: np.ndarray, pred: np.ndarray) -> np.ndarray:
    idx = np.array([CLASSES.index(c) for c in pred])
    return P[np.arange(len(P)), idx]


def _fit_iso(P, pred, y) -> IsotonicRegression:
    keep = pred != "unsure"
    iso = IsotonicRegression(out_of_bounds="clip")
    if keep.sum() >= 2:
        iso.fit(_chosen_prob(P, pred)[keep], (pred == y)[keep].astype(float))
    else:
        iso.fit([0.0, 1.0], [0.5, 1.0])
    return iso


def _calibrated(iso, P, pred) -> np.ndarray:
    return np.clip(iso.predict(_chosen_prob(P, pred)), 0.5, 1.0)


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    if len(conf) == 0:
        return 0.0
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.sum() * abs(conf[m].mean() - correct[m].mean())
    return float(total / len(conf))


def _full(model, M) -> np.ndarray:
    p = model.predict_proba(M)
    out = np.zeros((len(M), 3))
    for j, c in enumerate(model.classes_):
        out[:, CLASSES.index(c)] = p[:, j]
    return out


class Decider:
    def __init__(self):
        self.model = None
        self.iso: IsotonicRegression | None = None
        self.t_false, self.t_unsure = 0.5, 1.01
        self.cv_report: dict = {}

    def fit(self, X: list[dict], y: list[str], groups: list[int]) -> "Decider":
        M, y_arr, g = _matrix(X), np.array(y), np.array(groups)
        folds = list(GroupKFold(n_splits=min(5, len(set(groups)))).split(M, y_arr, g))
        P = np.zeros((len(y), 3))
        for tr, te in folds:
            P[te] = _full(_model().fit(M[tr], y_arr[tr]), M[te])
        # honest estimate: tune thresholds + calibration on 4 folds, apply to the 5th
        honest = np.empty(len(y), dtype=object)
        hconf = np.zeros(len(y))
        for _, te in folds:
            tr = np.setdiff1d(np.arange(len(y)), te)
            tf, tu = _tune(P[tr], y_arr[tr])
            iso = _fit_iso(P[tr], _policy(P[tr], tf, tu), y_arr[tr])
            honest[te] = _policy(P[te], tf, tu)
            hconf[te] = _calibrated(iso, P[te], honest[te])
        keep = honest != "unsure"
        # final policy: tuned on all OOF probabilities
        self.t_false, self.t_unsure = _tune(P, y_arr)
        dec = _policy(P, self.t_false, self.t_unsure)
        self.iso = _fit_iso(P, dec, y_arr)
        self.model = _model().fit(M, y_arr)
        self.cv_report = {
            "classes": CLASSES,
            "oof_proba": P.tolist(),
            "accuracy": float((honest == y_arr).mean()),
            "macro_f1": macro_f1(y_arr, honest),
            "ece": ece(hconf[keep], (honest == y_arr)[keep].astype(float)),
            "oof_pred": [str(v) for v in honest],
            "thresholds": {"t_false": self.t_false, "t_unsure": self.t_unsure},
        }
        return self

    def proba(self, x: dict) -> dict[str, float]:
        return dict(zip(CLASSES, _full(self.model, _matrix([x]))[0].tolist()))

    def predict(self, x: dict) -> tuple[str, float | None]:
        P = _full(self.model, _matrix([x]))
        label = str(_policy(P, self.t_false, self.t_unsure)[0])
        if label == "unsure":
            return label, None
        c = float(_calibrated(self.iso, P, np.array([label], dtype=object))[0])
        return label, c

    def predict_many(self, X: list[dict]) -> list[tuple[str, float | None]]:
        """``predict`` for many rows in one forest pass (same results, far faster)."""
        if not X:
            return []
        P = _full(self.model, _matrix(X))
        labels = _policy(P, self.t_false, self.t_unsure)
        conf = _calibrated(self.iso, P, labels)
        return [(str(l), None if l == "unsure" else float(c)) for l, c in zip(labels, conf)]
