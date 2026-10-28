"""A small, dependency-free logistic regression over sparse dict features.

Enough for a few thousand rows; deterministic (fixed order, no randomness).
"""
from __future__ import annotations

import math
from collections import defaultdict


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    e = math.exp(z)
    return e / (1 + e)


def logit(p: float, eps: float = 1e-4) -> float:
    p = min(1 - eps, max(eps, p))
    return math.log(p / (1 - p))


class LogReg:
    """Binary logistic regression, L2-regularised, trained with Adagrad."""

    def __init__(self, l2: float = 1e-3, epochs: int = 40, lr: float = 0.5):
        self.l2, self.epochs, self.lr = l2, epochs, lr
        self.w: dict[str, float] = defaultdict(float)
        self.b = 0.0

    def fit(self, X: list[dict[str, float]], y: list[int], sample_weight=None) -> "LogReg":
        n = len(X)
        sw = sample_weight or [1.0] * n
        g2 = defaultdict(lambda: 1e-8)
        gb2 = 1e-8
        pos = sum(w for w, t in zip(sw, y) if t) / sum(sw)
        self.b = logit(pos)  # start from the base rate
        for _ in range(self.epochs):
            for x, t, w in zip(X, y, sw):
                err = (self.predict_proba(x) - t) * w
                for f, v in x.items():
                    g = err * v + self.l2 * self.w[f]
                    g2[f] += g * g
                    self.w[f] -= self.lr * g / math.sqrt(g2[f])
                gb2 += err * err
                self.b -= self.lr * err / math.sqrt(gb2)
        return self

    def decision(self, x: dict[str, float]) -> float:
        w = self.w
        return self.b + sum(w.get(f, 0.0) * v for f, v in x.items())

    def predict_proba(self, x: dict[str, float]) -> float:
        return _sigmoid(self.decision(x))

    def top(self, k: int = 15):
        """Strongest features in each direction (for the demo / report)."""
        items = sorted(self.w.items(), key=lambda kv: kv[1])
        return {"towards_1": items[::-1][:k], "towards_0": items[:k]}
