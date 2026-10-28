"""Isotonic calibration (pool-adjacent-violators), dependency-free.

Maps a raw score to the observed frequency of being right, keeping the order of scores.
Used for parse confidence (how often a parse with score s is correct) and for the final
label confidence (how often a label given with confidence c is right).
"""
from __future__ import annotations

import bisect


class Isotonic:
    def __init__(self, prior_weight: float = 2.0):
        self.prior_weight = prior_weight
        self.xs: list[float] = []
        self.ys: list[float] = []

    def fit(self, scores, outcomes) -> "Isotonic":
        pts = sorted(zip(scores, outcomes))
        if not pts:
            self.xs, self.ys = [0.0], [0.5]
            return self
        base = sum(o for _, o in pts) / len(pts)
        # blocks: [sum_y, weight, max_x]; a weak pull to the base rate keeps tiny blocks sane
        blocks = []
        for x, y in pts:
            blocks.append([float(y), 1.0, x])
            while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
                s, w, mx = blocks.pop()
                blocks[-1][0] += s
                blocks[-1][1] += w
                blocks[-1][2] = mx
        self.xs = [b[2] for b in blocks]
        self.ys = [(b[0] + self.prior_weight * base) / (b[1] + self.prior_weight) for b in blocks]
        return self

    def __call__(self, score: float) -> float:
        i = bisect.bisect_left(self.xs, score)
        return self.ys[min(i, len(self.ys) - 1)]

    def table(self):
        return list(zip(self.xs, self.ys))
