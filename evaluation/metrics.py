"""Metrics for row labels: accuracy, per-class F1, calibration and confidence ranking.

The official scoring formula is not published, so we report several views: plain
accuracy (what "correctness" most likely means), probability quality (Brier, log loss)
and calibration (does 0.8 confidence mean right 80% of the time?).
"""
from __future__ import annotations

import math
import random
from collections import defaultdict

from hooly.data import LABELS


def confusion(gold, pred):
    m = {g: {p: 0 for p in LABELS} for g in LABELS}
    for g, p in zip(gold, pred):
        m[g][p] += 1
    return m


def per_class(conf):
    out = {}
    for c in LABELS:
        tp = conf[c][c]
        fp = sum(conf[g][c] for g in LABELS if g != c)
        fn = sum(conf[c][p] for p in LABELS if p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[c] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}
    return out


def brier(gold, probs):
    return sum(sum((p.get(c, 0) - (g == c)) ** 2 for c in LABELS) for g, p in zip(gold, probs)) / len(gold)


def log_loss(gold, probs, eps=1e-6):
    return -sum(math.log(max(eps, p.get(g, 0))) for g, p in zip(gold, probs)) / len(gold)


def reliability(correct, conf, bins=10):
    """Bins of reported confidence (0.5-1) vs actual accuracy, plus expected calibration error."""
    table = defaultdict(lambda: [0, 0, 0.0])  # n, correct, sum of confidence
    for ok, c in zip(correct, conf):
        b = min(bins - 1, int((c - 0.5) / 0.5 * bins))
        t = table[b]
        t[0], t[1], t[2] = t[0] + 1, t[1] + ok, t[2] + c
    rows, ece, n = [], 0.0, len(correct)
    for b in sorted(table):
        cnt, ok, s = table[b]
        acc, mean_c = ok / cnt, s / cnt
        ece += cnt / n * abs(acc - mean_c)
        rows.append({"bin": f"{0.5 + b * 0.05:.2f}-{0.55 + b * 0.05:.2f}", "n": cnt,
                     "confidence": round(mean_c, 3), "accuracy": round(acc, 3)})
    return rows, ece


def selective(correct, conf, coverages=(0.25, 0.5, 0.75, 1.0)):
    """Accuracy on the most confident X% of decisions: is high confidence actually better?"""
    order = sorted(range(len(conf)), key=lambda i: -conf[i])
    out = {}
    for cov in coverages:
        k = max(1, round(cov * len(order)))
        out[f"{int(cov * 100)}%"] = sum(correct[i] for i in order[:k]) / k
    return out


def bootstrap_accuracy(gold, pred, groups, n=1000, seed=0):
    """95% interval for accuracy, resampling whole pages (rows of a page are not independent)."""
    by_group = defaultdict(list)
    for g, p, grp in zip(gold, pred, groups):
        by_group[grp].append(g == p)
    keys = list(by_group)
    rng = random.Random(seed)
    accs = []
    for _ in range(n):
        sample = [x for k in rng.choices(keys, k=len(keys)) for x in by_group[k]]
        accs.append(sum(sample) / len(sample))
    accs.sort()
    return accs[int(0.025 * n)], accs[int(0.975 * n) - 1]


def summarize(records):
    """records: dicts with gold, label, probs, confidence, page."""
    gold = [r["gold"] for r in records]
    pred = [r["label"] for r in records]
    probs = [r["probs"] for r in records]
    conf = confusion(gold, pred)
    decided = [r for r in records if r["label"] != "unsure"]
    correct = [r["gold"] == r["label"] for r in decided]
    confs = [r["confidence"] for r in decided]
    rel, ece = reliability(correct, confs) if decided else ([], float("nan"))
    lo, hi = bootstrap_accuracy(gold, pred, [r["page"] for r in records])
    pc = per_class(conf)
    return {
        "n": len(records),
        "accuracy": sum(g == p for g, p in zip(gold, pred)) / len(gold),
        "accuracy_95ci": [lo, hi],
        "macro_f1": sum(v["f1"] for v in pc.values()) / len(LABELS),
        "per_class": pc,
        "confusion": conf,
        "brier": brier(gold, probs),
        "log_loss": log_loss(gold, probs),
        "ece": ece,
        "reliability": rel,
        "selective_accuracy": selective(correct, confs) if decided else {},
        "predicted": {l: pred.count(l) for l in LABELS},
    }
