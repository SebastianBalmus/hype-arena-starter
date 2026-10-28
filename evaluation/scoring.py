"""Independent diagnostics for classification experiment predictions.

This module deliberately performs no file I/O and uses only the standard library.
"""

from collections import Counter
import math

LABELS = ("true", "false", "unsure")
REASONS = ("decided", "model_unsure", "ambiguous_parse", "missing_evidence",
           "temporal_conflict", "low_confidence", "unsupported", "other")


def _finite_number(value):
    # bool is an int in Python, but is not a meaningful confidence/probability.
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _class_metrics(gold, pred):
    result = {}
    for label in LABELS:
        tp = sum(g == label and p == label for g, p in zip(gold, pred))
        fp = sum(g != label and p == label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))
        support = sum(g == label for g in gold)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        result[label] = {"precision": precision, "recall": recall, "f1": f1,
                         "support": support, "absent": support == 0}
    return result


def score_rows(gold_rows, predictions):
    """Return (classification metrics, output audit, scored canonical rows).

    IDs are compared as strings (the canonical CSV representation). First duplicate
    occurrence wins. Invalid/missing predictions are normalized to ``unsure`` only
    for diagnostic metrics; callers must honor ``eligible_for_comparison``.
    """
    gold_by_id, gold_errors = {}, []
    for i, row in enumerate(gold_rows):
        rid = str(row.get("id", ""))
        if not rid:
            gold_errors.append({"code": "missing_gold_id", "row": i})
        elif rid in gold_by_id:
            gold_errors.append({"code": "duplicate_gold_id", "id": rid, "row": i})
        else:
            gold_by_id[rid] = row
        if row.get("label") not in LABELS:
            gold_errors.append({"code": "invalid_gold_label", "id": rid, "row": i})

    pred_by_id, pred_errors = {}, []
    for i, row in enumerate(predictions):
        rid = str(row.get("id", ""))
        if not rid:
            pred_errors.append({"code": "missing_id", "row": i})
            continue
        if rid not in gold_by_id:
            pred_errors.append({"code": "unknown_id", "id": rid, "row": i})
        if rid in pred_by_id:
            pred_errors.append({"code": "duplicate_id", "id": rid, "row": i})
            continue
        pred_by_id[rid] = row

    scored, ys, ps = [], [], []
    reason_counts = Counter()
    decisions = []
    full_dist_ok = True
    multiclass_brier_sum = 0.0
    for rid, grow in gold_by_id.items():
        errors = []
        prow = pred_by_id.get(rid)
        if prow is None:
            errors.append("missing_prediction")
            label, confidence, reason = "unsure", None, "other"
            dist = None
        else:
            label = prow.get("label")
            confidence = prow.get("confidence")
            reason = prow.get("decision_reason")
            if label not in LABELS:
                errors.append("invalid_label")
                label = "unsure"
            if reason not in REASONS:
                errors.append("invalid_decision_reason")
                reason = "other"
            if label in ("true", "false"):
                if not _finite_number(confidence) or not 0.5 <= confidence <= 1.0:
                    errors.append("invalid_confidence")
                    label, confidence = "unsure", None
            elif confidence not in (None, ""):
                errors.append("unsure_has_confidence")
                confidence = None
            dist = [prow.get("p_true"), prow.get("p_false"), prow.get("p_unsure")]
        gold = grow.get("label")
        correct = label == gold if gold in LABELS else None
        out = {"id": rid, "gold": gold, "label": label, "confidence": confidence,
               "decision_reason": reason, "correct": correct,
               "output_valid": not errors, "error_codes": errors}
        scored.append(out)
        reason_counts[reason] += 1
        if gold in LABELS:
            ys.append(gold); ps.append(label)
        if label in ("true", "false") and _finite_number(confidence) and correct is not None:
            decisions.append((gold, correct, confidence))
        if not (dist and all(_finite_number(x) and x >= 0 for x in dist)
                and abs(sum(dist) - 1.0) <= 1e-6):
            full_dist_ok = False
        else:
            multiclass_brier_sum += sum((dist[j] - (gold == LABELS[j])) ** 2
                                        for j in range(3))
        if errors:
            pred_errors.extend({"code": e, "id": rid} for e in errors)

    n = len(scored)
    classes = _class_metrics(ys, ps)
    confusion = [[sum(g == a and p == b for g, p in zip(ys, ps)) for b in LABELS]
                 for a in LABELS]
    correct_n = sum(g == p for g, p in zip(ys, ps))
    d = len(decisions); dcorrect = sum(x[1] for x in decisions)
    bins = []
    bounds = [(0.5, .6), (.6, .7), (.7, .8), (.8, .9), (.9, 1.0000000001)]
    for lo, hi in bounds:
        vals = [x for x in decisions if lo <= x[2] < hi]
        bins.append({"range": f"[{lo:.1f},{'1.0' if hi > 1 else f'{hi:.1f}'})" if hi <= 1 else "[0.9,1.0]",
                     "count": len(vals),
                     "mean_confidence": sum(x[2] for x in vals) / len(vals) if vals else None,
                     "empirical_correctness": sum(x[1] for x in vals) / len(vals) if vals else None})
    by_gold = {lab: {"decided_count": sum(g == lab for g, _, _ in decisions),
                     "coverage": (sum(g == lab for g, _, _ in decisions) / classes[lab]["support"]
                                 if classes[lab]["support"] else None)} for lab in LABELS}
    metrics = {
        "rows": n, "accuracy": correct_n / n if n else None,
        "macro_f1": sum(classes[x]["f1"] for x in LABELS) / 3 if n else None,
        "per_class": classes, "confusion_matrix": {"labels": list(LABELS), "counts": confusion},
        "decided_count": d, "decided_correct": dcorrect,
        "decided_errors": d - dcorrect, "decided_coverage": d / n if n else None,
        "decided_accuracy": dcorrect / d if d else None,
        "decided_risk": (d - dcorrect) / d if d else None,
        "decided_by_gold_class": by_gold,
        "decision_reason_counts": dict(reason_counts),
        "correctness_brier": (sum((c - float(ok)) ** 2 for _, ok, c in decisions) / d if d else None),
        "reliability_bins": bins,
        "multiclass_brier": multiclass_brier_sum / n if n and full_dist_ok else None,
    }
    output_errors = {"eligible_for_comparison": not (gold_errors or pred_errors),
                     "missing_count": sum(rid not in pred_by_id for rid in gold_by_id),
                     "duplicate_count": sum(e["code"] in ("duplicate_id", "duplicate_gold_id") for e in gold_errors + pred_errors),
                     "unknown_count": sum(e["code"] == "unknown_id" for e in pred_errors),
                     "gold_errors": gold_errors, "errors": pred_errors}
    return metrics, output_errors, scored


if __name__ == "__main__":
    # Direct smoke checks for denominator, first duplicate, abstention and undefined metrics.
    m, e, s = score_rows(
        [{"id": "a", "label": "true"}, {"id": "b", "label": "true"}],
        [{"id": "a", "label": "true", "confidence": .8, "decision_reason": "decided"},
         {"id": "a", "label": "false", "confidence": .9, "decision_reason": "decided"}])
    assert m["rows"] == 2 and m["accuracy"] == .5 and m["decided_count"] == 1
    assert not e["eligible_for_comparison"] and s[1]["label"] == "unsure"
    m, _, _ = score_rows([], [])
    assert m["decided_accuracy"] is None and m["multiclass_brier"] is None
