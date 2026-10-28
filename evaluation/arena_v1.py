"""Run our models on the team's shared benchmark (arena-eval-v1, EVALUATION.md on the
`bogdan` branch): frozen splits in evaluation/splits-v1.json, access profile
batch_unlabelled_v1 (labelled training rows + unlabelled test text, no test forms), seed 42,
decision policy fixed_050 (emit true/false only if its probability >= 0.50, else unsure).

    python3 -m evaluation.arena_v1 [--models windows_plus always_true] [--tracks stratified temporal checker]

Not the full arena-eval-v1 artifact contract (no run.json/sanitised-file runner); it reports
the protocol's required label and confidence metrics on its exact splits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import replace

from evaluation.run import MODELS
from hooly.data import LABELS, build_names, load_lexicon, load_rows
from hooly.pipeline import Pipeline

HASHES = {"data/practice.csv": "52743befaa0fa38b89b5c802c0f953d242ae027c86839c9c0f05a9e69a86d8e9",
          "data/lexicon.json": "d0fd5e9bdc6b33ad3ec160001f50e00e7f8bf2b25a92779022f893db7c3d5507",
          "evaluation/splits-v1.json": "9699327af7bf80813eabb66a582593d5152683ee98abf36090a252b0b2723860"}


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def fixed_050(pred):
    if pred.label == "unsure":
        return "unsure", None
    p = pred.probs.get(pred.label, 0.0)
    return (pred.label, p) if p >= 0.5 else ("unsure", None)


def metrics(gold, emitted, conf, probs):
    n = len(gold)
    out = {"rows": n, "accuracy": sum(g == h for g, h in zip(gold, emitted)) / n, "per_class": {}}
    f1s = []
    for k in LABELS:
        tp = sum(g == k and h == k for g, h in zip(gold, emitted))
        fp = sum(g != k and h == k for g, h in zip(gold, emitted))
        fn = sum(g == k and h != k for g, h in zip(gold, emitted))
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        f1s.append(f1)
        out["per_class"][k] = {"precision": tp / (tp + fp) if tp + fp else 0.0,
                               "recall": tp / (tp + fn) if tp + fn else 0.0,
                               "f1": f1, "support": sum(g == k for g in gold)}
    out["macro_f1"] = sum(f1s) / 3
    out["confusion_matrix"] = [[sum(g == a and h == b for g, h in zip(gold, emitted)) for b in LABELS] for a in LABELS]
    D = [i for i, h in enumerate(emitted) if h in ("true", "false")]
    errs = sum(emitted[i] != gold[i] for i in D)
    out.update({"decided_count": len(D), "decided_correct": len(D) - errs, "decided_errors": errs,
                "decided_coverage": len(D) / n,
                "decided_accuracy": (1 - errs / len(D)) if D else None,
                "decided_risk": (errs / len(D)) if D else None,
                "correctness_brier": (sum((conf[i] - (emitted[i] == gold[i])) ** 2 for i in D) / len(D)) if D else None})
    bins = []
    for lo in (0.5, 0.6, 0.7, 0.8, 0.9):
        hi = lo + 0.1
        idx = [i for i in D if lo <= conf[i] < hi or (lo == 0.9 and conf[i] == 1.0)]
        bins.append({"bin": [lo, round(hi, 1)], "count": len(idx),
                     "mean_confidence": (sum(conf[i] for i in idx) / len(idx)) if idx else None,
                     "empirical_correctness": (sum(emitted[i] == gold[i] for i in idx) / len(idx)) if idx else None})
    out["reliability_bins"] = bins
    out["multiclass_brier"] = sum(sum((p.get(k, 0) - (g == k)) ** 2 for k in LABELS) for g, p in zip(gold, probs)) / n
    return out


def run_track(name, split, rows, lexicon, model_name, seed):
    train, test = set(split["train_ids"]), set(split["test_ids"])
    view = []
    for r in rows:
        if r.id in train:
            view.append(r)
        elif r.id in test:
            view.append(replace(r, label=None, atoms=None, meta={}))  # runtime shape: no label, no form
    names = build_names(lexicon, [r for r in view if r.atoms])
    model = Pipeline(MODELS[model_name], seed=seed).fit(view, lexicon, names)
    gold = {r.id: r.label for r in rows}
    preds = {r.id: model.predict(r) for r in view if r.id in test}
    ids = [i for i in split["test_ids"]]
    emitted, conf = zip(*(fixed_050(preds[i]) for i in ids))
    return metrics([gold[i] for i in ids], list(emitted), list(conf), [preds[i].probs for i in ids]), \
        [{"id": i, "label": e, "confidence": c} for i, e, c in zip(ids, emitted, conf)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["always_true", "windows_plus", "windows_votes"])
    ap.add_argument("--tracks", nargs="+", default=["stratified", "temporal", "checker"])
    ap.add_argument("--out", default="results/arena-v1")
    args = ap.parse_args(argv)
    for path, h in HASHES.items():
        if sha(path) != h:
            raise SystemExit(f"{path}: hash mismatch, run is not arena-eval-v1 compatible")
    manifest = json.load(open("evaluation/splits-v1.json"))
    rows, lexicon = load_rows("data/practice.csv"), load_lexicon("data/lexicon.json")
    seed = manifest["seed"]
    report = {"benchmark_version": manifest["benchmark_version"], "policy": "fixed_050",
              "access_profile": "batch_unlabelled_v1", "seed": seed, "results": {}}
    print("track | model | accuracy | macro F1 | F1 true | F1 false | F1 unsure | false P / R | decided cov | correctness Brier")
    print(" | ".join(["---"] * 10))
    for track in args.tracks:
        for m in args.models:
            t = time.time()
            res, preds = run_track(track, manifest["splits"][track], rows, lexicon, m, seed)
            report["results"].setdefault(track, {})[m] = {"metrics": res, "seconds": round(time.time() - t, 1)}
            pc = res["per_class"]
            print(f"{track} | {m} | {res['accuracy']:.3f} | {res['macro_f1']:.3f} | {pc['true']['f1']:.3f} | "
                  f"{pc['false']['f1']:.3f} | {pc['unsure']['f1']:.3f} | {pc['false']['precision']:.2f} / "
                  f"{pc['false']['recall']:.2f} | {res['decided_coverage']:.3f} | "
                  f"{res['correctness_brier'] if res['correctness_brier'] is None else round(res['correctness_brier'], 3)}")
            os.makedirs(os.path.join(args.out, "predictions"), exist_ok=True)
            with open(os.path.join(args.out, "predictions", f"{track}-{m}.csv"), "w", encoding="utf-8") as f:
                f.write("id,label,confidence\n")
                for p in preds:
                    f.write(f"{p['id']},{p['label']},{'' if p['confidence'] is None else round(p['confidence'], 4)}\n")
    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(f"\nmetrics: {args.out}/metrics.json")


if __name__ == "__main__":
    main()
