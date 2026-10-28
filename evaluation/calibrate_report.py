"""#5 Cross-fitted isotonic calibration of a saved CV report.

For each fold, maps fitted on the *other* folds' predictions ("when we said `label` with
probability p, how often were we right?", one map per label) are applied to that fold. A
`false` call whose calibrated chance of being right drops below 0.5 flips to `true`.
Also writes the map fitted on all folds, for the runtime.

    python3 -m evaluation.calibrate_report [--model hybrid] [--report path] [--save hooly/assets/calibration.json]
"""
import argparse
import glob
import json
import os

from evaluation.metrics import summarize
from hooly.calibrate import Isotonic


def fit_maps(records):
    maps = {}
    for lab in ("true", "false"):
        rs = [r for r in records if r["label"] == lab]
        maps[lab] = Isotonic().fit([r["probs"][lab] for r in rs], [int(r["gold"] == lab) for r in rs])
    return maps


def apply(rec, maps):
    r = dict(rec)
    lab = r["label"]
    if lab not in maps:
        return r
    c = maps[lab](r["probs"][lab])
    if lab == "false" and c < 0.5:
        lab, c = "true", 1 - c
    pu = r["probs"].get("unsure", 0.0)
    other = "false" if lab == "true" else "true"
    r["label"], r["confidence"] = lab, max(0.5, c)
    r["probs"] = {lab: c * (1 - pu), other: (1 - c) * (1 - pu), "unsure": pu}
    return r


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hybrid")
    ap.add_argument("--report", default=None)
    ap.add_argument("--save", default=None)
    args = ap.parse_args(argv)
    path = args.report or [p for p in sorted(glob.glob("evaluation/results/*-parsed.json"))
                           if args.model in json.load(open(p))["records"]][-1]
    recs = json.load(open(path))["records"][args.model]
    folds = sorted({r["fold"] for r in recs})
    out = []
    for k in folds:
        maps = fit_maps([r for r in recs if r["fold"] != k])
        out += [apply(r, maps) for r in recs if r["fold"] == k]
    before, after = summarize(recs), summarize(out)
    print(f"report: {path}  model: {args.model}")
    for name, s in (("raw", before), ("calibrated", after)):
        print(f"  {name:10s} acc {s['accuracy']:.3f} ({s['accuracy_95ci'][0]:.3f}-{s['accuracy_95ci'][1]:.3f})  "
              f"F1 false {s['per_class']['false']['f1']:.3f}  Brier {s['brier']:.3f}  ECE {s['ece']:.3f}  "
              f"predicted {s['predicted']}")
    print("  reliability after calibration (confidence -> accuracy):")
    for b in after["reliability"]:
        print(f"    {b['bin']}: n={b['n']:4d}  said {b['confidence']:.2f}  right {b['accuracy']:.2f}")
    if args.save:
        maps = fit_maps(recs)
        os.makedirs(os.path.dirname(args.save), exist_ok=True)
        with open(args.save, "w") as f:
            json.dump({lab: m.table() for lab, m in maps.items()}, f)
        print(f"  runtime map saved: {args.save}")
    return before, after


if __name__ == "__main__":
    main()
