"""Stress tests for the hidden data, end to end from text (parsed mode).

    python3 -m evaluation.robustness --split subjects [--models always_true windows_shape hybrid]
    python3 -m evaluation.robustness --split sparse --frac 0.1 --seeds 0 1 2

subjects  Subjects are split into 5 groups. Each round, every row whose facts involve a held-out
          subject loses its label and its gold facts and is scored; other rows that mention those
          subjects are hidden too (no anchors leak) but not scored. The Reader never sees their
          forms; the lexicon still lists them, as the runtime lexicon will.
sparse    Only a random fraction of pages keeps its labels (e.g. 10%); the rest is hidden and
          scored. The opposite extreme from the usual 80%.

Reports raw metrics and, separately, ECE after applying the shipped calibration map.
"""
from __future__ import annotations

import argparse
import random
import time
from dataclasses import replace

from evaluation.metrics import summarize
from evaluation.run import MODELS
from hooly.data import build_names, load_lexicon, load_rows
from hooly.folds import page_groups
from hooly.pipeline import Pipeline
from hooly.run import calibrated, load_calibration


def subject_rounds(rows, k=5, seed=0):
    subjects = sorted({a.subject for r in rows if r.atoms for a in r.atoms})
    random.Random(seed).shuffle(subjects)
    groups = [set(subjects[i::k]) for i in range(k)]
    for held in groups:
        primary = {r.id for r in rows if r.atoms and r.atoms[0].subject in held}
        touched = {r.id for r in rows if r.atoms and any(a.subject in held for a in r.atoms)}
        yield primary, touched  # score primary; hide everything touched


def sparse_rounds(rows, frac, seeds):
    for s in seeds:
        groups = page_groups([r for r in rows if r.label])
        random.Random(s).shuffle(groups)
        keep, n = set(), 0
        for g in groups:
            if n >= frac * len(rows):
                break
            keep.update(r.id for r in g)
            n += len(g)
        hidden = {r.id for r in rows if r.id not in keep}
        yield hidden, hidden


def run(model_name, rows, lexicon, rounds, maps, gold_facts=False):
    recs = []
    for k, (score_ids, hide_ids) in enumerate(rounds):
        if gold_facts:  # hidden rows keep their gold facts: the truth model alone, parser out of the picture
            view = [replace(r, label=None, meta={}) if r.id in hide_ids else r for r in rows]
        else:
            view = [replace(r, label=None, atoms=None, meta={}) if r.id in hide_ids else r for r in rows]
        names = build_names(lexicon, [r for r in view if r.atoms])
        model = (MODELS[model_name]() if gold_facts else Pipeline(MODELS[model_name])).fit(view, lexicon, names)
        gold = {r.id: r.label for r in rows}
        for r in view:
            if r.id in score_ids:
                p = model.predict(r)
                lab, conf = calibrated(p, maps)
                recs.append({"id": r.id, "fold": k, "page": r.page, "gold": gold[r.id], "label": p.label,
                             "confidence": p.confidence or 0.5, "probs": p.probs,
                             "cal_label": lab, "cal_conf": conf or 0.5})
    return recs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["subjects", "sparse"], required=True)
    ap.add_argument("--models", nargs="+", default=["always_true", "windows_shape", "hybrid"])
    ap.add_argument("--frac", type=float, default=0.1)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--gold", action="store_true", help="hidden rows keep gold facts (no parser)")
    args = ap.parse_args(argv)
    rows, lexicon = load_rows("data/practice.csv"), load_lexicon("data/lexicon.json")
    maps = load_calibration("hooly/assets/calibration.json")
    for m in args.models:
        t = time.time()
        rounds = list(subject_rounds(rows) if args.split == "subjects" else sparse_rounds(rows, args.frac, args.seeds))
        recs = run(m, rows, lexicon, rounds, maps, args.gold)
        s = summarize(recs)
        cal = summarize([{**r, "label": r["cal_label"], "confidence": r["cal_conf"]} for r in recs])
        print(f"{args.split}{' (gold facts)' if args.gold else ''} | {m} | n={s['n']} | acc {s['accuracy']:.3f} ({s['accuracy_95ci'][0]:.3f}-{s['accuracy_95ci'][1]:.3f}) | "
              f"macro F1 {s['macro_f1']:.3f} | F1 false {s['per_class']['false']['f1']:.3f} | ECE raw {s['ece']:.3f} | "
              f"calibrated: acc {cal['accuracy']:.3f}, ECE {cal['ece']:.3f} | {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
