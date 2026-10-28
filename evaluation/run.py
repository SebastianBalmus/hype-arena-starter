"""Run models through page-grouped cross-validation on the practice data.

    python3 -m evaluation.run                         # baselines, gold forms (oracle parser)
    python3 -m evaluation.run --models latest_before --mode parsed

--mode gold    test rows keep their gold atoms but lose their labels: measures the truth
               model alone, as if parsing were perfect.
--mode parsed  test rows lose labels and atoms: the model must parse the text itself.

Held-out rows stay in the input as unlabelled rows, as they would at runtime.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace

from hooly.baselines import AlwaysTrue, LatestBefore, Nearest
from hooly.temporal import ConservativeWindows, TemporalWindows, TemporalWindowsPlus
from hooly.combined import CredibilityOnly, WindowsCredibility
from hooly.evidence import WindowsEvidence, WindowsVotesStack
from hooly.data import build_names, load_lexicon, load_rows
from evaluation.folds import grouped_folds
from evaluation.metrics import summarize

MODELS = {
    "always_true": AlwaysTrue,
    "latest_before": LatestBefore,
    "nearest": Nearest,
    "windows": TemporalWindows,
    "windows_plus": TemporalWindowsPlus,
    "credibility_only": CredibilityOnly,
    "windows_cred": WindowsCredibility,
    "windows_evidence": WindowsEvidence,
    "windows_votes": WindowsVotesStack,
    "conservative": ConservativeWindows,
}


def run_model(factory, rows, lexicon, folds, mode):
    records = []
    for k, test_ids in enumerate(folds):
        view = []
        for r in rows:
            if r.id in test_ids:
                r = r.hidden()
                if mode == "parsed":
                    r = replace(r, atoms=None)
            view.append(r)
        # Name spellings may only come from what the model can see.
        names = build_names(lexicon, [r for r in view if r.atoms])
        model = factory().fit(view, lexicon, names)
        gold = {r.id: r for r in rows}
        for r in view:
            if r.id in test_ids:
                p = model.predict(r)
                records.append({"id": r.id, "fold": k, "page": r.page, "date": r.date,
                                "text": r.text, "gold": gold[r.id].label, "label": p.label,
                                "confidence": p.confidence if p.confidence is not None else 0.5,
                                "probs": p.probs, "trace": p.trace})
    return records


def fmt_table(results):
    cols = ["model", "acc", "95% CI", "macroF1", "F1 false", "F1 unsure", "brier", "ECE", "top50%acc"]
    lines = [" | ".join(cols), " | ".join("---" for _ in cols)]
    for name, s in results.items():
        lo, hi = s["accuracy_95ci"]
        lines.append(" | ".join([
            name, f"{s['accuracy']:.3f}", f"{lo:.3f}-{hi:.3f}", f"{s['macro_f1']:.3f}",
            f"{s['per_class']['false']['f1']:.3f}", f"{s['per_class']['unsure']['f1']:.3f}",
            f"{s['brier']:.3f}", f"{s['ece']:.3f}", f"{s['selective_accuracy'].get('50%', float('nan')):.3f}",
        ]))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--practice", default="data/practice.csv")
    ap.add_argument("--lexicon", default="data/lexicon.json")
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--mode", choices=["gold", "parsed"], default="gold")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="evaluation/results")
    args = ap.parse_args(argv)

    rows = load_rows(args.practice)
    lexicon = load_lexicon(args.lexicon)
    folds = grouped_folds(rows, args.folds, args.seed)
    print(f"{len(rows)} rows, {len({r.page for r in rows})} pages, "
          f"{args.folds} page-grouped folds of {[len(f) for f in folds]} rows, mode={args.mode}")

    results, all_records = {}, {}
    for name in args.models:
        t = time.time()
        recs = run_model(MODELS[name], rows, lexicon, folds, args.mode)
        results[name] = summarize(recs)
        all_records[name] = recs
        print(f"  {name}: {time.time() - t:.1f}s")
    print()
    print(fmt_table(results))

    os.makedirs(args.out, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(args.out, f"{stamp}-{args.mode}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "summary": results, "records": all_records}, f, indent=1, default=str)
    print(f"\nfull report: {path}")
    return results


if __name__ == "__main__":
    main()
