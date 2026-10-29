"""Parser quality on unseen subjects (subject-grouped rounds), vs gold forms.

    python3 -m evaluation.reader_shift [--threshold 0.3]
"""
import argparse
import collections
from dataclasses import replace

from evaluation.robustness import subject_rounds
from hooly.data import build_names, load_lexicon, load_rows
from hooly.reader import Reader, atom_match


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--dropout", type=float, default=0.35)
    a = ap.parse_args(argv)
    rows = load_rows("data/practice.csv"); lex = load_lexicon("data/lexicon.json")
    gold = {r.id: r.atoms for r in rows}
    tp = fp = fn = 0
    reasons = collections.Counter()
    for score_ids, hide_ids in subject_rounds(rows):
        view = [replace(r, label=None, atoms=None, meta={}) if r.id in hide_ids else r for r in rows]
        names = build_names(lex, [r for r in view if r.atoms])
        rd = Reader(a.threshold, 0.5, cue_dropout=a.dropout).fit([r for r in view if r.atoms], lex, names)
        rd.parse_all(view)
        P = {r.id: r.atoms for r in view if r.id in score_ids}
        for rid in score_ids:
            used = set()
            for g in gold[rid]:
                hit = next((i for i, p in enumerate(P[rid]) if i not in used and atom_match(p, g, names)), None)
                if hit is None:
                    fn += 1
                    ps = P[rid]
                    reasons["nothing parsed" if not ps else "wrong subject" if not any(p.subject == g.subject for p in ps)
                            else "wrong attribute" if not any(p.key == g.key for p in ps) else "wrong value"] += 1
                else:
                    used.add(hit)
                    tp += 1
            fp += len(P[rid]) - len(used)
    prec, rec = tp / max(1, tp + fp), tp / max(1, tp + fn)
    print(f"unseen subjects (threshold {a.threshold}, cue dropout {a.dropout}): precision {prec:.3f} recall {rec:.3f} "
          f"F1 {2 * prec * rec / max(1e-9, prec + rec):.3f}; misses: {reasons.most_common()}")


if __name__ == "__main__":
    main()
