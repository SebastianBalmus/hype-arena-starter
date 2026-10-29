"""Reader quality on held-out pages: atom precision / recall / F1 against gold forms.

    python3 -m evaluation.reader_eval
"""
import collections
from dataclasses import replace

from hooly.data import build_names, load_lexicon, load_rows
from hooly.folds import grouped_folds
from hooly.reader import Reader, atom_match


def main(k=5, seed=0, threshold=0.35, bool_threshold=0.5):
    rows = load_rows("data/practice.csv")
    lex = load_lexicon("data/lexicon.json")
    folds = grouped_folds(rows, k, seed)
    tp = fp = fn = 0
    subj_ok = attr_ok = n_gold = 0
    by_kind = collections.defaultdict(lambda: [0, 0, 0])
    for test in folds:
        view = [replace(r, atoms=None, label=None, meta={}) if r.id in test else r for r in rows]
        names = build_names(lex, [r for r in view if r.atoms])
        reader = Reader(threshold, bool_threshold).fit([r for r in view if r.id not in test], lex, names)
        parsed = reader.parse_all(view)
        gold = {r.id: r.atoms for r in rows}
        for rid in test:
            P, G = parsed.get(rid, []), gold[rid]
            used = set()
            for g in G:
                n_gold += 1
                hit = next((i for i, p in enumerate(P) if i not in used and atom_match(p, g, names)), None)
                kk = g.value["kind"]
                if hit is None:
                    fn += 1
                    by_kind[kk][2] += 1
                else:
                    used.add(hit)
                    tp += 1
                    by_kind[kk][0] += 1
                subj_ok += any(p.subject == g.subject for p in P)
                attr_ok += any(p.key == g.key for p in P)
            for i, p in enumerate(P):
                if i not in used:
                    fp += 1
                    by_kind[p.value["kind"]][1] += 1
    prec, rec = tp / max(1, tp + fp), tp / max(1, tp + fn)
    print(f"atoms: precision {prec:.3f}  recall {rec:.3f}  F1 {2 * prec * rec / max(1e-9, prec + rec):.3f}  "
          f"(tp {tp}, fp {fp}, fn {fn})")
    print(f"gold atoms whose subject appears among parsed atoms: {subj_ok / n_gold:.3f}; whose (subject, attribute): {attr_ok / n_gold:.3f}")
    for kk, (a, b, c) in sorted(by_kind.items(), key=lambda x: -(x[1][0] + x[1][2])):
        p_, r_ = a / max(1, a + b), a / max(1, a + c)
        print(f"  {kk:9s} precision {p_:.3f} recall {r_:.3f}  (tp {a}, fp {b}, fn {c})")


if __name__ == "__main__":
    main()
