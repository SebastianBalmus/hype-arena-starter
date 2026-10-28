"""Yes/no facts: precision and recall of the Reader per threshold, with wrong polarity counted
separately (right fact, wrong yes/no). First 3 page-grouped folds.

    python3 -m evaluation.bool_threshold
"""
import collections
from dataclasses import replace
from hooly.data import build_names, load_lexicon, load_rows
from hooly.folds import grouped_folds
from hooly.reader import Reader, atom_match
rows = load_rows("data/practice.csv"); lex = load_lexicon("data/lexicon.json")
folds = grouped_folds(rows, 5, 0)
G = {r.id: r for r in rows}
res = collections.defaultdict(lambda: [0, 0, 0, 0])
for test in folds[:3]:
    view = [replace(r, atoms=None, label=None, meta={}) if r.id in test else r for r in rows]
    names = build_names(lex, [r for r in view if r.atoms])
    rd = Reader(0.3, 0.5).fit([r for r in view if r.id not in test], lex, names)
    for t in (0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1):
        rd.bool_threshold = t
        v2 = [replace(r, atoms=None, meta={}) if r.id in test else r for r in view]
        parsed = rd.parse_all(v2)
        for rid in test:
            P = [p for p in parsed[rid] if p.value["kind"] == "boolean"]
            Gb = [g for g in G[rid].atoms if g.value["kind"] == "boolean"]
            for p in P:
                g = next((g for g in Gb if g.key == p.key), None)
                if g is None: res[t][1] += 1
                elif g.value["value"] == p.value["value"]: res[t][0] += 1
                else: res[t][3] += 1   # right fact, wrong polarity
            res[t][2] += sum(1 for g in Gb if not any(p.key == g.key for p in P))
for t, (tp, fp, fn, pol) in sorted(res.items(), reverse=True):
    print(f"t={t:.2f} tp {tp} wrong-polarity {pol} fp {fp} missed {fn} | precision {tp/max(1,tp+fp+pol):.2f} recall {tp/max(1,tp+fn+pol):.2f}")
