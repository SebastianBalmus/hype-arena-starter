"""How many gold (non-boolean) values does the extractor recover from practice sentences?

    python3 -m evaluation.extract_recall
"""
import collections
import sys

from hooly.data import build_names, load_lexicon, load_rows, name_values
from hooly.extract import Extractor, _singular
from hooly.values import satisfies


def main():
    rows = load_rows("data/practice.csv")
    lex = load_lexicon("data/lexicon.json")
    nouns = {_singular(a.value["unit"].lower()) for r in rows for a in r.atoms
             if a.value["kind"] == "quantity" and a.value["unit"].isalpha() and len(a.value["unit"]) >= 3}
    units = set(nouns)
    nouns |= {_singular(at.split("_")[-1]) for s in lex["subjects"] for at in s["attributes"]}
    names_norm = build_names(lex, rows)
    protected = [n for s in lex["subjects"] for n in [s["name"]] + s["aliases"]] + list(lex["events"])
    ex = Extractor(nouns=nouns, names=name_values(rows), protected=protected, units=units)
    hit, tot = collections.Counter(), collections.Counter()
    extra = 0
    misses = collections.defaultdict(list)
    for r in rows:
        found = ex.find(r.text)
        used = set()
        for a in r.atoms:
            k = a.value["kind"]
            if k == "boolean":
                continue
            tot[k] += 1
            ok = False
            for i, f in enumerate(found):
                if f.kind == k and f.value.get("op", "eq") == a.value.get("op", "eq") and \
                        satisfies({k2: v for k2, v in a.value.items() if k2 != "op"},
                                  {k2: v for k2, v in f.value.items() if k2 != "op"}, names_norm):
                    ok = True
                    used.add(i)
                    break
            hit[k] += ok
            if not ok:
                misses[k].append((r.text, a.value, [f.value for f in found]))
        extra += len(found) - len(used)
    total_hit, total = sum(hit.values()), sum(tot.values())
    print(f"recall: {total_hit}/{total} = {total_hit / total:.3f}   (extra values found: {extra})")
    for k in sorted(tot, key=lambda k: -tot[k]):
        print(f"  {k:9s} {hit[k]:4d}/{tot[k]:4d} = {hit[k] / tot[k]:.3f}")
    if "-v" in sys.argv:
        for k, ms in misses.items():
            print(f"\n--- missed {k}")
            for t, v, f in ms[:12]:
                print(f"  {t[:90]!r}\n      gold {v}\n      found {f}")


if __name__ == "__main__":
    main()
