"""Where exactly do our mistakes come from? Every held-out prediction, sorted by cause.

    python3 -m evaluation.error_map --report evaluation/results/<run>.json --model gated

parse   for each row, parsed facts vs gold facts: nothing parsed / wrong fact / missing fact /
        facts right
truth   for rows whose facts were parsed right: the timeline state of the deciding fact
"""
import argparse
import collections
import glob
import json

from hooly.data import build_names, load_lexicon, load_rows
from hooly.data import Atom
from hooly.reader import atom_match


def parse_category(parsed, gold, names):
    if not parsed:
        return "nothing parsed"
    P = [Atom(a["subject"], a["attribute"], a["value"]) for a in parsed]
    ok = [any(atom_match(p, g, names) for g in gold) for p in P]
    if not all(ok):
        return "a wrong fact parsed"
    if len(P) < len(gold):
        return "a fact missing"
    return "facts right"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=None)
    ap.add_argument("--model", default="gated")
    a = ap.parse_args(argv)
    path = a.report or [p for p in sorted(glob.glob("evaluation/results/*-parsed.json"))
                        if a.model in json.load(open(p))["records"]][-1]
    recs = json.load(open(path))["records"][a.model]
    rows = {r.id: r for r in load_rows("data/practice.csv")}
    lex = load_lexicon("data/lexicon.json")
    names = build_names(lex, list(rows.values()))
    cells = collections.Counter()
    states = collections.Counter()
    totals = collections.Counter()
    for r in recs:
        atoms = [x for x in r["trace"].get("atoms", []) if "subject" in x]
        cat = parse_category(atoms, rows[r["id"]].atoms, names)
        totals[cat] += 1
        if r["label"] == r["gold"]:
            continue
        kind = ("missed lie" if r["gold"] == "false" and r["label"] == "true" else
                "false alarm" if r["gold"] == "true" and r["label"] == "false" else
                "missed unsure" if r["gold"] == "unsure" else "other")
        cells[(cat, kind)] += 1
        if cat == "facts right" and atoms:
            w = min(atoms, key=lambda x: x.get("p_true", 1))
            states[(kind, w.get("state", "?").split("|")[0].rsplit("_", 1)[0] if w.get("state", "").startswith(("before", "after")) else w.get("state", "?").split("|")[0])] += 1
    n_err = sum(cells.values())
    print(f"report {path} | model {a.model} | {len(recs)} rows | {n_err} errors ({n_err / len(recs):.1%})\n")
    print(f"{'parse outcome':22s} {'rows':>5s} {'errors':>7s} {'missed lie':>11s} {'false alarm':>12s} {'missed unsure':>14s}")
    for cat in ("facts right", "a fact missing", "a wrong fact parsed", "nothing parsed"):
        e = {k: cells[(cat, k)] for k in ("missed lie", "false alarm", "missed unsure")}
        print(f"{cat:22s} {totals[cat]:5d} {sum(e.values()):7d} {e['missed lie']:11d} {e['false alarm']:12d} {e['missed unsure']:14d}")
    print("\nerrors on rows parsed right, by the deciding fact's timeline state:")
    for (kind, st), c in states.most_common(12):
        print(f"  {c:4d}  {kind:13s} {st}")


if __name__ == "__main__":
    main()
