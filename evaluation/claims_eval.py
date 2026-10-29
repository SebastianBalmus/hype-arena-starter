"""How many generated claims hold up against evidence the generator did not see?

For each page-grouped fold, claims are generated with that fold's labels hidden, then
checked against timelines built from *all* labelled practice rows:
  confirmed      the full data agrees
  contradicted   the full data disagrees (e.g. a change hidden pages reveal)
  unverifiable   the full data cannot tell (value falls in a gap, no recent sighting, ...)

    python3 -m evaluation.claims_eval
"""
import collections
from dataclasses import replace

from hooly.data import build_names, load_lexicon, load_rows
from hooly.folds import grouped_folds
from hooly.values import matches, normalize, satisfies
from hooly.writer import Writer, generate_claims


def run_value_at(tl, d):
    for r in tl.runs:
        if r.start <= d <= r.end:
            return r.values[0]
    return None


def check(c, full: Writer, names):
    f, t = c["form"], c["type"]
    if t == "change":
        tl = full.timelines.get((f["subject"], f["attribute"]))
        if not tl:
            return "unverifiable"
        idx_from = [i for i, r in enumerate(tl.runs) if any(matches(f["from"], v, names) for v in r.values)]
        idx_to = [i for i, r in enumerate(tl.runs) if any(matches(f["to"], v, names) for v in r.values)]
        if any(i < j for i in idx_from for j in idx_to):
            return "confirmed"
        return "contradicted" if idx_from and idx_to else "unverifiable"
    if t == "at_event":
        evd = full.event_dates().get(f["event"])
        tl = full.timelines.get((f["subject"], f["attribute"]))
        if not evd or not tl:
            return "unverifiable"
        v = run_value_at(tl, evd)
        if v is None:
            return "unverifiable"
        return "confirmed" if satisfies(f["value"], v, names) else "contradicted"
    if t == "relation":
        tl = full.timelines.get((f["target"], f["attribute"]))
        if not tl:
            return "unverifiable"
        vals = [r.values[0] for r in tl.runs]
        if all(satisfies(f["value"], v, names) for v in vals):
            return "confirmed"
        return "contradicted" if not any(satisfies(f["value"], v, names) for v in vals) else "unverifiable"
    if t == "fact":
        got = full.stable_now((f["subject"], f["attribute"]))
        if not got:
            return "unverifiable"
        return "confirmed" if satisfies(f["value"], got[0], names) else "contradicted"
    if t == "compare":
        a = full.stable_now((f["left"]["subject"], f["left"]["attribute"]))
        b = full.stable_now((f["right"]["subject"], f["right"]["attribute"]))
        if not a or not b:
            return "unverifiable"
        na, nb = normalize(a[0], names), normalize(b[0], names)
        if na.dim != nb.dim:
            return "contradicted"
        x, y = (na.value[0], nb.value[0]) if na.dim == ("date",) else (na.value, nb.value)
        ok = {"gt": x > y, "lt": x < y, "eq": x == y}[f["op"]]
        return "confirmed" if ok else "contradicted"
    return "unverifiable"


def main(k=5, seed=0):
    rows = load_rows("data/practice.csv")
    lex = load_lexicon("data/lexicon.json")
    names_full = build_names(lex, rows)
    full = Writer(rows, lex, names_full)
    tally = collections.defaultdict(collections.Counter)
    bad = []
    for test in grouped_folds(rows, k, seed):
        view = [replace(r, label=None) if r.id in test else r for r in rows]
        names = build_names(lex, [r for r in view if r.atoms])
        for c in generate_claims(view, lex, names):
            v = check(c, full, names_full)
            sub = c["type"] + ("/dates" if c["type"] == "compare" and c["form"]["op"] == "lt" else
                               "/amounts" if c["type"] == "compare" else "")
            tally[sub][v] += 1
            if v == "contradicted" and len(bad) < 12:
                bad.append((sub, c["text"]))
    print("type            claims  confirmed  contradicted  unverifiable   precision (confirmed / checkable)")
    tot = collections.Counter()
    for t, c in sorted(tally.items()):
        n = sum(c.values())
        tot.update(c)
        chk = c["confirmed"] + c["contradicted"]
        print(f"{t:15s} {n:6d}  {c['confirmed']:9d}  {c['contradicted']:12d}  {c['unverifiable']:12d}   "
              f"{(c['confirmed'] / chk) if chk else float('nan'):.3f}")
    chk = tot["confirmed"] + tot["contradicted"]
    print(f"{'all':15s} {sum(tot.values()):6d}  {tot['confirmed']:9d}  {tot['contradicted']:12d}  {tot['unverifiable']:12d}   {tot['confirmed'] / chk:.3f}")
    print("\ncontradicted examples:")
    for t, text in bad:
        print(f"  [{t}] {text}")


if __name__ == "__main__":
    main()
