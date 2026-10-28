"""Why does the Reader miss gold facts? Each missed fact of held-out pages (first 2 folds), by
cause: subject not among the candidates, yes/no fact below threshold, value not extracted,
pair score below threshold, or lost to another pair. Also samples name false positives.

    python3 -m evaluation.parse_misses
"""
import collections, re, random
from dataclasses import replace
from hooly.data import build_names, load_lexicon, load_rows
from hooly.folds import grouped_folds
from hooly.reader import Reader, atom_match
from hooly.values import satisfies
rows = load_rows("data/practice.csv"); lex = load_lexicon("data/lexicon.json")
folds = grouped_folds(rows, 5, 0)
why = collections.Counter(); ex = collections.defaultdict(list); fpn = []
for test in folds[:2]:
    view = [replace(r, atoms=None, label=None, meta={}) if r.id in test else r for r in rows]
    names = build_names(lex, [r for r in view if r.atoms])
    rd = Reader(0.3, 0.4).fit([r for r in view if r.id not in test], lex, names)
    pages = rd._page_index(view)
    for pr in pages.values():
        for r in pr:
            if r.atoms is None:
                r.meta["cands"] = rd._subjects(r, pages)
                r.meta["found"] = rd.extractor.find(r.text)
                r.atoms = rd.parse(r, pages)
                r.meta["probs"] = dict(rd.last_probs)
                # all pair scores for diagnosis
                sc = {}
                for i, f in enumerate(r.meta["found"]):
                    for key, feats in rd._pairs(r, f, r.meta["cands"]):
                        sc[(i, key)] = rd.model.predict_proba(feats)
                r.meta["sc"] = sc
    V = {v.id: v for v in view}
    for r in rows:
        if r.id not in test: continue
        v = V[r.id]
        for g in r.atoms:
            if any(atom_match(p, g, names) for p in v.atoms): continue
            if g.subject not in v.meta["cands"]:
                c = "subject not a candidate"
            elif g.value["kind"] == "boolean":
                c = "boolean below threshold"
            else:
                idx = [i for i, f in enumerate(v.meta["found"]) if f.kind == g.value["kind"] and satisfies({k:x for k,x in g.value.items() if k!="op"}, {k:x for k,x in f.value.items() if k!="op"}, names)]
                if not idx:
                    c = "value not extracted"
                else:
                    s = max(v.meta["sc"].get((i, g.key), -1) for i in idx)
                    if s < 0: c = "pair filtered (kind)"
                    elif s < 0.3: c = "score below threshold"
                    else: c = "lost to another pair"
                    if c != "pair filtered (kind)": c += f" ({'≥.15' if s>=.15 else '<.15'})" if s < 0.3 else ""
            why[(g.value["kind"], c)] += 1
            ex[(g.value["kind"], c)].append((r.text[:110], g.subject, g.attribute, g.value.get("value"), r.label))
        for p in v.atoms:
            if not any(atom_match(p, g, names) for g in r.atoms) and p.value["kind"] == "name":
                fpn.append((r.text[:100], p.subject, p.attribute, p.value["value"], [(g.subject, g.attribute, g.value.get("value")) for g in r.atoms][:2]))
for c, n in why.most_common(): print(f"{n:4d} {c}")
random.seed(1)
for c in ex:
    print("\n==", c)
    for e in random.sample(ex[c], min(10, len(ex[c]))): print("  ", e)
print("\n== name FPs", len(fpn))
for e in random.sample(fpn, min(25, len(fpn))): print("  ", e)
