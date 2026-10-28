"""Chunked CV: python3 -m evaluation.self_train_cv TH FOLD [FOLD...]; python3 -m scripts._st TH sum"""
import sys, json, os
import evaluation.run as R
from evaluation.metrics import summarize
from hooly.pipeline import Pipeline
from hooly.data import load_rows, load_lexicon
from evaluation.folds import grouped_folds
TH = float(sys.argv[1]) if sys.argv[1] != "none" else None
out = f"evaluation/results/st-{sys.argv[1]}"
os.makedirs(out, exist_ok=True)
if sys.argv[2] == "sum":
    recs = []
    for f in sorted(os.listdir(out)):
        recs += json.load(open(os.path.join(out, f)))
    s = summarize(recs)
    print(TH, "rows", len(recs), "acc", round(s["accuracy"], 4), "F1 false", round(s["per_class"]["false"]["f1"], 3), "ECE", round(s["ece"], 3))
    json.dump({"records": {"gated": recs}}, open(out + ".json", "w"), default=str)
    sys.exit()
class PST(Pipeline):
    def __init__(self, cls, **kw): super().__init__(cls, self_train=TH, **kw)
R.Pipeline = PST
rows = load_rows("data/practice.csv"); lex = load_lexicon("data/lexicon.json")
folds = grouped_folds(rows, 5, 0)
for k in map(int, sys.argv[2:]):
    recs = R.run_model(R.MODELS["gated"], rows, lex, [folds[k]], "parsed")
    for r in recs: r["fold"] = k
    json.dump(recs, open(f"{out}/fold{k}.json", "w"), default=str)
    print("fold", k, "done", sum(r["label"] == r["gold"] for r in recs), "/", len(recs))
