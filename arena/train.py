"""Offline training: builds every model artifact (runs inside ``docker build``)."""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pickle
import warnings

from scipy.optimize import OptimizeWarning
from sklearn.model_selection import GroupKFold

from arena.attributes import AttributeModel
from arena.data import load_practice
from arena.decide import Decider, row_features
from arena.evidence import EvidenceStore
from arena.extract import Extractor
from arena.kinds import KindIndex
from arena.lexicon import Lexicon, NameIndex
from arena.values import values_equal



def _names(gold) -> list[str]:
    return sorted({f.value["value"] for fs in gold.values() for f in fs if f.value.get("kind") == "name"})


def _examples(rows, gold):
    return [(r.text, f.value, f.subject, f.attribute) for r in rows for f in gold.get(r.id, [])]


def build_artifacts(practice_csv: str, lexicon_json: str, out_dir: str) -> dict:
    # sklearn passes an `iprint` option that newer scipy ignores with a warning.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unknown solver options", category=OptimizeWarning)
        return _build(practice_csv, lexicon_json, out_dir)


def _build(practice_csv: str, lexicon_json: str, out_dir: str) -> dict:
    rows, gold = load_practice(practice_csv)
    lexicon = Lexicon.load(lexicon_json)
    X, y, groups = [], [], []
    n_gold = n_hit = 0
    for tr, te in GroupKFold(n_splits=5).split(rows, groups=[r.page for r in rows]):
        train_rows = [rows[i] for i in tr]
        fold_rows = [rows[i] for i in te]
        fold_ids = {r.id for r in fold_rows}
        tgold = {r.id: gold[r.id] for r in train_rows}
        model = AttributeModel().fit(_examples(train_rows, gold))
        kinds = KindIndex.from_gold(tgold, train_rows)
        names = NameIndex.from_values(_names(tgold), lexicon)
        extracted = Extractor(lexicon, names, kinds, model).extract(fold_rows)
        store = EvidenceStore()
        for r in rows:
            if r.id in fold_ids:
                store.add_row(dataclasses.replace(r, label=""), extracted[r.id])
            else:
                store.add_row(r, gold[r.id])
        store.build()
        for r in fold_rows:
            facts = extracted[r.id]
            checks = [store.check(f, r.date) for f in facts]
            X.append(row_features(facts, checks, store.page_reliability.get(r.page, 0.8)))
            y.append(r.label)
            groups.append(r.page)
            for g in gold[r.id]:
                n_gold += 1
                n_hit += any(f.subject == g.subject and f.attribute == g.attribute
                             and values_equal(f.value, g.value) for f in facts)
    decider = Decider().fit(X, y, groups)
    oof = decider.cv_report.pop("oof_pred")
    n = len(y)
    tp = sum(p == "false" and t == "false" for p, t in zip(oof, y))
    pf = sum(p == "false" for p in oof)
    tf = sum(t == "false" for t in y)
    labels = sorted(set(y) | set(oof))
    metrics = {
        "label_accuracy": sum(p == t for p, t in zip(oof, y)) / n,
        "always_true_accuracy": sum(t == "true" for t in y) / n,
        "false_precision": tp / pf if pf else 0.0,
        "false_recall": tp / tf if tf else 0.0,
        "unsure_rate": sum(p == "unsure" for p in oof) / n,
        "fact_accuracy": n_hit / n_gold if n_gold else 0.0,
        "confusion": {t: {p: sum(a == t and b == p for a, b in zip(y, oof)) for p in labels} for t in labels},
    }
    os.makedirs(out_dir, exist_ok=True)
    artifacts = {
        "kinds.pkl": KindIndex.from_gold(gold, rows),
        "names.pkl": NameIndex.from_values(_names(gold), lexicon),
        "attribute_model.pkl": AttributeModel().fit(_examples(rows, gold)),
        "decider.pkl": decider,
    }
    for fname, obj in artifacts.items():
        with open(os.path.join(out_dir, fname), "wb") as f:
            pickle.dump(obj, f)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--practice", default="data/practice.csv")
    ap.add_argument("--lexicon", default="data/lexicon.json")
    a = ap.parse_args()
    print(json.dumps(build_artifacts(a.practice, a.lexicon, a.out), indent=2))


if __name__ == "__main__":
    main()
