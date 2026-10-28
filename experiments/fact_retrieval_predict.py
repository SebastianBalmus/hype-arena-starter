#!/usr/bin/env python3
"""Cross-fitted retrieval classifier with split-scoped claim features."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.kb_predict import fingerprint, match_training_forms, read_csv
from experiments.retrieval_predict import RetrievalClassifier, _safe_json
from retrieval.features import RetrievalIndex
from retrieval.claim_features import ClaimFeatureIndex


class CombinedIndex:
    """Text retrieval plus generated/observed claims, with candidate provenance."""

    def __init__(self, reference_rows, lexicon, views, mode, scope):
        self.rows = list(reference_rows)
        self.raw = RetrievalIndex(self.rows, lexicon, views=views, k=5)
        self.claims = ClaimFeatureIndex(self.rows, lexicon, mode=mode, limit=1000)
        self.feature_names_ = list(self.raw.feature_names_) + [
            "claim:" + n for n in self.claims.feature_names_
        ]
        self.scope = scope

    def transform(self, queries):
        raw_x, raw_trace = self.raw.transform(queries)
        claim_x, claim_trace = self.claims.transform(queries)
        import numpy as np
        features = np.hstack([raw_x, claim_x])
        for base, aux in zip(raw_trace, claim_trace):
            base["auxiliary_claims"] = aux
        return features, raw_trace

    def summary(self):
        claim_summary = self.claims.summary
        if callable(claim_summary):
            claim_summary = claim_summary()
        return {"text": self.raw.summary(), "claims": claim_summary, "scope": self.scope}

    def export_records(self):
        return self.claims.export_records()


class FactRetrievalClassifier(RetrievalClassifier):
    def __init__(self, lexicon, mode="observed", seed=42):
        if mode not in ("observed", "generated"):
            raise ValueError("mode must be observed or generated")
        super().__init__(lexicon, variant="raw_masked", seed=seed)
        self.mode = mode
        self.claim_catalogues_ = []

    def fit(self, training_rows):
        self.claim_catalogues_ = []
        return super().fit(training_rows)

    def _make_index(self, reference_rows):
        scope = f"fold-{len(self.claim_catalogues_)}" if len(self.claim_catalogues_) < 5 else "full"
        index = CombinedIndex(reference_rows, self.lexicon, self.views, self.mode, scope)
        summary = index.claims.summary
        if callable(summary):
            summary = summary()
        self.claim_catalogues_.append({
            "scope": scope,
            "reference_ids": [r["id"] for r in reference_rows],
            "records": index.export_records(),
            "summary": summary,
        })
        return index

    def diagnostics(self):
        result = super().diagnostics()
        result.update({"mode": self.mode, "claim_scopes": [x["scope"] for x in self.claim_catalogues_],
                       "claim_catalogue_counts": {x["scope"]: len(x["records"]) for x in self.claim_catalogues_}})
        return result


def _write_json(path, value):
    path.write_text(json.dumps(_safe_json(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--training-forms", required=True, type=Path)
    parser.add_argument("--lexicon", required=True, type=Path)
    parser.add_argument("--mode", choices=("observed", "generated"), default="observed")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        start = time.perf_counter()
        rows = read_csv(args.input)
        forms = match_training_forms(rows, read_csv(args.training_forms, forms=True))
        if not any(not r["label"] for r in rows):
            raise ValueError("input contains no pending rows")
        forms_by_id = {r["id"]: r["form"] for r in forms}
        training = [{**r, "form": forms_by_id[r["id"]]} for r in rows if r["label"]]
        lexicon = json.loads(args.lexicon.read_text(encoding="utf-8"))
        model = FactRetrievalClassifier(lexicon, args.mode, args.seed).fit(training)
        fit_end = time.perf_counter()
        pending = [r for r in rows if not r["label"]]
        predictions, traces = model.predict_rows(pending)
        infer_end = time.perf_counter()
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.artifacts.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "label", "confidence", "decision_reason", "p_true", "p_false", "p_unsure"])
        writer.writeheader(); writer.writerows(predictions)
    def jsonl(path, records):
        path.write_text("".join(json.dumps(_safe_json(r), ensure_ascii=False, allow_nan=False) + "\n" for r in records), encoding="utf-8")
    jsonl(args.artifacts / "traces.jsonl", traces)
    jsonl(args.artifacts / "oof_traces.jsonl", model.oof_traces_)
    _write_json(args.artifacts / "oof_folds.json", model.oof_folds_)
    _write_json(args.artifacts / "feature_names.json", model.feature_names_)
    scopes = {scope["scope"]: {"reference_ids": scope["reference_ids"], "records": scope["records"]}
              for scope in model.claim_catalogues_}
    with gzip.open(args.artifacts / "claim_catalogues.json.gz", "wt", encoding="utf-8") as stream:
        json.dump(_safe_json(scopes), stream, ensure_ascii=False, allow_nan=False)
    summaries = {s["scope"]: s["summary"] for s in model.claim_catalogues_}
    _write_json(args.artifacts / "claims_summary.json", {"mode": args.mode, "scopes": summaries})
    diagnostics = model.diagnostics()
    diagnostics["retrieval_index"] = model.full_index_.summary()
    diagnostics["training_source"] = {
        "row_count": len(training),
        "ids": [r["id"] for r in training],
        "fingerprints": {r["id"]: fingerprint(r) for r in training},
        "forms_sha256": hashlib.sha256(args.training_forms.read_bytes()).hexdigest(),
        "lexicon_sha256": hashlib.sha256(args.lexicon.read_bytes()).hexdigest(),
        "runtime_input_has_forms": False,
    }
    diagnostics["confidence_policy"] = {"threshold": 0.5, "probabilities": "uncalibrated logistic regression"}
    diagnostics["timings_seconds"] = {"fit": fit_end - start, "inference": infer_end - fit_end,
                                      "serialization": time.perf_counter() - infer_end}
    _write_json(args.artifacts / "diagnostics.json", diagnostics)


if __name__ == "__main__":
    main()
