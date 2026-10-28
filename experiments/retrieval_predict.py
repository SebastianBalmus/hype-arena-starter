#!/usr/bin/env python3
"""Cross-fitted retrieval-augmented text classifier."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
LABELS = ("true", "false", "unsure")
sys_path_root = str(ROOT)
import sys
if sys_path_root not in sys.path:
    sys.path.insert(0, sys_path_root)
from retrieval.features import RetrievalIndex


def read_runtime_rows(path: Path):
    rows, ids = [], set()
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        expected = ["id", "text", "date", "label"]
        if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)) or set(reader.fieldnames) != set(expected):
            raise ValueError(f"{path}: expected exactly columns {expected} in any order")
        for line, item in enumerate(reader, 2):
            if None in item:
                raise ValueError(f"{path}:{line}: malformed row has extra fields")
            row = {key: (item.get(key) or "").strip() for key in expected}
            row["label"] = row["label"].lower()
            if not row["id"] or row["id"] in ids:
                raise ValueError(f"{path}:{line}: missing or duplicate id {row['id']!r}")
            if not row["text"] or row["label"] not in ("", *LABELS):
                raise ValueError(f"{path}:{line}: empty text or invalid label")
            try:
                dt.date.fromisoformat(row["date"])
            except ValueError as exc:
                raise ValueError(f"{path}:{line}: invalid ISO date {row['date']!r}") from exc
            ids.add(row["id"])
            rows.append(row)
    return rows


def _text_vectorizer():
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), max_features=20000,
                                 sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")),
        ("char", TfidfVectorizer(analyzer="char", ngram_range=(3, 5),
                                 max_features=30000, sublinear_tf=True)),
    ])


def _safe_json(value):
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    if isinstance(value, (np.floating, float)):
        x = float(value)
        return x if math.isfinite(x) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


class RetrievalClassifier:
    def __init__(self, lexicon, variant="raw", seed=42):
        if variant not in ("raw", "raw_masked"):
            raise ValueError("variant must be raw or raw_masked")
        self.lexicon, self.variant, self.seed = lexicon, variant, seed
        self.views = ("raw", "masked") if variant == "raw_masked" else ("raw",)

    def fit(self, training_rows):
        self.training_rows_ = [dict(r) for r in training_rows]
        if any(r.get("label") not in LABELS for r in self.training_rows_):
            raise ValueError("training rows must all have supported labels")
        counts = Counter(r["label"] for r in self.training_rows_)
        if set(counts) != set(LABELS) or min(counts.values()) < 5:
            raise ValueError("each of true, false, unsure needs at least 5 rows for stratified 5-fold OOF")
        ids = [r["id"] for r in self.training_rows_]
        if len(set(ids)) != len(ids):
            raise ValueError("training IDs must be unique")
        texts = [r["text"] for r in self.training_rows_]
        labels = np.asarray([r["label"] for r in self.training_rows_])
        vectorizer = _text_vectorizer()
        with threadpool_limits(limits=2):
            text_matrix = vectorizer.fit_transform(texts)
        oof = None
        fold_assignments = np.full(len(ids), -1, dtype=int)
        oof_traces, fold_meta = [], []
        retrieval_names = None
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
        for fold, (ref_ix, query_ix) in enumerate(splitter.split(np.zeros(len(ids)), labels)):
            refs = [self.training_rows_[i] for i in ref_ix]
            # Query payload is deliberately stripped to runtime fields and blank label.
            queries = [{"id": self.training_rows_[i]["id"], "text": self.training_rows_[i]["text"],
                        "date": self.training_rows_[i]["date"], "label": ""} for i in query_ix]
            index = RetrievalIndex(refs, self.lexicon, views=self.views, k=5)
            if retrieval_names is None:
                retrieval_names = list(index.feature_names_)
            elif list(index.feature_names_) != retrieval_names:
                raise ValueError("inconsistent retrieval feature layout across folds")
            features, traces = index.transform(queries)
            if oof is None:
                oof = np.zeros((len(ids), features.shape[1]), dtype=float)
            oof[query_ix] = features
            fold_assignments[query_ix] = fold
            trace_by_id = {t["query_id"]: t for t in traces}
            for i in query_ix:
                raw = trace_by_id[self.training_rows_[i]["id"]]
                oof_traces.append({"query_id": raw["query_id"], "fold_index": fold,
                                   "neighbors": {k: v for k, v in raw["views"].items() if k != "class_neighbors"},
                                   **({"class_neighbors": raw["views"]["class_neighbors"]} if "class_neighbors" in raw["views"] else {})})
            fold_meta.append({"fold_index": fold, "query_ids": [ids[i] for i in query_ix],
                              "reference_ids": [ids[i] for i in ref_ix]})
        if np.any(fold_assignments < 0) or not np.isfinite(oof).all():
            raise ValueError("OOF feature construction incomplete or non-finite")
        if Counter(t["query_id"] for t in oof_traces) != Counter(ids):
            raise ValueError("OOF trace coverage does not match training rows")
        self.scaler_ = StandardScaler()
        scaled_oof = self.scaler_.fit_transform(oof)
        self.model_ = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000,
                                         random_state=self.seed)
        with threadpool_limits(limits=2):
            self.model_.fit(hstack([text_matrix, scaled_oof], format="csr"), labels)
        self.vectorizer_ = vectorizer
        self.oof_traces_ = oof_traces
        self.oof_folds_ = fold_meta
        self.feature_names_ = ["text:" + str(n) for n in vectorizer.get_feature_names_out()]
        self.feature_names_ += ["retrieval:" + n for n in retrieval_names]
        self.full_index_ = RetrievalIndex(self.training_rows_, self.lexicon, views=self.views, k=5)
        self.diagnostics_ = {"rows": len(ids), "label_counts": dict(counts), "folds": fold_meta,
                             "feature_count": len(self.feature_names_), "seed": self.seed,
                             "variant": self.variant, "probabilities": "uncalibrated logistic regression"}
        return self

    def predict_rows(self, pending_rows):
        if not pending_rows:
            return [], []
        queries = [{"id": r["id"], "text": r["text"], "date": r["date"], "label": ""} for r in pending_rows]
        with threadpool_limits(limits=2):
            raw_features, retrieval_traces = self.full_index_.transform(queries)
            matrix = hstack([self.vectorizer_.transform([r["text"] for r in pending_rows]),
                             self.scaler_.transform(raw_features)], format="csr")
            distributions = self.model_.predict_proba(matrix)
        traces, predictions = [], []
        for row, dist, retr in zip(pending_rows, distributions, retrieval_traces):
            probs = {label: 0.0 for label in LABELS}
            probs.update({str(k): float(v) for k, v in zip(self.model_.classes_, dist)})
            winner = max(LABELS, key=lambda l: probs[l])
            confidence = probs[winner]
            label = winner if winner == "unsure" or confidence >= 0.5 else "unsure"
            reason = "model_unsure" if winner == "unsure" else ("decided" if label != "unsure" else "low_confidence")
            predictions.append({"id": row["id"], "label": label,
                                "confidence": confidence if label != "unsure" else "",
                                "decision_reason": reason, **{"p_" + k: probs[k] for k in LABELS}})
            traces.append({"query_id": row["id"], "route": "retrieval_classifier",
                           "views": {k: v for k, v in retr["views"].items() if k != "class_neighbors"},
                           **({"class_neighbors": retr["views"]["class_neighbors"]} if "class_neighbors" in retr["views"] else {}),
                           "probability_inputs": {"class_order": list(LABELS), "probabilities": probs,
                                                  "calibrated": False, "decision_threshold": 0.5,
                                                  "neighbor_features": dict(zip(self.full_index_.feature_names_, raw_features[len(traces)].tolist()))}})
        return predictions, traces

    def diagnostics(self):
        return dict(self.diagnostics_)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--lexicon", required=True, type=Path)
    parser.add_argument("--variant", choices=("raw", "raw_masked"), default="raw")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        rows = read_runtime_rows(args.input)
        train = [r for r in rows if r["label"]]
        pending = [r for r in rows if not r["label"]]
        if not train:
            raise ValueError("input contains no labelled training rows")
        overlap = {r["id"] for r in train} & {r["id"] for r in pending}
        if overlap:
            raise ValueError("labelled and pending IDs overlap")
        lexicon = json.loads(args.lexicon.read_text(encoding="utf-8"))
        start = time.perf_counter()
        classifier = RetrievalClassifier(lexicon, args.variant, args.seed).fit(train)
        fit_end = time.perf_counter()
        predictions, traces = classifier.predict_rows(pending)
        infer_end = time.perf_counter()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.artifacts.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["id", "label", "confidence", "decision_reason", "p_true", "p_false", "p_unsure"])
            writer.writeheader(); writer.writerows(predictions)
        (args.artifacts / "traces.jsonl").write_text("".join(json.dumps(_safe_json(t), allow_nan=False) + "\n" for t in traces), encoding="utf-8")
        (args.artifacts / "oof_traces.jsonl").write_text("".join(json.dumps(_safe_json(t), allow_nan=False) + "\n" for t in classifier.oof_traces_), encoding="utf-8")
        (args.artifacts / "oof_folds.json").write_text(json.dumps(_safe_json(classifier.oof_folds_), indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (args.artifacts / "feature_names.json").write_text(json.dumps(classifier.feature_names_, indent=2) + "\n", encoding="utf-8")
        diagnostics = classifier.diagnostics()
        serial_end = time.perf_counter()
        diagnostics["timings_seconds"] = {"fit": fit_end-start, "inference": infer_end-fit_end, "serialization": serial_end-infer_end}
        diagnostics["retrieval_index"] = classifier.full_index_.summary()
        (args.artifacts / "diagnostics.json").write_text(json.dumps(_safe_json(diagnostics), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
