#!/usr/bin/env python3
"""Build Daria feature caches with fold-safe, order-invariant rich summaries.

This is a thin wrapper around ``integrations.daria_features``. It temporarily
patches only the feature-class factory; input validation, split sanitation,
calibration and output auditing remain in the existing builder.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from collections import Counter
from datetime import date
from statistics import fmean, pstdev
from urllib.parse import quote
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _token(value) -> str:
    """Encode a categorical value for a DictVectorizer feature name."""
    return quote(str(value), safe="") or "empty"


def _date_precision(value) -> str:
    if not isinstance(value, str):
        return "other"
    if len(value) == 4 and value.isdigit():
        return "year"
    if len(value) == 7 and value[4] == "-":
        return "month"
    if len(value) == 10 and value[4] == "-" and value[7] == "-":
        return "day"
    return "other"


def build_true_anchor_index(rows):
    """Index only labelled-true supplied atoms; pending parses never become anchors."""
    anchors = {}
    for row in rows:
        if row.label != "true" or not row.atoms:
            continue
        for atom in row.atoms:
            if "op" in atom.value:
                continue
            anchors.setdefault(atom.key, []).append((row.date, row.page, atom))
    for values in anchors.values():
        values.sort(key=lambda entry: (entry[0], entry[1]))
    return anchors


def summarize_rich_features(row, shape_atoms, anchors, atom_matcher, *, training=False, names=None):
    """Return only ``rich__`` features; no identifiers, text, or inferred truth are added.

    ``shape_atoms`` is the atom trace already produced by the temporal window model.
    Anchor gaps describe observation dates only; they do not imply persistence.
    """
    out = {}
    row_atoms = list(row.atoms or [])
    traces = list(shape_atoms or [])
    # The temporal shape emits one trace per atom in the same order. Defensive key
    # alignment keeps summaries stable if a caller supplies a reordered trace list.
    trace_by_key = {}
    for trace in traces:
        key = (trace.get("subject"), trace.get("attribute"))
        trace_by_key.setdefault(key, []).append(trace)
    aligned = []
    for atom in row_atoms:
        candidates = trace_by_key.get(atom.key, [])
        aligned.append(candidates.pop(0) if candidates else {})

    out["rich__summary_atom_count"] = float(len(row_atoms))
    subjects = {str(atom.subject) for atom in row_atoms}
    attributes = {str(atom.attribute) for atom in row_atoms}
    out["rich__summary_unique_subject_count"] = float(len(subjects))
    out["rich__summary_unique_attribute_count"] = float(len(attributes))
    for subject in subjects:
        out[f"rich__content_subject_{_token(subject)}"] = 1.0
    for attribute in attributes:
        out[f"rich__content_attribute_{_token(attribute)}"] = 1.0

    def histogram(prefix, values):
        counts = Counter(str(value) for value in values if value is not None)
        for value, count in counts.items():
            out[f"rich__summary_{prefix}_{_token(value)}"] = float(count)

    histogram("atom_kind", (atom.value.get("kind", "unknown") for atom in row_atoms))
    histogram("atom_op", (atom.value.get("op", "eq") for atom in row_atoms))
    histogram("atom_unit", (atom.value.get("unit") for atom in row_atoms))
    histogram("atom_period", (atom.value.get("per") for atom in row_atoms))
    histogram("date_precision", (_date_precision(atom.value.get("value"))
                                   for atom in row_atoms if atom.value.get("kind") == "date"))

    for field in ("state", "stability", "direction", "trend", "shape"):
        histogram(f"shape_{field}", (trace.get(field) for trace in aligned))

    def numeric_summary(prefix, values):
        values = [float(value) for value in values if value is not None]
        out[f"rich__summary_{prefix}_count"] = float(len(values))
        out[f"rich__summary_{prefix}_available"] = float(bool(values))
        if values:
            out[f"rich__summary_{prefix}_min"] = min(values)
            out[f"rich__summary_{prefix}_mean"] = fmean(values)
            out[f"rich__summary_{prefix}_max"] = max(values)
            out[f"rich__summary_{prefix}_std"] = pstdev(values)
        else:
            for suffix in ("min", "mean", "max", "std"):
                out[f"rich__summary_{prefix}_{suffix}"] = 0.0

    numeric_summary("atom_p_true", (trace.get("p_true") for trace in aligned))
    parse_q = [float(value) for value in (row.meta.get("atom_q") or {}).values()]
    out["rich__summary_parse_q_count"] = float(len(parse_q))
    out["rich__summary_parse_q_available"] = float(bool(parse_q))
    out["rich__summary_parse_q_min"] = min(parse_q) if parse_q else 0.0
    out["rich__summary_parse_q_mean"] = fmean(parse_q) if parse_q else 0.0

    query_day = date.fromisoformat(row.date)
    support_counts, anchor_counts = [], []
    past_gaps, future_gaps = [], []
    for atom in row_atoms:
        eligible = [entry for entry in anchors.get(atom.key, ())
                    if not (training and entry[1] == row.page)]
        anchor_counts.append(len(eligible))
        support = 0
        for anchor_date, _page, anchor_atom in eligible:
            if atom_matcher(anchor_atom, atom, names):
                support += 1
            anchor_day = date.fromisoformat(anchor_date)
            delta = (anchor_day - query_day).days
            if delta <= 0:
                past_gaps.append(float(-delta))
            if delta >= 0:
                future_gaps.append(float(delta))
        support_counts.append(support)

    out["rich__support_anchor_observation_count"] = float(sum(anchor_counts))
    out["rich__support_atoms_with_anchor_count"] = float(sum(count > 0 for count in anchor_counts))
    out["rich__support_matching_anchor_count"] = float(sum(support_counts))
    out["rich__support_atoms_with_matching_anchor_count"] = float(sum(count > 0 for count in support_counts))
    for label, gaps in (("past", past_gaps), ("future", future_gaps)):
        out[f"rich__support_{label}_gap_available"] = float(bool(gaps))
        out[f"rich__support_nearest_{label}_gap_days"] = min(gaps) if gaps else -1.0
        out[f"rich__support_mean_{label}_gap_days"] = fmean(gaps) if gaps else -1.0
    return out


def _augment_feature_class(base_class):
    class RichFeatureHybrid(base_class):
        def fit(self, rows, lexicon, names):
            # The Pipeline has already parsed pending rows, but only supplied labelled-true
            # rows qualify as anchors; pending parses remain excluded by the label gate.
            self._rich_anchors = build_true_anchor_index(rows)
            self._rich_names = names
            from hooly.values import satisfies

            def supports(anchor_atom, query_atom, value_names):
                return (anchor_atom.key == query_atom.key
                        and satisfies(query_atom.value, anchor_atom.value, value_names) is True)

            self._rich_atom_match = supports
            return super().fit(rows, lexicon, names)

        def _make_features(self, row, hp, training=False):
            out = super()._make_features(row, hp, training=training)
            _base, shape_atoms = self.shape.row_probs(
                row, exclude_page=row.page if training else None)
            out.update(summarize_rich_features(
                row, shape_atoms, self._rich_anchors, self._rich_atom_match,
                training=training, names=self._rich_names))
            return out

    RichFeatureHybrid.__name__ = "RichFeatureHybrid"
    return RichFeatureHybrid


def main():
    # Discover the selected upstream source before the old builder imports hooly.
    if "-h" in sys.argv or "--help" in sys.argv:
        from integrations.daria_features import main as builder_main
        builder_main()
        return
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source-dir", required=True)
    args, _unknown = parser.parse_known_args()
    source = str(Path(args.source_dir).resolve())
    if source not in sys.path:
        sys.path.insert(0, source)

    module = importlib.import_module("experiments.daria_sklearn")
    original_factory = module.build_truth_class

    def rich_factory(config=None, seed=42):
        return _augment_feature_class(original_factory(config, seed))

    module.build_truth_class = rich_factory
    try:
        from integrations.daria_features import main as builder_main
        builder_main()
    finally:
        module.build_truth_class = original_factory


if __name__ == "__main__":
    main()
