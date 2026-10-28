#!/usr/bin/env python3
"""Run Daria's native HybridStack on evaluator-supplied, split-sanitized CSVs.

The adapter deliberately never opens the practice corpus. Inputs are runtime
rows (id,text,date,label) and a separate train-only forms file.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path


RUNTIME_FIELDS = ["id", "text", "date", "label"]
FORM_FIELDS = ["id", "text", "date", "label", "form"]


def read_csv(path: str, fields: list[str]) -> list[dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != fields:
            raise ValueError(f"{path}: expected exact columns {fields}, got {reader.fieldnames}")
        rows = list(reader)
    if any(None in r for r in rows):
        raise ValueError(f"{path}: row has extra columns")
    if any(any(value is None for value in r.values()) for r in rows):
        raise ValueError(f"{path}: row has missing columns")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", required=True, help="Extracted Daria branch checkout")
    ap.add_argument("--input", required=True, help="Sanitized runtime CSV")
    ap.add_argument("--forms", required=True, help="Train-only CSV with form column")
    ap.add_argument("--split-json", required=True)
    ap.add_argument("--track", required=True, choices=("stratified", "temporal", "checker"))
    ap.add_argument("--model", choices=("hybrid", "gated"), default="gated")
    ap.add_argument("--lexicon", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--timings", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.source_dir).resolve()))
    from hooly.data import Atom, Row, assign_pages, build_names, load_lexicon
    from hooly import hmm
    HybridStack = hmm.HybridStack
    GatedHybrid = getattr(hmm, "GatedHybrid", HybridStack)
    from hooly.pipeline import Pipeline
    from hooly.reader import atom_match
    from hooly.calibrate import Isotonic

    manifest = json.loads(Path(args.split_json).read_text())
    split = manifest["splits"][args.track]
    train_ids = set(split["train_ids"])
    test_ids = set(split["test_ids"])
    fit_ids = set(split["fit_ids"])
    validation_ids = set(split["validation_ids"])
    if fit_ids & validation_ids or fit_ids | validation_ids != train_ids:
        raise ValueError("split fit/validation IDs must partition outer train IDs")

    raw = read_csv(args.input, RUNTIME_FIELDS)
    forms = read_csv(args.forms, FORM_FIELDS)
    by_id: dict[str, dict[str, str]] = {}
    for row in raw:
        if not row["id"] or row["id"] in by_id:
            raise ValueError("runtime input IDs must be nonempty and unique")
        by_id[row["id"]] = row
    if set(by_id) != train_ids | test_ids:
        raise ValueError("runtime input IDs must exactly equal this split's train and test IDs")
    for rid, row in by_id.items():
        label = row["label"].strip().lower()
        if rid in test_ids and label:
            raise ValueError(f"test row {rid} must have empty label")
        if rid in train_ids and label not in {"true", "false", "unsure"}:
            raise ValueError(f"training row {rid} must have a permitted label")

    form_by_id: dict[str, list[Atom]] = {}
    runtime_by_id = by_id
    for fr in forms:
        rid = fr["id"]
        if rid in form_by_id or rid not in train_ids or rid not in fit_ids | validation_ids:
            raise ValueError(f"form row {rid!r} is duplicate or outside outer training IDs")
        rr = runtime_by_id[rid]
        if (fr["text"], fr["date"], fr["label"].strip().lower()) != (
            rr["text"], rr["date"], rr["label"].strip().lower()
        ):
            raise ValueError(f"form/runtime fingerprint mismatch for {rid}")
        parsed = json.loads(fr["form"])
        form_by_id[rid] = [Atom(a["subject"], a["attribute"], a["value"]) for a in parsed]
    missing_forms = train_ids - set(form_by_id)
    if missing_forms:
        raise ValueError(f"missing training forms for {len(missing_forms)} rows")

    # Preserve the CSV's row order in both fit and pending batches.
    all_rows = []
    for i, rr in enumerate(raw):
        rid = rr["id"]
        atoms = form_by_id.get(rid)
        row = Row(rid, rr["text"], rr["date"], rr["label"].strip().lower() or None,
                  atoms, index=i)
        all_rows.append(row)
    assign_pages(all_rows)
    fit_rows = [r for r in all_rows if r.id in fit_ids]
    outer_train_rows = [r for r in all_rows if r.id in train_ids]
    gold_validation_atoms = {r.id: r.atoms for r in all_rows if r.id in validation_ids}
    # Inner calibration sees fit context and pending validation text in original
    # outer-training order. Gold validation atoms remain evaluator-side.
    calibration_rows = [replace(r, atoms=None, label=None, meta={}) if r.id in validation_ids
                        else replace(r, meta={}) for r in outer_train_rows]
    fit_names = build_names(load_lexicon(args.lexicon), fit_rows)
    outer_names = build_names(load_lexicon(args.lexicon), outer_train_rows)

    class SplitPipeline(Pipeline):
        def _parse_calibration(self, train_rows, lexicon, fitted_names):
            reader = self._reader().fit(fit_rows, lexicon, fit_names)
            copies = [replace(r, meta={}) for r in calibration_rows]
            reader.parse_all(copies)
            scores, correct = [], []
            gold_by_id = gold_validation_atoms
            for candidate in copies:
                if candidate.id not in validation_ids:
                    continue
                gold = gold_by_id[candidate.id] or []
                for atom in candidate.atoms or []:
                    scores.append(candidate.meta.get("parse_scores", {}).get(atom.key, 0.0))
                    correct.append(int(any(atom_match(atom, target, fit_names) for target in gold)))
            return Isotonic().fit(scores, correct)

    start = time.monotonic()
    lexicon = load_lexicon(args.lexicon)
    truth_model = GatedHybrid if args.model == "gated" else HybridStack
    model = SplitPipeline(truth_model, seed=args.seed).fit(all_rows, lexicon, outer_names)
    fit_seconds = time.monotonic() - start
    output = []
    traces_path = Path(args.output).with_name("traces.jsonl")
    trace_tmp = traces_path.with_suffix(traces_path.suffix + ".tmp")
    predict_start = time.monotonic()
    with trace_tmp.open("w", encoding="utf-8") as tf:
        for row in all_rows:
            if row.id not in test_ids:
                continue
            pred = model.predict(row)
            label = pred.label
            confidence = pred.probs.get(label, 0.0) if label in ("true", "false") else None
            if label in ("true", "false") and confidence < 0.5:
                label, confidence = "unsure", None
            probs = pred.probs
            reason = ('decided' if label != 'unsure' else
                      'model_unsure' if pred.label == 'unsure' else 'low_confidence')
            output.append({"id": row.id, "label": label, "confidence": confidence,
                           "decision_reason": reason, "p_true": probs.get("true"),
                           "p_false": probs.get("false"), "p_unsure": probs.get("unsure")})
            tf.write(json.dumps({"id": row.id, "probs": probs, "trace": pred.trace,
                                 "decision_reason": reason},
                                ensure_ascii=False, allow_nan=False) + "\n")
    traces_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(trace_tmp, traces_path)
    predict_seconds = time.monotonic() - predict_start
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    serialization_start = time.monotonic()
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label", "confidence", "decision_reason",
                                               "p_true", "p_false", "p_unsure"])
        writer.writeheader()
        writer.writerows(output)
    os.replace(tmp, out_path)
    audit = {"track": args.track, "model": args.model, "seed": args.seed, "train_rows": len(train_ids),
             "fit_rows": len(fit_ids), "validation_rows": len(validation_ids),
             "test_rows": len(test_ids), "forms_rows": len(form_by_id),
             "runtime_columns": RUNTIME_FIELDS, "forms_columns": FORM_FIELDS,
             "practice_file_read": False, "calibration_asset_loaded": False,
             "final_fit_uses_all_train_ids": True,
             "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
             "serialization_seconds": time.monotonic() - serialization_start}
    Path(args.timings).parent.mkdir(parents=True, exist_ok=True)
    Path(args.timings).write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
