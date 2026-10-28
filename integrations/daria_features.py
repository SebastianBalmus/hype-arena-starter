#!/usr/bin/env python3
"""Build split-scoped Daria feature caches from sanitized evaluator inputs."""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from integrations.daria_predict import read_csv, RUNTIME_FIELDS, FORM_FIELDS


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-dir', required=True)
    ap.add_argument('--input', required=True)
    ap.add_argument('--forms', required=True)
    ap.add_argument('--lexicon', required=True)
    ap.add_argument('--calibration-fit-ids', required=True,
                    help='JSON array of fit IDs, strictly within the current training pool')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    # Reject malformed or out-of-scope inputs before importing the upstream snapshot. This
    # keeps the boundary checks independently testable and prevents bad labels from silently
    # becoming training rows.
    raw = read_csv(args.input, RUNTIME_FIELDS)
    by_id = {r['id']: r for r in raw}
    if len(by_id) != len(raw) or '' in by_id:
        raise ValueError('Runtime IDs must be unique and nonempty')
    allowed_labels = {'', 'true', 'false', 'unsure'}
    for row in raw:
        if row['label'].strip().lower() not in allowed_labels:
            raise ValueError(f"Invalid runtime label for {row['id']!r}: {row['label']!r}")
    train_ids = {r['id'] for r in raw if r['label'].strip().lower()}
    forms = read_csv(args.forms, FORM_FIELDS)
    forms_by_id = {r['id']: r for r in forms}
    if len(forms_by_id) != len(forms) or set(forms_by_id) != train_ids:
        raise ValueError('Forms must cover exactly the labelled training pool')
    for rid, fr in forms_by_id.items():
        if any(fr[k] != by_id[rid][k] for k in RUNTIME_FIELDS):
            raise ValueError(f'Form fingerprint mismatch: {rid}')

    sys.path.insert(0, str(Path(args.source_dir).resolve()))
    from hooly.data import Atom, Row, assign_pages, build_names, load_lexicon
    from hooly.pipeline import Pipeline
    from hooly.reader import atom_match
    from hooly.calibrate import Isotonic
    from experiments.daria_sklearn import build_truth_class
    FeatureHybrid = build_truth_class()

    rows = [Row(r['id'], r['text'], r['date'], r['label'].strip().lower() or None,
                [Atom(a['subject'], a['attribute'], a['value'])
                 for a in json.loads(forms_by_id[r['id']]['form'])] if r['id'] in train_ids else None,
                index=i) for i, r in enumerate(raw)]
    assign_pages(rows)
    train = [r for r in rows if r.id in train_ids]
    cal_fit_ids = set(json.loads(Path(args.calibration_fit_ids).read_text()))
    if not cal_fit_ids or not cal_fit_ids < train_ids:
        raise ValueError('Calibration fit IDs must be a nonempty strict training subset')
    lexicon = load_lexicon(args.lexicon)
    names = build_names(lexicon, train)

    class SplitPipeline(Pipeline):
        def _parse_calibration(self, train_rows, lexicon, fitted_names):
            fit = [r for r in train if r.id in cal_fit_ids]
            held_ids = train_ids - cal_fit_ids
            fit_names = build_names(lexicon, fit)
            reader = self._reader().fit(fit, lexicon, fit_names)
            copies = [replace(r, label=None, atoms=None, meta={}) if r.id in held_ids
                      else replace(r, meta={}) for r in train]
            reader.parse_all(copies)
            gold = {r.id: r.atoms for r in train}
            scores, correct = [], []
            for row in copies:
                if row.id in held_ids:
                    for atom in row.atoms or []:
                        scores.append(row.meta.get('parse_scores', {}).get(atom.key, 0.0))
                        correct.append(int(any(atom_match(atom, a, fit_names)
                                               for a in gold[row.id] or [])))
            return Isotonic().fit(scores, correct)

    start = time.perf_counter()
    pipeline = SplitPipeline(FeatureHybrid, seed=42).fit(rows, lexicon, names)
    fit_seconds = time.perf_counter() - start
    pending = [r for r in rows if r.id not in train_ids]
    start = time.perf_counter()
    truth = pipeline.truth
    records = []
    for row in pending:
        p = truth.predict(row)
        records.append({'id': row.id, 'features': truth.feature_row(row),
                        'native': {'label': p.label, 'probs': p.probs},
                        'parsed_atoms': len(row.atoms or []),
                        'parse_q': list((row.meta.get('atom_q') or {}).values())})
    payload = {'train_features': truth.feature_train, 'train_labels': truth.target_train,
               'train_ids': truth.training_ids, 'pending': records,
               'timings': {'fit_seconds': fit_seconds,
                           'inference_seconds': time.perf_counter() - start},
               'audit': {'train_ids': sorted(train_ids), 'pending_ids': [r.id for r in pending],
                         'calibration_fit_ids': sorted(cal_fit_ids),
                         'forms_rows': len(forms), 'practice_file_read': False,
                         'packaged_assets_loaded': False,
                         'pending_labels_and_forms_absent': True}}
    Path(args.output).write_text(json.dumps(payload, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
