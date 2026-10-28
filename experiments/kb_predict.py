#!/usr/bin/env python3
"""Predict from split-permitted forms and a lexicon, with auditable evidence.

This experimental CLI never opens practice.csv. Its caller supplies a runtime
CSV and a separate forms CSV containing only the labelled rows in that runtime
input. A form is accepted only after matching ID, text, date and label.
"""

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from knowledge_base.parser import AssertionParser
from knowledge_base.store import EvidenceStore

LABELS = {'true', 'false', 'unsure'}


def read_csv(path, *, forms=False):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        expected = {'id', 'text', 'date', 'label'} | ({'form'} if forms else set())
        if set(reader.fieldnames or []) != expected or len(reader.fieldnames or []) != len(expected):
            raise ValueError(f'{path}: expected exactly {sorted(expected)}')
        rows = list(reader)
    seen = set()
    for row in rows:
        if not row['id'] or row['id'] in seen:
            raise ValueError(f'{path}: empty or duplicate ID {row["id"]!r}')
        seen.add(row['id'])
        if row['label'] not in LABELS | {''}:
            raise ValueError(f'{path}: invalid label on {row["id"]}')
        if date.fromisoformat(row['date']).isoformat() != row['date']:
            raise ValueError(f'{path}: expected YYYY-MM-DD date on {row["id"]}')
        if not row['text']:
            raise ValueError(f'{path}: empty text on {row["id"]}')
    return rows


def match_training_forms(runtime_rows, form_rows):
    """Prevent pending rows and reused IDs from retrieving packaged answers."""
    labelled = {row['id']: row for row in runtime_rows if row['label']}
    if {row['id'] for row in form_rows} != set(labelled):
        raise ValueError('Training forms must match exactly the labelled runtime IDs')
    for row in form_rows:
        source = labelled[row['id']]
        if any(row[key] != source[key] for key in ('id', 'text', 'date', 'label')):
            raise ValueError(f'Training form fingerprint mismatch: {row["id"]}')
        form = json.loads(row['form']) if isinstance(row['form'], str) else row['form']
        if not isinstance(form, list) or not form:
            raise ValueError(f'Invalid training form: {row["id"]}')
    return form_rows


def load_calibration(path):
    if path is None:
        return None
    calibration = json.loads(Path(path).read_text())
    for label in ('true', 'false'):
        bucket = calibration[label]
        confidence = bucket['confidence']
        if (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
                or not math.isfinite(confidence) or not 0 <= confidence <= 1):
            raise ValueError(f'Invalid calibration for {label}')
    return calibration


def fingerprint(row):
    return hashlib.sha256('\0'.join(row[k] for k in ('id', 'text', 'date')).encode()).hexdigest()


class KnowledgeBasePredictor:
    def __init__(self, training_rows, lexicon, *, calibration=None, mode='kb_only'):
        if mode not in ('kb_only', 'kb_hybrid'):
            raise ValueError(f'Unknown mode: {mode}')
        self.mode, self.calibration = mode, calibration
        self.parser = AssertionParser(training_rows, lexicon)
        self.store = EvidenceStore(training_rows, lexicon)
        self.fallback = None
        self.records = {r['id']: r for r in self.store.export_records()}
        if mode == 'kb_hybrid':
            from experiments.text_variants import make_model
            from threadpoolctl import threadpool_limits
            self.fallback = make_model('word_char_lr_balanced', 42)
            with threadpool_limits(limits=2):
                self.fallback.fit([r['text'] for r in training_rows],
                                  [r['label'] for r in training_rows])

    def predict(self, pending):
        predictions, traces, unresolved = [], [], []
        for row in pending:
            parsed = self.parser.parse(row['text'])
            queries = [self.store.query(atom, row['date']) for atom in parsed['atoms']]
            raw_label, reason = 'unsure', parsed['reason']
            if parsed['complete'] and queries:
                labels = [q['label'] for q in queries]
                if 'false' in labels:
                    raw_label, reason = 'false', 'decided'
                elif all(label == 'true' for label in labels):
                    raw_label, reason = 'true', 'decided'
                else:
                    reason = ('temporal_conflict' if any(q['reason'] == 'temporal_conflict'
                                                       for q in queries) else 'missing_evidence')
            confidence = None
            confidence_inputs = {'calibration': 'not_applied'}
            label = raw_label
            if label != 'unsure':
                bucket = self.calibration[label] if self.calibration else None
                confidence = bucket['confidence'] if bucket else 0.5
                confidence_inputs = (dict(bucket) if bucket else
                                     {'confidence': 0.5, 'status': 'uncalibrated_raw_validation_run'})
                if confidence < 0.5:
                    label, confidence, reason = 'unsure', None, 'low_confidence'
            if reason == 'parsed':
                reason = 'missing_evidence'
            prediction = {'id': row['id'], 'label': label, 'confidence': confidence,
                          'decision_reason': reason}
            evidence = {e['id']: e for query in queries for e in query['evidence']}
            premises = [{'source_id': rid, 'label': self.records[rid]['label'],
                         'atoms': self.records[rid]['atoms']}
                        for rid in evidence]
            trace = {
                'output_id': row['id'], 'output_kind': 'label',
                'input_fingerprint': fingerprint(row),
                'queried_date_or_temporal_condition': row['date'],
                'parsed_assertions': parsed['atoms'], 'parse': parsed,
                'source_row_ids_and_fingerprints': list(evidence.values()),
                'source_dates': sorted({e['date'] for e in evidence.values()}),
                'normalized_premises': premises,
                'inference_rule_ids': ['same_date_numeric_intervals_v1'] if queries else [],
                'intermediate_results': queries, 'assumptions': [],
                'conflicts': [q for q in queries if q['reason'] == 'temporal_conflict'],
                'confidence_inputs': confidence_inputs,
                'raw_kb_label': raw_label, 'kb_label': label, 'kb_reason': reason,
                'route': 'kb' if label != 'unsure' else 'abstain',
                'emitted_label': label, 'proof_status': 'not_verified',
            }
            predictions.append(prediction)
            traces.append(trace)
            if label == 'unsure':
                unresolved.append(len(predictions) - 1)
        if self.fallback is not None and unresolved:
            import baseline
            from threadpoolctl import threadpool_limits
            with threadpool_limits(limits=2):
                fallbacks = baseline.predict_rows(self.fallback, [pending[i] for i in unresolved])
            for index, fallback in zip(unresolved, fallbacks):
                reason = ('decided' if fallback['decision_reason'] == 'model_label'
                          else fallback['decision_reason'])
                predictions[index].update(label=fallback['label'],
                                          confidence=fallback['confidence'] or None,
                                          decision_reason=reason)
                traces[index].update(route='text_fallback', emitted_label=fallback['label'],
                                     fallback={'model': 'word_char_lr_balanced',
                                               'label': fallback['label'],
                                               'confidence': fallback['confidence'] or None,
                                               'confidence_status': 'uncalibrated',
                                               'decision_reason': reason})
        return predictions, traces


def write_jsonl(path, records):
    with path.open('w', encoding='utf-8') as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--training-forms', type=Path, required=True)
    parser.add_argument('--lexicon', type=Path, required=True)
    parser.add_argument('--mode', choices=('kb_only', 'kb_hybrid'), default='kb_only')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--calibration', type=Path)
    args = parser.parse_args()
    try:
        rows = read_csv(args.input)
        training = match_training_forms(rows, read_csv(args.training_forms, forms=True))
        pending = [r for r in rows if not r['label']]
        if not training or not pending:
            raise ValueError('Expected labelled training rows and blank-label pending rows')
        lexicon = json.loads(args.lexicon.read_text())
        calibration = load_calibration(args.calibration)
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))
    fit_start = time.perf_counter()
    model = KnowledgeBasePredictor(training, lexicon, calibration=calibration, mode=args.mode)
    fit_seconds = time.perf_counter() - fit_start
    inference_start = time.perf_counter()
    predictions, traces = model.predict(pending)
    inference_seconds = time.perf_counter() - inference_start
    serialization_start = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.artifacts.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['id', 'label', 'confidence', 'decision_reason'])
        writer.writeheader()
        writer.writerows(predictions)
    write_jsonl(args.artifacts / 'traces.jsonl', traces)
    write_jsonl(args.artifacts / 'kb_records.jsonl', model.store.export_records())
    diagnostics = {
        'mode': args.mode, 'parser': model.parser.summary(), 'store': model.store.summary(),
        'pending_rows': len(pending), 'parsed_complete': sum(t['parse']['complete'] for t in traces),
        'raw_kb_decisions': sum(t['raw_kb_label'] != 'unsure' for t in traces),
        'emitted_kb_decisions': sum(t['route'] == 'kb' for t in traces),
        'routes': dict(Counter(t['route'] for t in traces)),
        'kb_reasons': dict(Counter(t['kb_reason'] for t in traces)),
        'fit_seconds': fit_seconds, 'inference_seconds': inference_seconds,
        'serialization_seconds': time.perf_counter() - serialization_start,
        'confidence_status': ('uncalibrated_raw_validation_run' if not calibration else
                              'prior_only' if all(calibration[k]['count'] == 0
                                                  for k in ('true', 'false')) else
                              'validation_estimate_with_prior_fallback'),
        'proof_status': 'not_verified', 'generation_status': 'not_implemented',
    }
    (args.artifacts / 'diagnostics.json').write_text(json.dumps(diagnostics, indent=2) + '\n')


if __name__ == '__main__':
    main()
