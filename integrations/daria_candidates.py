#!/usr/bin/env python3
"""Fit declared candidate classifiers on a sanitized feature cache, without gold test data."""
import argparse
import json
import sys
import time
import traceback
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.daria_sklearn import CONFIGS, make_candidate, select_features
from evaluation.run_text_experiments import write_csv, write_json
from threadpoolctl import threadpool_limits


def probabilities_to_rows(ids, probabilities, classes):
    rows = []
    for rid, ps in zip(ids, probabilities):
        probs = {str(k): float(v) for k, v in zip(classes, ps)}
        label = max(probs, key=probs.get)
        confidence = probs[label] if label != 'unsure' else None
        reason = 'model_unsure' if label == 'unsure' else 'decided'
        if label != 'unsure' and confidence < .5:
            label, confidence, reason = 'unsure', None, 'low_confidence'
        rows.append({'id': rid, 'label': label, 'confidence': confidence,
                     'decision_reason': reason, **{f'p_{k}': probs.get(k, 0.0)
                                                   for k in ('true', 'false', 'unsure')}})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--candidate', choices=['native'] + [c['id'] for c in CONFIGS],
                    help='Fit only this frozen candidate (default: full declared sweep)')
    args = ap.parse_args()
    cache = json.loads(Path(args.cache).read_text())
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ids = [r['id'] for r in cache['pending']]
    pending = [r['features'] for r in cache['pending']]
    fields = ['id', 'label', 'confidence', 'decision_reason', 'p_true', 'p_false', 'p_unsure']
    native = [r['native']['probs'] for r in cache['pending']]
    write_csv(out / 'native.csv', probabilities_to_rows(ids,
              [[p[k] for k in ('true', 'false', 'unsure')] for p in native],
              ('true', 'false', 'unsure')), fields)
    timing = {'native': {'status': 'ok', 'fit_seconds': 0., 'inference_seconds': 0.}}
    with threadpool_limits(limits=2):
        for config in CONFIGS:
            name = config['id']
            if args.candidate and args.candidate != name:
                continue
            started = time.perf_counter()
            try:
                model = make_candidate(config, seed=42)
                with warnings.catch_warnings(record=True) as recorded:
                    warnings.simplefilter('always')
                    model.fit(select_features(cache['train_features'], config['features']),
                              cache['train_labels'])
                    fit_seconds = time.perf_counter() - started
                    start_predict = time.perf_counter()
                    probs = model.predict_proba(select_features(pending, config['features']))
                predictions = probabilities_to_rows(ids, probs, model.classes_)
                inference_seconds = time.perf_counter() - start_predict
                start_write = time.perf_counter()
                write_csv(out / f'{name}.csv', predictions, fields)
                timing[name] = {'status': 'ok', 'fit_seconds': fit_seconds,
                    'inference_seconds': inference_seconds,
                    'serialization_seconds': time.perf_counter() - start_write,
                    'warnings': sorted(set(str(w.message) for w in recorded))}
            except Exception as exc:
                timing[name] = {'status': 'failed', 'reason': str(exc),
                                'traceback': traceback.format_exc()}
            write_json(out / 'timings.json', timing)
            print(f'{name}: {timing[name]["status"]}', flush=True)


if __name__ == '__main__':
    main()
