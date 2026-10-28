#!/usr/bin/env python3
"""Fit declared feature variants on sanitized Daria caches; never read outer gold."""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.daria_feature_variants import CONFIGS, make_feature_model
from integrations.daria_candidates import probabilities_to_rows
from evaluation.run_text_experiments import write_csv, write_json
from evaluation.scoring import score_rows

FIELDS = ['id', 'label', 'confidence', 'decision_reason', 'p_true', 'p_false', 'p_unsure']


def fit_predict(config, cache, params):
    model = make_feature_model(config, params, seed=42)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        start = time.perf_counter()
        model.fit(cache['train_features'], cache['train_labels'])
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        ids = [r['id'] for r in cache['pending']]
        probs = model.predict_proba([r['features'] for r in cache['pending']])
        result = probabilities_to_rows(ids, probs, model.classes_)
        infer_seconds = time.perf_counter() - start
        width = model.named_steps['features'].transform(cache['train_features'][:1]).shape[1]
    return result, {'status': 'ok', 'fit_seconds': fit_seconds, 'inference_seconds': infer_seconds,
                    'feature_count': int(width), 'warnings': sorted(set(str(w.message) for w in recorded))}


def audit_cache(cache, train_ids=None, pending_ids=None):
    ids, labels = cache['train_ids'], cache['train_labels']
    pending = [r['id'] for r in cache['pending']]
    if len(ids) != len(set(ids)) or len(ids) != len(labels) or len(ids) != len(cache['train_features']):
        raise ValueError('Training IDs, labels and features must align uniquely')
    if len(pending) != len(set(pending)) or set(ids) & set(pending):
        raise ValueError('Pending IDs must be unique and disjoint from training')
    if set(ids) != set(cache['audit']['train_ids']) or set(pending) != set(cache['audit']['pending_ids']):
        raise ValueError('Cache audit IDs do not match feature contents')
    if not cache['audit']['pending_labels_and_forms_absent']:
        raise ValueError('Pending gold boundary failed')
    if train_ids is not None and set(ids) != set(train_ids):
        raise ValueError('Wrong training pool')
    if pending_ids is not None and set(pending) != set(pending_ids):
        raise ValueError('Wrong pending pool')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', required=True, choices=['cv', 'predict'])
    ap.add_argument('--spec', help='CV spec, containing tuning-pool labels only')
    ap.add_argument('--cache', help='Sanitized rich cache for prediction')
    ap.add_argument('--params', required=True)
    ap.add_argument('--output-dir', required=True)
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    params = json.loads(Path(args.params).read_text())
    start_total = time.perf_counter()
    with threadpool_limits(limits=2):
        if args.mode == 'cv':
            if not args.spec:
                ap.error('--spec required for cv')
            spec = json.loads(Path(args.spec).read_text())
            gold_labels = spec['fit_labels']
            folds = []
            for fold in spec['folds']:
                cache = json.loads(Path(fold['path']).read_text())
                audit_cache(cache, fold['train_ids'], fold['test_ids'])
                if dict(zip(cache['train_ids'], cache['train_labels'])) != {rid: gold_labels[rid] for rid in cache['train_ids']}:
                    raise ValueError('CV training label mismatch')
                folds.append((cache, [{'id': rid, 'label': gold_labels[rid]} for rid in fold['test_ids']]))
            results = {}
            for config in CONFIGS:
                name = config['id']
                try:
                    metrics = []
                    for k, (cache, gold) in enumerate(folds):
                        pred, timing = fit_predict(config, cache, params)
                        c, errors, _ = score_rows(gold, pred)
                        if not errors['eligible_for_comparison']:
                            raise ValueError(errors)
                        write_csv(out / f'{name}-fold{k}.csv', pred, FIELDS)
                        metrics.append({'classification': c, 'timings': timing})
                    results[name] = {'status': 'ok', 'folds': metrics,
                        'mean_macro_f1': float(np.mean([m['classification']['macro_f1'] for m in metrics])),
                        'std_macro_f1': float(np.std([m['classification']['macro_f1'] for m in metrics]))}
                except Exception as exc:
                    results[name] = {'status': 'failed', 'reason': str(exc), 'traceback': traceback.format_exc()}
                write_json(out / 'cv-results-progress.json', results)
                print(f'CV {name}: {results[name]["status"]} {results[name].get("mean_macro_f1", "")}', flush=True)
            viable = [c['id'] for c in CONFIGS if results[c['id']]['status'] == 'ok']
            if not viable or 'expanded_none' not in viable:
                raise RuntimeError('No viable candidate or baseline failed')
            chosen = max(viable, key=lambda name: results[name]['mean_macro_f1'])
            write_json(out / 'selection.json', {'selected': chosen, 'results': results,
                'selection': 'mean CV macro F1 after fixed050; ties by declared config order',
                'fixed_gb_params': params, 'source_pool': 'shared fit_ids only',
                'seconds': time.perf_counter() - start_total})
        else:
            if not args.cache:
                ap.error('--cache required for predict')
            cache = json.loads(Path(args.cache).read_text())
            audit_cache(cache)
            timings = {}
            for config in CONFIGS:
                name = config['id']
                try:
                    predictions, timing = fit_predict(config, cache, params)
                    start = time.perf_counter()
                    write_csv(out / f'{name}.csv', predictions, FIELDS)
                    timing['serialization_seconds'] = time.perf_counter() - start
                    timings[name] = timing
                except Exception as exc:
                    timings[name] = {'status': 'failed', 'reason': str(exc), 'traceback': traceback.format_exc()}
                write_json(out / 'timings.json', timings)
                print(f'predict {name}: {timings[name]["status"]}', flush=True)


if __name__ == '__main__':
    main()
