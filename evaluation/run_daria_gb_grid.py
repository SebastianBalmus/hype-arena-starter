#!/usr/bin/env python3
"""Tune GB using fold-rebuilt Daria features and evaluate frozen arena tracks."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, ParameterGrid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.run_text_experiments import (HASHES, TRACKS, digest, read_csv,
    write_csv, write_json, sanitized_rows, numeric_predictions, source_version)
from evaluation.scoring import score_rows
from experiments.daria_gb_grid import PARAM_GRID, BASE_PARAMS

METHOD = 'daria_gb_grid'
PREVIOUS = 'daria-sklearn-20261028-001'
FIELDS = ['id', 'label', 'confidence', 'decision_reason', 'p_true', 'p_false', 'p_unsure']
SCORED_FIELDS = ['id', 'gold', 'label', 'confidence', 'decision_reason', 'correct', 'output_valid', 'error_codes']
SOURCES = ('experiments/daria_gb_grid.py', 'evaluation/run_daria_gb_grid.py',
    'experiments/daria_sklearn.py', 'integrations/daria_features.py',
    'integrations/daria_candidates.py', 'integrations/daria_predict.py',
    'evaluation/run_text_experiments.py', 'evaluation/scoring.py', 'EVALUATION.md', 'requirements.txt')


def make_folds(rows, track):
    if track != 'temporal':
        return list(StratifiedKFold(n_splits=3, shuffle=True, random_state=42).split(
            np.arange(len(rows)), [r['label'] for r in rows]))
    dates = sorted({r['date'] for r in rows})
    result = []
    for train, test in TimeSeriesSplit(n_splits=3).split(dates):
        train_dates, test_dates = {dates[i] for i in train}, {dates[i] for i in test}
        assert max(train_dates) < min(test_dates)
        result.append((np.asarray([i for i, r in enumerate(rows) if r['date'] in train_dates]),
                       np.asarray([i for i, r in enumerate(rows) if r['date'] in test_dates])))
    return result


def score_file(gold, path):
    c, errors, scored = score_rows(gold, numeric_predictions(read_csv(path)))
    if not errors['eligible_for_comparison']:
        raise ValueError(f'Output audit failed: {path}: {errors}')
    return c, errors, scored


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-id', required=True)
    args = ap.parse_args()
    if not args.run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in args.run_id):
        ap.error('Invalid run ID')
    for file, expected in HASHES.items():
        if digest(ROOT / file) != expected:
            ap.error(f'Benchmark hash mismatch: {file}')
    previous = ROOT / 'results/comparisons' / PREVIOUS
    old_run = json.loads((previous / 'run.json').read_text())
    assert old_run['pinned_sha256'] == HASHES
    for file, expected in old_run['source_file_sha256'].items():
        if digest(ROOT / file) != expected:
            ap.error(f'Previous feature-cache method changed: {file}')
    out = ROOT / 'results/evaluations' / METHOD / args.run_id
    if out.exists():
        ap.error('Run exists; choose a new ID')
    out.mkdir(parents=True)
    rows = read_csv(ROOT / 'data/practice.csv')
    manifest = json.loads((ROOT / 'evaluation/splits-v1.json').read_text())
    env = os.environ.copy()
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        env[name] = '1'
    source = previous / 'source'
    assert all(digest(source / f) == h for f, h in old_run['upstream_source_sha256'].items())
    method_inputs = out / 'method-inputs'
    method_inputs.mkdir()
    run = {'benchmark_version': 'arena-eval-v1', 'method_id': METHOD, 'run_id': args.run_id,
        'pinned_sha256': HASHES, 'seed': 42, 'internal_native_fold_seed': 1,
        'source_commit': source_version(), 'source_file_sha256': {f: digest(ROOT / f) for f in SOURCES},
        'upstream_commit': old_run['upstream_commit'], 'upstream_source_sha256': old_run['upstream_source_sha256'],
        'dependencies': {p: importlib.metadata.version(p) for p in ('scikit-learn', 'numpy', 'scipy', 'joblib', 'threadpoolctl')},
        'python': platform.python_version(), 'splits': list(TRACKS), 'policies': ['fixed_050'],
        'access_profile': 'batch_unlabelled_v1', 'param_grid': PARAM_GRID,
        'parameter_combinations': len(ParameterGrid(PARAM_GRID)), 'cv_folds': 3,
        'cv_scope': 'Only shared fit_ids; full reader/calibration/timeline/HMM rebuilt for each CV fold. Shared validation is a single diagnostic after best CV params frozen.',
        'cv_policy': '3-fold stratified shuffled seed42 for stratified/checker; 3 expanding whole-date forward folds for temporal. These are inner training folds, not a new outer benchmark track.',
        'training_ids': {t: manifest['splits'][t]['train_ids'] for t in TRACKS},
        'fit_ids': {t: manifest['splits'][t]['fit_ids'] for t in TRACKS},
        'validation_ids': {t: manifest['splits'][t]['validation_ids'] for t in TRACKS},
        'test_ids': {t: manifest['splits'][t]['test_ids'] for t in TRACKS},
        'training_sources': 'Only each fold’s train labels/forms plus public lexicon, with its validation text unlabelled. Final fit uses full permitted outer train.',
        'factual_asset_provenance': 'Fresh full-pipeline CV features per fold. Inner/full outer caches reused only after exact row-pool, benchmark, source and cache hashes match earlier split-scoped run.',
        'leakage_audit': {'status': 'pass_logical_boundary', 'detail': 'Sanitized feature-builder subprocesses receive training-only forms, runtime labels blank on pending rows. Grid worker receives only fit_ids gold labels and fold-local caches. Gold shared-validation/outer labels and forms remain evaluator-side. No OS filesystem sandbox.'},
        'confidence': {'threshold': .5, 'semantics': 'Uncalibrated winning-class GB probability; native unsure preserved; no clamp',
                       'calibration': 'Training-only nested parse-score isotonic; no output calibration',
                       'selection': 'GridSearchCV mean 3-fold macro F1 after fixed050, all three classes, zero_division=0; tie by ParameterGrid order; refit=False. Never adjust from shared validation or outer test.'},
        'hardware': {'os': platform.platform(), 'machine': platform.machine(), 'cpu': platform.processor(),
            'grid_jobs': 2, 'threads_per_fit': 1, 'cpu_limit_enforced': None, 'memory_limit_bytes': None,
            'process_limit': None, 'timing': 'perf_counter native wall time; no constrained container'},
        'capabilities': {'classification': 'implemented', 'generation': 'not_implemented', 'proofs': 'not_implemented', 'container': 'not_run'},
        'model_asset_sha256': None, 'model_asset_reason': 'Fits in memory; fold caches, grids and predictions persisted with hashes.',
        'exact_reproduction_command': f'.venv/bin/python evaluation/run_daria_gb_grid.py --run-id NEW-UNIQUE-ID',
        'commands': [], 'inputs': {}, 'selections': {}, 'matched_controls': {}}
    write_json(out / 'run.json', run)

    def invoke(cmd, cwd, log=None):
        run['commands'].append(cmd)
        write_json(out / 'run.json', run)
        if log:
            with log.open('w') as stream:
                subprocess.run(cmd, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        else:
            subprocess.run(cmd, cwd=cwd, env=env, check=True)

    results = {}
    for track in TRACKS:
        split = manifest['splits'][track]
        fit_set = set(split['fit_ids'])
        fitting = [r for r in rows if r['id'] in fit_set]
        track_input = method_inputs / track
        track_input.mkdir()
        caches, fold_specs = [], []
        feature_start = time.perf_counter()
        for k, (train_index, test_index) in enumerate(make_folds(fitting, track)):
            train_ids = [fitting[i]['id'] for i in train_index]
            test_ids = [fitting[i]['id'] for i in test_index]
            cache_path = track_input / f'fold-{k}.json'
            print(f'{track}: rebuilding full feature pipeline for CV fold {k+1}/3 ({len(train_ids)} train, {len(test_ids)} validation)', flush=True)
            with tempfile.TemporaryDirectory(prefix='daria-gb-fold-') as temp:
                temp = Path(temp)
                write_csv(temp / 'runtime.csv', sanitized_rows(rows, train_ids, test_ids), ['id', 'text', 'date', 'label'])
                write_csv(temp / 'forms.csv', [r for r in rows if r['id'] in set(train_ids)], ['id', 'text', 'date', 'label', 'form'])
                write_json(temp / 'calibration-fit.json', train_ids[:int(len(train_ids) * .8)])
                run['inputs'][f'{track}/fold-{k}'] = {f: digest(temp / f) for f in ('runtime.csv', 'forms.csv', 'calibration-fit.json')}
                invoke([sys.executable, str(ROOT / 'integrations/daria_features.py'), '--source-dir', str(source),
                    '--input', str(temp / 'runtime.csv'), '--forms', str(temp / 'forms.csv'),
                    '--lexicon', str(ROOT / 'data/lexicon.json'), '--calibration-fit-ids', str(temp / 'calibration-fit.json'),
                    '--output', str(cache_path)], temp)
            cache = json.loads(cache_path.read_text())
            assert set(cache['train_ids']) == set(train_ids)
            assert set(cache['audit']['train_ids']) == set(train_ids)
            assert set(cache['audit']['pending_ids']) == set(test_ids)
            assert cache['audit']['pending_labels_and_forms_absent']
            run['inputs'][f'{track}/fold-{k}']['cache_sha256'] = digest(cache_path)
            caches.append({'path': str(cache_path), 'train_ids': train_ids, 'test_ids': test_ids})
            fold_specs.append({'train_indices': train_index.tolist(), 'test_indices': test_index.tolist(),
                'train_ids': train_ids, 'test_ids': test_ids,
                'train_date_min': min(fitting[i]['date'] for i in train_index),
                'train_date_max': max(fitting[i]['date'] for i in train_index),
                'validation_date_min': min(fitting[i]['date'] for i in test_index),
                'validation_date_max': max(fitting[i]['date'] for i in test_index)})
        feature_seconds = time.perf_counter() - feature_start
        write_json(track_input / 'cache-manifest.json', {'caches': caches})
        spec = {'cache_manifest': str(track_input / 'cache-manifest.json'),
                'ids': [r['id'] for r in fitting], 'labels': [r['label'] for r in fitting], 'folds': fold_specs}
        write_json(track_input / 'grid-spec.json', spec)
        track_out = out / 'fixed_050' / track
        track_out.mkdir(parents=True)
        print(f'{track}: searching {len(ParameterGrid(PARAM_GRID))} parameter combinations × 3 folds', flush=True)
        invoke([sys.executable, str(ROOT / 'experiments/daria_gb_grid.py'), 'search',
            '--spec', str(track_input / 'grid-spec.json'), '--output', str(track_out / 'grid-results.json')],
            track_input, track_out / 'grid.log')
        grid = json.loads((track_out / 'grid-results.json').read_text())
        chosen = grid['best_params']
        run['selections'][track] = {'best_params': chosen, 'cv_macro_f1': grid['best_score'],
                                   'frozen_before_shared_validation_and_outer_test': True}
        write_json(track_out / 'selected-params.json', chosen)
        write_json(out / 'run.json', run)
        print(f'{track}: frozen best={chosen}, CV macro F1={grid["best_score"]:.4f}', flush=True)
        # These original caches are already split-scoped. Verify exact sources, hashes,
        # train/pending pools, and absence of pending forms before reuse.
        phase_metrics = {}
        for phase, train_ids, test_ids in (
            ('inner', split['fit_ids'], split['validation_ids']),
            ('outer', split['train_ids'], split['test_ids'])):
            original = previous / track / phase / 'features.json'
            assert digest(original) == old_run['inputs'][f'{track}/{phase}']['feature_cache_sha256']
            cache = json.loads(original.read_text())
            assert set(cache['train_ids']) == set(train_ids)
            assert set(cache['audit']['train_ids']) == set(train_ids)
            assert set(cache['audit']['pending_ids']) == set(test_ids)
            assert cache['audit']['pending_labels_and_forms_absent']
            cache_path = track_input / f'{phase}-features.json'
            shutil.copyfile(original, cache_path)
            run['inputs'][f'{track}/{phase}'] = {'cache_sha256': digest(cache_path),
                'reused_from': str(original), 'source_run_sha256': digest(previous / 'run.json')}
            phase_out = track_out / 'validation' if phase == 'inner' else track_out
            phase_out.mkdir(exist_ok=True)
            start = time.perf_counter()
            invoke([sys.executable, str(ROOT / 'experiments/daria_gb_grid.py'), 'predict',
                '--cache', str(cache_path), '--params', str(track_out / 'selected-params.json'),
                '--output', str(phase_out / 'predictions.csv'), '--timings', str(phase_out / 'timings.json')], track_input)
            elapsed = time.perf_counter() - start
            gold = [r for r in rows if r['id'] in set(test_ids)]
            c, errors, scored = score_file(gold, phase_out / 'predictions.csv')
            write_csv(phase_out / 'scored_predictions.csv', scored, SCORED_FIELDS)
            metrics = {'benchmark_version': 'arena-eval-v1', 'method_id': METHOD, 'run_id': args.run_id,
                'split': track, 'policy': 'fixed_050', 'access_profile': 'batch_unlabelled_v1',
                'eligible_for_comparison': True, 'output_errors': errors,
                'classification': {'status': 'ok', **c},
                'generation': {'status': 'not_implemented', 'reason': 'Classification-only', 'verified_valid_yield_at_1000': None},
                'proofs': {'status': 'not_implemented', 'reason': 'Probabilistic model, no independent proof verifier', 'decision_proof_rate': None, 'grounded_decision_yield': None},
                'resources': {'status': 'not_run', 'reason': 'Constrained Linux amd64 benchmark not run', 'peak_ram_bytes': None,
                    'grid_search_seconds_native': grid['search_seconds'], 'cv_feature_build_seconds_native': feature_seconds,
                    'prediction_process_seconds_native': elapsed,
                    'shared_feature_build_timings_native': cache['timings'],
                    **json.loads((phase_out / 'timings.json').read_text())}}
            write_json(phase_out / 'metrics.json', metrics)
            phase_metrics[phase] = metrics
        # Rescore matching immutable controls. These use identical outer evidence pools.
        controls = {}
        for name, folder, rid in (
            ('untuned_gb', 'daria_sk_expanded_none_gb', PREVIOUS),
            ('native', 'daria_latest_native', PREVIOUS),
            ('always_true', 'always_true', 'text-variants-20261028-001'),
            ('tfidf_lr', 'tfidf_lr', 'text-variants-20261028-001'),
            ('tfidf_lr_balanced', 'tfidf_lr_balanced', 'text-variants-20261028-001')):
            base = ROOT / 'results/evaluations' / folder / rid
            crun = json.loads((base / 'run.json').read_text())
            assert crun['pinned_sha256'] == HASHES and crun['access_profile'] == 'batch_unlabelled_v1' and crun['seed'] == 42
            assert set(crun['training_ids'][track]) == set(split['train_ids'])
            assert set(crun['test_ids'][track]) == set(split['test_ids'])
            path = base / 'fixed_050' / track / 'predictions.csv'
            c, _, control_scored = score_file([r for r in rows if r['id'] in set(split['test_ids'])], path)
            ours = {r['id']: r for r in read_csv(track_out / 'scored_predictions.csv')}
            controls[name] = {'classification': c, 'run_path': str(base / 'run.json'),
                'run_sha256': digest(base / 'run.json'), 'predictions_sha256': digest(path),
                'fixed_control_errors': sum(not r['correct'] and ours[r['id']]['correct'] == 'True' for r in control_scored),
                'introduced_errors': sum(r['correct'] and ours[r['id']]['correct'] == 'False' for r in control_scored)}
        run['matched_controls'][track] = controls
        results[track] = phase_metrics
        print(f'{track}: shared validation F1={phase_metrics["inner"]["classification"]["macro_f1"]:.4f}, outer test F1={phase_metrics["outer"]["classification"]["macro_f1"]:.4f}, untuned GB F1={controls["untuned_gb"]["classification"]["macro_f1"]:.4f}', flush=True)
        write_json(out / 'run.json', run)
        write_json(out / 'summary.json', results)
    write_report(out, run, results)
    print(f'Completed: {out}', flush=True)


def write_report(out, run, results):
    lines = ['# GradientBoostingClassifier: fold-rebuilt GridSearchCV', '',
        'The expanded temporal feature set is unchanged. The single tested change is choosing gradient-boosting hyperparameters '
        'using GridSearchCV on shared fit_ids. Each CV fold rebuilds the full reader, parse calibration, windows and HMM from '
        'its own labelled rows/forms; fold validation is unlabelled text with no forms. This avoids applying ordinary CV to '
        'features already learned from all tuning labels.', '',
        f'Run `{run["run_id"]}`. {run["parameter_combinations"]} parameter combinations × 3 folds per track; '
        '972 CV classifier fits total. Search criterion: mean macro F1 after fixed_050. Ties use declared ParameterGrid order. '
        'The grid includes the unchanged GB parameters (80 trees, learning rate .05, depth 2, leaf minimum 1, subsample 1). '
        'CV runs only within frozen fit_ids. Shared validation is scored once after selection; it never changes selected parameters. '
        'Final classifier is refitted on complete outer train_ids using verified split-scoped cached features.', '',
        'CV is stratified three-fold with seed 42 on stratified/checker and expanding whole-date forward three-fold on temporal. '
        'These are internal training folds; the shared outer benchmark is unchanged. Parser calibration has a nested training-only '
        '80/20 original-order holdout. Native internal HMM grouping retains seed 1 and permitted training forms; it is not '
        'a completely cross-fitted reader within its own training features.', '',
        '| Track | Best CV F1 | Shared validation F1 | Untuned GB test F1 | Tuned test F1 | ΔF1 |',
        '|---|---:|---:|---:|---:|---:|']
    for track in TRACKS:
        old = run['matched_controls'][track]['untuned_gb']['classification']
        new = results[track]['outer']['classification']
        lines.append(f'| {track} | {run["selections"][track]["cv_macro_f1"]:.3f} | {results[track]["inner"]["classification"]["macro_f1"]:.3f} | {old["macro_f1"]:.3f} | {new["macro_f1"]:.3f} | {new["macro_f1"]-old["macro_f1"]:+.3f} |')
    lines += ['', 'Selected parameters:', '', '| Track | Trees | Learning rate | Max depth | Min samples per leaf | Subsample |', '|---|---:|---:|---:|---:|---:|']
    for track in TRACKS:
        p = run['selections'][track]['best_params']
        lines.append(f'| {track} | {p["n_estimators"]} | {p["learning_rate"]} | {p["max_depth"]} | {p["min_samples_leaf"]} | {p["subsample"]} |')
    lines += ['', '| Track | False precision / recall | Unsure F1 | Coverage | T/F errors / count | Risk | Brier | Search / feature build |',
              '|---|---:|---:|---:|---:|---:|---:|---|']
    for track in TRACKS:
        m = results[track]['outer']; c = m['classification']; p = c['per_class']; r = m['resources']
        lines.append(f'| {track} | {p["false"]["precision"]:.3f}/{p["false"]["recall"]:.3f} | {p["unsure"]["f1"]:.3f} | {c["decided_coverage"]:.3f} | {c["decided_errors"]}/{c["decided_count"]} | {c["decided_risk"]:.3f} | {c["correctness_brier"]:.3f} | {r["grid_search_seconds_native"]:.1f}s / {r["cv_feature_build_seconds_native"]:.1f}s native |')
    lines += ['', 'Matched controls (all were independently rescored from saved predictions with the same protocol, IDs, access and policy):', '',
        '| Track / control | Control F1 | Tuned ΔF1 | Control T/F errors/count | Fixed errors | Introduced errors |', '|---|---:|---:|---:|---:|---:|']
    for track in TRACKS:
        for name, value in run['matched_controls'][track].items():
            c = value['classification']; new = results[track]['outer']['classification']
            lines.append(f'| {track} / {name} | {c["macro_f1"]:.3f} | {new["macro_f1"]-c["macro_f1"]:+.3f} | {c["decided_errors"]}/{c["decided_count"]} | {value["fixed_control_errors"]} | {value["introduced_errors"]} |')
    lines += ['', 'Representative errors; no post-test retuning:', '', '| Track / ID | Gold | Prediction | Confidence | Reason |', '|---|---|---|---:|---|']
    errors = [dict(r, split=track) for track in TRACKS for r in read_csv(out / 'fixed_050' / track / 'scored_predictions.csv') if r['correct'] == 'False']
    errors.sort(key=lambda r: (r['split'] != 'stratified', r['gold'] == 'true', r['id']))
    for r in errors[:5]:
        lines.append(f'| {r["split"]}/{r["id"]} | {r["gold"]} | {r["label"]} | {r["confidence"]} | {r["decision_reason"]} |')
    lines += ['', 'No independent sentence-level causal audit was performed: these are observed classifier errors, not verified '
        'parser/evidence diagnoses. Probabilistic persistence, gold-form training versus parsed inference, scarce unsure examples '
        'and correlated rows remain limitations. Outer development results have already informed the choice of GB for this follow-up; '
        'the test sets are not pristine. CV-selected parameters are best within this finite grid and training protocol, not globally optimal.', '',
        'Generation, independently checked proofs, grounded yield and constrained peak RAM are unmeasured/null. Native macOS timings '
        'are supplementary. No independent generated-truth or submission-readiness claim is made. Confidence is uncalibrated.', '',
        'Full CV fold scores, timings, confusion matrices, per-class supports and reliability bins:', '']
    for track in TRACKS:
        lines.append(f'- [{track} full grid results](fixed_050/{track}/grid-results.json); [{track} outer metrics](fixed_050/{track}/metrics.json); [{track} shared-validation metrics](fixed_050/{track}/validation/metrics.json).')
    lines += ['', 'Reproduce with a new ID:', '', f'```sh\n{run["exact_reproduction_command"]}\n```', '']
    (out / 'report.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
