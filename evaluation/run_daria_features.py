#!/usr/bin/env python3
"""Compare frozen feature representations with fixed CV-tuned gradient boosting."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.run_text_experiments import (HASHES, TRACKS, digest, read_csv,
    write_csv, write_json, sanitized_rows, numeric_predictions, source_version)
from evaluation.scoring import score_rows
from experiments.daria_feature_variants import CONFIGS
from experiments.daria_feature_sweep import audit_cache

OLD_SWEEP = ROOT / 'results/comparisons/daria-sklearn-20261028-001'
OLD_GRID = ROOT / 'results/evaluations/daria_gb_grid/daria-gb-grid-20261028-002'
SOURCE_FILES = ('evaluation/run_daria_features.py', 'experiments/daria_feature_sweep.py',
    'experiments/daria_feature_variants.py', 'integrations/daria_rich_features.py',
    'experiments/daria_sklearn.py', 'integrations/daria_features.py', 'integrations/daria_predict.py',
    'integrations/daria_candidates.py', 'evaluation/scoring.py', 'evaluation/run_text_experiments.py',
    'EVALUATION.md', 'requirements.txt')
FIELDS = ['id', 'label', 'confidence', 'decision_reason', 'p_true', 'p_false', 'p_unsure']
SCORED_FIELDS = ['id', 'gold', 'label', 'confidence', 'decision_reason', 'correct', 'output_valid', 'error_codes']


def project_original(feature):
    return {k: v for k, v in feature.items() if k == '__text__' or k.startswith(('native__', 'expanded__'))}


def score_file(gold, path):
    c, errors, scored = score_rows(gold, numeric_predictions(read_csv(path)))
    if not errors['eligible_for_comparison']:
        raise ValueError(f'{path}: output audit failed: {errors}')
    return c, errors, scored


def compare_predictions(reference, actual):
    left = {r['id']: r for r in numeric_predictions(read_csv(reference))}
    right = {r['id']: r for r in numeric_predictions(read_csv(actual))}
    if left.keys() != right.keys():
        raise ValueError('Baseline parity: different IDs')
    maximum = 0.
    for rid, row in left.items():
        for field in FIELDS[1:]:
            a, b = row[field], right[rid][field]
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                maximum = max(maximum, abs(a - b))
            elif a != b:
                raise ValueError(f'Baseline parity mismatch: {rid}/{field}: {a!r} != {b!r}')
    if maximum > 1e-12:
        raise ValueError(f'Baseline probability mismatch: {maximum}')
    return {'status': 'exact' if maximum == 0 else 'within_1e-12', 'rows': len(left),
            'max_numeric_difference': maximum, 'reference_sha256': digest(reference), 'actual_sha256': digest(actual)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-id', required=True)
    args = ap.parse_args()
    if not args.run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in args.run_id):
        ap.error('Invalid run ID')
    for file, expected in HASHES.items():
        if digest(ROOT / file) != expected:
            ap.error(f'Benchmark hash mismatch: {file}')
    old = json.loads((OLD_SWEEP / 'run.json').read_text())
    grid_run = json.loads((OLD_GRID / 'run.json').read_text())
    assert old['pinned_sha256'] == grid_run['pinned_sha256'] == HASHES
    for file, expected in old['source_file_sha256'].items():
        assert digest(ROOT / file) == expected, file
    for file, expected in grid_run['source_file_sha256'].items():
        assert digest(ROOT / file) == expected, file
    source = OLD_SWEEP / 'source'
    assert all(digest(source / f) == h for f, h in old['upstream_source_sha256'].items())
    out = ROOT / 'results/comparisons' / args.run_id
    if out.exists():
        ap.error('Run exists; choose a new ID')
    out.mkdir(parents=True)
    rows = read_csv(ROOT / 'data/practice.csv')
    manifest = json.loads((ROOT / 'evaluation/splits-v1.json').read_text())
    configs = [c['id'] for c in CONFIGS]
    method_ids = {name: f'daria_features_{name}' for name in configs}
    method_ids['selected'] = 'daria_features_selected'
    run = {'benchmark_version': 'arena-eval-v1', 'run_id': args.run_id, 'seed': 42,
        'internal_native_fold_seed': 1, 'source_commit': source_version(), 'pinned_sha256': HASHES,
        'source_file_sha256': {f: digest(ROOT / f) for f in SOURCE_FILES},
        'upstream_commit': old['upstream_commit'], 'upstream_source_sha256': old['upstream_source_sha256'],
        'dependencies': {p: importlib.metadata.version(p) for p in ('scikit-learn', 'numpy', 'scipy', 'joblib', 'threadpoolctl')},
        'python': platform.python_version(), 'splits': list(TRACKS), 'policies': ['fixed_050'],
        'access_profile': 'batch_unlabelled_v1', 'feature_configs': list(CONFIGS),
        'training_ids': {t: manifest['splits'][t]['train_ids'] for t in TRACKS},
        'fit_ids': {t: manifest['splits'][t]['fit_ids'] for t in TRACKS},
        'validation_ids': {t: manifest['splits'][t]['validation_ids'] for t in TRACKS},
        'test_ids': {t: manifest['splits'][t]['test_ids'] for t in TRACKS},
        'training_sources': 'Only permitted labelled rows/forms and public lexicon. Rich cache rebuilt per CV fold, shared-validation fit pool and full outer train. Pending labels blank/forms absent.',
        'feature_asset_provenance': 'Fresh rich features preserve old features exactly; true-anchor summaries use permitted true rows only, excluding own-page anchors on training features. Vectorizers, selectors, scaling and reductions fit on training only.',
        'fixed_classifier': {'family': 'GradientBoostingClassifier', 'params_by_track': {t: grid_run['selections'][t]['best_params'] for t in TRACKS},
            'source_run_sha256': digest(OLD_GRID / 'run.json'), 'note': 'Hyperparameters frozen from previous training-side CV, never retuned in this feature comparison. Reusing those CV folds makes feature CV scores selection diagnostics, not unbiased estimates.'},
        'cv_policy': 'Exact previous shared-fit-pool CV fold IDs; stratified 3-fold seed42 for stratified/checker, whole-date expanding forward folds for temporal. Feature winner by mean fixed050 macro F1, ties by declared config order. Freeze before shared validation/outer test.',
        'confidence': {'threshold': .5, 'semantics': 'Uncalibrated winning-class GB probability; native unsure retained, below .5 abstains',
                       'calibration': 'Nested training-only parse-score isotonic; no output calibration'},
        'leakage_audit': {'status': 'pass_logical_boundary', 'detail': 'Separate cache builders receive sanitized runtime and train-only forms. CV worker receives only fit-pool labels. Prediction worker receives only sanitized caches. Evaluator retains shared-validation and outer answers. No OS filesystem isolation.'},
        'hardware': {'os': platform.platform(), 'machine': platform.machine(), 'cpu': platform.processor(), 'thread_env_limit': 2,
            'cpu_limit_enforced': None, 'memory_limit_bytes': None, 'process_limit': None, 'timing': 'perf_counter native wall time'},
        'model_asset_sha256': None, 'model_asset_reason': 'Models fitted in memory; split caches, predictions and hashes saved.',
        'capabilities': {'classification': 'implemented', 'generation': 'not_implemented', 'proofs': 'not_implemented', 'container': 'not_run'},
        'exact_reproduction_command': '.venv/bin/python evaluation/run_daria_features.py --run-id NEW-UNIQUE-ID',
        'inputs': {}, 'commands': [], 'selections': {}, 'baseline_parity': {}, 'matched_controls': {}, 'candidate_status': {}}
    env = os.environ.copy()
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        env[name] = '2'
    write_json(out / 'run.json', run)

    def invoke(cmd, cwd, log=None):
        run['commands'].append(cmd)
        write_json(out / 'run.json', run)
        if log:
            with log.open('w') as stream:
                subprocess.run(cmd, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        else:
            subprocess.run(cmd, cwd=cwd, env=env, check=True)

    def build_cache(track, phase, train_ids, pending_ids, calibration_ids):
        target = out / 'method-inputs' / track
        target.mkdir(parents=True, exist_ok=True)
        cache_path = target / f'{phase}-features.json'
        with tempfile.TemporaryDirectory(prefix='daria-rich-input-') as temp:
            temp = Path(temp)
            write_csv(temp / 'runtime.csv', sanitized_rows(rows, train_ids, pending_ids), ['id', 'text', 'date', 'label'])
            train_set = set(train_ids)
            write_csv(temp / 'forms.csv', [r for r in rows if r['id'] in train_set], ['id', 'text', 'date', 'label', 'form'])
            write_json(temp / 'calibration-fit.json', calibration_ids)
            run['inputs'][f'{track}/{phase}'] = {f: digest(temp / f) for f in ('runtime.csv', 'forms.csv', 'calibration-fit.json')}
            start = time.perf_counter()
            invoke([sys.executable, str(ROOT / 'integrations/daria_rich_features.py'), '--source-dir', str(source),
                '--input', str(temp / 'runtime.csv'), '--forms', str(temp / 'forms.csv'),
                '--lexicon', str(ROOT / 'data/lexicon.json'), '--calibration-fit-ids', str(temp / 'calibration-fit.json'),
                '--output', str(cache_path)], temp)
            elapsed = time.perf_counter() - start
        cache = json.loads(cache_path.read_text())
        audit_cache(cache, train_ids, pending_ids)
        run['inputs'][f'{track}/{phase}'].update({'cache_sha256': digest(cache_path), 'feature_process_seconds': elapsed})
        return cache_path, cache

    results = {name: {} for name in configs + ['selected']}
    for track in TRACKS:
        split = manifest['splits'][track]
        fold_results = json.loads((OLD_GRID / 'fixed_050' / track / 'grid-results.json').read_text())
        params = grid_run['selections'][track]['best_params']
        track_input = out / 'method-inputs' / track
        track_input.mkdir(parents=True, exist_ok=True)
        write_json(track_input / 'gb-params.json', params)
        folds = []
        for k, fold in enumerate(fold_results['folds']):
            train_ids, test_ids = fold['train_ids'], fold['test_ids']
            assert set(train_ids) | set(test_ids) <= set(split['fit_ids'])
            assert not set(train_ids) & set(test_ids)
            if track == 'temporal':
                assert fold['train_date_max'] < fold['validation_date_min']
            print(f'{track}: rebuilding rich CV fold {k+1}/3', flush=True)
            cache_path, cache = build_cache(track, f'cv-{k}', train_ids, test_ids, train_ids[:int(len(train_ids)*.8)])
            previous_fold_path = OLD_GRID / 'method-inputs' / track / f'fold-{k}.json'
            assert digest(previous_fold_path) == grid_run['inputs'][f'{track}/fold-{k}']['cache_sha256']
            previous_fold = json.loads(previous_fold_path.read_text())
            assert cache['train_ids'] == previous_fold['train_ids']
            assert [project_original(f) for f in cache['train_features']] == previous_fold['train_features']
            assert {r['id']: project_original(r['features']) for r in cache['pending']} == {r['id']: r['features'] for r in previous_fold['pending']}
            folds.append({'path': str(cache_path), 'train_ids': train_ids, 'test_ids': test_ids})
        fit_labels = {r['id']: r['label'] for r in rows if r['id'] in set(split['fit_ids'])}
        assert set(fit_labels) == set(split['fit_ids'])
        write_json(track_input / 'cv-spec.json', {'folds': folds, 'fit_labels': fit_labels})
        cv_out = out / track / 'cv'
        cv_out.mkdir(parents=True)
        print(f'{track}: comparing {len(CONFIGS)} feature pipelines in CV', flush=True)
        invoke([sys.executable, str(ROOT / 'experiments/daria_feature_sweep.py'), '--mode', 'cv',
            '--spec', str(track_input / 'cv-spec.json'), '--params', str(track_input / 'gb-params.json'),
            '--output-dir', str(cv_out)], track_input, cv_out / 'fit.log')
        selection = json.loads((cv_out / 'selection.json').read_text())
        assert abs(selection['results']['expanded_none']['mean_macro_f1'] - grid_run['selections'][track]['cv_macro_f1']) < 1e-12, 'Original CV baseline changed'
        run['selections'][track] = selection
        write_json(out / 'run.json', run)
        print(f'{track}: frozen feature choice={selection["selected"]}, CV F1={selection["results"][selection["selected"]]["mean_macro_f1"]:.4f}', flush=True)
        for phase, train_ids, pending_ids, cal_ids in (
            ('inner', split['fit_ids'], split['validation_ids'], split['fit_ids'][:int(len(split['fit_ids'])*.8)]),
            ('outer', split['train_ids'], split['test_ids'], split['fit_ids'])):
            print(f'{track}: rebuilding rich {phase} features', flush=True)
            cache_path, cache = build_cache(track, phase, train_ids, pending_ids, cal_ids)
            previous = json.loads((OLD_SWEEP / track / phase / 'features.json').read_text())
            assert digest(OLD_SWEEP / track / phase / 'features.json') == old['inputs'][f'{track}/{phase}']['feature_cache_sha256']
            assert cache['train_ids'] == previous['train_ids'] and cache['train_labels'] == previous['train_labels']
            assert [project_original(f) for f in cache['train_features']] == previous['train_features'], 'Original training features changed'
            prev_pending = {r['id']: r['features'] for r in previous['pending']}
            assert {r['id']: project_original(r['features']) for r in cache['pending']} == prev_pending, 'Original pending features changed'
            phase_out = out / track / phase
            phase_out.mkdir(parents=True)
            invoke([sys.executable, str(ROOT / 'experiments/daria_feature_sweep.py'), '--mode', 'predict',
                '--cache', str(cache_path), '--params', str(track_input / 'gb-params.json'), '--output-dir', str(phase_out)],
                track_input, phase_out / 'fit.log')
            timings = json.loads((phase_out / 'timings.json').read_text())
            run['candidate_status'][f'{track}/{phase}'] = {name: {'status': timing['status'],
                'reason': timing.get('reason')} for name, timing in timings.items()}
            if phase == 'inner':
                diagnostic = {}
                for name in configs:
                    if timings[name]['status'] == 'ok':
                        c, _, _ = score_file([r for r in rows if r['id'] in set(pending_ids)], phase_out / f'{name}.csv')
                        diagnostic[name] = c
                write_json(out / track / 'shared-validation.json', diagnostic)
                continue
            run['baseline_parity'][track] = compare_predictions(
                OLD_GRID / 'fixed_050' / track / 'predictions.csv', phase_out / 'expanded_none.csv')
            gold = [r for r in rows if r['id'] in set(pending_ids)]
            for name in configs + ['selected']:
                candidate = selection['selected'] if name == 'selected' else name
                target = ROOT / 'results/evaluations' / method_ids[name] / args.run_id / 'fixed_050' / track
                target.mkdir(parents=True, exist_ok=False)
                if timings[candidate]['status'] != 'ok':
                    write_json(target / 'failure.json', timings[candidate])
                    continue
                write_csv(target / 'predictions.csv', read_csv(phase_out / f'{candidate}.csv'), FIELDS)
                c, errors, scored = score_file(gold, target / 'predictions.csv')
                write_csv(target / 'scored_predictions.csv', scored, SCORED_FIELDS)
                metrics = {'benchmark_version': 'arena-eval-v1', 'method_id': method_ids[name], 'run_id': args.run_id,
                    'split': track, 'policy': 'fixed_050', 'access_profile': 'batch_unlabelled_v1',
                    'eligible_for_comparison': True, 'output_errors': errors, 'feature_candidate': candidate,
                    'fixed_gb_params': params, 'classification': {'status': 'ok', **c},
                    'generation': {'status': 'not_implemented', 'reason': 'Classification-only feature comparison', 'verified_valid_yield_at_1000': None},
                    'proofs': {'status': 'not_implemented', 'reason': 'Temporal summaries and scores are not independent proofs', 'decision_proof_rate': None, 'grounded_decision_yield': None},
                    'resources': {'status': 'not_run', 'reason': 'Constrained Linux amd64 benchmark not run', 'peak_ram_bytes': None,
                        'shared_feature_process_seconds_native': run['inputs'][f'{track}/{phase}']['feature_process_seconds'],
                        **timings[candidate]}}
                # Preserve resource status rather than candidate fit's local status.
                metrics['resources']['status'] = 'not_run'
                write_json(target / 'metrics.json', metrics)
                results[name][track] = metrics
            print(f'{track}: baseline test F1={results["expanded_none"][track]["classification"]["macro_f1"]:.4f}, selected F1={results["selected"][track]["classification"]["macro_f1"]:.4f}', flush=True)
        controls = {}
        for alias, folder, rid in (
            ('tuned_gb', 'daria_gb_grid', 'daria-gb-grid-20261028-002'),
            ('native', 'daria_latest_native', 'daria-sklearn-20261028-001'),
            ('always_true', 'always_true', 'text-variants-20261028-001'),
            ('tfidf_lr', 'tfidf_lr', 'text-variants-20261028-001'),
            ('tfidf_lr_balanced', 'tfidf_lr_balanced', 'text-variants-20261028-001')):
            base = ROOT / 'results/evaluations' / folder / rid
            crun = json.loads((base / 'run.json').read_text())
            assert crun['pinned_sha256'] == HASHES and crun['seed'] == 42 and crun['access_profile'] == 'batch_unlabelled_v1'
            assert set(crun['training_ids'][track]) == set(split['train_ids'])
            assert set(crun['test_ids'][track]) == set(split['test_ids'])
            path = base / 'fixed_050' / track / 'predictions.csv'
            c, _, _ = score_file([r for r in rows if r['id'] in set(split['test_ids'])], path)
            controls[alias] = {'classification': c, 'run_path': str(base / 'run.json'),
                               'run_sha256': digest(base / 'run.json'), 'predictions_sha256': digest(path)}
        run['matched_controls'][track] = controls
        write_json(out / 'summary.json', results)
        write_json(out / 'run.json', run)
    write_reports(out, run, results, method_ids)
    print(f'Completed: {out}', flush=True)


def write_reports(out, run, results, method_ids):
    def number(value):
        return 'n/a' if value is None else f'{value:.3f}'

    table = []
    for name, tracks in results.items():
        target = ROOT / 'results/evaluations' / method_ids[name] / run['run_id']
        target.mkdir(parents=True, exist_ok=True)
        write_json(target / 'run.json', {**run, 'method_id': method_ids[name], 'candidate': name})
        lines = [f'# {method_ids[name]}', '',
            'Feature representation experiment using fixed, previously CV-tuned GradientBoostingClassifier parameters per track. '
            'Full feature resources are rebuilt per fold; learned vectorizers, selection and reductions fit on training only. '
            'The selected method freezes its representation using mean three-fold CV macro F1 inside shared fit_ids. '
            'Shared validation is diagnostic only, and final assets use full outer train_ids. All outputs use fixed_050.', '',
            '| Track | Feature candidate | CV F1 | Shared validation F1 | Test F1 | Δ vs base | False P/R | Unsure F1 | Coverage | Errors/count | Risk | Brier |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for track, m in tracks.items():
            c = m['classification']; p = c['per_class']; candidate = m['feature_candidate']
            cv = run['selections'][track]['results'][candidate].get('mean_macro_f1')
            val = json.loads((out / track / 'shared-validation.json').read_text()).get(candidate, {}).get('macro_f1')
            baseline = results['expanded_none'][track]['classification']
            lines.append(f'| {track} | {candidate} | {number(cv)} | {number(val)} | {c["macro_f1"]:.3f} | {c["macro_f1"]-baseline["macro_f1"]:+.3f} | {p["false"]["precision"]:.3f}/{p["false"]["recall"]:.3f} | {p["unsure"]["f1"]:.3f} | {c["decided_coverage"]:.3f} | {c["decided_errors"]}/{c["decided_count"]} | {number(c["decided_risk"])} | {number(c["correctness_brier"])} |')
            table.append({'method': name, 'track': track, 'candidate': candidate, 'status': 'ok', 'reason': None, 'cv_macro_f1': cv,
                'shared_validation_macro_f1': val, 'test_macro_f1': c['macro_f1'], 'accuracy': c['accuracy'],
                'false_precision': p['false']['precision'], 'false_recall': p['false']['recall'],
                'unsure_f1': p['unsure']['f1'], 'coverage': c['decided_coverage'], 'decided_errors': c['decided_errors'],
                'decided_count': c['decided_count'], 'risk': c['decided_risk'], 'brier': c['correctness_brier']})
        for track in TRACKS:
            if track not in tracks:
                candidate = run['selections'][track]['selected'] if name == 'selected' else name
                failure = run['candidate_status'][f'{track}/outer'][candidate]
                table.append({'method': name, 'track': track, 'candidate': candidate,
                              'status': 'failed', 'reason': failure['reason']})
                lines += ['', f'Unavailable on {track}: {failure["reason"]}. See [failure artifact](fixed_050/{track}/failure.json).']
        lines += ['', 'Matched controls:', '', '| Track | Always true F1 | Word LR F1 | Balanced word LR F1 | Native F1 | Tuned GB F1 |', '|---|---:|---:|---:|---:|---:|']
        for track in tracks:
            cs = run['matched_controls'][track]
            values = [cs[k]['classification']['macro_f1'] for k in ('always_true', 'tfidf_lr', 'tfidf_lr_balanced', 'native', 'tuned_gb')]
            lines.append(f'| {track} | ' + ' | '.join(f'{v:.3f}' for v in values) + ' |')
        lines += ['', 'Detailed metrics, full confusion matrices and reliability bins: ' + ', '.join(
            f'[{track}](fixed_050/{track}/metrics.json)' for track in tracks) + '.', '',
            'Representative errors (diagnostic; no post-test changes):', '', '| Track / ID | Gold | Predicted | Confidence | Reason |', '|---|---|---|---:|---|']
        failures = [dict(r, split=track) for track in tracks for r in read_csv(target / 'fixed_050' / track / 'scored_predictions.csv') if r['correct'] == 'False']
        failures.sort(key=lambda r: (r['split'] != 'stratified', r['gold'] == 'true', r['id']))
        for r in failures[:5]:
            lines.append(f'| {r["split"]}/{r["id"]} | {r["gold"]} | {r["label"]} | {r["confidence"]} | {r["decision_reason"]} |')
        lines += ['', 'No independent sentence-level causal error audit was performed. Generated truth, proof rate, grounded yield '
            'and constrained peak RAM are unavailable/null. Fit/inference timings and effective feature counts are in the metric '
            'resource section; native macOS timings are supplementary. No submission-readiness claim.', '',
            'Limitations: these repeatedly used development splits are not pristine. Classifier hyperparameters were selected '
            'previously on the same fit-pool CV folds, so current CV results are selection diagnostics. Few gold unsure examples '
            'make macro F1 variable. Gold-form training versus parsed inference and native internal HMM partial cross-fitting '
            'remain limitations. Anchor support/gaps are features, not proof of persistence. Feature classifiers do not change '
            'the reader or make unlabelled observations trusted facts. No new dependencies or pretrained external facts were added.', '',
            f'Reproduce with a new ID: `{run["exact_reproduction_command"]}`', '']
        (target / 'report.md').write_text('\n'.join(lines))
    write_csv(out / 'candidate-results.csv', table, list(table[0]))
    lines = ['# Additional feature extraction experiments', '',
        '24 declared feature pipelines with fixed per-track gradient-boosting hyperparameters. Methods include order-invariant '
        'assertion summaries, typed assertion/subject/attribute bags, true-anchor support/date gaps, binary word counts, '
        'word-boundary character TF-IDF, chi-square selection, latent text features via SVD, hashing, F/MI feature selection, '
        'and scaled PCA. Text-only controls distinguish stand-alone text signal from adding text to temporal features.', '',
        '| Track | CV-selected representation | Previous GB F1 | Selected test F1 | Δ |', '|---|---|---:|---:|---:|']
    for track in TRACKS:
        a = results['expanded_none'][track]['classification']['macro_f1']
        b = results['selected'][track]['classification']['macro_f1']
        lines.append(f'| {track} | {run["selections"][track]["selected"]} | {a:.4f} | {b:.4f} | {b-a:+.4f} |')
    failures = [(track, name, detail['reason']) for phase, candidates in run['candidate_status'].items()
                for name, detail in candidates.items() if detail['status'] != 'ok'
                for track in [phase]]
    lines += ['', f'Failed phase/candidate fits: {len(failures)}.']
    for track, name, reason in failures:
        lines.append(f'- {track} / {name}: {reason}')
    for track in TRACKS:
        for name, detail in run['selections'][track]['results'].items():
            if detail['status'] != 'ok':
                lines.append(f'- {track} CV / {name}: {detail["reason"]}')
    lines += ['', 'Selection uses full-pipeline CV on frozen fit_ids; shared validation and outer labels never select the '
        'representation. Existing GB hyperparameters were kept fixed. The baseline representation exactly preserves all original '
        'features and is checked against the previous GB predictions on each track. These are development results with no '
        'guarantee of unseen generalization. Shared-validation diagnostics, all candidate scores and uncertainty/risk tradeoffs '
        'remain visible.', '', f'[All candidate results](candidate-results.csv). '
        f'[Selected-method detailed report](../../evaluations/daria_features_selected/{run["run_id"]}/report.md).', '',
        'Generation and independently verified proofs were not evaluated; native timings are supplementary.', '',
        f'```sh\n{run["exact_reproduction_command"]}\n```', '']
    (out / 'report.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
