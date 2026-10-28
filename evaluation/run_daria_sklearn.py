#!/usr/bin/env python3
"""Frozen sklearn sweep over latest Daria features, with training-side selection."""
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
from evaluation.run_text_experiments import (HASHES, TRACKS, CONTROLS, digest, read_csv,
    write_csv, write_json, sanitized_rows, numeric_predictions, source_version)
from evaluation.scoring import score_rows
from experiments.daria_sklearn import CONFIGS

SOURCE_FILES = ('experiments/daria_sklearn.py', 'integrations/daria_features.py',
                'integrations/daria_candidates.py', 'integrations/daria_predict.py',
                'evaluation/run_daria_sklearn.py', 'evaluation/run_text_experiments.py',
                'evaluation/scoring.py', 'EVALUATION.md', 'requirements.txt')
FIELDS = ['id', 'label', 'confidence', 'decision_reason', 'p_true', 'p_false', 'p_unsure']
SCORED_FIELDS = ['id', 'gold', 'label', 'confidence', 'decision_reason', 'correct',
                 'output_valid', 'error_codes']


def score_file(gold, path):
    c, errors, scored = score_rows(gold, numeric_predictions(read_csv(path)))
    if not errors['eligible_for_comparison']:
        raise ValueError(f'{path}: {errors}')
    return c, errors, scored


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--ref', default='5943f61', help='Pinned latest reviewed Daria commit')
    args = ap.parse_args()
    if not args.run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in args.run_id):
        ap.error('Invalid run ID')
    for path, expected in HASHES.items():
        if digest(ROOT / path) != expected:
            ap.error(f'Benchmark hash mismatch: {path}')
    destination = ROOT / 'results/comparisons' / args.run_id
    if destination.exists():
        ap.error('Run exists; use a new run ID')
    destination.mkdir(parents=True)
    source = destination / 'source'
    commit = subprocess.check_output(['git', 'rev-parse', args.ref], cwd=ROOT, text=True).strip()
    files = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', commit, 'hooly'], cwd=ROOT, text=True).splitlines()
    hashes = {}
    for file in files:
        if file.endswith('.py') and file not in ('hooly/run.py', 'hooly/writer.py'):
            path = source / file
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(subprocess.check_output(['git', 'show', f'{commit}:{file}'], cwd=ROOT))
            hashes[file] = digest(path)
    source_hashes = {f: digest(ROOT / f) for f in SOURCE_FILES}
    rows = read_csv(ROOT / 'data/practice.csv')
    manifest = json.loads((ROOT / 'evaluation/splits-v1.json').read_text())
    env = os.environ.copy()
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        env[key] = '2'
    names = ['native'] + [c['id'] for c in CONFIGS]
    methods = {n: 'daria_latest_native' if n == 'native' else f'daria_sk_{n}' for n in names}
    methods['selected'] = 'daria_sklearn_selected'
    common = {'benchmark_version': 'arena-eval-v1', 'run_id': args.run_id,
        'pinned_sha256': HASHES, 'seed': 42, 'splits': list(TRACKS), 'policies': ['fixed_050'],
        'access_profile': 'batch_unlabelled_v1', 'source_commit': source_version(),
        'upstream_commit': commit, 'upstream_source_sha256': hashes,
        'source_file_sha256': source_hashes,
        'dependencies': {p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'scikit-learn', 'joblib', 'threadpoolctl')},
        'python': platform.python_version(),
        'training_ids': {t: manifest['splits'][t]['train_ids'] for t in TRACKS},
        'fit_ids': {t: manifest['splits'][t]['fit_ids'] for t in TRACKS},
        'validation_ids': {t: manifest['splits'][t]['validation_ids'] for t in TRACKS},
        'test_ids': {t: manifest['splits'][t]['test_ids'] for t in TRACKS},
        'training_sources': 'Split training rows/forms and public lexicon only. Inner model selection rebuilds entire pipeline on fit_ids only; validation forms absent. Final fit uses full train_ids.',
        'factual_asset_provenance': 'Reader, temporal windows, HMM and parse calibrator rebuilt for each pool. Probabilistic timelines include unlabelled batch hypotheses, never proof premises.',
        'leakage_audit': {'status': 'pass_logical_boundary', 'detail': 'Feature extraction and candidate fitting run in subprocesses with sanitized runtime CSV, train-only forms or feature caches. Gold lives only in evaluator. Filesystem isolation not OS-enforced. Internal native HMM OOF hides labels but retains permitted training forms; training-feature optimism remains.'},
        'model_asset_sha256': None, 'model_asset_reason': 'Fitted in memory, not serialized; caches and their hashes retained.',
        'confidence': {'threshold': .5, 'semantics': 'Winning-class model probability, not evidence proof. Native/most candidates uncalibrated; LinearSVC and Ridge use training-only sigmoid CV calibration.',
                       'calibration': 'Parse isotonic on training-only holdout; final parser refit. Candidate calibration stays in fitting pool. Preprocessing inside estimator pipelines.',
                       'selection': 'Highest frozen inner-validation macro F1, tie by declared candidate order. Native included; no outer-test selection.'},
        'hardware': {'os': platform.platform(), 'machine': platform.machine(), 'cpu': platform.processor(), 'thread_env_limit': 2,
                     'cpu_limit_enforced': None, 'memory_limit_bytes': None, 'process_limit': None, 'timing': 'perf_counter native wall time'},
        'capabilities': {'classification': 'implemented', 'generation': 'not_implemented', 'proofs': 'not_implemented', 'constrained_container': 'not_run'},
        'exact_reproduction_command': f'.venv/bin/python evaluation/run_daria_sklearn.py --run-id NEW-UNIQUE-ID --ref {commit}',
        'candidate_configs': list(CONFIGS), 'inputs': {}, 'commands': [], 'selections': {}}
    write_json(destination / 'run.json', common)
    results = {n: {} for n in names + ['selected']}
    failures = {n: [] for n in names + ['selected']}
    for track in TRACKS:
        split = manifest['splits'][track]
        fit_ids, val_ids = set(split['fit_ids']), set(split['validation_ids'])
        if fit_ids & val_ids or fit_ids | val_ids != set(split['train_ids']):
            raise ValueError('Invalid inner partition')
        phase_paths = {}
        phase_seconds = {}
        selection = None
        for phase in ('inner', 'outer'):
            train_ids = split['fit_ids'] if phase == 'inner' else split['train_ids']
            test_ids = split['validation_ids'] if phase == 'inner' else split['test_ids']
            phase_dir = destination / track / phase
            phase_dir.mkdir(parents=True)
            # Inner parse calibration is nested inside fit_ids, deterministic original-order
            # 80/20 holdout. Outer parse calibration uses frozen shared fit/validation IDs.
            cal_ids = train_ids[:int(len(train_ids) * .8)] if phase == 'inner' else split['fit_ids']
            cache_path = phase_dir / 'features.json'
            with tempfile.TemporaryDirectory(prefix='daria-sk-input-') as temp:
                temp = Path(temp)
                write_csv(temp / 'runtime.csv', sanitized_rows(rows, train_ids, test_ids), ['id', 'text', 'date', 'label'])
                write_csv(temp / 'forms.csv', [r for r in rows if r['id'] in set(train_ids)], ['id', 'text', 'date', 'label', 'form'])
                write_json(temp / 'calibration-fit.json', cal_ids)
                common['inputs'][f'{track}/{phase}'] = {f: digest(temp / f) for f in ('runtime.csv', 'forms.csv', 'calibration-fit.json')}
                cmd = [sys.executable, str(ROOT / 'integrations/daria_features.py'), '--source-dir', str(source),
                    '--input', str(temp / 'runtime.csv'), '--forms', str(temp / 'forms.csv'), '--lexicon', str(ROOT / 'data/lexicon.json'),
                    '--calibration-fit-ids', str(temp / 'calibration-fit.json'), '--output', str(cache_path)]
                common['commands'].append(cmd)
                write_json(destination / 'run.json', common)
                print(f'{track}/{phase}: rebuilding reader, temporal windows and HMM', flush=True)
                start = time.perf_counter()
                subprocess.run(cmd, cwd=temp, env=env, check=True)
                phase_seconds[phase] = time.perf_counter() - start
            cache = json.loads(cache_path.read_text())
            assert set(cache['audit']['train_ids']) == set(train_ids)
            assert set(cache['audit']['pending_ids']) == set(test_ids)
            common['inputs'][f'{track}/{phase}']['feature_cache_sha256'] = digest(cache_path)
            cmd = [sys.executable, str(ROOT / 'integrations/daria_candidates.py'), '--cache', str(cache_path), '--output-dir', str(phase_dir / 'predictions')]
            common['commands'].append(cmd)
            with (phase_dir / 'fit.log').open('w') as log:
                subprocess.run(cmd, cwd=phase_dir, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
            phase_paths[phase] = phase_dir
            if phase == 'inner':
                scores, unavailable = {}, {}
                timings = json.loads((phase_dir / 'predictions/timings.json').read_text())
                for name in names:
                    if timings[name]['status'] != 'ok':
                        unavailable[name] = timings[name]
                        continue
                    c, _, _ = score_file([r for r in rows if r['id'] in val_ids], phase_dir / 'predictions' / f'{name}.csv')
                    scores[name] = c
                chosen = max(scores, key=lambda n: (scores[n]['macro_f1'], -names.index(n)))
                selection = {'selected': chosen, 'selection_metric': 'macro_f1', 'tie_break': 'declared candidate order',
                             'validation_metrics': scores, 'unavailable': unavailable,
                             'fit_ids': split['fit_ids'], 'validation_ids': split['validation_ids']}
                common['selections'][track] = selection
                write_json(destination / track / 'selection.json', selection)
                write_json(destination / 'run.json', common)
                print(f'{track}: frozen selection={chosen}, validation F1={scores[chosen]["macro_f1"]:.4f}', flush=True)
        outer = phase_paths['outer']
        timings = json.loads((outer / 'predictions/timings.json').read_text())
        cache_timing = json.loads((outer / 'features.json').read_text())['timings']
        gold = [r for r in rows if r['id'] in set(split['test_ids'])]
        for name in names + ['selected']:
            candidate = selection['selected'] if name == 'selected' else name
            target = ROOT / 'results/evaluations' / methods[name] / args.run_id / 'fixed_050' / track
            target.mkdir(parents=True, exist_ok=False)
            if timings[candidate]['status'] != 'ok':
                write_json(target / 'failure.json', timings[candidate])
                continue
            predictions = read_csv(outer / 'predictions' / f'{candidate}.csv')
            write_csv(target / 'predictions.csv', predictions, FIELDS)
            start = time.perf_counter()
            c, errors, scored = score_file(gold, target / 'predictions.csv')
            write_csv(target / 'scored_predictions.csv', scored, SCORED_FIELDS)
            metrics = {'benchmark_version': 'arena-eval-v1', 'method_id': methods[name], 'run_id': args.run_id,
                'split': track, 'policy': 'fixed_050', 'access_profile': 'batch_unlabelled_v1',
                'eligible_for_comparison': True, 'output_errors': errors, 'selected_candidate': candidate,
                'classification': {'status': 'ok', **c},
                'generation': {'status': 'not_implemented', 'reason': 'Classification-only experiment; upstream writer not evaluated', 'verified_valid_yield_at_1000': None},
                'proofs': {'status': 'not_implemented', 'reason': 'Probabilistic traces are not independent proofs', 'decision_proof_rate': None, 'grounded_decision_yield': None},
                'resources': {'status': 'not_run', 'reason': 'Offline Linux amd64 constrained benchmark not run',
                              'peak_ram_bytes': None, 'feature_process_seconds': phase_seconds['outer'],
                              'shared_features': cache_timing, 'candidate': timings[candidate],
                              'external_audit_seconds': time.perf_counter() - start}}
            write_json(target / 'metrics.json', metrics)
            results[name][track] = metrics
            failures[name].extend({'split': track, **r} for r in scored if not r['correct'])
        print(f'{track}: native={results["native"][track]["classification"]["macro_f1"]:.4f}; selected={results["selected"][track]["classification"]["macro_f1"]:.4f}', flush=True)
    # Re-score pinned matching controls instead of silently copying metrics.
    controls = {}
    for control in CONTROLS:
        base = ROOT / 'results/evaluations' / control / 'text-variants-20261028-001'
        run = json.loads((base / 'run.json').read_text())
        assert run['pinned_sha256'] == HASHES and run['seed'] == 42
        assert run['access_profile'] == 'batch_unlabelled_v1'
        controls[control] = {}
        for track in TRACKS:
            assert set(run['training_ids'][track]) == set(manifest['splits'][track]['train_ids'])
            assert set(run['test_ids'][track]) == set(manifest['splits'][track]['test_ids'])
            path = base / 'fixed_050' / track / 'predictions.csv'
            c, _, _ = score_file([r for r in rows if r['id'] in set(manifest['splits'][track]['test_ids'])], path)
            controls[control][track] = {'classification': c, 'prediction_sha256': digest(path),
                                       'run_sha256': digest(base / 'run.json'), 'run_path': str(base / 'run.json')}
    write_json(destination / 'controls.json', controls)
    write_json(destination / 'summary.json', results)
    write_json(destination / 'run.json', common)
    for name, tracks in results.items():
        out = ROOT / 'results/evaluations' / methods[name] / args.run_id
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / 'run.json', {**common, 'method_id': methods[name], 'candidate': name})
        lines = [f'# {methods[name]}', '', f'Run `{args.run_id}`, Daria `{commit[:7]}`, arena-eval-v1 / fixed_050.', '',
            'Latest reader and temporal/HMM features with a declared sklearn decision pipeline. '
            'The native control retains Daria’s binary stacker; sklearn candidates learn all three labels. '
            'Selected method chooses only using frozen inner validation and refits on full outer train. '
            'All candidates were declared before this sweep. These are development splits already used by the team.', '',
            '| Split | Candidate | Macro F1 | False P/R | Unsure F1 | Coverage | Errors/count | Risk | Brier | Native ΔF1 | Runtime / RAM |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
        for track, m in tracks.items():
            c, native = m['classification'], results['native'][track]['classification']
            p = c['per_class']; res = m['resources']
            risk = 'n/a' if c['decided_risk'] is None else f'{c["decided_risk"]:.3f}'
            brier = 'n/a' if c['correctness_brier'] is None else f'{c["correctness_brier"]:.3f}'
            lines.append(f'| {track} | {m["selected_candidate"]} | {c["macro_f1"]:.3f} | {p["false"]["precision"]:.3f}/{p["false"]["recall"]:.3f} | {p["unsure"]["f1"]:.3f} | {c["decided_coverage"]:.3f} | {c["decided_errors"]}/{c["decided_count"]} | {risk} | {brier} | {c["macro_f1"]-native["macro_f1"]:+.3f} | shared features {res["feature_process_seconds"]:.1f}s + fit {res["candidate"]["fit_seconds"]:.2f}s / unmeasured |')
        lines += ['', 'Generation, independent proof rate, grounded decision yield and constrained peak RAM are unavailable (null). '
            'Confusion matrices, class supports, reliability bins and full metrics are in each split’s `metrics.json`.', '',
            '| Split | Always true F1 | Word LR F1 | Balanced word LR F1 | Δ vs balanced LR |', '|---|---:|---:|---:|---:|']
        for track, m in tracks.items():
            cs = [controls[k][track]['classification']['macro_f1'] for k in CONTROLS]
            lines.append(f'| {track} | {cs[0]:.3f} | {cs[1]:.3f} | {cs[2]:.3f} | {m["classification"]["macro_f1"]-cs[2]:+.3f} |')
        lines += ['', 'Representative errors (diagnostic only; no post-test retuning):', '',
                  '| Split / ID | Gold | Predicted | Confidence | Reason |', '|---|---|---|---:|---|']
        fs = sorted(failures[name], key=lambda r: (r['split'] != 'stratified', r['gold'] == 'true', r['id']))
        for r in fs[:5]:
            lines.append(f'| {r["split"]} / {r["id"]} | {r["gold"]} | {r["label"]} | {r["confidence"]} | {r["decision_reason"]} |')
        lines += ['', 'No independent sentence-level causal audit was performed; errors can arise from extraction, temporal assumptions or classifier generalization. '
            'Native HMM out-of-fold training hides labels while retaining permitted training forms; this is not a fully cross-fitted extraction pipeline. '
            'Gold-form training versus parsed inference remains a distribution mismatch. Timelines assume persistence and use noisy batch observations; scores supply no factual proof. '
            'Tiny unsure support and selection across many candidates increase variance. Native macOS timing is supplementary; no submission-readiness claim.', '',
            f'Reproduce with a new ID: `{common["exact_reproduction_command"]}`', '']
        (out / 'report.md').write_text('\n'.join(lines))
    print(f'Completed: {destination}', flush=True)


if __name__ == '__main__':
    main()
