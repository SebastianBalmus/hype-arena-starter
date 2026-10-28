#!/usr/bin/env python3
"""Evaluate the split-scoped knowledge-base classifier under arena-eval-v1.

Gold labels/forms stay in the evaluator process. The method is invoked as a
separate process with sanitized row CSVs and a form file limited to its
permitted training IDs. This is a logical boundary, not an OS sandbox.
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import run_text_experiments as helpers
from evaluation.scoring import LABELS, score_rows

TRACKS = helpers.TRACKS
METHOD = ROOT / 'experiments/kb_predict.py'
METHOD_ID = 'kb_same_date'
METHODS = ('kb_same_date', 'kb_hybrid')
SOURCE_FILES = (
    'evaluation/run_kb_experiment.py', 'evaluation/run_text_experiments.py',
    'evaluation/scoring.py', 'experiments/kb_predict.py',
    'experiments/text_variants.py', 'experiments/predict.py', 'baseline.py',
    'requirements.txt', 'EVALUATION.md',
    '.agents/skills/hype-arena-evidence-policy/SKILL.md',
)


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def read_forms(rows, allowed_ids):
    allowed = set(allowed_ids)
    selected = [{key: row[key] for key in ('id', 'text', 'date', 'label', 'form')}
                for row in rows if row['id'] in allowed]
    if {row['id'] for row in selected} != allowed:
        raise ValueError('Training-form IDs do not match the permitted training IDs')
    return selected


def write_json(path, value):
    helpers.write_json(path, value)


def predictions(path):
    rows = helpers.numeric_predictions(helpers.read_csv(path))
    return rows


def run_method(input_path, forms_path, output_path, artifacts_path, mode,
               calibration_path=None):
    cmd = [sys.executable, str(METHOD), '--input', str(input_path),
           '--training-forms', str(forms_path), '--lexicon',
           str(ROOT / 'data/lexicon.json'), '--mode', mode, '--output',
           str(output_path), '--artifacts', str(artifacts_path)]
    if calibration_path is not None:
        cmd.extend(['--calibration', str(calibration_path)])
    env = os.environ.copy()
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS'):
        env[name] = '2'
    started = time.perf_counter()
    subprocess.run(cmd, cwd=ROOT, env=env, check=True, capture_output=True,
                   text=True)
    elapsed = time.perf_counter() - started
    if not Path(output_path).is_file():
        raise RuntimeError(f'Method did not create predictions: {output_path}')
    return cmd, elapsed


def calibration_from_validation(raw_predictions, gold_rows):
    """Laplace correctness estimate by emitted T/F class; unsure is an error."""
    gold = {str(row['id']): row['label'] for row in gold_rows}
    bins = {label: {'count': 0, 'correct': 0, 'confidence': None}
            for label in ('true', 'false')}
    for row in raw_predictions:
        label = row.get('label')
        if label not in bins:
            continue
        bucket = bins[label]
        bucket['count'] += 1
        bucket['correct'] += int(gold.get(str(row.get('id'))) == label)
    for bucket in bins.values():
        bucket['confidence'] = (bucket['correct'] + 1) / (bucket['count'] + 2)
        bucket['confidence_status'] = 'validation_calibrated' if bucket['count'] else 'prior_only'
    return bins


def score_predictions(gold, prediction_path, out_dir):
    audit_started = time.perf_counter()
    preds = predictions(prediction_path)
    metrics, errors, scored = score_rows(gold, preds)
    helpers.write_csv(out_dir / 'scored_predictions.csv', scored,
                      ['id', 'gold', 'label', 'confidence', 'decision_reason',
                       'correct', 'output_valid', 'error_codes'])
    if not errors['eligible_for_comparison']:
        raise RuntimeError(f'Invalid method output: {errors}')
    return metrics, errors, scored, time.perf_counter() - audit_started


def safe_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True,
                        help='Unique run ID; existing output is never replaced')
    args = parser.parse_args()
    if (not args.run_id or any(c not in
            'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_'
            for c in args.run_id)):
        parser.error('run ID must contain only letters, digits, hyphens or underscores')
    if not METHOD.exists():
        parser.error(f'Method CLI is not available: {METHOD}')
    for file, expected in helpers.HASHES.items():
        actual = helpers.digest(ROOT / file)
        if actual != expected:
            parser.error(f'Benchmark hash mismatch for {file}: {actual}')

    kb_sources = tuple(str(path.relative_to(ROOT)) for folder in ('experiments', 'knowledge_base')
                       for path in sorted((ROOT / folder).rglob('*.py')))
    source_files = tuple(dict.fromkeys(
        file for file in (*SOURCE_FILES, *kb_sources) if (ROOT / file).is_file()))
    source_hashes = {file: helpers.digest(ROOT / file) for file in source_files}
    rows = helpers.read_csv(ROOT / 'data/practice.csv')
    manifest = json.loads((ROOT / 'evaluation/splits-v1.json').read_text())
    if len(rows) != len({row['id'] for row in rows}):
        parser.error('Practice IDs are not unique')
    row_by_id = {row['id']: row for row in rows}
    destinations = {method: ROOT / 'results/evaluations' / method / args.run_id
                    for method in METHODS}
    for destination in destinations.values():
        if destination.exists():
            parser.error(f'Output already exists: {destination}')
    for destination in destinations.values():
        destination.mkdir(parents=True)
    timings, split_ids, split_hashes, results, scored_by_method = {}, {}, {}, {}, {}
    validation_details = {}
    commands = []
    with tempfile.TemporaryDirectory(prefix='hype-kb-eval-v1-') as temporary:
        temp = Path(temporary)
        for track in TRACKS:
            split = manifest['splits'][track]
            train_ids, test_ids = split['train_ids'], split['test_ids']
            fit_ids, validation_ids = split['fit_ids'], split['validation_ids']
            if set(fit_ids) | set(validation_ids) != set(train_ids) or set(fit_ids) & set(validation_ids):
                raise ValueError(f'{track}: fit/validation IDs do not partition outer train')
            test_set, validation_set = set(test_ids), set(validation_ids)
            # Inner validation: fit evidence is fit_ids only. Validation rows are
            # pending and have blank labels in runtime-shaped input.
            validation_input = helpers.sanitized_rows(rows, fit_ids, validation_ids)
            validation_forms = read_forms(rows, fit_ids)
            val_input_path = temp / f'{track}-validation.csv'
            val_forms_path = temp / f'{track}-validation-forms.csv'
            write_csv(val_input_path, validation_input, ['id', 'text', 'date', 'label'])
            write_csv(val_forms_path, validation_forms,
                      ['id', 'text', 'date', 'label', 'form'])
            val_gold = [{'id': row['id'], 'label': row['label']}
                        for row in rows if row['id'] in validation_set]
            val_pred_path = temp / f'{track}-validation-predictions.csv'
            val_artifacts = temp / f'{track}-validation-artifacts'
            val_cmd, val_elapsed = run_method(
                val_input_path, val_forms_path, val_pred_path, val_artifacts,
                'kb_only')
            commands.append(val_cmd)
            val_preds = predictions(val_pred_path)
            validation_prediction_counts = Counter(
                row.get('label') for row in val_preds if row.get('label') in LABELS)
            val_metrics, val_errors, _ = score_rows(val_gold, val_preds)
            if not val_errors['eligible_for_comparison']:
                raise RuntimeError(f'{track} validation predictions invalid: {val_errors}')
            calibration = calibration_from_validation(val_preds, val_gold)
            validation_artifact_dir = destinations['kb_same_date'] / 'validation' / track
            validation_artifact_dir.mkdir(parents=True, exist_ok=True)
            # Keep raw method output and diagnostics in result storage, separate
            # from every method input. No gold or scored rows are written here.
            (validation_artifact_dir / 'predictions.csv').write_bytes(val_pred_path.read_bytes())
            for artifact_name in ('traces.jsonl', 'kb_records.jsonl', 'diagnostics.json'):
                artifact_source = val_artifacts / artifact_name
                if not artifact_source.is_file():
                    raise RuntimeError(f'Method artifact missing: {artifact_source}')
                (validation_artifact_dir / artifact_name).write_bytes(artifact_source.read_bytes())
            write_json(validation_artifact_dir / 'calibration.json', {
                'method': 'Laplace-smoothed empirical correctness by predicted label',
                'threshold': 0.5, 'true': calibration['true'], 'false': calibration['false']})
            validation_details[track] = {
                'fit_ids': fit_ids, 'validation_ids': validation_ids,
                'raw_kb_only_macro_f1': val_metrics['macro_f1'],
                'raw_kb_only_metrics': val_metrics,
                'calibration': calibration,
                'input_sha256': helpers.digest(val_input_path),
                'training_forms_sha256': helpers.digest(val_forms_path),
                'method_process_seconds': val_elapsed,
                'diagnostics': safe_json(validation_artifact_dir / 'diagnostics.json'),
                'artifact_sha256': {
                    name: helpers.digest(validation_artifact_dir / name)
                    for name in ('traces.jsonl', 'kb_records.jsonl', 'diagnostics.json', 'calibration.json')
                },
            }

            # Final test: refit on complete outer train, with only test labels blank.
            test_input = helpers.sanitized_rows(rows, train_ids, test_ids)
            test_forms = read_forms(rows, train_ids)
            test_input_path = temp / f'{track}-test.csv'
            test_forms_path = temp / f'{track}-test-forms.csv'
            write_csv(test_input_path, test_input, ['id', 'text', 'date', 'label'])
            write_csv(test_forms_path, test_forms,
                      ['id', 'text', 'date', 'label', 'form'])
            test_gold = [{'id': row['id'], 'label': row['label']}
                         for row in rows if row['id'] in test_set]
            split_ids[track] = {'fit_ids': fit_ids, 'validation_ids': validation_ids,
                                'outer_train_ids': train_ids, 'test_ids': test_ids}
            split_hashes[track] = {
                'validation_input': helpers.digest(val_input_path),
                'validation_training_forms': helpers.digest(val_forms_path),
                'test_input': helpers.digest(test_input_path),
                'test_training_forms': helpers.digest(test_forms_path),
            }
            calibration_obj = {
                'method': 'Laplace-smoothed empirical correctness by predicted label',
                'threshold': 0.5,
                'true': calibration['true'], 'false': calibration['false'],
            }
            for mode, method_id in (('kb_only', 'kb_same_date'),
                                    ('kb_hybrid', 'kb_hybrid')):
                out_dir = destinations[method_id] / 'fixed_050' / track
                out_dir.mkdir(parents=True)
                calibration_path = out_dir / 'calibration.json'
                write_json(calibration_path, calibration_obj)
                prediction_path = out_dir / 'predictions.csv'
                artifacts_path = temp / f'{track}-{mode}-artifacts'
                cmd, elapsed = run_method(test_input_path, test_forms_path,
                                          prediction_path, artifacts_path, mode,
                                          calibration_path)
                commands.append(cmd)
                metrics, errors, scored, audit_seconds = score_predictions(
                    test_gold, prediction_path, out_dir)
                # Artifact bundle is method-generated and contains no evaluator gold.
                for name in ('traces.jsonl', 'kb_records.jsonl', 'diagnostics.json'):
                    src = artifacts_path / name
                    if not src.is_file():
                        raise RuntimeError(f'Method artifact missing: {src}')
                    (out_dir / name).write_bytes(src.read_bytes())
                artifact_hashes = {
                    name: helpers.digest(out_dir / name)
                    for name in ('traces.jsonl', 'kb_records.jsonl', 'diagnostics.json', 'calibration.json')
                }
                split_hashes[track].setdefault('exported_artifacts_sha256', {})[method_id] = artifact_hashes
                diagnostics = safe_json(out_dir / 'diagnostics.json') or {}
                test_prediction_counts = Counter(
                    row.get('label') for row in predictions(prediction_path)
                    if row.get('label') in LABELS)
                refit_distribution = {
                    label: {'validation_raw_count': validation_prediction_counts[label],
                            'test_refit_count': test_prediction_counts[label],
                            'validation_raw_share': (validation_prediction_counts[label] / sum(validation_prediction_counts.values())
                                                     if sum(validation_prediction_counts.values()) else None),
                            'test_refit_share': (test_prediction_counts[label] / sum(test_prediction_counts.values())
                                                 if sum(test_prediction_counts.values()) else None)}
                    for label in LABELS
                }
                metric_record = {
                    'benchmark_version': 'arena-eval-v1', 'method_id': method_id,
                    'run_id': args.run_id, 'split': track, 'policy': 'fixed_050',
                    'access_profile': 'batch_unlabelled_v1',
                    'eligible_for_comparison': errors['eligible_for_comparison'],
                    'output_errors': errors,
                    'artifact_sha256': artifact_hashes,
                    'classification': {'status': 'ok', **metrics},
                    'generation': {'status': 'not_implemented',
                                   'reason': 'Classification-only method',
                                   'verified_valid_yield_at_1000': None},
                    'proofs': {'status': 'not_verified',
                               'reason': 'Traces are exposed but no independent proof verifier was run',
                               'decision_proof_rate': None,
                               'grounded_decision_yield': None},
                    'resources': {
                        'status': 'not_run',
                        'reason': 'Native macOS timing only; constrained Linux benchmark not run',
                        'method_process_seconds': elapsed,
                        'peak_ram_bytes': None,
                        'fit_seconds': diagnostics.get('fit_seconds'),
                        'inference_seconds': diagnostics.get('inference_seconds'),
                        'serialization_seconds': diagnostics.get('serialization_seconds'),
                        'audit_seconds': audit_seconds,
                    },
                }
                write_json(out_dir / 'metrics.json', metric_record)
                results.setdefault(method_id, {})[track] = metric_record
                scored_by_method.setdefault(method_id, {})[track] = scored
                timings.setdefault(method_id, {})[track] = {
                    'validation_kb_only_process_seconds': val_elapsed,
                    'test_method_process_seconds': elapsed,
                    'evaluator_audit_seconds': audit_seconds,
                    'refit_prediction_distribution': refit_distribution,
                    'validation_calibration_counts': {
                        label: calibration[label]['count'] for label in ('true', 'false')},
                    'calibration_reliability_claim': 'not_established; small validation decision counts',
                    **diagnostics,
                }
                # Cross-model method metadata uses only the exact evaluator
                # command with sanitized files and the frozen IDs.
                print(f'{method_id:13} {track:10} macro_f1={metrics["macro_f1"]:.3f} '
                      f'coverage={metrics["decided_coverage"]:.3f}', flush=True)

    # Record dependency versions and file hashes; sklearn is used only by the
    # predeclared hybrid fallback, not by the KB evidence store.
    import numpy, scipy, sklearn, joblib, threadpoolctl
    versions = {'python': platform.python_version(), 'numpy': numpy.__version__,
                'scipy': scipy.__version__, 'scikit-learn': sklearn.__version__,
                'joblib': joblib.__version__, 'threadpoolctl': threadpoolctl.__version__}
    for method_id in METHODS:
        run_dir = destinations[method_id]
        run = {
            'benchmark_version': 'arena-eval-v1', 'method_id': method_id,
            'run_id': args.run_id, 'seed': 42, 'splits': list(TRACKS),
            'policies': ['fixed_050'], 'pinned_sha256': helpers.HASHES,
            'source_commit': helpers.source_version(),
            'source_file_sha256': source_hashes, 'dependencies': versions,
            'exact_reproduction_command': f'.venv/bin/python evaluation/run_kb_experiment.py --run-id {args.run_id}',
            'method_commands': commands,
            'training_sources': ('Split-permitted labelled rows and their forms, plus public lexicon. '
                                 'Validation calibration uses raw kb_only predictions on frozen validation rows; '
                                 'test refit uses the complete outer training pool.'),
            'training_ids': {track: split_ids[track]['outer_train_ids'] for track in TRACKS},
            'fit_ids': {track: split_ids[track]['fit_ids'] for track in TRACKS},
            'validation_ids': {track: split_ids[track]['validation_ids'] for track in TRACKS},
            'test_ids': {track: split_ids[track]['test_ids'] for track in TRACKS},
            'evidence_access_profile': 'batch_unlabelled_v1',
            'access_profile': 'batch_unlabelled_v1',
            'factual_asset_provenance': 'Rebuilt per split from permitted training-form rows and lexicon; no full-practice form cache.',
            'confidence': {
                'semantics': ('Laplace-smoothed validation empirical correctness by emitted true/false label; '
                              'hybrid unresolved fallback confidence is uncalibrated model probability.'),
                'calibration': 'kb_only validation predictions on validation_ids; gold unsure is an error for T/F decisions.',
                'threshold': 0.5, 'policy': 'fixed_050',
                'calibration_fit_ids': {track: split_ids[track]['fit_ids'] for track in TRACKS},
                'calibration_validation_ids': {track: split_ids[track]['validation_ids'] for track in TRACKS},
                'refit_ids': {track: split_ids[track]['outer_train_ids'] for track in TRACKS},
                'no_threshold_selection': True,
                'reliability_claim': 'No reliability guarantee; report empirical estimates with validation counts and prior-only status.',
            },
            'leakage_audit': {
                'status': 'pass_logical_boundary',
                'detail': 'Separate method subprocess receives sanitized runtime CSV and train-only forms CSV. Gold test labels/forms remain evaluator-side; no OS filesystem sandbox is enforced.',
            },
            'added_training_material': 'No external facts or pretrained factual assets; hybrid uses predeclared word_char_lr_balanced text model.',
            'assumptions': ['Same-date conservative evidence only; no time carry-forward.',
                            'Hybrid fallback is used only for unresolved KB rows.'],
            'split_input_sha256': split_hashes,
            'validation_calibration': validation_details,
            'timings': timings.get(method_id, {}),
            'hardware': {'os': platform.platform(), 'machine': platform.machine(),
                         'cpu': platform.processor(), 'cpu_limit_enforced': None,
                         'memory_limit_bytes': None, 'process_limit': None,
                         'thread_limit': 2, 'timing': 'time.perf_counter; native host'},
            'capabilities': {'classification': 'implemented', 'generation': 'not_implemented',
                             'proof_traces': 'available_unverified',
                             'independent_proof_verifier': 'not_run',
                             'constrained_container_benchmark': 'not_run',
                             'frontend': 'not_implemented'},
            'same_date_limitations': 'Conservative same-date evidence provides limited coverage; no temporal persistence or carry-forward is assumed.',
            'reproduction_note': 'Validation gold is held in evaluator memory and used only to fit the per-label Laplace confidence JSON. It is never written under method input or artifacts.',
        }
        write_json(run_dir / 'run.json', run)
        make_report(run_dir / 'report.md', method_id, results[method_id],
                    scored_by_method[method_id], args.run_id, validation_details)


def fmt(value):
    return 'n/a' if value is None else f'{value:.3f}'


def make_report(path, method_id, metrics_by_track, scored_by_track, run_id,
                validation_details):
    if method_id == 'kb_same_date':
        description = ('Same-date KB reasoning only: split-scoped training forms and lexicon build a '
                       'conservative evidence store. No time carry-forward is used.')
    else:
        description = ('Same-date KB reasoning with the predeclared word_char_lr_balanced classifier '
                       'for unresolved rows. Fallback classifier confidence is uncalibrated.')
    lines = [f'# {method_id}: arena-eval-v1', '', description, '',
             'The confidence calibration is fixed before test inference: validation raw KB decisions are grouped by predicted true/false label, and Laplace-smoothed correctness is applied at the fixed 0.50 threshold. No test labels or forms reach the method process. These are development comparisons; the practice corpus and related templates have informed prior work.', '',
             '| Split | Macro F1 | False precision / recall | Unsure F1 | Coverage | Errors / decided | Risk | Brier | Runtime | Peak RAM |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for track in TRACKS:
        m = metrics_by_track[track]['classification']
        r = metrics_by_track[track]['resources']
        p = m['per_class']
        lines.append(f"| {track} | {fmt(m['macro_f1'])} | {fmt(p['false']['precision'])} / {fmt(p['false']['recall'])} | {fmt(p['unsure']['f1'])} | {fmt(m['decided_coverage'])} | {m['decided_errors']} / {m['decided_count']} | {fmt(m['decided_risk'])} | {fmt(m['correctness_brier'])} | {r['method_process_seconds']:.2f}s native | not measured |")
    lines += ['', '## Validation calibration counts', '',
              'Laplace smoothing gives a 0.50 prior-only confidence for any predicted class with no raw validation KB decisions. Sparse calibration counts are descriptive point estimates; they do not establish reliability.', '',
              '| Split | Raw KB validation T/F decisions | True count | False count | Calibration note |',
              '| --- | ---: | ---: | ---: | --- |']
    for track in TRACKS:
        cal = validation_details[track]['calibration']
        total = cal['true']['count'] + cal['false']['count']
        note = ('No raw KB validation decisions were observed; both classes use prior-only 0.50 confidence.'
                if total == 0 else
                ', '.join(f'{label} prior-only at 0.50' for label in ('true', 'false')
                          if cal[label]['count'] == 0) or 'Sparse empirical estimates; reliability not established.')
        lines.append(f"| {track} | {total} | {cal['true']['count']} | {cal['false']['count']} | {note} |")
    lines += ['', 'Confusion matrices and reliability bins are in each track’s `metrics.json`. Grounded yield and proof acceptance are not measured: traces are available, but no independent proof verifier was run. Generation is not implemented.', '',
              '## Matched controls', '',
              'Controls are reported only when fixed_050 artifacts from `text-variants-20261028-001` exist for the matching track. Their access and benchmark metadata are checked before comparison.', '',
              '| Split | This method macro F1 | word_char_lr_balanced | tfidf_lr_balanced |',
              '| --- | ---: | ---: | ---: |']
    for track in TRACKS:
        vals = []
        for control in ('word_char_lr_balanced', 'tfidf_lr_balanced'):
            candidate = ROOT / 'results/evaluations' / control / 'text-variants-20261028-001' / 'fixed_050' / track / 'metrics.json'
            try:
                metric = json.loads(candidate.read_text())
                val = metric['classification']['macro_f1'] if metric.get('benchmark_version') == 'arena-eval-v1' and metric.get('policy') == 'fixed_050' and metric.get('access_profile') == 'batch_unlabelled_v1' else None
            except (OSError, ValueError, KeyError, TypeError):
                val = None
            vals.append(fmt(val))
        lines.append(f"| {track} | {fmt(metrics_by_track[track]['classification']['macro_f1'])} | {vals[0]} | {vals[1]} |")
    lines += ['', '## Representative errors', '',
              'The selected rows document emitted labels and confidence. A causal parsing/evidence diagnosis is not inferred from aggregate metrics; inspect the matching trace record for the executed rule and evidence references.', '',
              '| ID | Gold | Predicted | Confidence | Reason |', '| --- | --- | --- | ---: | --- |']
    failures = [row for row in scored_by_track['stratified'] if not row['correct']]
    failures.sort(key=lambda row: (row['gold'] == 'true', row['id']))
    if len(failures) < 5:
        failures.extend(row for track in ('temporal', 'checker')
                        for row in scored_by_track[track] if not row['correct'])
    for row in failures[:5]:
        lines.append(f"| {row['id']} | {row['gold']} | {row['label']} | {fmt(row['confidence'])} | {row['decision_reason']} |")
    lines += ['', '## Limitations and reproduction', '',
              'The same-date KB deliberately declines to carry observations across dates, so coverage is limited. The temporal track is a forward batch stress test. These are not untouched generalization test claims: the practice corpus and related templates have informed development. This is classification-only, not submission-ready. Native host timings are supplementary; constrained Linux resources, generation, frontend readiness and independent proof verification were not measured. No complete-practice form cache is available to inference.', '',
              f'```sh\n.venv/bin/python evaluation/run_kb_experiment.py --run-id NEW-UNIQUE-ID\n```', '',
              'Test gold labels/forms remain in the evaluator and are joined by ID after subprocess inference. Validation gold is used only to calculate the documented confidence calibration. The operating threshold is fixed at 0.50; no threshold selection is performed.', '']
    Path(path).write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
