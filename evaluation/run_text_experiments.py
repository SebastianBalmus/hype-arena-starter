#!/usr/bin/env python3
"""Run predeclared text classifiers under arena-eval-v1 fixed_050.

The evaluator reads gold data. A separate method process receives only a
split-specific runtime-shaped CSV, with blank test labels and no form column.
This is a logical input boundary, not an OS-enforced filesystem sandbox.
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
from evaluation.scoring import LABELS, score_rows
from experiments.text_variants import MODELS

HASHES = {
    'data/practice.csv': '52743befaa0fa38b89b5c802c0f953d242ae027c86839c9c0f05a9e69a86d8e9',
    'data/lexicon.json': 'd0fd5e9bdc6b33ad3ec160001f50e00e7f8bf2b25a92779022f893db7c3d5507',
    'evaluation/splits-v1.json': '9699327af7bf80813eabb66a582593d5152683ee98abf36090a252b0b2723860',
}
TRACKS = ('stratified', 'temporal', 'checker')
CONTROLS = ('always_true', 'tfidf_lr', 'tfidf_lr_balanced')
SOURCE_FILES = ('baseline.py', 'experiments/text_variants.py', 'evaluation/scoring.py',
                'evaluation/run_text_experiments.py', 'requirements.txt', 'EVALUATION.md')
DESCRIPTIONS = {
    'always_true': 'Majority-prior control, always true on these splits.',
    'tfidf_lr': 'Word unigrams/bigrams with unweighted logistic regression.',
    'tfidf_lr_balanced': 'Word unigrams/bigrams with class-balanced logistic regression.',
    'char_lr_balanced': 'Character 3–5-gram TF-IDF with class-balanced logistic regression.',
    'word_char_lr_balanced': 'Word plus character TF-IDF with class-balanced logistic regression.',
    'word_char_lr_unweighted': 'Word plus character TF-IDF with unweighted logistic regression.',
    'word_char_complement_nb': 'Word plus character TF-IDF with complement naive Bayes.',
    'word_lr_unsure_boost': 'Word TF-IDF and balanced logistic regression with twice the unsure weight.',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f'Empty CSV: {path}')
        return list(reader)


def write_csv(path, rows, fields):
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def numeric_predictions(rows):
    for row in rows:
        for key in ('confidence', 'p_true', 'p_false', 'p_unsure'):
            val = row.get(key, '')
            if val == '':
                row[key] = None
            else:
                try:
                    row[key] = float(val)
                except ValueError:
                    pass  # The scorer records invalid confidence/probability.
    return rows


def sanitized_rows(rows, train_ids, test_ids):
    """Use manifest IDs and original order; never include gold forms or test labels."""
    train, test = set(train_ids), set(test_ids)
    if train & test:
        raise ValueError('Split has overlapping train/test IDs')
    if len(train) != len(train_ids) or len(test) != len(test_ids):
        raise ValueError('Duplicate IDs in split manifest')
    result = [{key: ('' if key == 'label' and row['id'] in test else row[key])
               for key in ('id', 'text', 'date', 'label')}
              for row in rows if row['id'] in train | test]
    if {r['id'] for r in result} != train | test:
        raise ValueError('Split includes an ID absent from the dataset')
    if any(r['label'] for r in result if r['id'] in test):
        raise ValueError('Test label escaped input sanitization')
    return result


def source_version():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except (subprocess.SubprocessError, OSError):
        return None


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def percent(value):
    return 'n/a' if value is None else f'{value:.3f}'


def make_report(model_name, run_id, metrics_by_model, scored_by_split, command):
    lines = [f'# {model_name}: arena-eval-v1', '', DESCRIPTIONS[model_name], '',
             'Classification-only, text-only experiment. Fixed decision policy: `fixed_050`. '
             'All models were specified before looking at these results; no threshold or '
             'hyperparameter was selected using outer-test labels.', '',
             '| Split | Macro F1 | False precision / recall | Unsure F1 | Coverage | '
             'Decided errors / count | Risk | Correctness Brier | Grounded yield | '
             'Generation / verified yield | Proof / rate | Runtime / peak RAM |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |']
    for track in TRACKS:
        m = metrics_by_model[model_name][track]
        c, r = m['classification'], m['resources']
        p = c['per_class']
        lines.append(f"| {track} | {percent(c['macro_f1'])} | {percent(p['false']['precision'])} / "
                     f"{percent(p['false']['recall'])} | {percent(p['unsure']['f1'])} | "
                     f"{percent(c['decided_coverage'])} | {c['decided_errors']} / {c['decided_count']} | "
                     f"{percent(c['decided_risk'])} | {percent(c['correctness_brier'])} | n/a | "
                     f"not implemented / n/a | not implemented / n/a | "
                     f"{r['method_process_seconds']:.2f}s macOS / not measured |")
    lines += ['', 'Confusion matrices and reliability bins are in the per-split '
              '[metrics files](fixed_050/stratified/metrics.json); the temporal and checker '
              'files are in their respective sibling directories. Native macOS timings '
              'are supplementary, not constrained Linux measurements.', '',
              '## Matched controls', '',
              '| Split | This macro F1 | Always true | Word LR | Balanced word LR | '
              'Difference vs balanced word LR |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for track in TRACKS:
        f1 = metrics_by_model[model_name][track]['classification']['macro_f1']
        controls = [metrics_by_model[c][track]['classification']['macro_f1'] for c in CONTROLS]
        lines.append(f'| {track} | {f1:.3f} | {controls[0]:.3f} | {controls[1]:.3f} | '
                     f'{controls[2]:.3f} | {f1-controls[2]:+.3f} |')
    failures = [r for r in scored_by_split['stratified'] if not r['correct']]
    # Include minority classes before majority errors, without selecting new hyperparameters.
    failures.sort(key=lambda r: (r['gold'] == 'true', r['id']))
    if len(failures) < 5:
        failures.extend(r for track in ('temporal', 'checker') for r in scored_by_split[track]
                        if not r['correct'])
    lines += ['', '## Representative errors', '',
              'These classifiers do not parse assertions or consult evidence and dates. '
              'The table records the actual emitted reason; determining the factual cause '
              'would require a separate sentence-level review.', '',
              '| ID | Gold | Predicted | Confidence | Decision reason |',
              '| --- | --- | --- | ---: | --- |']
    for row in failures[:5]:
        lines.append(f"| {row['id']} | {row['gold']} | {row['label']} | "
                     f"{percent(row['confidence'])} | {row['decision_reason']} |")
    lines += ['', '## Limitations and reproduction', '',
              'Practice rows have already informed development, and related sentence templates '
              'may cross partitions. These are development results, not pristine generalization '
              'estimates. The temporal track is a forward batch stress test. Confidence scores '
              'are uncalibrated. Models supply no evidence proofs, claim generation, Docker '
              'benchmark, or submission-ready system.', '',
              'Inference inputs were produced with test labels blank and forms absent. The '
              'method is a separate process that reads the sanitized input; the host does '
              'not enforce a filesystem sandbox against other workspace files.', '',
              f'Run the following with a **new** run ID; original outputs are never overwritten:', '',
              f'```sh\n{command}\n```', '',
              'Development guidance: Kassis, Agarwal, He, Patel and Brueckner (2026), '
              '[Scientific Agent Skills](https://doi.org/10.48550/arXiv.2609.00065).', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True, help='Unique run ID; existing output is never replaced')
    args = parser.parse_args()
    if not args.run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in args.run_id):
        parser.error('run ID must contain only letters, digits, hyphens or underscores')
    for file, expected in HASHES.items():
        actual = digest(ROOT / file)
        if actual != expected:
            parser.error(f'Benchmark hash mismatch for {file}: {actual}')
    rows = read_csv(ROOT / 'data/practice.csv')
    manifest = json.loads((ROOT / 'evaluation/splits-v1.json').read_text())
    if len(rows) != len({r['id'] for r in rows}):
        parser.error('Practice IDs are not unique')
    destinations = {name: ROOT / 'results/evaluations' / name / args.run_id for name in MODELS}
    for path in destinations.values():
        if path.exists():
            parser.error(f'Output already exists: {path}')
    source_hashes = {file: digest(ROOT / file) for file in SOURCE_FILES}
    python_version = platform.python_version()
    import numpy, scipy, sklearn, joblib, threadpoolctl
    versions = {'python': python_version, 'numpy': numpy.__version__, 'scipy': scipy.__version__,
                'scikit-learn': sklearn.__version__, 'joblib': joblib.__version__,
                'threadpoolctl': threadpoolctl.__version__}
    results = {model: {} for model in MODELS}
    all_scored = {model: {} for model in MODELS}
    command = f'.venv/bin/python evaluation/run_text_experiments.py --run-id {args.run_id}'
    with tempfile.TemporaryDirectory(prefix='hype-eval-v1-') as tmp:
        tmp = Path(tmp)
        for track in TRACKS:
            split = manifest['splits'][track]
            train_ids, test_ids = split['train_ids'], split['test_ids']
            sanitized = sanitized_rows(rows, train_ids, test_ids)
            test_set = set(test_ids)
            gold = [{'id': r['id'], 'label': r['label']} for r in rows if r['id'] in test_set]
            input_path = tmp / f'{track}.csv'
            write_csv(input_path, sanitized, ['id', 'text', 'date', 'label'])
            input_hash = digest(input_path)
            for name in MODELS:
                out_dir = destinations[name] / 'fixed_050' / track
                out_dir.mkdir(parents=True)
                pred_path, timing_path = out_dir / 'predictions.csv', tmp / f'{track}-{name}-timing.json'
                method_command = [sys.executable, str(ROOT / 'experiments/text_variants.py'),
                                  '--input', str(input_path), '--model', name, '--output', str(pred_path),
                                  '--seed', '42', '--timings', str(timing_path)]
                env = os.environ.copy()
                for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
                    env[key] = '2'
                started = time.perf_counter()
                subprocess.run(method_command, cwd=ROOT, env=env, check=True, capture_output=True, text=True)
                process_seconds = time.perf_counter() - started
                predictions = numeric_predictions(read_csv(pred_path))
                audit_start = time.perf_counter()
                classification, errors, scored = score_rows(gold, predictions)
                audit_seconds = time.perf_counter() - audit_start
                write_csv(out_dir / 'scored_predictions.csv', scored,
                          ['id', 'gold', 'label', 'confidence', 'decision_reason', 'correct',
                           'output_valid', 'error_codes'])
                timings = json.loads(timing_path.read_text())
                metrics = {
                    'benchmark_version': 'arena-eval-v1', 'method_id': name, 'run_id': args.run_id,
                    'split': track, 'policy': 'fixed_050', 'access_profile': 'batch_unlabelled_v1',
                    'eligible_for_comparison': errors['eligible_for_comparison'], 'output_errors': errors,
                    'classification': {'status': 'ok', **classification},
                    'generation': {'status': 'not_implemented', 'reason': 'Classification-only method',
                                   'verified_valid_yield_at_1000': None},
                    'proofs': {'status': 'not_implemented', 'reason': 'No evidence proof functionality',
                               'decision_proof_rate': None, 'grounded_decision_yield': None},
                    'resources': {'status': 'not_run', 'reason': 'Constrained offline Linux amd64 benchmark not run',
                                  'fit_seconds_native': timings['fit_seconds'],
                                  'inference_seconds_native': timings['inference_seconds'],
                                  'serialization_seconds_native': timings['serialization_seconds'],
                                  'external_audit_seconds_native': audit_seconds,
                                  'method_process_seconds': process_seconds, 'peak_ram_bytes': None},
                }
                write_json(out_dir / 'metrics.json', metrics)
                results[name][track], all_scored[name][track] = metrics, scored
                print(f"{track:10} {name:26} macro_f1={classification['macro_f1']:.3f} "
                      f"false_recall={classification['per_class']['false']['recall']:.3f} "
                      f"eligible={errors['eligible_for_comparison']}", flush=True)
                if not errors['eligible_for_comparison']:
                    raise RuntimeError(f'{name}/{track} output failed comparison audit: {errors}')
                split.setdefault('_input_hash', input_hash)
    for name, destination in destinations.items():
        run = {
            'benchmark_version': 'arena-eval-v1', 'pinned_sha256': HASHES,
            'method_id': name, 'run_id': args.run_id, 'seed': 42,
            'splits': list(TRACKS), 'policies': ['fixed_050'], 'exact_reproduction_command': command,
            'method_command_template': 'python experiments/text_variants.py --input <sanitized-split.csv> '
                                       f'--model {name} --output <predictions.csv> --seed 42 --timings <timings.json>',
            'source_commit': source_version(), 'source_file_sha256': source_hashes,
            'dependencies': versions, 'model_asset_sha256': None,
            'model_asset_reason': 'Model is fitted in memory for each split and not serialized',
            'training_sources': 'Split training IDs, text and label only; date and ID are not features. '
                                'Practice forms and lexicon are not accessed by the method.',
            'training_ids': {s: manifest['splits'][s]['train_ids'] for s in TRACKS},
            'test_ids': {s: manifest['splits'][s]['test_ids'] for s in TRACKS},
            'sanitized_input_sha256': {s: manifest['splits'][s]['_input_hash'] for s in TRACKS},
            'access_profile': 'batch_unlabelled_v1',
            'factual_assets': 'None; text classifier does not establish world facts.',
            'leakage_audit': {'status': 'pass_logical_boundary',
                              'detail': 'Gold labels and forms removed from the method input; '
                                        'separate process. No OS-enforced workspace filesystem isolation.'},
            'confidence': {'semantics': 'uncalibrated class probability of winning label',
                           'calibration': 'none', 'threshold': 0.5, 'policy': 'fixed_050',
                           'selection': 'fixed by protocol, not tuned on test'},
            'hardware': {'os': platform.platform(), 'machine': platform.machine(),
                         'cpu': platform.processor(), 'thread_limit': 2, 'cpu_limit_enforced': None,
                         'memory_limit_bytes': None, 'process_limit': None,
                         'timing': 'perf_counter; process wall time includes startup, fit, inference and writing'},
            'capabilities': {'classification': 'implemented', 'generation': 'not_implemented',
                             'proofs': 'not_implemented', 'independent_truth_verifier': 'not_run',
                             'constrained_container_benchmark': 'not_run', 'frontend': 'not_implemented'},
        }
        write_json(destination / 'run.json', run)
        (destination / 'report.md').write_text(
            make_report(name, args.run_id, results, all_scored[name], command), encoding='utf-8')
    print('\nAll classification results are development comparisons, not final generalization scores.')


if __name__ == '__main__':
    main()
