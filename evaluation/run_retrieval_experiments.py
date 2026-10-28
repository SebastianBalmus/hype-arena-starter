#!/usr/bin/env python3
"""Compare predeclared cross-fitted retrieval features on frozen splits."""

import argparse
import json
import math
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
from evaluation import run_text_experiments as shared
from evaluation.scoring import score_rows

CONTROL = 'word_char_lr_balanced'
CANDIDATES = (CONTROL, 'retrieval_raw', 'retrieval_raw_masked')
SELECTED = 'retrieval_selected'
ARTIFACTS = ('traces.jsonl', 'oof_traces.jsonl', 'oof_folds.json',
             'feature_names.json', 'diagnostics.json')


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def audit_retrieval_artifacts(directory, train_rows, pending_ids):
    """Verify fold membership and every exposed retrieval label independently."""
    train_by_id = {r['id']: r for r in train_rows}
    train_ids, pending_ids = set(train_by_id), set(pending_ids)
    folds = json.loads((directory / 'oof_folds.json').read_text())
    fold_by_id, held_out = {}, []
    for fold in folds:
        index = fold['fold_index']
        if index in fold_by_id:
            raise ValueError('Duplicate OOF fold index')
        query, refs = fold['query_ids'], fold['reference_ids']
        if (len(query) != len(set(query)) or len(refs) != len(set(refs))
                or set(query) & set(refs) or set(query) | set(refs) != train_ids):
            raise ValueError('OOF fold does not partition permitted training IDs')
        fold_by_id[index] = fold
        held_out.extend(query)
    if Counter(held_out) != Counter(train_ids):
        raise ValueError('OOF folds must hold out every training row exactly once')

    def neighbors(trace, allowed, key):
        count = 0
        for view in trace[key].values():
            for neighbor in view:
                rid = neighbor['id']
                similarity = neighbor['similarity']
                if (rid not in allowed or neighbor['label'] != train_by_id[rid]['label']
                        or neighbor['date'] != train_by_id[rid]['date']
                        or not math.isfinite(similarity) or not 0 < similarity <= 1.000001):
                    raise ValueError('Neighbor label, ID or similarity failed retrieval audit')
                count += 1
        for classes in trace.get('class_neighbors', {}).values():
            for group in classes.values():
                for neighbor in group:
                    rid = neighbor['id']
                    if (rid not in allowed or neighbor['label'] != train_by_id[rid]['label']
                            or neighbor['date'] != train_by_id[rid]['date']
                            or not math.isfinite(neighbor['similarity'])
                            or not 0 < neighbor['similarity'] <= 1.000001):
                        raise ValueError('Per-class neighbor failed retrieval boundary audit')
                    count += 1
        return count

    oof = read_jsonl(directory / 'oof_traces.jsonl')
    if Counter(t['query_id'] for t in oof) != Counter(train_ids):
        raise ValueError('OOF traces must cover each training ID exactly once')
    citations = 0
    for trace in oof:
        fold = fold_by_id[trace['fold_index']]
        if trace['query_id'] not in set(fold['query_ids']):
            raise ValueError('OOF trace assigned to wrong fold')
        citations += neighbors(trace, set(fold['reference_ids']), 'neighbors')
    pending = read_jsonl(directory / 'traces.jsonl')
    if Counter(t['query_id'] for t in pending) != Counter(pending_ids):
        raise ValueError('Pending traces do not match canonical held-out IDs')
    for trace in pending:
        citations += neighbors(trace, train_ids, 'views')
    return {'status': 'pass', 'oof_rows': len(oof), 'pending_rows': len(pending),
            'folds': len(folds), 'neighbor_references_checked': citations,
            'self_or_excluded_fold_neighbors': 0}


def run_candidate(name, input_path, output_dir):
    output_dir.mkdir(parents=True)
    predictions = output_dir / 'predictions.csv'
    if name == CONTROL:
        command = [sys.executable, str(ROOT / 'experiments/text_variants.py'),
                   '--input', str(input_path), '--model', CONTROL,
                   '--output', str(predictions), '--seed', '42',
                   '--timings', str(output_dir / 'diagnostics.json')]
    else:
        variant = 'raw' if name == 'retrieval_raw' else 'raw_masked'
        command = [sys.executable, str(ROOT / 'experiments/retrieval_predict.py'),
                   '--input', str(input_path), '--lexicon', str(ROOT / 'data/lexicon.json'),
                   '--variant', variant, '--output', str(predictions),
                   '--artifacts', str(output_dir), '--seed', '42']
    env = os.environ.copy()
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS'):
        env[key] = '2'
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    if completed.returncode:
        raise RuntimeError(f'{name} failed: {completed.stderr[-5000:]}')
    if name != CONTROL:
        for artifact in ARTIFACTS:
            if not (output_dir / artifact).is_file():
                raise ValueError(f'Missing method artifact: {artifact}')
    return command, elapsed


def score_output(directory, gold, train_rows, name, run_id, track, phase, elapsed):
    predictions = shared.numeric_predictions(shared.read_csv(directory / 'predictions.csv'))
    started = time.perf_counter()
    metrics, errors, scored = score_rows(gold, predictions)
    if not errors['eligible_for_comparison']:
        raise ValueError(f'Invalid {name}/{phase}/{track} outputs: {errors}')
    audit = ({'status': 'not_applicable', 'reason': 'Text-only control has no retrieval features'}
             if name == CONTROL else
             audit_retrieval_artifacts(directory, train_rows, [r['id'] for r in gold]))
    audit_seconds = time.perf_counter() - started
    shared.write_csv(directory / 'scored_predictions.csv', scored,
                     ['id', 'gold', 'label', 'confidence', 'decision_reason', 'correct',
                      'output_valid', 'error_codes'])
    diagnostics = json.loads((directory / 'diagnostics.json').read_text())
    timing = diagnostics.get('timings_seconds', diagnostics)
    record = {
        'benchmark_version': 'arena-eval-v1', 'method_id': name, 'run_id': run_id,
        'split': track, 'phase': phase, 'policy': 'fixed_050',
        'access_profile': 'batch_unlabelled_v1', 'eligible_for_comparison': True,
        'output_errors': errors, 'retrieval_boundary_audit': audit,
        'classification': {'status': 'ok', **metrics},
        'generation': {'status': 'not_implemented', 'reason': 'Classification-only method',
                       'verified_valid_yield_at_1000': None},
        'proofs': {'status': 'not_implemented',
                   'reason': 'Neighbor traces are model features, not factual proofs',
                   'decision_proof_rate': None, 'grounded_decision_yield': None},
        'resources': {'status': 'not_run', 'reason': 'Native timing; constrained Linux run not measured',
                      'method_process_seconds': elapsed, 'audit_seconds_native': audit_seconds,
                      'fit_seconds_native': timing.get('fit', timing.get('fit_seconds')),
                      'inference_seconds_native': timing.get('inference', timing.get('inference_seconds')),
                      'serialization_seconds_native': timing.get('serialization', timing.get('serialization_seconds')),
                      'peak_ram_bytes': None},
    }
    shared.write_json(directory / 'metrics.json', record)
    return record


def report(destination, name, results, selections, run_id):
    lines = [f'# {name}: retrieval-feature experiment', '',
             'Frozen arena-eval-v1 development splits; fixed_050; seed 42. '
             'All candidates were declared before their outer-test scores were computed. '
             'Prior experiments have already informed development on these splits.', '',
             '| Track | Validation macro F1 | Test macro F1 | False precision / recall | Unsure F1 | Coverage | T/F errors / decisions |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for track in shared.TRACKS:
        source = selections[track]['selected_model'] if name == SELECTED else name
        validation = results[source]['validation'][track]['classification']
        test = results[source]['test'][track]['classification']
        false = test['per_class']['false']
        lines.append(f"| {track} | {validation['macro_f1']:.3f} | {test['macro_f1']:.3f} | "
                     f"{false['precision']:.3f} / {false['recall']:.3f} | "
                     f"{test['per_class']['unsure']['f1']:.3f} | {test['decided_coverage']:.3f} | "
                     f"{test['decided_errors']} / {test['decided_count']} |")
    lines += ['', '## Method and boundaries', '',
              'The unchanged control uses balanced word/character TF-IDF logistic regression. '
              'Retrieval candidates append standardized neighbor features learned from five '
              'stratified folds within the permitted training pool. Each training query receives '
              'references and vocabulary fitted on its other four folds only. A full training '
              'index is used only for pending-row inference. Query labels and forms are never '
              'retrieval inputs. Numeric agreement and similarity are heuristics; neither establishes truth.', '',
              'The raw candidate retrieves on original wording. The raw_masked candidate adds '
              'a second retrieval view with numeric surfaces replaced by a placeholder. '
              'Nearest-neighbor labels, date gaps, entity overlap, numeric overlap and '
              'currency/period matches are supplied to balanced logistic regression at C=1. '
              'Class probabilities are uncalibrated; the fixed 0.50 gate applies after prediction.', '',
              'Inner fit/validation predictions select among the unchanged control and both '
              'retrieval candidates. Selection is recorded before test inference. Each candidate '
              'is then refitted on the complete outer training pool. The selected method reuses '
              'that candidate output rather than fitting it twice. Validation/test text are '
              'only pending queries; no pending labels or forms enter training.', '',
              'Inputs are sanitized and method subprocesses are separate. Filesystem access '
              'is not OS-enforced. Related templates may cross the frozen partitions. '
              'Generation, accepted proofs, frontend and constrained runtime are unavailable.', '',
              '## Validation selection', '', '| Track | Selected candidate |', '| --- | --- |']
    for track, selection in selections.items():
        lines.append(f"| {track} | {selection['selected_model']} |")
    lines += ['', 'Neighbor traces, fold membership and feature names are stored with the '
              'retrieval candidates. The evaluator audits every exposed neighbor against '
              'the permitted reference IDs and labels. The selected method references those '
              'candidate artifacts through selection.json.', '',
              '## Reproduce', '',
              '```sh\n.venv/bin/python evaluation/run_retrieval_experiments.py --run-id NEW-UNIQUE-ID\n```', '']
    (destination / 'report.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    if not args.run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_'
                              for c in args.run_id):
        parser.error('Use letters, digits, hyphens and underscores for run ID')
    for path, expected in shared.HASHES.items():
        if shared.digest(ROOT / path) != expected:
            parser.error(f'Benchmark hash mismatch: {path}')
    rows = shared.read_csv(ROOT / 'data/practice.csv')
    manifest = json.loads((ROOT / 'evaluation/splits-v1.json').read_text())
    sources = ['baseline.py', 'experiments/text_variants.py', 'experiments/retrieval_predict.py',
               'retrieval/features.py', 'evaluation/run_retrieval_experiments.py',
               'evaluation/run_text_experiments.py', 'evaluation/scoring.py',
               'requirements.txt', 'EVALUATION.md']
    source_hashes = {path: shared.digest(ROOT / path) for path in sources}
    destinations = {name: ROOT / 'results/evaluations' / name / args.run_id
                    for name in (*CANDIDATES, SELECTED)}
    for destination in destinations.values():
        if destination.exists():
            parser.error(f'Run already exists: {destination}')
    results = {name: {'validation': {}, 'test': {}} for name in CANDIDATES}
    selections, commands, input_hashes = {}, {}, {}
    with tempfile.TemporaryDirectory(prefix='hype-retrieval-eval-') as temp:
        temp = Path(temp)
        # Finish all candidate validation fits and freeze selection before any test fit.
        for phase in ('validation', 'test'):
            for track in shared.TRACKS:
                split = manifest['splits'][track]
                train_ids = split['fit_ids'] if phase == 'validation' else split['train_ids']
                pending_ids = split['validation_ids'] if phase == 'validation' else split['test_ids']
                train_set, pending_set = set(train_ids), set(pending_ids)
                sanitized = shared.sanitized_rows(rows, train_ids, pending_ids)
                path = temp / f'{phase}-{track}.csv'
                shared.write_csv(path, sanitized, ['id', 'text', 'date', 'label'])
                input_hashes[f'{phase}/{track}'] = shared.digest(path)
                train_rows = [r for r in sanitized if r['id'] in train_set]
                gold = [{'id': r['id'], 'label': r['label']} for r in rows if r['id'] in pending_set]
                for name in CANDIDATES:
                    outdir = destinations[name] / ('validation' if phase == 'validation' else 'fixed_050') / track
                    command, elapsed = run_candidate(name, path, outdir)
                    commands[f'{name}/{phase}/{track}'] = command
                    metric = score_output(outdir, gold, train_rows, name, args.run_id,
                                          track, phase, elapsed)
                    results[name][phase][track] = metric
                    print(f"{phase:10} {track:10} {name:25} macro_f1="
                          f"{metric['classification']['macro_f1']:.3f}", flush=True)
                if phase == 'validation':
                    scores = {n: results[n][phase][track]['classification']['macro_f1'] for n in CANDIDATES}
                    winner = max(CANDIDATES, key=lambda n: (scores[n], -CANDIDATES.index(n)))
                    selections[track] = {'selected_model': winner, 'validation_macro_f1': scores,
                                         'tie_break': 'declared candidate ordering, baseline first',
                                         'selection_uses_test': False}
                    selected_dir = destinations[SELECTED] / 'fixed_050' / track
                    selected_dir.mkdir(parents=True)
                    shared.write_json(selected_dir / 'selection.json', selections[track])
                else:
                    winner = selections[track]['selected_model']
                    source_dir = destinations[winner] / 'fixed_050' / track
                    target_dir = destinations[SELECTED] / 'fixed_050' / track
                    for filename in ('predictions.csv', 'scored_predictions.csv'):
                        (target_dir / filename).write_bytes((source_dir / filename).read_bytes())
                    metric = {**results[winner]['test'][track], 'method_id': SELECTED,
                              'selected_model': winner, 'candidate_artifact_directory': str(source_dir.relative_to(ROOT))}
                    shared.write_json(target_dir / 'metrics.json', metric)
    import numpy, scipy, sklearn, joblib, threadpoolctl
    versions = {'python': platform.python_version(), **{m.__name__: m.__version__
                for m in (numpy, scipy, sklearn, joblib, threadpoolctl)}}
    for name, destination in destinations.items():
        artifact_hashes = {str(path.relative_to(destination)): shared.digest(path)
                           for path in sorted(destination.rglob('*')) if path.is_file()}
        run = {
            'benchmark_version': 'arena-eval-v1', 'method_id': name, 'run_id': args.run_id,
            'seed': 42, 'policies': ['fixed_050'], 'splits': list(shared.TRACKS),
            'pinned_sha256': shared.HASHES, 'source_commit': shared.source_version(),
            'source_file_sha256': source_hashes, 'exported_artifacts_sha256': artifact_hashes,
            'dependencies': versions,
            'exact_reproduction_command': f'.venv/bin/python evaluation/run_retrieval_experiments.py --run-id {args.run_id}',
            'method_commands': commands, 'sanitized_input_sha256': input_hashes,
            'training_ids': {t: manifest['splits'][t]['train_ids'] for t in shared.TRACKS},
            'fit_ids': {t: manifest['splits'][t]['fit_ids'] for t in shared.TRACKS},
            'validation_ids': {t: manifest['splits'][t]['validation_ids'] for t in shared.TRACKS},
            'test_ids': {t: manifest['splits'][t]['test_ids'] for t in shared.TRACKS},
            'training_sources': 'Only labelled permitted runtime rows; text and labels, dates and public lexicon for retrieval. No forms.',
            'access_profile': 'batch_unlabelled_v1',
            'factual_assets': 'None; retrieved labels are probabilistic model features and never new world evidence.',
            'added_training_material': 'None; no synthetic data, external facts or pretrained models.',
            'model_asset_sha256': None,
            'model_asset_reason': 'Models and TF-IDF indices fitted in memory; audit artifacts exported separately.',
            'leakage_audit': {'status': 'pass_logical_boundary',
                              'detail': 'Separate method subprocesses use runtime-shaped sanitized CSVs. '
                                        'OOF training retrieval excludes entire query fold; evaluator audits neighbor IDs and labels. '
                                        'No OS-enforced filesystem sandbox.'},
            'confidence': {'semantics': 'uncalibrated logistic-regression winning-class probability',
                           'calibration': 'none', 'threshold': 0.5, 'threshold_selection': 'none'},
            'selection': selections,
            'hardware': {'os': platform.platform(), 'machine': platform.machine(),
                         'cpu': platform.processor(), 'thread_limit': 2, 'memory_limit_bytes': None,
                         'cpu_limit_enforced': None, 'process_limit': None, 'timing': 'perf_counter, native host'},
            'capabilities': {'classification': 'implemented', 'generation': 'not_implemented',
                             'proofs': 'not_implemented', 'frontend': 'not_implemented',
                             'constrained_container_benchmark': 'not_run'},
            'limitations': 'Repeated development on these outer splits; related templates can cross partitions. '
                           'OOF reference pools are smaller than the final inference index, a train/inference distribution difference.',
        }
        shared.write_json(destination / 'run.json', run)
        report(destination, name, results, selections, args.run_id)


if __name__ == '__main__':
    main()
