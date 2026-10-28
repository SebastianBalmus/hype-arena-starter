#!/usr/bin/env python3
"""Evaluate fold-scoped Madalin claim features against unchanged retrieval."""

import argparse
import gzip
import hashlib
import importlib.metadata
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
from evaluation import run_retrieval_experiments as retrieval_eval
from evaluation import run_text_experiments as shared

CONTROL = 'retrieval_raw_masked'
CANDIDATES = (CONTROL, 'madalin_observed_features', 'madalin_generated_features')
SELECTED = 'madalin_facts_selected'


def audit_claims(directory, train_rows, pending_ids):
    """Check every synthetic reference's transitive source boundary, not its truth."""
    by_id = {r['id']: r for r in train_rows}
    with gzip.open(directory / 'claim_catalogues.json.gz', 'rt', encoding='utf-8') as stream:
        catalogues = json.load(stream)
    folds = {f"fold-{f['fold_index']}": f for f in json.loads((directory / 'oof_folds.json').read_text())}
    if set(catalogues) != set(folds) | {'full'}:
        raise ValueError('Missing or unexpected claim catalogue scope')
    records_by_scope, total_records = {}, 0
    for scope, catalogue in catalogues.items():
        allowed = set(by_id) if scope == 'full' else set(folds[scope]['reference_ids'])
        if set(catalogue['reference_ids']) != allowed:
            raise ValueError('Claim catalogue uses unexpected reference pool')
        records = catalogue['records']
        if len(records) > 1000 or len(records) != len({r['candidate_id'] for r in records}):
            raise ValueError('Claim catalogue exceeds cap or has duplicate candidate IDs')
        for record in records:
            ids = record['source_ids']
            if not ids or len(set(ids)) != len(ids) or not set(ids) <= allowed:
                raise ValueError('Claim candidate source is missing or outside its fold')
            fingerprints = record['source_fingerprints']
            if set(fingerprints) != set(ids) or record['status'] != 'unverified_auxiliary':
                raise ValueError('Claim candidate provenance or verification status is invalid')
            for rid in ids:
                source = by_id[rid]
                fingerprint = hashlib.sha256('\0'.join(source[k] for k in ('id', 'text', 'date')).encode()).hexdigest()
                if source['label'] != 'true' or fingerprints[rid] != fingerprint:
                    raise ValueError('Claim source is not a matching true permitted row')
            if set(record['source_dates']) != {by_id[rid]['date'] for rid in ids}:
                raise ValueError('Claim source dates differ from original observations')
        records_by_scope[scope] = {r['candidate_id']: r for r in records}
        total_records += len(records)
    citations = 0
    for phase, filename in [('oof', 'oof_traces.jsonl'), ('pending', 'traces.jsonl')]:
        traces = retrieval_eval.read_jsonl(directory / filename)
        for trace in traces:
            scope = f"fold-{trace['fold_index']}" if phase == 'oof' else 'full'
            auxiliary = trace['auxiliary_claims']
            if auxiliary['query_id'] != trace['query_id']:
                raise ValueError('Auxiliary trace belongs to a different query')
            for neighbor in auxiliary['neighbors']:
                record = records_by_scope[scope].get(neighbor['candidate_id'])
                if (record is None or neighbor['source_ids'] != record['source_ids']
                        or neighbor['source_dates'] != record['source_dates']
                        or not 0 < neighbor['similarity'] <= 1.000001):
                    raise ValueError('Auxiliary neighbor does not match the correct fold catalogue')
                citations += 1
    return {'status': 'pass', 'catalogue_scopes': len(catalogues),
            'candidate_records_checked': total_records, 'neighbor_citations_checked': citations,
            'full_catalogue_count': len(catalogues['full']['records']),
            'full_catalogue_types': dict(Counter(r['type'] for r in catalogues['full']['records'])),
            'truth_verification': 'not_verified', 'all_sources_permitted_true': True}


def run_method(name, input_path, forms_path, output):
    output.mkdir(parents=True)
    if name == CONTROL:
        command = [sys.executable, str(ROOT / 'experiments/retrieval_predict.py'),
                   '--input', str(input_path), '--lexicon', str(ROOT / 'data/lexicon.json'),
                   '--variant', 'raw_masked', '--output', str(output / 'predictions.csv'),
                   '--artifacts', str(output), '--seed', '42']
    else:
        mode = 'observed' if name == 'madalin_observed_features' else 'generated'
        command = [sys.executable, str(ROOT / 'experiments/fact_retrieval_predict.py'),
                   '--input', str(input_path), '--training-forms', str(forms_path),
                   '--lexicon', str(ROOT / 'data/lexicon.json'), '--mode', mode,
                   '--output', str(output / 'predictions.csv'), '--artifacts', str(output), '--seed', '42']
    env = os.environ.copy()
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        env[key] = '2'
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    if completed.returncode:
        raise RuntimeError(f'{name} failed: {completed.stderr[-6000:]}')
    for artifact in retrieval_eval.ARTIFACTS:
        if not (output / artifact).is_file():
            raise ValueError(f'Missing method artifact: {artifact}')
    return command, elapsed


def write_report(destination, name, results, selections, run_id):
    lines = [f'# {name}: Madalin claim-feature integration', '',
             'arena-eval-v1, fixed_050, seed 42. Development results on previously '
             'explored frozen splits. Generator candidates are auxiliary model features, '
             'not independently verified observations.', '',
             '| Track | Validation macro F1 | Test macro F1 | False P/R | Unsure F1 | Coverage | Errors/decided | Risk | Brier | Runtime |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for track in shared.TRACKS:
        candidate = selections[track]['selected_model'] if name == SELECTED else name
        val = results[candidate]['validation'][track]['classification']
        metric = results[candidate]['test'][track]
        c = metric['classification']; f = c['per_class']['false']
        lines.append(f"| {track} | {val['macro_f1']:.3f} | {c['macro_f1']:.3f} | "
                     f"{f['precision']:.3f}/{f['recall']:.3f} | {c['per_class']['unsure']['f1']:.3f} | "
                     f"{c['decided_coverage']:.3f} | {c['decided_errors']}/{c['decided_count']} | "
                     f"{c['decided_risk']:.3f} | {c['correctness_brier']:.3f} | "
                     f"{metric['resources']['method_process_seconds']:.2f}s native |")
    lines += ['', 'Confusion matrices and reliability bins are in each metrics.json. '
              'Grounded yield, independently verified generation yield and accepted proof rate '
              'are unavailable. Peak RAM and constrained Linux resources were not measured.', '',
              '## Paired changes and reference controls', '',
              '| Track | This macro F1 | Retrieval control | Always true | Word LR | Balanced word LR |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for track in shared.TRACKS:
        candidate = selections[track]['selected_model'] if name == SELECTED else name
        values = [results[candidate]['test'][track]['classification']['macro_f1'],
                  results[CONTROL]['test'][track]['classification']['macro_f1']]
        for control in shared.CONTROLS:
            metric = ROOT / 'results/evaluations' / control / 'text-variants-20261028-001/fixed_050' / track / 'metrics.json'
            data = json.loads(metric.read_text())
            if not (data['benchmark_version'] == 'arena-eval-v1' and data['policy'] == 'fixed_050'
                    and data['access_profile'] == 'batch_unlabelled_v1'):
                raise ValueError('Historical control has incompatible metadata')
            values.append(data['classification']['macro_f1'])
        lines.append('| ' + track + ' | ' + ' | '.join(f'{v:.3f}' for v in values) + ' |')
    lines += ['', '## Design and boundaries', '',
              'The reference is unchanged raw+number-masked retrieval. The observed candidate '
              'renders dated atomic statements from permitted true training forms with the '
              'teammate value renderer. The generated candidate calls Madalin’s pinned '
              'enumerate_candidates/select_claims implementation on true training annotations, '
              'with no pending/unsure/false positive observations and at most 1,000 candidates '
              'per index. Each training fold regenerates its own assets from its other four '
              'folds. All query labels are blank and query forms are absent.', '',
              'Claim-feature catalogues preserve source IDs, fingerprints, observation dates '
              'and temporal assumptions. Current fact/comparison candidates retain the contract '
              'as-of date; publication dates are not silently substituted for that date. '
              'Generator intervals, freshness and relation overlap remain assumptions. '
              'Candidates never enter the trusted world evidence store or become synthetic '
              'true-labelled training targets. All output probabilities remain uncalibrated.', '',
              'Selection uses the frozen inner fit/validation rows; the winner is recorded '
              'before any test fit. Final refits use the complete outer training pool. '
              'All methods share the same fixed 0.50 decision policy. Method inputs are '
              'sanitized CSV files and split-permitted training forms; gold answers remain '
              'evaluator-side. No OS-enforced filesystem sandbox exists.', '',
              'The selected method references its candidate’s saved artifact directory '
              'instead of refitting twice. Source and exported artifact hashes, upstream '
              'commit, exact row-ID pools and native timing are in run.json.', '',
              '## Validation selection', '', '| Track | Selected candidate |', '| --- | --- |']
    for track in shared.TRACKS:
        lines.append(f"| {track} | {selections[track]['selected_model']} |")
    lines += ['', '## Representative failures', '',
              'The rows below show the executed classifier outcome. Neighbor/claim-feature '
              'traces describe its inputs; no unsupported causal explanation is inferred.', '',
              '| ID | Gold | Predicted | Confidence | Reason |', '| --- | --- | --- | ---: | --- |']
    selected = selections['stratified']['selected_model'] if name == SELECTED else name
    path = destination.parent.parent / selected / run_id / 'fixed_050/stratified/scored_predictions.csv'
    failures = [r for r in shared.read_csv(path) if r['correct'] == 'False']
    failures.sort(key=lambda r: (r['gold'] == 'true', r['id']))
    for row in failures[:5]:
        confidence = f"{float(row['confidence']):.3f}" if row['confidence'] else 'null'
        lines.append(f"| {row['id']} | {row['gold']} | {row['label']} | {confidence} | {row['decision_reason']} |")
    lines += ['', '## Reproduce', '',
              '```sh\n.venv/bin/python evaluation/run_fact_integration.py --run-id NEW-UNIQUE-ID\n```', '',
              'Submission claim generation, independent truth/proof verification, frontend '
              'integration and constrained-container readiness are not implemented by this adapter.', '']
    (destination / 'report.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    if not args.run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_'
                              for c in args.run_id):
        parser.error('Invalid run ID')
    for path, expected in shared.HASHES.items():
        if shared.digest(ROOT / path) != expected:
            parser.error(f'Benchmark hash mismatch: {path}')
    upstream = json.loads((ROOT / 'integrations/madalin-source.json').read_text())
    for path, expected in upstream['files'].items():
        if shared.digest(ROOT / path) != expected:
            parser.error(f'Imported source differs from pinned Madalin commit: {path}')
    manifest = json.loads((ROOT / 'evaluation/splits-v1.json').read_text())
    rows = shared.read_csv(ROOT / 'data/practice.csv')
    destinations = {name: ROOT / 'results/evaluations' / name / args.run_id for name in (*CANDIDATES, SELECTED)}
    for destination in destinations.values():
        if destination.exists():
            parser.error(f'Run already exists: {destination}')
    sources = ['baseline.py', 'experiments/text_variants.py', 'experiments/kb_predict.py',
               'experiments/retrieval_predict.py', 'experiments/fact_retrieval_predict.py',
               'retrieval/features.py', 'retrieval/claim_features.py', 'requirements.txt',
               'evaluation/run_fact_integration.py', 'evaluation/run_retrieval_experiments.py',
               'evaluation/run_text_experiments.py', 'evaluation/scoring.py', 'EVALUATION.md',
               'integrations/madalin-source.json', *upstream['files']]
    source_hashes = {p: shared.digest(ROOT / p) for p in sources}
    results = {n: {'validation': {}, 'test': {}} for n in CANDIDATES}
    selections, commands, input_hashes = {}, {}, {}
    with tempfile.TemporaryDirectory(prefix='hype-madalin-integration-') as temp:
        temp = Path(temp)
        for phase in ('validation', 'test'):
            for track in shared.TRACKS:
                split = manifest['splits'][track]
                train_ids = split['fit_ids'] if phase == 'validation' else split['train_ids']
                pending_ids = split['validation_ids'] if phase == 'validation' else split['test_ids']
                train_set, pending_set = set(train_ids), set(pending_ids)
                inputs = shared.sanitized_rows(rows, train_ids, pending_ids)
                training = [r for r in rows if r['id'] in train_set]
                gold = [{'id': r['id'], 'label': r['label']} for r in rows if r['id'] in pending_set]
                input_path, form_path = temp / f'{phase}-{track}.csv', temp / f'{phase}-{track}-forms.csv'
                shared.write_csv(input_path, inputs, ['id', 'text', 'date', 'label'])
                shared.write_csv(form_path, training, ['id', 'text', 'date', 'label', 'form'])
                input_hashes[f'{phase}/{track}'] = {'runtime': shared.digest(input_path),
                                                        'training_forms': shared.digest(form_path)}
                for name in CANDIDATES:
                    destination = destinations[name] / ('validation' if phase == 'validation' else 'fixed_050') / track
                    command, elapsed = run_method(name, input_path, form_path, destination)
                    commands[f'{name}/{phase}/{track}'] = command
                    metric = retrieval_eval.score_output(destination, gold, training, name, args.run_id, track, phase, elapsed)
                    if name != CONTROL:
                        metric['auxiliary_claims'] = audit_claims(destination, training, pending_ids)
                        metric['generation']['reason'] = 'Submission generation absent; internal candidates used only as classifier features'
                    shared.write_json(destination / 'metrics.json', metric)
                    results[name][phase][track] = metric
                    count = metric.get('auxiliary_claims', {}).get('full_catalogue_count', 0)
                    print(f"{phase:10} {track:10} {name:28} macro_f1={metric['classification']['macro_f1']:.3f} candidates={count}", flush=True)
                if phase == 'validation':
                    scores = {n: results[n][phase][track]['classification']['macro_f1'] for n in CANDIDATES}
                    winner = max(CANDIDATES, key=lambda n: (scores[n], -CANDIDATES.index(n)))
                    selections[track] = {'selected_model': winner, 'validation_macro_f1': scores,
                                         'tie_break': 'declared ordering, retrieval control first', 'selection_uses_test': False}
                    target = destinations[SELECTED] / 'fixed_050' / track
                    target.mkdir(parents=True)
                    shared.write_json(target / 'selection.json', selections[track])
                else:
                    winner = selections[track]['selected_model']
                    source = destinations[winner] / 'fixed_050' / track
                    target = destinations[SELECTED] / 'fixed_050' / track
                    for filename in ('predictions.csv', 'scored_predictions.csv'):
                        (target / filename).write_bytes((source / filename).read_bytes())
                    metric = {**results[winner]['test'][track], 'method_id': SELECTED,
                              'selected_model': winner, 'candidate_artifact_directory': str(source.relative_to(ROOT))}
                    shared.write_json(target / 'metrics.json', metric)
    packages = ['numpy', 'scipy', 'scikit-learn', 'joblib', 'threadpoolctl', 'jsonschema',
                'referencing', 'jsonschema-specifications', 'attrs', 'rpds-py', 'typing-extensions']
    versions = {'python': platform.python_version(), **{p: importlib.metadata.version(p) for p in packages}}
    for name, destination in destinations.items():
        hashes = {str(p.relative_to(destination)): shared.digest(p) for p in sorted(destination.rglob('*')) if p.is_file()}
        run = {
            'benchmark_version': 'arena-eval-v1', 'method_id': name, 'run_id': args.run_id,
            'seed': 42, 'splits': list(shared.TRACKS), 'policies': ['fixed_050'],
            'access_profile': 'batch_unlabelled_v1', 'pinned_sha256': shared.HASHES,
            'source_commit': shared.source_version(), 'source_file_sha256': source_hashes,
            'upstream_generator': upstream, 'dependencies': versions,
            'exported_artifacts_sha256': hashes, 'sanitized_input_sha256': input_hashes,
            'exact_reproduction_command': f'.venv/bin/python evaluation/run_fact_integration.py --run-id {args.run_id}',
            'method_commands': commands, 'selection': selections,
            'training_ids': {t: manifest['splits'][t]['train_ids'] for t in shared.TRACKS},
            'fit_ids': {t: manifest['splits'][t]['fit_ids'] for t in shared.TRACKS},
            'validation_ids': {t: manifest['splits'][t]['validation_ids'] for t in shared.TRACKS},
            'test_ids': {t: manifest['splits'][t]['test_ids'] for t in shared.TRACKS},
            'training_sources': 'Only permitted training text/date/labels/forms and public lexicon. Control ignores forms.',
            'factual_assets': 'None added to trusted world evidence. Madalin candidates are separate unverified auxiliary features.',
            'added_training_material': 'Derived candidate texts from permitted true-row forms only; no independent world facts or external pretrained assets.',
            'confidence': {'semantics': 'uncalibrated winning LR class probability', 'threshold': 0.5,
                           'calibration': 'none', 'threshold_selection': 'none'},
            'leakage_audit': {'status': 'pass_logical_boundary',
                              'detail': 'Sanitized subprocess runtime and exact training-form fingerprint checks. '
                                        'Each OOF generator uses only fold reference rows; source fingerprints and all exposed references audited. '
                                        'Gold answers evaluator-only; no OS filesystem sandbox.'},
            'model_asset_sha256': None, 'model_asset_reason': 'Models fitted in memory; source and generated catalogue hashes recorded.',
            'assumptions': 'Generator freshness/interval/as-of rules are disclosed in candidate records and are not independently verified truth.',
            'capabilities': {'classification': 'implemented', 'auxiliary_claim_generation': 'implemented',
                             'submission_claim_generation': 'not_implemented', 'independent_truth_verifier': 'not_run',
                             'proofs': 'not_implemented', 'frontend': 'not_implemented', 'constrained_container_benchmark': 'not_run'},
            'hardware': {'os': platform.platform(), 'machine': platform.machine(), 'cpu': platform.processor(),
                         'thread_limit': 2, 'cpu_limit_enforced': None, 'memory_limit_bytes': None,
                         'process_limit': None, 'timing': 'perf_counter, native host'},
            'limitations': 'Existing development splits; related templates may cross partitions. '
                           'OOF reference pools smaller than inference index. Candidate source admissibility does not prove generated truth.',
        }
        shared.write_json(destination / 'run.json', run)
        write_report(destination, name, results, selections, args.run_id)


if __name__ == '__main__':
    main()
