"""User-requested synthetic-only -> original-practice diagnostic, not arena-eval-v1."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.run_text_experiments import read_csv, write_csv, write_json, HASHES
from evaluation.scoring import score_rows
from experiments.daria_feature_variants import CONFIGS
from experiments.daria_feature_sweep import fit_predict, audit_cache
from baseline import make_model, predict_rows


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    dest = ROOT / 'results/evaluations/synthetic_transfer' / sys.argv[1]
    dest.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    synth_path = ROOT / 'data/augmentation-20261028-2000/practice_additional_2000.csv'
    train = read_csv(synth_path)
    gold = read_csv(ROOT / 'data/practice.csv')
    for p, expected in HASHES.items():
        assert sha(ROOT / p) == expected, p
    assert len(train) == 2000 and len(gold) == 1897
    assert not {r['id'] for r in train} & {r['id'] for r in gold}
    inputs = dest / 'inputs'
    inputs.mkdir()
    fields = ['id', 'text', 'date', 'label']
    runtime = [{k: r[k] for k in fields} for r in train]
    runtime += [{k: ('' if k == 'label' else r[k]) for k in fields} for r in gold]
    write_csv(inputs / 'runtime.csv', runtime, fields)
    write_csv(inputs / 'forms.csv', train, fields + ['form'])
    write_json(inputs / 'calibration-fit.json', [r['id'] for r in train[:1600]])
    source = dest / 'source'
    commit = subprocess.check_output(['git', 'rev-parse', '5943f61'], cwd=ROOT, text=True).strip()
    for f in subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', commit, 'hooly'], cwd=ROOT, text=True).splitlines():
        if f.endswith('.py') and f not in ('hooly/run.py', 'hooly/writer.py'):
            p = source / f
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(subprocess.check_output(['git', 'show', f'{commit}:{f}'], cwd=ROOT))
    params = json.loads((ROOT / 'results/comparisons/daria-features-20261028-001/method-inputs/stratified/gb-params.json').read_text())
    command = [sys.executable, str(ROOT / 'integrations/daria_features.py'), '--source-dir', str(source), '--input', str(inputs/'runtime.csv'), '--forms', str(inputs/'forms.csv'), '--lexicon', str(ROOT/'data/lexicon.json'), '--calibration-fit-ids', str(inputs/'calibration-fit.json'), '--output', str(dest/'features.json')]
    metadata = {'benchmark_version': 'synthetic-transfer-diagnostic-v1', 'seed': 42, 'policy': 'fixed_050', 'upstream_commit': commit,
        'method': 'Daria expanded_none + gradient boosting', 'parameters': params,
        'selection': 'Frozen primary-track CV-selected configuration from saved comparison; no tuning on original targets.',
        'training_rows': 2000, 'test_rows': 1897, 'training_sha256': sha(synth_path), 'pinned_sha256': HASHES,
        'eligible_for_clean_generalization_claim': False,
        'contamination': 'Synthetic training rows derive from original evaluation texts, forms and labels. Diagnostic only; not independent data-quality confirmation.',
        'direct_input_boundary': 'Only synthetic labels/forms supplied to model. Original text/date visible as unlabelled batch. Gold is evaluator-only; no OS filesystem isolation.',
        'native_limitations': 'Native HMM OOF retains permitted synthetic training forms. Probabilistic timelines/persistence assumptions are inherited; confidence is not proof.',
        'standard_tracks': 'Not rerun: user explicitly requested a different training/test partition; frozen v1 manifests unchanged. Results are not v1-comparable.',
        'generation': 'not_implemented', 'proofs': 'not_verified', 'container_resources': 'not_run',
        'command': command, 'runner_sha256': sha(__file__)}
    write_json(dest/'run.json', metadata)
    print('Rebuilding parser, calibration, timelines and feature model from synthetic rows only.', flush=True)
    with (dest/'fit.log').open('w') as log:
        subprocess.run(command, cwd=inputs, check=True, stdout=log, stderr=subprocess.STDOUT)
    cache = json.loads((dest/'features.json').read_text())
    audit_cache(cache, [r['id'] for r in train], [r['id'] for r in gold])
    config = next(c for c in CONFIGS if c['id'] == 'expanded_none')
    predictions, timing = fit_predict(config, cache, params)
    results = {'daria_expanded_gb': (predictions, timing)}
    for name in ('always_true', 'tfidf_lr', 'tfidf_lr_balanced'):
        t = time.perf_counter()
        model = make_model(name, seed=42)
        model.fit([r['text'] for r in train], [r['label'] for r in train])
        pred = predict_rows(model, gold, min_confidence=.5)
        for r in pred:
            r['decision_reason'] = 'decided' if r['label'] != 'unsure' else 'low_confidence'
        results[name] = (pred, {'total_seconds': time.perf_counter()-t})
    ancestor_ids = {json.loads(l)['source_id'] for l in (ROOT/'data/augmentation-20261028-2000/provenance.jsonl').read_text().splitlines()}
    summaries = {}
    for name, (pred, timing) in results.items():
        out = dest/name
        out.mkdir()
        metrics, errors, scored = score_rows(gold, pred)
        assert errors['eligible_for_comparison'], errors
        write_csv(out/'predictions.csv', [{k:r.get(k) for k in ['id','label','confidence','decision_reason']} for r in pred], ['id','label','confidence','decision_reason'])
        write_csv(out/'scored_predictions.csv', scored, list(scored[0]))
        slices = {}
        for label, ids in [('source_has_descendants', ancestor_ids), ('source_has_no_direct_descendants', {r['id'] for r in gold}-ancestor_ids)]:
            slices[label] = score_rows([r for r in gold if r['id'] in ids], [r for r in pred if r['id'] in ids])[0]
        write_json(out/'metrics.json', {'classification': metrics, 'output_errors':errors, 'timings':timing, 'diagnostic_slices':slices, 'eligible_for_clean_generalization_claim':False})
        summaries[name] = metrics
    primary = summaries['daria_expanded_gb']
    lines = ['# Synthetic-only training → original practice predictions', '', 'Training: 2,000 synthetic rows only. Test: all 1,897 original rows, supplied with empty labels and no forms. Seed 42; fixed 0.50 decision policy. Daria expanded features + gradient boosting, frozen settings selected in the prior primary-track CV comparison.', '', '**This is a source-dependent transfer diagnostic, not independent confirmation that synthetic data are correct.** Generated training examples were built from these original test labels/forms/texts. All parser/calibration/temporal resources are rebuilt using synthetic training inputs only, but that indirect ancestry leakage remains.', '', f'Accuracy: {primary["accuracy"]:.4f}. Macro F1: {primary["macro_f1"]:.4f}.', '', '| Class | Precision | Recall | F1 | Support |', '|---|---:|---:|---:|---:|']
    for label,m in primary['per_class'].items():
        lines.append(f'| {label} | {m["precision"]:.4f} | {m["recall"]:.4f} | {m["f1"]:.4f} | {m["support"]} |')
    lines += ['', 'Confusion matrix: gold rows and prediction columns ordered true, false, unsure.', '', '```json', json.dumps(primary['confusion_matrix']), '```', '', '| Control | Accuracy | Macro F1 |', '|---|---:|---:|']
    for name,m in summaries.items(): lines.append(f'| {name} | {m["accuracy"]:.4f} | {m["macro_f1"]:.4f} |')
    lines += ['', 'The prior-only always_true control abstains under fixed_050 because synthetic true prevalence is 45%, below the decision threshold. It is not forced to emit an invalid confidence.', '', 'Per-class metrics, all-row predictions, errors, reliability diagnostics and direct-ancestor slices are saved beside this report. Neither slice is independent: all share the supplied fictional universe and full-corpus augmentation design. Native/macOS timing is supplementary; constrained image resources, generation and independent proof verification were not run. No baseline results, runtime assets, original data or shared splits were overwritten.', '', 'Reproduce from repository root: `.venv/bin/python evaluation/run_synthetic_transfer.py NEW-UNIQUE-RUN-ID`.', '']
    (dest/'report.md').write_text('\n'.join(lines))
    metadata['timings'] = {'features': cache['timings'], 'classifier': results['daria_expanded_gb'][1], 'total_seconds':time.perf_counter()-start}
    write_json(dest/'run.json', metadata)
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == '__main__':
    main()
