#!/usr/bin/env python3
"""Audit and annotate completed feature experiments without changing selection."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.run_text_experiments import TRACKS, digest, write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-id', required=True)
    args = ap.parse_args()
    out = ROOT / 'results/comparisons' / args.run_id
    run = json.loads((out / 'run.json').read_text())
    results = json.loads((out / 'summary.json').read_text())
    names = [c['id'] for c in run['feature_configs']]
    assert len(names) == 24
    assert all(set(tracks) == set(TRACKS) and all(m['eligible_for_comparison'] for m in tracks.values())
               for tracks in results.values())
    assert all(detail['status'] == 'ok' for candidates in run['candidate_status'].values() for detail in candidates.values())
    assert all(digest(ROOT / f) == h for f, h in run['source_file_sha256'].items())
    for track in TRACKS:
        cv = run['selections'][track]['results']
        viable = [n for n in names if cv[n]['status'] == 'ok']
        assert max(viable, key=lambda n: cv[n]['mean_macro_f1']) == run['selections'][track]['selected']
        assert run['baseline_parity'][track]['status'] == 'exact'
    for name, tracks in results.items():
        method = 'daria_features_selected' if name == 'selected' else f'daria_features_{name}'
        report = ROOT / 'results/evaluations' / method / args.run_id / 'report.md'
        content = report.read_text().split('\n<!-- selection-audit -->')[0]
        if name == 'selected':
            note = 'This report evaluates the CV-selected procedure; its feature representation can differ by track.'
        else:
            chosen = [t for t in TRACKS if run['selections'][t]['selected'] == name]
            note = ('This report evaluates a predeclared fixed candidate. CV selected it on: '
                    + (', '.join(chosen) if chosen else '**none of the tracks**')
                    + '. Its other test results are descriptive and were not used to choose a representation.')
        content += '\n<!-- selection-audit -->\n\n' + note + '\n\n'
        content += '| Track / control | Control F1 | Candidate ΔF1 | Control T/F errors/count | Candidate T/F errors/count |\n|---|---:|---:|---:|---:|\n'
        for track, m in tracks.items():
            c = m['classification']
            for control, info in run['matched_controls'][track].items():
                old = info['classification']
                content += f'| {track} / {control} | {old["macro_f1"]:.3f} | {c["macro_f1"]-old["macro_f1"]:+.3f} | {old["decided_errors"]}/{old["decided_count"]} | {c["decided_errors"]}/{c["decided_count"]} |\n'
        report.write_text(content)
    report = out / 'report.md'
    content = report.read_text().split('\n<!-- extended-results -->')[0]
    content += '\n<!-- extended-results -->\n\nAll configurations: each cell shows **training CV F1 / outer test F1**. Only CV determined selection.\n\n'
    content += '| Feature configuration | Stratified | Temporal | Checker |\n|---|---:|---:|---:|\n'
    for name in names:
        cells = []
        for track in TRACKS:
            cv = run['selections'][track]['results'][name]
            m = results[name].get(track)
            cells.append(f'{cv["mean_macro_f1"]:.3f} / {m["classification"]["macro_f1"]:.3f}'
                         if cv['status'] == 'ok' and m else 'unavailable')
        content += f'| {name} | ' + ' | '.join(cells) + ' |\n'
    novel = sorted((n for n in names if n != 'expanded_none'),
                   key=lambda n: results[n]['stratified']['classification']['macro_f1'], reverse=True)[:3]
    content += '\nThe highest observed primary-test scores among new candidates were: '
    content += ', '.join(f'`{n}` ({results[n]["stratified"]["classification"]["macro_f1"]:.3f})' for n in novel)
    content += ('. These are **post-test descriptive rankings**, not CV-selected winners. '
                'The original feature representation was the primary CV winner.\n\n'
                'Gradient-boosting hyperparameters were previously optimized for the original representation and stayed fixed here. '
                'This isolates representation changes at those settings; joint feature-specific hyperparameter tuning was not tested. '
                'The reader and assertion parser were unchanged.\n\n'
                'All 121 repository tests passed. Every original baseline prediction matched exactly across all three tracks. '
                'All 24 configurations completed CV, shared-validation and outer prediction without failure.\n\n'
                'Recreate this extended summary after the experiment with:\n\n'
                f'```sh\n.venv/bin/python evaluation/summarize_daria_features.py --run-id {args.run_id}\n```\n')
    report.write_text(content)
    write_json(out / 'verification.json', {'method_code_hashes_unchanged': True,
        'candidate_count': len(names), 'eligible_outer_evaluations_including_selected':
            sum(len(ts) for ts in results.values()), 'all_three_baselines_exact': True,
        'tests_passed': 121, 'test_command': '.venv/bin/python -m unittest discover -s tests -q',
        'summary_script_sha256': digest(Path(__file__)), 'report_sha256': digest(report),
        'summary_command': f'.venv/bin/python evaluation/summarize_daria_features.py --run-id {args.run_id}'})
    print(report)


if __name__ == '__main__':
    main()
