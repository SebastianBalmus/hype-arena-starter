#!/usr/bin/env python3
"""Predict blank-label runtime rows with a selected Daria/sklearn configuration."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.daria_sklearn import CONFIGS
from integrations.daria_predict import read_csv, RUNTIME_FIELDS


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-dir', required=True, help='Pinned Daria hooly source directory')
    ap.add_argument('--input', required=True, help='id,text,date,label CSV; pending labels empty')
    ap.add_argument('--forms', required=True, help='Training-only id,text,date,label,form CSV')
    ap.add_argument('--lexicon', default=str(ROOT / 'data/lexicon.json'))
    ap.add_argument('--candidate', required=True, choices=['native'] + [c['id'] for c in CONFIGS])
    ap.add_argument('--output', required=True, help='Classification predictions; no generated claims')
    args = ap.parse_args()
    rows = read_csv(args.input, RUNTIME_FIELDS)
    labelled_ids = [r['id'] for r in rows if r['label']]
    if len(labelled_ids) < 10:
        ap.error('At least ten training rows required for parse calibration')
    env = os.environ.copy()
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        env[key] = '2'
    with tempfile.TemporaryDirectory(prefix='daria-sk-predict-') as temp:
        temp = Path(temp)
        (temp / 'calibration-fit.json').write_text(json.dumps(labelled_ids[:int(len(labelled_ids) * .8)]))
        subprocess.run([sys.executable, str(ROOT / 'integrations/daria_features.py'),
            '--source-dir', str(Path(args.source_dir).resolve()), '--input', str(Path(args.input).resolve()),
            '--forms', str(Path(args.forms).resolve()), '--lexicon', str(Path(args.lexicon).resolve()),
            '--calibration-fit-ids', str(temp / 'calibration-fit.json'), '--output', str(temp / 'features.json')],
            cwd=temp, env=env, check=True)
        subprocess.run([sys.executable, str(ROOT / 'integrations/daria_candidates.py'),
            '--cache', str(temp / 'features.json'), '--output-dir', str(temp / 'predictions'),
            '--candidate', args.candidate], cwd=temp, env=env, check=True)
        result = temp / 'predictions' / f'{args.candidate}.csv'
        if not result.exists():
            raise RuntimeError((temp / 'predictions/timings.json').read_text())
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        staged = output.with_suffix(output.suffix + '.tmp')
        shutil.copyfile(result, staged)
        os.replace(staged, output)


if __name__ == '__main__':
    main()
