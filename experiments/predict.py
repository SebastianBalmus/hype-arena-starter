#!/usr/bin/env python3
"""Train one text experiment and label previously unseen CSV rows."""

import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import baseline
from experiments.text_variants import MODELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='CSV with blank labels to predict')
    parser.add_argument('--train', type=Path, default=ROOT / 'data/practice.csv')
    parser.add_argument('--model', choices=MODELS, default='word_char_lr_balanced')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/variant-labels.csv')
    args = parser.parse_args()
    try:
        train = [row for row in baseline.read_rows(args.train) if row['label']]
        pending = [row for row in baseline.read_rows(args.input) if not row['label']]
        if not train:
            raise ValueError('Training CSV has no labelled rows')
        overlap = {r['id'] for r in train} & {r['id'] for r in pending}
        if overlap:
            raise ValueError('Blank input IDs also appear in the labelled training file; '
                             'use the evaluation runner for holdouts: ' + ', '.join(sorted(overlap)[:5]))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='hype-predict-') as folder:
            folder = Path(folder)
            method_input = folder / 'input.csv'
            method_output = folder / 'predictions.csv'
            with method_input.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['id', 'text', 'date', 'label'])
                writer.writeheader()
                writer.writerows(train + pending)
            if pending:
                subprocess.run([sys.executable, str(ROOT / 'experiments/text_variants.py'),
                                '--input', str(method_input), '--model', args.model,
                                '--output', str(method_output), '--seed', '42'], check=True)
                with method_output.open(encoding='utf-8', newline='') as stream:
                    predictions = list(csv.DictReader(stream))
            else:
                predictions = []
            with args.output.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['id', 'type', 'text', 'form', 'label', 'confidence'])
                writer.writeheader()
                for row in predictions:
                    writer.writerow({'id': row['id'], 'type': '', 'text': '', 'form': '',
                                     'label': row['label'], 'confidence': row['confidence']})
        print(f'Wrote {len(predictions)} classifications to {args.output}.')
        print('Classification only: output does not contain the required 1,000 claims.')
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
