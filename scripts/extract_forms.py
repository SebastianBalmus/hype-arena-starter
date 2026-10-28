"""Extract practice-shaped assertion forms without reading labels or gold forms."""

import argparse
import csv
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from entity_linking.pipeline import extract_rows
from entity_linking.values import exact_json
from entity_linking.io import read_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Fresh directory for predictions.csv and traces.jsonl')
    parser.add_argument('--model', type=Path, default=ROOT / 'models/potion-base-8M')
    parser.add_argument('--lexicon', type=Path, default=ROOT / 'data/lexicon.json')
    parser.add_argument('--registry', type=Path, default=ROOT / 'data/entity_registry.json')
    parser.add_argument('--threshold', type=float, default=.4)
    parser.add_argument('--margin', type=float, default=.02)
    args = parser.parse_args()
    started = time.perf_counter()
    rows, _ = read_rows(args.input)
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / 'predictions.csv').open('w', newline='') as csv_file, (args.output / 'traces.jsonl').open('w') as trace_file:
        writer = csv.DictWriter(csv_file, fieldnames=['id', 'text', 'date', 'predicted_form'])
        writer.writeheader()
        for prediction in extract_rows(rows, args.model, args.lexicon, args.registry, args.threshold, args.margin):
            writer.writerow({k: exact_json(prediction[k]) if k == 'predicted_form' else prediction[k] for k in writer.fieldnames})
            trace_file.write(exact_json(prediction) + '\n')
    print(f'{len(rows)} rows extracted in {time.perf_counter() - started:.3f}s')


if __name__ == '__main__':
    main()
