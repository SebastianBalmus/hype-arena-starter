"""Score full predicted assertion lists against practice forms after inference."""

import argparse
from collections import Counter
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from entity_linking.report import write_forms_report as write_report
from entity_linking.pipeline import extract_rows
from entity_linking.evaluation import fact_key, scores, evaluate
from entity_linking.io import read_rows, write_jsonl
from entity_linking.values import exact_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'data/practice.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/forms')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh output directory')
    rows, _ = read_rows(args.input)
    started = time.perf_counter()
    predictions = list(extract_rows(rows, ROOT / 'models/potion-base-8M', ROOT / 'data/lexicon.json', ROOT / 'data/entity_registry.json'))
    inference_seconds = time.perf_counter() - started
    # No gold forms are accessed by the extractor above.
    lexicon = json.loads((ROOT / 'data/lexicon.json').read_text())
    aliases = {name.casefold(): e['name'].casefold() for e in lexicon['subjects'] for name in [e['name'], *e['aliases']]}
    metrics, by_attribute, errors = evaluate(rows, predictions, aliases)
    from jsonschema import Draft202012Validator
    validator = Draft202012Validator(json.loads((ROOT / 'contracts/claimed_value.schema.json').read_text()))
    allowed = {e['name']: set(e['attributes']) for e in lexicon['subjects']}
    for prediction in predictions:
        for fact in prediction['predicted_form']:
            assert set(fact) == {'subject', 'attribute', 'value'}
            assert fact['attribute'] in allowed[fact['subject']]
            validator.validate(fact['value'])
    summary = {'method': 'structural_rules_then_lexicon_similarity', 'metrics': metrics,
               'full_value_schema_and_lexicon_checks_passed': True, 'inference_seconds_including_preprocessing': inference_seconds,
               'scope': 'full practice form shape; typed literals plus conservative source-name and Boolean candidates',
               'uses_gold_for_inference': False, 'fits_similarity_references_or_thresholds': False,
               'trace_statuses': dict(Counter(t['status'] for p in predictions for t in p['traces'])),
               'limitations': ['Development diagnostics on known practice data, not untouched held-out results.',
                              'Implicit subjects, contextual names, negation and composition remain only partially handled.',
                              'Predicted forms describe sentence assertions; they are not verified world evidence.'],
               'source_sha256': {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [args.input, ROOT / 'data/lexicon.json',
                                           *[ROOT / 'entity_linking' / name for name in [
                                               'pipeline.py', 'linker.py', 'values.py', 'context.py',
                                               'candidates.py', 'forms.py', 'relationships.py',
                                               'structural_rules.py', 'evaluation.py', 'json_utils.py']]]}}
    args.output.mkdir(parents=True)
    write_jsonl(args.output / 'traces.jsonl', predictions)
    write_jsonl(args.output / 'errors.jsonl', errors)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (args.output / 'by_attribute.json').write_text(json.dumps(by_attribute, indent=2) + '\n')
    with (args.output / 'predictions.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'text', 'date', 'predicted_form'])
        writer.writeheader()
        for p in predictions:
            writer.writerow({k: exact_json(p[k]) if k == 'predicted_form' else p[k] for k in writer.fieldnames})
    write_report(args.output / 'inspection.html', rows, predictions, aliases)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
