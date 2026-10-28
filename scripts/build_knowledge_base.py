"""Build an offline index with source labels and dated assertion provenance."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from entity_linking.report import write_kb_report as write_report

from entity_linking.knowledge_base import EvidenceIndex
from entity_linking.io import read_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'data/practice.csv')
    parser.add_argument('--lexicon', type=Path, default=ROOT / 'data/lexicon.json')
    parser.add_argument('--packaged', type=Path, help='Verified packaged forms for runtime input without forms')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/knowledge_base')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh output directory')
    lexicon = json.loads(args.lexicon.read_text())
    rows, _ = read_rows(args.input)
    packaged = EvidenceIndex.load(args.packaged, lexicon) if args.packaged else None
    index = EvidenceIndex.from_rows(rows, lexicon, packaged, source_policy='true_only')
    # Independent development-time contract validation, outside runtime lookup.
    from jsonschema import Draft202012Validator
    validator = Draft202012Validator(json.loads((ROOT / 'contracts/claimed_value.schema.json').read_text()))
    for observation in index.to_dict()['observations']:
        validator.validate(observation['value'])
        assert observation['source_label'] == 'true'
    args.output.mkdir(parents=True)
    path = args.output / 'knowledge_base.json'
    index.save(path)
    summary = {**index.summary(), 'policy': 'Only trusted true rows; no false, unsure or unlabelled assertions',
               'form_source': 'Provided forms or fingerprint-verified packaged forms; never predicted forms',
               'temporal_policy': 'Row dates are observation dates; no carry-forward or inferred effective dates',
               'value_schema_validation_passed': True,
               'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
               'knowledge_base_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
               'runtime_code_sha256': hashlib.sha256((ROOT / 'entity_linking/knowledge_base.py').read_bytes()).hexdigest()}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    write_report(index, args.output / 'inspection.html')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
