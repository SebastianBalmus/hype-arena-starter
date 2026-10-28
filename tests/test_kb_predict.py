"""Integration checks for the evidence boundary and emitted reasoning traces."""

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from experiments.kb_predict import KnowledgeBasePredictor, match_training_forms

ROOT = Path(__file__).resolve().parents[1]
LEXICON = {'subjects': [{'name': 'Orbit', 'aliases': [], 'attributes': ['users']}]}


def row(rid, text, label='true', number=50, day='2026-01-01'):
    return {'id': rid, 'text': text, 'date': day, 'label': label,
            'form': json.dumps([{'subject': 'Orbit', 'attribute': 'users',
                                 'value': {'kind': 'quantity', 'value': number, 'unit': 'user'}}])}


def runtime(source):
    return {k: source[k] for k in ('id', 'text', 'date', 'label')}


class KnowledgeBasePredictorTests(unittest.TestCase):
    def setUp(self):
        self.training = row('train', 'Orbit serves 50 users.')

    def test_support_and_refutation_have_source_traces(self):
        model = KnowledgeBasePredictor([self.training], LEXICON)
        pending = [runtime(row('same', 'Orbit serves 50 users.', '')),
                   runtime(row('wrong', 'Orbit serves 60 users.', ''))]
        predictions, traces = model.predict(pending)
        self.assertEqual([p['label'] for p in predictions], ['true', 'false'])
        self.assertEqual([t['route'] for t in traces], ['kb', 'kb'])
        self.assertEqual(traces[1]['source_row_ids_and_fingerprints'][0]['id'], 'train')
        self.assertEqual(Decimal(traces[1]['parsed_assertions'][0]['value']['value']), 60)
        self.assertEqual(traces[1]['proof_status'], 'not_verified')

    def test_no_evidence_outside_observation_date(self):
        model = KnowledgeBasePredictor([self.training], LEXICON)
        predictions, traces = model.predict([runtime(row('future', 'Orbit serves 50 users.', '',
                                                         day='2026-01-02'))])
        self.assertEqual(predictions[0]['label'], 'unsure')
        self.assertIsNone(predictions[0]['confidence'])
        self.assertEqual(traces[0]['kb_reason'], 'missing_evidence')

    def test_validation_confidence_below_half_abstains(self):
        model = KnowledgeBasePredictor([self.training], LEXICON,
                                       calibration={'true': {'confidence': .4},
                                                    'false': {'confidence': .8}})
        predictions, traces = model.predict([runtime(row('pending', 'Orbit serves 50 users.', ''))])
        self.assertEqual(predictions[0]['label'], 'unsure')
        self.assertEqual(predictions[0]['decision_reason'], 'low_confidence')
        self.assertEqual(traces[0]['raw_kb_label'], 'true')

    def test_false_training_form_teaches_parser_without_positive_fact(self):
        source = row('bad', 'Orbit serves 50 users.', 'false')
        model = KnowledgeBasePredictor([source], LEXICON)
        predictions, traces = model.predict([runtime(row('pending', 'Orbit serves 60 users.', ''))])
        self.assertTrue(traces[0]['parse']['complete'])
        self.assertEqual(predictions[0]['label'], 'unsure')

    def test_pending_forms_and_reused_ids_are_rejected(self):
        pending = row('pending', 'Orbit serves 60 users.', '')
        inputs = [runtime(self.training), runtime(pending)]
        with self.assertRaisesRegex(ValueError, 'labelled runtime IDs'):
            match_training_forms(inputs, [self.training, pending])
        changed = {**self.training, 'text': 'Orbit serves 51 users.'}
        with self.assertRaisesRegex(ValueError, 'fingerprint mismatch'):
            match_training_forms(inputs, [changed])

    def test_hybrid_routes_only_unresolved_rows_to_text(self):
        class Fallback:
            classes_ = ['false', 'true', 'unsure']

            def predict_proba(self, texts):
                return [[.8, .1, .1] for _ in texts]

        model = KnowledgeBasePredictor([self.training], LEXICON)
        model.fallback = Fallback()
        pending = [runtime(row('supported', 'Orbit serves 50 users.', '')),
                   runtime(row('unknown', 'A wholly new sentence.', ''))]
        predictions, traces = model.predict(pending)
        self.assertEqual([p['label'] for p in predictions], ['true', 'false'])
        self.assertEqual([t['route'] for t in traces], ['kb', 'text_fallback'])
        self.assertEqual(traces[1]['source_row_ids_and_fingerprints'], [])
        self.assertEqual(traces[1]['fallback']['confidence_status'], 'uncalibrated')

    def test_cli_outputs_pending_ids_without_gold_columns(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            inputs = [runtime(self.training), runtime(row('pending', 'Orbit serves 60 users.', ''))]
            for name, rows in [('runtime.csv', inputs), ('forms.csv', [self.training])]:
                with (temp / name).open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            (temp / 'lexicon.json').write_text(json.dumps(LEXICON))
            subprocess.run([sys.executable, str(ROOT / 'experiments/kb_predict.py'),
                            '--input', str(temp / 'runtime.csv'), '--training-forms', str(temp / 'forms.csv'),
                            '--lexicon', str(temp / 'lexicon.json'), '--output', str(temp / 'output.csv'),
                            '--artifacts', str(temp / 'artifacts')], check=True, capture_output=True, text=True)
            with (temp / 'output.csv').open() as stream:
                result = list(csv.DictReader(stream))
            self.assertEqual(result[0]['id'], 'pending')
            self.assertEqual(result[0]['label'], 'false')
            self.assertEqual(set(result[0]), {'id', 'label', 'confidence', 'decision_reason'})
            records = [json.loads(line) for line in (temp / 'artifacts/kb_records.jsonl').read_text().splitlines()]
            self.assertEqual({r['id'] for r in records}, {'train'})


if __name__ == '__main__':
    unittest.main()
