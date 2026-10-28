import csv
import tempfile
import unittest
from pathlib import Path

import baseline


def row(identifier, text, label='true', date='2026-08-01'):
    return {'id': identifier, 'text': text, 'label': label, 'date': date}


class BaselineTests(unittest.TestCase):
    def test_checker_split_is_deterministic_and_disjoint_by_id(self):
        rows = [row(f'r{i}', f'text {i}', ('true', 'false', 'unsure')[i % 3])
                for i in range(30)]
        train1, test1 = baseline.split_rows(rows, 'checker')
        train2, test2 = baseline.split_rows(rows, 'checker')
        train_ids, test_ids = {r['id'] for r in train1}, {r['id'] for r in test1}
        self.assertEqual(train_ids, {r['id'] for r in train2})
        self.assertEqual(test_ids, {r['id'] for r in test2})
        self.assertFalse(train_ids & test_ids)
        self.assertEqual(train_ids | test_ids, {r['id'] for r in rows})

    def test_tfidf_vocabulary_is_learned_from_training_text_only(self):
        train = [row('a', 'shared alpha', 'true'), row('b', 'shared beta', 'false')]
        held_out = [row('c', 'shared quarantineword')]
        model = baseline.make_model('tfidf_lr')
        model.fit([r['text'] for r in train], [r['label'] for r in train])
        vocabulary = model.named_steps['tfidf'].vocabulary_
        self.assertIn('shared', vocabulary)
        self.assertNotIn('quarantineword', vocabulary)
        baseline.predict_rows(model, held_out)
        self.assertNotIn('quarantineword', model.named_steps['tfidf'].vocabulary_)

    def test_read_rows_discards_forms_and_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'rows.csv'
            with path.open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=['id', 'text', 'date', 'label', 'form'])
                writer.writeheader()
                writer.writerow({'id': 'x', 'text': 'claim', 'date': '2026-08-01',
                                 'label': ' TRUE ', 'form': 'do not use'})
            loaded = baseline.read_rows(path)
            self.assertEqual(loaded, [row('x', 'claim')])
            with path.open('a', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=['id', 'text', 'date', 'label', 'form'])
                writer.writerow({'id': 'x', 'text': 'duplicate', 'date': '2026-08-02',
                                 'label': 'false', 'form': 'ignored'})
            with self.assertRaisesRegex(ValueError, 'duplicate id'):
                baseline.read_rows(path)

    def test_prediction_emits_only_unlabelled_input_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            train_path, input_path, output_path = (Path(temp) / name for name in
                                                   ('train.csv', 'input.csv', 'out.csv'))
            fields = ['id', 'text', 'date', 'label']
            train_rows = [row(f't{i}', text, label) for i, (text, label) in enumerate([
                ('the verified launch succeeded', 'true'), ('verified evidence exists', 'true'),
                ('the successful launch works', 'true'), ('confirmed launch facts', 'true'),
                ('the false report failed', 'false'), ('report was false', 'false'),
                ('the uncertain launch details', 'unsure'), ('uncertain claim details', 'unsure')])]
            input_rows = [row('known', 'the verified launch succeeded', 'true'),
                          row('pending', 'an unseen claim', '')]
            for path, values in ((train_path, train_rows), (input_path, input_rows)):
                with path.open('w', newline='', encoding='utf-8') as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(values)
            import subprocess
            import sys
            result = subprocess.run(
                [sys.executable, 'baseline.py', 'predict', '--train', str(train_path),
                 '--input', str(input_path), '--output', str(output_path), '--model', 'always_true'],
                cwd=baseline.ROOT, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            with output_path.open(newline='', encoding='utf-8') as stream:
                predictions = list(csv.DictReader(stream))
            self.assertEqual([p['id'] for p in predictions], ['pending'])
            self.assertEqual(predictions[0]['label'], 'true')

    def test_unsure_predictions_have_blank_confidence_and_decisions_stay_in_bounds(self):
        class FixedModel:
            classes_ = ['false', 'true', 'unsure']

            def __init__(self, values):
                self.values = values

            def predict_proba(self, texts):
                return self.values

        rows = [row('native', 'native unsure'), row('low', 'low confidence'),
                row('edge', 'boundary confidence')]
        predictions = baseline.predict_rows(
            FixedModel([[0.1, 0.1, 0.8], [0.35, 0.4, 0.25], [0.0, 0.5, 0.5]]),
            rows, min_confidence=0.5)
        self.assertEqual(predictions[0]['label'], 'unsure')
        self.assertEqual(predictions[0]['confidence'], '')
        self.assertEqual(predictions[0]['decision_reason'], 'model_unsure')
        self.assertEqual(predictions[1]['label'], 'unsure')
        self.assertEqual(predictions[1]['confidence'], '')
        self.assertEqual(predictions[1]['decision_reason'], 'low_confidence')
        self.assertEqual(predictions[2]['label'], 'true')  # argmax tie follows class order
        for prediction in predictions:
            if prediction['label'] != 'unsure':
                self.assertGreaterEqual(prediction['confidence'], 0.5)
                self.assertLessEqual(prediction['confidence'], 1.0)


if __name__ == '__main__':
    unittest.main()
