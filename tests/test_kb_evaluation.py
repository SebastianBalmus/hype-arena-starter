"""Focused evaluator checks that do not invoke the KB method or run an experiment."""

import unittest

from evaluation import run_kb_experiment as runner
from evaluation import run_text_experiments as helpers


class CalibrationTests(unittest.TestCase):
    def test_unsure_gold_counts_as_error_and_empty_class_uses_prior(self):
        raw = [
            {'id': 'u', 'label': 'true'},
            {'id': 't', 'label': 'true'},
        ]
        gold = [{'id': 'u', 'label': 'unsure'}, {'id': 't', 'label': 'true'}]
        calibrated = runner.calibration_from_validation(raw, gold)

        self.assertEqual(calibrated['true']['count'], 2)
        self.assertEqual(calibrated['true']['correct'], 1)
        self.assertEqual(calibrated['true']['confidence'], 0.5)
        self.assertEqual(calibrated['true']['confidence_status'], 'validation_calibrated')
        self.assertEqual(calibrated['false']['count'], 0)
        self.assertEqual(calibrated['false']['confidence'], 0.5)
        self.assertEqual(calibrated['false']['confidence_status'], 'prior_only')


class SanitizationBoundaryTests(unittest.TestCase):
    def test_pending_labels_are_blank_and_forms_are_training_only(self):
        source = [
            {'id': 'fit', 'text': 'known', 'date': '2026-01-01',
             'label': 'true', 'form': '{"source":"fit"}'},
            {'id': 'pending', 'text': 'pending', 'date': '2026-01-02',
             'label': 'false', 'form': '{"source":"pending"}'},
        ]
        inference = helpers.sanitized_rows(source, ['fit'], ['pending'])
        training_forms = runner.read_forms(source, ['fit'])

        self.assertEqual(inference[1]['label'], '')
        self.assertEqual([row['id'] for row in training_forms], ['fit'])
        self.assertNotIn('form', inference[1])
        self.assertNotIn('pending', {row['id'] for row in training_forms})
        with self.assertRaises(ValueError):
            runner.read_forms(source, ['missing'])


if __name__ == '__main__':
    unittest.main()
