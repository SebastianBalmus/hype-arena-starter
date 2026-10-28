import unittest
from unittest.mock import patch

import numpy as np

from entity_linking import EntityLinker, build_registry
from entity_linking.forms import FormExtractor
from entity_linking.relationships import LexiconRelationshipRanker


class TrapEncoder:
    """Deliberately favors price/provider over version in fallback queries."""
    dim = 3

    def encode(self, texts):
        return np.asarray([[0, 1, 0] if text in {'version', 'api version', 'cli version'} else [1, 0, 0]
                           for text in texts], dtype=np.float32)


class IntegratedRuleTests(unittest.TestCase):
    def setUp(self):
        self.registry = build_registry({'subjects': [
            {'name': 'Assistant', 'aliases': [], 'attributes': ['version', 'api_version', 'price', 'cloud_provider']},
            {'name': 'Plan', 'aliases': [], 'attributes': ['price', 'annual_price', 'extra_seat_price', 'free_trial', 'refund_window', 'customers', 'regions']},
            {'name': 'Cloud', 'aliases': [], 'attributes': ['cli_version']},
            {'name': 'Android', 'aliases': [], 'attributes': ['version']},
            {'name': 'Phone', 'aliases': [], 'attributes': ['shipping_android']},
            {'name': 'Person', 'aliases': [], 'attributes': ['joined', 'previous_employer']},
            {'name': 'Other', 'aliases': [], 'attributes': ['price', 'customers']}], 'events': []})
        self.linker = EntityLinker(self.registry)
        self.ranker = LexiconRelationshipRanker(TrapEncoder(), self.registry)
        self.extractor = FormExtractor(self.linker, self.ranker)

    def extract(self, text):
        return self.extractor.extract([{'id': 'x', 'date': '2026-10-28', 'text': text}])[0]

    def test_version_overrides_embedding_and_skips_query_encoding(self):
        with patch.object(self.ranker.model, 'encode', side_effect=AssertionError('Direct rule must bypass embeddings')):
            result = self.extract('Assistant 1.5 ships today.')
        self.assertEqual(result['predicted_form'], [{'subject': 'Assistant', 'attribute': 'version', 'value': {'kind': 'version', 'value': '1.5'}}])
        self.assertEqual(result['traces'][0]['method'], 'structural_rule')
        self.assertIsNone(result['traces'][0]['binding']['score'])

    def test_api_cli_and_device_shipping_are_distinct(self):
        for text, subject, attribute in [('Assistant API v3 is available.', 'Assistant', 'api_version'),
                                          ('Cloud CLI 2.3 ships today.', 'Cloud', 'cli_version'),
                                          ('Phone ships with Android 17.0.', 'Phone', 'shipping_android')]:
            result = self.extract(text)
            self.assertEqual([(f['subject'], f['attribute']) for f in result['predicted_form']], [(subject, attribute)])

    def test_prices_are_selected_by_billing_and_seat_cues(self):
        for text, attribute in [('Plan costs €49 per month.', 'price'),
                                ('Plan costs €490 per year.', 'annual_price'),
                                ('Each further seat on Plan costs €15 per month.', 'extra_seat_price')]:
            result = self.extract(text)
            self.assertEqual([f['attribute'] for f in result['predicted_form']], [attribute])

    def test_count_and_inherited_subject(self):
        result = self.extract('Plan has 20 customers and covers 3 regions.')
        self.assertEqual([f['attribute'] for f in result['predicted_form']], ['customers', 'regions'])
        self.assertEqual(result['traces'][1]['subject_resolution'], 'same_sentence_unique_entity')

    def test_inheritance_does_not_cross_sentence_or_ambiguous_entities(self):
        self.assertEqual(self.extract('Plan is available. It covers 3 regions.')['predicted_form'], [])
        self.assertEqual(self.extract('Plan and Other are available, with 20 customers.')['predicted_form'], [])

    def test_trial_and_refund_duration_bindings_remain_separate(self):
        result = self.extract('Plan has a trial of 7 days and refunds are offered for 14 days.')
        self.assertEqual([f['attribute'] for f in result['predicted_form']], ['free_trial', 'refund_window'])

    def test_two_credible_price_rules_conflict_instead_of_falling_back(self):
        result = self.extract("Plan's annual extra seat price is €15 per year.")
        self.assertEqual(result['predicted_form'], [])
        trace = next(t for t in result['traces'] if t['value']['kind'] == 'money')
        self.assertEqual(trace['status'], 'rule_conflict')
        self.assertEqual({c['attribute'] for c in trace['candidates']}, {'annual_price', 'extra_seat_price'})

    def test_rules_do_not_bypass_negation_or_hypothetical_scope(self):
        for text in ['Plan does not cost €49 per month.', 'If Plan costs €49 per month, we subscribe.']:
            self.assertEqual(self.extract(text)['predicted_form'], [])

    def test_clear_name_syntax_is_accepted_and_broad_names_are_unresolved(self):
        accepted = self.extract('Assistant is hosted on Nimbus Labs.')
        self.assertEqual(accepted['predicted_form'], [{'subject': 'Assistant', 'attribute': 'cloud_provider', 'value': {'kind': 'name', 'value': 'Nimbus Labs'}}])
        unresolved = self.extract('Assistant is powered by Nimbus Labs.')
        self.assertEqual(unresolved['predicted_form'], [])
        self.assertTrue(any(t['status'] == 'name_requires_structural_evidence' for t in unresolved['traces']))

    def test_gold_and_labels_cannot_change_rule_output(self):
        row = {'id': 'x', 'date': '2026-10-28', 'text': 'Assistant 1.5 ships today.'}
        before = self.extractor.extract([row])
        row.update(label='false', form='malformed')
        self.assertEqual(self.extractor.extract([row]), before)


if __name__ == '__main__':
    unittest.main()
