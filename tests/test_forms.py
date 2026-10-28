from decimal import Decimal
import unittest
from unittest.mock import patch

import numpy as np

from entity_linking import EntityLinker, build_registry
from entity_linking.forms import FormExtractor, value_form
from entity_linking.relationships import LexiconRelationshipRanker
from entity_linking.values import exact_json


class FormEncoder:
    dim = 7

    def encode(self, texts):
        return np.asarray([[float(any(word in text.casefold() for word in group)) for group in [
            ['join', 'started'], ['previous', 'employer'], ['team', 'engineer'],
            ['cost', 'price'], ['open source'], ['image input'], ['__unused__']]]
            for text in texts], dtype=np.float32)


class FormTests(unittest.TestCase):
    def setUp(self):
        self.registry = build_registry({'subjects': [
            {'name': 'Holdan', 'aliases': [], 'attributes': ['joined', 'previous_employer', 'team_size']},
            {'name': 'Product', 'aliases': [], 'attributes': ['price', 'open_source', 'image_input', 'offline_editing', 'carbon_neutral', 'ai_answers', 'on_device_gemini']}], 'events': []})
        self.linker = EntityLinker(self.registry)
        self.extractor = FormExtractor(self.linker, LexiconRelationshipRanker(FormEncoder(), self.registry))

    def extract(self, text):
        return self.extractor.extract([{'id': 'x', 'text': text, 'date': '2026-10-28'}])[0]

    def test_emits_practice_shape_for_date_and_quantity(self):
        result = self.extract('Holdan joined on 12 January 2026 and Holdan has a team of 4 engineers.')
        self.assertEqual(result['predicted_form'], [
            {'subject': 'Holdan', 'attribute': 'joined', 'value': {'kind': 'date', 'value': '2026-01-12'}},
            {'subject': 'Holdan', 'attribute': 'team_size', 'value': {'kind': 'quantity', 'value': 4, 'unit': 'engineer'}}])

    def test_name_can_be_extracted_from_source_without_being_a_lexicon_entity(self):
        result = self.extract("Holdan's previous employer is Nimbus Cloud.")
        self.assertIn({'subject': 'Holdan', 'attribute': 'previous_employer',
                       'value': {'kind': 'name', 'value': 'Nimbus Cloud'}}, result['predicted_form'])
        trace = next(t for t in result['traces'] if t['value']['kind'] == 'name')
        self.assertEqual(result['text'][trace['value']['start']:trace['value']['end']], 'Nimbus Cloud')

    def test_boolean_negation_is_local(self):
        result = self.extract('Product is open source but Product does not support image input.')
        self.assertEqual(result['predicted_form'], [
            {'subject': 'Product', 'attribute': 'open_source', 'value': {'kind': 'boolean', 'value': True}},
            {'subject': 'Product', 'attribute': 'image_input', 'value': {'kind': 'boolean', 'value': False}}])

    def test_has_no_feature_is_negative_without_embedding_query(self):
        with patch.object(self.extractor.ranker.model, 'encode', side_effect=AssertionError('Boolean rule must not encode a query')):
            result = self.extract('Product has no image input yet.')
        self.assertEqual(result['predicted_form'], [
            {'subject': 'Product', 'attribute': 'image_input', 'value': {'kind': 'boolean', 'value': False}}])
        self.assertEqual(result['traces'][0]['method'], 'literal_boolean_predicate')
        self.assertIsNone(result['traces'][0]['binding']['similarity'])

    def test_conversational_no_does_not_negate_following_predicate(self):
        result = self.extract('No, Product supports image input.')
        self.assertEqual(result['predicted_form'][0]['value']['value'], True)

    def test_feature_before_subject_and_postfix_negation(self):
        for text in ['Image input is not enabled in Product.', 'Image input is disabled in Product.']:
            result = self.extract(text)
            self.assertEqual(result['predicted_form'], [
                {'subject': 'Product', 'attribute': 'image_input', 'value': {'kind': 'boolean', 'value': False}}])

    def test_contraction_and_two_clauses_have_separate_polarities(self):
        result = self.extract("Product isn't open source but Product supports image input.")
        self.assertEqual([f['value']['value'] for f in result['predicted_form']], [False, True])

    def test_ambiguous_negation_and_fee_statement_do_not_assert_feature(self):
        for text in ['Product is not only open source.', 'Product is not necessarily open source.',
                     'Product does not charge for image input.', 'Image input is not disabled in Product.']:
            self.assertEqual(self.extract(text)['predicted_form'], [])

    def test_offline_wording_has_positive_and_negative_polarity(self):
        for text, expected in [('Product works offline.', True), ('Product does not work offline.', False),
                               ("Product doesn't work offline.", False), ('No, Product works offline.', True),
                               ('Product has no problems working offline.', True)]:
            result = self.extract(text)
            self.assertEqual(result['predicted_form'], [
                {'subject': 'Product', 'attribute': 'offline_editing', 'value': {'kind': 'boolean', 'value': expected}}])
        self.assertEqual(self.extract('Product does not promise to work offline.')['predicted_form'], [])

    def test_additional_boolean_wording_is_lexicon_constrained(self):
        for text, attribute in [('Product runs carbon neutral.', 'carbon_neutral'),
                                ('Product writes AI answers above links.', 'ai_answers'),
                                ('Product runs Gemini on the device.', 'on_device_gemini')]:
            result = self.extract(text)
            self.assertIn({'subject': 'Product', 'attribute': attribute, 'value': {'kind': 'boolean', 'value': True}}, result['predicted_form'])
        self.assertEqual(self.extract('Holdan works offline.')['predicted_form'], [])

    def test_negated_numeric_is_not_a_positive_form(self):
        result = self.extract('Product does not cost €49 per month.')
        self.assertEqual(result['predicted_form'], [])
        self.assertTrue(any(t['status'] == 'negated_non_boolean_assertion' for t in result['traces']))

    def test_hypothetical_and_alternative_do_not_become_flat_facts(self):
        for text in ['If Product costs €49 per month, we will subscribe.', 'Product costs €49 or €59 per month.']:
            self.assertEqual(self.extract(text)['predicted_form'], [])

    def test_implicit_subject_stays_unresolved(self):
        result = self.extract('He has a team of 4 engineers.')
        self.assertEqual(result['predicted_form'], [])
        self.assertIn({'stage': 'subject', 'status': 'no_explicit_subject'}, result['unresolved'])

    def test_exact_values_operator_and_no_annotation_metadata(self):
        value = {'kind': 'money', 'value': Decimal('49.123456789123456789'), 'currency': 'EUR', 'per': 'month',
                 'op': 'gt', 'surface': 'over EUR 49.123456789123456789/month', 'start': 0, 'basis': 'seat'}
        result = value_form(value)
        self.assertEqual(set(result), {'kind', 'value', 'currency', 'per', 'op'})
        self.assertIn('49.123456789123456789', exact_json(result))
        self.assertNotIn('per', value_form({'kind': 'money', 'value': 49, 'currency': 'EUR', 'per': None}))

    def test_inference_is_independent_of_gold_form_and_label(self):
        row = {'id': 'x', 'text': 'Product costs €49 per month.', 'date': '2026-10-28'}
        before = self.extractor.extract([row])
        row.update(form='not valid JSON', label='false')
        self.assertEqual(self.extractor.extract([row]), before)


if __name__ == '__main__':
    unittest.main()
