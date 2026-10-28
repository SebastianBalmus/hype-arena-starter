from copy import deepcopy
from decimal import Decimal
from itertools import product
import json
from pathlib import Path
import tempfile
import unittest

from entity_linking.knowledge_base import EvidenceIndex
from entity_linking.json_utils import exact_json


LEXICON = {'subjects': [{'name': 'Product', 'aliases': ['Alias'], 'attributes': ['price', 'version', 'enabled']}], 'events': []}


def row(rid='one', label='true', day='2026-01-01', amount=Decimal('49.123456789123456789')):
    return {'id': rid, 'label': label, 'text': f'Product costs {amount} EUR.', 'date': day,
            'form': exact_json([{'subject': 'Product', 'attribute': 'price',
                                 'value': {'kind': 'money', 'value': amount, 'currency': 'EUR', 'per': 'month'}}])}


class KnowledgeBaseTests(unittest.TestCase):
    def test_saved_index_is_multiline_and_preserves_exact_decimal(self):
        index = EvidenceIndex.from_rows([row()], LEXICON)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'index.json'
            index.save(path)
            saved = path.read_text()
            self.assertIn('\n  "sources": [\n', saved)
            self.assertIn('49.123456789123456789', saved)
            self.assertEqual(EvidenceIndex.load(path, LEXICON).to_dict(), index.to_dict())

    def test_all_labels_are_stored_but_default_lookup_only_returns_true_evidence(self):
        rows = [row('t'), row('f', 'false', amount=59), row('u', 'unsure', amount=69), row('blank', '', amount=79)]
        index = EvidenceIndex.from_rows(rows, LEXICON, source_policy='all_labels')
        self.assertEqual(index.summary()['source_rows'], 4)
        self.assertEqual(index.summary()['observations'], 4)
        self.assertEqual({o['source_label'] for o in index.to_dict()['observations']}, {'true', 'false', 'unsure', ''})
        self.assertEqual([o['source_id'] for o in index.lookup('Product', 'price', '2026-01-01')['observations']], ['t'])
        self.assertEqual(len(index.lookup('Product', 'price', '2026-01-01', include_untrusted=True)['observations']), 4)
        self.assertEqual(len(index.history('Product', 'price')), 1)
        self.assertEqual(len(index.history('Product', 'price', include_untrusted=True)), 4)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'index.json'
            index.save(path)
            loaded = EvidenceIndex.load(path, LEXICON)
            self.assertEqual(loaded.to_dict(), index.to_dict())
            self.assertEqual(len(loaded.lookup('Product', 'price', '2026-01-01')['observations']), 1)


    def test_invalid_source_policy_is_rejected(self):
        with self.assertRaises(ValueError):
            EvidenceIndex.from_rows([], LEXICON, source_policy='everything_is_true')

    def test_only_true_rows_enter_and_malformed_excluded_forms_are_never_read(self):
        rows = [row()]
        for label in ['false', 'unsure', '']:
            r = row(label or 'empty', label=label); r['form'] = 'invalid JSON'; rows.append(r)
        index = EvidenceIndex.from_rows(rows, LEXICON)
        self.assertEqual(index.summary()['observations'], 1)
        self.assertEqual(index.summary()['source_labels'], {'true': 1})

    def test_no_temporal_carry_forward_and_alias_lookup(self):
        index = EvidenceIndex.from_rows([row()], LEXICON)
        self.assertEqual(len(index.lookup('ALIAS', 'price', '2026-01-01')['observations']), 1)
        self.assertEqual(index.lookup('Product', 'price', '2026-01-02')['observations'], [])
        self.assertEqual(index.lookup('Product', 'price', '2025-12-31')['observations'], [])
        self.assertEqual(len(index.history('Product', 'price')), 1)

    def test_multiple_values_and_repeated_sources_are_preserved(self):
        index = EvidenceIndex.from_rows([row('a', amount=49), row('b', amount=59), row('c', amount=49)], LEXICON)
        found = index.lookup('Product', 'price', '2026-01-01')
        self.assertEqual(len(found['observations']), 3)
        self.assertEqual(found['distinct_predicates'], 2)

    def test_exact_decimal_operator_and_partial_date_payload_survive_roundtrip(self):
        original = row()
        facts = json.loads(original['form'], parse_float=Decimal); facts[0]['value']['op'] = 'gt'; original['form'] = exact_json(facts)
        index = EvidenceIndex.from_rows([original], LEXICON)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'index.json'; index.save(path); loaded = EvidenceIndex.load(path, LEXICON)
        value = loaded.lookup('Product', 'price', '2026-01-01')['observations'][0]['value']
        self.assertEqual(value['value'], Decimal('49.123456789123456789'))
        self.assertEqual(value['op'], 'gt')
        self.assertEqual(value['per'], 'month')

    def test_true_row_can_assert_a_false_boolean_and_multiple_atoms(self):
        source = row()
        facts = json.loads(source['form'], parse_float=Decimal)
        facts.append({'subject': 'Product', 'attribute': 'enabled', 'value': {'kind': 'boolean', 'value': False}})
        source['form'] = exact_json(facts)
        index = EvidenceIndex.from_rows([source], LEXICON)
        self.assertEqual(index.summary()['observations'], 2)
        self.assertFalse(index.lookup('Product', 'enabled', '2026-01-01')['observations'][0]['value']['value'])

    def test_partial_date_precision_and_units_are_not_reinterpreted(self):
        lexicon = deepcopy(LEXICON); lexicon['subjects'][0]['attributes'] += ['launched', 'storage']
        source = row()
        source['form'] = exact_json([
            {'subject': 'Product', 'attribute': 'launched', 'value': {'kind': 'date', 'value': '2025-11'}},
            {'subject': 'Product', 'attribute': 'storage', 'value': {'kind': 'quantity', 'value': 1, 'unit': 'GB'}}])
        index = EvidenceIndex.from_rows([source], lexicon)
        self.assertEqual(index.lookup('Product', 'launched', '2026-01-01')['observations'][0]['value']['value'], '2025-11')
        self.assertEqual(index.lookup('Product', 'storage', '2026-01-01')['observations'][0]['value']['unit'], 'GB')
        self.assertEqual(index.lookup('Product', 'launched', '2025-11-01')['observations'], [])

    def test_packaged_forms_require_true_label_and_exact_id_text_date(self):
        original = row(); packaged = EvidenceIndex.from_rows([original], LEXICON)
        runtime = {k:v for k,v in original.items() if k != 'form'}
        self.assertEqual(EvidenceIndex.from_rows([runtime], LEXICON, packaged).summary()['observations'], 1)
        for field, changed in [('label', ''), ('label', 'false'), ('label', 'unsure'), ('id', 'reused'), ('text', 'Different sentence'), ('date', '2026-01-02')]:
            altered = {**runtime, field: changed}
            self.assertEqual(EvidenceIndex.from_rows([altered], LEXICON, packaged).summary()['observations'], 0)

    def test_runtime_filtered_packaged_index_can_be_saved_and_reloaded(self):
        source = row(); packaged = EvidenceIndex.from_rows([source], LEXICON); del source['form']
        index = EvidenceIndex.from_rows([source], LEXICON, packaged)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'index.json'; index.save(path)
            loaded = EvidenceIndex.load(path, LEXICON)
        self.assertEqual(loaded.to_dict(), index.to_dict())

    def test_query_results_and_inputs_cannot_mutate_index(self):
        source = row(); index = EvidenceIndex.from_rows([source], LEXICON)
        source['text'] = 'changed'
        result = index.lookup('Product', 'price', '2026-01-01'); result['observations'][0]['value']['value'] = 0
        self.assertNotEqual(index.lookup('Product', 'price', '2026-01-01')['observations'][0]['value']['value'], 0)

    def test_unknown_attribute_invalid_value_and_duplicate_id_are_rejected(self):
        for facts in [[{'subject':'Product', 'attribute':'invented', 'value':{'kind':'boolean','value':True}}],
                      [{'subject':'Product', 'attribute':'price', 'value':{'kind':'money','value':True,'currency':'EUR'}}]]:
            r = row(); r['form'] = exact_json(facts)
            with self.assertRaises(ValueError): EvidenceIndex.from_rows([r], LEXICON)
        with self.assertRaises(ValueError): EvidenceIndex.from_rows([row(),row()], LEXICON)

    def test_saved_observations_or_source_labels_cannot_be_tampered(self):
        payload = EvidenceIndex.from_rows([row()], LEXICON).to_dict()
        for alteration in ['observation','label','text']:
            changed = deepcopy(payload)
            if alteration == 'observation': changed['observations'][0]['value']['value'] = 0
            elif alteration == 'label': changed['sources'][0]['label'] = 'false'
            else: changed['sources'][0]['text'] = 'changed'
            with tempfile.TemporaryDirectory() as d:
                path = Path(d) / 'index.json'; path.write_text(exact_json(changed))
                with self.assertRaises(ValueError): EvidenceIndex.load(path, LEXICON)

    def test_label_gating_and_order_invariance_over_all_small_label_mixtures(self):
        for labels in product(['true','false','unsure',''], repeat=3):
            rows = [row(str(i), label=label, amount=i+1) for i,label in enumerate(labels)]
            for ordering in [rows, list(reversed(rows))]:
                found = EvidenceIndex.from_rows(ordering, LEXICON).lookup('Product','price','2026-01-01')['observations']
                self.assertEqual({o['source_id'] for o in found}, {str(i) for i,label in enumerate(labels) if label == 'true'})


if __name__ == '__main__':
    unittest.main()
