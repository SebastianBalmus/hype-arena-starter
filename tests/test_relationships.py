import unittest

import numpy as np

from entity_linking import EntityLinker, build_registry
from entity_linking.relationships import LexiconRelationshipRanker, local_context
from entity_linking.values import annotate


class TinyEncoder:
    """Deterministic semantic stand-in; tests decisions independently of POTION."""
    dim = 3

    def encode(self, texts):
        return np.array([[float('cost' in text or 'price' in text),
                          float('retain' in text or 'history' in text),
                          .01] for text in texts])


class RelationshipTests(unittest.TestCase):
    def setUp(self):
        self.registry = build_registry({'subjects': [
            {'name': 'Alpha', 'aliases': [], 'attributes': ['price', 'history_retention']},
            {'name': 'Beta', 'aliases': [], 'attributes': ['price']}], 'events': []})
        self.linker = EntityLinker(self.registry)

    def record(self, text):
        return {'id': 'example', 'date': '2026-10-28', 'text': text, **annotate(text, self.linker)}








    def test_context_spans_preserve_original_wording(self):
        row = self.record('Alpha retains history for 90 days.')
        value = row['values'][0]
        result = local_context(row, value)
        self.assertEqual(row['text'][result['start']:result['end']], result['surface'])
        self.assertNotIn('alpha', result['embedding_text'])


    def test_lexicon_only_references_include_every_attribute_without_training(self):
        self.registry['entities'][0]['attributes'].append('previous_employer')
        ranker = LexiconRelationshipRanker(TinyEncoder(), self.registry)
        self.assertEqual({r['attribute']: r['text'] for r in ranker.references},
                         {'price': 'price', 'history_retention': 'history retention', 'previous_employer': 'previous employer'})
        self.assertTrue(all(set(r) == {'attribute', 'text', 'source'} for r in ranker.references))

    def test_lexicon_only_has_no_type_filter_or_distance_penalty(self):
        ranker = LexiconRelationshipRanker(TinyEncoder(), self.registry)
        # Deliberately odd money/history combination still ranks every attribute:
        # the untyped lexicon supplies no basis for a hard type exclusion.
        row = self.record('Alpha retains history for €49 per month.')
        result = ranker.propose([row])[0]['relationships'][0]
        self.assertEqual({c['attribute'] for c in result['candidates']}, {'price', 'history_retention'})
        self.assertTrue(all(c['score'] == c['similarity'] and 'attachment_penalty' not in c for c in result['candidates']))
        self.assertEqual(result['binding']['attribute'], 'history_retention')

    def test_lexicon_only_ranks_concepts_without_a_typed_value(self):
        ranker = LexiconRelationshipRanker(TinyEncoder(), self.registry)
        result = ranker.propose([self.record('Alpha retains history.')])[0]
        self.assertEqual(result['relationships'], [])
        self.assertEqual(result['concept_matches'][0]['binding']['attribute'], 'history_retention')
        self.assertIsNone(result['concept_matches'][0]['value'])

    def test_lexicon_only_shared_attribute_owners_abstain(self):
        ranker = LexiconRelationshipRanker(TinyEncoder(), self.registry)
        result = ranker.propose([self.record('Alpha and Beta cost €49 per month.')])[0]['relationships'][0]
        self.assertEqual(result['status'], 'ambiguous_candidates')
        self.assertIsNone(result['binding'])

    def test_lexicon_only_ignores_form_and_label(self):
        ranker = LexiconRelationshipRanker(TinyEncoder(), self.registry)
        row = self.record('Alpha retains history for 90 days.')
        expected = ranker.propose([row])
        row.update(form='invalid', label='false')
        self.assertEqual(ranker.propose([row]), expected)


if __name__ == '__main__':
    unittest.main()
