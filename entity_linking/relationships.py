"""Offline POTION proposal ranking. Similarity is extraction evidence, not truth."""

import json
from pathlib import Path
import re

import numpy as np

from entity_linking.context import sentence_bounds, local_context, entities_for


def normalized_embeddings(model, texts):
    if not texts:
        return np.empty((0, model.dim), dtype=np.float32)
    vectors = np.asarray(model.encode(texts), dtype=np.float32)
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)


class LexiconRelationshipRanker:
    """Pure cosine against lexicon attribute names; no learned type/reference data."""

    def __init__(self, model, registry, threshold=.4, margin=.02):
        self.model = model
        self.entities = {e['id']: e for e in registry['entities'] if e.get('active', True)}
        self.threshold, self.margin = threshold, margin
        attributes = sorted({a for e in self.entities.values() for a in e['attributes']})
        self.references = [{'attribute': a, 'text': a.replace('_', ' '), 'source': 'lexicon_attribute'} for a in attributes]
        self.indices = {a: i for i, a in enumerate(attributes)}
        self.vectors = normalized_embeddings(model, [r['text'] for r in self.references])

    @classmethod
    def load(cls, model_path, registry_path, threshold=.4, margin=.02, lexicon_path=None):
        from model2vec import StaticModel
        model_path = Path(model_path)
        if not model_path.is_dir():
            raise FileNotFoundError(f'Local model directory missing: {model_path}')
        registry = json.loads(Path(registry_path).read_text())
        if lexicon_path is not None:
            from entity_linking import build_registry
            registry = build_registry(json.loads(Path(lexicon_path).read_text()), registry)
        return cls(StaticModel.from_pretrained(model_path), registry, threshold, margin)

    def _rank(self, scores, mentions):
        candidates = {}
        for mention in mentions:
            entity = self.entities[mention['entity_id']]
            for attribute in entity['attributes']:
                score = float(scores[self.indices[attribute]])
                candidate = {'entity_id': entity['id'], 'subject': entity['name'], 'attribute': attribute,
                             'similarity': score, 'score': score, 'entity_span': [mention['start'], mention['end']],
                             'matched_reference': self.references[self.indices[attribute]]}
                candidates.setdefault((entity['id'], attribute), candidate)
        ranked = sorted(candidates.values(), key=lambda c: (-c['score'], c['entity_id'], c['attribute']))
        gap = ranked[0]['score'] - ranked[1]['score'] if len(ranked) > 1 else None
        accepted = bool(ranked and ranked[0]['score'] >= self.threshold and (gap is None or gap >= max(self.margin, 1e-6)))
        status = 'accepted' if accepted else 'no_explicit_entity' if not ranked else 'low_score' if ranked[0]['score'] < self.threshold else 'ambiguous_candidates'
        return {'status': status, 'margin': gap, 'binding': ranked[0] if accepted else None, 'candidates': ranked[:3]}

    def propose(self, records):
        value_contexts, concept_contexts = [], []
        for record in records:
            contexts = [local_context(record, v) for v in record['values']]
            # Use actual source wording, without artificial kind/unit suffixes.
            for context in contexts:
                context['embedding_text'] = context['wording']
            value_contexts.append(contexts)
            concepts = []
            seen = set()
            for mention in record['mentions']:
                left, right = sentence_bounds(record, mention)
                key = mention['entity_id'], left, right
                if key in seen:
                    continue
                seen.add(key)
                wording = record['text'][left:right]
                for span in sorted(record['mentions'], key=lambda m: m['start'], reverse=True):
                    if left <= span['start'] and span['end'] <= right:
                        wording = wording[:span['start'] - left] + ' ' + wording[span['end'] - left:]
                wording = re.sub(r'\d+(?:[.,]\d+)*', ' ', wording)
                wording = re.sub(r'\W+', ' ', wording.casefold()).strip()
                concepts.append((mention, {'wording': wording, 'embedding_text': wording,
                                           'start': left, 'end': right, 'surface': record['text'][left:right]}))
            concept_contexts.append(concepts)
        texts = sorted({c['embedding_text'] for row in value_contexts for c in row} |
                       {c['embedding_text'] for row in concept_contexts for _, c in row})
        scores = normalized_embeddings(self.model, texts) @ self.vectors.T
        similarities = dict(zip(texts, scores))
        output = []
        for record, contexts, concepts in zip(records, value_contexts, concept_contexts):
            relationships = [
                {'value': value, 'context': context,
                 **self._rank(similarities[context['embedding_text']], entities_for(record, value)),
                 'method': 'potion_cosine_lexicon_only'}
                for value, context in zip(record['values'], contexts)]
            concept_matches = [
                {'entity_id': mention['entity_id'], 'context': context,
                 **self._rank(similarities[context['embedding_text']], [mention]),
                 'method': 'potion_cosine_lexicon_only', 'value': None}
                for mention, context in concepts]
            output.append({'id': record['id'], 'text': record['text'], 'date': record['date'],
                           'relationships': relationships, 'concept_matches': concept_matches,
                           'scope': 'lexicon_only_attribute_similarity; concept_matches_do_not_extract_values'})
        return output
