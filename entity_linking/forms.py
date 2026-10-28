"""Sentence-to-form baseline with lexicon-only attribute similarity."""

import re

from entity_linking.relationships import normalized_embeddings
from entity_linking.candidates import (
    boolean_candidates, boolean_wording_candidates, clause_bounds, name_candidates,
    scope_reason, typed_candidates, value_form,
)
from entity_linking.structural_rules import prepare_candidate, resolve_choices
from entity_linking.values import annotate, exact_json


class FormExtractor:
    def __init__(self, linker, ranker):
        self.linker, self.ranker = linker, ranker

    def extract(self, rows):
        # Explicit field selection keeps all supplied labels/forms out of inference.
        records = [{'id': row['id'], 'date': row['date'], 'text': row['text'], **annotate(row['text'], self.linker)} for row in rows]
        candidates_by_row = [typed_candidates(r) + name_candidates(r) + boolean_candidates(r, self.ranker.entities)
                             + boolean_wording_candidates(r, self.ranker.entities) for r in records]
        candidates_by_row = [[prepare_candidate(record, candidate, self.ranker.entities, clause_bounds) for candidate in group]
                             for record, group in zip(records, candidates_by_row)]
        texts = sorted({p['context']['embedding_text'] for group in candidates_by_row for p in group if not p['rule_choices']})
        vectors = normalized_embeddings(self.ranker.model, texts) @ self.ranker.vectors.T
        scores = dict(zip(texts, vectors))
        results = []
        for record, candidates in zip(records, candidates_by_row):
            traces = []
            for candidate in candidates:
                method = 'potion_cosine_lexicon_only'
                if candidate['rule_choices']:
                    result = resolve_choices(candidate['rule_choices'], record, self.ranker.entities)
                    if candidate.get('deterministic'):
                        method = 'explicit_boolean_wording' if candidate['value']['rule'].startswith('explicit_boolean_wording:') else 'literal_boolean_predicate'
                    else:
                        method = 'structural_rule'
                else:
                    result = self.ranker._rank(scores[candidate['context']['embedding_text']], candidate['owners'])
                    if candidate.get('rule_blocked'):
                        result.update(binding=None, fallback_status=result['status'], status=candidate['rule_blocked'])
                if result['binding'] and candidate.get('required_attribute') and result['binding']['attribute'] != candidate['required_attribute']:
                    result.update(binding=None, status='boolean_predicate_attribute_mismatch')
                traces.append({'value': candidate['value'], 'context': candidate['context'], **result,
                               'stage': candidate.get('stage', candidate['value']['kind']), 'method': method,
                               'subject_resolution': candidate.get('subject_resolution', 'explicit_local_pattern')})
            forms, candidate_forms, seen = [], [], set()
            for trace in traces:
                value = value_form(trace['value'])
                best = trace['candidates'][0] if trace['candidates'] else None
                trace['candidate_form'] = {'subject': best['subject'], 'attribute': best['attribute'], 'value': value} if best else None
                if trace['candidate_form']:
                    candidate_forms.append(trace['candidate_form'])
                reason = scope_reason(record, trace['value'])
                if reason:
                    trace.update(binding=None, status=reason)
                trace['accepted_form'] = None
                if trace['binding']:
                    form = {'subject': trace['binding']['subject'], 'attribute': trace['binding']['attribute'], 'value': value}
                    key = exact_json(form)
                    if key not in seen:
                        seen.add(key)
                        forms.append(form)
                    trace['accepted_form'] = form
            unresolved = [{'stage': t['stage'], 'status': t['status'], 'surface': t['value']['surface'],
                           'start': t['value']['start'], 'end': t['value']['end']} for t in traces if not t['accepted_form']]
            if not record['mentions']:
                unresolved.append({'stage': 'subject', 'status': 'no_explicit_subject'})
            if not traces:
                unresolved.append({'stage': 'assertion', 'status': 'no_supported_value_candidate'})
            results.append({'id': record['id'], 'text': record['text'], 'date': record['date'],
                            'predicted_form': forms, 'candidate_form': candidate_forms, 'unresolved': unresolved,
                            'traces': traces, 'normalized_text': record['normalized_text'],
                            'composition': {'status': 'unresolved' if re.search(r'\b(?:or|unless|if)\b', record['text'], re.I) else 'not_semantically_resolved',
                                            'source_text': record['text']},
                            'method': 'structural_rules_then_lexicon_similarity',
                            'extraction_complete': False})
        return results
