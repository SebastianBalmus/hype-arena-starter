"""Typed, named and Boolean assertion candidates and source-scope checks."""

import re
from entity_linking.context import local_context, sentence_bounds


VALUE_FIELDS = {
    'quantity': ('unit',), 'duration': ('unit',), 'money': ('currency', 'per'),
    'percent': (), 'date': (), 'version': (), 'name': (), 'boolean': (),
}


NAME_PATTERN = re.compile(r"\b[A-Z][A-Za-z’'&-]*(?:\s+[A-Z][A-Za-z’'&-]*){0,4}\b")


STOP_NAMES = set(('Before From The This Today On In Our Release Host Board Meanwhile After Since As At '
                  'It He She His Her They Their We Its January February March April May June July August '
                  'September October November December Monday Tuesday Wednesday Thursday Friday Saturday Sunday').split())


NEGATION = re.compile(r"\b(?:not|never|no longer|without)\b|n['’]t\b", re.I)


HYPOTHETICAL = re.compile(r'\b(?:if|might|could|would|hypothetically)\b', re.I)


BOOLEAN_WORDING_RULES = {
    'offline_editing': r'\b(?:works?|working|edits?|editing)\s+offline\b',
    'carbon_neutral': r'\b(?:runs?|operates?)\s+(?:not\s+)?carbon[\s-]+neutral\b',
    'ai_answers': r'\b(?:writes?|shows?|displays?)\s+(?:no\s+)?AI\s+answers\b',
    'on_device_gemini': r'\b(?:runs?|uses?)\s+Gemini\s+on\s+(?:the\s+)?device\b',
}


def value_form(value):
    """Strip annotation metadata; retain exact contract fields and operators."""
    kind = value['kind']
    result = {'kind': kind, 'value': value['value']}
    for key in VALUE_FIELDS[kind]:
        if key in value and value[key] is not None:
            result[key] = value[key]
    if kind not in {'name', 'boolean'} and value.get('op', 'eq') != 'eq':
        result['op'] = value['op']
    return result


def clause_bounds(record, span):
    left, right = sentence_bounds(record, span)
    # Do not mistake commas in thousands for clause separators.
    boundaries = [m.span() for m in re.finditer(r';|,(?!\d)|\b(?:and|but|whereas|while)\b', record['text'][left:right], re.I)]
    start, end = left, right
    for a, b in boundaries:
        if left + b <= span['start']:
            start = max(start, left + b)
        elif left + a >= span['end']:
            end = min(end, left + a)
    return start, end


def scope_reason(record, value):
    left, right = clause_bounds(record, value)
    text = record['text'][left:right]
    if HYPOTHETICAL.search(text):
        return 'hypothetical_scope'
    if re.search(r'\bor\b', text, re.I):
        return 'alternative_scope'
    if value.get('approximate'):
        return 'approximate_value'
    if value.get('kind') != 'boolean' and (value.get('negated_prefix') or NEGATION.search(text)):
        return 'negated_non_boolean_assertion'
    return None


def typed_candidates(record):
    result = []
    for value in record['values']:
        context = local_context(record, value)
        left, right = clause_bounds(record, value)
        left, right = max(left, context['start']), min(right, context['end'])
        wording = record['text'][left:right]
        for mention in sorted(record['mentions'], key=lambda m: m['start'], reverse=True):
            if mention['start'] < right and left < mention['end']:
                a, b = max(left, mention['start']) - left, min(right, mention['end']) - left
                wording = wording[:a] + ' ' + wording[b:]
        wording = re.sub(r'\d+(?:[.,]\d+)*', ' ', wording)
        wording = re.sub(r'[^\w\s-]', ' ', wording)
        wording = re.sub(r'\s+', ' ', wording.casefold()).strip()
        context = {'start': left, 'end': right, 'surface': record['text'][left:right],
                   'wording': wording, 'embedding_text': wording}
        a, b = clause_bounds(record, value)
        owners = [m for m in record['mentions'] if a <= m['start'] and m['end'] <= b]
        result.append({'value': value, 'context': context, 'owners': owners, 'stage': 'typed_literal'})
    return result


def name_candidates(record):
    """Source-grounded names, including capitalized names absent from lexicon."""
    candidates = {(m['start'], m['end']): {**m, 'name': m['canonical_name'], 'target_entity_id': m['entity_id']}
                  for m in record['mentions']}
    protected = record['mentions'] + record['values']
    for match in NAME_PATTERN.finditer(record['text']):
        if any(match.start() < s['end'] and s['start'] < match.end() for s in protected):
            continue
        if match.group() in STOP_NAMES:
            continue
        candidates[match.span()] = {'start': match.start(), 'end': match.end(), 'surface': match.group(), 'name': match.group()}
    result = []
    for name in candidates.values():
        if len(name['name']) > 200:
            continue
        left, right = sentence_bounds(record, name)
        owners = [m for m in record['mentions'] if left <= m['start'] and m['end'] <= right
                  and m['entity_id'] != name.get('target_entity_id')
                  and (m['end'] <= name['start'] or name['end'] <= m['start'])]
        preceding = [m for m in owners if m['end'] <= name['start']]
        following = [m for m in owners if name['end'] <= m['start']]
        owner = None
        if preceding:
            proposed = max(preceding, key=lambda m: m['end'])
            between = record['text'][proposed['end']:name['start']]
            if len(between.split()) <= 16 and re.search(r"\b(?:is|was|at|from|by|in|of|named|called|with|as|to)\b|['’]s", between, re.I):
                owner = proposed
        if owner is None and following:
            proposed = min(following, key=lambda m: m['start'])
            between = record['text'][name['end']:proposed['start']]
            if len(between.split()) <= 12 and re.search(r"\b(?:is|was|of|at)\b", between, re.I):
                owner = proposed
        if owner is None:
            continue
        value = {'kind': 'name', 'value': name['name'], 'start': name['start'], 'end': name['end'],
                 'surface': record['text'][name['start']:name['end']], 'rule': 'source_name_and_local_relation'}
        context = local_context(record, value)
        # The target is a name, not relationship wording.
        wording = context['wording']
        if not name.get('target_entity_id'):
            wording = re.sub(re.escape(name['surface'].casefold()), ' ', wording)
        context['embedding_text'] = re.sub(r'\s+', ' ', wording).strip()
        result.append({'value': value, 'context': context, 'owners': [owner]})
    return result


def boolean_candidates(record, entities):
    """Conservative literal attribute-phrase predicates, with local polarity."""
    output = []
    seen = set()
    for mention in record['mentions']:
        left, right = sentence_bounds(record, mention)
        for attribute in entities[mention['entity_id']]['attributes']:
            words = attribute.split('_')
            if words[0] == 'is':
                words = words[1:]
            # A bare noun such as 'price' is not a Boolean predicate.
            if len(words) < 2:
                continue
            pattern = r'(?<!\w)' + r'[\s-]+'.join(map(re.escape, words)) + r'(?!\w)'
            for match in re.finditer(pattern, record['text'][left:right], re.I):
                start, end = left + match.start(), left + match.end()
                if any(start < s['end'] and s['start'] < end for s in record['mentions'] + record['values']):
                    continue
                pseudo = {'start': start, 'end': end}
                a, b = clause_bounds(record, pseudo)
                clause = record['text'][a:b]
                if not (a <= mention['start'] and mention['end'] <= b):
                    continue
                if any(a <= v['start'] < b for v in record['values']):
                    continue
                prefix = record['text'][a:start]
                suffix = record['text'][end:b]
                # Recognize both "ENTITY has no FEATURE" and "FEATURE is not
                # enabled in ENTITY", keeping polarity within this clause.
                before = re.search(r"\b(?:is|are|was|were|has|have|had|stays?|remains?|serves?\s+as|supports?|offers?|includes?|provides?|(?:isn|aren|wasn|weren|hasn|haven)['’]t)\s+(?:(?:not|no|never|longer|now|still|currently|already|yet|also|fully|completely|a|an|any|the|this|that)\s+)*$", prefix, re.I)
                after = re.match(r'\s+(?:is|are|was|were)\s+(?:(?:not|no longer)\s+)?(?:enabled|available|supported|included|active|disabled|off|unavailable)\b', suffix, re.I)
                if not before and not after:
                    continue
                if re.search(r'\b(?:not only|not necessarily|not without|no evidence|cannot confirm)\b', clause, re.I):
                    continue
                negative_before = bool(NEGATION.search(prefix) or re.search(r"\bno\s*$|\bcannot\b|\bcan['’]t\b", prefix, re.I))
                negative_after = bool(after and re.search(r'\b(?:not|no longer|disabled|off|unavailable)\b', after.group(), re.I))
                # Do not reduce double negation mechanically to a feature value.
                if after and re.search(r'\bnot\s+(?:disabled|off|unavailable)\b', after.group(), re.I):
                    continue
                if (mention['entity_id'], start, end) in seen:
                    continue
                seen.add((mention['entity_id'], start, end))
                value = {'kind': 'boolean', 'value': not (negative_before or negative_after),
                         'start': start, 'end': end, 'surface': record['text'][start:end],
                         'rule': 'explicit_attribute_phrase_predicate'}
                context = {'start': a, 'end': b, 'surface': clause, 'wording': clause, 'embedding_text': clause}
                output.append({'value': value, 'context': context, 'owners': [mention], 'required_attribute': attribute,
                               'deterministic': True})
    return output


def boolean_wording_candidates(record, entities):
    """Small explicit paraphrase rules; output attributes remain lexicon-bound."""
    output, seen = [], set()
    for attribute, pattern in BOOLEAN_WORDING_RULES.items():
        for match in re.finditer(pattern, record['text'], re.I):
            a, b = clause_bounds(record, {'start': match.start(), 'end': match.end()})
            clause = record['text'][a:b]
            if re.search(r'\b(?:not only|not necessarily|not without|no evidence|cannot confirm)\b', clause, re.I):
                continue
            owners = {m['entity_id']: m for m in record['mentions'] if a <= m['start'] and m['end'] <= b
                      and attribute in entities[m['entity_id']]['attributes']}
            if len(owners) != 1:
                continue
            owner = next(iter(owners.values()))
            key = owner['entity_id'], attribute, a, b
            if key in seen:
                continue
            seen.add(key)
            predicate = record['text'][a:match.end()]
            prefix = record['text'][a:match.start()]
            if re.search(r'\b(?:promise|guarantee|confirm|deny)\w*\s+(?:to\s+)?$', prefix, re.I):
                continue
            # "has no problems working offline" does not negate working offline.
            negative = bool(re.search(r"\b(?:not|never|no longer|cannot|(?:can|doesn|don|isn|aren|wasn|weren)['’]t)\s*$", prefix, re.I)
                            or re.search(r'\b(?:not|no|never)\b', match.group(), re.I))
            value = {'kind': 'boolean', 'value': not negative, 'start': match.start(), 'end': match.end(),
                     'surface': match.group(), 'rule': f'explicit_boolean_wording:{attribute}'}
            context = {'start': a, 'end': b, 'surface': clause, 'wording': clause, 'embedding_text': clause}
            output.append({'value': value, 'context': context, 'owners': [owner],
                           'required_attribute': attribute, 'deterministic': True})
    return output
