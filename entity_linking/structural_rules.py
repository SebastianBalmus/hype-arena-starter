"""Source-grounded binding rules. No labels, forms or learned resources are read."""

import re

from entity_linking.context import local_context, sentence_bounds
from entity_linking.values import singular


def local_owners(record, value, clause_bounds, inherit=False):
    left, right = clause_bounds(record, value)
    owners = {m['entity_id']: m for m in record['mentions'] if left <= m['start'] and m['end'] <= right}
    inherited = False
    if not owners and inherit:
        left, right = sentence_bounds(record, value)
        owners = {m['entity_id']: m for m in record['mentions'] if left <= m['start'] and m['end'] <= right}
        if len(owners) != 1:
            return [], False
        inherited = True
    return list(owners.values()), inherited


def version_choices(record, value, entities):
    """Distinguish product/API/CLI/device versions using explicit constructions."""
    output = []
    for device in record['mentions']:
        if 'shipping_android' not in entities[device['entity_id']]['attributes'] or device['end'] > value['start']:
            continue
        gap = record['text'][device['end']:value['start']]
        if len(gap.split()) <= 12 and not re.search(r'[.!?;](?:\s|$)', gap) and re.search(r'\bandroid\b', gap, re.I) and re.search(r'\b(?:ships? with|runs?|receives?|software on)\b', gap, re.I):
            output.append((device['entity_id'], 'shipping_android', 'device_runs_or_ships_android'))
        prefix = record['text'][max(0, device['start'] - 20):device['start']]
        if re.search(r'\bsoftware on\s+(?:the\s+)?$', prefix, re.I) and re.fullmatch(r':\s*(?:Hooly\s+)?Android\s*', gap, re.I):
            output.append((device['entity_id'], 'shipping_android', 'software_on_device_android'))
    if output:
        return output
    preceding = [m for m in record['mentions'] if m['end'] <= value['start']]
    if preceding:
        mention = max(preceding, key=lambda m: m['end'])
        gap = record['text'][mention['end']:value['start']]
        if re.fullmatch(r"\s*(?:(?:API|CLI|build|release|version|at|runs|sits|speaks)\s*|[:v])*\s*", gap, re.I):
            attrs = entities[mention['entity_id']]['attributes']
            context = record['text'][mention['end']:min(len(record['text']), value['end'] + 30)]
            if 'api_version' in attrs and re.search(r'\b(?:api|interface)\b', context, re.I):
                output.append((mention['entity_id'], 'api_version', 'explicit_api_or_interface_version'))
            elif 'cli_version' in attrs and re.search(r'\bcli\b', gap, re.I):
                output.append((mention['entity_id'], 'cli_version', 'explicit_cli_version'))
            elif 'version' in attrs:
                output.append((mention['entity_id'], 'version', 'entity_adjacent_product_version'))
            elif 'stable_version' in attrs:
                output.append((mention['entity_id'], 'stable_version', 'entity_adjacent_stable_version'))
    if not output:
        following = [m for m in record['mentions'] if value['end'] <= m['start']]
        if following:
            mention = min(following, key=lambda m: m['start'])
            if re.fullmatch(r'\s+of\s+', record['text'][value['end']:mention['start']], re.I) and 'version' in entities[mention['entity_id']]['attributes']:
                output.append((mention['entity_id'], 'version', 'version_of_explicit_entity'))
    return output


DURATION_CUES = {
    'free_trial': r'\btrial\b', 'refund_window': r'\brefund',
    'history_retention': r'\bhistory\b', 'trash_retention': r'\b(?:trash|bin)\b',
    'support_first_response': r'\bfirst\s+(?:response|reply)\b',
    'first_reply': r'\bfirst\s+(?:response|reply)\b',
    'median_response_time': r'\bmedian\b.{0,20}\b(?:response|answer)|\b(?:response|answer).{0,20}\bmedian\b',
    'p99_response_time': r'\bp99\b', 'release_cadence': r'\b(?:release|version|build)\b.{0,35}\bevery\b',
    'video_upload_limit': r'\bvideo\b', 'battery_life': r'\bbattery\b.{0,20}\b(?:last|life)',
    'data_retention': r'\b(?:retain|keep|store)\w*\s+(?:uploads|data)\b|\bdata\s+retention\b',
}
MONEY_CUES = {
    'ticket_price': r'\bticket\b', 'rent': r'\brent\b',
    'annual_price': r'\b(?:annual|yearly)\b|\ba year\b|\bper year\b',
    'extra_seat_price': r'\b(?:extra|additional|further)\s+seats?\b',
    'ad_free_price': r'\bad[- ]free\b', 'series_a_amount': r'\bseries a\b',
    'cost_per_hire': r'\bcost per hire\b',
}


def tokens(text):
    return ' '.join(singular(w) for w in re.findall(r'[a-z0-9]+', text.casefold().replace('_', ' ')))


def matching_choices(owners, entities, attributes, family):
    return [(m['entity_id'], attr, family) for m in owners for attr in attributes if attr in entities[m['entity_id']]['attributes']]


def typed_choices(record, value, owners, inherited, entities, clause_bounds):
    """Return all credible choices, so ambiguity is explicit rather than ordered."""
    choices = []
    a, b = clause_bounds(record, value)
    context = local_context(record, value)
    text = record['text'][max(a, context['start']):min(b, context['end'])]
    blocked = None
    if value['kind'] == 'version':
        return version_choices(record, value, entities), None
    if value['kind'] == 'quantity':
        attrs = {attr for m in owners for attr in entities[m['entity_id']]['attributes'] if tokens(attr) == tokens(value['unit'])}
        choices.extend(matching_choices(owners, entities, attrs, 'count_noun_exact_attribute'))
    if value['kind'] == 'duration':
        attrs = [attr for attr, cue in DURATION_CUES.items() if re.search(cue, text, re.I)]
        choices.extend(matching_choices(owners, entities, attrs, 'duration_explicit_cue'))
    if value['kind'] == 'money':
        attrs = [attr for attr, cue in MONEY_CUES.items() if re.search(cue, text, re.I)]
        if attrs:
            choices.extend(matching_choices(owners, entities, attrs, 'money_specific_cue'))
            if not choices:
                blocked = 'specific_price_attribute_unavailable'
        elif value.get('per') == 'month' or re.search(r'\b(?:costs?|priced|price)\b', text, re.I):
            choices.extend(matching_choices(owners, entities, ['price'], 'money_base_price'))
    if value['kind'] == 'date':
        for mention in owners:
            if mention['end'] > value['start'] or 'joined' not in entities[mention['entity_id']]['attributes']:
                continue
            gap = record['text'][mention['end']:value['start']]
            if len(gap.split()) <= 14 and re.match(r'\s+(?:started\s+(?:at|working)|joined\b|arrived\s+at\b)', gap, re.I):
                choices.append((mention['entity_id'], 'joined', 'employment_start_pattern'))
    if inherited and not choices and not blocked:
        # Tested inheritance cues are used only when one entity exists in the
        # sentence. Generic phrase/date rules remain disabled for other cases.
        words = tokens(text)
        attrs = {attr for owner in owners for attr in entities[owner['entity_id']]['attributes']
                 if len(tokens(attr).split()) >= 2 and re.search(r'(?<!\w)' + re.escape(tokens(attr)) + r'(?!\w)', words)}
        choices.extend(matching_choices(owners, entities, attrs, 'inherited_literal_attribute'))
        if value['kind'] == 'date':
            attrs = [attr for attr in ['announced', 'released', 'launched', 'opened', 'published', 'founded'] if re.search(r'\b' + attr + r'\b', text, re.I)]
            choices.extend(matching_choices(owners, entities, attrs, 'inherited_date_verb'))
        if value['kind'] == 'quantity' and value['unit'] in {'engineer', 'member', 'person'} and re.search(r'\b(?:team|group|crew)\b', text, re.I):
            choices.extend(matching_choices(owners, entities, ['team_size'], 'inherited_team_count'))
    return choices, blocked


def name_choices(record, candidate, entities):
    choices = []
    value = candidate['value']
    for owner in candidate['owners']:
        attr = None
        if owner['end'] <= value['start']:
            gap = record['text'][owner['end']:value['start']]
            match = re.fullmatch(r"\s*['’]s\s+(ceo|cto|previous employer)\s+is\s*", gap, re.I)
            if match:
                attr = match[1].casefold().replace(' ', '_')
            elif re.fullmatch(r'\s+(?:runs?|is hosted)\s+on\s*', gap, re.I):
                attr = 'cloud_provider'
        elif value['end'] <= owner['start']:
            gap = record['text'][value['end']:owner['start']]
            match = re.fullmatch(r'\s*,?\s*(?:the\s+)?(ceo|cto)\s+of\s*', gap, re.I)
            if match:
                attr = match[1].casefold()
        if attr in entities[owner['entity_id']]['attributes']:
            choices.append((owner['entity_id'], attr, 'name_local_syntax'))
    return choices


def prepare_candidate(record, candidate, entities, clause_bounds):
    """Plan a single decision per source value before any embedding query."""
    candidate = dict(candidate)
    if candidate.get('deterministic'):
        candidate['rule_choices'] = [(candidate['owners'][0]['entity_id'], candidate['required_attribute'], candidate['value']['rule'])]
        return candidate
    if candidate['value']['kind'] == 'name':
        choices = name_choices(record, candidate, entities)
        blocked = None if choices else 'name_requires_structural_evidence'
    else:
        owners, inherited = local_owners(record, candidate['value'], clause_bounds, inherit=True)
        candidate['owners'] = owners
        candidate['subject_resolution'] = 'same_sentence_unique_entity' if inherited else 'explicit_clause'
        choices, blocked = typed_choices(record, candidate['value'], owners, inherited, entities, clause_bounds)
        if inherited and not choices and not blocked:
            blocked = 'inherited_subject_without_binding_cue'
    candidate['rule_choices'] = choices
    candidate['rule_blocked'] = blocked
    return candidate


def resolve_choices(choices, record, entities):
    grouped = {}
    for entity_id, attribute, rule in choices:
        if attribute not in entities[entity_id]['attributes']:
            raise ValueError('Structural rule proposed an attribute outside the lexicon')
        grouped.setdefault((entity_id, attribute), set()).add(rule)
    alternatives = []
    for (entity_id, attribute), rules in sorted(grouped.items()):
        mention = next(m for m in record['mentions'] if m['entity_id'] == entity_id)
        alternatives.append({'entity_id': entity_id, 'subject': entities[entity_id]['name'], 'attribute': attribute,
                             'entity_span': [mention['start'], mention['end']], 'rules': sorted(rules),
                             'score': None, 'similarity': None})
    return {'binding': alternatives[0] if len(alternatives) == 1 else None,
            'candidates': alternatives, 'margin': None, 'status': 'accepted' if len(alternatives) == 1 else 'rule_conflict'}
