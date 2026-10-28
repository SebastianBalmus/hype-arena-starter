"""Dated source assertions, with explicit labels and trusted-evidence filtering."""

from collections import Counter
from copy import deepcopy
from datetime import date
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re

from entity_linking.json_utils import exact_json


def _hash(value):
    return hashlib.sha256(exact_json(value).encode('utf-8')).hexdigest()


def _label(row):
    label = row.get('label', '')
    if not isinstance(label, str) or label not in {'true', 'false', 'unsure', ''}:
        raise ValueError('Source label must be true, false, unsure or empty')
    return label


def _date(value):
    if not isinstance(value, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
        raise ValueError('Observation date must be YYYY-MM-DD')
    date.fromisoformat(value)
    return value


def source_fingerprint(row):
    """Exact ID/text/date identity; do not normalize source text."""
    if any(not isinstance(row.get(key), str) or not row[key] for key in ('id', 'text', 'date')):
        raise ValueError('Source requires nonempty string ID, text and date')
    _date(row['date'])
    return _hash([row['id'], row['text'], row['date']])


def validate_value(value):
    """Validate the contract's value structure without a runtime schema dependency."""
    if not isinstance(value, dict):
        raise ValueError('Value must be an object')
    kind = value.get('kind')
    fields = {'quantity': {'unit'}, 'duration': {'unit'}, 'money': {'currency'},
              'percent': set(), 'date': set(), 'version': set(), 'name': set(), 'boolean': set()}
    if kind not in fields:
        raise ValueError('Unsupported value kind')
    required = {'kind', 'value'} | fields[kind]
    allowed = required | ({'op'} if kind not in {'name', 'boolean'} else set()) | ({'per'} if kind == 'money' else set())
    if not required <= set(value) or not set(value) <= allowed:
        raise ValueError('Invalid value fields')
    if value.get('op', 'eq') not in {'eq', 'lt', 'le', 'gt', 'ge'}:
        raise ValueError('Invalid comparison operator')
    raw = value['value']
    if kind in {'quantity', 'duration', 'money', 'percent'}:
        if isinstance(raw, bool) or not isinstance(raw, (int, float, Decimal)) or not Decimal(str(raw)).is_finite():
            raise ValueError('Expected a finite numeric value')
        if kind in {'duration', 'money'} and raw < 0:
            raise ValueError('Negative money/duration is outside the contract')
    elif kind == 'boolean':
        if type(raw) is not bool:
            raise ValueError('Boolean value must be true or false')
    elif not isinstance(raw, str):
        raise ValueError('Expected a string value')
    if kind == 'name' and (not raw.strip() or len(raw) > 200):
        raise ValueError('Invalid name value')
    if kind == 'version' and not re.fullmatch(r'\d+(?:\.\d+){0,3}', raw):
        raise ValueError('Invalid version value')
    if kind == 'date':
        if not re.fullmatch(r'\d{4}(?:-\d{2}(?:-\d{2})?)?', raw):
            raise ValueError('Invalid date value')
        date.fromisoformat(raw + ('-01-01' if len(raw) == 4 else '-01' if len(raw) == 7 else ''))
    if kind == 'quantity' and (not isinstance(value['unit'], str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,39}', value['unit']) or value['unit'] in {'ms', 's', 'min', 'h', 'day', 'week', 'month', 'year'}):
        raise ValueError('Invalid quantity unit')
    if kind == 'duration' and value['unit'] not in {'ms', 's', 'min', 'h', 'day', 'week', 'month', 'year'}:
        raise ValueError('Invalid duration unit')
    if kind == 'money' and (not isinstance(value['currency'], str) or not re.fullmatch(r'[A-Z]{3}', value['currency'])):
        raise ValueError('Invalid currency')
    if kind == 'money' and 'per' in value and value['per'] not in {'day', 'week', 'month', 'year'}:
        raise ValueError('Invalid payment period')


class EvidenceIndex:
    """Index[canonical subject, attribute][observation date] -> all observations."""

    def __init__(self, lexicon, sources, build_stats=None, source_policy='true_only'):
        if source_policy not in {'true_only', 'all_labels'}:
            raise ValueError('Unknown source policy')
        self.source_policy = source_policy
        self.lexicon_sha256 = hashlib.sha256(json.dumps(lexicon, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        self._aliases, self._attributes = {}, {}
        for entity in lexicon['subjects']:
            self._attributes[entity['name']] = set(entity['attributes'])
            for name in [entity['name'], *entity.get('aliases', [])]:
                self._aliases.setdefault(name.casefold().strip(), set()).add(entity['name'])
        self._sources, self._observations, self._index = {}, [], {}
        self.build_stats = deepcopy(build_stats or {})
        for original in sources:
            source = deepcopy(original)
            label = _label(source)
            if self.source_policy == 'true_only' and label != 'true':
                raise ValueError('Knowledge-base sources must be labelled true')
            if source['id'] in self._sources:
                raise ValueError('Duplicate source ID')
            fingerprint = source_fingerprint(source)
            if source.get('fingerprint', fingerprint) != fingerprint:
                raise ValueError('Source fingerprint mismatch')
            form = source['form']
            if not isinstance(form, list):
                raise ValueError('Trusted form must be an assertion list')
            form_hash = _hash(form)
            if source.get('form_sha256', form_hash) != form_hash:
                raise ValueError('Source form checksum mismatch')
            origin = source.get('form_origin', 'provided_form')
            if origin not in {'provided_form', 'verified_packaged_form'}:
                raise ValueError('Unsupported form origin')
            source.update(fingerprint=fingerprint, form_sha256=form_hash, form_origin=origin)
            self._sources[source['id']] = source
            for ordinal, fact in enumerate(form):
                if not isinstance(fact, dict) or set(fact) != {'subject', 'attribute', 'value'}:
                    raise ValueError('Assertion must contain subject, attribute and value')
                subject = self.resolve_subject(fact['subject'])
                attribute = fact['attribute']
                if attribute not in self._attributes[subject]:
                    raise ValueError(f'Attribute {attribute!r} is not allowed for {subject}')
                validate_value(fact['value'])
                observation = {'subject': subject, 'attribute': attribute, 'value': deepcopy(fact['value']),
                               'observation_date': source['date'], 'source_id': source['id'],
                               'source_label': label, 'source_fingerprint': fingerprint,
                               'source_form_sha256': form_hash, 'assertion_index': ordinal, 'form_origin': origin}
                self._observations.append(observation)
                self._index.setdefault((subject, attribute), {}).setdefault(source['date'], []).append(observation)

    def resolve_subject(self, subject):
        if not isinstance(subject, str):
            raise ValueError('Subject must be a name or published alias')
        matches = self._aliases.get(subject.casefold().strip(), set())
        if len(matches) != 1:
            raise ValueError(f'Unknown or ambiguous subject: {subject!r}')
        return next(iter(matches))

    @classmethod
    def from_rows(cls, rows, lexicon, packaged=None, source_policy='true_only'):
        """Store labelled claims; only true rows are positive evidence by default."""
        if source_policy not in {'true_only', 'all_labels'}:
            raise ValueError('Unknown source policy')
        if packaged is not None:
            expected = hashlib.sha256(json.dumps(lexicon, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            if packaged.lexicon_sha256 != expected:
                raise ValueError('Packaged knowledge base uses a different lexicon')
        sources, seen, labels, skipped = [], set(), Counter(), []
        for row in rows:
            if row.get('id') in seen:
                raise ValueError('Duplicate input ID')
            seen.add(row.get('id'))
            label = _label(row)
            labels[label or 'unlabelled'] += 1
            if source_policy == 'true_only' and label != 'true':
                continue
            source_fingerprint(row)
            raw = row.get('form')
            if raw is not None and raw != '':
                form = json.loads(raw, parse_float=Decimal) if isinstance(raw, str) else deepcopy(raw)
                origin = 'provided_form'
            else:
                form = packaged.verified_form(row) if packaged is not None else None
                origin = 'verified_packaged_form'
            if form is None:
                skipped.append({'id': row['id'], 'reason': 'no_verified_form_for_true_row' if source_policy == 'true_only' else 'no_verified_form'})
                continue
            sources.append({'id': row['id'], 'text': row['text'], 'date': row['date'], 'label': label,
                            'form': form, 'form_origin': origin})
        skipped_key = 'true_rows_without_verified_forms' if source_policy == 'true_only' else 'rows_without_verified_forms'
        return cls(lexicon, sources, {'input_rows_by_label': dict(labels), skipped_key: skipped}, source_policy)

    def verified_form(self, row):
        """A reused ID alone is insufficient, and an unlabelled row is never a seed."""
        if _label(row) != 'true':
            return None
        source = self._sources.get(row.get('id'))
        if source is None or source_fingerprint(row) != source['fingerprint']:
            return None
        return deepcopy(source['form'])

    def lookup(self, subject, attribute, on_date, *, include_untrusted=False):
        subject = self.resolve_subject(subject)
        if attribute not in self._attributes[subject]:
            raise ValueError('Attribute is not in this subject\'s lexicon')
        _date(on_date)
        observations = self._index.get((subject, attribute), {}).get(on_date, [])
        if not include_untrusted:
            observations = [o for o in observations if o['source_label'] == 'true']
        return {'subject': subject, 'attribute': attribute, 'date': on_date,
                'status': 'observations_found' if observations else 'no_same_date_observations',
                'observations': deepcopy(observations),
                'distinct_predicates': len({_hash(o['value']) for o in observations}),
                'temporal_policy': 'same_observation_date_only; no persistence inferred'}

    def history(self, subject, attribute, *, include_untrusted=False):
        """Audit history; callers must not treat it as proof at arbitrary dates."""
        subject = self.resolve_subject(subject)
        if attribute not in self._attributes[subject]:
            raise ValueError('Attribute is not in this subject\'s lexicon')
        dates = self._index.get((subject, attribute), {})
        return deepcopy([o for day in sorted(dates) for o in dates[day]
                         if include_untrusted or o['source_label'] == 'true'])

    def summary(self):
        date_groups = [observations for dates in self._index.values() for observations in dates.values()]
        return {'source_rows': len(self._sources), 'observations': len(self._observations),
                'source_policy': self.source_policy,
                'observations_by_source_label': dict(Counter(o['source_label'] or 'unlabelled' for o in self._observations)),
                'source_labels': dict(Counter(s['label'] for s in self._sources.values())),
                'subjects': len({s for s, a in self._index}), 'subject_attribute_pairs': len(self._index),
                'subject_attribute_date_groups': len(date_groups),
                'observation_dates': len({s['date'] for s in self._sources.values()}),
                'values_by_kind': dict(Counter(o['value']['kind'] for o in self._observations)),
                'comparison_observations': sum(o['value'].get('op', 'eq') != 'eq' for o in self._observations),
                'same_date_groups_with_multiple_predicates': sum(len({_hash(o['value']) for o in group}) > 1 for group in date_groups),
                'multiple_predicates_note': 'Retained separately; not automatically declared contradictions.',
                'build_stats': deepcopy(self.build_stats)}

    def to_dict(self):
        return deepcopy({'schema_version': 1, 'source_policy': self.source_policy, 'lexicon_sha256': self.lexicon_sha256,
                         'sources': list(self._sources.values()), 'observations': self._observations,
                         'build_stats': self.build_stats})

    def save(self, path):
        path = Path(path)
        with path.open('x', encoding='utf-8') as handle:
            handle.write(exact_json(self.to_dict(), indent=2) + '\n')

    @classmethod
    def load(cls, path, lexicon):
        payload = json.loads(Path(path).read_text(), parse_float=Decimal)
        if payload.get('schema_version') != 1:
            raise ValueError('Unsupported knowledge-base schema')
        result = cls(lexicon, payload['sources'], payload.get('build_stats'), payload.get('source_policy', 'true_only'))
        if payload['lexicon_sha256'] != result.lexicon_sha256:
            raise ValueError('Knowledge-base lexicon checksum mismatch')
        if exact_json(payload['observations']) != exact_json(result._observations):
            raise ValueError('Saved observations do not match their trusted source forms')
        return result
