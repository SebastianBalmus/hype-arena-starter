"""Reusable source-span and local wording utilities."""

import re


def sentence_bounds(record, value):
    text = record['text']
    # Do not split decimals or periods inside recognized names/literals.
    spans = record['mentions'] + record['values']
    ends = [m.end() for m in re.finditer(r'[.!?](?=\s|$)', text)
            if not any(s['start'] <= m.start() < s['end'] for s in spans)]
    left = max([0] + [end for end in ends if end <= value['start']])
    right = min([len(text)] + [end for end in ends if end >= value['end']])
    return left, right


def local_context(record, value):
    """Bound wording by adjacent literals; strip entities and numeric magnitudes."""
    text = record['text']
    left, right = sentence_bounds(record, value)
    others = [v for v in record['values'] if v['start'] != value['start']]
    left = max([left] + [v['end'] for v in others if left <= v['end'] <= value['start']])
    right = min([right] + [v['start'] for v in others if value['end'] <= v['start'] <= right])
    # Keep at most eight words on each side, preserving original offsets.
    before = list(re.finditer(r'\S+', text[left:value['start']]))
    after = list(re.finditer(r'\S+', text[value['end']:right]))
    if len(before) > 8:
        left += before[-8].start()
    if len(after) > 8:
        right = value['end'] + after[7].end()
    fragment = text[left:right]
    spans = [m for m in record['mentions'] if m['start'] < right and m['end'] > left]
    for mention in sorted(spans, key=lambda m: m['start'], reverse=True):
        start, end = max(left, mention['start']) - left, min(right, mention['end']) - left
        fragment = fragment[:start] + ' ' + fragment[end:]
    fragment = re.sub(r'\d+(?:[.,]\d+)*', ' ', fragment)
    fragment = re.sub(r'[^\w\s-]', ' ', fragment)
    wording = re.sub(r'\s+', ' ', fragment.casefold()).strip()
    # Unit/kind is deliberate extraction information; no truth label or form.
    return {'wording': wording, 'embedding_text': f"{wording} {value['kind']} {value.get('unit', '')}",
            'start': left, 'end': right, 'surface': text[left:right]}


def entities_for(record, value):
    left, right = sentence_bounds(record, value)
    return [m for m in record['mentions'] if left <= m['start'] and m['end'] <= right]
