#!/usr/bin/env python3
"""Reproducible research probes, not an Arena submission.

Uses Python's standard library. The oracle probes deliberately read the gold
forms of test rows; they diagnose evidence/timeline quality, not parsing skill.
No test labels or test forms enter the evidence store or retrieval index.
"""

import argparse
import collections
import datetime as dt
import itertools
import json
import math
import re
from pathlib import Path
import csv


ROOT = Path(__file__).resolve().parents[1]


def load():
    lexicon = json.loads((ROOT / 'data/lexicon.json').read_text())
    with (ROOT / 'data/practice.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row['facts'] = json.loads(row['form'])
        row['day'] = dt.date.fromisoformat(row['date']).toordinal()
    aliases = {name.casefold(): subject['name']
               for subject in lexicon['subjects']
               for name in [subject['name']] + subject['aliases']}
    return rows, lexicon, aliases


UNITS = {
    'B': ('data', 1), 'KB': ('data', 1000), 'MB': ('data', 10**6),
    'GB': ('data', 10**9), 'TB': ('data', 10**12),
    'mm': ('length', .001), 'cm': ('length', .01), 'm': ('length', 1),
    'km': ('length', 1000), 'in': ('length', .0254),
    'inch': ('length', .0254), 'g': ('mass', .001), 'kg': ('mass', 1),
    'Hz': ('frequency', 1), 'kHz': ('frequency', 1000),
    'MHz': ('frequency', 10**6), 'GHz': ('frequency', 10**9),
    'mW': ('power', .001), 'W': ('power', 1), 'kW': ('power', 1000),
}
DURATIONS = {
    'ms': ('seconds', .001), 's': ('seconds', 1), 'min': ('seconds', 60),
    'h': ('seconds', 3600), 'day': ('seconds', 86400),
    'week': ('seconds', 604800), 'month': ('months', 1),
    'year': ('months', 12),
}


def canonical(value, aliases):
    kind, val = value['kind'], value['value']
    if kind == 'name':
        return kind, aliases.get(val.casefold(), val.casefold())
    if kind in ('quantity', 'duration'):
        table = UNITS if kind == 'quantity' else DURATIONS
        dimension, factor = table.get(value['unit'], (value['unit'], 1))
        return kind, dimension, round(float(val) * factor, 9)
    if kind == 'money':
        return kind, value['currency'], value.get('per', 'one_off'), float(val)
    return kind, val


def comparable(value, aliases):
    c = canonical(value, aliases)
    if c[0] in ('name', 'boolean'):
        return None
    if c[0] == 'date':
        # Candidate counting only: avoid mixing date precision.
        return c[0], len(c[1])
    return c[:-1]


def metrics(pairs):
    pairs = list(pairs)
    decided = [(gold, pred) for gold, pred in pairs if pred != 'unsure']
    confusion = collections.Counter(f'{gold}->{pred}' for gold, pred in pairs)
    f1s = []
    for label in ('true', 'false', 'unsure'):
        tp = sum(g == label and p == label for g, p in pairs)
        fp = sum(g != label and p == label for g, p in pairs)
        fn = sum(g == label and p != label for g, p in pairs)
        f1s.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
    false_rows = sum(g == 'false' for g, _ in pairs)
    return {
        'rows': len(pairs),
        'three_way_accuracy': round(sum(g == p for g, p in pairs) / len(pairs), 4),
        'decided_coverage': round(len(decided) / len(pairs), 4),
        'decided_precision': round(sum(g == p for g, p in decided) / len(decided), 4)
                             if decided else None,
        'macro_f1': round(sum(f1s) / 3, 4),
        'false_recall': round(sum(g == p == 'false' for g, p in pairs) / false_rows, 4)
                        if false_rows else None,
        'confusion': dict(sorted(confusion.items())),
    }


def profile(rows, lexicon, aliases):
    slots = collections.defaultdict(list)
    exact = collections.defaultdict(set)
    texts = collections.defaultdict(list)
    names = [name for s in lexicon['subjects'] for name in [s['name']] + s['aliases']]
    no_name = 0
    for row in rows:
        texts[row['text']].append(row)
        no_name += not any(re.search(r'(?<!\w)' + re.escape(n) + r'(?!\w)',
                                    row['text'], re.I) for n in names)
        for fact in row['facts']:
            key = fact['subject'], fact['attribute']
            slots[key].append((row, fact))
            if row['label'] == 'true' and fact['value'].get('op', 'eq') == 'eq':
                exact[key].add(canonical(fact['value'], aliases))
    missing = [list(k) for k, obs in slots.items()
               if not any(r['label'] == 'true' for r, _ in obs)]
    latest = {}
    for key, observations in slots.items():
        valid = [(row, fact) for row, fact in observations
                 if row['label'] == 'true' and fact['value'].get('op', 'eq') == 'eq']
        if valid:
            latest[key] = max(valid, key=lambda item: (item[0]['day'],
                                                     len(str(item[1]['value']['value']))))
    exact_subject_mentions = 0
    for row in rows:
        for fact in row['facts']:
            subject = next(s for s in lexicon['subjects'] if s['name'] == fact['subject'])
            exact_subject_mentions += any(
                re.search(r'(?<!\w)' + re.escape(n) + r'(?!\w)', row['text'], re.I)
                for n in [subject['name']] + subject['aliases'])
    groups = collections.defaultdict(list)
    for key, (row, fact) in latest.items():
        dimension = comparable(fact['value'], aliases)
        if dimension:
            groups[dimension].append(key)
    pairs = sum(len(v) * (len(v) - 1) // 2 for v in groups.values())
    same_attribute_pairs = sum(a[1] == b[1] for values in groups.values()
                               for a, b in itertools.combinations(values, 2))
    changes = 0
    for key, observations in slots.items():
        valid = [(r, f) for r, f in observations
                 if r['label'] == 'true' and f['value'].get('op', 'eq') == 'eq']
        values = set()
        for (r1, f1), (r2, f2) in itertools.combinations(valid, 2):
            if r1['day'] > r2['day']:
                r1, r2, f1, f2 = r2, r1, f2, f1
            a, b = canonical(f1['value'], aliases), canonical(f2['value'], aliases)
            if r1['day'] < r2['day'] and a != b and a[:-1] == b[:-1]:
                if a[0] == 'date' and len(a[-1]) != len(b[-1]):
                    continue
                values.add((a, b))
        changes += len(values)
    return {
        'rows': len(rows), 'labels': dict(collections.Counter(r['label'] for r in rows)),
        'subjects': len(lexicon['subjects']),
        'lexicon_attribute_slots': sum(len(s['attributes']) for s in lexicon['subjects']),
        'events': len(lexicon['events']), 'date_range': [min(r['date'] for r in rows),
                                                        max(r['date'] for r in rows)],
        'atoms': sum(len(r['facts']) for r in rows),
        'atoms_per_row': dict(collections.Counter(len(r['facts']) for r in rows)),
        'atom_kinds': dict(collections.Counter(f['value']['kind'] for r in rows
                                             for f in r['facts'])),
        'slots_with_true_rows': len(slots) - len(missing),
        'slots_with_exact_true_values': len(exact),
        'canonical_exact_true_values': sum(map(len, exact.values())),
        'slots_with_multiple_canonical_exact_true_values': sum(len(v) > 1 for v in exact.values()),
        'slots_without_true_rows': missing,
        'rows_without_whole_word_alias_match': no_name,
        'atoms_with_explicit_whole_word_subject_alias': exact_subject_mentions,
        'multi_atom_rows': sum(len(r['facts']) > 1 for r in rows),
        'multi_atom_false_rows': sum(len(r['facts']) > 1 and r['label'] == 'false' for r in rows),
        'identical_texts_with_multiple_labels': [
            {'text': t, 'rows': [[r['id'], r['date'], r['label']] for r in obs]}
            for t, obs in texts.items() if len(set(r['label'] for r in obs)) > 1],
        'candidate_capacity_NOT_verified_claims': {
            'latest_true_exact_slots': len(latest),
            'latest_true_evidence_months': dict(sorted(collections.Counter(
                r['date'][:7] for r, _ in latest.values()).items())),
            'unordered_dimension_compatible_pairs': pairs,
            'same_attribute_subset': same_attribute_pairs,
            'ordered_observed_value_changes_same_kind_dimension': changes,
            'warning': 'Counts ignore as-of validity, partial-date compatibility, renderability, '
                       'and hidden changes; these are candidate pools, not certified truths.'},
    }


def oracle_predict(train, test, aliases, mode):
    evidence = collections.defaultdict(list)
    for row in train:
        if row['label'] != 'true':
            continue
        for fact in row['facts']:
            if fact['value'].get('op', 'eq') == 'eq':
                evidence[(fact['subject'], fact['attribute'])].append((row['day'], fact['value']))

    def atom_status(fact, day):
        # Deliberately simple: exact assertions only; all other predicates abstain.
        if fact['value'].get('op', 'eq') != 'eq':
            return 'unsure'
        observations = evidence[(fact['subject'], fact['attribute'])]
        if mode == 'same_day':
            candidates = [(d, v) for d, v in observations if d == day]
        elif mode == 'latest_past':
            candidates = [(d, v) for d, v in observations if d <= day]
            if candidates:
                newest = max(d for d, _ in candidates)
                candidates = [(d, v) for d, v in candidates if d == newest]
        else:
            candidates = observations
            if candidates:
                distance = min(abs(d - day) for d, _ in candidates)
                candidates = [(d, v) for d, v in candidates if abs(d - day) == distance]
        values = {canonical(v, aliases) for _, v in candidates}
        query = canonical(fact['value'], aliases)
        if not values:
            return 'unsure'
        if values == {query}:
            return 'true'
        if query not in values and all(v[:-1] == query[:-1] for v in values):
            # For dates, a year and a day need compatible interval reasoning.
            # This probe declines different precisions rather than calling false.
            if query[0] == 'date' and any(len(v[-1]) != len(query[-1]) for v in values):
                return 'unsure'
            return 'false'
        return 'unsure'

    results = []
    for row in test:
        states = [atom_status(f, row['day']) for f in row['facts']]
        pred = 'false' if 'false' in states else 'true' if all(s == 'true' for s in states) else 'unsure'
        results.append((row['label'], pred))
    return metrics(results)


def features(text):
    words = re.findall(r'\w+|[€$£%]', text.casefold())
    return collections.Counter(words + [' '.join(pair) for pair in zip(words, words[1:])])


def retrieval_predict(train, test):
    # Word unigrams + bigrams; training-only IDF; cosine nearest neighbour.
    counters = [features(r['text']) for r in train]
    frequencies = collections.Counter(term for terms in counters for term in terms)
    idf = {term: math.log((1 + len(train)) / (1 + freq)) + 1
           for term, freq in frequencies.items()}
    postings = collections.defaultdict(list)
    for index, terms in enumerate(counters):
        weights = {term: (1 + math.log(count)) * idf[term] for term, count in terms.items()}
        norm = math.sqrt(sum(w*w for w in weights.values())) or 1
        for term, weight in weights.items():
            postings[term].append((index, weight/norm))
    plain, dated = [], []
    for row in test:
        terms = features(row['text'])
        weights = {term: (1 + math.log(count)) * idf[term] for term, count in terms.items() if term in idf}
        norm = math.sqrt(sum(w*w for w in weights.values())) or 1
        similarities = collections.defaultdict(float)
        for term, weight in weights.items():
            for index, other in postings[term]:
                similarities[index] += weight/norm * other
        plain_index = max(similarities, key=similarities.get) if similarities else 0
        dated_index = max(similarities, key=lambda i: similarities[i] /
                          (1 + abs(train[i]['day'] - row['day']) / 90)) if similarities else 0
        plain.append((row['label'], train[plain_index]['label']))
        dated.append((row['label'], train[dated_index]['label']))
    return {'tfidf_1nn': metrics(plain), 'tfidf_1nn_date_penalty_90_days': metrics(dated)}


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    rows, lexicon, aliases = load()
    results = {'profile': profile(rows, lexicon, aliases), 'experiments': {}}
    splits = {
        'every_fifth_row_hidden': ([r for i, r in enumerate(rows) if i % 5],
                                  [r for i, r in enumerate(rows) if i % 5 == 0]),
        'forward_holdout_from_2026_09_01': ([r for r in rows if r['date'] < '2026-09-01'],
                                          [r for r in rows if r['date'] >= '2026-09-01']),
    }
    for name, (train, test) in splits.items():
        experiments = {'train_rows': len(train), 'test_rows': len(test),
                       'always_true': metrics((r['label'], 'true') for r in test)}
        experiments.update(retrieval_predict(train, test))
        for mode in ('same_day', 'latest_past', 'nearest_date'):
            experiments[f'GOLD_FORM_DIAGNOSTIC_{mode}'] = oracle_predict(train, test, aliases, mode)
        results['experiments'][name] = experiments
    rendered = json.dumps(results, indent=2, ensure_ascii=False) + '\n'
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end='')


if __name__ == '__main__':
    run()
