"""Reference normalization and extraction metrics, separate from inference."""

from collections import Counter, defaultdict
from decimal import Decimal
import json


FACTORS = {
    "B": ("B", "1"), "KB": ("B", "1000"), "MB": ("B", "1000000"),
    "GB": ("B", "1000000000"), "TB": ("B", "1000000000000"),
    "mm": ("m", ".001"), "cm": ("m", ".01"), "m": ("m", "1"), "km": ("m", "1000"),
    "inch": ("m", ".0254"), "in": ("m", ".0254"), "g": ("g", "1"), "kg": ("g", "1000"),
    "mAh": ("mAh", "1"), "Hz": ("Hz", "1"), "kHz": ("Hz", "1000"),
    "MHz": ("Hz", "1000000"), "GHz": ("Hz", "1000000000"),
    "mW": ("W", ".001"), "W": ("W", "1"), "kW": ("W", "1000"),
}


TIME = {"ms": ("s", ".001"), "s": ("s", "1"), "min": ("s", "60"), "h": ("s", "3600"),
        "day": ("s", "86400"), "week": ("s", "604800"), "month": ("month", "1"), "year": ("month", "12")}


def signature(value):
    kind, op = value["kind"], value.get("op", "eq")
    if kind in ("quantity", "duration"):
        unit = value["unit"]
        canonical, factor = (FACTORS if kind == "quantity" else TIME).get(unit, (unit, "1"))
        return kind, op, canonical, Decimal(str(value["value"])) * Decimal(factor)
    if kind == "money":
        return kind, op, value["currency"], value.get("per"), Decimal(str(value["value"]))
    if kind == "percent":
        return kind, op, Decimal(str(value["value"]))
    return kind, op, value["value"]


def fact_key(fact, aliases):
    value = fact['value']
    value_key = signature(value)
    if value['kind'] == 'name':
        name = value['value'].casefold().strip()
        value_key = ('name', 'eq', aliases.get(name, name))
    subject = fact['subject'].casefold().strip()
    return aliases.get(subject, subject), fact['attribute'], value_key


def scores(tp, predicted, expected):
    precision = tp / predicted if predicted else 0
    recall = tp / expected if expected else 0
    return {'correct': tp, 'predicted': predicted, 'reference': expected, 'precision': precision, 'recall': recall,
            'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0}


def evaluate(rows, predictions, aliases):
    tp = predicted_count = expected_count = exact = nonempty_exact = 0
    kinds = defaultdict(Counter)
    attributes = defaultdict(Counter)
    errors = []
    for row, prediction in zip(rows, predictions):
        if (row['id'], row['text'], row['date']) != (prediction['id'], prediction['text'], prediction['date']):
            raise ValueError('Prediction ID/text/date mismatch')
        gold = {fact_key(f, aliases): f for f in json.loads(row['form'], parse_float=Decimal)}
        guessed = {fact_key(f, aliases): f for f in prediction['predicted_form']}
        common = set(gold) & set(guessed)
        tp += len(common)
        expected_count += len(gold)
        predicted_count += len(guessed)
        exact += set(gold) == set(guessed)
        nonempty_exact += bool(gold) and set(gold) == set(guessed)
        for collection, field in [(gold, 'reference'), (guessed, 'predicted')]:
            for key, fact in collection.items():
                kinds[fact['value']['kind']][field] += 1
                attributes[fact['attribute']][field] += 1
        for key in common:
            kinds[gold[key]['value']['kind']]['correct'] += 1
            attributes[gold[key]['attribute']]['correct'] += 1
        if set(gold) != set(guessed):
            errors.append({'id': row['id'], 'text': row['text'], 'reference_form': list(gold.values()),
                           'predicted_form': prediction['predicted_form'],
                           'missing': [gold[k] for k in gold if k not in guessed],
                           'extra': [guessed[k] for k in guessed if k not in gold]})
    by_kind = {k: scores(v['correct'], v['predicted'], v['reference']) for k, v in sorted(kinds.items())}
    by_attribute = {k: scores(v['correct'], v['predicted'], v['reference']) for k, v in sorted(attributes.items())}
    result = {**scores(tp, predicted_count, expected_count), 'rows': len(rows),
              'exact_form_rows': exact, 'exact_form_row_rate': exact / len(rows) if rows else 0,
              'nonempty_exact_form_rows': nonempty_exact,
              'rows_with_predictions': sum(bool(p['predicted_form']) for p in predictions),
              'by_kind': by_kind,
              'macro_attribute_f1': sum(m['f1'] for m in by_attribute.values()) / len(by_attribute) if by_attribute else 0}
    return result, by_attribute, errors
