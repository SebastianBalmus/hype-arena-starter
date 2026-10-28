"""Exact Decimal JSON serialization for the knowledge base."""

from decimal import Decimal
import json


def exact_json(value, *, indent=None):
    if indent is not None:
        def formatted(item, level):
            if isinstance(item, (dict, list, tuple)) and item:
                padding = ' ' * (indent * (level + 1))
                if isinstance(item, dict):
                    parts = [json.dumps(k, ensure_ascii=False) + ': ' + formatted(v, level + 1) for k, v in item.items()]
                    left, right = '{', '}'
                else:
                    parts = [formatted(v, level + 1) for v in item]
                    left, right = '[', ']'
                return left + '\n' + padding + (',\n' + padding).join(parts) + '\n' + ' ' * (indent * level) + right
            return exact_json(item)
        return formatted(value, 0)
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("Nonfinite decimal")
        return str(value)
    if isinstance(value, dict):
        return "{" + ",".join(json.dumps(key, ensure_ascii=False) + ":" + exact_json(item) for key, item in value.items()) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(exact_json(item) for item in value) + "]"
    return json.dumps(value, ensure_ascii=False, allow_nan=False)
