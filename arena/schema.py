"""Draft 2020-12 validation of an entry against contracts/entry.schema.json."""
from __future__ import annotations

import json
import os
from functools import lru_cache

from jsonschema import Draft202012Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

_CONTRACTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "contracts")


def _load(name: str) -> dict:
    with open(os.path.join(_CONTRACTS, name), encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def _validator() -> Draft202012Validator:
    entry = _load("entry.schema.json")
    value = _load("claimed_value.schema.json")
    value_resource = Resource.from_contents(value, default_specification=DRAFT202012)
    registry = Registry().with_resources(
        [
            (entry["$id"], Resource.from_contents(entry, default_specification=DRAFT202012)),
            (value["$id"], value_resource),
            ("claimed_value.schema.json", value_resource),
        ]
    )
    return Draft202012Validator(entry, registry=registry)


def validate_entry(entry: dict) -> list[str]:
    """Every schema violation of ``entry``, as ``path: message``; empty when valid."""
    errors = sorted(_validator().iter_errors(entry), key=lambda e: list(map(str, e.absolute_path)))
    out = []
    for err in errors:
        path = "/".join(str(p) for p in err.absolute_path) or "(root)"
        out.append(f"{path}: {err.message}")
    return out
