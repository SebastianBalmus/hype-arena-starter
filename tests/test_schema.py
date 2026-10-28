import glob
import json

import pytest

from arena.schema import validate_entry

GOOD = sorted(glob.glob("contracts/examples/entry/good/*.json"))
BAD = sorted(glob.glob("contracts/examples/entry/bad/*.json"))


def _load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def test_examples_found():
    assert len(GOOD) == 7 and len(BAD) == 13


@pytest.mark.parametrize("path", GOOD)
def test_good_entries_validate(path):
    assert validate_entry(_load(path)) == []


@pytest.mark.parametrize("path", BAD)
def test_bad_entries_fail(path):
    assert validate_entry(_load(path)) != []


def test_value_schema_is_resolved():
    entry = _load("contracts/examples/entry/good/fact.json")
    entry["form"]["value"]["currency"] = "eur"
    assert validate_entry(entry) != []
    entry["form"]["value"]["currency"] = "EUR"
    entry["form"]["value"]["unit"] = "GB"
    assert validate_entry(entry) != []
