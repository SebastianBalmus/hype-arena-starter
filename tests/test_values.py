import glob
import json

import pytest

from arena.values import (
    compare,
    has_negation,
    normalize,
    parse_values,
    render,
    satisfies,
    split_sentences,
    values_equal,
)


def Q(value, unit, op=None):
    v = {"kind": "quantity", "value": value, "unit": unit}
    if op:
        v["op"] = op
    return v


def M(value, currency, per=None, op=None):
    v = {"kind": "money", "value": value, "currency": currency}
    if per:
        v["per"] = per
    if op:
        v["op"] = op
    return v


def P(value, op=None):
    v = {"kind": "percent", "value": value}
    if op:
        v["op"] = op
    return v


def D(value, op=None):
    v = {"kind": "date", "value": value}
    if op:
        v["op"] = op
    return v


def V(value):
    return {"kind": "version", "value": value}


def T(value, unit, op=None):
    v = {"kind": "duration", "value": value, "unit": unit}
    if op:
        v["op"] = op
    return v


# Every "Written | Read as" row of contracts/value-forms.md (Quantity, counted nouns,
# Money, Percent, Date, Version, Duration, Comparisons).
READ_CASES = [
    # Quantity
    ("128 GB of storage", Q(128, "GB")),
    ("128GB", Q(128, "GB")),
    ("2 terabytes", Q(2, "TB")),
    ("a 5,000 mAh battery", Q(5000, "mAh")),
    ("weighs 1.2 kg", Q(1.2, "kg")),
    ('a 6.1" display', Q(6.1, "in")),
    ("a 6.1-inch display", Q(6.1, "in")),
    ("3.2 GHz", Q(3.2, "GHz")),
    ("a 65-watt charger", Q(65, "W")),
    ("water resistant to 100 m", Q(100, "m")),
    ("1 GB", Q(1000, "MB")),
    # Counted nouns
    ("speaks 40 languages", Q(40, "language")),
    ("speaks forty languages", Q(40, "language")),
    ("trusted by 10,000 companies", Q(10000, "company")),
    ("1,200 paying customers", Q(1200, "customer")),
    ("10,000 of our customers", Q(10000, "customer")),
    ("1.2 million users", Q(1200000, "user")),
    ("1M tests", Q(1000000, "test")),
    ("10k steps", Q(10000, "step")),
    ("twenty-four engineers", Q(24, "engineer")),
    ("a dozen partners", Q(12, "partner")),
    ("12 people", Q(12, "person")),
    ("zero outages", Q(0, "outage")),
    ("2,048 customers", Q(2048, "customer")),
    ("speaks 30 languages and never misses a beat", Q(30, "language")),
    # Money
    ("€49 a month", M(49, "EUR", "month")),
    ("€49/mo", M(49, "EUR", "month")),
    ("49 euros per month", M(49, "EUR", "month")),
    ("EUR 49 monthly", M(49, "EUR", "month")),
    ("€49,99 a month", M(49.99, "EUR", "month")),
    ("€10 per user per month", M(10, "EUR", "month")),
    ("AI Pro from €39/month", M(39, "EUR", "month")),
    ("£9 a week", M(9, "GBP", "week")),
    ("€490 per annum", M(490, "EUR", "year")),
    ("$299.99", M(299.99, "USD")),
    ("US$12", M(12, "USD")),
    ("a €12M seed round", M(12000000, "EUR")),
    ("€1.2bn", M(1200000000, "EUR")),
    ("1,999 lei", M(1999, "RON")),
    ("€49 a month or €490 a year", M(490, "EUR", "year")),
    ("€49 a month or €490 a year", M(49, "EUR", "month")),
    # Percent
    ("99.99% uptime", P(99.99)),
    ("99.9 % uptime", P(99.9)),
    ("50 percent", P(50)),
    ("50 per cent", P(50)),
    ("fifty percent", P(50)),
    ("99.999% uptime", P(99.999)),
    # Date
    ("2026-03-12", D("2026-03-12")),
    ("2026-03", D("2026-03")),
    ("12 March 2026", D("2026-03-12")),
    ("12th March 2026", D("2026-03-12")),
    ("the 12th of March 2026", D("2026-03-12")),
    ("12March 2026", D("2026-03-12")),
    ("March 12, 2026", D("2026-03-12")),
    ("Mar 12th 2026", D("2026-03-12")),
    ("since 1 Mar. 2026", D("2026-03-01")),
    ("March 2026", D("2026-03")),
    ("Sept 2026", D("2026-09")),
    ("founded in 2025", D("2025")),
    # Version
    ("v2.3", V("2.3")),
    ("version 2.3", V("2.3")),
    ("ver. 2.3", V("2.3")),
    ("v3", V("3")),
    ("2.3.1", V("2.3.1")),
    ("Hooly Gemini 2.2", V("2.2")),
    # Duration
    ("battery lasts 14 days", T(14, "day")),
    ("a 14-day trial", T(14, "day")),
    ("a 14-day trial", T(2, "week")),
    ("14days", T(14, "day")),
    ("two weeks", T(2, "week")),
    ("a two-week trial", T(2, "week")),
    ("for a fortnight", T(2, "week")),
    ("24 hours", T(24, "h")),
    ("24h", T(24, "h")),
    ("30 min", T(30, "min")),
    ("p95 latency 1.8 s", T(1.8, "s")),
    ("500 ms", T(500, "ms")),
    ("12 months", T(1, "year")),
    ("for a week", T(1, "week")),
    # Comparisons
    ("over 10,000 customers", Q(10000, "customer", "gt")),
    ("10,000+ customers", Q(10000, "customer", "ge")),
    ("at least 12 languages", Q(12, "language", "ge")),
    ("40 languages or more", Q(40, "language", "ge")),
    ("uploads up to 25 MB", Q(25, "MB", "le")),
    ("no more than 2 GB", Q(2, "GB", "le")),
    ("2 GB or less", Q(2, "GB", "le")),
    ("fewer than 50 employees", Q(50, "employee", "lt")),
    ("under $300", M(300, "USD", op="lt")),
    ("over 99% accuracy", P(99, "gt")),
    ("answers in under a second", T(1, "s", "lt")),
    ("within an hour", T(1, "h", "le")),
    ("before March 2026", D("2026-03", "lt")),
    ("after 2025", D("2025", "gt")),
    ("by 1 May 2026", D("2026-05-01", "le")),
]

# Every row of the "Not read" table of contracts/value-forms.md.
NOT_READ_CASES = [
    ("12/03/2026", D("2026-03-12")),
    ("12.03.2026", D("2026-03-12")),
    ("March 12", D("2026-03-12")),
    ("Q4 2026", D("2026")),
    ("12 March 2026", D("2026-03")),
    ("1.000.000 users", Q(1000000, "user")),
    ("1,5 GB", Q(1.5, "GB")),
    ("1,5 GB", Q(5, "GB")),
    ("99,9% uptime", P(9)),
    ("1.500 mAh", Q(1500, "mAh")),
    ("99.6% uptime", P(6)),
    ("we do not offer 99.9% uptime", P(99.9)),
    ("no 14-day trial", T(14, "day")),
    ("not founded by Gavin Bellson", {"kind": "name", "value": "Gavin Bellson"}),
    ("about 10,000 customers", Q(10000, "customer")),
    ("~10 GB", Q(10, "GB")),
    ("40 languages or so", Q(40, "language")),
    ("10–20 GB", Q(20, "GB")),
    ("10 GB to 20 GB", Q(10, "GB")),
    ("between 10 and 20 GB", Q(20, "GB")),
    ("1 or 2 days", T(2, "day")),
    ("10m users", Q(10000000, "user")),
    ("2026 customers", Q(2026, "customer")),
    ("half an hour", T(30, "min")),
    ("1,000 requests a day", T(1, "day")),
    ("€49 a month", M(49, "EUR")),
    ("5 pounds", M(5, "GBP")),
    ("Hooly Gemini 11", V("11")),
    ("v2.3", V("2.3.0")),
    ("30 days", T(1, "month")),
    ("over 10,000 customers", Q(10000, "customer")),
    ("10,000 of our happy loyal customers", Q(10000, "customer")),
]


@pytest.mark.parametrize("text,expected", READ_CASES)
def test_reads(text, expected):
    got = parse_values(text)
    assert any(values_equal(s.value, expected) and s.value.get("op") == expected.get("op") for s in got), [
        s.value for s in got
    ]


@pytest.mark.parametrize("text,not_expected", NOT_READ_CASES)
def test_not_read(text, not_expected):
    got = parse_values(text)
    assert not any(
        values_equal(s.value, not_expected) and s.value.get("op") == not_expected.get("op") for s in got
    ), [s.value for s in got]


def test_render_roundtrip():
    files = glob.glob("contracts/examples/claimed_value/good/*.json")
    assert files
    for f in files:
        with open(f) as fh:
            v = json.load(fh)
        if v["kind"] in ("name", "boolean") or "op" in v:
            continue
        assert any(values_equal(s.value, v) and "op" not in s.value for s in parse_values(render(v))), f


def test_split_and_negation():
    assert split_sentences("Hooly AI Pro is our plan. It costs €59 per month.") == [
        "Hooly AI Pro is our plan.",
        "It costs €59 per month.",
    ]
    assert len(split_sentences("Hooly Inc. costs e.g. €5 a month.")) == 1
    assert has_negation("The app isn't SOC 2 certified") and not has_negation("Hooly passed its SOC 2 audit")


def test_compare_and_satisfies():
    m = lambda v: {"kind": "money", "value": v, "currency": "EUR", "per": "month"}  # noqa: E731
    assert compare(m(199), m(59)) == 1 and compare(m(59), {"kind": "money", "value": 590, "currency": "EUR", "per": "year"}) is None
    assert compare({"kind": "date", "value": "2026-03"}, {"kind": "date", "value": "2026-03-12"}) is None
    assert compare({"kind": "duration", "value": 14, "unit": "day"}, {"kind": "duration", "value": 2, "unit": "week"}) == 0
    assert satisfies({"kind": "quantity", "value": 120, "unit": "customer"}, {"kind": "quantity", "value": 100, "unit": "customer", "op": "gt"})


# --- Additional checks of spec rules beyond the tables -------------------------------


def test_render_forms():
    assert render(M(59, "EUR", "month")) == "€59 per month"
    assert render(M(299.99, "USD")) == "$299.99"
    assert render(M(9, "GBP", "week")) == "£9 per week"
    assert render(M(59, "CHF", "month")) == "59 CHF per month"
    assert render(Q(2048, "customer")) == "2,048 customers"
    assert render(Q(1, "customer")) == "1 customer"
    assert render(Q(12, "person")) == "12 people"
    assert render(Q(3, "company")) == "3 companies"
    assert render(Q(128, "GB")) == "128 GB"
    assert render(P(99.9)) == "99.9%"
    assert render(D("2026-03-12")) == "12 March 2026"
    assert render(D("2026-03")) == "March 2026"
    assert render(D("2025")) == "2025"
    assert render(V("2.3")) == "v2.3"
    assert render(T(500, "ms")) == "500 ms"
    assert render(T(1.8, "s")) == "1.8 s"
    assert render(T(30, "min")) == "30 min"
    assert render(T(24, "h")) == "24 hours"
    assert render(T(14, "day")) == "14 days"
    assert render(T(2, "week")) == "2 weeks"
    assert render(T(3, "month")) == "3 months"
    assert render(T(1, "year")) == "1 year"


@pytest.mark.parametrize(
    "v",
    [
        M(59, "EUR", "month"), M(49.99, "EUR", "year"), M(12000000, "EUR"), M(10, "CHF", "month"),
        Q(2026, "customer"), Q(1, "company"), Q(1.234, "GB"), Q(5000, "mAh"), Q(40, "language"),
        P(99.999), D("2026-03-12"), D("2026-09"), D("2025"), V("2.3.1"),
        T(1, "h"), T(1.5, "s"), T(2, "week"), T(10, "year"),
    ],
)
def test_render_reads_back(v):
    got = parse_values(render(v))
    assert any(values_equal(s.value, v) and "op" not in s.value for s in got), (render(v), [s.value for s in got])


def test_invisible_characters_and_soft_hyphen():
    assert not any(values_equal(s.value, Q(128, "GB")) for s in parse_values("128​GB"))
    assert not any(values_equal(s.value, Q(128, "GB")) for s in parse_values("128 GB​ of storage"))
    assert not any(values_equal(s.value, Q(40, "language")) for s in parse_values("speaks ⁠40 languages"))
    assert not any(values_equal(s.value, P(99.9)) for s in parse_values("99.9%﻿ uptime"))
    assert any(values_equal(s.value, Q(40, "language")) for s in parse_values("speaks 40 lan­guages"))


def test_change_phrase_values_read_each_side():
    vals = [s.value for s in parse_values("Hooly AI Pro went from €49 per month to €59 per month.")]
    assert any(values_equal(v, M(49, "EUR", "month")) for v in vals)
    assert any(values_equal(v, M(59, "EUR", "month")) for v in vals)
    vals = [s.value for s in parse_values("Hooly AI Pro went from €49 to €59 per month.")]
    assert any(values_equal(v, M(49, "EUR")) for v in vals)


def test_spans_point_into_text():
    text = "Hooly AI Pro costs €59 per month."
    (span,) = parse_values(text)
    assert text[span.start:span.end] == "€59 per month"


def test_normalize_drops_op_and_converts():
    assert normalize(Q(1, "GB", "gt")) == {"kind": "quantity", "value": 1000000000, "unit": "B"}
    assert normalize(Q(3, "Customers")) == {"kind": "quantity", "value": 3, "unit": "customer"}
    assert values_equal(Q(128, "gigabyte"), Q(128, "GB"))
    assert not values_equal(T(30, "day"), T(1, "month"))
