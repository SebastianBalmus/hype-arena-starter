"""Typed values: normalization, matching and comparison-operator checks.

A value is the dict shape of contracts/claimed_value.schema.json, e.g.
{"kind": "money", "value": 59, "currency": "EUR", "per": "month"}.

Two values from different rows can describe the same fact in different ways:
14 days vs 2 weeks, 1 GB vs 1,000 MB, 2026-01 vs 2026-01-12, Nimbus vs Nimbus Cloud.
`compare` tells those apart from real contradictions.
"""
from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

EQUAL = "equal"                # same value
COMPATIBLE = "compatible"      # one is a less precise version of the other (dates)
DIFFERENT = "different"        # same dimension, different value: a contradiction
INCOMPARABLE = "incomparable"  # different kinds/dimensions: says nothing

# Factors to a base unit per dimension (data in bytes, length in m, mass in g, ...).
_FACTORS = {
    "data": {"B": 1, "KB": 10**3, "MB": 10**6, "GB": 10**9, "TB": 10**12},
    "length": {"mm": Decimal("0.001"), "cm": Decimal("0.01"), "m": 1, "km": 1000, "in": Decimal("0.0254")},
    "mass": {"g": 1, "kg": 1000},
    "charge": {"mAh": 1},
    "frequency": {"Hz": 1, "kHz": 10**3, "MHz": 10**6, "GHz": 10**9},
    "power": {"mW": Decimal("0.001"), "W": 1, "kW": 1000},
}
_UNIT_ALIASES = {
    "kb": "KB", "mb": "MB", "gb": "GB", "tb": "TB", "byte": "B", "kilobyte": "KB", "megabyte": "MB",
    "gigabyte": "GB", "gig": "GB", "terabyte": "TB",
    "millimetre": "mm", "millimeter": "mm", "centimetre": "cm", "centimeter": "cm", "metre": "m",
    "meter": "m", "kilometre": "km", "kilometer": "km", "inch": "in", '"': "in", "″": "in",
    "gram": "g", "gramme": "g", "kilogram": "kg", "kilo": "kg",
    "hertz": "Hz", "kilohertz": "kHz", "megahertz": "MHz", "gigahertz": "GHz",
    "milliwatt": "mW", "watt": "W", "kilowatt": "kW",
}
_UNIT_DIM = {u: dim for dim, units in _FACTORS.items() for u in units}

# Durations: fixed units convert to seconds; months/years only to each other (never to days).
_FIXED_SECONDS = {"ms": Decimal("0.001"), "s": 1, "min": 60, "h": 3600, "day": 86400, "week": 604800}
_CALENDAR_MONTHS = {"month": 1, "year": 12}
_DURATION_ALIASES = {
    "sec": "s", "second": "s", "minute": "min", "hr": "h", "hour": "h", "d": "day",
    "wk": "week", "fortnight": "week", "mo": "month", "mth": "month", "yr": "year", "decade": "year",
}
_DURATION_SCALE = {"fortnight": 2, "decade": 10}


def _dec(x) -> Decimal:
    return Decimal(str(x))


def _singular(noun: str) -> str:
    n = noun.lower()
    if n == "people":
        return "person"
    if n.endswith("ies") and len(n) > 4:
        return n[:-3] + "y"
    if n.endswith("s") and not n.endswith("ss") and len(n) > 3:
        return n[:-1]
    return n


def name_key(name: str) -> str:
    """Case- and space-insensitive key for names ("Gavin Bellson" == "GavinBellson")."""
    return "".join(name.lower().split())


class NameNormalizer:
    """Maps name spellings to one canonical name.

    Built from the lexicon (subject names + aliases) and every name value seen in the data.
    A short form that is a token-prefix of exactly one longer known name maps to it
    (Nimbus -> Nimbus Cloud, Gavin -> Gavin Bellson); ambiguous short forms stay as they are.
    """

    def __init__(self, lexicon: dict | None = None, extra_names=()):
        self.canon: dict[str, str] = {}
        full_names: set[str] = set()
        for s in (lexicon or {}).get("subjects", []):
            self.canon[name_key(s["name"])] = s["name"]
            full_names.add(s["name"])
            for a in s.get("aliases", []):
                self.canon[name_key(a)] = s["name"]
        for n in extra_names:
            full_names.add(n)
            self.canon.setdefault(name_key(n), n)
        tokens = {n: n.lower().split() for n in full_names}
        for short in full_names:
            st = tokens[short]
            longer = [n for n, t in tokens.items() if len(t) > len(st) and t[: len(st)] == st]
            targets = {self.canon.get(name_key(n), n) for n in longer}
            if len(targets) == 1:
                self.canon[name_key(short)] = targets.pop()

    def __call__(self, name: str) -> str:
        return self.canon.get(name_key(name), name)


@dataclass(frozen=True)
class Norm:
    """A value reduced to a comparable form: a dimension plus a magnitude/token."""
    dim: tuple
    value: object       # Decimal for amounts, (start, end) dates for dates, str/bool otherwise
    precision: int = 0  # dates: 4 (year), 7 (month) or 10 (day)


def _date_interval(s: str) -> tuple[date, date]:
    parts = [int(p) for p in s.split("-")]
    if len(parts) == 1:
        return date(parts[0], 1, 1), date(parts[0], 12, 31)
    if len(parts) == 2:
        y, m = parts
        return date(y, m, 1), date(y, m, calendar.monthrange(y, m)[1])
    return date(*parts), date(*parts)


def normalize(v: dict, names: NameNormalizer | None = None) -> Norm:
    kind = v["kind"]
    if kind == "quantity":
        unit = v["unit"]
        unit = _UNIT_ALIASES.get(unit, _UNIT_ALIASES.get(unit.lower(), unit))
        if unit in _UNIT_DIM:
            dim = _UNIT_DIM[unit]
            return Norm(("quantity", dim), _dec(v["value"]) * _dec(_FACTORS[dim][unit]))
        return Norm(("quantity", "count", _singular(unit)), _dec(v["value"]))
    if kind == "money":
        return Norm(("money", v["currency"], v.get("per") or "once"), _dec(v["value"]))
    if kind == "percent":
        return Norm(("percent",), _dec(v["value"]))
    if kind == "duration":
        raw = v["unit"].lower() if len(v["unit"]) > 1 else v["unit"]
        scale = _DURATION_SCALE.get(raw, 1)
        unit = _DURATION_ALIASES.get(raw, raw)
        amount = _dec(v["value"]) * scale
        if unit in _FIXED_SECONDS:
            return Norm(("duration", "fixed"), amount * _dec(_FIXED_SECONDS[unit]))
        return Norm(("duration", "calendar"), amount * _CALENDAR_MONTHS[unit])
    if kind == "date":
        return Norm(("date",), _date_interval(v["value"]), len(v["value"]))
    if kind == "version":
        return Norm(("version",), str(v["value"]))
    if kind == "name":
        n = names(v["value"]) if names else v["value"]
        return Norm(("name",), name_key(n))
    if kind == "boolean":
        return Norm(("boolean",), bool(v["value"]))
    raise ValueError(f"unknown value kind: {kind!r}")


def compare(a: dict, b: dict, names: NameNormalizer | None = None) -> str:
    """Relation between two exact values (comparison operators are ignored here)."""
    na, nb = normalize(a, names), normalize(b, names)
    if na.dim != nb.dim:
        return INCOMPARABLE
    if na.dim == ("date",):
        (a0, a1), (b0, b1) = na.value, nb.value
        if (a0, a1) == (b0, b1):
            return EQUAL
        if (a0 <= b0 and b1 <= a1) or (b0 <= a0 and a1 <= b1):
            return COMPATIBLE
        return DIFFERENT
    return EQUAL if na.value == nb.value else DIFFERENT


def matches(a: dict, b: dict, names: NameNormalizer | None = None) -> bool:
    """True when two values do not contradict each other (equal or compatible)."""
    return compare(a, b, names) in (EQUAL, COMPATIBLE)


def satisfies(claimed: dict, actual: dict, names: NameNormalizer | None = None) -> bool | None:
    """Does an exact `actual` value make `claimed` (which may carry an op) true?

    Returns None when the two cannot be compared, or when date precision leaves it open.
    """
    op = claimed.get("op", "eq")
    if op == "eq":
        rel = compare(claimed, actual, names)
        return None if rel == INCOMPARABLE else rel in (EQUAL, COMPATIBLE)
    nc, na = normalize(claimed, names), normalize(actual, names)
    if nc.dim != na.dim or nc.dim in (("name",), ("boolean",), ("version",)):
        return None
    if nc.dim == ("date",):
        (c0, c1), (a0, a1) = nc.value, na.value
        sure_true = {"lt": a1 < c0, "le": a1 <= c1, "gt": a0 > c1, "ge": a0 >= c0}[op]
        sure_false = {"lt": a0 >= c0, "le": a0 > c1, "gt": a1 <= c1, "ge": a1 < c0}[op]
        return True if sure_true else False if sure_false else None
    x, y = na.value, nc.value
    return {"lt": x < y, "le": x <= y, "gt": x > y, "ge": x >= y}[op]
