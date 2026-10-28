"""The value reader of contracts/value-forms.md.

``parse_values`` reads every quantity, money, percent, date, version and duration a text
states plainly; ``normalize``/``values_equal``/``compare``/``satisfies`` compare typed values;
``render`` writes a value in a canonical form ``parse_values`` reads back. Names and yes/no
values are not read here.
"""

from __future__ import annotations

import calendar
import math
import re
from dataclasses import dataclass
from decimal import Decimal

__all__ = [
    "ValueSpan",
    "split_sentences",
    "has_negation",
    "parse_values",
    "normalize",
    "values_equal",
    "compare",
    "satisfies",
    "render",
]


@dataclass
class ValueSpan:
    start: int
    end: int
    value: dict


# --------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------

_INVISIBLE = "​⁠﻿"
_SOFT_HYPHEN = "­"

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7,
    "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
_MONTH_NAMES = [
    "", "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December",
]
_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))
_WEEKDAYS = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}

_ISO_CODES = [
    "EUR", "USD", "GBP", "CHF", "RON", "PLN", "CZK", "HUF", "SEK", "NOK", "DKK",
    "JPY", "CNY", "INR", "CAD", "AUD",
]
_SYMBOL_CURRENCY = {"US$": "USD", "€": "EUR", "$": "USD", "£": "GBP"}
_WORD_CURRENCY = {"euro": "EUR", "euros": "EUR", "dollar": "USD", "dollars": "USD", "lei": "RON"}
_RENDER_SYMBOL = {"EUR": "€", "USD": "$", "GBP": "£"}

# canonical physical unit -> (base unit, factor to base)
_PHYS = {
    "B": ("B", Decimal(1)), "KB": ("B", Decimal(10) ** 3), "MB": ("B", Decimal(10) ** 6),
    "GB": ("B", Decimal(10) ** 9), "TB": ("B", Decimal(10) ** 12),
    "mm": ("m", Decimal("0.001")), "cm": ("m", Decimal("0.01")), "m": ("m", Decimal(1)),
    "km": ("m", Decimal(1000)), "in": ("m", Decimal("0.0254")),
    "g": ("g", Decimal(1)), "kg": ("g", Decimal(1000)),
    "mAh": ("mAh", Decimal(1)),
    "Hz": ("Hz", Decimal(1)), "kHz": ("Hz", Decimal(10) ** 3), "MHz": ("Hz", Decimal(10) ** 6),
    "GHz": ("Hz", Decimal(10) ** 9),
    "mW": ("W", Decimal("0.001")), "W": ("W", Decimal(1)), "kW": ("W", Decimal(1000)),
}
# symbols, case-sensitive
_PHYS_SYMBOLS = {
    "B": "B", "KB": "KB", "kB": "KB", "kb": "KB", "MB": "MB", "mb": "MB", "GB": "GB",
    "gb": "GB", "TB": "TB", "tb": "TB", "mm": "mm", "cm": "cm", "m": "m", "km": "km",
    "in": "in", "g": "g", "kg": "kg", "mAh": "mAh", "Hz": "Hz", "kHz": "kHz", "MHz": "MHz",
    "GHz": "GHz", "mW": "mW", "W": "W", "kW": "kW",
}
# words, any case: (singular, plural or None) -> canonical
_PHYS_WORDS = [
    ("byte", "bytes", "B"), ("kilobyte", "kilobytes", "KB"), ("megabyte", "megabytes", "MB"),
    ("gigabyte", "gigabytes", "GB"), ("gig", "gigs", "GB"), ("terabyte", "terabytes", "TB"),
    ("millimetre", "millimetres", "mm"), ("millimeter", "millimeters", "mm"),
    ("centimetre", "centimetres", "cm"), ("centimeter", "centimeters", "cm"),
    ("metre", "metres", "m"), ("meter", "meters", "m"),
    ("kilometre", "kilometres", "km"), ("kilometer", "kilometers", "km"),
    ("inch", "inches", "in"),
    ("gram", "grams", "g"), ("gramme", "grammes", "g"),
    ("kilogram", "kilograms", "kg"), ("kilo", "kilos", "kg"),
    ("hertz", None, "Hz"), ("kilohertz", None, "kHz"), ("megahertz", None, "MHz"),
    ("gigahertz", None, "GHz"),
    ("milliwatt", "milliwatts", "mW"), ("watt", "watts", "W"), ("kilowatt", "kilowatts", "kW"),
]
_PHYS_WORD_MAP = {}
for _s, _p, _c in _PHYS_WORDS:
    _PHYS_WORD_MAP[_s] = _c
    if _p:
        _PHYS_WORD_MAP[_p] = _c

# canonical duration unit -> (comparison base, factor)
_DUR = {
    "ms": ("s", Decimal("0.001")), "s": ("s", Decimal(1)), "min": ("s", Decimal(60)),
    "h": ("s", Decimal(3600)), "day": ("s", Decimal(86400)), "week": ("s", Decimal(604800)),
    "month": ("month", Decimal(1)), "year": ("month", Decimal(12)),
}
# symbols, case-sensitive: written -> (canonical, multiplier)
_DUR_SYMBOLS = {"ms": ("ms", 1), "s": ("s", 1), "h": ("h", 1), "d": ("day", 1)}
# words, any case
_DUR_WORDS = [
    ("second", "s", 1), ("sec", "s", 1), ("minute", "min", 1), ("min", "min", 1),
    ("hour", "h", 1), ("hr", "h", 1), ("day", "day", 1), ("week", "week", 1),
    ("wk", "week", 1), ("fortnight", "week", 2), ("month", "month", 1), ("mo", "month", 1),
    ("mth", "month", 1), ("year", "year", 1), ("yr", "year", 1), ("decade", "year", 10),
]
_DUR_WORD_MAP = {}
for _w, _c, _m in _DUR_WORDS:
    _DUR_WORD_MAP[_w] = (_c, _m)
    _DUR_WORD_MAP[_w + "s"] = (_c, _m)
for _k in ("ms", "s", "min", "h", "day", "week", "month", "year"):
    _DUR_WORD_MAP.setdefault(_k, (_k, 1))

_RENDER_DUR = {
    "ms": ("ms", "ms"), "s": ("s", "s"), "min": ("min", "min"), "h": ("hour", "hours"),
    "day": ("day", "days"), "week": ("week", "weeks"), "month": ("month", "months"),
    "year": ("year", "years"),
}

_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
    "eighteen", "nineteen",
]
_TENS = ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_ONES_VAL = {w: i for i, w in enumerate(_ONES)}
_TENS_VAL = {w: 20 + 10 * i for i, w in enumerate(_TENS)}

# --------------------------------------------------------------------------------------
# Regexes
# --------------------------------------------------------------------------------------

_SP = r"[ \t ]"
_GAP1 = rf"(?:{_SP}+(?:\n{_SP}*)?|\n{_SP}*)"  # at least one space, never a blank line
_GAP1_RE = re.compile(_GAP1)

_NUM_RE = re.compile(r"(?<![A-Za-z0-9_.,])(?<![A-Za-z]-)\d+(?:[.,]\d+)*")

_ones_alt = "|".join(sorted(_ONES, key=len, reverse=True))
_ones19_alt = "|".join(_ONES[1:10])
_tens_alt = "|".join(_TENS)
_WORDNUM_RE = re.compile(
    rf"(?i)\b(?:(?:a|{_ones19_alt}){_SP}+hundred|a{_SP}+dozen"
    rf"|(?:{_tens_alt})(?:[-\s](?:{_ones19_alt})(?![A-Za-z]))?|(?:{_ones_alt}))\b"
)
_dur_word_alt = "|".join(sorted({w for w, _, _ in _DUR_WORDS}, key=len, reverse=True))
_A_DUR_RE = re.compile(
    rf"(?i)\b(?:under|within|in|for|over|than|at{_SP}+least|at{_SP}+most|to|only|just|about|around)"
    rf"{_SP}+(an?)(?={_SP}+(?:{_dur_word_alt})\b)"
)

_SCALE_DIGIT_RE = re.compile(
    rf"{_SP}*(?i:(thousand|million|billion|bn))(?![A-Za-z])|(M)(?![A-Za-z0-9])|(k)(?![A-Za-z0-9])"
)
_SCALE_WORD_RE = re.compile(rf"{_SP}+(?i:(thousand|million|billion))\b")
_SCALES = {"thousand": 10**3, "k": 10**3, "million": 10**6, "m": 10**6, "billion": 10**9, "bn": 10**9}

_iso_alt = "|".join(_ISO_CODES)
_CUR_BEFORE_RE = re.compile(rf"(?:(US\$|€|\$|£)|(?<![A-Za-z])({_iso_alt})){_SP}?$")
_CUR_AFTER_RE = re.compile(
    rf"{_SP}*(US\$|€|\$|£)|{_SP}*({_iso_alt})(?![A-Za-z])|(?:{_SP}+|-)(?i:(euros?|dollars?|lei))(?![A-Za-z])"
)
_PER_SEAT = rf"(?:{_SP}*/{_SP}*(?i:user|seat)|{_SP}+(?i:per){_SP}+(?i:user|seat))"
_PERIOD_RES = [
    (re.compile(rf"{_SP}*/{_SP}*(?i:(month|mo|mth|day|week|wk|year|yr))(?![A-Za-z])"), None),
    (re.compile(rf"{_SP}+(?i:a|an|per){_SP}+(?i:(month|day|week|year))(?![A-Za-z])"), None),
    (re.compile(rf"{_SP}+(?i:per){_SP}+(?i:annum)(?![A-Za-z])"), "year"),
    (re.compile(rf"{_SP}+(?i:(monthly|daily|weekly|yearly|annually))(?![A-Za-z])"), None),
]
_PERIOD_WORD = {
    "month": "month", "mo": "month", "mth": "month", "monthly": "month",
    "day": "day", "daily": "day", "week": "week", "wk": "week", "weekly": "week",
    "year": "year", "yr": "year", "yearly": "year", "annually": "year",
}
_PCT_RE = re.compile(rf"{_SP}*%|(?:{_SP}*|-)(?i:percent|per{_SP}+cent)(?![A-Za-z])")

_V_RE = re.compile(r"(?<![A-Za-z0-9.])[vV](\d+(?:\.\d+){0,3})(?![\d]|\.\d)")
_VER_WORD_RE = re.compile(rf"(?i)\b(?:version|ver\.?){_SP}*(\d+(?:\.\d+){{0,3}})(?![\d]|\.\d)")
_SKIP_RE = re.compile(r"(?<![\d])\d+(?:[/:]\d+)+")

_ORD = r"(?:st|nd|rd|th)?"
_DATE_RES = [
    ("iso", re.compile(r"(?<![\d\-/.])(\d{4})-(\d{2})(?:-(\d{2}))?(?![\d\-])")),
    ("dmy", re.compile(
        rf"(?i)(?<![\w.,])(?:the{_SP}+)?(\d{{1,2}}){_ORD}(?:{_SP}+of)?{_SP}*({_MONTH_ALT})\b\.?,?{_SP}+(\d{{4}})(?!\d)"
    )),
    ("mdy", re.compile(
        rf"(?i)\b({_MONTH_ALT})\b\.?{_SP}+(\d{{1,2}}){_ORD},?{_SP}+(\d{{4}})(?!\d)"
    )),
    ("my", re.compile(rf"(?i)\b({_MONTH_ALT})\b\.?,?{_SP}+(\d{{4}})(?!\d)")),
]


def _phrase_re(phrases):
    alts = sorted(phrases, key=len, reverse=True)
    body = "|".join(re.escape(p).replace(r"\ ", rf"{_SP}+") for p in alts)
    return body


_GEN_OPS = {
    "over": "gt", "more than": "gt", "greater than": "gt", "higher than": "gt", "above": "gt",
    "exceeding": "gt", "in excess of": "gt", "upwards of": "gt",
    "under": "lt", "less than": "lt", "fewer than": "lt", "lower than": "lt", "below": "lt",
    "up to": "le", "at most": "le", "no more than": "le", "not more than": "le",
    "maximum": "le", "max": "le", "max.": "le",
    "at least": "ge", "no less than": "ge", "not less than": "ge", "no fewer than": "ge",
    "minimum": "ge",
}
_DUR_OPS = dict(_GEN_OPS, within="le")
_DATE_OPS = {
    "before": "lt", "prior to": "lt", "after": "gt", "by": "le", "no later than": "le",
    "no earlier than": "ge",
}
_AFTER_OPS = {
    "or less": "le", "or fewer": "le", "or under": "le", "or below": "le", "or lower": "le",
    "or shorter": "le", "or more": "ge", "or higher": "ge", "or above": "ge", "or longer": "ge",
    "or over": "ge",
}


def _before_re(table):
    return re.compile(rf"(?i)(?<![A-Za-z])({_phrase_re(table)}){_SP}*$")


_GEN_OP_RE = _before_re(_GEN_OPS)
_DUR_OP_RE = _before_re(_DUR_OPS)
_DATE_OP_RE = _before_re(_DATE_OPS)
_AFTER_OP_RE = re.compile(rf"(?i){_SP}*,?{_SP}*({_phrase_re(_AFTER_OPS)})(?![A-Za-z])")

_APPROX_WORDS = [
    "about", "around", "approximately", "approx", "roughly", "nearly", "almost", "circa", "ca",
    "some", "estimated", "est", "close to", "more or less",
]
_APPROX_BEFORE_RE = re.compile(
    rf"(?i)(?:(?<![A-Za-z])(?:{_phrase_re(_APPROX_WORDS)})\.?|[~≈±]){_SP}*$"
)
_APPROX_AFTER_RE = re.compile(rf"(?i)(?:-?ish(?![A-Za-z])|{_SP}*,?{_SP}*(?:or{_SP}+so|give{_SP}+or{_SP}+take)(?![A-Za-z]))")
_ISH_RE = re.compile(r"(?i)-?ish(?![A-Za-z])")

_NEG_RE = re.compile(
    r"(?i)\b(?:not|never|none|nobody|nothing|neither|nor|without|cannot)\b|\bno\b(?!\.\s*\d)|[A-Za-z]n['’]t\b"
)
_NEG_OP_RE = re.compile(rf"(?i)\b(?:no|not){_SP}+(?:more|less|fewer|later|earlier){_SP}+than\b")

_QUARTER_RE = re.compile(rf"(?<![A-Za-z0-9])[QH][1-4]{_SP}*$")
_CAP_WORD_RE = re.compile(rf"(?<![A-Za-z0-9])([A-Z][A-Za-z0-9]*){_SP}$")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z][A-Za-z0-9]*)*(?![A-Za-z0-9])")

_SENT_END_RE = re.compile(r"[.!?]+[\"'”’)\]]*(?=\s|$)")
_ABBREVIATIONS = {
    "e.g.", "i.e.", "vs.", "approx.", "inc.", "ltd.", "co.", "corp.", "dr.", "mr.", "mrs.",
    "ms.", "prof.", "st.",
}
# Points that end nothing when a number follows: "No. 1", "ver. 2.3", "1 Mar. 2026".
_ABBREV_BEFORE_NUMBER = {"no.", "ver."} | {m + "." for m in _MONTHS if len(m) <= 4}

# Words that end the search for a counted noun.
_HALT = {
    "and", "or", "but", "nor", "to", "than", "in", "on", "at", "for", "with", "by", "from",
    "per", "as", "that", "which", "who", "whom", "whose", "if", "when", "while", "since",
    "until", "over", "under", "into", "onto", "more", "less", "fewer", "a", "an", "is", "are",
    "was", "were", "be", "been", "being", "has", "have", "had", "does", "do", "did", "will",
    "would", "can", "could", "should", "may", "might", "must", "shall", "costs", "cost",
    "speaks", "supports", "includes", "offers", "runs", "lasts", "weighs", "holds", "comes",
    "goes", "works", "makes", "takes", "gets", "needs", "requires", "provides", "allows",
    "enables", "means", "seems", "remains", "becomes", "uses", "says", "gives", "adds", "plus",
    "minus", "out", "off", "up", "down", "so", "after", "before", "during", "about", "around",
    "between", "among", "within", "without", "not", "no", "never", "via", "vs", "versus",
    "through", "across", "times", "percent", "cent",
} | set(_MONTHS) | _WEEKDAYS
_DETERMINERS = {"of", "the", "our", "their", "its", "your", "my", "his", "her", "these", "those", "this", "that"}
_NOT_NOUN = {
    "its", "his", "hers", "ours", "yours", "theirs", "this", "thus", "yes", "always", "perhaps",
    "unless", "whereas", "versus", "besides", "towards", "afterwards", "us", "as", "plus",
    "less", "is", "was", "has", "does", "alas", "series", "species", "news",
}
_IRREGULAR = {"person": "people", "child": "children", "man": "men", "woman": "women", "mouse": "mice"}
_IRREGULAR_PLURAL = {v: k for k, v in _IRREGULAR.items()}
_INVARIANT = {"series", "species", "news"}

# Capitalised words that are not a product name before a version number.
_CAP_STOP = {
    "at", "in", "on", "for", "from", "to", "by", "the", "a", "an", "only", "just", "over",
    "under", "up", "about", "around", "with", "within", "of", "and", "or", "but", "since",
    "after", "before", "is", "it", "its", "our", "we", "this", "that", "these", "those", "then",
    "now", "version", "ver", "v", "page", "step", "section", "chapter", "table", "figure", "no",
    "costs", "weighs", "lasts", "than", "between", "as", "approx", "about", "nearly",
} | set(_MONTHS) | _WEEKDAYS


# --------------------------------------------------------------------------------------
# Units
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Unit:
    regex: re.Pattern
    kind: str  # quantity | duration
    canon: str
    mult: int
    written: str


def _build_units():
    units = []
    for sym, canon in _PHYS_SYMBOLS.items():
        units.append(_Unit(re.compile(re.escape(sym) + r"(?![A-Za-z0-9])"), "quantity", canon, 1, sym))
    for sing, plur, canon in _PHYS_WORDS:
        body = re.escape(sing) if not plur else f"(?:{re.escape(plur)}|{re.escape(sing)})"
        units.append(_Unit(re.compile(f"(?i:{body})(?![A-Za-z0-9])"), "quantity", canon, 1, sing))
    for sym, (canon, mult) in _DUR_SYMBOLS.items():
        units.append(_Unit(re.compile(re.escape(sym) + r"(?![A-Za-z0-9])"), "duration", canon, mult, sym))
    for word, canon, mult in _DUR_WORDS:
        units.append(
            _Unit(re.compile(f"(?i:{re.escape(word)}s?)(?![A-Za-z0-9])"), "duration", canon, mult, word)
        )
    return units


_UNITS = _build_units()
_INCH_MARK_RE = re.compile(r"[\"″]")


def _match_unit(clean, pos, kinds=("quantity", "duration"), gaps=("att", "sp", "hy")):
    """Longest unit after ``pos``: (end, unit) or None."""
    starts = []
    if "att" in gaps:
        starts.append((pos, "att"))
    if "sp" in gaps:
        m = _GAP1_RE.match(clean, pos)
        if m:
            starts.append((m.end(), "sp"))
    if "hy" in gaps and clean.startswith("-", pos):
        starts.append((pos + 1, "hy"))
    best = None
    if "quantity" in kinds and "att" in gaps and _INCH_MARK_RE.match(clean, pos):
        best = (pos + 1, _Unit(_INCH_MARK_RE, "quantity", "in", 1, '"'))
    for upos, gap in starts:
        for unit in _UNITS:
            if unit.kind not in kinds:
                continue
            m = unit.regex.match(clean, upos)
            if not m:
                continue
            # "in" spaced before a word is the preposition, not an inch: "1 in 5 users".
            if unit.written == "in" and gap == "sp" and re.match(rf"{_SP}+[A-Za-z0-9]", clean[m.end():]):
                continue
            if best is None or m.end() > best[0]:
                best = (m.end(), unit)
    return best


# --------------------------------------------------------------------------------------
# Numbers
# --------------------------------------------------------------------------------------


def _tonum(d: Decimal):
    if d == d.to_integral_value():
        return int(d)
    return float(d)


def _dec(x) -> Decimal:
    if isinstance(x, Decimal):
        return x
    if isinstance(x, bool):
        raise TypeError("boolean is not a number")
    if isinstance(x, int):
        return Decimal(x)
    return Decimal(repr(float(x)))


def _classify(raw: str):
    """(value or None, flag) for a written digit number, per value-forms "Numbers"."""
    if "," in raw and "." in raw:
        if re.fullmatch(r"\d{1,3}(?:,\d{3})+\.\d+", raw):
            return Decimal(raw.replace(",", "")), "num"
        return None, "bad"
    if "," in raw:
        if re.fullmatch(r"\d{1,3}(?:,\d{3})+", raw):
            return Decimal(raw.replace(",", "")), "int"
        if re.fullmatch(r"\d+,\d{2}", raw):
            return Decimal(raw.replace(",", ".")), "comma"  # a price only
        return None, "bad"
    dots = raw.count(".")
    if dots >= 2:
        return None, "multi"
    if dots == 1:
        if len(raw.split(".")[1]) == 3:
            return Decimal(raw), "amb3"  # a percentage only
        return Decimal(raw), "dec"
    if len(raw) == 4 and 1900 <= int(raw) <= 2099:
        return Decimal(raw), "year"
    return Decimal(raw), "int"


def _word_value(s: str) -> int:
    s = re.sub(r"\s+", " ", s.lower().replace("-", " ")).strip()
    parts = s.split(" ")
    if parts[-1] == "dozen":
        return 12
    if parts[-1] == "hundred":
        return 100 * (1 if parts[0] == "a" else _ONES_VAL[parts[0]])
    total = 0
    for p in parts:
        total += _TENS_VAL.get(p, 0) or _ONES_VAL.get(p, 0)
    return total


def _version_ok(raw: str) -> bool:
    groups = raw.split(".")
    if len(groups) > 4:
        return False
    if all(len(g) == 3 for g in groups[1:]):
        return False  # points between thousands: 1.000.000
    if re.fullmatch(r"\d{1,2}\.\d{1,2}\.\d{2,4}", raw):
        return False  # a numeric date: 12.03.2026
    return True


# --------------------------------------------------------------------------------------
# Plurals
# --------------------------------------------------------------------------------------


def _pluralize(noun: str) -> str:
    if noun in _IRREGULAR:
        return _IRREGULAR[noun]
    if noun in _INVARIANT:
        return noun
    if len(noun) > 1 and noun.endswith("y") and noun[-2] not in "aeiou":
        return noun[:-1] + "ies"
    if noun.endswith(("s", "x", "z", "ch", "sh")):
        return noun + "es"
    return noun + "s"


def _singularize(word: str) -> str:
    w = word.lower()
    if w in _IRREGULAR_PLURAL:
        return _IRREGULAR_PLURAL[w]
    if w in _INVARIANT or w in _IRREGULAR:
        return w
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if w.endswith(("sses", "xes", "zzes", "ches", "shes")):
        return w[:-2]
    if len(w) > 2 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def _noun_key(unit: str) -> str:
    """One key for a counted noun's singular and plural spellings."""
    return _singularize(_pluralize(_singularize(unit)))


def _looks_plural(w: str) -> bool:
    lw = w.lower()
    if lw in _IRREGULAR_PLURAL:
        return True
    if lw in _NOT_NOUN or lw in _DETERMINERS:
        return False
    return len(lw) >= 3 and lw.endswith("s") and not lw.endswith(("ss", "us", "is"))


# --------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------


@dataclass
class _Tok:
    start: int
    end: int
    raw: str
    value: Decimal | None
    flag: str  # int num dec amb3 comma bad multi year word a


@dataclass
class _Atom:
    start: int
    end: int
    value: dict | None  # None: a bare number, or one that reads as nothing
    plus: bool = False
    approx: bool = False
    excluded: bool = False


def _prepare(text: str):
    """Drop soft hyphens between letters; map clean offsets back to the text's."""
    chars, idx = [], []
    n = len(text)
    for i, c in enumerate(text):
        if c == _SOFT_HYPHEN and 0 < i < n - 1 and text[i - 1].isalpha() and text[i + 1].isalpha():
            continue
        chars.append(c)
        idx.append(i)
    idx.append(n)
    clean = "".join(chars)
    poison = [c in _INVISIBLE or c == _SOFT_HYPHEN for c in clean]
    return clean, idx, poison


def _valid_date(y: int, m: int, d: int | None) -> bool:
    if not 1 <= m <= 12:
        return False
    if d is None:
        return True
    return 1 <= d <= calendar.monthrange(y, m)[1]


def _month_ok(written: str) -> bool:
    # "may" is the month only when written "May".
    return not (written.lower() == "may" and written[0] == "m")


def _find_dates(clean):
    found = []
    for kind, rx in _DATE_RES:
        for m in rx.finditer(clean):
            if kind == "iso":
                y, mo = int(m.group(1)), int(m.group(2))
                d = int(m.group(3)) if m.group(3) else None
            elif kind == "dmy":
                if not _month_ok(m.group(2)):
                    continue
                d, mo, y = int(m.group(1)), _MONTHS[m.group(2).lower()], int(m.group(3))
            elif kind == "mdy":
                if not _month_ok(m.group(1)):
                    continue
                mo, d, y = _MONTHS[m.group(1).lower()], int(m.group(2)), int(m.group(3))
            else:
                if not _month_ok(m.group(1)):
                    continue
                mo, d, y = _MONTHS[m.group(1).lower()], None, int(m.group(2))
            if not _valid_date(y, mo, d):
                continue
            val = f"{y:04d}-{mo:02d}" + (f"-{d:02d}" if d is not None else "")
            found.append((m.start(), m.end(), val))
    found.sort(key=lambda t: (t[0], -(t[1] - t[0])))
    out, last = [], -1
    for s, e, v in found:
        if s >= last:
            out.append((s, e, v))
            last = e
    return out


def _period_at(clean, pos):
    """(end, period) of a billing period after a price, or (pos, None)."""
    starts = [pos]
    m = re.compile(_PER_SEAT).match(clean, pos)
    if m:
        starts.insert(0, m.end())
    for p in starts:
        for rx, fixed in _PERIOD_RES:
            m = rx.match(clean, p)
            if m:
                per = fixed or _PERIOD_WORD[m.group(1).lower()]
                return m.end(), per
    return pos, None


def _money(start, pos, clean, value, currency, plus, approx):
    end, per = _period_at(clean, pos)
    v = {"kind": "money", "value": _tonum(value), "currency": currency}
    if per:
        v["per"] = per
    return _Atom(start, end, v, plus=plus, approx=approx)


def _counted_noun(clean, pos, value, taken):
    """(end, noun) for a counted noun after ``pos``: up to two words between."""
    for i in range(3):
        g = _GAP1_RE.match(clean, pos)
        if not g:
            return None
        m = _WORD_RE.match(clean, g.end())
        if not m or any(taken[m.start():m.end()]):
            return None
        w = m.group(0)
        lw = w.lower()
        if lw in _HALT:
            return None
        if "-" not in w:
            if value == 1:
                if i == 0 and lw not in _NOT_NOUN and lw not in _DETERMINERS:
                    return m.end(), _singularize(lw)
                return None
            if _looks_plural(w):
                return m.end(), _singularize(lw)
        if value == 1:
            return None
        pos = m.end()
    return None


def _extend(clean, tok: _Tok, taken) -> _Atom | None:
    s, e = tok.start, tok.end
    if tok.flag == "a":
        u = _match_unit(clean, e, kinds=("duration",), gaps=("sp",))
        if not u:
            return None
        end, unit = u
        return _Atom(s, end, {"kind": "duration", "value": unit.mult, "unit": unit.canon})

    nothing = _Atom(s, e, None)
    if tok.flag == "multi":
        if _version_ok(tok.raw):
            return _Atom(s, e, {"kind": "version", "value": tok.raw})
        return nothing

    value, flag = tok.value, tok.flag
    approx = bool(_ISH_RE.match(clean, e))
    pos = e

    # currency before the amount
    cur, cstart = None, s
    if flag != "word":
        m = _CUR_BEFORE_RE.search(clean[max(0, s - 4):s])
        if m:
            sym, iso = m.group(1), m.group(2)
            cur = _SYMBOL_CURRENCY[sym] if sym else iso
            cstart = s - (len(m.group(0)))

    scaled = False
    if value is not None and flag in ("int", "num", "dec", "year", "comma", "word"):
        rx = _SCALE_WORD_RE if flag == "word" else _SCALE_DIGIT_RE
        m = rx.match(clean, pos)
        if m and flag != "comma":
            word = next(g for g in m.groups() if g)
            value = value * _SCALES[word.lower()]
            pos, scaled = m.end(), True

    plus = False
    if clean.startswith("+", pos):
        plus, pos = True, pos + 1

    if cur:
        if value is None or flag == "amb3":
            return nothing
        return _money(cstart, pos, clean, value, cur, plus, approx)

    m = _CUR_AFTER_RE.match(clean, pos)
    if m:
        if value is None or flag == "amb3":
            return nothing
        sym, iso, word = m.groups()
        currency = _SYMBOL_CURRENCY[sym] if sym else iso or _WORD_CURRENCY[word.lower()]
        return _money(s, m.end(), clean, value, currency, plus, approx)

    m = _PCT_RE.match(clean, pos)
    if m:
        if value is None or flag == "comma":
            return nothing
        return _Atom(s, m.end(), {"kind": "percent", "value": _tonum(value)}, plus=plus, approx=approx)

    if value is None or flag in ("comma", "amb3"):
        return nothing

    u = _match_unit(clean, pos)
    if u:
        end, unit = u
        if flag == "year" and unit.canon == "s" and end == pos + 1:
            return nothing  # "the 1990s"
        if unit.kind == "duration":
            v = {"kind": "duration", "value": _tonum(value * unit.mult), "unit": unit.canon}
        else:
            v = {"kind": "quantity", "value": _tonum(value), "unit": unit.canon}
        return _Atom(s, end, v, plus=plus, approx=approx)

    if flag == "year" and not plus:
        if _QUARTER_RE.search(clean[max(0, s - 6):s]):
            return nothing  # Q4 2026, H1 2026: not dates
        return _Atom(s, pos, {"kind": "date", "value": tok.raw})

    if flag == "dec" and not scaled and not plus:
        m = _CAP_WORD_RE.search(clean[max(0, s - 40):s])
        if m and m.group(1).lower() not in _CAP_STOP:
            return _Atom(s, pos, {"kind": "version", "value": tok.raw})

    if value == value.to_integral_value() and value >= 0:
        c = _counted_noun(clean, pos, value, taken)
        if c:
            end, noun = c
            return _Atom(s, end, {"kind": "quantity", "value": _tonum(value), "unit": noun}, plus=plus, approx=approx)

    return _Atom(s, pos, None)


def _range_pairs(clean, atoms):
    """Mark both ends of every range as excluded."""
    for a, b in zip(atoms, atoms[1:]):
        between = clean[a.end:b.start]
        before = clean[max(0, a.start - 12):a.start]
        is_range = False
        if re.fullmatch(rf"{_SP}*[-–—]{_SP}*", between):
            is_range = True
        elif re.fullmatch(rf"{_GAP1}(?i:to){_GAP1}", between):
            same = a.value is None or b.value is None or a.value["kind"] == b.value["kind"]
            # "from X to Y" is a change phrase, each side read on its own (claim-forms.md).
            is_range = same and not re.search(rf"(?i)(?<![A-Za-z])from{_SP}*$", before)
        elif re.fullmatch(rf"{_GAP1}(?i:or){_GAP1}", between):
            is_range = a.value is None  # "1 or 2 days"; "€49 a month or €490 a year" is a choice
        elif re.fullmatch(rf"{_GAP1}(?i:and){_GAP1}", between):
            is_range = bool(re.search(rf"(?i)(?<![A-Za-z])between{_SP}+$", before))
        if is_range:
            a.excluded = b.excluded = True


def _clause_start(clean, pos):
    i = pos - 1
    while i >= 0:
        c = clean[i]
        if c in ";:!?•·|":
            return i + 1
        if c in ".,":
            if not (0 < i < len(clean) - 1 and clean[i - 1].isdigit() and clean[i + 1].isdigit()):
                return i + 1
        if c == "\n":
            j = i - 1
            while j >= 0 and clean[j] in " \t ":
                j -= 1
            if j >= 0 and clean[j] == "\n":
                return i + 1
        i -= 1
    return 0


def _negated(segment: str) -> bool:
    return bool(_NEG_RE.search(_NEG_OP_RE.sub(" ", segment)))


def parse_values(text: str) -> list[ValueSpan]:
    """Every quantity, money, percent, date, version and duration ``text`` states plainly."""
    clean, idx, poison = _prepare(text)
    n = len(clean)
    taken = [False] * n
    atoms: list[_Atom] = []

    def free(s, e):
        return not any(taken[s:e])

    def take(s, e):
        for i in range(s, e):
            taken[i] = True

    for s, e, v in _find_dates(clean):
        if free(s, e):
            take(s, e)
            atoms.append(_Atom(s, e, {"kind": "date", "value": v}))
    for rx in (_V_RE, _VER_WORD_RE):
        for m in rx.finditer(clean):
            if free(m.start(), m.end()):
                take(m.start(), m.end())
                atoms.append(_Atom(m.start(), m.end(), {"kind": "version", "value": m.group(1)}))
    for m in _SKIP_RE.finditer(clean):  # 12/03/2026, 24/7, 10:30 read as nothing
        if free(m.start(), m.end()):
            take(m.start(), m.end())

    toks: list[_Tok] = []
    for m in _WORDNUM_RE.finditer(clean):
        if free(m.start(), m.end()):
            toks.append(_Tok(m.start(), m.end(), m.group(0), Decimal(_word_value(m.group(0))), "word"))
    for m in _NUM_RE.finditer(clean):
        if free(m.start(), m.end()):
            val, flag = _classify(m.group(0))
            toks.append(_Tok(m.start(), m.end(), m.group(0), val, flag))
    for m in _A_DUR_RE.finditer(clean):
        if free(m.start(1), m.end(1)):
            toks.append(_Tok(m.start(1), m.end(1), m.group(1), Decimal(1), "a"))
    toks.sort(key=lambda t: (t.start, -(t.end - t.start)))

    last_end = -1
    for tok in toks:
        if tok.start < last_end or not free(tok.start, tok.end):
            continue
        atom = _extend(clean, tok, taken)
        if atom is None:
            continue
        take(atom.start, atom.end)
        atoms.append(atom)
        last_end = atom.end

    atoms.sort(key=lambda a: a.start)
    _range_pairs(clean, atoms)

    out: list[ValueSpan] = []
    for a in atoms:
        if a.value is None or a.excluded or a.approx:
            continue
        kind = a.value["kind"]
        st, en = a.start, a.end
        pre = clean[max(0, st - 60):st]
        if _APPROX_BEFORE_RE.search(pre) or _APPROX_AFTER_RE.match(clean, en):
            continue
        op = "ge" if a.plus else None
        op_rx = _DATE_OP_RE if kind == "date" else _DUR_OP_RE if kind == "duration" else _GEN_OP_RE
        table = _DATE_OPS if kind == "date" else _DUR_OPS if kind == "duration" else _GEN_OPS
        m = op_rx.search(pre)
        if m:
            op = op or table[re.sub(r"\s+", " ", m.group(1).lower())]
            st = st - (len(pre) - m.start())
        if kind != "date":
            m = _AFTER_OP_RE.match(clean, en)
            if m:
                op = op or _AFTER_OPS[re.sub(r"\s+", " ", m.group(1).lower())]
                en = m.end()
        if _negated(clean[_clause_start(clean, st):st]):
            continue
        if any(poison[i] for i in range(max(0, st - 1), min(n, en + 1))):
            continue
        value = dict(a.value)
        if op and op != "eq":
            value["op"] = op
        out.append(ValueSpan(idx[st], idx[en], value))
    return out


# --------------------------------------------------------------------------------------
# Sentences and negation
# --------------------------------------------------------------------------------------


def split_sentences(text: str) -> list[str]:
    """Sentences end at . ! ? before a space, or at a blank line; not after abbreviations."""
    out = []
    for block in re.split(r"\n[ \t]*\n", text):
        block = block.replace("\n", " ")
        start = 0
        for m in _SENT_END_RE.finditer(block):
            if m.group(0).rstrip("\"'”’)]") == ".":
                tm = re.search(r"(\S+)$", block[start:m.start() + 1])
                written = tm.group(1).lstrip("([{\"'“‘") if tm else ""
                token = written.lower()
                rest = block[m.end():].lstrip()
                if token in _ABBREVIATIONS or re.fullmatch(r"[A-Z]\.", written):
                    continue
                if token in _ABBREV_BEFORE_NUMBER and rest[:1].isdigit():
                    continue
            piece = block[start:m.end()].strip()
            if piece:
                out.append(piece)
            start = m.end()
        tail = block[start:].strip()
        if tail:
            out.append(tail)
    return out


def has_negation(text: str) -> bool:
    """not, no, never, none, nobody, nothing, neither, nor, without, cannot, or a word in n't.

    Comparison words such as "no more than" are part of their value, not a negation.
    """
    return _negated(text)


# --------------------------------------------------------------------------------------
# Comparing
# --------------------------------------------------------------------------------------


def _phys_canon(unit: str):
    if unit in _PHYS_SYMBOLS:
        return _PHYS_SYMBOLS[unit]
    return _PHYS_WORD_MAP.get(unit.lower())


def _dur_canon(unit: str):
    if unit in _DUR_SYMBOLS:
        return _DUR_SYMBOLS[unit]
    return _DUR_WORD_MAP.get(unit.lower())


def normalize(v: dict) -> dict:
    """Comparable canonical form; drops ``op``."""
    kind = v.get("kind")
    out = {k: val for k, val in v.items() if k != "op"}
    if kind == "quantity":
        unit = str(v["unit"])
        canon = _phys_canon(unit)
        if canon:
            base, factor = _PHYS[canon]
            out["value"] = _tonum(_dec(v["value"]) * factor)
            out["unit"] = base
        else:
            out["value"] = _tonum(_dec(v["value"]))
            out["unit"] = _noun_key(unit)
    elif kind == "duration":
        canon, mult = _dur_canon(str(v["unit"])) or (str(v["unit"]), 1)
        base, factor = _DUR.get(canon, (canon, Decimal(1)))
        out["value"] = _tonum(_dec(v["value"]) * mult * factor)
        out["unit"] = base
    elif kind == "money":
        out["value"] = _tonum(_dec(v["value"]))
        out["currency"] = str(v["currency"]).upper()
    elif kind == "percent":
        out["value"] = _tonum(_dec(v["value"]))
    return out


def _num_eq(a, b) -> bool:
    return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)


def _same_dimension(a: dict, b: dict) -> bool:
    kind = a["kind"]
    if kind in ("quantity", "duration"):
        return a["unit"] == b["unit"]
    if kind == "money":
        return a["currency"] == b["currency"] and a.get("per") == b.get("per")
    return True


def values_equal(a: dict, b: dict) -> bool:
    if not a or not b or a.get("kind") != b.get("kind"):
        return False
    kind = a["kind"]
    try:
        na, nb = normalize(a), normalize(b)
    except (KeyError, TypeError, ValueError, ArithmeticError):
        return False
    if kind in ("quantity", "duration", "money", "percent"):
        return _same_dimension(na, nb) and _num_eq(na["value"], nb["value"])
    if kind == "name":
        return " ".join(str(na["value"]).split()).casefold() == " ".join(str(nb["value"]).split()).casefold()
    return na.get("value") == nb.get("value")


def _date_range(s: str):
    parts = [int(p) for p in s.split("-")]
    y = parts[0]
    if len(parts) == 1:
        return (y, 1, 1), (y, 12, 31)
    m = parts[1]
    if len(parts) == 2:
        return (y, m, 1), (y, m, calendar.monthrange(y, m)[1])
    return (y, m, parts[2]), (y, m, parts[2])


def _sign(x) -> int:
    return (x > 0) - (x < 0)


def compare(a: dict, b: dict) -> int | None:
    """-1, 0 or 1 after normalisation; None when the two do not compare."""
    if not a or not b or a.get("kind") != b.get("kind"):
        return None
    kind = a["kind"]
    if kind in ("name", "boolean"):
        return None
    try:
        na, nb = normalize(a), normalize(b)
    except (KeyError, TypeError, ValueError, ArithmeticError):
        return None
    if kind in ("quantity", "duration", "money", "percent"):
        if not _same_dimension(na, nb):
            return None
        if _num_eq(na["value"], nb["value"]):
            return 0
        return _sign(float(na["value"]) - float(nb["value"]))
    if kind == "date":
        sa, sb = str(na["value"]), str(nb["value"])
        if sa == sb:
            return 0
        (a0, a1), (b0, b1) = _date_range(sa), _date_range(sb)
        if sa.count("-") == sb.count("-"):
            return _sign((a0 > b0) - (a0 < b0))
        if a1 < b0:
            return -1
        if a0 > b1:
            return 1
        return None
    if kind == "version":
        sa, sb = str(na["value"]), str(nb["value"])
        if sa == sb:
            return 0
        ta, tb = [int(p) for p in sa.split(".")], [int(p) for p in sb.split(".")]
        width = max(len(ta), len(tb))
        ta, tb = ta + [0] * (width - len(ta)), tb + [0] * (width - len(tb))
        if ta == tb:
            return None  # 2.3 is not 2.3.0, nor above or below it
        return -1 if ta < tb else 1
    return None


def satisfies(exact: dict, claimed: dict) -> bool:
    """Whether ``claimed``'s op (default eq) holds for ``exact``."""
    op = claimed.get("op", "eq")
    if op == "eq":
        return values_equal(exact, claimed)
    c = compare(exact, claimed)
    if c is None:
        return False
    return {"gt": c > 0, "ge": c >= 0, "lt": c < 0, "le": c <= 0}.get(op, False)


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------


def _fmt(x, allow_three=False) -> str:
    d = _dec(x)
    neg = d < 0
    d = abs(d)
    s = format(d.normalize(), "f")
    whole, _, frac = s.partition(".")
    out = f"{int(whole):,}"
    if frac:
        if len(frac) == 3 and not allow_three:
            frac += "0"  # 1.500 would read as nothing
        out += "." + frac
    return ("-" if neg else "") + out


_RENDER_OPS = {"gt": "over ", "lt": "under ", "le": "up to ", "ge": "at least "}
_RENDER_DATE_OPS = {"gt": "after ", "lt": "before ", "le": "by ", "ge": "no earlier than "}


def render(v: dict) -> str:
    """Canonical text that ``parse_values`` reads back as ``v``."""
    kind = v.get("kind")
    op = v.get("op", "eq")
    if kind == "money":
        cur = str(v["currency"]).upper()
        n = _fmt(v["value"])
        text = f"{_RENDER_SYMBOL[cur]}{n}" if cur in _RENDER_SYMBOL else f"{n} {cur}"
        if v.get("per"):
            text += f" per {v['per']}"
    elif kind == "quantity":
        unit = str(v["unit"])
        canon = _phys_canon(unit)
        n = _fmt(v["value"])
        if canon:
            text = f"{n} {canon}"
        else:
            noun = _singularize(unit)
            text = f"{n} {noun if _dec(v['value']) == 1 else _pluralize(noun)}"
    elif kind == "percent":
        text = f"{_fmt(v['value'], allow_three=True)}%"
    elif kind == "date":
        parts = str(v["value"]).split("-")
        if len(parts) == 3:
            text = f"{int(parts[2])} {_MONTH_NAMES[int(parts[1])]} {parts[0]}"
        elif len(parts) == 2:
            text = f"{_MONTH_NAMES[int(parts[1])]} {parts[0]}"
        else:
            text = parts[0]
        return _RENDER_DATE_OPS.get(op, "") + text
    elif kind == "version":
        text = f"v{v['value']}"
    elif kind == "duration":
        canon, mult = _dur_canon(str(v["unit"])) or (str(v["unit"]), 1)
        value = _dec(v["value"]) * mult
        sing, plur = _RENDER_DUR.get(canon, (canon, canon))
        text = f"{_fmt(value)} {sing if value == 1 else plur}"
    elif kind == "boolean":
        return "yes" if v.get("value") else "no"
    else:
        return str(v.get("value", ""))
    return _RENDER_OPS.get(op, "") + text
