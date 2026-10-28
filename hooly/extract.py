"""Value extraction: find typed values in a sentence (the grammar of contracts/value-forms.md).

Returns every candidate value with its span. Deciding which values are claims, and about
which subject and attribute, is the job of hooly/reader.py. Booleans have no value token of
their own; the reader finds them through attribute words, using `negated()` for polarity.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september",
     "october", "november", "december"], 1)}
MONTH_ABBR = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9,
              "sept": 9, "oct": 10, "nov": 11, "dec": 12}
_MONTH = r"(?P<mon>January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec)\.?"

_UNITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
          "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
          "nineteen"]
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
         "eighty": 80, "ninety": 90}
WORD_NUMBERS = {w: i for i, w in enumerate(_UNITS)}
for t, v in _TENS.items():
    WORD_NUMBERS[t] = v
    for i in range(1, 10):
        WORD_NUMBERS[f"{t}-{_UNITS[i]}"] = v + i
WORD_NUMBERS["a dozen"] = 12
WORD_NUMBERS["a hundred"] = 100
for i in range(1, 10):
    WORD_NUMBERS[f"{_UNITS[i]} hundred"] = i * 100
_WORDNUM = "(?:" + "|".join(sorted((re.escape(w) for w in WORD_NUMBERS), key=len, reverse=True)) + ")"

# digits with thousands commas or a decimal point; validated by _number()
_DIG = r"\d[\d,.]*\d|\d"
_SCALE = r"(?:\s?(?P<scale>million|billion|thousand|bn|M|k)(?![A-Za-z]))?"
_SCALES = {"thousand": 10**3, "k": 10**3, "million": 10**6, "M": 10**6, "billion": 10**9, "bn": 10**9}

NEGATIONS = re.compile(r"\b(not|no|never|none|nobody|nothing|neither|nor|without|cannot)\b|\w+n't\b", re.I)
_CLAUSE_END = re.compile(r"[.,;:!?•·|]")
_APPROX = re.compile(r"(about|around|approximately|approx\.?|roughly|nearly|almost|circa|ca\.|some|estimated|est\.|close to|more or less|~|≈|±)\s*$", re.I)
_OPS_BEFORE = [
    (r"no more than|not more than|up to|at most|maximum|max", "le"),
    (r"no less than|not less than|no fewer than|at least|minimum", "ge"),
    (r"more than|greater than|higher than|in excess of|upwards of|exceeding|over|above", "gt"),
    (r"less than|fewer than|lower than|under|below", "lt"),
]
_DUR_OPS = [(r"within", "le")] + _OPS_BEFORE
_OPS_AFTER = [(r"or more|or higher|or above|or longer|or over", "ge"),
              (r"or less|or fewer|or under|or below|or lower|or shorter", "le")]
_DATE_OPS = [(r"no later than|by", "le"), (r"no earlier than", "ge"), (r"before|prior to", "lt"), (r"after", "gt")]


@dataclass
class Found:
    value: dict
    start: int
    end: int

    @property
    def kind(self):
        return self.value["kind"]


def _number(s: str, decimal_comma: bool = False, percent: bool = False):
    """Parse digits per the value grammar; None when ambiguous (1,5 / 1.000.000 / 1.500)."""
    if decimal_comma and re.fullmatch(r"\d+,\d{1,2}", s):
        return float(s.replace(",", "."))
    if re.fullmatch(r"\d{1,3}(,\d{3})+(\.\d+)?", s) or re.fullmatch(r"\d+", s):
        return float(s.replace(",", ""))
    m = re.fullmatch(r"\d+\.(\d+)", s)
    if m:
        if len(m.group(1)) == 3 and not percent:
            return None
        return float(s)
    return None


def _num_value(x: float):
    return int(x) if float(x).is_integer() else x


def negated(text: str, pos: int) -> bool:
    """Is there a negation earlier in the clause that contains position `pos`?"""
    start = 0
    for m in _CLAUSE_END.finditer(text, 0, pos):
        start = m.end()
    return bool(NEGATIONS.search(text[start:pos]))


_YES_NO = re.compile(r"\s*(yes|no)\b", re.I)


def negated_at(text: str, pos: int) -> bool:
    """Polarity of a yes/no fact whose words sit at `pos`: a negation anywhere in that clause
    ("is not discontinued", "Series A not yet raised"), or a bare answer after a colon
    ("Desktop mode: no", "Audited: yes")."""
    pos = min(max(0, pos), len(text))
    start = 0
    for m in _CLAUSE_END.finditer(text, 0, pos):
        if not _decimal_point(text, m.start()):
            start = m.end()
    end, colon = len(text), None
    for m in _CLAUSE_END.finditer(text, pos):
        if not _decimal_point(text, m.start()):
            end, colon = m.start(), m.end() if text[m.start()] == ":" else None
            break
    if pos >= len(text):  # no cue word found: the whole sentence decides
        return bool(NEGATIONS.search(text))
    if colon is not None:  # "Desktop mode: no", "AI editing: not live", "Hub status: ... does not serve"
        ans = _YES_NO.match(text, colon)
        if ans:
            return ans.group(1).lower() == "no"
        nxt = _CLAUSE_END.search(text, colon)
        end = nxt.start() if nxt else len(text)
    return bool(NEGATIONS.search(text[start:end]))


def _decimal_point(text: str, i: int) -> bool:
    return text[i] in ".," and 0 < i < len(text) - 1 and text[i - 1].isdigit() and text[i + 1].isdigit()


def _op_before(text: str, start: int, table) -> str | None:
    head = text[max(0, start - 25):start].lower().rstrip()
    for pat, op in table:
        if re.search(r"(?:^|\W)(?:" + pat + r")$", head):
            return op
    return None


def _op_after(text: str, end: int) -> str | None:
    tail = text[end:end + 15].lower()
    for pat, op in _OPS_AFTER:
        if re.match(r"\s+(?:" + pat + r")\b", tail):
            return op
    return None


class Extractor:
    def __init__(self, nouns=(), names=(), protected=(), units=None):
        """nouns: known counted nouns (singular); units: nouns actually seen as counted units
        (stricter, used to tell "4.1 stars" from a version); names: name spellings to find as
        name values; protected: names whose digits are part of the name, not values."""
        self.nouns = {n.lower() for n in nouns}
        self.units = {n.lower() for n in units} if units is not None else self.nouns
        self._protected = [re.compile(r"(?<![\w])" + r"\s*".join(map(re.escape, n.split())) + r"(?!\w|\.\d)", re.I)
                           for n in sorted({n for n in protected if re.search(r"\d", n)}, key=len, reverse=True)]
        self.names = sorted({n for n in names if n.strip()}, key=len, reverse=True)
        self._name_re = [(n, re.compile(r"(?<![\w])" + r"\s*".join(map(re.escape, n.split())) + r"(?![\w])", re.I))
                         for n in self.names]

    # ---- one pass per kind; later passes skip spans already taken ----
    def find(self, text: str) -> list[Found]:
        taken: list[tuple[int, int]] = []
        out: list[Found] = []
        for rx in self._protected:  # "Hooly Android 17", "Pixel 9", "Hooly I/O 2026": not values
            for m in rx.finditer(text):
                if all(m.end() <= s or m.start() >= e for s, e in taken):
                    taken.append((m.start(), m.end()))

        def free(a, b):
            return all(b <= s or a >= e for s, e in taken)

        def add(value, a, b, op=None, op_table=None):
            if not free(a, b):
                return
            if _APPROX.search(text[max(0, a - 22):a]):
                taken.append((a, b))  # an approximation states nothing
                return
            op = op or (_op_before(text, a, op_table) if op_table else None) or _op_after(text, b)
            if op and op != "eq":
                value = {**value, "op": op}
            out.append(Found(value, a, b))
            taken.append((a, b))

        for f in (self._dates, self._money, self._percent, self._versions, self._durations,
                  self._quantities, self._counts, self._names):
            f(text, add, free)
        return sorted(out, key=lambda x: x.start)

    def _dates(self, text, add, free):
        pats = [
            r"\b(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})\b",
            r"\b(?:the\s+)?(?P<d>\d{1,2})(?:st|nd|rd|th)?(?:\s+of)?\s*" + _MONTH + r",?\s+(?P<y>\d{4})\b",
            r"\b" + _MONTH + r"\s+(?P<d>\d{1,2})(?:st|nd|rd|th)?,?\s+(?P<y>\d{4})\b",
            r"\b" + _MONTH + r",?\s+(?P<y>\d{4})\b",
            r"\b(?P<y>\d{4})-(?P<m>\d{2})\b",
            r"\b(?:in|since|from|until|till|by|of|during|before|after|founded|year)\s(?P<y>(?:19|20)\d{2})\b",
        ]
        for i, p in enumerate(pats):
            bare_year = i == len(pats) - 1
            for m in re.finditer(p, text):
                g = m.groupdict()
                y = int(g["y"])
                mon = g.get("mon")
                mo = (MONTHS.get(mon.lower()) or MONTH_ABBR.get(mon.lower().rstrip("."))) if mon else (int(g["m"]) if g.get("m") else None)
                if mo is not None and not 1 <= mo <= 12:
                    continue
                d = int(g["d"]) if g.get("d") else None
                if d is not None and not 1 <= d <= 31:
                    continue
                val = f"{y:04d}" + (f"-{mo:02d}" if mo else "") + (f"-{d:02d}" if d else "")
                a = m.start("y") if bare_year else m.start()
                add({"kind": "date", "value": val}, a, m.end(), op_table=_DATE_OPS)

    _CUR = {"€": "EUR", "$": "USD", "£": "GBP", "US$": "USD", "euro": "EUR", "euros": "EUR",
            "dollar": "USD", "dollars": "USD", "lei": "RON"}
    _ISO = "EUR|USD|GBP|CHF|RON|PLN|CZK|HUF|SEK|NOK|DKK|JPY|CNY|INR|CAD|AUD"
    _PER = (r"(?:\s*(?:per|/)\s*(?!month|year|week|day|annum|mo\b|yr\b|wk\b)(?:user|seat|terabyte|gigabyte|tb|gb|device|member|employee)s?\b)?"
            r"\s*(?P<per>/\s?(?:month|mo|year|yr|week|wk|day)\b|per\s+(?:month|year|week|day|annum)\b|"
            r"(?:an?|each|every)\s+(?:month|year|week|day)\b|monthly\b|yearly\b|annually\b|weekly\b|daily\b)?")

    def _money(self, text, add, free):
        pre = r"(?P<cur>US\$|€|\$|£|\b(?:" + self._ISO + r"))\s?(?P<amt>" + _DIG + r")" + _SCALE + self._PER
        post = r"(?<![\w.,])(?P<amt>" + _DIG + r")" + _SCALE + r"\s?(?P<cur>\b(?:" + self._ISO + r")\b|euros?\b|dollars?\b|lei\b)" + self._PER
        for p in (pre, post):
            for m in re.finditer(p, text):
                amt = _number(m.group("amt"), decimal_comma=True)
                if amt is None:
                    continue
                if m.group("scale"):
                    amt *= _SCALES[m.group("scale")]
                cur = self._CUR.get(m.group("cur").lower() if m.group("cur").lower() in self._CUR else m.group("cur"), m.group("cur").upper())
                val = {"kind": "money", "value": _num_value(amt), "currency": cur}
                per = (m.group("per") or "").lower()
                for word, p2 in (("month", "month"), ("mo", "month"), ("annum", "year"), ("annually", "year"),
                                 ("year", "year"), ("yr", "year"), ("week", "week"), ("wk", "week"), ("day", "day"), ("daily", "day")):
                    if word in per:
                        val["per"] = p2
                        break
                add(val, m.start(), m.end(), op_table=_OPS_BEFORE)

    def _percent(self, text, add, free):
        for m in re.finditer(r"(?<![\w.,])(?P<amt>" + _DIG + r")\s?(?:%|percent\b|per cent\b)", text, re.I):
            x = _number(m.group("amt"), percent=True)
            if x is not None:
                add({"kind": "percent", "value": _num_value(x)}, m.start(), m.end(), op_table=_OPS_BEFORE)
        for m in re.finditer(r"\b(?P<w>" + _WORDNUM + r")\s+(?:percent|per cent)\b", text, re.I):
            add({"kind": "percent", "value": WORD_NUMBERS[m.group("w").lower()]}, m.start(), m.end(), op_table=_OPS_BEFORE)

    def _versions(self, text, add, free):
        pats = [(r"\b(?:v|version\s|ver\.\s?)(?P<v>\d+(?:\.\d+)*)\b", re.I, False),
                (r"(?<![\w.])(?P<v>\d+\.\d+\.\d+(?:\.\d+)*)(?!\w|\.\d)", 0, True),
                (r"\b[A-Z][A-Za-z0-9-]{1,30}\s(?P<v>\d+\.\d+)(?!\w|\.\d)", 0, True)]
        for p, flags, from_group in pats:
            for m in re.finditer(p, text, flags):
                nxt = re.match(r"\s+([A-Za-z]+)", text[m.end():])
                if from_group and nxt and _singular(nxt.group(1).lower()) in self.units:
                    continue  # "4.1 stars" is a count, not a version
                add({"kind": "version", "value": m.group("v")}, m.start("v") if from_group else m.start(), m.end())

    _DUR = {"ms": "ms", "millisecond": "ms", "s": "s", "sec": "s", "second": "s", "min": "min", "minute": "min",
            "h": "h", "hr": "h", "hour": "h", "d": "day", "day": "day", "wk": "week", "week": "week",
            "fortnight": "week", "mo": "month", "mth": "month", "month": "month", "yr": "year", "year": "year",
            "decade": "year"}

    def _durations(self, text, add, free):
        unit = r"(?P<u>milliseconds?|ms|seconds?|secs?|s|minutes?|mins?|hours?|hrs?|h|days?|d|weeks?|wks?|fortnights?|months?|mths?|mo|years?|yrs?|decades?)(?![A-Za-z])"
        for m in re.finditer(r"(?<![\w.,])(?P<n>" + _DIG + r"|" + _WORDNUM + r")(?:\s|-)?" + unit, text, re.I):
            raw_u = m.group("u")
            if len(raw_u) == 1 and raw_u != raw_u.lower():
                continue
            n = m.group("n")
            x = WORD_NUMBERS.get(n.lower()) if not n[0].isdigit() else _number(n)
            if x is None:
                continue
            u = raw_u.lower()
            base = u[:-1] if u.endswith("s") and u not in ("ms", "s") and u[:-1] in self._DUR else u
            if base not in self._DUR:
                continue
            scale = 2 if base == "fortnight" else 10 if base == "decade" else 1
            add({"kind": "duration", "value": _num_value(x * scale), "unit": self._DUR[base]}, m.start(), m.end(), op_table=_DUR_OPS)
        for m in re.finditer(r"\b(?:under|within|in|for|over|than|at least|at most|to|only|just)\s+(?P<a>an?)\s+(?P<u>millisecond|second|minute|hour|day|week|month|year)\b", text, re.I):
            add({"kind": "duration", "value": 1, "unit": self._DUR[m.group("u").lower()]}, m.start("a"), m.end(), op_table=_DUR_OPS)

    _QUNITS = {"B": "B", "KB": "KB", "kB": "KB", "kb": "KB", "MB": "MB", "mb": "MB", "GB": "GB", "gb": "GB",
               "TB": "TB", "tb": "TB", "mm": "mm", "cm": "cm", "m": "m", "km": "km", "in": "in", "g": "g",
               "kg": "kg", "mAh": "mAh", "Hz": "Hz", "kHz": "kHz", "MHz": "MHz", "GHz": "GHz", "mW": "mW",
               "W": "W", "kW": "kW", '"': "in", "″": "in"}
    _QWORDS = {"byte": "B", "kilobyte": "KB", "megabyte": "MB", "gigabyte": "GB", "gig": "GB", "terabyte": "TB",
               "millimetre": "mm", "millimeter": "mm", "centimetre": "cm", "centimeter": "cm", "metre": "m",
               "meter": "m", "kilometre": "km", "kilometer": "km", "inch": "in", "gram": "g", "gramme": "g",
               "kilogram": "kg", "kilo": "kg", "hertz": "Hz", "kilohertz": "kHz", "megahertz": "MHz",
               "gigahertz": "GHz", "milliwatt": "mW", "watt": "W", "kilowatt": "kW"}

    def _quantities(self, text, add, free):
        sym = r"(?P<u>KB|kB|kb|MB|mb|GB|gb|TB|tb|mAh|kHz|MHz|GHz|Hz|mW|kW|mm|cm|km|kg|B|W|m|g|in)(?![A-Za-z])"
        word = r"(?P<w>bytes?|kilobytes?|megabytes?|gigabytes?|gigs?|terabytes?|millimet(?:re|er)s?|centimet(?:re|er)s?|kilomet(?:re|er)s?|met(?:re|er)s?|inch(?:es)?|grams?|grammes?|kilograms?|kilos?|hertz|kilohertz|megahertz|gigahertz|milliwatts?|kilowatts?|watts?)(?![A-Za-z])"
        for m in re.finditer(r"(?<![\w.,])(?P<n>" + _DIG + r")(?:\s|-)?(?:" + sym + r"|" + word + r"|(?P<q>[\"″]))", text):
            x = _number(m.group("n"))
            if x is None:
                continue
            if m.group("u"):
                u = self._QUNITS[m.group("u")]
                if m.group("u") == "in" and not re.match(r"\s", text[m.start("u") - 1:m.start("u")] or "x") is None:
                    pass
                if m.group("u") == "in":  # "in" right after a number is too ambiguous ("3 in 4")
                    continue
            elif m.group("w"):
                w = m.group("w").lower()
                w = w[:-2] if w.endswith("ches") else w[:-1] if w.endswith("s") and w[:-1] in self._QWORDS else w
                u = self._QWORDS.get(w)
                if not u:
                    continue
            else:
                u = "in"
            add({"kind": "quantity", "value": _num_value(x), "unit": u}, m.start(), m.end(), op_table=_OPS_BEFORE)

    _STOP = {"of", "the", "a", "an", "and", "or", "to", "in", "on", "at", "for", "per", "from", "by", "with",
             "more", "than", "our", "its", "their", "his", "her", "new", "total", "is", "are", "was", "were",
             "has", "have", "it", "that", "this", "which", "who", "as", "so", "now", "each", "every", "times",
             "within", "since", "until", "across", "over", "under", "about", "around", "into", "through",
             "after", "before", "during", "via", "up", "down", "out", "than", "while", "when", "where"}

    def _counts(self, text, add, free):
        for m in re.finditer(r"(?<![\w.,€$£])(?P<n>" + _DIG + r"|" + _WORDNUM + r")" + _SCALE + r"(?:\+)?(?=\s+[A-Za-z])", text, re.I):
            n = m.group("n")
            if n[0].isdigit():
                if re.fullmatch(r"(19|20)\d{2}", n) and not m.group("scale"):
                    continue  # a plain four-digit 1900-2099 is a year
                x = _number(n)
            else:
                x = WORD_NUMBERS.get(n.lower())
            if x is None:
                continue
            if m.group("scale"):
                x *= _SCALES[m.group("scale")]
            window = re.split(r"\d", text[m.end():m.end() + 60], maxsplit=1)[0]
            words = list(re.finditer(r"[A-Za-z][A-Za-z-]*", window))[:3]
            noun, end = None, None
            run = []  # the words right after the number, up to a function word or punctuation
            for w in words:
                gap = window[(run[-1].end() if run else 0):w.start()]
                if w.group().lower() in self._STOP or re.search(r"[,;:.!?()]", gap):
                    break
                run.append(w)
            sing = lambda w: _singular(w.group().lower())
            in_units = [w for w in run if sing(w) in self.units]
            in_nouns = [w for w in run if sing(w) in self.nouns]
            pick = (in_units[-1] if in_units else in_nouns[-1] if in_nouns else run[0] if run else
                    next((w for w in words if sing(w) in self.units or sing(w) in self.nouns), None))
            if pick is not None:  # the head of "4 data centre sites" is "sites"
                noun, end = _singular(pick.group().lower()), m.end() + pick.end()
            if noun is None:
                continue
            op = "ge" if text[m.end():m.end() + 1] == "+" else None
            add({"kind": "quantity", "value": _num_value(x), "unit": noun}, m.start(), end, op=op, op_table=_OPS_BEFORE)

    def _names(self, text, add, free):
        for n, rx in self._name_re:
            for m in rx.finditer(text):
                if free(m.start(), m.end()):
                    add({"kind": "name", "value": n}, m.start(), m.end())


def _singular(noun: str) -> str:
    if noun == "people":
        return "person"
    if noun.endswith("ies") and len(noun) > 4:
        return noun[:-3] + "y"
    if noun.endswith(("ches", "shes", "sses", "xes")):
        return noun[:-2]
    if noun.endswith("s") and not noun.endswith("ss") and len(noun) > 3:
        return noun[:-1]
    return noun
