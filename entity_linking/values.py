"""Typed literal annotation, without subject attribution or truth inference.

Rules read text only. Entity spans are protected. Ambiguous ranges and numeric
spellings are reported, not silently interpreted as exact scalar values.
"""

import calendar
from collections import Counter
from datetime import date
from decimal import Decimal, localcontext
import json
import re


ONES = dict(zip("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split(), range(20)))
TENS = dict(zip("twenty thirty forty fifty sixty seventy eighty ninety".split(), range(20, 100, 10)))
WORD = "(?:" + "|".join([*ONES, *TENS]) + ")"
SMALL = "(?:" + "|".join(TENS) + ")(?:[ -]+" + "(?:" + "|".join(ONES) + "))?|(?:" + "|".join(ONES) + ")"
WORD_NUMBER = rf"(?:a dozen|a hundred|(?:{SMALL})[ -]+hundred(?:[ -]+(?:and[ -]+)?(?:{SMALL}))?|{SMALL})(?![A-Za-z])"
DIGITS = r"[+-]?\d+(?:[.,]\d+)*"
NUMBER = rf"(?:{DIGITS}|{WORD_NUMBER})"
SCALE_WORD = r"(?:thousand|million|billion|bn|(?-i:M|k))"
SCALE = rf"(?:\s*{SCALE_WORD}(?![A-Za-z0-9]))?"
AMOUNT = NUMBER + SCALE
NUMBERS = re.compile(rf"(?<![\w.,+-])(?P<number>{NUMBER})(?P<scale>\s*{SCALE_WORD}(?![A-Za-z0-9]))?(?!\d|[.,]\d)", re.I)
MULTIPLIERS = {"thousand": 1000, "million": 1000000, "billion": 1000000000, "bn": 1000000000, "M": 1000000, "k": 1000}
CURRENCIES = {"€": "EUR", "$": "USD", "US$": "USD", "£": "GBP", "euro": "EUR", "euros": "EUR",
              "dollar": "USD", "dollars": "USD", "lei": "RON"}
ISO = "EUR USD GBP CHF RON PLN CZK HUF SEK NOK DKK JPY CNY INR CAD AUD".split()
CURRENCY = r"(?:US\$|€|\$|£|" + "|".join(ISO) + r"|euros?|dollars?|lei)"
MONEY = [re.compile(rf"(?<!\w)(?P<currency>{CURRENCY})\s*(?P<amount>{AMOUNT})(?!\d|[.,]\d)", re.I),
         re.compile(rf"(?<![\w.,])(?P<amount>{AMOUNT})\s*(?P<currency>{CURRENCY})(?!\w)", re.I)]
PERIOD = re.compile(r"\s*(?:(?:per\s+(?P<basis>user|seat|terabyte|gigabyte|TB|GB)|/(?P<seat>seat))\s*)?(?:(?:/|per\s+|a\s+|each\s+)(?P<period>day|week|month|mo|year|yr)|(?P<adverb>daily|weekly|monthly|annually|yearly)|per\s+annum)(?![A-Za-z])", re.I)
PERIODS = {"mo": "month", "yr": "year", "daily": "day", "weekly": "week", "monthly": "month", "annually": "year", "yearly": "year"}
MONTHS = {}
for index in range(1, 13):
    MONTHS[calendar.month_name[index].lower()] = index
    MONTHS[calendar.month_abbr[index].lower()] = index
MONTHS["sept"] = 9
MONTH = "(?:" + "|".join(sorted(MONTHS, key=len, reverse=True)) + r")\.?"
DATES = [
    (re.compile(r"(?<![\w\d-])(?P<year>\d{4})-(?P<month>\d{2})(?:-(?P<day>\d{2}))?(?![\d-])"), "iso_date"),
    (re.compile(rf"(?<!\w)(?:the\s+)?(?P<day>\d{{1,2}})(?:st|nd|rd|th)?(?:\s+of)?\s*(?P<month>{MONTH})\s+(?P<year>\d{{4}})(?!\d)", re.I), "day_month_year"),
    (re.compile(rf"\b(?P<month>{MONTH})\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?[,]?\s+(?P<year>\d{{4}})(?!\d)", re.I), "month_day_year"),
    (re.compile(rf"\b(?P<month>{MONTH})\s+(?P<year>\d{{4}})(?!\d)", re.I), "month_year"),
]
VERSIONS = [re.compile(r"(?<!\w)(?:v|version\s+|ver\.\s*)(?P<value>\d+(?:\.\d+){0,3})(?![\w.]\d)", re.I),
            re.compile(r"(?<![\w.])(?P<value>\d+(?:\.\d+){2,3})(?![\w.]\d)")]
PERCENT = re.compile(r"\s*(?:%|percent\b|per\s+cent\b)", re.I)
PHYSICAL = {symbol: symbol for symbol in "B KB MB GB TB mm cm m km in g kg mAh Hz kHz MHz GHz mW W kW".split()}
for canonical, words in {
    "B": "byte", "KB": "kilobyte", "MB": "megabyte", "GB": "gigabyte gig", "TB": "terabyte",
    "mm": "millimetre millimeter", "cm": "centimetre centimeter", "m": "metre meter", "km": "kilometre kilometer", "in": "inch",
    "g": "gram gramme", "kg": "kilogram kilo", "Hz": "hertz", "kHz": "kilohertz", "MHz": "megahertz", "GHz": "gigahertz",
    "mW": "milliwatt", "W": "watt", "kW": "kilowatt",
}.items():
    for word in words.split():
        PHYSICAL[word] = canonical
        PHYSICAL[word + "s"] = canonical
PHYSICAL.update({"kB": "KB", "kb": "KB", "mb": "MB", "gb": "GB", "tb": "TB", "inches": "in"})
DURATIONS = {}
for canonical, words in {"ms": "ms millisecond milliseconds", "s": "s sec secs second seconds",
                         "min": "min mins minute minutes", "h": "h hr hrs hour hours", "day": "d day days",
                         "week": "week weeks wk wks", "month": "month months mo mth", "year": "year years yr yrs"}.items():
    DURATIONS.update({word: canonical for word in words.split()})
ADJECTIVES = set("paying active concurrent supported major included new different total simultaneous monthly daily open available registered unique full output input of our its the".split())
ADJECTIVES.update({"data", "centre", "center", "storage"})
STOP = set("a an and or to at in on for with from by than as is are was were has have had can will would should must does do did remains remain ships ship serves serve per after before since until then now today ago about roughly around approximately of our its the into out through over under up down more less within".split())
STOP.update("bring brings brought keep keeps tell tells speak speaks cost costs went read reads shipped landed land took take hold holds understand understands cover covers run runs answer answers save saves start starts send sends reached reach reviewed review off".split())
STOP.update(MONTHS)
STOP.update({"them", "us", "it", "him", "her", "they", "those", "these", "certified"})
COUNT_NOUNS = set("user customer language engineer attendee registration session speaker sponsor seat request company partner outage token parameter format region country device model file service function connector connection phone star sensor megapixel point office desk test step person member director role recruiter page guide ticket subscriber webinar incident founder command workshop template editor app application feature event sdk tool payment team tonne pair benchmark plan addition report employee project version day week month year byte".split())
COUNT_NOUNS.discard("benchmark")
OPS = {"gt": ["more than", "greater than", "higher than", "in excess of", "over", "above", "exceeding", "after"],
       "lt": ["less than", "fewer than", "lower than", "under", "below", "before", "prior to"],
       "le": ["no more than", "not more than", "no later than", "at most", "up to", "within", "by"],
       "ge": ["no less than", "not less than", "no fewer than", "no earlier than", "at least"]}
OP_PHRASES = sorted([(phrase, op) for op, phrases in OPS.items() for phrase in phrases], key=lambda item: -len(item[0]))
APPROX = re.compile(r"(?i)(?:\b(?:about|around|approximately|approx\.?|roughly|nearly|almost|circa|estimated|close to)|[~≈±])\s*$")
NEGATION = re.compile(r"(?i)\b(?:not|no|never|without|cannot)\b|\b\w+n['’]t\b")


def exact_json(value):
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("Nonfinite decimal")
        return str(value)
    if isinstance(value, dict):
        return "{" + ",".join(json.dumps(key, ensure_ascii=False) + ":" + exact_json(item) for key, item in value.items()) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(exact_json(item) for item in value) + "]"
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def number(raw, comma=False, percent=False):
    raw = raw.strip()
    match = re.fullmatch(rf"(?P<number>{NUMBER})(?P<scale>\s*{SCALE_WORD})?", raw, re.I)
    if not match:
        raise ValueError("unsupported_number")
    value = match["number"].lower()
    if value[0].isdigit() or value[0] in "+-":
        if re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", value):
            value = value.replace(",", "")
        elif "," in value:
            if comma and re.fullmatch(r"[+-]?\d+,\d{1,2}", value):
                value = value.replace(",", ".")
            else:
                raise ValueError("ambiguous_numeric_spelling")
        if value.count(".") > 1 or (re.fullmatch(r"[+-]?\d+\.\d{3}", value) and not percent):
            raise ValueError("ambiguous_numeric_spelling")
        result = Decimal(value)
    else:
        result = Decimal(0)
        for word in value.replace("-", " ").split():
            if word in ("a", "and"):
                continue
            if word == "dozen":
                result = Decimal(12)
            elif word == "hundred":
                result = (result or 1) * 100
            else:
                result += ONES.get(word, TENS.get(word, 0))
    scale = (match["scale"] or "").strip()
    if scale:
        # A lower-case attached m is a length unit, not a million multiplier.
        if scale == "m":
            raise ValueError("ambiguous_scale")
        multiplier = Decimal(MULTIPLIERS[scale if scale in MULTIPLIERS else scale.lower()])
        with localcontext() as context:
            context.prec = max(28, len(result.as_tuple().digits) + len(multiplier.as_tuple().digits) + 1)
            result = result * multiplier
    return int(result) if result == result.to_integral_value() else result


def singular(word):
    low = word.lower()
    if low == "people":
        return "person"
    if low in {"series", "species"}:
        return low
    if low.endswith(("ches", "shes", "sses", "xes", "zes")):
        return low[:-2]
    if low.endswith("ies"):
        return low[:-3] + "y"
    return low[:-1] if low.endswith("s") and not low.endswith("ss") else low


def parse_values(text, protected=(), entity_mentions=()):
    occupied = list(protected)
    values, unresolved = [], []

    def free(start, end):
        return not any(start < right and left < end for left, right in occupied)

    def reject(start, end, reason):
        if free(start, end):
            occupied.append((start, end))
            unresolved.append({"start": start, "end": end, "surface": text[start:end], "reason": reason})

    def add(start, end, kind, value, rule, **extra):
        if not free(start, end):
            return
        prefix = text[:start]
        op, op_start = "eq", start
        for phrase, candidate_op in OP_PHRASES:
            found = re.search(r"(?i)(?<!\w)" + re.escape(phrase) + r"\s*$", prefix)
            if found and free(found.start(), end):
                # 'after 3 days' describes a duration relation; retain its word
                # as context rather than treating the duration itself as >3 days.
                if kind != "date" and phrase in {"before", "after", "prior to", "by", "no later than", "no earlier than"}:
                    break
                op, op_start = candidate_op, found.start()
                break
        approximate = bool(APPROX.search(text[:op_start])) or bool(re.match(r"\s*(?:or so|give or take|[- ]ish)\b", text[end:], re.I))
        if end < len(text) and text[end] == "+" and kind in ("quantity", "percent"):
            end += 1
            op = "ge"
        clause = re.split(r"[;:!?]|\.(?=\s)|,(?!\d)", text[:op_start])[-1]
        occupied.append((op_start, end))
        values.append({"kind": kind, "value": value, "op": op, "approximate": approximate,
                       "negated_prefix": bool(NEGATION.search(clause)), "start": op_start, "end": end,
                       "surface": text[op_start:end], "rule": rule, **extra})

    # Entity names own their embedded years and product numbers. Invalid/range
    # spans are also protected so fragments cannot be mislabeled as scalars.
    for pattern, reason in [(r"\bSOC[ -]*2\b(?:\s+Type\s+(?:III|II|I)\b)?", "identifier_not_scalar"),
                            (r"\bnet[ -]+zero\b", "idiom_not_scalar")]:
        for match in re.finditer(pattern, text, re.I):
            reject(*match.span(), reason)
    ranges = re.compile(rf"(?<![\w.,])(?:between\s+)?(?:€|\$|£)?{DIGITS}(?:\s+[A-Za-z%]+)?\s*(?:[-–—]|\bto\b|\band\b|\bor\b)\s*(?:€|\$|£)?{DIGITS}(?:\s*[A-Za-z%]+)?", re.I)
    for match in ranges.finditer(text):
        if re.match(r"\d{4}-\d{2}(?:-\d{2})?", match.group()):
            continue
        if " and " in match.group().lower() and not match.group().lower().startswith("between"):
            continue
        reject(*match.span(), "range_or_choice_not_resolved")
    for match in re.finditer(r"(?<!\d)\d{1,3}(?:[ \u00a0]\d{3})+(?!\d)", text):
        reject(*match.span(), "ambiguous_space_grouped_number")
    for pattern, rule in DATES:
        for match in pattern.finditer(text):
            if not free(*match.span()):
                continue
            parts = match.groupdict()
            month = parts["month"].rstrip(".").lower()
            month = int(month) if month.isdigit() else MONTHS[month]
            year, day = int(parts["year"]), int(parts.get("day") or 1)
            try:
                date(year, month, day)
            except ValueError:
                reject(*match.span(), "invalid_calendar_date")
                continue
            value = f"{year:04d}-{month:02d}" + (f"-{day:02d}" if parts.get("day") else "")
            add(*match.span(), "date", value, rule, precision="day" if parts.get("day") else "month")
    for pattern in VERSIONS:
        for match in pattern.finditer(text):
            add(*match.span(), "version", match["value"], "explicit_version")
    for match in re.finditer(r"\b(?:CLI|release)\s+(?P<value>\d+\.\d+(?:\.\d+){0,2})(?![\w.]\d)", text, re.I):
        add(*match.span("value"), "version", match["value"], "version_context_label")
    for pattern in MONEY:
        for match in pattern.finditer(text):
            if not free(*match.span()):
                continue
            try:
                value = number(match["amount"], comma=True)
            except ValueError as error:
                reject(*match.span(), str(error))
                continue
            currency = match["currency"]
            currency = CURRENCIES.get(currency, CURRENCIES.get(currency.lower(), currency.upper()))
            end, extra = match.end(), {"currency": currency, "per": None}
            period = PERIOD.match(text, end)
            if period:
                period_name = period["period"] or period["adverb"] or "year"
                extra["per"] = PERIODS.get(period_name.lower(), period_name.lower())
                if period["basis"] or period["seat"]:
                    extra["basis"] = (period["basis"] or period["seat"]).lower()
                end = period.end()
            add(match.start(), end, "money", value, "currency_amount", **extra)
    for match in NUMBERS.finditer(text):
        if not free(*match.span()):
            continue
        end = match.end()
        plus = text[end:end + 1] == "+"
        unit_start = end + int(plus)
        suffix_op = {"op": "ge"} if plus else {}
        percent = PERCENT.match(text, unit_start)
        if percent:
            try:
                add(match.start(), percent.end(), "percent", number(match.group(), percent=True), "percent_suffix", **suffix_op)
            except ValueError as error:
                reject(match.start(), percent.end(), str(error))
            continue
        tail = re.match(r'[ -]*(?P<unit>[A-Za-z]+|["″])(?![A-Za-z])', text[unit_start:])
        word = tail["unit"] if tail else ""
        unit = PHYSICAL.get(word, PHYSICAL.get(word.lower()) if len(word) > 2 else None)
        duration = DURATIONS.get(word.lower()) if word else None
        if word in ('"', '″'):
            unit = "in"
        implicit_version = re.fullmatch(r"\d+\.\d+", match["number"]) and any(
            mention["end"] <= match.start() and not text[mention["end"]:match.start()].strip()
            for mention in entity_mentions)
        if word == "in":
            following = text[unit_start + tail.end():]
            measurement_context = re.search(r"\b(?:screen|display|size|measures|measured|height|length|width|wide|tall|across)\b",
                                            text[max(0, match.start() - 50):match.start()] + following[:25], re.I)
            if not measurement_context and (implicit_version or re.match(r"\s+[A-Za-z]", following)):
                unit = None
        if implicit_version and not unit and not duration and (not word or word.lower() in STOP or
                (singular(word) not in COUNT_NOUNS and word.lower() not in ADJECTIVES)):
            add(*match.span(), "version", match["number"], "decimal_after_entity")
            continue
        try:
            value = number(match.group())
        except ValueError as error:
            reject(*match.span(), str(error))
            continue
        if unit:
            add(match.start(), unit_start + tail.end(), "quantity", value, "physical_unit", unit=unit, unit_type="physical", **suffix_op)
        elif duration:
            add(match.start(), unit_start + tail.end(), "duration", value, "duration_unit", unit=duration, **suffix_op)
        elif re.fullmatch(r"(?:19|20)\d{2}", match.group()):
            # Quarters/halves are not calendar dates.
            if not re.search(r"\b[QH][1-4]\s*$", text[:match.start()]):
                add(*match.span(), "date", match.group(), "bare_year", precision="year")
        else:
            noun_match = re.match(r"[ -]*(?:[A-Za-z]+[ -]+){0,2}[A-Za-z]+", text[unit_start:])
            chosen = None
            if noun_match:
                words = list(re.finditer(r"[A-Za-z]+", noun_match.group()))
                for candidate in words:
                    noun = candidate.group().lower()
                    if noun in ADJECTIVES:
                        continue
                    if noun not in STOP:
                        chosen = candidate
                    break
            if chosen:
                add(match.start(), unit_start + chosen.end(), "quantity", value, "counted_noun_heuristic",
                    unit=singular(chosen.group()), unit_type="count", **suffix_op)
            else:
                unresolved.append({"start": match.start(), "end": match.end(), "surface": match.group(),
                                   "reason": "number_without_supported_type"})
    # 'under a second' is a duration; 'requests a day' is left as a rate phrase.
    articles = re.compile(r"\b(?:a|an)\s+(second|minute|hour|day|week|month|year)\b", re.I)
    for match in articles.finditer(text):
        if re.search(r"\b(?:under|within|in|for|over|than|to|only|just|at least|at most)\s*$", text[:match.start()], re.I):
            add(*match.span(), "duration", 1, "article_duration", unit=DURATIONS[match[1].lower()])
    return sorted(values, key=lambda item: item["start"]), sorted(unresolved, key=lambda item: item["start"])


def annotate(text, linker):
    entity_result = linker.normalize(text)
    entities = [dict(mention) for mention in entity_result["mentions"]]
    ambiguous = entity_result["ambiguous_mentions"]
    protected = [(item["start"], item["end"]) for item in entities + ambiguous]
    protected += [match.span() for match in re.finditer(r"<[A-Z]+:\d+>", text)]
    values, unresolved = parse_values(text, protected, entities)
    counters = Counter()
    for value in values:
        counters[value["kind"]] += 1
        value["value_id"] = counters[value["kind"]]
        value["token"] = f"<{value['kind'].upper()}:{value['value_id']}>"
    pieces, cursor, length = [], 0, 0
    for item in sorted(entities + values, key=lambda item: item["start"]):
        prefix = text[cursor:item["start"]]
        pieces.append(prefix)
        length += len(prefix)
        token = item.get("token", f"<ENTITY:{item.get('entity_id')}>")
        item["normalized_start"], item["normalized_end"] = length, length + len(token)
        pieces.append(token)
        length += len(token)
        cursor = item["end"]
    pieces.append(text[cursor:])
    return {"original_text": text, "entity_normalized_text": entity_result["normalized_text"],
            "normalized_text": "".join(pieces), "mentions": entities,
            "ambiguous_mentions": ambiguous, "suppressed_mentions": entity_result.get("suppressed_mentions", []),
            "values": values, "unresolved_values": unresolved}
