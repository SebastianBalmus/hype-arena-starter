"""Local referee: does an entry's text state its claim in the forms of contracts/claim-forms.md?

``check_claim`` returns the reasons an entry is invalid; an empty list means the text reads.
It is deliberately strict: a claim it accepts must read for the real referee.
"""
from __future__ import annotations

import re

from arena.lexicon import Lexicon, has_attribute_words
from arena.schema import validate_entry
from arena.values import compare, has_negation, parse_values, render, split_sentences, values_equal

__all__ = ["check_claim", "COMPARE_PHRASES", "EVENT_PHRASES", "REFUSED_DATE_WORDS", "TIME_WORDS"]

_B = r"(?<![A-Za-z0-9])"
_E = r"(?![A-Za-z0-9])"

# ---------------------------------------------------------------------------------------
# Published lists (contracts/claim-forms.md)
# ---------------------------------------------------------------------------------------

COMPARE_PHRASES = {
    "amounts": {
        "gt": ["more … than", "higher … than", "greater … than", "larger … than", "bigger … than", "longer … than"],
        "lt": ["less … than", "fewer … than", "lower … than", "smaller … than", "shorter … than", "cheaper … than"],
        "eq": ["the same … as", "as much … as", "as many … as", "as long as", "as high as", "equal to"],
    },
    "dates": {
        "gt": ["after", "later than"],
        "lt": ["before", "earlier than"],
        "eq": ["on the same day as", "the same day as"],
    },
}

EVENT_PHRASES = ["at", "during", "on the day of", "at the time of"]

REFUSED_DATE_WORDS = [
    "January", "February", "March", "April", "May", "June", "July", "August", "September",
    "October", "November", "December", "spring", "summer", "autumn", "winter",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
]

TIME_WORDS = [
    "ago", "yesterday", "tomorrow", "once", "formerly", "previously", "originally", "initially",
    "recently", "lately", "soon", "eventually", "then", "before", "after", "since", "until", "till",
    "earlier", "later", "used to", "last week", "last month", "last quarter", "last year",
    "next week", "next month", "next quarter", "next year", "in the past", "in the future",
    "at the time", "at that time",
]

_NUMBER_WORDS = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    "hundred", "thousand", "million", "billion", "dozen", "dozens", "bn",
    "twice", "double", "triple", "thrice", "quadruple", "half", "once", "several",
]

# Attributes that hold dates, for telling the dates table from the amounts table when no
# kind index is given (all date-kind attributes of the practice dataset, plus *_date).
_DATE_ATTRIBUTES = {
    "announced", "beta_date", "beta_opened", "chair_since", "date", "founded",
    "general_availability_date", "joined", "latest_release_date", "launched", "lease_end",
    "net_zero_date", "next_meeting", "next_round", "opened", "planned_general_availability",
    "published", "released", "soc2_audit_start", "stable_release", "stable_release_date",
    "training_data_cutoff", "generally_available",
}

_STRIP_NEG_RE = re.compile(
    r"(?i)(?<=[A-Za-z])n['’]t" + _E + "|" + _B + r"(?:not|never|none|nobody|nothing|neither|nor|without|cannot|no)" + _E
)
_INVISIBLE_RE = re.compile("[​‌‍⁠﻿­]")
_NEG_EXTRA_RE = re.compile(r"(?i)" + _B + r"(?:no|not)" + _E)  # "no more than" is no negation for has_negation


def _words_re(phrase: str) -> str:
    out = ""
    for i, tok in enumerate(phrase.split()):
        if tok == "…":
            out += r"(?:\s+[A-Za-z0-9'’-]+){0,3}?"
        else:
            out += (r"\s+" if i else "") + re.escape(tok)
    return out


def _compile_list(words: list[str], flags=re.IGNORECASE) -> re.Pattern[str]:
    alts = sorted((_words_re(w) for w in words), key=len, reverse=True)
    return re.compile(_B + "(?:" + "|".join(alts) + ")" + _E, flags)


_PHRASE_RES = {
    table: {op: _compile_list(ps) for op, ps in ops.items()} for table, ops in COMPARE_PHRASES.items()
}
_ANY_PHRASE_RE = _compile_list([p for ops in COMPARE_PHRASES.values() for ps in ops.values() for p in ps])
_REFUSED_RE = re.compile(
    _B + "(?:May|(?i:" + "|".join(w + ("s?" if w.endswith("day") else "") for w in REFUSED_DATE_WORDS if w != "May") + "|"
    + "|".join(sorted((_words_re(w) for w in TIME_WORDS), key=len, reverse=True)) + "))" + _E
)
_NUMBER_RE = re.compile(r"\d|" + _B + "(?i:" + "|".join(_NUMBER_WORDS) + ")" + _E)
_EVENT_PHRASE_RE = re.compile(
    r"(?i)" + _B + "(?:" + "|".join(_words_re(p) for p in sorted(EVENT_PHRASES, key=len, reverse=True))
    + r")\s+(?:the\s+)?$"
)
_FROM_TO_RE = re.compile(r"(?i)" + _B + r"(?:(up|down)\s+)?from\s+(.+?)\s+to\s+")
_UPDOWN_FROM_RE = re.compile(r"(?i),\s*(up|down)\s+from\s+")


# ---------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------


def _within(start: int, end: int, spans) -> bool:
    return any(a <= start and end <= b for a, b in spans)


def _negated(sentence: str) -> bool:
    return has_negation(sentence) or bool(_NEG_EXTRA_RE.search(sentence))


def _refused(sentence: str, allowed) -> list[str]:
    return [m.group(0) for m in _REFUSED_RE.finditer(sentence) if not _within(m.start(), m.end(), allowed)]


def _numbers(sentence: str, allowed) -> list[str]:
    return [m.group(0) for m in _NUMBER_RE.finditer(sentence) if not _within(m.start(), m.end(), allowed)]


def _same_kind(a: dict, b: dict) -> bool:
    """Is ``a`` a value of ``b``'s kind, as the one-value rule reads it?"""
    if a.get("kind") != b.get("kind"):
        return False
    if a["kind"] in ("quantity", "duration", "money"):
        return compare(a, b) is not None
    return True


def _clause_start(s: str, pos: int) -> int:
    i = pos - 1
    while i >= 0:
        c = s[i]
        if c in ";:!?•·|":
            return i + 1
        if c in ".," and not (0 < i < len(s) - 1 and s[i - 1].isdigit() and s[i + 1].isdigit()):
            return i + 1
        i -= 1
    return 0


def _clause_end(s: str, pos: int) -> int:
    i = pos
    while i < len(s):
        c = s[i]
        if c in ";:!?•·|":
            return i
        if c in ".," and not (0 < i < len(s) - 1 and s[i - 1].isdigit() and s[i + 1].isdigit()):
            return i
        i += 1
    return len(s)


def _name_spans(text: str, name: str, lexicon: Lexicon, mentions=None) -> list[tuple[int, int]]:
    """Where ``text`` names ``name``: the lexicon's spellings, else the name itself (whole words,
    any case, spaces optional), never inside another entity's longer name."""
    mentions = lexicon.mentions(text) if mentions is None else mentions
    canon = lexicon.canonical(name)
    if canon is not None:
        return [(m.start, m.end) for m in mentions if m.subject == canon]
    toks = name.split()
    if not toks:
        return []
    rx = re.compile(r"(?<!\w)" + r"\s*".join(re.escape(t) for t in toks) + r"(?!\w)", re.IGNORECASE)
    return [
        (m.start(), m.end())
        for m in rx.finditer(text)
        if not any(mm.start < m.end() and m.start() < mm.end for mm in mentions)
    ]


def _looks_date(attribute: str) -> bool:
    return attribute in _DATE_ATTRIBUTES or attribute.endswith("_date")


# ---------------------------------------------------------------------------------------
# The value of a fact, at_event or relation sentence
# ---------------------------------------------------------------------------------------


def _value_rules(s, value, attribute, needed, lexicon, mentions, allow_names=()) -> list[str]:
    """One value, one date, no negation (yes or no aside), no refused word.

    ``needed`` are the spans of the names (and event phrase) the form needs."""
    errs: list[str] = []
    kind = value["kind"]
    allowed = list(needed)
    spans = [sp for sp in parse_values(s) if not _within(sp.start, sp.end, needed)]

    if kind == "boolean":
        if not has_attribute_words(s, attribute):
            errs.append(f"the sentence does not hold the words of {attribute!r}")
        if (not has_negation(s)) != bool(value["value"]):
            errs.append(f"the sentence states {'no' if has_negation(s) else 'yes'}")
        for sp in spans:
            if sp.value["kind"] == "date":
                errs.append(f"a date of its own: {s[sp.start:sp.end]!r}")
        # parse_values skips what follows a negation: look again with the negation words blanked out
        blank = _STRIP_NEG_RE.sub(lambda m: " " * len(m.group(0)), s)
        seen = {(sp.start, sp.end) for sp in spans}
        for sp in parse_values(blank):
            if sp.value["kind"] == "date" and (sp.start, sp.end) not in seen and not _within(sp.start, sp.end, needed):
                errs.append(f"a date of its own: {s[sp.start:sp.end]!r}")
        for m in re.finditer(_B + r"(?:1[0-9]|20)\d\d" + _E, s):
            if not _within(m.start(), m.end(), needed):
                errs.append(f"a year of its own: {m.group(0)!r}")
    else:
        if _negated(s):
            errs.append("a negation")
        if kind == "name":
            hits = [h for h in _name_spans(s, value["value"], lexicon, mentions) if not _within(*h, needed)]
            if not hits:
                errs.append(f"the sentence does not name {value['value']!r}")
            allowed += hits
            value_canon = lexicon.canonical(value["value"])
            for m in mentions:
                if m.subject != value_canon and m.subject not in allow_names and not _within(m.start, m.end, allowed):
                    errs.append(f"another name: {m.surface!r}")
            for sp in spans:
                if sp.value["kind"] == "date" and not _within(sp.start, sp.end, allowed):
                    errs.append(f"a date of its own: {s[sp.start:sp.end]!r}")
        else:
            hits = [sp for sp in spans if "op" not in sp.value and values_equal(sp.value, value)]
            if not hits:
                errs.append(f"the sentence does not state {render(value)}")
            for sp in spans:
                if sp in hits:
                    continue
                if _same_kind(sp.value, value):
                    errs.append(f"another value of its kind: {s[sp.start:sp.end]!r}")
                elif sp.value["kind"] == "date":
                    errs.append(f"a date of its own: {s[sp.start:sp.end]!r}")
            allowed += [(sp.start, sp.end) for sp in hits]
    for w in _refused(s, allowed):
        errs.append(f"a date or time word: {w!r}")
    return errs


def _subject_spans(s, canon, mentions, errs, what="subject"):
    spans = [(m.start, m.end) for m in mentions if m.subject == canon]
    if not spans:
        errs.append(f"the sentence does not name the {what} {canon!r}")
    return spans


# ---------------------------------------------------------------------------------------
# Per type: the reasons one sentence does not hold the claim
# ---------------------------------------------------------------------------------------


def _fact(s, form, lexicon, ctx):
    errs: list[str] = []
    mentions = lexicon.mentions(s)
    needed = _subject_spans(s, ctx["subject"], mentions, errs)
    errs += _value_rules(s, form["value"], form["attribute"], needed, lexicon, mentions, {ctx["subject"]})
    return errs


def _at_event(s, form, lexicon, ctx):
    errs: list[str] = []
    mentions = lexicon.mentions(s)
    needed = _subject_spans(s, ctx["subject"], mentions, errs)
    event_spans = []
    for m in mentions:
        if m.subject != ctx["event"]:
            continue
        pm = _EVENT_PHRASE_RE.search(s[: m.start])
        if pm:
            event_spans.append((pm.start(), m.end))
    if not event_spans:
        errs.append(f"no event phrase right before a name of {ctx['event']!r}")
    needed = needed + event_spans
    errs += _value_rules(
        s, form["value"], form["attribute"], needed, lexicon, mentions, {ctx["subject"], ctx["event"]}
    )
    return errs


def _relation(s, form, lexicon, ctx):
    errs: list[str] = []
    mentions = lexicon.mentions(s)
    needed = _subject_spans(s, ctx["subject"], mentions, errs)
    if not has_attribute_words(s, form["link"]):
        errs.append(f"the sentence does not hold the relation words of {form['link']!r}")
    errs += _value_rules(
        s, form["value"], form["attribute"], needed, lexicon, mentions, {ctx["subject"], ctx["target"]}
    )
    return errs


def _side_states(side: str, value: dict, attribute: str, lexicon: Lexicon, whole: bool) -> bool:
    """Does one side of a change phrase state ``value`` on its own, at its start?"""
    text = side.strip()
    if not text:
        return False
    kind = value["kind"]
    if kind == "boolean":
        return has_attribute_words(text, attribute) and (not has_negation(text)) == bool(value["value"])
    if kind == "name":
        hits = _name_spans(text, value["value"], lexicon)
        return any(a == 0 and (not whole or b == len(text)) for a, b in hits)
    spans = parse_values(text)
    if not spans:
        return False
    sp = spans[0]
    if sp.start != 0 or "op" in sp.value or not values_equal(sp.value, value):
        return False
    return not whole or (sp.end == len(text) and len(spans) == 1)


def _direction_ok(direction, frm, to) -> bool:
    if direction is None:
        return not values_equal(frm, to)
    if frm["kind"] in ("name", "boolean"):
        return False
    c = compare(to, frm)
    return c == (1 if direction.lower() == "up" else -1)


def _change(s, form, lexicon, ctx):
    errs: list[str] = []
    mentions = lexicon.mentions(s)
    _subject_spans(s, ctx["subject"], mentions, errs)
    frm, to, attr = form["from"], form["to"], form["attribute"]
    if frm["kind"] != "boolean" and _negated(s):
        errs.append("a negation")
    found = False
    for m in _FROM_TO_RE.finditer(s):
        y = s[m.end(): _clause_end(s, m.end())]
        if (
            _side_states(m.group(2), frm, attr, lexicon, whole=True)
            and _side_states(y, to, attr, lexicon, whole=False)
            and _direction_ok(m.group(1), frm, to)
        ):
            found = True
    for m in _UPDOWN_FROM_RE.finditer(s):
        before = s[_clause_start(s, m.start()): m.start()].strip()
        after = s[m.end(): _clause_end(s, m.end())]
        if frm["kind"] in ("name", "boolean"):
            continue
        spans = parse_values(before)
        to_ok = bool(spans) and spans[-1].end == len(before) and "op" not in spans[-1].value \
            and values_equal(spans[-1].value, to)
        if to_ok and _side_states(after, frm, attr, lexicon, whole=False) and _direction_ok(m.group(1), frm, to):
            found = True
    if not found:
        errs.append(f"no change phrase from {render(frm)} to {render(to)}")
    return errs


def _compare(s, form, lexicon, ctx):
    errs: list[str] = []
    mentions = lexicon.mentions(s)
    lefts = [m for m in mentions if m.subject == ctx["left"]]
    rights = [m for m in mentions if m.subject == ctx["right"]]
    if not lefts:
        errs.append(f"the sentence does not name the left subject {ctx['left']!r}")
    if not rights:
        errs.append(f"the sentence does not name the right subject {ctx['right']!r}")
    table = "dates" if ctx["dates"] else "amounts"
    phrases = [(m.start(), m.end()) for m in _PHRASE_RES[table][form["op"]].finditer(s)]
    triple = None
    for lm in lefts:
        for ps, pe in phrases:
            for rm in rights:
                if rm is not lm and lm.end <= ps and pe <= rm.start:
                    triple = (ps, pe)
    if triple is None:
        errs.append(f"no left name, {form['op']} phrase of the {table} list, right name in that order")
    if _negated(s):
        errs.append("a negation")
    names = [(m.start, m.end) for m in lefts + rights]
    for sp in parse_values(s):
        if not _within(sp.start, sp.end, names):
            errs.append(f"a value: {s[sp.start:sp.end]!r}")
    for w in _numbers(s, names):
        errs.append(f"a number: {w!r}")
    for w in _refused(s, names + ([triple] if triple else [])):
        errs.append(f"a date or time word: {w!r}")
    return errs


_CHECKERS = {"fact": _fact, "compare": _compare, "change": _change, "at_event": _at_event, "relation": _relation}


# ---------------------------------------------------------------------------------------
# Names and attributes against the lexicon
# ---------------------------------------------------------------------------------------


def _resolve(lexicon, name, what, errs):
    canon = lexicon.canonical(name)
    if canon is None:
        errs.append(f"{what} {name!r} is not in the lexicon")
    return canon


def _has_attribute(lexicon, canon, attribute, errs):
    if canon is not None and attribute not in lexicon.attributes(canon):
        errs.append(f"{canon!r} has no attribute {attribute!r}")


def _context(ctype, form, lexicon, kinds) -> tuple[dict, list[str]]:
    errs: list[str] = []
    ctx: dict = {}
    if ctype == "compare":
        lf, rf = form["left"], form["right"]
        ctx["left"] = _resolve(lexicon, lf["subject"], "subject", errs)
        ctx["right"] = _resolve(lexicon, rf["subject"], "subject", errs)
        _has_attribute(lexicon, ctx["left"], lf["attribute"], errs)
        _has_attribute(lexicon, ctx["right"], rf["attribute"], errs)
        if ctx["left"] is not None and (ctx["left"], lf["attribute"]) == (ctx["right"], rf["attribute"]):
            errs.append("a fact does not compare with itself")
        sides = []
        for attr in (lf["attribute"], rf["attribute"]):
            kind = kinds.kind(attr) if kinds is not None else None
            if kind in ("name", "boolean"):
                errs.append(f"{attr!r} holds a {kind}, which does not compare")
            sides.append((kind, kind == "date" if kind else _looks_date(attr)))
        if sides[0][0] and sides[1][0] and sides[0][0] != sides[1][0]:
            errs.append(f"a {sides[0][0]} does not compare with a {sides[1][0]}")
        if sides[0][1] != sides[1][1]:
            errs.append("a date does not compare with an amount")
        ctx["dates"] = sides[0][1]
        return ctx, errs
    ctx["subject"] = _resolve(lexicon, form["subject"], "subject", errs)
    if ctype == "relation":
        _has_attribute(lexicon, ctx["subject"], form["link"], errs)
        ctx["target"] = _resolve(lexicon, form["target"], "target", errs)
        _has_attribute(lexicon, ctx["target"], form["attribute"], errs)
        return ctx, errs
    _has_attribute(lexicon, ctx["subject"], form["attribute"], errs)
    if ctype == "at_event":
        ctx["event"] = _resolve(lexicon, form["event"], "event", errs)
        if ctx["event"] is not None and ctx["event"] not in lexicon.events:
            errs.append(f"{ctx['event']!r} is not an event")
    return ctx, errs


# ---------------------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------------------


def check_claim(entry: dict, lexicon: Lexicon, kinds=None) -> list[str]:
    """Why ``entry`` is invalid for the referee; ``[]`` when its text states its claim.

    ``kinds`` is optional: anything with ``kind(attribute) -> str | None`` (such as a
    ``KindIndex``). For a compare it says which phrase table applies and refuses names and
    yes-or-no values; without it, date attributes are told by name.
    """
    errs = validate_entry(entry)
    if errs:
        return ["schema: " + e for e in errs]
    ctype, form, text = entry["type"], entry["form"], entry["text"]
    ctx, errs = _context(ctype, form, lexicon, kinds)
    if errs:
        return errs
    sentences = split_sentences(text)
    if ctype == "compare":
        count = 0
        for s in sentences:
            spans = sorted((m.start(), m.end()) for m in _ANY_PHRASE_RE.finditer(s))
            end = -1
            for a, b in spans:
                if a >= end:
                    count += 1
                end = max(end, b)
        if count > 1:
            errs.append("the text holds a second compare phrase")
    best: list[str] | None = None
    for s in sentences:
        if _INVISIBLE_RE.search(s):
            reasons = ["an invisible character"]
        else:
            reasons = _CHECKERS[ctype](s, form, lexicon, ctx)
        if not reasons:
            return errs
        if best is None or len(reasons) < len(best):
            best = reasons
    return errs + (best or ["the text holds no sentence"])
