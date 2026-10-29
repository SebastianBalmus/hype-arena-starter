"""Writer + Referee: exactly 1,000 new claims, every one true, each with its confidence.

Truth comes only from labelled-true rows (their timelines say which value held when).
Candidates, safest first:
  change      two labelled-true values of a fact, in time order (as-of date irrelevant)
  compare     two past dates (joined, launched, ...: they cannot change any more)
  relation    a link's target (e.g. Hooly's CTO) and that target's stable fact
  at_event    a value whose certain core covers the event's final date
  compare     two recent amounts of one kind with a wide gap (robust to small changes)
  fact        immutable past dates, or values confirmed recently
Each claim is rendered with conservative templates in the value forms of
contracts/value-forms.md, then re-read by our Referee (hooly.extract): the subject must be
named, the value stated exactly once, no competing value, date, time word or negation; a
compare holds no number at all. Only claims the Referee accepts are kept.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from datetime import date

from .data import Row
from .extract import NEGATIONS, Extractor, _singular
from .temporal import Timeline
from .values import compare as vcompare, matches, normalize, satisfies, EQUAL, COMPATIBLE

AS_OF = date(2026, 10, 29)
MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
               "October", "November", "December"]
REFUSED = ["january", "february", "march", "april", "may", "june", "july", "august", "september",
           "october", "november", "december", "spring", "summer", "autumn", "winter", "monday",
           "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "ago", "yesterday",
           "tomorrow", "once", "formerly", "previously", "originally", "initially", "recently",
           "lately", "soon", "eventually", "then", "before", "after", "since", "until", "till",
           "earlier", "later", "used to", "last week", "last month", "last quarter", "last year",
           "next week", "next month", "next quarter", "next year", "in the past", "in the future",
           "at the time", "at that time", "last", "next", "previous", "current", "today", "now"]
PAST_DATE_WORDS = {"joined", "founded", "launched", "released", "opened", "started", "announced",
                   "shipped", "acquired", "incorporated", "beta_opened", "stable_release_date", "date"}
CURRENCY = {"EUR": "€", "USD": "$", "GBP": "£"}
DUR_WORDS = {"ms": ("millisecond", "milliseconds"), "s": ("second", "seconds"), "min": ("minute", "minutes"),
             "h": ("hour", "hours"), "day": ("day", "days"), "week": ("week", "weeks"),
             "month": ("month", "months"), "year": ("year", "years")}
QUNIT = {"in": "-inch"}
from .extract import WORD_NUMBERS as _WN
_NUMBER_WORDS = re.compile(r"(?<![a-z-])(" + "|".join(sorted((re.escape(w) for w in list(_WN) + ["one", "hundred", "thousand", "million", "billion", "dozen"]), key=len, reverse=True)) + r")(?![a-z-])", re.I)


def _d(s):
    return date.fromisoformat(s)


# ---------------------------------------------------------------- rendering
def fmt_num(x, allow3=False):
    x = float(x)
    if x.is_integer():
        return f"{int(x):,}"
    s = repr(x)
    if "e" in s:
        return None
    dec = s.split(".")[1]
    if len(dec) > 2 and not (allow3 and len(dec) == 3):
        return None  # 1.500 is ambiguous; long decimals are not worth the risk
    whole = f"{int(x):,}" if abs(x) >= 1000 else str(int(x)) if x >= 0 else "-" + str(int(abs(x)))
    return f"{whole}.{dec}"


def plural(noun, n):
    if float(n) == 1:
        return noun
    if noun.endswith("y") and noun[-2:-1] not in "aeiou":
        return noun[:-1] + "ies"
    if noun.endswith(("s", "x", "ch", "sh")):
        return noun + "es"
    if noun == "person":
        return "people"
    return noun + "s"


def render(v: dict) -> str | None:
    k = v["kind"]
    if "op" in v:
        return None
    if k == "quantity":
        n = fmt_num(v["value"])
        if n is None:
            return None
        u = v["unit"]
        if u in QUNIT:
            return n + QUNIT[u]
        if u in ("B", "KB", "MB", "GB", "TB", "mm", "cm", "m", "km", "g", "kg", "mAh", "Hz", "kHz", "MHz", "GHz", "mW", "W", "kW"):
            return f"{n} {u}"
        return f"{n} {plural(u, v['value'])}"
    if k == "money":
        n = fmt_num(v["value"])
        if n is None:
            return None
        cur = CURRENCY.get(v["currency"])
        s = f"{cur}{n}" if cur else f"{v['currency']} {n}"
        return s + (f" per {v['per']}" if v.get("per") else "")
    if k == "percent":
        n = fmt_num(v["value"], allow3=True)
        return None if n is None else f"{n}%"
    if k == "duration":
        n = fmt_num(v["value"])
        if n is None or v["unit"] not in DUR_WORDS:
            return None
        one, many = DUR_WORDS[v["unit"]]
        return f"{n} {one if float(v['value']) == 1 else many}"
    if k == "date":
        parts = v["value"].split("-")
        if len(parts) == 3:
            return f"{int(parts[2])} {MONTH_NAMES[int(parts[1]) - 1]} {parts[0]}"
        if len(parts) == 2:
            return f"{MONTH_NAMES[int(parts[1]) - 1]} {parts[0]}"
        return f"in {parts[0]}"
    if k == "version":
        return f"version {v['value']}"
    if k == "name":
        return v["value"]
    return None


def attr_words(attr: str) -> str:
    return attr.replace("_", " ")


def safe_words(text: str) -> bool:
    """No refused time word, month or season, and no digit (outside names it would be a value)."""
    t = " " + text.lower() + " "
    if re.search(r"\d", text):
        return False
    return not any(re.search(r"(?<![a-z])" + re.escape(w) + r"(?![a-z])", t) for w in REFUSED)


# ---------------------------------------------------------------- the Referee
class Referee:
    """Re-reads a rendered claim with our own value parser and the contract's rules."""

    def __init__(self, lexicon, names, input_texts):
        self.lexicon = lexicon
        self.names = names
        self.attrs = {s["name"]: set(s["attributes"]) for s in lexicon["subjects"]}
        self.spell = {}
        for s in lexicon["subjects"]:
            for n in [s["name"]] + s.get("aliases", []):
                self.spell[n.lower()] = s["name"]
        name_spellings = [n for s in lexicon["subjects"] for n in [s["name"]] + s.get("aliases", [])]
        self.extractor = Extractor(nouns=set(), names=name_spellings, protected=name_spellings + list(lexicon.get("events", [])),
                                   units=set())
        self.inputs = {" ".join(t.lower().split()).rstrip(".!? ") for t in input_texts}
        self.subject_res = sorted(((n, re.compile(r"(?<![\w])" + r"\s*".join(map(re.escape, n.split())) + r"(?![\w])", re.I))
                                   for n in name_spellings), key=lambda x: -len(x[0]))

    def mentions(self, text):
        taken, out = [], set()
        for n, rx in self.subject_res:
            for m in rx.finditer(text):
                if all(m.end() <= s or m.start() >= e for s, e in taken):
                    taken.append((m.start(), m.end()))
                    out.add(self.spell[n.lower()])
        return out

    def _values(self, text, value_kind_unit=None):
        vals = [f.value for f in self.extractor.find(text) if f.kind != "name"]
        return vals

    def ok(self, claim) -> tuple[bool, str]:
        text, form, typ = claim["text"], claim["form"], claim["type"]
        if len(text) > 300 or re.search(r"[.!?]\s", text.strip()[:-1]):
            return False, "length or more than one sentence"
        if " ".join(text.lower().split()).rstrip(".!? ") in self.inputs:
            return False, "copies an input row"
        subjects = [form.get("subject")] if typ != "compare" else [form["left"]["subject"], form["right"]["subject"]]
        attrs = [form.get("attribute")] if typ != "compare" else [form["left"]["attribute"], form["right"]["attribute"]]
        if typ == "relation":
            subjects.append(form["target"]); attrs.append(form["attribute"])
            if form["link"] not in self.attrs.get(form["subject"], set()):
                return False, "link not an attribute of the subject"
        if typ == "relation":
            pairs = [(form["subject"], form["link"]), (form["target"], form["attribute"])]
        else:
            pairs = list(zip(subjects, attrs))
        for s, a in pairs:
            if s not in self.attrs or a not in self.attrs[s]:
                return False, f"not in lexicon: {s}.{a}"
        named = self.mentions(text)
        need = [form["left"]["subject"], form["right"]["subject"]] if typ == "compare" else [form["subject"]]
        if any(n not in named for n in need):
            return False, "subject not named"
        vals = self._values(text)
        if typ == "compare":
            bare = self._strip_names(text)
            if vals or re.search(r"\d", bare) or _NUMBER_WORDS.search(bare):
                return False, "compare states a number"
            if not safe_words(self._strip_names(text).replace(" before ", " ").replace(" after ", " ")):
                return False, "time word in compare"
            return True, "ok"
        if typ == "change":
            want = [form["from"], form["to"]]
            found = [v for v in vals if v.get("kind") == form["from"]["kind"]]
            if len(found) != 2 or not all(any(satisfies(w, f, self.names) and vcompare(w, f, self.names) == EQUAL for f in found) for w in want):
                return False, f"change values not read back: {found}"
            if NEGATIONS.search(text):
                return False, "negation"
            return True, "ok"
        v = form["value"]
        if v["kind"] == "boolean":
            neg = bool(NEGATIONS.search(text))
            if neg == v["value"]:
                return False, "polarity"
            if attr_words(form["attribute"]).lower() not in text.lower():
                return False, "yes/no without attribute words"
            if vals:
                return False, "stray value in yes/no"
            return True, "ok"
        if NEGATIONS.search(text):
            return False, "negation"
        same = [x for x in vals if x["kind"] == v["kind"]]
        if v["kind"] == "name":
            if v["value"].lower() not in text.lower():
                return False, "name value missing"
            return True, "ok"
        if len(same) != 1 or vcompare(same[0], v, self.names) != EQUAL or "op" in same[0]:
            return False, f"value not read back exactly once: {same}"
        if any(x["kind"] == "date" for x in vals if x is not same[0]) and v["kind"] != "date":
            return False, "stray date"
        return True, "ok"

    def _strip_names(self, text):
        for n, rx in self.subject_res:
            text = rx.sub(" ", text)
        for e in self.lexicon.get("events", []):
            text = re.sub(re.escape(e), " ", text, flags=re.I)
        return text


# ---------------------------------------------------------------- candidates
class Writer:
    def __init__(self, rows: list[Row], lexicon, names, extra_true=frozenset()):
        """extra_true: ids of unlabelled rows the system itself labelled true with high confidence;
        they count as evidence only in the second tier (lower claim confidence)."""
        self.lexicon, self.names = lexicon, names
        self.attrs = {s["name"]: list(s["attributes"]) for s in lexicon["subjects"]}
        self.lex_names = {n.lower(): s["name"] for s in lexicon["subjects"] for n in [s["name"]] + s.get("aliases", [])}
        obs = defaultdict(list)
        for r in rows:
            if (r.label == "true" or r.id in extra_true) and r.atoms:
                for a in r.atoms:
                    if "op" not in a.value:
                        obs[a.key].append((_d(r.date), a.value, r.id))
        self.timelines = {k: Timeline(o, names) for k, o in obs.items()}
        self.last_seen = {k: max(d for d, _, _ in o) for k, o in obs.items()}
        self.events = list(lexicon.get("events", []))

    def stable_now(self, key):
        """Value of a fact on 29 Oct if it can be trusted: (value, why) or None."""
        tl = self.timelines.get(key)
        if not tl or tl.multi or not tl.runs:
            return None
        last = tl.runs[-1]
        v = last.values[0]
        if v["kind"] == "date" and _d(self._iso_end(v)) < last.end and key[1] in PAST_DATE_WORDS | {"date"}:
            return v, "past date"
        if v["kind"] == "date" and _d(self._iso_end(v)) < last.start and any(w in key[1] for w in ("joined", "founded", "launched", "released", "opened")):
            return v, "past date"
        # a fact that has changed before may change again: its last sighting must be fresher
        # (fact claims are judged on 29 Oct; ~94% held under the 30-day rule, the misses volatile)
        recent = 21 if len(tl.runs) > 1 else 30
        if (AS_OF - last.end).days <= recent and (last.end - last.start).days >= 0:
            return v, "seen recently"
        if len(tl.runs) == 1 and (last.end - last.start).days >= 120 and (AS_OF - last.end).days <= 75:
            return v, "long-stable"
        return None

    @staticmethod
    def _iso_end(v):
        p = v["value"].split("-")
        if len(p) == 1:
            return f"{p[0]}-12-31"
        if len(p) == 2:
            import calendar
            return f"{p[0]}-{p[1]}-{calendar.monthrange(int(p[0]), int(p[1]))[1]:02d}"
        return v["value"]

    def subject_name(self, s):
        return s

    # ---- each type yields (confidence, claim) ----
    def changes(self):
        for key, tl in self.timelines.items():
            if tl.multi or len(tl.runs) < 2:
                continue
            for i in range(len(tl.runs)):
                for j in range(i + 1, len(tl.runs)):
                    a, b = tl.runs[i].values[0], tl.runs[j].values[0]
                    if a["kind"] in ("boolean",) or matches(a, b, self.names):
                        continue
                    if a["kind"] == "name" and (a["value"].lower() not in self.lex_names or b["value"].lower() not in self.lex_names):
                        continue
                    ra, rb = render(a), render(b)
                    if not ra or not rb:
                        continue
                    subj = key[0]
                    words = attr_words(key[1])
                    lead = f"The {words} of {subj}" if safe_words(words) else subj
                    text = f"{lead} went from {ra} to {rb}."
                    conf = 0.97 if j == i + 1 else 0.96
                    yield conf, {"type": "change", "text": text,
                                 "form": {"subject": subj, "attribute": key[1], "from": a, "to": b}, "key": key}

    def date_compares(self):
        past = []
        for key in self.timelines:
            got = self.stable_now(key)
            if got and got[0]["kind"] == "date" and got[1] == "past date" and len(got[0]["value"]) == 10:
                past.append((key, got[0]))
        for i in range(len(past)):
            for j in range(i + 1, len(past)):
                (k1, v1), (k2, v2) = past[i], past[j]
                if k1 == k2 or v1["value"] == v2["value"]:
                    continue
                if v1["value"] > v2["value"]:
                    (k1, v1), (k2, v2) = (k2, v2), (k1, v1)
                w1, w2 = attr_words(k1[1]), attr_words(k2[1])
                left = f"{k1[0]}" + (f" ({w1})" if safe_words(w1) else "")
                right = f"{k2[0]}" + (f" ({w2})" if safe_words(w2) else "")
                text = f"The {w1} of {k1[0]} came before the {w2} of {k2[0]}." if safe_words(w1) and safe_words(w2) \
                    else f"{k1[0]} came before {k2[0]}."
                yield 0.95, {"type": "compare", "text": text,
                             "form": {"left": {"subject": k1[0], "attribute": k1[1]},
                                      "right": {"subject": k2[0], "attribute": k2[1]}, "op": "lt"},
                             "key": (k1, k2)}

    def amount_compares(self):
        now = []
        for key in self.timelines:
            got = self.stable_now(key)
            if got and got[0]["kind"] in ("quantity", "money", "percent", "duration") and got[1] != "past date":
                try:
                    now.append((key, got[0], normalize(got[0], self.names)))
                except ValueError:
                    pass
        by_dim = defaultdict(list)
        for key, v, n in now:
            by_dim[n.dim].append((key, v, n))
        for dim, items in by_dim.items():
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    (k1, v1, n1), (k2, v2, n2) = items[i], items[j]
                    if k1 == k2 or n1.value <= 0 or n2.value <= 0:
                        continue
                    big, small = ((k1, n1), (k2, n2)) if n1.value > n2.value else ((k2, n2), (k1, n1))
                    ratio = float(big[1].value / small[1].value)
                    if ratio < 1.5:
                        continue
                    w1, w2 = attr_words(big[0][1]), attr_words(small[0][1])
                    if safe_words(w1) and safe_words(w2) and len(w1.split()) <= 3:
                        text = f"{big[0][0]} has more {w1} than {small[0][0]} has {w2}."
                        if len(w1.split()) > 3:
                            continue
                    else:
                        text = f"{big[0][0]} is higher than {small[0][0]} on this measure."
                    conf = 0.9 if ratio >= 3 else 0.85
                    yield conf, {"type": "compare", "text": text,
                                 "form": {"left": {"subject": big[0][0], "attribute": big[0][1]},
                                          "right": {"subject": small[0][0], "attribute": small[0][1]}, "op": "gt"},
                                 "key": (big[0], small[0])}

    def relations(self):
        for (subj, link), tl in self.timelines.items():
            if tl.multi or not tl.runs:
                continue
            for run in tl.runs:
                v = run.values[0]
                if v["kind"] != "name":
                    continue
                target = self.lex_names.get(v["value"].lower())
                if not target or target == subj:
                    continue
                for attr in self.attrs.get(target, []):
                    ttl = self.timelines.get((target, attr))
                    if not ttl or ttl.multi or len(ttl.runs) != 1:
                        continue  # only target facts that never changed
                    tv = ttl.runs[0].values[0]
                    r = render(tv)
                    if not r or tv["kind"] == "boolean":
                        continue
                    lw = attr_words(link)
                    if not safe_words(lw):
                        continue
                    aw = attr_words(attr)
                    text = (f"For the {lw} of {subj}, the {aw} is {r}." if safe_words(aw)
                            else f"The {lw} of {subj}: {r}.")
                    if tv["kind"] == "name" and tv["value"].lower() not in self.lex_names:
                        continue
                    yield 0.9, {"type": "relation", "text": text,
                                "form": {"subject": subj, "link": link, "target": target, "attribute": attr, "value": tv},
                                "key": (subj, link, attr)}

    def event_dates(self):
        out = {}
        for ev in self.events:
            tl = self.timelines.get((ev, "date"))
            if tl and not tl.multi and tl.runs:
                v = tl.runs[-1].values[0]
                if len(v["value"]) == 10:
                    out[ev] = _d(v["value"])
        return out

    def at_events(self):
        evd = self.event_dates()
        for key, tl in self.timelines.items():
            if tl.multi or key[0] in evd:
                continue
            for ev, d in evd.items():
                for run in tl.runs:
                    if run.start <= d <= run.end:
                        v = run.values[0]
                        r = render(v)
                        if not r or v["kind"] == "boolean":
                            continue
                        if v["kind"] == "name" and v["value"].lower() not in self.lex_names:
                            continue
                        w = attr_words(key[1])
                        text = (f"At the time of {ev}, the {w} of {key[0]} was {r}." if safe_words(w)
                                else f"At the time of {ev}, {key[0]} stood at {r}.")
                        yield 0.92, {"type": "at_event", "text": text,
                                     "form": {"subject": key[0], "attribute": key[1], "event": ev, "value": v},
                                     "key": key}

    def facts(self):
        for key in self.timelines:
            got = self.stable_now(key)
            if not got:
                continue
            v, why = got
            r = render(v)
            if not r or v["kind"] == "boolean":
                continue
            if v["kind"] == "name" and v["value"].lower() not in self.lex_names:
                continue
            w = attr_words(key[1])
            text = f"The {w} of {key[0]} is {r}." if safe_words(w) else f"{key[0]}: {r}."
            conf = 0.93 if why == "past date" else 0.85 if why == "seen recently" else 0.8
            yield conf, {"type": "fact", "text": text, "form": {"subject": key[0], "attribute": key[1], "value": v},
                         "key": key}


def _form_key(c):
    return repr(sorted((k, repr(v)) for k, v in c["form"].items()))


def _candidates(writer):
    cands = []
    for gen in (writer.changes, writer.date_compares, writer.relations, writer.at_events,
                writer.amount_compares, writer.facts):
        cands += list(gen())
    cands.sort(key=lambda x: -x[0])
    return cands


def generate_claims(rows, lexicon, names, model=None, n=1000, deadline=None, per_key=6, report=None,
                    extra_true=frozenset(), progress=None):
    ref = Referee(lexicon, names, [r.text for r in rows])
    out, seen, per = [], set(), Counter()
    rejected, tiers = Counter(), Counter()
    tier_sources = [("labelled", _candidates(Writer(rows, lexicon, names)), 1.0)]
    if extra_true:
        tier_sources.append(("system-labelled", _candidates(Writer(rows, lexicon, names, extra_true)), 0.9))
    if progress:
        progress("writer", "complete", {
            "total_candidates": sum(len(cands) for _, cands, _ in tier_sources),
            "tiers": {tier: len(cands) for tier, cands, _ in tier_sources},
            "preview_limit": 20,
            "candidates": [{"type": c["type"], "text": c["text"], "form": c["form"],
                            "confidence": round(conf * scale, 3)}
                           for _, cands, scale in tier_sources for conf, c in cands[:20]][:20],
        })
        progress("referee", "running")
    # Fill with the safest claims first; relax the per-fact cap before using weaker evidence.
    for tier, cands, scale in tier_sources:
        for cap in (per_key, 10, 20):
            for conf, c in cands:
                if len(out) >= n:
                    break
                fk = _form_key(c)
                if fk in seen:
                    continue
                ks = c["key"] if isinstance(c["key"], tuple) and c["key"] and isinstance(c["key"][0], tuple) else (c["key"],)
                if any(per[k] >= cap for k in ks):
                    continue
                ok, why = ref.ok(c)
                if not ok:
                    rejected[why.split(":")[0]] += 1
                    seen.add(fk)
                    continue
                seen.add(fk)
                for k in ks:
                    per[k] += 1
                out.append({**c, "confidence": round(conf * scale, 3)})
                tiers[f"{tier} (cap {cap})"] += 1
    if report is not None:
        report.update({"kept": len(out), "rejected": dict(rejected), "tiers": dict(tiers),
                       "by_type": dict(Counter(c["type"] for c in out))})
    for i, c in enumerate(out, 1):
        c["id"] = f"claim-{i:04d}"
        c.pop("key", None)
    if progress:
        progress("referee", "complete", {
            "kept": len(out), "rejected": dict(rejected), "tiers": dict(tiers),
            "by_type": dict(Counter(c["type"] for c in out)),
            "preview_limit": 20, "accepted": out[:20],
        })
    return out
