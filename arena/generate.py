"""Claim generation: enumerate true claims from the evidence store, keep those the local referee reads."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date as _date

from arena.claimforms import check_claim
from arena.evidence import EvidenceStore, Interval
from arena.lexicon import Lexicon, NameIndex, attribute_words
from arena.model import AS_OF
from arena.values import compare, normalize, render

__all__ = ["Candidate", "enumerate_candidates", "select_claims"]

STALE_DAYS = 150
_AMOUNT_KINDS = ("quantity", "duration", "money", "percent", "version")
_COMPARE_WORDS = {
    "amounts": {"gt": "higher than", "lt": "lower than", "eq": "equal to"},
    "dates": {"gt": "later than", "lt": "earlier than", "eq": "on the same day as"},
}
_FLIP = {"gt": "lt", "lt": "gt", "eq": "eq"}


@dataclass
class Candidate:
    type: str
    form: dict
    text: str
    score: float
    subject: str


def _days_to_as_of(d: str) -> int:
    try:
        return (_date.fromisoformat(AS_OF) - _date.fromisoformat(d)).days
    except ValueError:
        return 10**6


def _strong(iv: Interval, pages: int) -> bool:
    return iv.trusted or len(iv.pages) >= pages


def _current_ok(iv: Interval | None) -> bool:
    return iv is not None and _strong(iv, 3) and _days_to_as_of(iv.end) <= STALE_DAYS


def _score(base: float, ivs: list[Interval]) -> float:
    pages = min(len(iv.pages) for iv in ivs)
    trust = 1.0 if all(iv.trusted for iv in ivs) else 0.9
    return base * min(1.0, 0.6 + 0.1 * pages) * trust


def _normalise_text(t: str) -> str:
    return " ".join(t.lower().split()).rstrip(".!? ")


def _day(v: dict) -> bool:
    return v.get("kind") == "date" and len(str(v.get("value"))) == 10


def _margin_ok(a: dict, b: dict, c: int) -> bool:
    if c == 0:
        return True
    if a["kind"] == "version":
        return False
    try:
        x, y = float(normalize(a)["value"]), float(normalize(b)["value"])
    except (KeyError, TypeError, ValueError):
        return False
    top = max(abs(x), abs(y))
    return top > 0 and abs(x - y) / top >= 0.10


def _valid(cand: Candidate, lexicon: Lexicon, kinds) -> bool:
    entry = {"id": "c-0000", "type": cand.type, "text": cand.text, "form": cand.form, "tag": "true", "confidence": 0.9}
    return check_claim(entry, lexicon, kinds=kinds) == []


def _facts(store, lexicon, names, out):
    for s, a in store.keys():
        iv = store.current(s, a)
        if not _current_ok(iv):
            continue
        v = iv.value
        kind = v.get("kind")
        if kind == "name" and not (lexicon.canonical(str(v["value"])) or str(v["value"]) in names.names):
            continue
        A = attribute_words(a)
        if kind == "boolean":
            text = f"{s} has {A}." if v["value"] else f"{s} does not have {A}."
            base = 0.75
        else:
            text = f"The {A} of {s} is {render(v)}."
            base = 0.80
        form = {"subject": s, "attribute": a, "value": v}
        out.append(Candidate("fact", form, text, _score(base, [iv]), s))


def _changes(store, lexicon, out):
    for s, a in store.keys():
        ivs = store.intervals(s, a)
        A = attribute_words(a)
        for p, q in zip(ivs, ivs[1:]):
            if p.value.get("kind") == "boolean" or not (_strong(p, 2) and _strong(q, 2)):
                continue
            text = f"The {A} of {s} went from {render(p.value)} to {render(q.value)}."
            form = {"subject": s, "attribute": a, "from": p.value, "to": q.value}
            out.append(Candidate("change", form, text, _score(0.90, [p, q]), s))


def _compares(store, out):
    by_attr: dict[str, list[tuple[str, Interval]]] = {}
    for s, a in store.keys():
        iv = store.current(s, a)
        if _current_ok(iv) and iv.value.get("kind") in _AMOUNT_KINDS + ("date",):
            by_attr.setdefault(a, []).append((s, iv))
    for a, items in by_attr.items():
        A = attribute_words(a)
        for i, (s1, iv1) in enumerate(items):
            for s2, iv2 in items[i + 1:]:
                if s1 == s2:
                    continue
                c = compare(iv1.value, iv2.value)
                if c is None:
                    continue
                if c < 0:  # one orientation per pair: the larger or later fact on the left
                    (s1, iv1), (s2, iv2), c = (s2, iv2), (s1, iv1), -c
                if iv1.value["kind"] == "date":
                    group = "dates"
                    if c == 0 and not (_day(iv1.value) and _day(iv2.value)):
                        continue
                else:
                    group = "amounts"
                    if not _margin_ok(iv1.value, iv2.value, c):
                        continue
                op = "eq" if c == 0 else "gt"
                text = f"The {A} of {s1} is {_COMPARE_WORDS[group][op]} the {A} of {s2}."
                form = {"left": {"subject": s1, "attribute": a}, "right": {"subject": s2, "attribute": a}, "op": op}
                out.append(Candidate("compare", form, text, _score(0.85, [iv1, iv2]), s1))


def _at_events(store, lexicon, out):
    keys = store.keys()
    events = set(lexicon.events)
    for e in lexicon.events:
        eiv = store.current(e, "date")
        if eiv is None or not _day(eiv.value) or not _strong(eiv, 2):
            continue
        d = str(eiv.value["value"])
        for s, a in keys:
            if s in events:
                continue
            for iv in store.intervals(s, a):
                if not (iv.start <= d <= iv.end) or iv.value.get("kind") == "boolean" or not _strong(iv, 2):
                    continue
                text = f"The {attribute_words(a)} of {s} was {render(iv.value)} at {e}."
                form = {"subject": s, "attribute": a, "event": e, "value": iv.value}
                out.append(Candidate("at_event", form, text, _score(0.85, [iv, eiv]), s))


def _overlap(a: Interval, a_last: bool, b: Interval, b_last: bool) -> bool:
    """Observation spans overlap; a latest interval is taken to hold on (a value stays until seen to change)."""
    end_a, end_b = ("9999" if a_last else a.end), ("9999" if b_last else b.end)
    return a.start <= end_b and b.start <= end_a


def _relations(store, lexicon, names, out):
    keys = store.keys()
    attrs_of: dict[str, list[str]] = {}
    for s, a in keys:
        attrs_of.setdefault(s, []).append(a)
    for s, link in keys:
        links = store.intervals(s, link)
        for li, liv in enumerate(links):
            if liv.value.get("kind") != "name" or not _strong(liv, 2):
                continue
            raw = str(liv.value["value"])
            target = lexicon.canonical(raw) or names.canonical(raw)
            if target == s or target not in attrs_of:
                continue
            for a in attrs_of[target]:
                if a not in lexicon.attributes(target):
                    continue
                tivs = store.intervals(target, a)
                for ti, iv in enumerate(tivs):
                    if iv.value.get("kind") == "boolean" or not _strong(iv, 2):
                        continue
                    if not _overlap(iv, ti == len(tivs) - 1, liv, li == len(links) - 1):
                        continue
                    text = f"The {attribute_words(a)} of the {attribute_words(link)} of {s} was {render(iv.value)}."
                    form = {"subject": s, "link": link, "target": target, "attribute": a, "value": iv.value}
                    out.append(Candidate("relation", form, text, _score(0.85, [liv, iv]), s))


def enumerate_candidates(store: EvidenceStore, lexicon: Lexicon, kinds, names: NameIndex) -> list[Candidate]:
    raw: list[Candidate] = []
    _facts(store, lexicon, names, raw)
    _changes(store, lexicon, raw)
    _compares(store, raw)
    _at_events(store, lexicon, raw)
    _relations(store, lexicon, names, raw)
    return [c for c in raw if len(c.text) <= 300 and _valid(c, lexicon, kinds)]


def _form_key(c: Candidate) -> str:
    f = c.form
    if c.type == "compare":
        l, r, op = (f["left"]["subject"], f["left"]["attribute"]), (f["right"]["subject"], f["right"]["attribute"]), f["op"]
        if r < l:
            l, r, op = r, l, _FLIP[op]
        return json.dumps(["compare", l, r, op])
    return json.dumps([c.type, f], sort_keys=True, ensure_ascii=False)


def select_claims(
    cands: list[Candidate],
    lexicon: Lexicon,
    input_texts: set[str],
    n: int = 1000,
    per_subject_cap: int = 80,
    kinds=None,
) -> list[dict]:
    ordered = sorted(cands, key=lambda c: (-c.score, c.type, c.text))
    seen_forms: set[str] = set()
    seen_texts: set[str] = set()
    per_subject: dict[str, int] = {}
    out: list[dict] = []
    for c in ordered:
        if len(out) >= n:
            break
        norm = _normalise_text(c.text)
        key = _form_key(c)
        if norm in input_texts or norm in seen_texts or key in seen_forms:
            continue
        if per_subject.get(c.subject, 0) >= per_subject_cap:
            continue
        entry = {
            "id": f"c-{len(out) + 1:04d}",
            "type": c.type,
            "text": c.text,
            "form": c.form,
            "confidence": round(min(0.99, max(0.5, c.score)), 4),
        }
        if check_claim({**entry, "tag": "true"}, lexicon, kinds=kinds):
            continue
        seen_forms.add(key)
        seen_texts.add(norm)
        per_subject[c.subject] = per_subject.get(c.subject, 0) + 1
        out.append(entry)
    return out
