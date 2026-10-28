"""Conservative, date-scoped numeric evidence store for Hype Arena."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import date as _date
from decimal import Decimal, InvalidOperation


_OUT_OF_SCOPE = re.compile(r"\b(last quarter|up from|from\s+\S+\s+to|used to|previously|formerly|before|after|next|will|planned|expected|target(?:s|ed|ing)?|goal(?:s|ed)?|in the future|since|until|(?:in|during|by|before|after|since|until)\s+(?:19|20)\d{2}|(?:in|during|by|before|after|since|until)\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?))\b", re.I)
_NUMERIC = {"money", "quantity", "duration", "percent"}
_OPS = {"eq", "gt", "ge", "lt", "le"}
_DURATION_UNITS = {"ms", "s", "min", "h", "day", "week", "month", "year"}


def _valid_date(value):
    if not isinstance(value, str):
        return False
    try:
        return _date.fromisoformat(value).isoformat() == value
    except ValueError:
        return False


def _decimal(v):
    try:
        n = Decimal(str(v))
        return n if n.is_finite() else None
    except (InvalidOperation, ValueError, TypeError):
        return None


def _read_form(form):
    if isinstance(form, str):
        try:
            form = json.loads(form)
        except (ValueError, TypeError):
            return []
    if isinstance(form, dict):
        return [form]
    return form if isinstance(form, list) else []


def _scope(a):
    """Only dimensions that can safely be compared; never convert currencies."""
    kind = a.get("kind")
    if kind == "money":
        if not isinstance(a.get("currency"), str) or not re.fullmatch(r"[A-Z]{3}", a["currency"]):
            return None
        if a.get("per") not in {None, "day", "week", "month", "year"}:
            return None
        return (kind, a.get("currency"), a.get("per"))
    if kind in {"quantity", "duration"}:
        unit = a.get("unit")
        if not isinstance(unit, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,39}", unit):
            return None
        if kind == "quantity" and unit in _DURATION_UNITS:
            return None
        if kind == "duration" and unit not in _DURATION_UNITS:
            return None
        # Conservative conversions for common duration units.
        if kind == "duration" and unit in {"s", "min", "h", "day", "week"}:
            factors = {"s": 1, "min": 60, "h": 3600, "day": 86400, "week": 604800}
            return (kind, "seconds")
        return (kind, unit, 1)
    if kind == "percent":
        return (kind,)
    return None


def _interval(a):
    n = _decimal(a.get("value"))
    op = a.get("op", "eq")
    if n is None or op not in _OPS:
        return None
    scope = _scope(a)
    if scope is None:
        return None
    if a.get("kind") in {"money", "quantity", "duration"} and n < 0:
        return None
    factors = {"s": 1, "min": 60, "h": 3600, "day": 86400, "week": 604800}
    factor = factors.get(a.get("unit"), 1) if a.get("kind") == "duration" else 1
    n *= factor
    if op == "eq": return (n, True, n, True)
    if op == "gt": return (n, False, None, False)
    if op == "ge": return (n, True, None, False)
    if op == "lt": return (None, False, n, False)
    return (None, False, n, True)


def _intersect(x, y):
    lo, li, hi, hii = x
    ylo, yli, yhi, yhii = y
    if ylo is not None and (lo is None or ylo > lo):
        lo, li = ylo, yli
    elif ylo == lo and ylo is not None:
        li = li and yli
    if yhi is not None and (hi is None or yhi < hi):
        hi, hii = yhi, yhii
    elif yhi == hi and yhi is not None:
        hii = hii and yhii
    if lo is not None and hi is not None and (lo > hi or (lo == hi and not (li and hii))):
        return None
    return (lo, li, hi, hii)


def _disjoint(x, y):
    return _intersect(x, y) is None


class EvidenceStore:
    def __init__(self, training_rows: list[dict], lexicon: dict):
        self.lexicon = lexicon or {}
        self.allowed = {s["name"]: set(s.get("attributes", [])) for s in self.lexicon.get("subjects", []) if isinstance(s, dict) and s.get("name")}
        self.records = []
        self._facts = {}
        for row in training_rows:
            label = str(row.get("label", "")).lower()
            if label not in {"true", "false"}:
                continue
            source = {k: row.get(k) for k in ("id", "text", "date")}
            fingerprint = hashlib.sha256("\0".join(str(source[k] or "") for k in ("id", "text", "date")).encode()).hexdigest()
            annotated_atoms = _read_form(row.get("form"))
            atoms = []
            for raw in annotated_atoms:
                if not isinstance(raw, dict) or raw.get("subject") not in self.allowed or raw.get("attribute") not in self.allowed.get(raw.get("subject"), set()):
                    continue
                val = raw.get("value")
                if not isinstance(val, dict) or val.get("kind") not in _NUMERIC or _interval(val) is None:
                    continue
                atom = {"subject": raw["subject"], "attribute": raw["attribute"], "value": val}
                atoms.append(atom)
            blocked = self._out_of_scope(str(row.get("text", "")))
            rec = {**source, "label": label, "fingerprint": fingerprint, "form": row.get("form"), "atoms": atoms, "scope_blocked": blocked}
            self.records.append(rec)
            if blocked or not _valid_date(row.get("date")):
                continue
            # A false compound means at least one atom fails; it cannot refute each atom.
            indexed_atoms = atoms if label == "true" or len(annotated_atoms) == 1 else []
            for atom in indexed_atoms:
                key = (atom["subject"], atom["attribute"], str(row["date"]))
                self._facts.setdefault(key, []).append({"atom": atom, "label": label, "record": rec})

    def query(self, atom: dict, date: str) -> dict:
        if not _valid_date(date):
            return {"label": "unsure", "reason": "unsupported", "evidence": []}
        subject, attribute = atom.get("subject"), atom.get("attribute")
        if subject not in self.allowed or attribute not in self.allowed.get(subject, set()):
            return {"label": "unsure", "reason": "unsupported", "evidence": []}
        value = atom.get("value")
        target = _interval(value) if isinstance(value, dict) and value.get("kind") in _NUMERIC else None
        if target is None:
            return {"label": "unsure", "reason": "unsupported", "evidence": []}
        refs = self._facts.get((subject, attribute, str(date)), [])
        positives = [r for r in refs if r["label"] == "true" and _scope(r["atom"]["value"]) == _scope(value)]
        intervals = [_interval(r["atom"]["value"]) for r in positives]
        # Mutually impossible positive observations are preserved as a conflict.
        if any(_disjoint(a, b) for i, a in enumerate(intervals) for b in intervals[i + 1:]):
            return {"label": "unsure", "reason": "temporal_conflict", "evidence": [self._provenance(r["record"]) for r in positives]}
        support = [r for r in positives if _intersect(_interval(r["atom"]["value"]), target) == _interval(r["atom"]["value"])]
        refute = [r for r in positives if _disjoint(_interval(r["atom"]["value"]), target)]
        # A false row only excludes its identical predicate; never infer an alternative.
        negatives = [r for r in refs if r["label"] == "false" and _same_predicate(r["atom"]["value"], value)]
        if support and negatives:
            involved = support + negatives
            return {"label": "unsure", "reason": "temporal_conflict", "evidence": [self._provenance(r["record"]) for r in involved]}
        if support and refute:
            return {"label": "unsure", "reason": "temporal_conflict", "evidence": [self._provenance(r["record"]) for r in support + refute]}
        if support:
            return {"label": "true", "reason": "decided", "evidence": [self._provenance(r["record"]) for r in support]}
        if refute or negatives:
            chosen = refute + negatives
            return {"label": "false", "reason": "decided", "evidence": [self._provenance(r["record"]) for r in chosen]}
        return {"label": "unsure", "reason": "missing_evidence", "evidence": []}

    @staticmethod
    def _provenance(record):
        return {"id": record["id"], "date": record["date"], "fingerprint": record["fingerprint"]}

    def _out_of_scope(self, text):
        # Entity names such as a product called "Next" are not temporal cues.
        masked = text
        aliases = []
        for subject in self.lexicon.get("subjects", []):
            if isinstance(subject, dict) and subject.get("name"):
                aliases.extend([subject["name"], *subject.get("aliases", [])])
        for alias in sorted((a for a in aliases if isinstance(a, str) and a), key=len, reverse=True):
            masked = re.sub(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", " ", masked, flags=re.I)
        return bool(_OUT_OF_SCOPE.search(masked))

    def summary(self):
        indexed = sum(1 for records in self._facts.values() for r in records if r["label"] == "true" or len(r["record"]["atoms"]) == 1)
        return {"source_rows": len(self.records), "numeric_atoms": sum(len(r["atoms"]) for r in self.records), "indexed_atoms": indexed, "constraints": sum(1 for r in self.records if r["label"] == "false" and len(_read_form(r["form"])) > 1), "scope_blocked_rows": sum(r["scope_blocked"] for r in self.records), "labels": {k: sum(r["label"] == k for r in self.records) for k in ("true", "false")}}

    def export_records(self):
        return json.loads(json.dumps(self.records))


def _same_predicate(a, b):
    """Compare predicate identity after canonical Decimal and unit normalization."""
    if _scope(a) != _scope(b):
        return False
    ia, ib = _interval(a), _interval(b)
    return ia is not None and ia == ib
