"""Fold-scoped typed claim references for retrieval classifiers.

Candidates in this module are statistical references only. They are never inserted
into a world evidence store and never establish a fact about a query row.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from datetime import date
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from arena.evidence import EvidenceStore
from arena.generate import enumerate_candidates, select_claims
from arena.kinds import KindIndex
from arena.lexicon import Lexicon, NameIndex, attribute_words
from arena.model import AS_OF, Fact, Row
from arena.values import compare, render, values_equal

TYPES = ("fact", "compare", "change", "at_event", "relation")
_NUM_RE = re.compile(r"(?<!\w)(?:[$€£]\s*)?\d[\d,.]*(?:\s*(?:%|percent|USD|EUR|GBP|dollars?|euros?|pounds?))?", re.I)
_SCOPE_CUE_RE = re.compile(r"\b(?:last quarter|previous quarter|last year|previously|formerly|historically|planned|plan(?:ned)? to|target(?:ed)?|forecast(?:ed)?|projected|will|next quarter|next year|by 2027)\b", re.I)
_NAMES: dict[str, str] = {}


def _date(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _fingerprint(row: dict) -> str:
    payload = "\0".join(str(row.get(k, "")) for k in ("id", "text", "date"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _value(form: dict) -> dict | None:
    if isinstance(form.get("value"), dict):
        return form["value"]
    return None


def _compatible(a: dict | None, b: dict | None) -> bool:
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    if a.get("kind") != b.get("kind"):
        return False
    # values_equal normalizes compatible physical units and money periods/currency.
    # For non-equal values, compare() is safe only when it recognizes the same domain.
    try:
        return values_equal(a, b) or compare(a, b) is not None
    except (KeyError, TypeError, ValueError):
        return False


def _masked(text: str) -> str:
    return _NUM_RE.sub(" NUM ", text)


class ClaimFeatureIndex:
    """Build typed candidate references from explicitly supplied labelled rows.

    ``observed`` renders atomic facts from trusted true-row annotations at the
    source date. ``generated`` runs the team's candidate generator over those same
    true annotations; generated records remain unverified auxiliary candidates.
    """

    def __init__(self, reference_rows: list[dict], lexicon: dict,
                 mode: str = "observed", limit: int = 1000):
        if mode not in ("observed", "generated"):
            raise ValueError("mode must be observed or generated")
        if limit < 0 or limit > 1000:
            raise ValueError("limit must be between 0 and 1000")
        self.mode = mode
        self.limit = int(limit)
        self.lexicon_data = lexicon
        self.lexicon = self._lexicon(lexicon)
        self.reference_rows = [dict(r) for r in reference_rows]
        ids = [str(r.get("id", "")) for r in self.reference_rows]
        if any(not rid for rid in ids) or len(set(ids)) != len(ids):
            raise ValueError("reference IDs must be nonempty and unique")
        self._rows = {str(r["id"]): r for r in self.reference_rows}
        self._facts: dict[str, list[dict]] = {}
        self._fact_index: dict[tuple[str, str], list[tuple[str, dict]]] = defaultdict(list)
        self._true_rows: list[dict] = []
        self._true_ids: set[str] = set()
        self._load_facts()
        self.records = self._observed_records() if mode == "observed" else self._generated_records()
        self.records = sorted(self.records, key=lambda x: (x["candidate_id"], x["text"]))[:self.limit]
        for rec in self.records:
            rec["_entity_cache"] = sorted(self._query_entities(rec["text"]))
        self._fit()
        self.feature_names_ = self._feature_names()
        self._summary = {
            "mode": self.mode,
            "candidate_count": len(self.records),
            "reference_row_count": len(self.reference_rows),
            "positive_source_row_count": len(self._true_rows),
            "limit": self.limit,
            "candidate_status": "unverified_auxiliary",
            "candidate_type_counts": {typ: sum(r["type"] == typ for r in self.records) for typ in TYPES},
            "source_policy": "only labelled true reference rows contribute positive candidate facts",
            "generator_assumptions": self._assumptions(),
        }

    def summary(self) -> dict:
        return json.loads(json.dumps(self._summary, ensure_ascii=False))

    @staticmethod
    def _lexicon(data: dict) -> Lexicon:
        subjects = data.get("subjects", []) if isinstance(data, dict) else []
        if isinstance(subjects, list):
            smap = {str(s["name"]): list(s.get("attributes", [])) for s in subjects if isinstance(s, dict) and s.get("name")}
            amap = {str(s["name"]): list(s.get("aliases", [])) for s in subjects if isinstance(s, dict) and s.get("name")}
            return Lexicon(smap, amap, list(data.get("events", [])))
        # Compatibility with the older dictionary-shaped lexicon.
        smap = {str(k): list(v.get("attributes", [])) for k, v in data.items() if isinstance(v, dict)}
        amap = {str(k): list(v.get("aliases", [])) for k, v in data.items() if isinstance(v, dict)}
        return Lexicon(smap, amap, [])

    def _load_facts(self) -> None:
        for row in self.reference_rows:
            rid = str(row["id"])
            raw = row.get("form")
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw) if raw else []
                except (ValueError, TypeError):
                    raw = []
            if not isinstance(raw, list):
                raw = []
            facts = []
            for item in raw:
                if not isinstance(item, dict):
                    continue
                canonical = self.lexicon.canonical(str(item.get("subject", "")))
                attr, value = str(item.get("attribute", "")), item.get("value")
                if canonical and attr in self.lexicon.attributes(canonical) and isinstance(value, dict) and value.get("kind"):
                    facts.append({**item, "subject": canonical})
            self._facts[rid] = facts
            for fact in facts:
                self._fact_index[(fact["subject"], fact["attribute"])].append((rid, fact))
            if row.get("label") == "true":
                self._true_rows.append(row)
                self._true_ids.add(rid)

    def _record(self, *, typ: str, text: str, form: dict, sources: list[str], score: float = 0.0,
                scope_uncertain: bool = False, assumptions: list[str] | None = None,
                temporal_kind: str = "source_exact") -> dict:
        sources = sorted(set(sources))
        source_rows = [self._rows[rid] for rid in sources]
        return {
            "candidate_id": "claim-" + hashlib.sha256(json.dumps([typ, form, sources], sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16],
            "type": typ, "text": text, "form": form,
            "source_ids": sources,
            "source_fingerprints": {str(r["id"]): _fingerprint(r) for r in source_rows},
            "source_dates": sorted({str(r.get("date", "")) for r in source_rows}),
            "status": "unverified_auxiliary",
            "generator_score": float(score),
            "scope_uncertain": bool(scope_uncertain),
            "assumptions": sorted(set(assumptions or [])),
            "temporal_kind": temporal_kind,
        }

    def _observed_records(self) -> list[dict]:
        grouped: dict[str, dict] = {}
        for row in self._true_rows:
            rid, day = str(row["id"]), str(row.get("date", ""))
            if _date(day) is None:
                continue
            for fact in self._facts[rid]:
                v = _value(fact)
                if not v or v.get("op") not in (None, "eq"):
                    continue
                subject = self.lexicon.canonical(str(fact.get("subject", ""))) or str(fact.get("subject", ""))
                attr = str(fact.get("attribute", ""))
                if not subject or not attr:
                    continue
                form = {"subject": subject, "attribute": attr, "value": v}
                key = json.dumps([subject, attr, v, day], sort_keys=True, ensure_ascii=False)
                rec = grouped.get(key)
                if rec is None:
                    valtext = render(v)
                    text = (f"{subject} has {attribute_words(attr)}." if v.get("kind") == "boolean" and v.get("value")
                            else f"{subject} does not have {attribute_words(attr)}." if v.get("kind") == "boolean"
                            else f"The {attribute_words(attr)} of {subject} is {valtext}.")
                    rec = grouped[key] = {"type": "fact", "text": text, "form": form, "source_ids": [],
                                          "source_fingerprints": {}, "source_dates": [], "status": "unverified_auxiliary",
                                          "generator_score": 0.0, "scope_uncertain": self._historical_scope(str(row.get("text", ""))),
                                          "assumptions": [], "temporal_kind": "source_exact"}
                rec["source_ids"].append(rid)
                rec["source_fingerprints"][rid] = _fingerprint(row)
                rec["scope_uncertain"] = rec["scope_uncertain"] or self._historical_scope(str(row.get("text", "")))
        for rec in grouped.values():
            rec["source_ids"] = sorted(set(rec["source_ids"]))
            rec["source_dates"] = sorted({str(self._rows[rid].get("date", "")) for rid in rec["source_ids"]})
            rec["candidate_id"] = "claim-" + hashlib.sha256(json.dumps(
                [rec["type"], rec["form"], rec["source_dates"], rec["source_ids"]],
                sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
        return list(grouped.values())

    def _generated_records(self) -> list[dict]:
        true_rows = self._true_rows
        gold: dict[str, list[Fact]] = {}
        rows: list[Row] = []
        for page, row in enumerate(sorted(true_rows, key=lambda x: (str(x.get("date", "")), str(x["id"])) )):
            rid = str(row["id"])
            gold[rid] = [Fact(str(f["subject"]), str(f["attribute"]), dict(f["value"]))
                         for f in self._facts[rid] if isinstance(f.get("value"), dict)]
            rows.append(Row(rid, str(row.get("text", "")), str(row.get("date", "")), "true", page))
        kinds = KindIndex.from_gold(gold, rows)
        values = [str(f.value["value"]) for fs in gold.values() for f in fs if f.value.get("kind") == "name"]
        names = NameIndex.from_values(values, self.lexicon)
        store = EvidenceStore()
        for row in rows:
            store.add_row(row, gold[row.id])
        store.build(iterations=0)
        candidates = enumerate_candidates(store, self.lexicon, kinds, names)
        input_texts = {" ".join(str(r.get("text", "")).lower().split()).rstrip(".!? ") for r in true_rows}
        selected = select_claims(candidates, self.lexicon, input_texts, n=self.limit, kinds=kinds)
        out = []
        for entry in selected:
            groups = self._premise_groups(entry)
            if not groups or any(not group for group in groups):
                continue
            sources = sorted({source_id for group in groups for source_id in group})
            temporal_kind, assumptions = self._candidate_temporal(entry)
            # Candidate text/form stay generator output. EvidenceStore intervals and the
            # generator's as-of rules are documented as assumptions, not proven continuity.
            extra = {}
            if entry["type"] == "at_event":
                event_iv = store.current(entry["form"].get("event"), "date")
                if event_iv and event_iv.value.get("kind") == "date":
                    extra["event_date"] = str(event_iv.value.get("value", ""))
            if entry["type"] == "relation":
                extra["relation_source_date_intersection"] = sorted(set(self._source_dates(groups[0])) & set(self._source_dates(groups[1])))
            out.append(self._record(typ=entry["type"], text=entry["text"], form=entry["form"],
                                    sources=sources, score=float(entry.get("confidence", 0.0)),
                                    scope_uncertain=True, assumptions=assumptions,
                                    temporal_kind=temporal_kind) | extra)
        return out

    def _premise_sources(self, entry: dict) -> tuple[list[str], str, list[str]]:
        groups = self._premise_groups(entry)
        return sorted({rid for group in groups for rid in group}), *self._candidate_temporal(entry)

    @staticmethod
    def _candidate_temporal(entry: dict) -> tuple[str, list[str]]:
        return {
            "fact": ("as_of_candidate", ["latest_observation_as_of_2026-10-29"]),
            "compare": ("as_of_candidate", ["latest_observation_as_of_2026-10-29"]),
            "change": ("observed_transition", ["interval_compaction_used_by_generator"]),
            "at_event": ("event_date_candidate", ["interval_contains_event_date_assumed_by_generator"]),
            "relation": ("relation_overlap_candidate", ["30_day_latest_interval_extension_may_be_used"]),
        }.get(entry["type"], ("unknown", ["unsupported_candidate_type"]))

    def _historical_scope(self, text: str) -> bool:
        # Remove recognized entity surfaces so names containing temporal words do not trigger cues.
        spans = sorted(self.lexicon.mentions(text), key=lambda m: m.start, reverse=True)
        clean = text
        for mention in spans:
            clean = clean[:mention.start] + " " * (mention.end - mention.start) + clean[mention.end:]
        return bool(_SCOPE_CUE_RE.search(clean))

    def _premise_groups(self, entry: dict) -> list[list[str]]:
        typ, form = entry["type"], entry["form"]
        def matches(subject, attr, value=None):
            found = []
            for rid, fact in self._fact_index.get((subject, attr), []):
                if rid in self._true_ids and (value is None or (_compatible(_value(fact), value) and values_equal(_value(fact), value))):
                    found.append(rid)
            return sorted(set(found))
        if typ == "fact": return [matches(form["subject"], form["attribute"], form["value"])]
        if typ == "compare": return [matches(form["left"]["subject"], form["left"]["attribute"]), matches(form["right"]["subject"], form["right"]["attribute"])]
        if typ == "change": return [matches(form["subject"], form["attribute"], form["from"]), matches(form["subject"], form["attribute"], form["to"])]
        if typ == "at_event": return [matches(form["subject"], form["attribute"], form["value"]), matches(form["event"], "date")]
        if typ == "relation": return [matches(form["subject"], form["link"], {"kind": "name", "value": form["target"]}), matches(form["target"], form["attribute"], form["value"])]
        return []

    def _source_dates(self, ids: list[str]) -> list[str]:
        return sorted({str(self._rows[rid].get("date", "")) for rid in ids})

    def _assumptions(self) -> list[str]:
        return [] if self.mode == "observed" else [
            "generator_as_of_date_is_2026-10-29",
            "latest_observation_as_of_candidates_do_not_prove_persistence",
            "change_candidates_depend_on_compacted_observation_intervals",
            "event_and_relation_candidates_use_generator_interval_rules",
        ]

    def _fit(self) -> None:
        self.texts = [_masked(r["text"]) for r in self.records]
        self.vectorizer = None
        self.matrix = None
        if self.texts:
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20000, sublinear_tf=True)
            self.matrix = self.vectorizer.fit_transform(self.texts)

    @staticmethod
    def _feature_names() -> list[str]:
        names = []
        for typ in TYPES:
            names.extend([f"{typ}.max_similarity", f"{typ}.mean_top5_similarity", f"{typ}.candidate_count",
                          f"{typ}.shared_entity", f"{typ}.typed_value_agreement", f"{typ}.source_date_exact",
                          f"{typ}.nearest_source_gap_years", f"{typ}.temporal_compatible",
                          f"{typ}.scope_uncertain"])
        names.append("claim.any_similarity")
        return names

    def _query_entities(self, text: str) -> set[str]:
        return {m.subject for m in self.lexicon.mentions(text)}

    def _typed_agreement(self, text: str, rec: dict, spans: list[dict]) -> bool:
        form = rec["form"]
        subject = form.get("subject") or (form.get("left") or {}).get("subject")
        attribute = form.get("attribute") or (form.get("left") or {}).get("attribute")
        if subject and attribute and not any(m.subject == subject and attribute_words(attribute).casefold() in text.casefold() for m in self.lexicon.mentions(text)):
            return False
        values = []
        if rec["type"] in ("fact", "at_event"):
            values = [form.get("value")]
        elif rec["type"] == "change":
            values = [form.get("from"), form.get("to")]
        elif rec["type"] == "compare":
            # Only compare when a query number can be parsed; avoid cross-kind matches.
            values = []
        return any(_compatible(v, q) and values_equal(v, q) for v in values if v for q in spans)

    def _temporal(self, rec: dict, query_date: str) -> tuple[bool, bool, float]:
        qd = _date(query_date)
        source_dates = [_date(x) for x in rec["source_dates"]]
        source_dates = [d for d in source_dates if d is not None]
        exact = bool(qd and qd in source_dates)
        kind = rec["temporal_kind"]
        if kind == "source_exact":
            compatible = exact
        elif kind in ("as_of_candidate",):
            compatible = bool(qd and query_date == AS_OF)
        elif kind == "event_date_candidate":
            compatible = bool(qd and rec.get("event_date") == query_date)
        elif kind == "relation_overlap_candidate":
            compatible = bool(qd and query_date in rec.get("relation_source_date_intersection", []))
        else:  # transitions are retrieval candidates, never a current-state assertion
            compatible = False
        gap = min((abs((qd - d).days) for d in source_dates), default=730) if qd else 730
        if kind == "source_exact" and rec.get("scope_uncertain"):
            compatible = False
        return exact, compatible, min(gap / 365.25, 2.0)

    def transform(self, queries: list[dict]) -> tuple[np.ndarray, list[dict]]:
        for q in queries:
            if q.get("label") not in (None, ""):
                raise ValueError("query labels must be blank")
            if str(q.get("id", "")) in self._rows:
                raise ValueError("query ID overlaps a reference ID")
        nfeat = len(self.feature_names_)
        X = np.zeros((len(queries), nfeat), dtype=float)
        traces = []
        for qi, query in enumerate(queries):
            text = str(query.get("text", ""))
            from arena.values import parse_values
            parsed_values = [span.value for span in parse_values(text)]
            query_scope_uncertain = self._historical_scope(text)
            sim = (cosine_similarity(self.vectorizer.transform([_masked(text)]), self.matrix).ravel()
                   if self.vectorizer is not None and self.matrix is not None else np.zeros(0))
            entities = self._query_entities(text)
            offset = 0
            trace_neighbors = []
            global_max = 0.0
            for typ in TYPES:
                indices = [i for i, r in enumerate(self.records) if r["type"] == typ]
                scored = sorted(((float(sim[i]), self.records[i]) for i in indices if sim[i] > 0),
                                key=lambda x: (-x[0], x[1]["candidate_id"]))
                sims = [s for s, _ in scored]
                top = scored[:5]
                maxsim = sims[0] if sims else 0.0
                mean5 = float(np.mean([s for s, _ in top])) if top else 0.0
                positive_top = [(s, r) for s, r in top if s > 0.0]
                shared = any(entities.intersection(r["_entity_cache"]) for _, r in positive_top)
                typed = any(self._typed_agreement(text, r, parsed_values) for _, r in positive_top)
                temporal = [self._temporal(r, str(query.get("date", ""))) for _, r in positive_top]
                exact = any(t[0] for t in temporal)
                compat = any(t[1] for t in temporal)
                if query_scope_uncertain and typ == "fact":
                    compat = False
                gap = min((t[2] for t in temporal), default=2.0)
                vals = [maxsim, mean5, float(len(indices)), float(shared), float(typed), float(exact), gap, float(compat),
                        float(query_scope_uncertain or any(r["scope_uncertain"] for _, r in positive_top))]
                X[qi, offset:offset + len(vals)] = vals
                offset += len(vals)
                global_max = max(global_max, maxsim)
                for score, rec in top:
                    _, compatible, _ = self._temporal(rec, str(query.get("date", "")))
                    if query_scope_uncertain and rec["type"] == "fact":
                        compatible = False
                    trace_neighbors.append({"candidate_id": rec["candidate_id"], "type": typ, "similarity": score,
                                           "source_ids": list(rec["source_ids"]), "source_dates": list(rec["source_dates"]),
                                           "temporal_compatible": compatible, "status": rec["status"]})
            X[qi, -1] = global_max
            traces.append({"query_id": str(query.get("id", "")), "neighbors": trace_neighbors})
        return X, traces

    def export_records(self) -> list[dict]:
        public = [{k: v for k, v in record.items() if not k.startswith("_")} for record in self.records]
        return json.loads(json.dumps(public, ensure_ascii=False))
