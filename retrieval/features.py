"""Split-scoped, non-probabilistic nearest-neighbour features."""
from __future__ import annotations

import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

LABELS = ("true", "false", "unsure")
NUM_RE = re.compile(r"(?<!\w)(?:[$€£]\s*)?\d[\d,]*(?:\.\d+)?\s*(?:%|percent|USD|EUR|GBP|dollars?|euros?|pounds?)?", re.I)
CURRENCY_RE = re.compile(r"\b(?:USD|EUR|GBP|dollars?|euros?|pounds?)\b|[$€£]", re.I)
PERIOD_RE = re.compile(r"(?:\bper\s+(?:month|year|week|user)\b|\b(?:monthly|annually|yearly|weekly)\b|/\s*(?:mo|yr|month|year)\b)", re.I)
DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")


def _date(value: Any):
    try:
        d = np.datetime64(str(value)[:10], "D")
        return None if np.isnat(d) else d
    except (ValueError, TypeError):
        return None


def _entity_terms(lexicon: dict):
    terms = []
    canonical = {}
    subjects = lexicon.get("subjects", []) if isinstance(lexicon, dict) else []
    if isinstance(subjects, list):
        for subject in subjects:
            if isinstance(subject, dict):
                name = str(subject.get("name", ""))
                aliases = [str(a) for a in subject.get("aliases", []) if isinstance(a, (str, int, float))]
                for term in [name, *aliases]:
                    if term:
                        terms.append(term)
                        canonical[term.casefold()] = name.casefold()
        if terms:
            return sorted({x for x in terms if x}, key=lambda x: (-len(x), x.casefold())), canonical
    for key, value in lexicon.items():
        if isinstance(value, dict):
            terms.append(str(value.get("name", key)))
            aliases = value.get("aliases", [])
            terms.extend(str(a) for a in aliases if isinstance(a, (str, int, float)))
        elif isinstance(value, list):
            terms.append(str(key))
            terms.extend(str(a) for a in value if isinstance(a, (str, int, float)))
        else:
            terms.append(str(key))
    normalized = sorted({x for x in terms if x}, key=lambda x: (-len(x), x.casefold()))
    return normalized, {x.casefold(): x.casefold() for x in normalized}


def _entities(text: str, entity_terms) -> frozenset[str]:
    terms, canonical = entity_terms
    # Longest aliases win: suppress any shorter hit contained in a longer matched span.
    retained = []
    spans = []
    for term in terms:
        for match in re.finditer(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text, re.I):
            if not any(a <= match.start() and match.end() <= b for a, b in spans):
                spans.append(match.span())
                retained.append(canonical.get(term.casefold(), term.casefold()))
    return frozenset(retained)


def _numeric(text: str):
    raw = [m.group(0) for m in NUM_RE.finditer(text)]
    vals = []
    for value in raw:
        s = re.sub(r"[^\d.]", "", value).replace(",", "")
        try:
            v = Decimal(s)
            if v.is_finite(): vals.append(v)
        except (InvalidOperation, ValueError):
            pass
    return raw, vals


def _currency(text: str) -> frozenset[str]:
    result = set()
    for m in CURRENCY_RE.finditer(text):
        token = m.group(0).casefold()
        result.add({"$": "usd", "€": "eur", "£": "gbp", "dollar": "usd", "dollars": "usd",
                    "euro": "eur", "euros": "eur", "pound": "gbp", "pounds": "gbp"}.get(token, token))
    return frozenset(result)


def _period(text: str) -> frozenset[str]:
    value = text.casefold()
    result = set()
    aliases = {"month": "month", "monthly": "month", "mo": "month", "year": "year",
               "yearly": "year", "annually": "year", "yr": "year", "week": "week", "weekly": "week", "user": "user"}
    for m in PERIOD_RE.finditer(value):
        words = re.findall(r"[a-z]+", m.group(0))
        for word in words:
            if word in aliases: result.add(aliases[word])
    return frozenset(result)


def _overlap(a, b):
    """Jaccard overlap for parsed number sets; a heuristic, with no unit inference."""
    return len(set(a) & set(b)) / len(set(a) | set(b)) if a and b else 0.0


class RetrievalIndex:
    """TF-IDF retrieval features over only the explicitly supplied references.

    Similarities and agreement indicators are descriptive features, never truth proofs.
    """

    def __init__(self, reference_rows: list[dict], lexicon: dict,
                 views: tuple[str, ...] = ("raw",), k: int = 5):
        if any(v not in ("raw", "masked") for v in views) or not views:
            raise ValueError("views must contain raw and/or masked")
        if k < 1:
            raise ValueError("k must be positive")
        self.views, self.k = tuple(views), int(k)
        self.lexicon = lexicon
        self.entity_terms = _entity_terms(lexicon)
        self.reference_rows = list(reference_rows)
        ids = [str(r.get("id", "")) for r in self.reference_rows]
        if any(not x for x in ids) or len(set(ids)) != len(ids):
            raise ValueError("reference IDs must be nonempty and unique")
        if any(r.get("label") not in LABELS for r in self.reference_rows):
            raise ValueError("reference labels must be true, false, or unsure")
        self.ids = ids
        self._meta = []
        for r in self.reference_rows:
            rt = str(r.get("text", ""))
            _, vals = _numeric(rt)
            self._meta.append({"date": _date(r.get("date")), "entities": _entities(rt, self.entity_terms),
                               "numbers": vals, "currency": _currency(rt), "period": _period(rt)})
        self._refs = {v: [self._view(str(r.get("text", "")), v) for r in self.reference_rows] for v in self.views}
        self._vectorizers, self._matrices = {}, {}
        for view in self.views:
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20000,
                                          sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")
            try:
                matrix = vectorizer.fit_transform(self._refs[view])
            except ValueError:  # empty vocabulary
                vectorizer, matrix = None, None
            self._vectorizers[view], self._matrices[view] = vectorizer, matrix
        self.feature_names_ = self._make_feature_names()

    @staticmethod
    def _view(text: str, view: str) -> str:
        return NUM_RE.sub(" NUM ", text) if view == "masked" else text

    def _make_feature_names(self):
        names = []
        base = [f"max_sim_{l}" for l in LABELS] + [f"mean_topk_sim_{l}" for l in LABELS]
        base += [f"weighted_share_{l}" for l in LABELS]
        base += [f"nearest_dategap_signed_{l}" for l in LABELS] + [f"nearest_dategap_abs_{l}" for l in LABELS]
        base += [f"nearest_same_date_{l}" for l in LABELS] + [f"nearest_shared_entity_{l}" for l in LABELS]
        base += [f"nearest_numeric_equal_{l}" for l in LABELS] + [f"nearest_numeric_overlap_{l}" for l in LABELS]
        base += [f"nearest_currency_match_{l}" for l in LABELS] + [f"nearest_period_match_{l}" for l in LABELS]
        base += ["neighbor_similarity", "neighbor_label_entropy", "neighbor_disagreement",
                 "same_entity_neighbor_share", "same_date_neighbor_share"]
        for view in self.views:
            names.extend(f"{view}_{name}" for name in base)
        return names

    def summary(self):
        return {"reference_count": len(self.reference_rows), "views": list(self.views), "k": self.k,
                "feature_names": list(self.feature_names_), "labels": list(LABELS),
                "similarity_is_probability": False, "numeric_agreement_is_proof": False}

    def transform(self, query_rows: list[dict]):
        if set(self.ids) & {str(r.get("id", "")) for r in query_rows}:
            raise ValueError("query IDs must be disjoint from reference IDs")
        output, traces = [], []
        for query in query_rows:
            all_feats, view_trace = [], {}
            text = str(query.get("text", ""))
            q_entities = _entities(text, self.entity_terms)
            _, q_nums = _numeric(text)
            q_currency, q_period = _currency(text), _period(text)
            for view in self.views:
                vectorizer, matrix = self._vectorizers[view], self._matrices[view]
                sims = np.zeros(len(self.reference_rows), dtype=float)
                if vectorizer is not None and len(self.reference_rows):
                    qvec = vectorizer.transform([self._view(text, view)])
                    if qvec.nnz:
                        sims = cosine_similarity(qvec, matrix).ravel()
                order = sorted((i for i, s in enumerate(sims) if s > 0), key=lambda i: (-sims[i], self.ids[i]))[:self.k]
                features = {name: 0.0 for name in self._make_feature_names() if name.startswith(view + "_")}
                class_ix = {l: [i for i, r in enumerate(self.reference_rows) if r["label"] == l] for l in LABELS}
                class_neighbors = {}
                for label in LABELS:
                    ix = class_ix[label]
                    features[f"{view}_max_sim_{label}"] = max((float(sims[i]) for i in ix), default=0.0)
                    topclass = sorted((i for i in ix if sims[i] > 0), key=lambda i: (-sims[i], self.ids[i]))[:self.k]
                    class_neighbors[label] = [self._neighbor(i, sims[i], query, q_entities, q_nums, q_currency, q_period)
                                              for i in topclass]
                    features[f"{view}_mean_topk_sim_{label}"] = float(np.mean([sims[i] for i in topclass])) if topclass else 0.0
                    weights = [float(sims[i]) for i in order]
                    features[f"{view}_weighted_share_{label}"] = (sum(float(sims[i]) for i in order if self.reference_rows[i]["label"] == label) / sum(weights)) if weights else 0.0
                    nearest = topclass[0] if topclass else None
                    if nearest is not None:
                        ref = self.reference_rows[nearest]
                        meta = self._meta[nearest]
                        qd, rd = _date(query.get("date")), meta["date"]
                        gap = int((qd - rd).astype(int)) if qd is not None and rd is not None else 0
                        features[f"{view}_nearest_dategap_signed_{label}"] = float(np.clip(gap / 365.0, -2, 2))
                        features[f"{view}_nearest_dategap_abs_{label}"] = float(min(abs(gap) / 365.0, 2))
                        features[f"{view}_nearest_same_date_{label}"] = float(gap == 0 and qd is not None and rd is not None)
                        features[f"{view}_nearest_shared_entity_{label}"] = float(bool(q_entities & meta["entities"]))
                        features[f"{view}_nearest_numeric_equal_{label}"] = float(bool(q_nums and meta["numbers"] and set(q_nums) == set(meta["numbers"])))
                        features[f"{view}_nearest_numeric_overlap_{label}"] = _overlap(q_nums, meta["numbers"])
                        features[f"{view}_nearest_currency_match_{label}"] = float(bool(q_currency and meta["currency"] and q_currency == meta["currency"]))
                        features[f"{view}_nearest_period_match_{label}"] = float(bool(q_period and meta["period"] and q_period == meta["period"]))
                labels = [self.reference_rows[i]["label"] for i in order]
                counts = Counter(labels)
                probs = [counts[l] / len(labels) for l in LABELS] if labels else []
                features[f"{view}_neighbor_similarity"] = float(np.mean([sims[i] for i in order])) if order else 0.0
                features[f"{view}_neighbor_label_entropy"] = float(-sum(p * np.log(p) for p in probs if p))
                features[f"{view}_neighbor_disagreement"] = float(1 - max(probs)) if probs else 0.0
                features[f"{view}_same_entity_neighbor_share"] = float(np.mean([bool(q_entities & self._meta[i]["entities"]) for i in order])) if order else 0.0
                qdate = _date(query.get("date"))
                features[f"{view}_same_date_neighbor_share"] = float(np.mean([qdate is not None and self._meta[i]["date"] is not None and qdate == self._meta[i]["date"] for i in order])) if order else 0.0
                all_feats.extend(features[n] for n in self.feature_names_ if n.startswith(view + "_"))
                view_trace[view] = [self._neighbor(i, sims[i], query, q_entities, q_nums, q_currency, q_period)
                                    for i in order]
                view_trace.setdefault("class_neighbors", {})
                view_trace["class_neighbors"][view] = class_neighbors
            output.append(np.asarray(all_feats, dtype=float))
            traces.append({"query_id": query.get("id"), "views": view_trace})
        return (np.vstack(output) if output else np.zeros((0, len(self.feature_names_))), traces)

    def _neighbor(self, i, similarity, query, q_entities, q_nums, q_currency, q_period):
        ref, meta = self.reference_rows[i], self._meta[i]
        qd, rd = _date(query.get("date")), meta["date"]
        gap = int((qd-rd).astype(int)) if qd is not None and rd is not None else None
        return {"id": self.ids[i], "label": ref["label"], "date": ref.get("date"),
                "similarity": float(similarity), "shared_entity": bool(q_entities & meta["entities"]),
                "numeric_equal": bool(q_nums and meta["numbers"] and set(q_nums) == set(meta["numbers"])),
                "numeric_overlap": _overlap(q_nums, meta["numbers"]), "date_gap_days": gap,
                "currency_match": bool(q_currency and meta["currency"] and q_currency == meta["currency"]),
                "period_match": bool(q_period and meta["period"] and q_period == meta["period"])}
