"""Rank which candidate attribute a sentence states for a given value."""
from __future__ import annotations

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from arena.lexicon import attribute_words
from arena.values import parse_values


def _prepare(text: str, value: dict | None, subject: str) -> str:
    value = value or {}
    out, pos, seen = [], 0, set()
    for sp in sorted(parse_values(text), key=lambda s: s.start):
        if sp.start < pos:
            continue
        kind = (sp.value or {}).get("kind", "value")
        seen.add(kind)
        out.append(text[pos:sp.start])
        out.append(f" __{kind.upper()}__ ")
        pos = sp.end
    out.append(text[pos:])
    s = "".join(out)
    kind = value.get("kind")
    if kind and kind not in seen:
        s += f" __{str(kind).upper()}__"
    unit = value.get("unit") or value.get("currency") or ""
    per = value.get("per") or ""
    subj = re.sub(r"\W+", "_", subject or "")
    return f"{s} __SUBJ_{subj}__ __UNIT_{unit}__ __PER_{per}__"


class AttributeModel:
    def __init__(self):
        self.pipe: Pipeline | None = None
        self.classes: set[str] = set()

    def fit(self, examples: list[tuple[str, dict, str, str]]) -> "AttributeModel":
        X = [_prepare(t, v, s) for t, v, s, _ in examples]
        y = [a for *_, a in examples]
        feats = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
        ])
        self.pipe = Pipeline([("f", feats), ("c", LogisticRegression(C=4.0, max_iter=2000))])
        self.pipe.fit(X, y)
        self.classes = set(self.pipe.classes_)
        return self

    def rank(self, text: str, value: dict | None, subject: str, candidates: list[str]) -> list[tuple[str, float]]:
        if not candidates:
            return []
        probs = dict(zip(self.pipe.classes_, self.pipe.predict_proba([_prepare(text, value, subject)])[0]))
        low = text.lower()
        scores = []
        for c in candidates:
            if c in self.classes:
                scores.append(float(probs[c]))
            else:
                toks = re.findall(r"[a-z0-9]+", attribute_words(c).lower())
                have = sum(1 for t in toks if re.search(r"(?<![a-z0-9])" + re.escape(t), low))
                scores.append(0.5 * have / len(toks) if toks else 0.0)
        total = sum(scores)
        scores = [s / total for s in scores] if total > 0 else [1 / len(candidates)] * len(candidates)
        return sorted(zip(candidates, scores), key=lambda p: -p[1])
