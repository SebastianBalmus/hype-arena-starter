"""E5: credibility from how a sentence reads (after SLiMFast, Rekatsinas et al. 2017).

Classic truth discovery learns a reliability per *source*; the runtime input has no source
ids. SLiMFast instead predicts reliability from features with a discriminative model and
can use labelled ground truth. Here: P(row is a lie | its text), a logistic regression over
style features. Numbers are masked so the model learns *how* things are said (first person,
superlatives, round numbers, "over"), not *which* values were lies.
"""
from __future__ import annotations

import re

from .data import Row
from .learn import LogReg

_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
_TOKEN = re.compile(r"<num>|[a-z]+(?:'[a-z]+)?|[€$£%!?:]")
_FIRST_PERSON = {"i", "me", "my", "we", "our", "us", "i'm", "we're"}
_HYPE = {"best", "most", "record", "every", "ever", "fastest", "biggest", "largest", "only",
         "all", "any", "everyone", "nobody", "even", "already", "proud", "amazing", "vertical",
         "competitor", "competitors", "humans", "personal", "beat", "beats", "unmatched", "world"}
_BOUND = re.compile(r"\b(over|more than|at least|up to|under|less than|fewer than|nearly|almost)\b")


def featurize(text: str) -> dict[str, float]:
    t = text.lower()
    numbers = _NUM.findall(t)
    toks = _TOKEN.findall(_NUM.sub(" <num> ", t))
    f: dict[str, float] = {}
    for a in toks:
        f["w:" + a] = 1.0
    for a, b in zip(toks, toks[1:]):
        f["b:" + a + "_" + b] = 1.0
    if _FIRST_PERSON & set(toks):
        f["first_person"] = 1.0
    hype = len(_HYPE & set(toks))
    if hype:
        f["hype"] = min(2.0, float(hype))
    if "!" in text:
        f["exclaim"] = 1.0
    if _BOUND.search(t):
        f["bound_word"] = 1.0
    for n in numbers:
        digits = n.replace(",", "").split(".")[0]
        if len(digits) >= 3 and digits.endswith("00"):
            f["round_number"] = 1.0
        if len(digits) >= 7:
            f["huge_number"] = 1.0
    if len(numbers) >= 2:
        f["several_numbers"] = 1.0
    n_words = sum(1 for a in toks if a.isalpha())
    f["len:" + ("short" if n_words <= 5 else "mid" if n_words <= 12 else "long")] = 1.0
    if not text.rstrip().endswith((".", "!", "?")):
        f["no_final_punct"] = 1.0  # slide- or spec-sheet-like fragment
    return f


class Credibility:
    """P(row is false | text). Trained on true vs false rows (unsure rows left out)."""

    def __init__(self, l2: float = 3e-3):
        self.l2 = l2

    def fit(self, rows: list[Row]) -> "Credibility":
        data = [(featurize(r.text), int(r.label == "false")) for r in rows if r.label in ("true", "false")]
        self.model = LogReg(l2=self.l2).fit([x for x, _ in data], [y for _, y in data])
        return self

    def p_false(self, text: str) -> float:
        return self.model.predict_proba(featurize(text))
