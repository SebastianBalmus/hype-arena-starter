"""Fact extraction: sentences -> typed (subject, attribute, value) facts, using page context."""
from __future__ import annotations

import re

from arena.kinds import KindIndex, bigrams, unit_of
from arena.lexicon import Lexicon, NameIndex, has_attribute_words
from arena.model import Fact, Row
from arena.values import has_negation, parse_values, split_sentences

CONTEXT_PRIORS = (0.6, 0.45, 0.35)
_CLAUSE_SPLIT = re.compile(r"[,;:]|\s+(?:but|although|though|while|whereas)\s+", re.IGNORECASE)


class Extractor:
    def __init__(self, lexicon: Lexicon, names: NameIndex, kinds: KindIndex, model,
                 gold: dict[str, list[Fact]] | None = None, gold_rows: dict[str, Row] | None = None):
        self.lexicon, self.names, self.kinds, self.model = lexicon, names, kinds, model
        self.gold, self.gold_rows = gold or {}, gold_rows or {}
        self._cache: dict = {}

    # -- helpers -------------------------------------------------------------
    def _subject_candidates(self, mentions, pos: int, context: list[str]) -> list[tuple[str, float]]:
        """(subject, prior): nearest mention before ``pos``, other mentions, then page context."""
        out: list[tuple[str, float]] = []
        seen: set[str] = set()
        before = [m for m in mentions if m.end <= pos]
        first = before[-1] if before else None
        order = ([first] if first else []) + [m for m in mentions if m is not first]
        for m in order:
            if m.subject not in seen:
                seen.add(m.subject)
                out.append((m.subject, 1.0))
        for subj, prior in zip(context, CONTEXT_PRIORS):
            if subj not in seen:
                seen.add(subj)
                out.append((subj, prior))
        return out

    def _attrs(self, subject: str, value: dict) -> list[str]:
        kind = value["kind"]
        cands = [a for a in self.lexicon.attributes(subject) if self.kinds.kind(a) in (kind, None)]
        if kind in ("quantity", "duration", "money") and cands:
            want = unit_of(value)
            keep = [a for a in cands if self.kinds.kind(a) is None or self.kinds.unit(a) in (None, want)]
            if keep:
                cands = keep
        return cands

    def _rank(self, sentence: str, value: dict, subj: str, cands: list[str]):
        key = (sentence, repr(sorted(value.items())), subj, tuple(cands))
        hit = self._cache.get(key)
        if hit is None:
            ranked = self.model.rank(sentence, value, subj, cands)
            hit = ranked[0] if ranked else (None, 0.0)
            if len(self._cache) > 50000:
                self._cache.clear()
            self._cache[key] = hit
        return hit

    def _best(self, sentence: str, value: dict, subjects, used: set, skip=None):
        best = None
        for subj, prior in sorted(subjects, key=lambda t: -t[1]):
            if best is not None and prior <= best[0]:
                break  # a rank score is at most 1, so this subject cannot win
            if skip and skip(subj):
                continue
            cands = [a for a in self._attrs(subj, value) if (subj, a) not in used]
            if not cands:
                continue
            attr, s = self._rank(sentence, value, subj, cands)
            if attr is None:
                continue
            score = prior * s
            if best is None or score > best[0]:
                best = (score, subj, attr)
        return best

    # -- per sentence --------------------------------------------------------
    def _sentence(self, sentence: str, context: list[str]):
        mentions = self.lexicon.mentions(sentence)
        spans = parse_values(sentence)
        used: set[tuple[str, str]] = set()
        facts: list[Fact] = []
        options = []
        for span in spans:
            subs = self._subject_candidates(mentions, span.start, context)
            b = self._best(sentence, span.value, subs, set())
            if b:
                options.append((b[0], span, subs))
        options.sort(key=lambda t: -t[0])
        for _, span, subs in options:
            b = self._best(sentence, span.value, subs, used)
            if b:
                used.add((b[1], b[2]))
                facts.append(Fact(b[1], b[2], dict(span.value), round(min(b[0], 1.0), 4)))
        taken = [(s.start, s.end) for s in spans]
        for start, end, name in self.names.find(sentence):
            if any(start < e and s < end for s, e in taken):
                continue
            subs = self._subject_candidates(mentions, start, context)
            hit = {m.subject for m in mentions if m.start < end and start < m.end}
            value = {"kind": "name", "value": name}
            b = self._best(sentence, value, subs, used, skip=lambda s: s in hit or s == name)
            if b:
                used.add((b[1], b[2]))
                facts.append(Fact(b[1], b[2], value, round(min(b[0], 1.0), 4)))
        # booleans
        subjects = [m.subject for m in mentions] or context[:1]
        clauses = [c for c in _CLAUSE_SPLIT.split(sentence) if c.strip()] or [sentence]
        grams = {c: bigrams(c) for c in clauses}
        for subj in dict.fromkeys(subjects):
            for attr in self.lexicon.attributes(subj):
                if self.kinds.kind(attr) != "boolean":
                    continue
                cues = self.kinds.cues(attr)
                for c in clauses:
                    if has_attribute_words(c, attr) or any(q in grams[c] for q in cues):
                        facts.append(Fact(subj, attr, {"kind": "boolean", "value": not has_negation(c)}, 0.8))
                        break
        return facts, mentions

    # -- public --------------------------------------------------------------
    def extract(self, rows: list[Row]) -> dict[str, list[Fact]]:
        contexts: dict[int, list[str]] = {}
        out: dict[str, list[Fact]] = {}
        for row in rows:
            ctx = contexts.setdefault(row.page, [])
            g, gr = self.gold.get(row.id), self.gold_rows.get(row.id)
            if g is not None and gr is not None and gr.text == row.text and gr.date == row.date:
                out[row.id] = [Fact(f.subject, f.attribute, dict(f.value), 1.0) for f in g]
                for f in out[row.id]:
                    _push(ctx, f.subject)
                continue
            facts: list[Fact] = []
            for sentence in split_sentences(row.text):
                fs, mentions = self._sentence(sentence, ctx[:3])
                facts.extend(fs)
                for m in mentions:
                    _push(ctx, m.subject)
            out[row.id] = facts
        return out


def _push(ctx: list[str], subject: str) -> None:
    if subject in ctx:
        ctx.remove(subject)
    ctx.insert(0, subject)
    del ctx[3:]
