"""End to end: the Reader parses rows that have no atoms, then a truth model labels them.

Labelled practice rows keep their gold atoms (at runtime they are recovered from the
packaged practice data by id + text + date); every other row is parsed from its text.

Parse confidence (#4): inside the training data, a second Reader trained without a slice of
pages parses that slice, which shows how often a parse with score s is actually right
(isotonic calibration). Each parsed atom carries that probability; the truth model pulls an
unsure parse towards the prior instead of trusting it fully.
"""
from __future__ import annotations

import random
from dataclasses import replace

from .calibrate import Isotonic
from .folds import page_groups
from .models import Model
from .reader import Reader, atom_match


class Pipeline(Model):
    def __init__(self, truth_cls, threshold: float = 0.3, bool_threshold: float = 0.4, seed: int = 0,
                 parse_aware: bool = True, self_train: float | None = None):
        self.truth = truth_cls()
        self.name = self.truth.name
        self.threshold, self.bool_threshold, self.seed = threshold, bool_threshold, seed
        self.parse_aware = parse_aware
        self.self_train = self_train  # e.g. 0.9: retrain the Reader on its own parses this sure

    def _reader(self):
        return Reader(self.threshold, self.bool_threshold, seed=self.seed)

    def _parse_calibration(self, train_rows, lexicon, names):
        groups = page_groups(train_rows)
        random.Random(self.seed).shuffle(groups)
        held, n = set(), 0
        for g in groups:
            if n >= len(train_rows) // 5:
                break
            held.update(r.id for r in g)
            n += len(g)
        inner = self._reader().fit([r for r in train_rows if r.id not in held], lexicon, names)
        copies = [replace(r, atoms=None, meta={}) if r.id in held else r for r in train_rows]
        inner.parse_all(copies)
        gold = {r.id: r.atoms for r in train_rows}
        scores, ok, fb_scores, fb_ok = [], [], [], []
        for c in copies:
            if c.id in held:
                fb = c.meta.get("fallback_keys", set())
                for a in c.atoms:
                    hit = int(any(atom_match(a, g, names) for g in gold[c.id]))
                    if a.key in fb:  # second-chance facts get their own confidence map
                        fb_scores.append(c.meta["parse_scores"].get(a.key, 0.0)); fb_ok.append(hit)
                    else:
                        scores.append(c.meta["parse_scores"].get(a.key, 0.0)); ok.append(hit)
        cal = Isotonic().fit(scores, ok)
        cal.fallback = Isotonic().fit(fb_scores, fb_ok) if len(fb_scores) >= 10 else None
        self.n_fallback_cal = len(fb_scores)
        return cal

    def fit(self, rows, lexicon, names):
        train_rows = [r for r in rows if r.atoms]
        self.parse_cal = self._parse_calibration(train_rows, lexicon, names) if self.parse_aware else None
        self.reader = self._reader().fit(train_rows, lexicon, names)
        self.reader.parse_all(rows)  # fills .atoms of rows that had none
        if self.self_train and self.parse_cal:
            self._self_train(rows, train_rows, lexicon, names)
        if self.parse_cal:
            for r in rows:
                if r.meta.get("parsed"):
                    fb = r.meta.get("fallback_keys", set())
                    fcal = getattr(self.parse_cal, "fallback", None)
                    r.meta["atom_q"] = {k: (fcal(p) if k in fb and fcal else self.parse_cal(p))
                                        for k, p in r.meta["parse_scores"].items()}
        self.truth.fit(rows, lexicon, names)
        return self

    def _self_train(self, rows, train_rows, lexicon, names):
        """#1 self-training: unlabelled rows whose every parsed fact is sure (calibrated parse
        probability >= self_train) become extra training sentences for a second Reader, which
        then re-parses every unlabelled row. Rows on those pages are re-parsed with their own
        page's statistics left out, so a parse cannot confirm itself."""
        pseudo = [r for r in rows if r.meta.get("parsed") and r.atoms and
                  all(self.parse_cal(p) >= self.self_train for p in r.meta["parse_scores"].values())]
        self.n_pseudo = len(pseudo)
        if not pseudo:
            return
        reader = self._reader().fit(train_rows + pseudo, lexicon, names)
        reader.self_pages = {r.page for r in pseudo}
        for r in rows:
            if r.meta.get("parsed"):
                r.atoms = None
                for k in ("parsed", "parse_probs", "parse_scores"):
                    r.meta.pop(k, None)
        reader.parse_all(rows)
        self.reader = reader

    def predict(self, row):
        p = self.truth.predict(row)
        p.trace["parsed"] = bool(row.meta.get("parsed"))
        p.trace["parse_probs"] = row.meta.get("parse_probs", {})
        return p
