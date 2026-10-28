"""Reader: sentence -> atoms (subject, attribute, value), with a parse probability each.

1. Values come from hooly.extract (the value grammar).
2. Candidate subjects: names in the sentence (longest first) plus subjects carried over from
   earlier rows of the same page ("He ...", "Its ...", "The platform ...").
3. Every (subject, attribute) pair whose type fits the value is scored by a logistic
   regression trained on practice forms: cue words per attribute, attribute words in the
   sentence, counted noun vs attribute, closeness to known values of that fact, and how the
   subject was found. The best pair above a threshold becomes an atom.
4. Yes/no attributes have no value token: they are scored through their words, and the
   sentence's negation gives the value.
"""
from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict

from .data import Atom, Row
from .extract import Extractor, _singular, negated_at
from .learn import LogReg
from .values import normalize, satisfies

_WORD = re.compile(r"[a-z][a-z0-9]*")  # "trade-in" -> trade, in
_STOP = {"the", "a", "an", "of", "and", "or", "to", "in", "on", "at", "for", "by", "with", "is", "are", "was",
         "were", "be", "it", "its", "this", "that", "from", "as", "has", "have", "had", "now", "our", "we", "i",
         "you", "your", "their", "they", "he", "she", "his", "her", "so", "than", "into", "per", "each", "all"}
AMOUNT_KINDS = ("quantity", "money", "percent", "duration")


def words(text: str) -> list[str]:
    return [_singular(w) for w in _WORD.findall(text.lower()) if w not in _STOP]


_SUFFIXES = sorted(["ations", "ation", "ings", "ing", "ions", "ion", "ments", "ment", "ers", "er", "ed", "es", "s",
                    "ly", "ity", "able"], key=len, reverse=True)


def light_stem(w: str) -> str:
    """Suffix stripping: editing / edit -> edit, encryption / encrypted -> encrypt, shipped -> ship."""
    for suf in _SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            w = w[: -len(suf)]
            break
    if len(w) > 3 and w[-1] == w[-2] and w[-1] not in "aeiouls":
        w = w[:-1]
    if len(w) > 3 and w.endswith("e"):
        w = w[:-1]
    return w


def stem_keys(w: str) -> set[str]:
    """Keys under which two words count as the same stem (STEMMER = "prefix" | "light")."""
    if Reader.STEMMER == "prefix":
        return {stem(w)}
    keys = {light_stem(w)}
    if len(w) >= 5:
        keys.add(w[:5])
    return keys


def stem(w: str) -> str:
    """Crude stem: encrypts / encrypted / encryption -> encry, weighs / weight -> weigh."""
    return w[:5] if len(w) > 5 else w


_MONTHS = {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
           "november", "december"}


def _clause(text: str, pos: int) -> str:
    """The clause around position pos (between punctuation marks)."""
    pos = min(max(0, pos), len(text))
    start = max([m.end() for m in re.finditer(r"[.,;:!?]\s", text[:pos])], default=0)
    m = re.search(r"[.,;:!?](\s|$)", text[pos:])
    return text[start: pos + m.start()] if m else text[start:]


def attr_tokens(attr: str) -> set[str]:
    toks = set()
    for t in attr.lower().split("_"):
        toks.add(_singular(t))
        m = re.fullmatch(r"([a-z]+)(\d+)", t)  # soc2 -> soc, 2
        if m:
            toks.add(m.group(1))
    return toks


def guess_kinds(attr: str) -> set[str]:
    """Value kinds an attribute name suggests, for attributes never seen in training."""
    a = attr.lower()
    t = attr_tokens(a)
    if a.startswith(("is_", "has_", "had_", "raised_", "supports_", "offers_")) or a.endswith(("_certified", "_enabled", "_available")):
        return {"boolean"}
    if "version" in t:
        return {"version"}
    if t & {"price", "cost", "rent", "amount", "revenue", "fee", "bounty", "salary", "budget", "funding", "valuation"}:
        return {"money"}
    if t & {"date", "joined", "launched", "released", "opened", "founded", "started", "since", "end", "opening",
            "release", "deadline", "meeting", "beta", "start"} or a.endswith(("_on", "_at")):
        return {"date"}
    if t & {"share", "rate", "percent", "uptime", "sla", "satisfaction", "accuracy", "attendance", "margin", "growth",
            "churn", "conversion", "score"}:
        return {"percent", "quantity"}
    if t & {"time", "duration", "window", "retention", "length", "latency", "delay", "cadence", "trial", "recover",
            "response", "lifetime", "warranty", "life"}:
        return {"duration"}
    if t & {"ceo", "cto", "cfo", "coo", "lead", "head", "chair", "city", "venue", "headquarters", "provider", "employer",
            "auditor", "investor", "founder", "partner", "region", "name", "owner", "host", "manager", "director"}:
        return {"name"}
    return {"quantity"}


_MODAL = re.compile(r"\b(will|should|would|could|might|plans? to|planned|planning to|expects? to|expected to|"
                    r"aims? to|intends? to|is set to|are set to|soon)\b", re.I)


class Reader:
    STEMMER = "prefix"      # "light": suffix stripping plus 5-letter prefix (tested: -2 rows, off)
    PARTIAL_CAPS = False    # name words only when capitalised (tested: -1 row, off)
    BOOL_MODAL = True       # no yes/no fact from a clause that is a plan or a hedge ("should be available")

    def __init__(self, threshold: float = 0.35, bool_threshold: float = 0.5, seed: int = 0, cue_dropout: float = 0.35):
        self.threshold, self.bool_threshold, self.seed = threshold, bool_threshold, seed
        self.cue_dropout = cue_dropout
        self._drop = None  # attribute names treated as unseen for the current training row

    # ------------------------------------------------------------------ training
    def fit(self, rows: list[Row], lexicon: dict, names) -> "Reader":
        self.lexicon, self.names = lexicon, names
        self.subject_attrs = {s["name"]: list(s["attributes"]) for s in lexicon["subjects"]}
        self.spellings = []  # (regex, canonical subject)
        for s in lexicon["subjects"]:
            for n in [s["name"]] + list(s.get("aliases", [])):
                # "Hooly Android 16.0" names Hooly Android (version 16.0), not Hooly Android 16
                rx = re.compile(r"(?<![\w])" + r"\s*".join(map(re.escape, n.split())) + r"(?!\w|\.\d)", re.I)
                self.spellings.append((len(n), rx, s["name"]))
        self.spellings.sort(key=lambda x: -x[0])
        # distinctive name words ("Berlin" -> Hooly Berlin Office, "the Plus plan" -> Hooly Photos Plus)
        tok_subj = defaultdict(set)
        for s in lexicon["subjects"]:
            for n in [s["name"]] + list(s.get("aliases", [])):
                for t in re.findall(r"[a-z][a-z0-9]*", n.lower()):
                    tok_subj[t].add(s["name"])
        self.partials = [(re.compile(r"(?<![\w])" + re.escape(t) + r"(?![\w])", re.I), sorted(subs))
                         for t, subs in tok_subj.items()
                         if len(subs) <= 2 and len(t) > 2 and t not in _STOP and t not in _MONTHS and t != "hooly"]
        train = [r for r in rows if r.atoms]
        # Per-page copies of every statistic: features for a training row are computed with
        # its own page left out, as they will be for an unseen row (no self-inclusion leak).
        self.pg = defaultdict(lambda: defaultdict(Counter))
        self.pg_known = defaultdict(lambda: defaultdict(list))
        self.kinds = defaultdict(set)       # (subject, attribute) -> kinds seen
        self.kinds_by_name = defaultdict(set)
        self.known = defaultdict(list)      # (subject, attribute) -> values seen (any label)
        self.attr_count = Counter()         # attribute name -> atoms
        self.subj_attr = Counter()
        self.cue = defaultdict(Counter)     # attribute name -> word counts
        self.word_df = Counter()
        units, name_vals = set(), set()
        for r in train:
            ws = set(words(self._plain(r.text)))
            pg = self.pg[r.page]
            for a in r.atoms:
                self.kinds[a.key].add(a.value["kind"])
                self.kinds_by_name[a.attribute].add(a.value["kind"])
                self.known[a.key].append(a.value)
                self.pg_known[r.page][a.key].append(a.value)
                self.attr_count[a.attribute] += 1
                pg["attr_count"][a.attribute] += 1
                self.subj_attr[a.key] += 1
                pg["subj_attr"][a.key] += 1
                self.cue[a.attribute].update(ws)
                pg["cue:" + a.attribute].update(ws)
                self.word_df.update(ws)
                pg["word_df"].update(ws)
                if a.value["kind"] == "quantity" and a.value["unit"].isalpha() and len(a.value["unit"]) >= 3:
                    units.add(_singular(a.value["unit"].lower()))
                if a.value["kind"] == "name":
                    name_vals.add(a.value["value"])
        self.n_atoms = max(1, sum(self.attr_count.values()))
        self.unit_attr = defaultdict(Counter)  # counted noun / unit -> attribute name
        self.local_cue = defaultdict(Counter)  # attribute name -> words right before its value
        self.local_df = Counter()
        for r in train:
            for a in r.atoms:
                u = self._unit(a.value)
                if u:
                    self.unit_attr[u][a.attribute] += 1
                    self.pg[r.page]["unit:" + u][a.attribute] += 1
        # The company itself is the usual unnamed subject ("our headquarters", "38 employees"):
        # the subject with the most attributes.
        self.default_subject = max(self.subject_attrs, key=lambda s: len(self.subject_attrs[s])) if self.subject_attrs else None
        nouns = set(units) | {_singular(at.split("_")[-1]) for at_list in self.subject_attrs.values() for at in at_list}
        protected = [n for s in lexicon["subjects"] for n in [s["name"]] + s.get("aliases", [])] + list(lexicon.get("events", []))
        self.extractor = Extractor(nouns=nouns, names=name_vals, protected=protected, units=units)

        for r in train:  # local cue words: the few words just before each gold value
            found = self._find(r.text)
            for a in r.atoms:
                for f in found:
                    if f.kind == a.value["kind"] and satisfies(self._exact(a.value), self._exact(f.value), self.names):
                        lw = set(self._local_words(self._plain(r.text), f.start, f.end))
                        self.local_cue[a.attribute].update(lw)
                        self.local_df.update(lw)
                        self.pg[r.page]["lcue:" + a.attribute].update(lw)
                        self.pg[r.page]["local_df"].update(lw)
                        break
        rng = random.Random(self.seed)
        X, y, Xb, yb = [], [], [], []
        pages = self._page_index(rows)
        all_names = sorted({a for attrs in self.subject_attrs.values() for a in attrs})
        for r in train:
            found = self._find(r.text)
            cands = self._subjects(r, pages, gold=True)
            self._ex = r.page
            # cue dropout: some attribute names (gold or not) look unseen for this row, so the
            # scorer also learns to rank attributes it has no cue statistics for
            self._drop = {a for a in all_names if rng.random() < self.cue_dropout}
            for f in found:
                gold_keys = {a.key for a in r.atoms if a.value["kind"] == f.kind and
                             satisfies(self._exact(a.value), self._exact(f.value), self.names)}
                pairs = self._pairs(r, f, cands)
                pos = [p for p in pairs if p[0] in gold_keys]
                neg = [p for p in pairs if p[0] not in gold_keys]
                rng.shuffle(neg)
                for key, feats in pos + neg[:20]:
                    X.append(feats)
                    y.append(int(key in gold_keys))
            gold_bool = {a.key for a in r.atoms if a.value["kind"] == "boolean"}
            bpairs = self._bool_pairs(r, cands)
            bneg = [p for p in bpairs if p[0] not in gold_bool]
            rng.shuffle(bneg)
            for key, feats, _ in [p for p in bpairs if p[0] in gold_bool] + bneg[:15]:
                Xb.append(feats)
                yb.append(int(key in gold_bool))
        self._ex = None
        self._drop = None
        self.model = LogReg(l2=1e-3, epochs=15, lr=0.3).fit(X, y)
        self.bool_model = LogReg(l2=1e-3, epochs=15, lr=0.3).fit(Xb, yb)
        return self

    @staticmethod
    def _unit(v):
        if v["kind"] == "quantity":
            return _singular(v["unit"].lower())
        if v["kind"] in ("duration", "money"):
            return v.get("unit") or v.get("per") or v["kind"]
        return None

    @staticmethod
    def _local_words(text, start, end, before=6, after=3):
        pre = words(text[:start])[-before:]
        post = words(text[end:])[:after]
        return pre + post

    _ex = None  # page whose own statistics are left out (set while building training features)

    def _get(self, table: str, glob: Counter, key):
        v = glob.get(key, 0)
        if self._ex is not None:
            v -= self.pg[self._ex][table].get(key, 0) if self._ex in self.pg else 0
        return v

    def _n_atoms(self):
        n = self.n_atoms
        if self._ex is not None and self._ex in self.pg:
            n -= sum(self.pg[self._ex]["attr_count"].values())
        return max(1, n)

    def _local_cue(self, attr, lw):
        n_a = self._get("attr_count", self.attr_count, attr)
        if n_a <= 0:
            return 0.0
        n = self._n_atoms()
        score = 0.0
        for w in lw:
            c = self._get("lcue:" + attr, self.local_cue[attr], w)
            df = self._get("local_df", self.local_df, w)
            lift = math.log((c + 0.5) / (n_a + 1)) - math.log((df + 0.5) / (n + 1))
            if lift > 0:
                score += lift
        return score / math.sqrt(len(lw) + 1)

    def _known(self, key):
        vals = self.known.get(key, [])
        if self._ex is None or self._ex not in self.pg_known:
            return vals
        own = self.pg_known[self._ex].get(key, [])
        if not own:
            return vals
        out, drop = [], list(own)
        for v in vals:
            if v in drop:
                drop.remove(v)
            else:
                out.append(v)
        return out

    delex = False  # cue words from the sentence with lexicon names blanked out (Sebi's delexicalisation)

    def _plain(self, text: str) -> str:
        """The sentence with lexicon names blanked out (same length, so offsets still hold):
        cue words then describe the statement ("costs", "weighs"), not the product."""
        if not self.delex:
            return text
        chars = list(text)
        for s, e, _ in self.mentions(text):
            chars[s:e] = " " * (e - s)
        return "".join(chars)

    _SPEAKER = re.compile(r"^([A-Z][\w.'-]*(?: [A-Z][\w.'-]*){0,2}):\s")

    def _find(self, text: str):
        """Values in a sentence, minus an interview speaker tag ("Gavin: ...")."""
        found = self.extractor.find(text)
        m = self._SPEAKER.match(text)
        if m:
            found = [f for f in found if not (f.kind == "name" and f.start < m.end(1) + 1)]
        return found

    @staticmethod
    def _exact(v):
        return {k: x for k, x in v.items() if k != "op"}

    # ------------------------------------------------------------------ candidates
    def mentions(self, text: str) -> list[tuple[int, int, str]]:
        taken, out = [], []
        for _, rx, subj in self.spellings:
            for m in rx.finditer(text):
                if all(m.end() <= s or m.start() >= e for s, e in taken):
                    taken.append((m.start(), m.end()))
                    out.append((m.start(), m.end(), subj))
        return sorted(out)

    def _page_index(self, rows):
        pages = defaultdict(list)
        for r in sorted(rows, key=lambda r: r.index):
            pages[r.page].append(r)
        return pages

    def _subjects(self, row: Row, pages, gold=False):
        """subject -> (source, rank, positions), by priority: named in the sentence, named inside
        a longer name, earlier rows of the page (nearest first), a distinctive name word, later
        rows of the page (their text only), the company itself."""
        out = {}
        taken = []
        for s, e, subj in self.mentions(row.text):
            out.setdefault(subj, ("text", 0, []))[2].append((s, e))
            taken.append((s, e))
        for s, e, subj in self.nested(row.text, taken):
            if subj not in out:
                out[subj] = ("nested", 0, [(s, e)])
        page = pages.get(row.page, [])
        rank = 0
        for prev in reversed([p for p in page if p.index < row.index]):
            subs = ([a.subject for a in prev.atoms] if prev.atoms else []) + [m[2] for m in self.mentions(prev.text)]
            for subj in subs:
                if subj not in out:
                    rank += 1
                    out[subj] = ("context", rank, [])
        for rx, subs in self.partials:
            for m in rx.finditer(row.text):
                if any(s <= m.start() < e for s, e in taken):
                    continue
                if self.PARTIAL_CAPS and not row.text[m.start()].isupper():
                    continue
                for subj in subs:
                    if subj not in out:
                        out[subj] = ("partial", len(subs), [(m.start(), m.end())])
        rank = 0
        for nxt in [p for p in page if p.index > row.index]:
            for m in self.mentions(nxt.text):
                if m[2] not in out:
                    rank += 1
                    out[m[2]] = ("later", rank, [])
        if self.default_subject and self.default_subject not in out:
            out[self.default_subject] = ("default", 0, [])
        return out

    def nested(self, text: str, taken) -> list[tuple[int, int, str]]:
        """Shorter names inside a longer mention: "Hooly Cloud" in "Hooly Cloud Next 2026"."""
        out = []
        for _, rx, subj in self.spellings:
            for m in rx.finditer(text):
                if any(s <= m.start() and m.end() <= e and (m.start(), m.end()) != (s, e) for s, e in taken):
                    out.append((m.start(), m.end(), subj))
        return out

    def _unseen(self, attr) -> bool:
        return attr not in self.kinds_by_name or (self._drop is not None and attr in self._drop)

    def _kind_ok(self, key, kind):
        if self._unseen(key[1]):
            return 1 if kind in guess_kinds(key[1]) else 0.5
        if key in self.kinds:
            return 2 if kind in self.kinds[key] else 0
        if key[1] in self.kinds_by_name:
            return 2 if kind in self.kinds_by_name[key[1]] else 0
        return 1 if kind in guess_kinds(key[1]) else 0

    def _cue(self, attr: str, ws: set[str]) -> float:
        n_a = self._get("attr_count", self.attr_count, attr)
        if n_a <= 0:
            return 0.0
        n = self._n_atoms()
        score = 0.0
        for w in ws:
            c = self._get("cue:" + attr, self.cue[attr], w)
            df = self._get("word_df", self.word_df, w)
            lift = math.log((c + 0.5) / (n_a + 1)) - math.log((df + 0.5) / (n + 1))
            if lift > 0:
                score += lift
        return score / math.sqrt(len(ws) + 1)

    def _subject_feats(self, subj, info, f_span=None, n_text_subjects=0):
        source, rank, positions = info
        feats = {}
        if source == "text":
            feats["subj_text"] = 1.0
            if f_span and positions:
                d = min(abs(f_span[0] - e) if f_span[0] >= e else abs(s - f_span[1]) for s, e in positions)
                feats["subj_dist"] = math.exp(-d / 30)
            if n_text_subjects == 1:
                feats["only_text_subject"] = 1.0
        elif source == "context":
            feats["subj_context"] = 1.0 / rank
            feats["subj_context_far"] = min(rank, 12) / 12
            if n_text_subjects:
                feats["context_but_text_has_other"] = 1.0
        elif source == "nested":
            feats["subj_nested"] = 1.0
        elif source == "partial":
            feats["subj_partial"] = 1.0 / rank
            if f_span and positions:
                s, e = positions[0]
                d = abs(f_span[0] - e) if f_span[0] >= e else abs(s - f_span[1])
                feats["partial_dist"] = math.exp(-d / 30)
            if n_text_subjects:
                feats["partial_but_text_has_other"] = 1.0
        elif source == "later":
            feats["subj_later"] = 1.0 / rank
            if n_text_subjects:
                feats["later_but_text_has_other"] = 1.0
        else:
            feats["subj_default"] = 1.0
            if n_text_subjects:
                feats["default_but_text_has_other"] = 1.0
        return feats

    def _pairs(self, row, f, cands):
        ws = set(words(self._plain(row.text)))
        wst = {k for w in ws for k in stem_keys(w)}
        n_text = sum(1 for v in cands.values() if v[0] == "text")
        unit = _singular(f.value["unit"].lower()) if f.kind == "quantity" else None
        u_key = self._unit(f.value)
        u_tot = (sum(self.unit_attr[u_key].values()) - (sum(self.pg[self._ex]["unit:" + u_key].values())
                 if self._ex is not None and self._ex in self.pg else 0)) if u_key else 0
        lw = self._local_words(self._plain(row.text), f.start, f.end)
        out = []
        for subj, info in cands.items():
            for attr in self.subject_attrs.get(subj, []):
                key = (subj, attr)
                ok = self._kind_ok(key, f.kind)
                if not ok:
                    continue
                unseen = self._unseen(attr)
                feats = {"bias": 1.0, {2: "kind_seen", 1: "kind_guessed", 0.5: "kind_unguessed"}[ok]: 1.0}
                at = attr_tokens(attr)
                overlap = len(at & ws) / len(at)
                if overlap:
                    feats["attr_words"] = overlap
                    if unseen:
                        feats["attr_words:unseen"] = overlap
                st = sum(1 for t in at if stem_keys(t) & wst) / len(at)
                if st > overlap:
                    feats["attr_stems"] = st - overlap
                if unit and unit in at:
                    feats["unit_in_attr"] = 1.0
                if unseen:
                    feats["attr_unseen"] = 1.0
                    feats["attr_prior"] = math.log(1 / len(self.subject_attrs[subj]))
                    known = []
                else:
                    feats["cue"] = self._cue(attr, ws)
                    if u_tot > 0:
                        feats["unit_assoc"] = math.log((self._get("unit:" + u_key, self.unit_attr[u_key], attr) + 0.1) / (u_tot + 1))
                    feats["local_cue"] = self._local_cue(attr, lw)
                    tot = sum(self._get("subj_attr", self.subj_attr, (subj, a2)) for a2 in self.subject_attrs[subj])
                    feats["attr_prior"] = math.log((self._get("subj_attr", self.subj_attr, key) + 1) / (tot + len(self.subject_attrs[subj])))
                    known = self._known(key)
                if known:
                    feats["has_known"] = 1.0
                    if any(satisfies(self._exact(f.value), k, self.names) for k in known):
                        feats["known_match"] = 1.0
                    elif f.kind in AMOUNT_KINDS:
                        feats["magnitude"] = self._closeness(f.value, known)
                if f.kind == "name" and f.value["value"].lower() == subj.lower():
                    feats["value_is_subject"] = 1.0
                feats.update(self._subject_feats(subj, info, (f.start, f.end), n_text))
                out.append((key, feats))
        return out

    def _closeness(self, v, known):
        try:
            nv = normalize(self._exact(v), self.names)
        except ValueError:
            return 0.0
        best = 0.0
        for k in known:
            try:
                nk = normalize(k, self.names)
            except ValueError:
                continue
            if nk.dim == nv.dim and nk.value > 0 and nv.value > 0:
                d = abs(math.log10(float(nv.value)) - math.log10(float(nk.value)))
                best = max(best, math.exp(-3 * d))
        return best

    def _bool_pairs(self, row, cands):
        ws = set(words(self._plain(row.text)))
        wst = {k for w in ws for k in stem_keys(w)}
        n_text = sum(1 for v in cands.values() if v[0] == "text")
        out = []
        for subj, info in cands.items():
            for attr in self.subject_attrs.get(subj, []):
                key = (subj, attr)
                if not self._kind_ok(key, "boolean"):
                    continue
                at = attr_tokens(attr)
                hits = at & ws
                unseen = self._unseen(attr)
                shits = {t for t in at if stem_keys(t) & wst}
                feats = {"bias": 1.0, "attr_words": len(hits) / len(at), "all_attr_words": float(hits == at),
                         "attr_stems": len(shits) / len(at), "all_attr_stems": float(shits == at)}
                if unseen:
                    feats["attr_unseen"] = 1.0
                    feats["attr_words:unseen"] = len(hits) / len(at)
                else:
                    feats["cue"] = self._cue(attr, ws)
                feats.update(self._subject_feats(subj, info, None, n_text))
                pos = len(row.text)
                for t in hits | shits:
                    m = re.search(r"\b" + re.escape(min(stem_keys(t), key=len)), row.text.lower())
                    if m:
                        pos = min(pos, m.start())
                out.append((key, feats, pos))
        return out

    # ------------------------------------------------------------------ inference
    self_pages = frozenset()  # pages that lent pseudo-labelled sentences (self-training)

    def parse(self, row: Row, pages) -> list[Atom]:
        self._ex = row.page if row.page in self.self_pages else None
        try:
            return self._parse(row, pages)
        finally:
            self._ex = None

    def _parse(self, row: Row, pages) -> list[Atom]:
        cands = self._subjects(row, pages)
        best_for_key = {}
        options = []  # every (p, value, pair): assign greedily, one pair per value and per fact
        for i, f in enumerate(self._find(row.text)):
            for key, feats in self._pairs(row, f, cands):
                p = self.model.predict_proba(feats)
                if p >= self.threshold:
                    options.append((p, i, key, f))
        used_values = set()
        for p, i, key, f in sorted(options, key=lambda o: -o[0]):
            if i in used_values or key in best_for_key:
                continue
            used_values.add(i)
            best_for_key[key] = (p, Atom(key[0], key[1], f.value))
        for key, feats, pos in self._bool_pairs(row, cands):
            p = self.bool_model.predict_proba(feats)
            if self.BOOL_MODAL and _MODAL.search(_clause(row.text, pos)):
                continue
            if p >= self.bool_threshold and (key not in best_for_key or best_for_key[key][0] < p):
                best_for_key[key] = (p, Atom(key[0], key[1], {"kind": "boolean", "value": not negated_at(row.text, pos)}))
        out = []
        for p, a in best_for_key.values():
            a = Atom(a.subject, a.attribute, a.value)
            out.append((a, p))
        self.last_probs = {a.key: p for a, p in out}
        return [a for a, _ in out]

    def parse_all(self, rows: list[Row]) -> dict[str, list[Atom]]:
        """Parse rows in page order so context from earlier rows is available."""
        pages = self._page_index(rows)
        parsed, probs = {}, {}
        for page_rows in pages.values():
            for r in page_rows:
                if r.atoms is None:
                    atoms = self.parse(r, pages)
                    r.atoms = atoms          # later rows of the page use it as context
                    r.meta["parsed"] = True
                    r.meta["parse_probs"] = {f"{k[0]}|{k[1]}": round(p, 3) for k, p in self.last_probs.items()}
                    r.meta["parse_scores"] = dict(self.last_probs)
                    parsed[r.id] = atoms
        return parsed


def atom_match(p: Atom, g: Atom, names) -> bool:
    """Does a parsed atom state the same fact and value as a gold atom?"""
    if p.key != g.key or p.value["kind"] != g.value["kind"]:
        return False
    if p.value.get("op", "eq") != g.value.get("op", "eq"):
        return False
    strip = lambda v: {k: x for k, x in v.items() if k != "op"}
    return bool(satisfies(strip(g.value), strip(p.value), names))
