# Hype Arena System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A non-LLM pipeline that labels Hooly universe rows true/false/unsure with calibrated confidence and writes up to 1,000 referee-valid true claims, plus a live demo page, packaged as a Docker image.

**Architecture:** One package `arena/`: a value parser and lexicon matcher feed a fact extractor; extracted facts go into a dated evidence store (value intervals + page reliability); a gradient-boosting decider turns per-fact checks into labels; a template generator turns trusted intervals into claims that a local referee validates. A batch CLI and a FastAPI server share `arena.pipeline.run`.

**Tech Stack:** Python 3.12, numpy 2.1.3, scikit-learn 1.5.2, jsonschema 4.23.0, fastapi 0.115.5, uvicorn 0.32.1; pytest 8.3.3 and httpx 0.27.2 for tests; plain HTML/CSS/JS frontend; Docker (python:3.12-slim, linux/amd64).

**Spec:** `docs/superpowers/specs/2026-10-28-hype-arena-system-design.md` (authoritative contract: `contracts/README.md`, `contracts/claim-forms.md`, `contracts/value-forms.md`, `contracts/*.schema.json`).

## Global Constraints

- Local work only: branch `hype-arena-system`, local commits, never `git push`, never open a PR, never clone.
- No LLM or pretrained language model anywhere in `arena/`; no network at run time.
- Output columns exactly `id,type,text,form,label,confidence`; label rows: empty `type,text,form`; `unsure` → empty confidence; confidences in [0.5, 1]; claim text ≤ 300 chars; claim id matches `^[A-Za-z0-9_.-]{1,64}$`; output ≤ 16 MiB; written to `/output/output.tmp` then `os.replace` to `/output/output.csv`.
- As-of date for `fact` and `compare` claims: `2026-10-29` (constant `AS_OF` in `arena/model.py`).
- Writable paths at run time: `/tmp` (≤ 128 MB) and `/output` only; `HOME=/tmp`.
- Image: `docker build --platform linux/amd64 .`, ≤ 3 GiB, last Dockerfile line `USER 10001`.
- Python env for development: `.venv` created with `uv venv --python 3.12 .venv`; run tests as `.venv/bin/pytest`.
- Fewer than 1,000 valid claims → output fewer; never pad with unvalidated claims.

## Review Focus

1. Row text containing commas, double quotes or line breaks must round-trip through `write_output` and be read back by `scripts/check_output.py` → test in Task 1.
2. A runtime lexicon with a subject/attribute never seen in practice must not crash extraction and must still produce candidate facts via attribute-name overlap → test in Task 6.
3. A row from which nothing can be extracted must still get a label row (`true` with the prior confidence), never be dropped → test in Task 10.
4. A deadline already passed or a SIGTERM mid-run must leave a complete `/output/output.csv` (the safe table) and exit 0 → test in Task 10.
5. A claim whose normalized text equals an input row, or a compare that is the reverse of one already chosen, must be excluded → test in Task 9.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `requirements.txt`, `requirements-dev.txt`, `pytest.ini`, `.gitignore`, `.dockerignore` | deps and tooling |
| `arena/model.py` | `Row`, `Fact`, `AS_OF` |
| `arena/data.py` | read universe/practice CSV, page ids, atomic output writer |
| `arena/values.py` | value-forms.md parser, normaliser, comparator, renderer, sentence split, negation |
| `arena/lexicon.py` | `Lexicon` (subjects, aliases, attributes, events, mentions), `NameIndex` (name values), attribute words |
| `arena/schema.py` | entry JSON-schema validation |
| `arena/claimforms.py` | local referee for the five claim types |
| `arena/kinds.py` | `KindIndex`: attribute → value kind/unit learned from practice |
| `arena/attributes.py` | `AttributeModel`: TF-IDF + logistic regression attribute ranker |
| `arena/extract.py` | `Extractor`: sentence → facts with page context and gold reuse |
| `arena/evidence.py` | `EvidenceStore`: observations, intervals, reliability, checks |
| `arena/decide.py` | row features, `Decider` classifier + isotonic calibration |
| `arena/generate.py` | claim candidates, templates, validation, ranking |
| `arena/train.py` | offline build of `artifacts/` with cross-fitted metrics |
| `arena/pipeline.py` | `load_artifacts`, `run` |
| `arena/cli.py` | batch entrypoint with safe output, deadline, SIGTERM |
| `arena/server.py`, `web/index.html`, `web/app.js`, `web/styles.css` | live demo |
| `Dockerfile` | image |

---

### Task 1: Scaffold, shared model, data IO

**Files:**
- Create: `requirements.txt`, `requirements-dev.txt`, `pytest.ini`, `.gitignore`, `arena/__init__.py`, `arena/model.py`, `arena/data.py`, `tests/__init__.py`, `tests/test_data.py`

**Interfaces:**
- Produces:
  - `arena.model.AS_OF = "2026-10-29"`
  - `@dataclass(frozen=True) class Row: id: str; text: str; date: str; label: str; page: int` (label ∈ `{"true","false","unsure",""}`)
  - `@dataclass class Fact: subject: str; attribute: str; value: dict; confidence: float = 1.0` (`value` follows `claimed_value.schema.json`, may hold `op`)
  - `arena.data.read_universe(path: str) -> list[Row]` — `page` = 0-based index of the run of consecutive rows sharing one `date`, in file order
  - `arena.data.load_practice(path: str) -> tuple[list[Row], dict[str, list[Fact]]]` — rows (with labels) and gold facts by row id
  - `arena.data.write_output(out_dir: str, labels: list[tuple[str, str, float | None]], claims: list[dict]) -> str` — claims are dicts with keys `id,type,text,form(dict),confidence`; writes header then label rows then claim rows (claim `label` = `true`, `form` = `json.dumps(form, ensure_ascii=False)`), via `output.tmp` + `os.replace`; returns final path

- [ ] **Step 1: Environment**

`requirements.txt`: `numpy==2.1.3`, `scikit-learn==1.5.2`, `jsonschema==4.23.0`, `fastapi==0.115.5`, `uvicorn==0.32.1`. `requirements-dev.txt`: `-r requirements.txt`, `pytest==8.3.3`, `httpx==0.27.2`. `pytest.ini`: `[pytest]` / `testpaths = tests`. `.gitignore`: `.venv/`, `__pycache__/`, `artifacts/`, `.pytest_cache/`.
Run: `uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements-dev.txt`
Expected: installs without error.

- [ ] **Step 2: Write the failing tests** in `tests/test_data.py`

```python
def test_read_universe_pages(tmp_path):
    p = tmp_path / "u.csv"
    p.write_text("﻿id,text,date,label\na,x,2026-01-01,true\nb,y,2026-01-01,\nc,z,2026-02-01,\nd,w,2026-01-01,false\n", encoding="utf-8")
    rows = read_universe(str(p))
    assert [r.page for r in rows] == [0, 0, 1, 2]
    assert rows[1].label == "" and rows[0].label == "true"

def test_load_practice_gold():
    rows, gold = load_practice("data/practice.csv")
    assert len(rows) == 1897 and len(gold) == 1897
    assert gold["p-1"][0].subject == "Holdan" and gold["p-1"][0].attribute == "joined"
    assert gold["p-1"][0].value == {"kind": "date", "value": "2026-01-12"}

def test_write_output_roundtrip_quoting(tmp_path):
    labels = [("x-1", "true", 0.9), ("x-2", "unsure", None)]
    claims = [{"id": "c-1", "type": "fact", "text": 'Says "hi", then\nleaves.', "form": {"a": "b,c"}, "confidence": 0.8}]
    path = write_output(str(tmp_path), labels, claims)
    rows = list(csv.DictReader(open(path, encoding="utf-8", newline="")))
    assert rows[1]["confidence"] == "" and rows[2]["text"] == 'Says "hi", then\nleaves.'
    assert json.loads(rows[2]["form"]) == {"a": "b,c"} and rows[2]["label"] == "true"
    assert not (tmp_path / "output.tmp").exists()
```

- [ ] **Step 3: Run** `.venv/bin/pytest tests/test_data.py -v` — Expected: FAIL (ImportError).
- [ ] **Step 4: Implement** `arena/model.py` and `arena/data.py` (csv module, `encoding="utf-8-sig"` on read, `newline=""`).
- [ ] **Step 5: Run** `.venv/bin/pytest tests/test_data.py -v` — Expected: 3 passed.
- [ ] **Step 6: Commit** `git add -A && git commit -m "feat: scaffold, shared model and data IO"`

---

### Task 2: Value parser (`arena/values.py`)

**Files:**
- Create: `arena/values.py`, `tests/test_values.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `@dataclass class ValueSpan: start: int; end: int; value: dict` (`value` includes `op` only when not eq)
  - `split_sentences(text: str) -> list[str]` — ends at `.`/`!`/`?` before a space or at a blank line; not after the abbreviations listed in value-forms.md ("General rules")
  - `has_negation(text: str) -> bool` — words not, no, never, none, nobody, nothing, neither, nor, without, cannot, or any word ending in `n't` (also `n’t`)
  - `parse_values(text: str) -> list[ValueSpan]` — every value of kinds quantity, money, percent, date, version, duration stated plainly. Excluded: after a negation in its clause, approximations, either end of a range, values touching invisible characters (U+200B, U+2060, U+00AD not between letters, U+FEFF). Comparison words set `op`. Names and booleans are NOT parsed here.
  - `normalize(v: dict) -> dict` — comparable canonical form: data units to bytes, length to metres, mass to grams, frequency to Hz, power to W; durations ms/s/min/h/day/week to seconds, month/year to months (12 months = 1 year; never months↔days); counted nouns singular lower-case; drops `op`
  - `values_equal(a: dict, b: dict) -> bool` — same kind and equal after `normalize`; money also needs same currency and same `per` (absent = one-off); dates/versions compare as written strings
  - `compare(a: dict, b: dict) -> int | None` — −1/0/1 after normalisation; `None` when kinds/units/currency/period differ, for names and booleans, and for dates of different precision whose ranges overlap (`2026-03` vs `2026-03-12`)
  - `satisfies(exact: dict, claimed: dict) -> bool` — `claimed.op` (default eq) holds for `exact`
  - `render(v: dict) -> str` — canonical text that `parse_values` reads back as `v`: money `€59 per month` (`$`, `£` for USD/GBP, otherwise `59 CHF`); thousands commas; no trailing zeros; quantity `2,048 customers` (plural noun when value ≠ 1: `y`→`ies`, s/x/ch/sh→`es`, `person`→`people`) or `128 GB` for symbol units; percent `99.9%`; date `12 March 2026` / `March 2026` / `2025`; version `v2.3`; duration `500 ms`, `1.8 s`, `30 min`, `24 hours`, `14 days`, `2 weeks`, `3 months`, `1 year`

- [ ] **Step 1: Write the failing tests.** Transcribe every row of every "Written | Read as" table in `contracts/value-forms.md` (Quantity, counted nouns, Money, Percent, Date, Version, Duration, Comparisons) into one parametrised test, and every row of the "Not read" table into another:

```python
@pytest.mark.parametrize("text,expected", READ_CASES)          # e.g. ("1 GB", {"kind":"quantity","value":1000,"unit":"MB"})
def test_reads(text, expected):
    got = parse_values(text)
    assert any(values_equal(s.value, expected) and s.value.get("op") == expected.get("op") for s in got)

@pytest.mark.parametrize("text,not_expected", NOT_READ_CASES)   # e.g. ("10–20 GB", {"kind":"quantity","value":20,"unit":"GB"})
def test_not_read(text, not_expected):
    assert not any(values_equal(s.value, not_expected) and s.value.get("op") == not_expected.get("op") for s in parse_values(text))

def test_render_roundtrip():
    for f in glob.glob("contracts/examples/claimed_value/good/*.json"):
        v = json.load(open(f))
        if v["kind"] in ("name", "boolean") or "op" in v: continue
        assert any(values_equal(s.value, v) for s in parse_values(render(v))), f

def test_split_and_negation():
    assert split_sentences("Hooly AI Pro is our plan. It costs €59 per month.") == ["Hooly AI Pro is our plan.", "It costs €59 per month."]
    assert len(split_sentences("Hooly Inc. costs e.g. €5 a month.")) == 1
    assert has_negation("The app isn't SOC 2 certified") and not has_negation("Hooly passed its SOC 2 audit")

def test_compare_and_satisfies():
    m = lambda v: {"kind": "money", "value": v, "currency": "EUR", "per": "month"}
    assert compare(m(199), m(59)) == 1 and compare(m(59), {"kind": "money", "value": 590, "currency": "EUR", "per": "year"}) is None
    assert compare({"kind":"date","value":"2026-03"}, {"kind":"date","value":"2026-03-12"}) is None
    assert compare({"kind":"duration","value":14,"unit":"day"}, {"kind":"duration","value":2,"unit":"week"}) == 0
    assert satisfies({"kind":"quantity","value":120,"unit":"customer"}, {"kind":"quantity","value":100,"unit":"customer","op":"gt"})
```

- [ ] **Step 2: Run** `.venv/bin/pytest tests/test_values.py -v` — Expected: FAIL (ImportError).
- [ ] **Step 3: Implement** `arena/values.py` — compiled regexes over the token stream; read numbers per "Numbers" rules (ambiguous forms like `1.500`, `99,9`, `1.000.000` read as nothing; decimal comma only in prices; four-digit 1900–2099 plain numbers are years); up to two words between a number and its counted noun.
- [ ] **Step 4: Run** `.venv/bin/pytest tests/test_values.py -v` — Expected: all passed.
- [ ] **Step 5: Commit** `git add arena/values.py tests/test_values.py && git commit -m "feat: value parser per value-forms.md"`

---

### Task 3: Lexicon and name index (`arena/lexicon.py`)

**Files:**
- Create: `arena/lexicon.py`, `tests/test_lexicon.py`

**Interfaces:**
- Produces:
  - `@dataclass class Mention: start: int; end: int; subject: str; surface: str`
  - `class Lexicon` with `Lexicon.load(path: str) -> Lexicon`, `.subjects: dict[str, list[str]]` (canonical → attributes), `.events: list[str]`, `canonical(name: str) -> str | None` (case-insensitive, runs of spaces collapsed, also matches with spaces removed), `attributes(subject: str) -> list[str]`, `mentions(text: str) -> list[Mention]` (all names + aliases, longest match first, whole words, non-overlapping, left to right)
  - `attribute_words(attribute: str) -> str` — `"soc2_certified"` → `"soc2 certified"`
  - `has_attribute_words(text: str, attribute: str) -> bool` — any case; between words a space, hyphen or nothing; between a word's letters and digits a space, hyphen or nothing (`SOC 2 certified`, `SOC2-certified` both match `soc2_certified`)
  - `class NameIndex` with `NameIndex.from_values(names: list[str], lexicon: Lexicon) -> NameIndex`, `canonical(name: str) -> str` (maps a short form to the unique longer known name starting with it: `Gavin` → `Gavin Bellson`, `Nimbus` → `Nimbus Cloud`, else returns the name), `find(text: str) -> list[tuple[int, int, str]]` (canonical names found as whole words, longest first)

- [ ] **Step 1: Write the failing tests**

```python
L = Lexicon.load("data/lexicon.json")
def test_canonical_and_mentions():
    assert L.canonical("hooly   ai pro") == "Hooly AI Pro" and L.canonical("HoolyAIPro") == "Hooly AI Pro"
    assert [m.subject for m in L.mentions("Hooly AI Pro costs more than Hooly.")] == ["Hooly AI Pro", "Hooly"]
    assert [m.subject for m in L.mentions("The Workspace plan and Hooly I/O 2026")] == ["Hooly Workspace", "Hooly I/O 2026"]
    assert L.mentions("Hoolyx is not a name") == []
def test_attribute_words():
    assert has_attribute_words("Hooly has never been SOC2-certified.", "soc2_certified")
    assert has_attribute_words("Hooly is SOC 2 certified", "soc2_certified")
    assert not has_attribute_words("Hooly passed a SOC 2 audit", "soc2_certified")
def test_name_index():
    ni = NameIndex.from_values(["Gavin Bellson", "Nimbus Cloud", "Berlin"], L)
    assert ni.canonical("Gavin") == "Gavin Bellson" and ni.canonical("Berlin") == "Berlin"
    assert [n for _, _, n in ni.find("Gavin Bellson moved to Berlin")] == ["Gavin Bellson", "Berlin"]
```

- [ ] **Step 2: Run** `.venv/bin/pytest tests/test_lexicon.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement** `arena/lexicon.py` (one regex alternation sorted by length descending, `\b`-style boundaries that also treat `/` and `.` inside names literally).
- [ ] **Step 4: Run** — Expected: all passed.
- [ ] **Step 5: Commit** `git add arena/lexicon.py tests/test_lexicon.py && git commit -m "feat: lexicon mentions and name index"`

---

### Task 4: Local referee — schema + claim forms

**Files:**
- Create: `arena/schema.py`, `arena/claimforms.py`, `tests/test_schema.py`, `tests/test_claimforms.py`

**Interfaces:**
- Consumes: `values.parse_values`, `values.values_equal`, `values.split_sentences`, `values.has_negation` (Task 2); `Lexicon`, `has_attribute_words` (Task 3).
- Produces:
  - `arena.schema.validate_entry(entry: dict) -> list[str]` — Draft 2020-12 validation of `entry.schema.json` with `claimed_value.schema.json` resolved locally (use `referencing.Registry` keyed by both files' `$id` and the relative ref)
  - `arena.claimforms.check_claim(entry: dict, lexicon: Lexicon) -> list[str]` — empty list = valid. Checks: schema; subject/event/target names resolve in the lexicon; attributes belong to their subject (`link` to `subject`, `attribute` to `target` for relation; `event` in `lexicon.events`); then the text rules of `contracts/claim-forms.md` for the entry's type — one sentence holds the required names, phrase and values; the referee value reading (`parse_values`) finds the claim's value(s) exactly and no other value of the same kind; no extra date, no month/season/weekday or time word from the "Refused" table outside names/phrase/value; no negation except yes-or-no; yes-or-no needs `has_attribute_words`; compare: left name, one phrase of the op's list (dates vs amounts table), right name, no number/value/date; change: a change phrase with `up`/`down` direction consistent; at_event: an event phrase (`at`, `during`, `on the day of`, `at the time of`) right before the event name (optional `the`); relation: `has_attribute_words(text, link)`

- [ ] **Step 1: Write the failing tests.** `tests/test_schema.py`: every file in `contracts/examples/entry/good` → `validate_entry == []`; every file in `contracts/examples/entry/bad` → non-empty. `tests/test_claimforms.py`: every row of every example table in `contracts/claim-forms.md` (Fact ×2 tables, Compare ×2 tables, Change, At an event, Relation) as a parametrised case `(type, form, text, readable: bool)`, building the entry with `id="t"`, `tag="true"`, `confidence=0.9`, using the form each table states (e.g. Fact table 1: `{"subject":"Hooly AI Pro","attribute":"price","value":{"kind":"money","value":59,"currency":"EUR","per":"month"}}`; SOC 2 table: `{"subject":"Hooly","attribute":"soc2_certified","value":{"kind":"boolean","value":false}}`; Compare tables: left/right as stated with the row's `op`; Change: from €49/month to €59/month; At an event: Hooly Gemini `languages` 12 language at `Hooly I/O 2026`; Relation: Hooly `cto` → Bertram Gilfoil `joined` 2026-07-01):

```python
@pytest.mark.parametrize("ctype,form,text,ok", CASES)
def test_claim_forms(ctype, form, text, ok):
    entry = {"id": "t", "type": ctype, "text": text, "form": form, "tag": "true", "confidence": 0.9}
    assert (check_claim(entry, L) == []) is ok, check_claim(entry, L)
```

Also all `contracts/examples/entry/good/*.json` pass `check_claim`.

- [ ] **Step 2: Run** `.venv/bin/pytest tests/test_schema.py tests/test_claimforms.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement** `arena/schema.py` and `arena/claimforms.py`.
- [ ] **Step 4: Run** — Expected: all passed.
- [ ] **Step 5: Commit** `git add arena/schema.py arena/claimforms.py tests/test_schema.py tests/test_claimforms.py && git commit -m "feat: local referee for claim forms and schema"`

---

### Task 5: Kind index and fact extractor (`arena/kinds.py`, `arena/extract.py`)

**Files:**
- Create: `arena/kinds.py`, `arena/extract.py`, `tests/test_extract.py`

**Interfaces:**
- Consumes: `Row`, `Fact` (Task 1); `parse_values`, `split_sentences`, `has_negation` (Task 2); `Lexicon`, `NameIndex`, `has_attribute_words` (Task 3); `AttributeModel.rank` (Task 6 — until Task 6 lands, tests use a stub with the same signature that returns candidates with uniform scores).
- Produces:
  - `class KindIndex` with `KindIndex.from_gold(gold: dict[str, list[Fact]]) -> KindIndex`, `kind(attribute: str) -> str | None`, `unit(attribute: str) -> str | None` (most common unit/currency+per), `cues(attribute: str) -> list[str]` (for boolean attributes: up to 5 lowercase word bigrams most specific to rows holding that attribute)
  - `class Extractor(lexicon: Lexicon, names: NameIndex, kinds: KindIndex, model, gold: dict[str, list[Fact]] | None = None, gold_rows: dict[str, Row] | None = None)` with `extract(rows: list[Row]) -> dict[str, list[Fact]]` (every row id present, possibly `[]`)

Pinned algorithm for `extract` (rows processed in input order; context is per `page`):
1. Gold reuse: if `row.id` in `gold` and `gold_rows[row.id]` has the same `text` and `date`, return the gold facts (confidence 1.0) and push their subjects onto the page context.
2. For each sentence: `mentions = lexicon.mentions(sentence)`; context stack = up to 3 most recent subjects of this page (most recent first).
3. Subject candidates for a value span: the nearest mention before the span, then other mentions in the sentence, then the context stack; prior 1.0 for an explicit mention, 0.6, 0.45, 0.35 for context positions 1–3.
4. Attribute candidates for (subject, span): `lexicon.attributes(subject)` whose `kinds.kind` equals the span's kind, or is `None` (attribute unseen in practice); for quantity spans with a known unit, keep attributes whose unit has the same dimension/noun.
5. Score = prior × `model.rank(sentence, span.value, subject, candidates)` top score; take the best (subject, attribute) per span; a (subject, attribute) pair is used once per sentence.
6. Name facts: each `names.find(sentence)` hit (not itself a mention of the chosen subject) is matched to name-kind attributes the same way.
7. Boolean facts: for each subject candidate, boolean attributes whose `has_attribute_words` or any `kinds.cues` bigram occurs in the sentence → `{"kind":"boolean","value": not has_negation(clause)}`.
8. Push explicit mentions onto the page context after the row.

- [ ] **Step 1: Write the failing tests**

```python
def test_gold_reuse_requires_same_text(practice):        # fixture: rows, gold from load_practice
    rows, gold = practice
    ex = make_extractor(gold=gold, gold_rows={r.id: r for r in rows})
    assert ex.extract([rows[0]])["p-1"] == gold["p-1"]
    changed = Row("p-1", "Holdan joined on 2 February 2026.", rows[0].date, "", 0)
    assert ex.extract([changed])["p-1"] != gold["p-1"]

def test_explicit_subject_date():
    out = make_extractor().extract([Row("a", "Holdan started at Hooly on 12 January 2026.", "2026-01-12", "", 0)])
    assert any(f.subject == "Holdan" and f.attribute == "joined" and f.value == {"kind": "date", "value": "2026-01-12"} for f in out["a"])

def test_pronoun_uses_page_context():
    rows = [Row("a", "Holdan started at Hooly on 12 January 2026.", "2026-01-12", "", 0),
            Row("b", "He begins with a small crew of 4 engineers, and he is hiring.", "2026-01-12", "", 0)]
    out = make_extractor().extract(rows)
    assert any(f.subject == "Holdan" and f.attribute == "team_size" and f.value["value"] == 4 for f in out["b"])

def test_boolean_polarity():
    out = make_extractor().extract([Row("a", "Hooly Chrome does not include an AI sidebar yet.", "2026-07-08", "", 0)])
    assert any(f.attribute == "ai_sidebar" and f.value == {"kind": "boolean", "value": False} for f in out["a"])

def test_every_row_returned():
    out = make_extractor().extract([Row("z", "Nothing to see here", "2026-01-01", "", 0)])
    assert out == {"z": []}
```

(`make_extractor` builds `KindIndex` and `NameIndex` from practice gold and uses the real `AttributeModel` fitted on practice gold once Task 6 exists; with the stub before that.)

- [ ] **Step 2: Run** `.venv/bin/pytest tests/test_extract.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement** `arena/kinds.py` and `arena/extract.py` per the pinned algorithm.
- [ ] **Step 4: Run** — Expected: all passed.
- [ ] **Step 5: Commit** `git add arena/kinds.py arena/extract.py tests/test_extract.py && git commit -m "feat: fact extractor with page context"`

---

### Task 6: Attribute ranker (`arena/attributes.py`)

**Files:**
- Create: `arena/attributes.py`, `tests/test_attributes.py`

**Interfaces:**
- Consumes: `parse_values` (Task 2), `attribute_words` (Task 3), `Fact` (Task 1).
- Produces: `class AttributeModel` with `fit(examples: list[tuple[str, dict, str, str]]) -> AttributeModel` (tuples of sentence text, value dict, subject, attribute), `rank(text: str, value: dict | None, subject: str, candidates: list[str]) -> list[tuple[str, float]]` (sorted descending, scores sum to 1 over candidates; empty candidates → `[]`), picklable.

Approach: input string = text with each parsed value span replaced by `__{KIND}__`, plus ` __SUBJ_{subject}__ __UNIT_{unit}__`; `FeatureUnion` of `TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True)` and `TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), sublinear_tf=True)`; `LogisticRegression(C=4.0, max_iter=2000)`. For a candidate in the model's classes use its probability; for a candidate not in the classes use `0.5 * overlap`, where overlap = share of `attribute_words(candidate)` tokens present in the lowercased text; renormalise over candidates (uniform when all zero).

- [ ] **Step 1: Write the failing tests**

```python
def test_rank_known_attribute(fitted):          # fitted on all practice gold facts
    r = fitted.rank("Hooly AI Pro costs €59 per month.", {"kind":"money","value":59,"currency":"EUR","per":"month"},
                    "Hooly AI Pro", ["price", "annual_price", "refund_window"])
    assert r[0][0] == "price"
def test_unseen_attribute_overlap(fitted):
    r = fitted.rank("Hooly Vault now has 12 vault shards.", {"kind":"quantity","value":12,"unit":"shard"},
                    "Hooly Vault", ["vault_shards", "minimum_nodes"])
    assert r[0][0] == "vault_shards" and abs(sum(s for _, s in r) - 1) < 1e-6
def test_leave_page_out_accuracy():
    # GroupKFold(5) by page over practice single-fact rows with gold subject given; candidates = subject's lexicon attributes of the gold kind
    assert attribute_accuracy() >= FLOOR   # FLOOR = measured value − 0.03, written into the test with the measured value in a comment
```

- [ ] **Step 2: Run** `.venv/bin/pytest tests/test_attributes.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement** `arena/attributes.py`; measure `attribute_accuracy()` and set `FLOOR`.
- [ ] **Step 4: Run** `.venv/bin/pytest tests/test_attributes.py tests/test_extract.py -v` — Expected: all passed (switch `make_extractor` in Task 5 tests to the real model).
- [ ] **Step 5: Commit** `git add arena/attributes.py tests/test_attributes.py tests/test_extract.py && git commit -m "feat: attribute ranker"`

---

### Task 7: Evidence store (`arena/evidence.py`)

**Files:**
- Create: `arena/evidence.py`, `tests/test_evidence.py`

**Interfaces:**
- Consumes: `Row`, `Fact`, `AS_OF` (Task 1); `values_equal`, `compare`, `satisfies` (Task 2).
- Produces:
  - `@dataclass class Interval: value: dict; start: str; end: str; pages: set[int]; trusted: bool; weight: float`
  - `@dataclass class AtomCheck: state: str; status: str; support: int; trusted: bool; days: int` — `state` ∈ `true|false|unknown`; `status` ∈ `in|gap|before|after|none`; `support` = distinct pages of the interval used; `days` = distance in days from the date to the nearest observation of that interval (0 inside)
  - `class EvidenceStore()` with `add_row(row: Row, facts: list[Fact]) -> None`, `build(iterations: int = 3) -> None`, `intervals(subject: str, attribute: str) -> list[Interval]`, `check(fact: Fact, date: str) -> AtomCheck`, `current(subject: str, attribute: str) -> Interval | None` (value holding on `AS_OF`: the last interval), `keys() -> list[tuple[str, str]]`, `page_reliability: dict[int, float]`

Pinned algorithm:
- Observation weight: label `true` → trusted, weight 3.0; label `''` → `page_reliability[page] * fact.confidence`; label `unsure` → `0.3 * page_reliability[page]`; label `false` with one fact → a refutation of (key, value, date), not a vote; label `false` with several facts → ignored. Observations whose value has an `op` other than eq never vote (they are only checked).
- Per key: per date, the winning value = largest summed weight among `values_equal` groups (trusted wins over untrusted on a date). Sequence of winners sorted by date.
- Lie filter: drop a winner that is untrusted, supported by one page, and whose previous and next winners exist and share a different value.
- Merge consecutive equal winners into `Interval`s.
- `check`: refutation for that key/value/date → `false`, trusted. Else locate the date: inside an interval → `in`; between intervals → `gap` (use the earlier interval when both sides share a value, else no interval → `unknown`); before the first → `before` (first interval); after the last → `after` (last interval); no intervals → `none`/`unknown`. With an interval: `satisfies(interval.value, fact.value)` → `true`, else `false`.
- Reliability: start 0.8 for every page; pages with labels start at `(n_true + 1) / (n_labelled + 2)`; each iteration recompute unlabelled pages as `(agree + 1.6) / (n + 2)` where agree counts their observations whose check (ignoring the observation itself is not required) is `true`.

- [ ] **Step 1: Write the failing tests**

```python
def price(v): return {"kind": "money", "value": v, "currency": "EUR", "per": "month"}
def store_with(obs):   # obs: list of (row_id, date, page, label, value)
    s = EvidenceStore()
    for rid, d, page, label, v in obs:
        s.add_row(Row(rid, "t", d, label, page), [Fact("Hooly AI Pro", "price", v)])
    s.build(); return s

def test_change_over_time():
    s = store_with([("a","2025-10-07",0,"true",price(49)),("b","2026-01-19",1,"",price(49)),("c","2026-03-18",2,"",price(49)),
                    ("d","2026-04-01",3,"true",price(59)),("e","2026-07-22",4,"",price(59))])
    assert s.check(Fact("Hooly AI Pro","price",price(49)), "2026-02-10").state == "true"
    assert s.check(Fact("Hooly AI Pro","price",price(59)), "2026-02-10").state == "false"
    assert s.check(Fact("Hooly AI Pro","price",price(49)), "2026-10-26").state == "false"
    assert [i.value["value"] for i in s.intervals("Hooly AI Pro","price")] == [49, 59]

def test_lone_lie_dropped():
    s = store_with([("a","2026-01-01",0,"",price(49)),("b","2026-02-01",1,"",price(99)),("c","2026-03-01",2,"",price(49))])
    assert [i.value["value"] for i in s.intervals("Hooly AI Pro","price")] == [49]

def test_trusted_beats_crowd_same_day():
    s = store_with([("a","2026-05-01",0,"true",price(59))] + [(f"x{i}","2026-05-01",i+1,"",price(69)) for i in range(3)])
    assert s.current("Hooly AI Pro","price").value["value"] == 59

def test_bound_and_refutation():
    s = EvidenceStore()
    q = lambda v, op=None: {"kind":"quantity","value":v,"unit":"customer", **({"op":op} if op else {})}
    s.add_row(Row("a","t","2026-01-12","true",0), [Fact("Hooly Cloud","customers",q(120))])
    s.add_row(Row("b","t","2026-01-13","false",1), [Fact("Hooly Cloud","customers",q(500))])
    s.build()
    assert s.check(Fact("Hooly Cloud","customers",q(100,"gt")), "2026-01-20").state == "true"
    c = s.check(Fact("Hooly Cloud","customers",q(500)), "2026-01-13")
    assert c.state == "false" and c.trusted
```

- [ ] **Step 2: Run** `.venv/bin/pytest tests/test_evidence.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement** `arena/evidence.py` per the pinned algorithm.
- [ ] **Step 4: Run** — Expected: all passed.
- [ ] **Step 5: Commit** `git add arena/evidence.py tests/test_evidence.py && git commit -m "feat: dated evidence store with intervals and reliability"`

---

### Task 8: Decider and offline training (`arena/decide.py`, `arena/train.py`)

**Files:**
- Create: `arena/decide.py`, `arena/train.py`, `tests/test_decide.py`

**Interfaces:**
- Consumes: Tasks 1–7.
- Produces:
  - `FEATURES: list[str]` — exactly: `n_facts, n_true, n_false, n_unknown, verdict_true, verdict_false, verdict_unknown, any_trusted_false, all_trusted_true, min_support, max_days, any_before, any_after, any_gap, any_none, min_fact_conf, page_rel, has_op, n_bool, n_name`
  - `row_features(facts: list[Fact], checks: list[AtomCheck], page_rel: float) -> dict[str, float]` (verdict: any check `false` → false; all `true` and ≥ 1 fact → true; else unknown; no facts → `verdict_unknown=1`, `n_facts=0`)
  - `class Decider` with `fit(X: list[dict], y: list[str], groups: list[int]) -> Decider`, `predict(x: dict) -> tuple[str, float | None]` (label, confidence; `None` for `unsure`), `cv_report: dict`; model `HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0)`; calibration: out-of-fold (GroupKFold 5) max-probability vs correctness → `IsotonicRegression(out_of_bounds="clip")`; confidence = `min(1.0, max(0.5, iso(p)))`
  - `arena.train.build_artifacts(practice_csv: str, lexicon_json: str, out_dir: str) -> dict` — writes `kinds.pkl`, `names.pkl`, `attribute_model.pkl`, `decider.pkl`, `metrics.json` (pickle); returns metrics with keys `label_accuracy`, `always_true_accuracy`, `false_precision`, `false_recall`, `unsure_rate`, `fact_accuracy`
  - CLI: `python -m arena.train [--out artifacts]`

Pinned training flow: GroupKFold(5) on practice pages. For each fold: fit `AttributeModel` on the other folds' gold facts; extract the fold's rows with gold reuse disabled for them; build an `EvidenceStore` over all practice rows where other-fold rows carry labels + gold facts and fold rows carry `label=""` + extracted facts; `row_features` for fold rows → out-of-fold matrix. `fact_accuracy` = share of fold gold facts reproduced exactly (subject, attribute, `values_equal` value). Fit `Decider` on the out-of-fold matrix; then refit `AttributeModel` on all gold and save everything.

- [ ] **Step 1: Write the failing tests**

```python
def test_row_features_verdicts():
    t = AtomCheck("true","in",3,True,0); f = AtomCheck("false","in",2,False,0)
    x = row_features([FACT, FACT], [t, f], 0.8)
    assert set(x) == set(FEATURES) and x["verdict_false"] == 1 and x["n_false"] == 1
    assert row_features([], [], 0.8)["verdict_unknown"] == 1

def test_build_artifacts(tmp_path):
    m = build_artifacts("data/practice.csv", "data/lexicon.json", str(tmp_path))
    for f in ["kinds.pkl","names.pkl","attribute_model.pkl","decider.pkl","metrics.json"]:
        assert (tmp_path / f).exists()
    assert 0.80 <= m["always_true_accuracy"] <= 0.86
    assert m["label_accuracy"] >= FLOOR      # FLOOR = measured − 0.02, measured value in a comment
```

- [ ] **Step 2: Run** `.venv/bin/pytest tests/test_decide.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement** `arena/decide.py` and `arena/train.py`; run `.venv/bin/python -m arena.train --out artifacts` and record the printed metrics in the commit message.
- [ ] **Step 4: Run** — Expected: all passed.
- [ ] **Step 5: Commit** `git add arena/decide.py arena/train.py tests/test_decide.py && git commit -m "feat: decider and offline training (label_accuracy=<value>)"`

---

### Task 9: Claim generation (`arena/generate.py`)

**Files:**
- Create: `arena/generate.py`, `tests/test_generate.py`

**Interfaces:**
- Consumes: `EvidenceStore`, `Interval` (Task 7); `render`, `compare` (Task 2); `Lexicon`, `attribute_words`, `NameIndex` (Task 3); `check_claim` (Task 4); `KindIndex` (Task 5); `AS_OF` (Task 1).
- Produces:
  - `@dataclass class Candidate: type: str; form: dict; text: str; score: float; subject: str`
  - `enumerate_candidates(store: EvidenceStore, lexicon: Lexicon, kinds: KindIndex, names: NameIndex) -> list[Candidate]`
  - `select_claims(cands: list[Candidate], lexicon: Lexicon, input_texts: set[str], n: int = 1000, per_subject_cap: int = 80) -> list[dict]` → dicts `{id, type, text, form, confidence}` with ids `c-0001`…; keeps only candidates with `check_claim(entry, lexicon) == []` (entry built with `tag="true"`); drops texts whose normalisation (lowercase, collapsed spaces, trailing `.!? ` stripped) is in `input_texts` (already normalised the same way); dedupes on form (a compare and its reverse with flipped op are one); sorts by score descending; caps per subject (left subject for compare)

Pinned rules (`A` = attribute words, `V` = `render(value)`):

| Type | Candidate when | Template | Base |
| --- | --- | --- | --- |
| fact | `store.current(S,a)` exists, value kind not boolean/name-unresolved, interval trusted or ≥ 3 pages, last observation within 150 days of `AS_OF` | `The {A} of {S} is {V}.` | 0.80 |
| fact (boolean) | same, kind boolean | true: `{S} has {A}.` false: `{S} does not have {A}.` | 0.75 |
| change | consecutive intervals of a key with different values, each trusted or ≥ 2 pages; not boolean | `The {A} of {S} went from {V_from} to {V_to}.` | 0.90 |
| compare | same attribute, two subjects whose `current` values pass the fact rule, `compare` not None; amounts need relative margin ≥ 10% (eq needs exact equality) | amounts: `The {A} of {L} is higher than / lower than / equal to the {A} of {R}.`; dates: `… is later than / earlier than / on the same day as …` (same-day only for day precision) | 0.85 |
| at_event | event subject with a `date` interval of day precision that is `current` and trusted or ≥ 2 pages; any key whose interval contains that date with status `in` | `The {A} of {S} was {V} at {E}.` | 0.85 |
| relation | link = name-kind attribute of S whose interval value canonicalises (via `names`/`lexicon.canonical`) to subject T; any (T, a) interval overlapping the link interval | `The {A} of the {link words} of {S} was {V}.` | 0.85 |

score = base × min(1, 0.6 + 0.1 × distinct pages of the weakest interval used) × (1.0 if all used intervals trusted else 0.9); confidence = min(0.99, max(0.5, score)).

- [ ] **Step 1: Write the failing tests** (build a store from rows: AI Pro price 49 → 59 with two pages each, one trusted; Workspace price 199 trusted on 2026-10-07; Hooly cto Bertram Gilfoil trusted 2026-07-02; Bertram Gilfoil joined 2026-07-01 trusted; Hooly I/O 2026 date 2026-03-24 trusted; Hooly Gemini languages 12 on 2026-03-24 and 2026-08-20 trusted)

```python
def test_expected_claims(store, L, kinds, names):
    texts = {c.text for c in enumerate_candidates(store, L, kinds, names)}
    assert "The price of Hooly AI Pro went from €49 per month to €59 per month." in texts
    assert "The price of Hooly Workspace is higher than the price of Hooly AI Pro." in texts
    assert "The languages of Hooly Gemini was 12 languages at Hooly I/O 2026." in texts
    assert "The joined of the cto of Hooly was 1 July 2026." in texts

def test_select_validates_dedupes_and_novelty(store, L, kinds, names):
    cands = enumerate_candidates(store, L, kinds, names)
    taken = select_claims(cands, L, {"the price of hooly ai pro went from €49 per month to €59 per month"})
    assert all(check_claim({**c, "tag": "true"}, L) == [] for c in taken)
    assert not any(c["type"] == "change" and c["form"]["subject"] == "Hooly AI Pro" for c in taken)
    pairs = [(c["form"]["left"]["subject"], c["form"]["right"]["subject"]) for c in taken if c["type"] == "compare"]
    assert len({frozenset(p) for p in pairs}) == len(pairs)
    assert len({c["id"] for c in taken}) == len(taken) and len(taken) <= 1000

def test_subject_cap(store, L, kinds, names):
    taken = select_claims(enumerate_candidates(store, L, kinds, names), L, set(), per_subject_cap=1)
    subs = [c["form"].get("subject") or c["form"]["left"]["subject"] for c in taken]
    assert len(subs) == len(set(subs))
```

If a pinned template fails `check_claim` for a case in this test, fix `claimforms` only if it contradicts `contracts/claim-forms.md`; otherwise adjust the template and update this table and test together.

- [ ] **Step 2: Run** `.venv/bin/pytest tests/test_generate.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement** `arena/generate.py`.
- [ ] **Step 4: Run** — Expected: all passed.
- [ ] **Step 5: Commit** `git add arena/generate.py tests/test_generate.py && git commit -m "feat: claim generation with local referee"`

---

### Task 10: Pipeline and batch CLI (`arena/pipeline.py`, `arena/cli.py`)

**Files:**
- Create: `arena/pipeline.py`, `arena/cli.py`, `tests/test_pipeline.py`

**Interfaces:**
- Consumes: Tasks 1–9.
- Produces:
  - `@dataclass class Artifacts: kinds: KindIndex; names: NameIndex; model: AttributeModel; decider: Decider; gold: dict[str, list[Fact]]; gold_rows: dict[str, Row]`
  - `load_artifacts(art_dir: str, practice_csv: str) -> Artifacts`
  - `@dataclass class RunResult: labels: list[tuple[str, str, float | None]]; claims: list[dict]; traces: dict[str, dict]` (trace per row id: `facts` as dicts, `checks` as dicts, `features`)
  - `run(rows: list[Row], lexicon: Lexicon, art: Artifacts, background: list[Row] | None = None, generate: bool = True, n_claims: int = 1000) -> RunResult` — labels only for rows with `label == ""` from `rows` (background rows are evidence only); extraction runs over background + rows together
  - `arena.cli.main(argv: list[str] | None = None) -> int` with options `--universe /universe --output /output --artifacts /app/artifacts --practice /app/data/practice.csv`; sequence: install SIGTERM handler (writes nothing new, exits 0 after the current safe write); write safe table (every unlabelled row `true`, 0.6; no claims) within 60 s; if `ARENA_DEADLINE` − now < 120 s stop; run labels; write; if time remains > 120 s generate claims; write; return 0

- [ ] **Step 1: Write the failing tests**

```python
def test_end_to_end_practice_shape(tmp_path):
    uni = tmp_path / "universe"; out = tmp_path / "output"; uni.mkdir(); out.mkdir()
    subprocess.run([sys.executable, "scripts/input.py", "data/practice.csv", str(uni / "universe.csv")], check=True)
    shutil.copy("data/lexicon.json", uni / "lexicon.json")
    assert main(["--universe", str(uni), "--output", str(out), "--artifacts", ART, "--practice", "data/practice.csv"]) == 0
    res = subprocess.run([sys.executable, "scripts/check_output.py", str(out), "data/practice.csv", str(uni / "universe.csv")], capture_output=True, text=True)
    problems = [l for l in res.stdout.splitlines() if l.startswith("  ")]
    assert all("new claims, not exactly" in p for p in problems), res.stdout
    assert "380 of the 380 hidden rows labelled" in res.stdout

def test_unparseable_row_gets_label(art, L):
    r = run([Row("z", "Nothing to see here", "2026-01-01", "", 0)], L, art, generate=False)
    assert r.labels[0][0] == "z" and r.labels[0][1] in ("true", "false", "unsure")

def test_deadline_passed_leaves_safe_output(tmp_path, monkeypatch):
    # universe with 3 unlabelled rows; ARENA_DEADLINE = now − 1
    monkeypatch.setenv("ARENA_DEADLINE", str(int(time.time()) - 1))
    assert main([...]) == 0
    rows = list(csv.DictReader(open(out / "output.csv")))
    assert len(rows) == 3 and all(r["label"] == "true" and r["confidence"] == "0.6" for r in rows)
```

(`ART` = an `artifacts` dir built once per session by a fixture calling `build_artifacts`.)

- [ ] **Step 2: Run** `.venv/bin/pytest tests/test_pipeline.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement** `arena/pipeline.py` and `arena/cli.py` (`if __name__ == "__main__": sys.exit(main())`).
- [ ] **Step 4: Run** `.venv/bin/pytest -v` (whole suite) — Expected: all passed; note claim count printed by the checker.
- [ ] **Step 5: Commit** `git add arena/pipeline.py arena/cli.py tests/test_pipeline.py && git commit -m "feat: pipeline and batch CLI with safe output"`

---

### Task 11: Live server and demo page (`arena/server.py`, `web/`)

**Files:**
- Create: `arena/server.py`, `web/index.html`, `web/app.js`, `web/styles.css`, `tests/test_server.py`

**Interfaces:**
- Consumes: `load_artifacts`, `run` (Task 10); `Lexicon` (Task 3); `load_practice` (Task 1).
- Produces: FastAPI `app` (run: `python -m arena.server`, port from `PORT`, default 8000, host 0.0.0.0); startup loads `data/lexicon.json`, artifacts dir from `ARENA_ARTIFACTS` (default `artifacts`), practice rows as `background`.
  - `GET /api/health` → `{"ok": true, "rows_background": int}`
  - `GET /api/sample` → `{"csv": str}` — `sample/sample.csv` if present, else the first 20 practice rows with `label` emptied (columns `id,text,date,label`)
  - `POST /api/run` body `{"csv": str}` (columns `id,text,date,label`) → `{"rows": [{"id","text","date","label","confidence","facts":[...],"checks":[...]}], "claims": [first 20 claims], "timings": {"extract_ms","evidence_ms","decide_ms","generate_ms"}}`; bad CSV → 400 with `{"detail": str}`
  - `GET /` → `web/index.html`; `/static/*` → `web/`

Page (use the frontend-design skill for the visual direction): left column = the architecture diagram as inline SVG with five stages (Extract, Evidence store, Decide, Generate, Output) that highlight in turn while a run is in flight; right column = CSV textarea preloaded from `/api/sample`, a Run button, a results table (id, text, label chip, confidence bar), and a detail panel for the selected row showing extracted facts and each check's state/status/support. Works offline (no CDN), at 1280×720 for the recording and at phone width.

- [ ] **Step 1: Write the failing tests**

```python
client = TestClient(app)
def test_health_and_sample():
    assert client.get("/api/health").json()["ok"] is True
    assert client.get("/api/sample").json()["csv"].startswith("id,text,date,label")
def test_run_returns_labels():
    csv_text = "id,text,date,label\ns-1,Hooly AI Pro costs €59 per month.,2026-05-10,\n"
    body = client.post("/api/run", json={"csv": csv_text}).json()
    assert body["rows"][0]["id"] == "s-1" and body["rows"][0]["label"] in ("true", "false", "unsure")
    assert body["rows"][0]["facts"][0]["attribute"] == "price"
def test_bad_csv():
    assert client.post("/api/run", json={"csv": "nope"}).status_code == 400
def test_index_served():
    r = client.get("/"); assert r.status_code == 200 and "<svg" in r.text
```

- [ ] **Step 2: Run** `.venv/bin/pytest tests/test_server.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement** `arena/server.py` and the `web/` files.
- [ ] **Step 4: Run** `.venv/bin/pytest tests/test_server.py -v` — Expected: all passed. Then `ARENA_ARTIFACTS=artifacts .venv/bin/python -m arena.server` and load `http://localhost:8000` — Expected: page renders, Run fills the table.
- [ ] **Step 5: Commit** `git add arena/server.py web tests/test_server.py && git commit -m "feat: live demo server and page"`

---

### Task 12: Docker image and full check

**Files:**
- Create: `Dockerfile`, `.dockerignore`
- Modify: `README.md` (append a "Running our system" section)

**Interfaces:**
- Consumes: everything.
- Produces: image whose default command runs `python -m arena.cli` with defaults `/universe`, `/output`, `/app/artifacts`, `/app/data/practice.csv`; `docker run -p 8000:8000 <image> python -m arena.server` serves the demo.

Dockerfile essentials: `FROM --platform=linux/amd64 python:3.12-slim`; `WORKDIR /app`; `ENV HOME=/tmp XDG_CACHE_HOME=/tmp PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 ARENA_ARTIFACTS=/app/artifacts`; `pip install --no-cache-dir -r requirements.txt`; copy `arena/ web/ data/ contracts/`; `RUN python -m arena.train --out /app/artifacts`; `RUN chmod -R a+rX /app`; `CMD ["python", "-m", "arena.cli"]`; last line `USER 10001`. `.dockerignore`: `.venv`, `.git`, `artifacts`, `tests`, `docs`, `*.pdf`, `__pycache__`.

- [ ] **Step 1: Build** `docker build --platform linux/amd64 -t hype-arena .` — Expected: success; `docker image inspect hype-arena --format '{{.Size}}'` < 3221225472.
- [ ] **Step 2: Run the official check** `MINUTES=10 scripts/check.sh hype-arena` — Expected: exit code 0 from the container, "380 of the 380 hidden rows labelled", only a claim-count problem allowed if fewer than 1,000 valid claims; record wall time.
- [ ] **Step 3: Demo smoke test** `docker run --rm -d -p 8000:8000 --name ha-demo hype-arena python -m arena.server && curl -s localhost:8000/api/health && docker rm -f ha-demo` — Expected: `{"ok":true,...}`.
- [ ] **Step 4: README** — how to build, run the batch, run the demo, and run tests.
- [ ] **Step 5: Commit** `git add Dockerfile .dockerignore README.md && git commit -m "feat: docker image and run docs"`

---

## Execution order and parallelism

- Wave 1: Task 1.
- Wave 2 (parallel): Task 2, Task 3.
- Wave 3 (parallel): Task 4, Task 6, Task 7 (Task 7 only needs Task 2's interface).
- Wave 4: Task 5 (needs 2, 3, 6).
- Wave 5: Task 8, then Task 9 (Task 9 can start after 4 and 7, in parallel with 8).
- Wave 6: Task 10; then Task 11 and Task 12 in parallel.
