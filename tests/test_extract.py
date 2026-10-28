import pytest
from sklearn.model_selection import GroupKFold

from arena.attributes import AttributeModel
from arena.data import load_practice
from arena.extract import Extractor
from arena.kinds import KindIndex
from arena.lexicon import Lexicon, NameIndex
from arena.model import Row
from arena.values import values_equal

PRACTICE = "data/practice.csv"
LEXICON = "data/lexicon.json"


def _examples(rows, gold):
    by_id = {r.id: r for r in rows}
    return [(by_id[i].text, f.value, f.subject, f.attribute) for i, fs in gold.items() if i in by_id for f in fs]


def _build(rows, gold, model, use_gold=False):
    lex = Lexicon.load(LEXICON)
    sub = {r.id: gold[r.id] for r in rows if r.id in gold}
    kinds = KindIndex.from_gold(sub, rows)
    names = NameIndex.from_values(
        [f.value["value"] for fs in sub.values() for f in fs if f.value["kind"] == "name"], lex)
    if use_gold:
        return Extractor(lex, names, kinds, model, gold=sub, gold_rows={r.id: r for r in rows})
    return Extractor(lex, names, kinds, model)


@pytest.fixture(scope="module")
def practice():
    return load_practice(PRACTICE)


@pytest.fixture(scope="module")
def fitted(practice):
    rows, gold = practice
    return AttributeModel().fit(_examples(rows, gold))


@pytest.fixture(scope="module")
def make_extractor(practice, fitted):
    rows, gold = practice

    def make(gold_reuse=False):
        return _build(rows, gold, fitted, use_gold=gold_reuse)
    return make


def test_gold_reuse_requires_same_text(practice, make_extractor):
    rows, gold = practice
    ex = make_extractor(gold_reuse=True)
    assert ex.extract([rows[0]])[rows[0].id] == gold[rows[0].id]
    changed = Row(rows[0].id, "Holdan joined on 2 February 2026.", rows[0].date, "", 0)
    assert ex.extract([changed])[rows[0].id] != gold[rows[0].id]


def test_explicit_subject_date(make_extractor):
    out = make_extractor().extract([Row("a", "Holdan started at Hooly on 12 January 2026.", "2026-01-12", "", 0)])
    assert any(f.subject == "Holdan" and f.attribute == "joined"
               and f.value == {"kind": "date", "value": "2026-01-12"} for f in out["a"])


def test_pronoun_uses_page_context(make_extractor):
    rows = [Row("a", "Holdan started at Hooly on 12 January 2026.", "2026-01-12", "", 0),
            Row("b", "He begins with a small crew of 4 engineers, and he is hiring.", "2026-01-12", "", 0)]
    out = make_extractor().extract(rows)
    assert any(f.subject == "Holdan" and f.attribute == "team_size" and f.value["value"] == 4 for f in out["b"])


def test_boolean_polarity(make_extractor):
    out = make_extractor().extract([Row("a", "Hooly Chrome does not include an AI sidebar yet.", "2026-07-08", "", 0)])
    assert any(f.attribute == "ai_sidebar" and f.value == {"kind": "boolean", "value": False} for f in out["a"])


def test_every_row_returned(make_extractor):
    out = make_extractor().extract([Row("z", "Nothing to see here", "2026-01-01", "", 0)])
    assert out == {"z": []}


def fact_accuracy(rows, gold) -> float:
    """Share of gold facts reproduced exactly, extracting without gold reuse, model fitted on other page folds."""
    hits = total = 0
    for tr, te in GroupKFold(5).split(rows, groups=[r.page for r in rows]):
        train = [rows[i] for i in tr]
        model = AttributeModel().fit(_examples(train, gold))
        ex = _build(train, gold, model)
        test = [rows[i] for i in te]
        out = ex.extract(test)
        for r in test:
            for g in gold[r.id]:
                total += 1
                hits += any(f.subject == g.subject and f.attribute == g.attribute and values_equal(f.value, g.value)
                            for f in out[r.id])
    return hits / total


def test_fact_accuracy_floor(practice):
    rows, gold = practice
    acc = fact_accuracy(rows, gold)
    print(f"fact_accuracy={acc:.4f}")
    # measured 0.7523 (GroupKFold(5) by page, no gold reuse); floor = measured - 0.03
    assert acc >= 0.7223
