import json

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


def _build(rows, gold, model, use_gold=False, lexicon_path=LEXICON):
    lex = Lexicon.load(lexicon_path)
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


def fact_scores(rows, gold) -> tuple[float, float]:
    """(recall, precision) on held-out page folds, no gold reuse, model fitted on the other folds."""
    hits = total = extracted = correct = 0
    for tr, te in GroupKFold(5).split(rows, groups=[r.page for r in rows]):
        train = [rows[i] for i in tr]
        model = AttributeModel().fit(_examples(train, gold))
        ex = _build(train, gold, model)
        test = [rows[i] for i in te]
        out = ex.extract(test)

        def match(f, g):
            return f.subject == g.subject and f.attribute == g.attribute and values_equal(f.value, g.value)
        for r in test:
            for g in gold[r.id]:
                total += 1
                hits += any(match(f, g) for f in out[r.id])
            for f in out[r.id]:
                extracted += 1
                correct += any(match(f, g) for g in gold[r.id])
    return hits / total, correct / extracted


def fact_accuracy(rows, gold) -> float:
    return fact_scores(rows, gold)[0]


def test_fact_accuracy_floor(practice):
    rows, gold = practice
    recall, precision = fact_scores(rows, gold)
    print(f"fact_accuracy(recall)={recall:.4f} precision={precision:.4f}")
    # measured recall 0.7540, precision 0.7912 (GroupKFold(5) by page, no gold reuse); floors = measured - 0.03
    assert recall >= 0.7240
    assert precision >= 0.7612


def _lexicon_with(tmp_path, subject, attribute):
    data = json.load(open(LEXICON, encoding="utf-8"))
    for s in data["subjects"]:
        if s["name"] == subject:
            s["attributes"].append(attribute)
    path = tmp_path / "lexicon.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def test_unseen_quantity_attribute(practice, fitted, tmp_path):
    rows, gold = practice
    path = _lexicon_with(tmp_path, "Hooly Vault", "vault_shards")
    ex = _build(rows, gold, fitted, lexicon_path=path)
    out = ex.extract([Row("u", "Hooly Vault now runs 12 vault shards.", "2026-05-01", "", 0)])
    assert any(f.subject == "Hooly Vault" and f.attribute == "vault_shards" and f.value["value"] == 12
               and f.value["kind"] == "quantity" for f in out["u"])


def test_unseen_boolean_attribute(practice, fitted, tmp_path):
    rows, gold = practice
    path = _lexicon_with(tmp_path, "Hooly Vault", "cold_storage")
    ex = _build(rows, gold, fitted, lexicon_path=path)
    out = ex.extract([Row("u", "Hooly Vault does not offer cold storage.", "2026-05-01", "", 0),
                      Row("v", "Hooly Vault offers cold storage.", "2026-05-01", "", 1)])
    assert any(f.attribute == "cold_storage" and f.value == {"kind": "boolean", "value": False} for f in out["u"])
    assert any(f.attribute == "cold_storage" and f.value == {"kind": "boolean", "value": True} for f in out["v"])
