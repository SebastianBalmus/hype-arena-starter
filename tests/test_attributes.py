import pickle

import pytest
from sklearn.model_selection import GroupKFold

from arena.attributes import AttributeModel
from arena.data import load_practice
from arena.lexicon import Lexicon

PRACTICE = "data/practice.csv"
LEXICON = "data/lexicon.json"


def _examples(rows, gold):
    by_id = {r.id: r for r in rows}
    return [(by_id[i].text, f.value, f.subject, f.attribute) for i, fs in gold.items() if i in by_id for f in fs]


@pytest.fixture(scope="module")
def fitted():
    rows, gold = load_practice(PRACTICE)
    return AttributeModel().fit(_examples(rows, gold))


def attribute_accuracy() -> float:
    rows, gold = load_practice(PRACTICE)
    lex = Lexicon.load(LEXICON)
    single = [r for r in rows if len(gold[r.id]) == 1]
    groups = [r.page for r in single]
    hits = total = 0
    for tr, te in GroupKFold(5).split(single, groups=groups):
        train_ids = {single[i].id for i in tr}
        sub = [r for r in rows if r.id in train_ids]
        kinds = {}
        for r in sub:
            for f in gold[r.id]:
                kinds[f.attribute] = f.value.get("kind")
        model = AttributeModel().fit(_examples(sub, gold))
        for i in te:
            r, f = single[i], gold[single[i].id][0]
            k = f.value.get("kind")
            cands = [a for a in lex.attributes(f.subject) if kinds.get(a, k) == k]
            if f.attribute not in cands:
                cands.append(f.attribute)
            hits += model.rank(r.text, f.value, f.subject, cands)[0][0] == f.attribute
            total += 1
    return hits / total


def test_rank_known_attribute(fitted):
    r = fitted.rank("Hooly AI Pro costs €59 per month.",
                    {"kind": "money", "value": 59, "currency": "EUR", "per": "month"},
                    "Hooly AI Pro", ["price", "annual_price", "refund_window"])
    assert r[0][0] == "price"


def test_unseen_attribute_overlap(fitted):
    r = fitted.rank("Hooly Vault now has 12 vault shards.", {"kind": "quantity", "value": 12, "unit": "shard"},
                    "Hooly Vault", ["vault_shards", "minimum_nodes"])
    assert r[0][0] == "vault_shards" and abs(sum(s for _, s in r) - 1) < 1e-6


def test_empty_and_pickle(fitted):
    assert fitted.rank("x", None, "S", []) == []
    m = pickle.loads(pickle.dumps(fitted))
    r = m.rank("Hooly AI Pro costs €59 per month.", {"kind": "money", "value": 59}, "Hooly AI Pro",
               ["price", "annual_price"])
    assert abs(sum(s for _, s in r) - 1) < 1e-6


def test_leave_page_out_accuracy():
    acc = attribute_accuracy()
    FLOOR = 0.887  # measured 0.9168 minus 0.03
    assert acc >= FLOOR
