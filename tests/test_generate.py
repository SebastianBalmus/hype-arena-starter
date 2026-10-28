import pytest

from arena.claimforms import check_claim
from arena.data import load_practice
from arena.evidence import EvidenceStore
from arena.generate import enumerate_candidates, select_claims
from arena.kinds import KindIndex
from arena.lexicon import Lexicon, NameIndex
from arena.model import Fact, Row

L = Lexicon.load("data/lexicon.json")


def price(v):
    return {"kind": "money", "value": v, "currency": "EUR", "per": "month"}


def date(v):
    return {"kind": "date", "value": v}


def name(v):
    return {"kind": "name", "value": v}


def lang(v):
    return {"kind": "quantity", "value": v, "unit": "language"}


@pytest.fixture(scope="module")
def parts():
    obs = [
        # (id, date, page, label, subject, attribute, value)
        ("a1", "2026-01-10", 0, "true", "Hooly AI Pro", "price", price(49)),
        ("a2", "2026-02-10", 1, "", "Hooly AI Pro", "price", price(49)),
        ("a3", "2026-04-01", 2, "true", "Hooly AI Pro", "price", price(59)),
        ("a4", "2026-09-01", 3, "", "Hooly AI Pro", "price", price(59)),
        ("b1", "2026-10-07", 4, "true", "Hooly Workspace", "price", price(199)),
        ("c1", "2026-07-02", 5, "true", "Hooly", "cto", name("Bertram Gilfoil")),
        ("d1", "2026-07-01", 6, "true", "Bertram Gilfoil", "joined", date("2026-07-01")),
        ("e1", "2026-03-24", 7, "true", "Hooly I/O 2026", "date", date("2026-03-24")),
        ("g1", "2026-03-24", 7, "true", "Hooly Gemini", "languages", lang(12)),
        ("g2", "2026-08-20", 8, "true", "Hooly Gemini", "languages", lang(12)),
    ]
    store = EvidenceStore()
    gold = {}
    for rid, d, page, label, s, a, v in obs:
        store.add_row(Row(rid, "t", d, label, page), [Fact(s, a, v)])
        gold[rid] = [Fact(s, a, v)]
    store.build()
    kinds = KindIndex.from_gold(gold)
    names = NameIndex.from_values(["Bertram Gilfoil"], L)
    return store, kinds, names


@pytest.fixture
def store(parts):
    return parts[0]


@pytest.fixture
def kinds(parts):
    return parts[1]


@pytest.fixture
def names(parts):
    return parts[2]


def test_expected_claims(store, kinds, names):
    texts = {c.text for c in enumerate_candidates(store, L, kinds, names)}
    assert "The price of Hooly AI Pro went from €49 per month to €59 per month." in texts
    assert "The price of Hooly Workspace is higher than the price of Hooly AI Pro." in texts
    assert "The languages of Hooly Gemini was 12 languages at Hooly I/O 2026." in texts
    assert "The joined of the cto of Hooly was 1 July 2026." in texts


def test_select_validates_dedupes_and_novelty(store, kinds, names):
    cands = enumerate_candidates(store, L, kinds, names)
    taken = select_claims(cands, L, {"the price of hooly ai pro went from €49 per month to €59 per month"})
    assert all(check_claim({**c, "tag": "true"}, L) == [] for c in taken)
    assert not any(c["type"] == "change" and c["form"]["subject"] == "Hooly AI Pro" for c in taken)
    pairs = [(c["form"]["left"]["subject"], c["form"]["right"]["subject"]) for c in taken if c["type"] == "compare"]
    assert len({frozenset(p) for p in pairs}) == len(pairs)
    assert len({c["id"] for c in taken}) == len(taken) and len(taken) <= 1000
    assert [c["id"] for c in taken][:2] == ["c-0001", "c-0002"]
    assert all(0.5 <= c["confidence"] <= 1 and len(c["text"]) <= 300 for c in taken)


def test_subject_cap(store, kinds, names):
    taken = select_claims(enumerate_candidates(store, L, kinds, names), L, set(), per_subject_cap=1)
    subs = [c["form"].get("subject") or c["form"]["left"]["subject"] for c in taken]
    assert len(subs) == len(set(subs))


def test_practice_universe_regression_guard():
    rows, gold = load_practice("data/practice.csv")
    store = EvidenceStore()
    for r in rows:
        store.add_row(r, gold.get(r.id, []))
    store.build()
    kinds = KindIndex.from_gold(gold, rows)
    values = [f.value["value"] for fs in gold.values() for f in fs if f.value.get("kind") == "name"]
    names = NameIndex.from_values(values, L)
    cands = enumerate_candidates(store, L, kinds, names)
    texts = {" ".join(r.text.lower().split()).rstrip(".!? ") for r in rows}
    taken = select_claims(cands, L, texts, n=100000, kinds=kinds)
    assert all(check_claim({**c, "tag": "true"}, L, kinds=kinds) == [] for c in taken)
    # measured 1318 on the practice universe: at_event 481, fact 382, change 299, compare 131, relation 25
    assert len(taken) >= 659
