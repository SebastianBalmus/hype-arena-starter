from arena.evidence import EvidenceStore
from arena.model import Fact, Row


def price(v): return {"kind": "money", "value": v, "currency": "EUR", "per": "month"}


def store_with(obs):   # obs: list of (row_id, date, page, label, value)
    s = EvidenceStore()
    for rid, d, page, label, v in obs:
        s.add_row(Row(rid, "t", d, label, page), [Fact("Hooly AI Pro", "price", v)])
    s.build()
    return s


def test_change_over_time():
    s = store_with([("a", "2025-10-07", 0, "true", price(49)), ("b", "2026-01-19", 1, "", price(49)),
                    ("c", "2026-03-18", 2, "", price(49)),
                    ("d", "2026-04-01", 3, "true", price(59)), ("e", "2026-07-22", 4, "", price(59))])
    assert s.check(Fact("Hooly AI Pro", "price", price(49)), "2026-02-10").state == "true"
    assert s.check(Fact("Hooly AI Pro", "price", price(59)), "2026-02-10").state == "false"
    assert s.check(Fact("Hooly AI Pro", "price", price(49)), "2026-10-26").state == "false"
    assert [i.value["value"] for i in s.intervals("Hooly AI Pro", "price")] == [49, 59]


def test_lone_lie_dropped():
    s = store_with([("a", "2026-01-01", 0, "", price(49)), ("b", "2026-02-01", 1, "", price(99)),
                    ("c", "2026-03-01", 2, "", price(49))])
    assert [i.value["value"] for i in s.intervals("Hooly AI Pro", "price")] == [49]


def test_trusted_beats_crowd_same_day():
    s = store_with([("a", "2026-05-01", 0, "true", price(59))] +
                   [(f"x{i}", "2026-05-01", i + 1, "", price(69)) for i in range(3)])
    assert s.current("Hooly AI Pro", "price").value["value"] == 59


def test_bound_and_refutation():
    s = EvidenceStore()
    q = lambda v, op=None: {"kind": "quantity", "value": v, "unit": "customer", **({"op": op} if op else {})}
    s.add_row(Row("a", "t", "2026-01-12", "true", 0), [Fact("Hooly Cloud", "customers", q(120))])
    s.add_row(Row("b", "t", "2026-01-13", "false", 1), [Fact("Hooly Cloud", "customers", q(500))])
    s.build()
    assert s.check(Fact("Hooly Cloud", "customers", q(100, "gt")), "2026-01-20").state == "true"
    c = s.check(Fact("Hooly Cloud", "customers", q(500)), "2026-01-13")
    assert c.state == "false" and c.trusted


def test_status_and_none():
    s = store_with([("a", "2026-03-01", 0, "true", price(49)), ("b", "2026-04-01", 1, "true", price(49))])
    f = Fact("Hooly AI Pro", "price", price(49))
    assert s.check(f, "2026-03-15").status == "in"
    c = s.check(f, "2026-02-01")
    assert c.status == "before" and c.state == "true" and c.days == 28
    assert s.check(f, "2026-05-01").status == "after"
    n = s.check(Fact("X", "y", price(1)), "2026-05-01")
    assert (n.state, n.status) == ("unknown", "none")
    assert s.keys() == [("Hooly AI Pro", "price")]


def test_gap_different_values_unknown():
    s = store_with([("a", "2026-01-01", 0, "true", price(49)), ("b", "2026-06-01", 1, "true", price(59))])
    c = s.check(Fact("Hooly AI Pro", "price", price(49)), "2026-03-01")
    assert (c.state, c.status) == ("unknown", "gap")


def test_reliability_labelled_pages():
    s = store_with([("a", "2026-01-01", 0, "true", price(49)), ("b", "2026-02-01", 0, "true", price(49))])
    assert abs(s.page_reliability[0] - 3 / 4) < 1e-9
