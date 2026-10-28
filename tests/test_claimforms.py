import glob
import json

import pytest

from arena.claimforms import check_claim
from arena.lexicon import Lexicon

L = Lexicon.load("data/lexicon.json")


def eur_month(v):
    return {"kind": "money", "value": v, "currency": "EUR", "per": "month"}


PRICE_FACT = {"subject": "Hooly AI Pro", "attribute": "price", "value": eur_month(59)}
SOC2_FACT = {"subject": "Hooly", "attribute": "soc2_certified", "value": {"kind": "boolean", "value": False}}


def price_compare(op):
    return {
        "left": {"subject": "Hooly Workspace", "attribute": "price"},
        "right": {"subject": "Hooly AI Pro", "attribute": "price"},
        "op": op,
    }


def date_compare(op):
    return {
        "left": {"subject": "Hooly I/O 2026", "attribute": "date"},
        "right": {"subject": "The May 2026 outage", "attribute": "date"},
        "op": op,
    }


CHANGE = {"subject": "Hooly AI Pro", "attribute": "price", "from": eur_month(49), "to": eur_month(59)}
AT_EVENT = {
    "subject": "Hooly Gemini",
    "attribute": "languages",
    "event": "Hooly I/O 2026",
    "value": {"kind": "quantity", "value": 12, "unit": "language"},
}
RELATION = {
    "subject": "Hooly",
    "link": "cto",
    "target": "Bertram Gilfoil",
    "attribute": "joined",
    "value": {"kind": "date", "value": "2026-07-01"},
}

# Every row of every example table of contracts/claim-forms.md.
CASES = [
    # Fact: Hooly AI Pro's price, €59 per month
    ("fact", PRICE_FACT, "Hooly AI Pro costs €59 per month.", True),
    ("fact", PRICE_FACT, "At €59 per month, Hooly AI Pro is our best seller.", True),
    ("fact", PRICE_FACT, "Hooly AI Pro costs over €59 per month.", False),
    ("fact", PRICE_FACT, "Hooly AI Pro costs about €59 per month.", False),
    ("fact", PRICE_FACT, "Hooly AI Pro costs €59.", False),
    ("fact", PRICE_FACT, "It costs €59 per month.", False),
    ("fact", PRICE_FACT, "Hooly AI Pro is our plan. It costs €59 per month.", False),
    ("fact", PRICE_FACT, "Hooly costs €59 per month.", False),
    ("fact", PRICE_FACT, "Hooly AI Pro costs €59 per month, and never more.", False),
    ("fact", PRICE_FACT, "Hooly AI Pro costs €59 per month or €590 per year.", True),
    ("fact", PRICE_FACT, "Hooly AI Pro costs €59 per month, up from €49 per month.", False),
    ("fact", PRICE_FACT, "In March 2026, Hooly AI Pro cost €59 per month.", False),
    ("fact", PRICE_FACT, "In April, Hooly AI Pro cost €59 per month.", False),
    ("fact", PRICE_FACT, "Hooly AI Pro used to cost €59 per month.", False),
    # Fact: Hooly's SOC 2 certification, no
    ("fact", SOC2_FACT, "Hooly is not SOC 2 certified.", True),
    ("fact", SOC2_FACT, "Hooly has never been SOC2-certified.", True),
    ("fact", SOC2_FACT, "Hooly is fully SOC 2 certified.", False),
    ("fact", SOC2_FACT, "Hooly has never passed a SOC 2 audit.", False),
    ("fact", SOC2_FACT, "Hooly is not profitable.", False),
    # Compare: Hooly Workspace's price vs Hooly AI Pro's
    ("compare", price_compare("gt"), "Hooly Workspace costs more than Hooly AI Pro.", True),
    ("compare", price_compare("gt"), "Hooly Workspace is more expensive than Hooly AI Pro.", True),
    ("compare", price_compare("gt"), "The Workspace plan costs a higher monthly price than Hooly AI Pro.", True),
    ("compare", price_compare("eq"), "Hooly Workspace costs the same as Hooly AI Pro.", True),
    ("compare", price_compare("eq"), "Hooly Workspace costs as much as Hooly AI Pro.", True),
    ("compare", price_compare("lt"), "Hooly Workspace costs more than Hooly AI Pro.", False),
    ("compare", price_compare("lt"), "Hooly AI Pro costs less than Hooly Workspace.", False),
    ("compare", price_compare("gt"), "Hooly Workspace costs more than its rivals.", False),
    ("compare", price_compare("gt"), "Hooly Workspace outprices Hooly AI Pro.", False),
    ("compare", price_compare("gt"), "Hooly Workspace does not cost more than Hooly AI Pro.", False),
    ("compare", price_compare("gt"), "Hooly Workspace costs more than Hooly AI Pro, and less than Hooly AI Free.", False),
    ("compare", price_compare("gt"), "Hooly Workspace is our biggest plan. It costs more than Hooly AI Pro.", False),
    ("compare", price_compare("gt"), "Hooly Workspace costs more than Hooly.", False),
    ("compare", price_compare("gt"), "Hooly Workspace comes after Hooly AI Pro.", False),
    ("compare", price_compare("gt"), "Hooly Workspace costs more per month than Hooly AI Pro.", True),
    ("compare", price_compare("gt"), "Hooly Workspace costs more than Hooly AI Pro. We checked in June 2026.", True),
    ("compare", price_compare("gt"), "Hooly Workspace costs €100 more than Hooly AI Pro.", False),
    ("compare", price_compare("gt"), "Hooly Workspace costs 100 more than Hooly AI Pro.", False),
    ("compare", price_compare("gt"), "Hooly Workspace costs a hundred more than Hooly AI Pro.", False),
    ("compare", price_compare("gt"), "In June 2026, Hooly Workspace cost more than Hooly AI Pro.", False),
    ("compare", price_compare("gt"), "In June, Hooly Workspace cost more than Hooly AI Pro.", False),
    ("compare", price_compare("gt"), "Hooly Workspace used to cost more than Hooly AI Pro.", False),
    ("compare", price_compare("gt"), "Hooly Workspace cost more than Hooly AI Pro before the outage.", False),
    # Compare: date of Hooly I/O 2026 vs date of the May 2026 outage
    ("compare", date_compare("lt"), "Hooly I/O 2026 took place before the May 2026 outage.", True),
    ("compare", date_compare("lt"), "Hooly I/O 2026 came earlier than the outage.", True),
    ("compare", date_compare("eq"), "Hooly I/O took place on the same day as the outage.", True),
    ("compare", date_compare("gt"), "Hooly I/O 2026 was more than the May 2026 outage.", False),
    ("compare", date_compare("lt"), "Hooly I/O 2026 took place two weeks before the May 2026 outage.", False),
    ("compare", date_compare("lt"), "Hooly I/O 2026 took place before the outage, in March 2026.", False),
    # Change: Hooly AI Pro's price, from €49 per month to €59 per month
    ("change", CHANGE, "Hooly AI Pro went from €49 per month to €59 per month.", True),
    ("change", CHANGE, "The price of Hooly AI Pro rose from €49 per month to €59 per month in April.", True),
    ("change", CHANGE, "Hooly AI Pro costs €59 per month, up from €49 per month.", True),
    ("change", CHANGE, "Hooly AI Pro went up from €49 per month to €59 per month.", True),
    ("change", CHANGE, "Hooly AI Pro costs €59 per month, down from €49 per month.", False),
    ("change", CHANGE, "Hooly AI Pro went down from €49 per month to €59 per month.", False),
    ("change", CHANGE, "Hooly AI Pro went from €59 per month to €49 per month.", False),
    ("change", CHANGE, "Hooly AI Pro went from €49 to €59 per month.", False),
    ("change", CHANGE, "Hooly AI Pro cost €49 per month and now costs €59 per month.", False),
    ("change", CHANGE, "Hooly AI Pro never went from €49 per month to €59 per month.", False),
    # At an event: Hooly Gemini's languages at Hooly I/O 2026, 12 languages
    ("at_event", AT_EVENT, "Hooly Gemini spoke 12 languages at Hooly I/O 2026.", True),
    ("at_event", AT_EVENT, "At Hooly I/O, Hooly Gemini spoke a dozen languages.", True),
    ("at_event", AT_EVENT, "Hooly Gemini supported 12 languages on the day of Hooly I/O 2026.", True),
    ("at_event", AT_EVENT, "Hooly Gemini spoke 12 languages at the time of Hooly I/O 2026.", True),
    ("at_event", AT_EVENT, "Hooly Gemini spoke 12 languages before Hooly I/O 2026.", False),
    ("at_event", AT_EVENT, "Hooly Gemini spoke 12 languages in March.", False),
    ("at_event", AT_EVENT, "At Hooly I/O 2026, Hooly spoke 12 languages.", False),
    ("at_event", AT_EVENT, "Hooly Gemini spoke 12 languages at Hooly I/O 2026, up from 6 languages.", False),
    ("at_event", AT_EVENT, "In March 2026, at Hooly I/O 2026, Hooly Gemini spoke 12 languages.", False),
    # Relation: Hooly's CTO (Bertram Gilfoil) joined on 1 July 2026
    ("relation", RELATION, "The CTO of Hooly joined on 1 July 2026.", True),
    ("relation", RELATION, "Hooly's CTO started on 2026-07-01.", True),
    ("relation", RELATION, "Hooly's chief technology officer joined on 1 July 2026.", False),
    ("relation", RELATION, "Bertram Gilfoil joined Hooly on 1 July 2026.", False),
    ("relation", RELATION, "The CTO joined on 1 July 2026.", False),
    ("relation", RELATION, "The CTO of Hooly, who left on 3 May 2027, joined on 1 July 2026.", False),
    ("relation", RELATION, "The CTO of Hooly joined on 1 July 2026, after Richard Hendrix.", False),
]


def _entry(ctype, form, text):
    return {"id": "t", "type": ctype, "text": text, "form": form, "tag": "true", "confidence": 0.9}


def test_every_table_row_is_a_case():
    assert len(CASES) == 14 + 5 + 23 + 6 + 10 + 9 + 7


@pytest.mark.parametrize("ctype,form,text,ok", CASES)
def test_claim_forms(ctype, form, text, ok):
    entry = _entry(ctype, form, text)
    assert (check_claim(entry, L) == []) is ok, check_claim(entry, L)


@pytest.mark.parametrize("path", sorted(glob.glob("contracts/examples/entry/good/*.json")))
def test_good_examples_pass(path):
    with open(path, encoding="utf-8") as fh:
        entry = json.load(fh)
    assert check_claim(entry, L) == []


@pytest.mark.parametrize("path", sorted(glob.glob("contracts/examples/entry/bad/*.json")))
def test_bad_examples_fail(path):
    with open(path, encoding="utf-8") as fh:
        entry = json.load(fh)
    assert check_claim(entry, L) != []


# Further cases from the general rules of claim-forms.md and value-forms.md.
EXTRA = [
    # lexicon: unknown subject, attribute the subject lacks, unknown event, relation attribute on target
    ("fact", {"subject": "Hoolyx", "attribute": "price", "value": eur_month(59)}, "Hoolyx costs €59 per month.", False),
    ("fact", {"subject": "Hooly AI Pro", "attribute": "languages", "value": eur_month(59)}, "Hooly AI Pro costs €59 per month.", False),
    ("at_event", dict(AT_EVENT, event="Hooly Gemini"), "Hooly Gemini spoke 12 languages at Hooly Gemini.", False),
    ("relation", dict(RELATION, attribute="price"), "The CTO of Hooly joined on 1 July 2026.", False),
    ("relation", dict(RELATION, link="languages"), "The languages of Hooly joined on 1 July 2026.", False),
    # names: any case and spacing, alias
    ("fact", PRICE_FACT, "HOOLY  AI PRO costs €59 per month.", True),
    ("fact", PRICE_FACT, "HoolyAIPro costs €59 per month.", True),
    ("fact", dict(PRICE_FACT, subject="hooly ai pro"), "Hooly AI Pro costs €59 per month.", True),
    # value written another way, value in another unit that converts
    ("fact", PRICE_FACT, "Hooly AI Pro costs €59/mo.", True),
    ("fact", PRICE_FACT, "Hooly AI Pro costs 59 euros a month.", True),
    ("fact", PRICE_FACT, "Hooly AI Pro costs at least €59 per month.", False),
    ("fact", PRICE_FACT, "Hooly AI Pro costs €59 per month. It used to cost €49 per month.", True),
    ("fact", PRICE_FACT, "Hooly AI Pro costs €59 per month on Mondays.", False),
    ("fact", PRICE_FACT, "Hooly AI Pro costs €59 per month this summer.", False),
    ("fact", PRICE_FACT, "Hooly AI Pro may cost €59 per month.", True),
    ("fact", PRICE_FACT, "Hooly AI Pro costs €59​ per month.", False),
    ("fact", {"subject": "Hooly", "attribute": "employees", "value": {"kind": "quantity", "value": 24, "unit": "employee"}},
     "Hooly has twenty-four employees.", True),
    ("fact", {"subject": "Hooly", "attribute": "employees", "value": {"kind": "quantity", "value": 24, "unit": "employee"}},
     "Hooly has 24 employees, and 3 offices.", True),
    ("fact", {"subject": "Hooly", "attribute": "employees", "value": {"kind": "quantity", "value": 24, "unit": "employee"}},
     "Hooly has 24 employees, up from 20 employees.", False),
    ("fact", {"subject": "Hooly AI Pro", "attribute": "free_trial", "value": {"kind": "duration", "value": 2, "unit": "week"}},
     "Hooly AI Pro has a 14-day free trial.", True),
    ("fact", {"subject": "Hooly Gemini", "attribute": "version", "value": {"kind": "version", "value": "2.2"}},
     "Hooly Gemini 2.2 is the latest release.", True),
    ("fact", {"subject": "Hooly", "attribute": "founded", "value": {"kind": "date", "value": "2025"}},
     "Hooly was founded in 2025.", True),
    ("fact", {"subject": "Hooly", "attribute": "founded", "value": {"kind": "date", "value": "2025"}},
     "Hooly was founded in March 2025.", False),
    # a value inside a name the form does not need counts
    ("fact", {"subject": "Hooly Gemini", "attribute": "languages", "value": {"kind": "quantity", "value": 12, "unit": "language"}},
     "Hooly Gemini spoke 12 languages at Hooly I/O 2026.", False),
    # yes or no stated yes
    ("fact", dict(SOC2_FACT, value={"kind": "boolean", "value": True}), "Hooly is SOC 2 certified.", True),
    ("fact", dict(SOC2_FACT, value={"kind": "boolean", "value": True}), "Hooly is SOC 2 certified since 2026.", False),
    ("fact", {"subject": "Hooly", "attribute": "had_major_outage", "value": {"kind": "boolean", "value": False}},
     "Hooly hasn't had major outage.", True),
    # names as values
    ("fact", {"subject": "Hooly", "attribute": "cto", "value": {"kind": "name", "value": "Bertram Gilfoil"}},
     "The CTO of Hooly is BertramGilfoil.", True),
    ("fact", {"subject": "Hooly", "attribute": "cto", "value": {"kind": "name", "value": "Bertram Gilfoil"}},
     "The CTO of Hooly is not Bertram Gilfoil.", False),
    ("fact", {"subject": "Hooly", "attribute": "cto", "value": {"kind": "name", "value": "Bertram Gilfoil"}},
     "The CTO of Hooly is Dinesh Chugtay.", False),
    ("fact", {"subject": "Hooly", "attribute": "cto", "value": {"kind": "name", "value": "Bertram Gilfoil"}},
     "Bertram Gilfoil, not Dinesh Chugtay, is the CTO of Hooly.", False),
    ("fact", {"subject": "Hooly Cloud", "attribute": "reliability_lead", "value": {"kind": "name", "value": "Nimbus Cloud"}},
     "The reliability lead of Hooly Cloud is nimbus cloud.", True),
    ("fact", {"subject": "Hooly Cloud", "attribute": "reliability_lead", "value": {"kind": "name", "value": "Nimbus Cloud"}},
     "The reliability lead of Hooly Cloud is Nimbus Clouds.", False),
    # compare
    ("compare", price_compare("lt"), "Hooly AI Pro costs less than Hooly Workspace.".replace("Hooly AI Pro", "X"), False),
    ("compare", {"left": {"subject": "Hooly AI Pro", "attribute": "price"}, "right": {"subject": "Hooly Workspace", "attribute": "price"}, "op": "lt"},
     "Hooly AI Pro costs less than Hooly Workspace.", True),
    ("compare", price_compare("gt"), "Hooly Workspace costs no more than Hooly AI Pro.", False),
    ("compare", price_compare("gt"), "Hooly Workspace costs more than Hooly AI Pro. It is less than Hooly AI Free.", False),
    ("compare", price_compare("gt"), "Hooly Workspace costs more than one Hooly AI Pro.", False),
    ("compare", price_compare("gt"), "Hooly Workspace costs more than Hooly AI Pro on Sundays.", False),
    ("compare", price_compare("gt"), "Hooly Workspace costs more than Hooly AI Pro since the outage.", False),
    ("compare", {"left": {"subject": "Hooly AI Pro", "attribute": "price"}, "right": {"subject": "Hooly AI Pro", "attribute": "price"}, "op": "eq"},
     "Hooly AI Pro costs the same as Hooly AI Pro.", False),
    ("compare", {"left": {"subject": "Hooly Workspace", "attribute": "minimum_seats"},
                 "right": {"subject": "Hooly Workspace", "attribute": "maximum_seats"}, "op": "lt"},
     "Hooly Workspace allows fewer minimum seats than Hooly Workspace.", True),
    ("compare", date_compare("gt"), "Hooly I/O 2026 took place after the May 2026 outage.", True),
    ("compare", date_compare("gt"), "Hooly I/O 2026 took place later than the outage.", True),
    ("compare", date_compare("lt"), "Hooly I/O 2026 took place earlier than the outage in May.", False),
    # change
    ("change", CHANGE, "Hooly AI Pro went from €49/mo to €59 a month.", True),
    ("change", CHANGE, "Hooly AI Pro went from over €49 per month to €59 per month.", False),
    ("change", CHANGE, "Hooly AI Pro went from €49 per month to over €59 per month.", False),
    ("change", CHANGE, "Hooly AI Pro is a plan. Its price went from €49 per month to €59 per month.", False),
    ("change", dict(CHANGE, **{"from": eur_month(59), "to": eur_month(49)}), "Hooly AI Pro costs €49 per month, down from €59 per month.", True),
    ("change", {"subject": "Hooly Gemini", "attribute": "languages",
                "from": {"kind": "quantity", "value": 6, "unit": "language"}, "to": {"kind": "quantity", "value": 12, "unit": "language"}},
     "Hooly Gemini went from 6 languages to 12 languages.", True),
    ("change", {"subject": "Hooly", "attribute": "cto", "from": {"kind": "name", "value": "Dinesh Chugtay"}, "to": {"kind": "name", "value": "Bertram Gilfoil"}},
     "The CTO of Hooly went from Dinesh Chugtay to Bertram Gilfoil.", True),
    ("change", {"subject": "Hooly", "attribute": "cto", "from": {"kind": "name", "value": "Dinesh Chugtay"}, "to": {"kind": "name", "value": "Bertram Gilfoil"}},
     "The CTO of Hooly went up from Dinesh Chugtay to Bertram Gilfoil.", False),
    ("change", {"subject": "Hooly", "attribute": "soc2_certified", "from": {"kind": "boolean", "value": False}, "to": {"kind": "boolean", "value": True}},
     "Hooly went from not SOC 2 certified to SOC 2 certified.", True),
    ("change", {"subject": "Hooly", "attribute": "soc2_certified", "from": {"kind": "boolean", "value": True}, "to": {"kind": "boolean", "value": False}},
     "Hooly went from not SOC 2 certified to SOC 2 certified.", False),
    # at an event
    ("at_event", AT_EVENT, "Hooly Gemini spoke 12 languages during Hooly I/O 2026.", True),
    ("at_event", AT_EVENT, "Hooly Gemini spoke 12 languages at the Hooly I/O 2026.", True),
    ("at_event", AT_EVENT, "Hooly Gemini spoke 12 languages at Hooly I/O 2026 last year.", False),
    ("at_event", AT_EVENT, "Hooly Gemini did not speak 12 languages at Hooly I/O 2026.", False),
    ("at_event", dict(AT_EVENT, event="The May 2026 outage"), "Hooly Gemini spoke 12 languages during the May 2026 outage.", True),
    ("at_event", dict(AT_EVENT, event="The May 2026 outage"), "Hooly Gemini spoke 12 languages at the outage.", True),
    # relation
    ("relation", RELATION, "The CTO of Hooly joined on 1 July 2026, in June 2026.", False),
    ("relation", RELATION, "The CTO of Hooly never joined on 1 July 2026.", False),
    ("relation", RELATION, "The CTO of Hooly joined in July 2026.", False),
]


@pytest.mark.parametrize("ctype,form,text,ok", EXTRA)
def test_general_rules(ctype, form, text, ok):
    entry = _entry(ctype, form, text)
    assert (check_claim(entry, L) == []) is ok, check_claim(entry, L)


def test_schema_errors_come_first():
    entry = _entry("fact", PRICE_FACT, "Hooly AI Pro costs €59 per month.")
    entry["confidence"] = 0.2
    errs = check_claim(entry, L)
    assert errs and "schema" in errs[0]


def test_compare_kinds_from_index():
    class Kinds:
        def kind(self, attribute):
            return {"price": "money", "date": "date", "cto": "name"}.get(attribute)

    gt = price_compare("gt")
    assert check_claim(_entry("compare", gt, "Hooly Workspace costs more than Hooly AI Pro."), L, kinds=Kinds()) == []
    names = {"left": {"subject": "Hooly", "attribute": "cto"}, "right": {"subject": "Hooly", "attribute": "cto"}, "op": "eq"}
    assert check_claim(_entry("compare", names, "Hooly is the same as Hooly."), L, kinds=Kinds()) != []
    mixed = {"left": {"subject": "Hooly I/O 2026", "attribute": "date"}, "right": {"subject": "Hooly AI Pro", "attribute": "price"}, "op": "lt"}
    assert check_claim(_entry("compare", mixed, "Hooly I/O 2026 is cheaper than Hooly AI Pro."), L, kinds=Kinds()) != []
