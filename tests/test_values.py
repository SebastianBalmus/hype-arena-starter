"""Value matching: the cases that would otherwise look like contradictions."""
import glob
import json
import unittest

from hooly.values import (COMPATIBLE, DIFFERENT, EQUAL, INCOMPARABLE, NameNormalizer,
                          compare, normalize, satisfies)


def q(v, u): return {"kind": "quantity", "value": v, "unit": u}
def dur(v, u): return {"kind": "duration", "value": v, "unit": u}
def dt(s): return {"kind": "date", "value": s}
def money(v, per=None): return {"kind": "money", "value": v, "currency": "EUR", **({"per": per} if per else {})}
def name(s): return {"kind": "name", "value": s}


class TestCompare(unittest.TestCase):
    def test_units_convert(self):
        self.assertEqual(compare(q(1, "GB"), q(1000, "MB")), EQUAL)
        self.assertEqual(compare(dur(14, "day"), dur(2, "week")), EQUAL)
        self.assertEqual(compare(dur(12, "month"), dur(1, "year")), EQUAL)
        self.assertEqual(compare(q(6.1, "inch"), q(6.1, "in")), EQUAL)

    def test_months_never_convert_to_days(self):
        self.assertEqual(compare(dur(30, "day"), dur(1, "month")), INCOMPARABLE)

    def test_counted_nouns(self):
        self.assertEqual(compare(q(40, "languages"), q(40, "language")), EQUAL)
        self.assertEqual(compare(q(12, "people"), q(12, "person")), EQUAL)
        self.assertEqual(compare(q(40, "language"), q(40, "region")), INCOMPARABLE)

    def test_money_period_is_part_of_the_dimension(self):
        self.assertEqual(compare(money(59, "month"), money(59, "month")), EQUAL)
        self.assertEqual(compare(money(59, "month"), money(590, "year")), INCOMPARABLE)
        self.assertEqual(compare(money(59, "month"), money(59)), INCOMPARABLE)

    def test_date_precision(self):
        self.assertEqual(compare(dt("2026-01"), dt("2026-01-12")), COMPATIBLE)
        self.assertEqual(compare(dt("2026"), dt("2026-03")), COMPATIBLE)
        self.assertEqual(compare(dt("2026-02"), dt("2026-01-12")), DIFFERENT)

    def test_decimals_are_exact(self):
        self.assertEqual(compare({"kind": "percent", "value": 99.9}, {"kind": "percent", "value": 99.90}), EQUAL)
        self.assertEqual(compare({"kind": "version", "value": "2.3"}, {"kind": "version", "value": "2.3.0"}), DIFFERENT)

    def test_short_names(self):
        names = NameNormalizer({"subjects": []}, extra_names=["Nimbus Cloud", "Nimbus", "Gavin Bellson", "Gavin"])
        self.assertEqual(compare(name("Nimbus"), name("Nimbus Cloud"), names), EQUAL)
        self.assertEqual(compare(name("gavin bellson"), name("GavinBellson"), names), EQUAL)

    def test_ambiguous_short_name_is_left_alone(self):
        names = NameNormalizer({"subjects": []}, extra_names=["Ada Lovelace", "Ada King", "Ada"])
        self.assertEqual(names("Ada"), "Ada")


class TestSatisfies(unittest.TestCase):
    def test_bounds(self):
        self.assertTrue(satisfies({**q(100, "customer"), "op": "gt"}, q(140, "customer")))
        self.assertFalse(satisfies({**q(5000, "attendee"), "op": "gt"}, q(2600, "attendee")))
        self.assertTrue(satisfies({**q(50, "GB"), "op": "le"}, q(50, "GB")))

    def test_date_bounds_with_precision(self):
        self.assertTrue(satisfies({**dt("2026-03"), "op": "lt"}, dt("2026-02-10")))
        self.assertFalse(satisfies({**dt("2026-03"), "op": "lt"}, dt("2026-03-10")))
        self.assertIsNone(satisfies({**dt("2026-03-15"), "op": "lt"}, dt("2026-03")))

    def test_incomparable(self):
        self.assertIsNone(satisfies(q(3, "region"), q(3, "zone")))


class TestContractExamples(unittest.TestCase):
    def test_good_examples_normalize(self):
        files = glob.glob("contracts/examples/claimed_value/good/*.json")
        self.assertTrue(files)
        for path in files:
            with open(path) as f:
                normalize(json.load(f))


if __name__ == "__main__":
    unittest.main()
