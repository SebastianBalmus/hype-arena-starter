import json
import unittest

from knowledge_base.store import EvidenceStore


LEXICON = {"subjects": [{"name": "Hooly", "aliases": [], "attributes": ["price", "users", "retention", "duration"]}]}


def row(i, text, date, label, atoms):
    return {"id": i, "text": text, "date": date, "label": label, "form": json.dumps(atoms)}


def atom(attribute, value):
    return {"subject": "Hooly", "attribute": attribute, "value": value}


class EvidenceStoreTests(unittest.TestCase):
    def test_date_only_and_fingerprint_provenance(self):
        q = atom("price", {"kind": "money", "value": 10, "currency": "EUR"})
        store = EvidenceStore([row("r1", "Price is 10 EUR", "2026-01-01", "true", [q])], LEXICON)
        self.assertEqual(store.query(q, "2026-01-01")["label"], "true")
        self.assertEqual(store.query(q, "2026-01-02")["reason"], "missing_evidence")
        provenance = store.query(q, "2026-01-01")["evidence"][0]
        self.assertEqual(provenance["id"], "r1")
        self.assertEqual(len(provenance["fingerprint"]), 64)
        self.assertEqual(store.summary()["numeric_atoms"], 1)

    def test_false_compound_including_unsupported_atom_does_not_refute_numeric_atom(self):
        numeric = atom("users", {"kind": "quantity", "value": 10, "unit": "customer"})
        boolean = atom("retention", {"kind": "boolean", "value": True})
        store = EvidenceStore([row("f", "Users are 10 and retention holds", "2026-01-01", "false", [numeric, boolean])], LEXICON)
        self.assertEqual(store.query(numeric, "2026-01-01")["reason"], "missing_evidence")
        self.assertEqual(store.export_records()[0]["atoms"], [numeric])

    def test_dimensions_and_billing_period_are_not_collapsed(self):
        fact = atom("price", {"kind": "money", "value": 10, "currency": "EUR", "per": "month"})
        store = EvidenceStore([row("r", "10 EUR monthly", "2026-01-01", "true", [fact])], LEXICON)
        for query in (
            atom("price", {"kind": "money", "value": 10, "currency": "EUR", "per": "year"}),
            atom("price", {"kind": "money", "value": 10, "currency": "USD", "per": "month"}),
        ):
            self.assertEqual(store.query(query, "2026-01-01")["label"], "unsure")

    def test_inequality_implication_and_exclusion(self):
        fact = atom("users", {"kind": "quantity", "value": 10, "unit": "customer", "op": "ge"})
        store = EvidenceStore([row("r", "At least 10 customers", "2026-01-01", "true", [fact])], LEXICON)
        self.assertEqual(store.query(atom("users", {"kind": "quantity", "value": 5, "unit": "customer", "op": "gt"}), "2026-01-01")["label"], "true")
        self.assertEqual(store.query(atom("users", {"kind": "quantity", "value": 9, "unit": "customer", "op": "lt"}), "2026-01-01")["label"], "false")
        self.assertEqual(store.query(atom("users", {"kind": "quantity", "value": 10, "unit": "customer"}), "2026-01-01")["label"], "unsure")

    def test_open_closed_equal_bounds_intersect_correctly(self):
        ge = atom("users", {"kind": "quantity", "value": 10, "unit": "customer", "op": "ge"})
        gt = atom("users", {"kind": "quantity", "value": 10, "unit": "customer", "op": "gt"})
        store = EvidenceStore([row("b", "More than 10", "2026-01-01", "true", [gt])], LEXICON)
        exact10 = atom("users", {"kind": "quantity", "value": 10, "unit": "customer"})
        self.assertEqual(store.query(exact10, "2026-01-01")["label"], "false")

    def test_conflicting_true_observations_preserved(self):
        rows = [row("a", "10 customers", "2026-01-01", "true", [atom("users", {"kind": "quantity", "value": 10, "unit": "customer"})]), row("b", "12 customers", "2026-01-01", "true", [atom("users", {"kind": "quantity", "value": 12, "unit": "customer"})])]
        store = EvidenceStore(rows, LEXICON)
        out = store.query(atom("users", {"kind": "quantity", "value": 10, "unit": "customer"}), "2026-01-01")
        self.assertEqual((out["label"], out["reason"]), ("unsure", "temporal_conflict"))
        self.assertEqual({e["id"] for e in out["evidence"]}, {"a", "b"})

    def test_false_single_atom_refutes_canonical_equivalent_only(self):
        false_atom = atom("users", {"kind": "quantity", "value": 10, "unit": "customer"})
        store = EvidenceStore([row("f", "Users were 10", "2026-01-01", "false", [false_atom])], LEXICON)
        equivalent = atom("users", {"kind": "quantity", "value": 10.0, "unit": "customer", "op": "eq"})
        other = atom("users", {"kind": "quantity", "value": 12, "unit": "customer"})
        self.assertEqual(store.query(equivalent, "2026-01-01")["label"], "false")
        self.assertEqual(store.query(other, "2026-01-01")["reason"], "missing_evidence")

    def test_positive_other_value_and_false_target_corroborate_false(self):
        true5 = atom("users", {"kind": "quantity", "value": 5, "unit": "customer"})
        false6 = atom("users", {"kind": "quantity", "value": 6, "unit": "customer"})
        store = EvidenceStore([row("t", "5 customers", "2026-01-01", "true", [true5]), row("f", "6 customers", "2026-01-01", "false", [false6])], LEXICON)
        out = store.query(false6, "2026-01-01")
        self.assertEqual((out["label"], out["reason"]), ("false", "decided"))

    def test_scope_guard_and_invalid_dimensions(self):
        q = atom("users", {"kind": "quantity", "value": 10, "unit": "customer"})
        store = EvidenceStore([row("f", "Last quarter users were 10", "2026-01-01", "false", [q])], LEXICON)
        self.assertEqual(store.query(q, "2026-01-01")["reason"], "missing_evidence")
        self.assertEqual(store.summary()["scope_blocked_rows"], 1)
        invalid = atom("users", {"kind": "quantity", "value": float("inf"), "unit": "customer"})
        invalid_unit = atom("users", {"kind": "quantity", "value": 10, "unit": ""})
        for bad in (invalid, invalid_unit):
            self.assertEqual(store.query(bad, "2026-01-01")["reason"], "unsupported")
        self.assertEqual(store.query(q, "2026-02-30")["reason"], "unsupported")

    def test_past_year_and_planned_targets_are_not_current_evidence(self):
        q = atom("users", {"kind": "quantity", "value": 10, "unit": "customer"})
        rows = [row("past", "Users were 10 in 2025", "2026-01-01", "true", [q]),
                row("target", "Target is 10 users", "2026-01-01", "true", [q])]
        store = EvidenceStore(rows, LEXICON)
        self.assertEqual(store.query(q, "2026-01-01")["reason"], "missing_evidence")
        self.assertEqual(store.summary()["scope_blocked_rows"], 2)

    def test_nonnegative_numeric_domains_and_duration_unit_kinds(self):
        for value in (
            {"kind": "money", "value": -1, "currency": "EUR"},
            {"kind": "quantity", "value": -1, "unit": "customer"},
            {"kind": "duration", "value": -1, "unit": "day"},
            {"kind": "quantity", "value": 1, "unit": "day"},
        ):
            source = atom("users", value)
            store = EvidenceStore([row("bad", "Invalid value", "2026-01-01", "true", [source])], LEXICON)
            self.assertEqual(store.summary()["numeric_atoms"], 0)

    def test_temporal_keyword_in_entity_name_is_masked(self):
        lexicon = {"subjects": [{"name": "Hooly Next", "aliases": [], "attributes": ["price"]}]}
        q = {"subject": "Hooly Next", "attribute": "price", "value": {"kind": "money", "value": 10, "currency": "EUR"}}
        store = EvidenceStore([row("n", "Hooly Next costs 10 EUR", "2026-01-01", "true", [q])], lexicon)
        self.assertEqual(store.query(q, "2026-01-01")["label"], "true")


if __name__ == "__main__":
    unittest.main()
