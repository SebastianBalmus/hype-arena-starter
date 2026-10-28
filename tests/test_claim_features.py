import unittest

import numpy as np

from retrieval.claim_features import ClaimFeatureIndex


LEXICON = {
    "subjects": [
        {"name": "Hooly Workspace", "aliases": ["Workspace"], "attributes": ["price", "users", "owner"]},
        {"name": "Hooly Desk", "aliases": ["Desk"], "attributes": ["price", "users"]},
        {"name": "Launch", "aliases": [], "attributes": ["date"]},
    ],
    "events": ["Launch"],
}


def row(rid, text, label="true", form=None, date="2026-10-29"):
    return {"id": rid, "text": text, "date": date, "label": label,
            "form": [] if form is None else form}


def fact(subject, attribute, value, kind="money", currency="USD", per="month"):
    data = {"kind": kind, "value": value}
    if kind == "money":
        data.update(currency=currency, per=per)
    return {"subject": subject, "attribute": attribute, "value": data}


class ClaimFeatureTests(unittest.TestCase):
    def test_false_compounds_never_become_positive_references(self):
        refs = [
            row("t1", "Workspace costs ten dollars monthly", form=[fact("Hooly Workspace", "price", "10")]),
            row("f1", "Workspace has 99 users and costs 99 dollars", "false",
                [fact("Hooly Workspace", "price", "99"),
                 fact("Hooly Workspace", "users", "99", kind="quantity")]),
            row("u1", "Workspace costs 12 dollars", "unsure", [fact("Hooly Workspace", "price", "12")]),
        ]
        idx = ClaimFeatureIndex(refs, LEXICON, mode="observed")
        records = idx.export_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["source_ids"], ["t1"])
        self.assertEqual(records[0]["status"], "unverified_auxiliary")

    def test_duplicate_observations_group_with_fingerprinted_sources(self):
        f = [fact("Workspace", "price", "10")]
        idx = ClaimFeatureIndex([row("a", "x", form=f), row("b", "y", form=f)], LEXICON)
        rec = idx.export_records()[0]
        self.assertEqual(rec["source_ids"], ["a", "b"])
        self.assertEqual(set(rec["source_fingerprints"]), {"a", "b"})

    def test_query_labels_forms_ignored_and_reference_overlap_rejected(self):
        idx = ClaimFeatureIndex([row("r", "Workspace price", form=[fact("Hooly Workspace", "price", "10")])], LEXICON)
        feats, traces = idx.transform([{"id": "q", "text": "Workspace price 10", "date": "2026-10-29", "label": "", "form": [{"value": "ignored"}]}])
        self.assertEqual(feats.shape, (1, len(idx.feature_names_)))
        self.assertEqual(traces[0]["query_id"], "q")
        with self.assertRaises(ValueError):
            idx.transform([{"id": "r", "text": "anything", "date": "2026-10-29", "label": ""}])
        with self.assertRaises(ValueError):
            idx.transform([{"id": "q2", "text": "anything", "date": "2026-10-29", "label": "true"}])

    def test_source_date_is_not_promoted_to_as_of(self):
        idx = ClaimFeatureIndex([row("r", "Workspace costs ten dollars", form=[fact("Hooly Workspace", "price", "10")], date="2026-10-10")], LEXICON)
        rec = idx.export_records()[0]
        self.assertEqual(rec["source_dates"], ["2026-10-10"])
        self.assertEqual(rec["temporal_kind"], "source_exact")
        _, traces = idx.transform([{"id": "q", "text": rec["text"], "date": "2026-10-29", "label": ""}])
        self.assertFalse(traces[0]["neighbors"][0]["temporal_compatible"])

    def test_plain_historical_observation_is_compatible_on_its_own_date(self):
        idx = ClaimFeatureIndex([row("r", "Workspace price is $10", form=[fact("Hooly Workspace", "price", 10)], date="2026-05-01")], LEXICON)
        rec = idx.export_records()[0]
        _, traces = idx.transform([{"id": "q", "text": rec["text"], "date": "2026-05-01", "label": ""}])
        self.assertFalse(rec["scope_uncertain"])
        self.assertTrue(traces[0]["neighbors"][0]["temporal_compatible"])

    def test_explicit_last_quarter_context_is_not_temporally_compatible(self):
        idx = ClaimFeatureIndex([row("r", "Workspace price was $10 last quarter", form=[fact("Hooly Workspace", "price", 10)], date="2026-10-01")], LEXICON)
        rec = idx.export_records()[0]
        _, traces = idx.transform([{"id": "q", "text": rec["text"] + " last quarter", "date": "2026-10-01", "label": ""}])
        self.assertTrue(rec["scope_uncertain"])
        self.assertFalse(traces[0]["neighbors"][0]["temporal_compatible"])

    def test_rerouted_event_uses_latest_observed_event_date_not_max_date(self):
        refs = [
            row("price", "Workspace price $10", form=[fact("Hooly Workspace", "price", 10)], date="2026-07-02"),
            row("old_event", "Launch planned July 10", form=[{"subject": "Launch", "attribute": "date", "value": {"kind": "date", "value": "2026-07-10"}}], date="2026-10-01"),
            row("new_event", "Launch rescheduled July 2", form=[{"subject": "Launch", "attribute": "date", "value": {"kind": "date", "value": "2026-07-02"}}], date="2026-10-20"),
        ]
        idx = ClaimFeatureIndex(refs, LEXICON, mode="generated", limit=100)
        event_records = [r for r in idx.export_records() if r["type"] == "at_event"]
        self.assertTrue(event_records)
        self.assertTrue(all(r["event_date"] == "2026-07-02" for r in event_records))

    def test_value_compatibility_respects_currency_and_period(self):
        refs = [row("r", "Workspace price", form=[fact("Hooly Workspace", "price", "10", currency="USD", per="month")])]
        idx = ClaimFeatureIndex(refs, LEXICON)
        from arena.values import parse_values
        good = idx._typed_agreement("Workspace price is $10 per month", idx.export_records()[0], [s.value for s in parse_values("Workspace price is $10 per month")])
        bad = idx._typed_agreement("Workspace price is €10 per year", idx.export_records()[0], [s.value for s in parse_values("Workspace price is €10 per year")])
        self.assertTrue(good)
        self.assertFalse(bad)

    def test_generated_mode_is_provenance_gated_and_capped(self):
        refs = [
            row("a", "Workspace price is $10 per month", form=[fact("Hooly Workspace", "price", 10)]),
            row("b", "Workspace price is $10 per month", form=[fact("Hooly Workspace", "price", 10)], date="2026-10-28"),
            row("n", "Workspace is owned by Desk", form=[{"subject": "Hooly Workspace", "attribute": "owner", "value": {"kind": "name", "value": "Hooly Desk"}}]),
        ]
        idx = ClaimFeatureIndex(refs, LEXICON, mode="generated", limit=5)
        self.assertLessEqual(len(idx.export_records()), 5)
        self.assertGreater(len(idx.export_records()), 0)
        self.assertIsInstance(idx.summary(), dict)
        for rec in idx.export_records():
            self.assertTrue(rec["source_ids"])
            self.assertEqual(rec["status"], "unverified_auxiliary")
            self.assertTrue(rec["scope_uncertain"])
            self.assertNotEqual(rec["generator_score"], None)

    def test_generated_candidates_need_matching_true_premises(self):
        idx = ClaimFeatureIndex([row("r", "Workspace price", form=[fact("Hooly Workspace", "price", "10")])], LEXICON, mode="generated")
        entry = {"type": "fact", "form": {"subject": "Hooly Workspace", "attribute": "users", "value": {"kind": "quantity", "value": "4", "unit": "user"}}}
        self.assertEqual(idx._premise_sources(entry)[0], [])

    def test_observed_candidates_on_different_dates_have_distinct_ids(self):
        f = [fact("Hooly Workspace", "price", 10)]
        idx = ClaimFeatureIndex([row("a", "Workspace price ten", form=f, date="2026-10-01"),
                                row("b", "Workspace price ten", form=f, date="2026-10-29")], LEXICON)
        records = idx.export_records()
        self.assertEqual(len(records), 2)
        self.assertNotEqual(records[0]["candidate_id"], records[1]["candidate_id"])

    def test_each_generated_claim_premise_must_have_a_source(self):
        idx = ClaimFeatureIndex([row("r", "Workspace price", form=[fact("Hooly Workspace", "price", 10)])], LEXICON)
        cases = [
            {"type": "compare", "form": {"left": {"subject": "Hooly Workspace", "attribute": "price"}, "right": {"subject": "Hooly Desk", "attribute": "price"}, "op": "gt"}},
            {"type": "change", "form": {"subject": "Hooly Workspace", "attribute": "price", "from": {"kind": "money", "value": 9, "currency": "USD", "per": "month"}, "to": {"kind": "money", "value": 10, "currency": "USD", "per": "month"}}},
            {"type": "at_event", "form": {"subject": "Hooly Workspace", "attribute": "price", "value": {"kind": "money", "value": 10, "currency": "USD", "per": "month"}, "event": "Launch"}},
            {"type": "relation", "form": {"subject": "Hooly Workspace", "link": "owner", "target": "Hooly Desk", "attribute": "price", "value": {"kind": "money", "value": 10, "currency": "USD", "per": "month"}}},
        ]
        for entry in cases:
            groups = idx._premise_groups(entry)
            self.assertTrue(groups)
            self.assertTrue(any(not g for g in groups))

    def test_zero_similarity_neighbors_do_not_set_semantic_features(self):
        idx = ClaimFeatureIndex([row("r", "Workspace price", form=[fact("Hooly Workspace", "price", 10)])], LEXICON)
        feats, traces = idx.transform([{"id": "q", "text": "unrelatedwordzz", "date": "2026-10-29", "label": ""}])
        self.assertEqual(feats[0, idx.feature_names_.index("fact.shared_entity")], 0)
        self.assertEqual(feats[0, idx.feature_names_.index("fact.typed_value_agreement")], 0)
        self.assertTrue(all(n["similarity"] == 0 for n in traces[0]["neighbors"]))


if __name__ == "__main__":
    unittest.main()
