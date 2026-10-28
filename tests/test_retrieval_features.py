import unittest

from retrieval.features import RetrievalIndex


LEXICON = {"Hooly Workspace": {"aliases": ["Workspace", "HW"]},
           "Hooly AI Pro": {"aliases": ["AI Pro"]}}
ACTUAL_LEXICON = {"subjects": [
    {"name": "Hooly", "aliases": ["Hooly AI"]},
    {"name": "Hooly Workspace", "aliases": ["Workspace"]},
    {"name": "Hooly AI Pro", "aliases": []}], "events": []}


def row(i, text, label="true", date="2026-01-01"):
    return {"id": i, "text": text, "label": label, "date": date}


class RetrievalFeatureTests(unittest.TestCase):
    def test_masked_view_ignores_number_difference(self):
        refs = [row("a", "Hooly Workspace costs 10 USD monthly", "true")]
        idx = RetrievalIndex(refs, LEXICON, views=("raw", "masked"))
        _, traces = idx.transform([row("q", "Hooly Workspace costs 900 USD monthly", "false")])
        sims = {v: (traces[0]["views"][v][0]["similarity"] if traces[0]["views"][v] else 0)
                for v in ("raw", "masked")}
        self.assertGreater(sims["masked"], sims["raw"])

    def test_longest_alias_match_and_trace(self):
        idx = RetrievalIndex([row("r", "Hooly AI Pro has 5 users")], LEXICON)
        _, trace = idx.transform([row("q", "Hooly AI Pro has 8 users")])
        self.assertTrue(trace[0]["views"]["raw"][0]["shared_entity"])

    def test_empty_vocabulary_and_oov_have_no_neighbors(self):
        empty = RetrievalIndex([row("r", "!!!")], {})
        feats, traces = empty.transform([row("q", "???")])
        self.assertEqual(len(traces[0]["views"]["raw"]), 0)
        self.assertTrue((feats == 0).all())
        idx = RetrievalIndex([row("r", "alpha beta")], {})
        _, traces = idx.transform([row("q", "unseen gamma")])
        self.assertEqual(traces[0]["views"]["raw"], [])

    def test_ties_are_stable_by_reference_id_and_label_shares(self):
        refs = [row("z", "same phrase", "false"), row("a", "same phrase", "true")]
        idx = RetrievalIndex(refs, {}, k=2)
        feats, traces = idx.transform([row("q", "same phrase")])
        self.assertEqual([n["id"] for n in traces[0]["views"]["raw"]], ["a", "z"])
        names = idx.feature_names_
        self.assertEqual(feats[0, names.index("raw_weighted_share_true")], .5)
        self.assertGreater(feats[0, names.index("raw_neighbor_label_entropy")], 0)

    def test_query_label_ignored_and_date_features(self):
        idx = RetrievalIndex([row("r", "Workspace 5 users", "true", "2025-01-01")], LEXICON)
        q1 = row("q1", "Workspace 5 users", "false", "2026-01-01")
        q2 = dict(q1, id="q2", label="true")
        features, _ = idx.transform([q1, q2])
        self.assertTrue((features[0] == features[1]).all())
        self.assertAlmostEqual(features[0, idx.feature_names_.index("raw_nearest_dategap_abs_true")], 365/365)

    def test_self_id_rejected_and_fit_labels_validated(self):
        idx = RetrievalIndex([row("r", "alpha")], {})
        with self.assertRaises(ValueError):
            idx.transform([row("r", "alpha")])
        with self.assertRaises(ValueError):
            RetrievalIndex([row("x", "alpha", "maybe")], {})

    def test_reference_text_only_drives_vocabulary(self):
        idx = RetrievalIndex([row("r", "reference phrase")], {})
        _, traces = idx.transform([row("q", "totally novel query words")])
        self.assertEqual(traces[0]["views"]["raw"], [])

    def test_actual_lexicon_resolves_aliases_and_distinct_long_names(self):
        idx = RetrievalIndex([row("r", "Hooly Workspace costs 10 EUR per month")], ACTUAL_LEXICON)
        _, traces = idx.transform([row("q", "Workspace costs 10 EUR /month")])
        self.assertTrue(traces[0]["views"]["raw"][0]["shared_entity"])
        # The shorter Hooly hit is suppressed inside the longer Hooly AI Pro alias.
        from retrieval.features import _entities
        self.assertEqual(_entities("Hooly AI Pro", idx.entity_terms), frozenset({"hooly ai pro"}))
        self.assertEqual(_entities("Workspace", idx.entity_terms), frozenset({"hooly workspace"}))
        self.assertTrue(traces[0]["views"]["raw"][0]["period_match"])

    def test_numeric_overlap_and_unit_compatibility(self):
        idx = RetrievalIndex([row("r", "price 10 USD per month")], {})
        _, trace = idx.transform([row("q", "price 20 EUR annually")])
        neighbor = trace[0]["views"]["raw"][0]
        self.assertFalse(neighbor["numeric_equal"])
        self.assertEqual(neighbor["numeric_overlap"], 0.0)
        self.assertFalse(neighbor["currency_match"])
        self.assertFalse(neighbor["period_match"])
        _, matching = idx.transform([row("q2", "price 10 dollars monthly")])
        neighbor = matching[0]["views"]["raw"][0]
        self.assertTrue(neighbor["numeric_equal"])
        self.assertEqual(neighbor["numeric_overlap"], 1.0)
        self.assertTrue(neighbor["currency_match"])
        self.assertTrue(neighbor["period_match"])

    def test_per_class_nearest_is_not_limited_to_global_top_five(self):
        refs = [row(str(i), "alpha common beta", "true") for i in range(6)]
        refs.append(row("z", "alpha rare", "false"))
        idx = RetrievalIndex(refs, {}, k=1)
        feats, trace = idx.transform([row("q", "alpha common beta rare")])
        self.assertTrue(any(n["id"] != "z" for n in trace[0]["views"]["raw"]))
        # The false-class-specific nearest neighbor is still used for its metadata,
        # even though it is outside the global top-k trace.
        ix = idx.feature_names_.index("raw_nearest_numeric_equal_false")
        self.assertGreaterEqual(ix, 0)
        self.assertEqual(feats.shape[1], len(idx.feature_names_))
        self.assertEqual([n["id"] for n in trace[0]["views"]["class_neighbors"]["raw"]["false"]], ["z"])


if __name__ == "__main__":
    unittest.main()
