"""Pure feature-helper tests; no upstream Daria snapshot or gold corpus is loaded."""

from dataclasses import dataclass, field
import unittest

from integrations.daria_rich_features import build_true_anchor_index, summarize_rich_features


@dataclass(frozen=True)
class FakeAtom:
    subject: str
    attribute: str
    value: dict

    @property
    def key(self):
        return self.subject, self.attribute


@dataclass
class FakeRow:
    id: str
    date: str
    label: str | None
    atoms: list | None
    page: int
    meta: dict = field(default_factory=dict)


class DariaRichFeatureTests(unittest.TestCase):
    def setUp(self):
        self.name = FakeAtom("Hooly", "employees", {"kind": "quantity", "value": 10, "unit": "person"})
        self.region = FakeAtom("Hooly Berlin Office", "opened", {"kind": "date", "value": "2026-03"})
        self.name_trace = {"subject": "Hooly", "attribute": "employees", "state": "core_match", "p_true": .8,
                           "stability": "stable", "direction": "flat", "trend": "flat", "shape": "same"}
        self.region_trace = {"subject": "Hooly Berlin Office", "attribute": "opened", "state": "before_match_near",
                             "p_true": .6, "stability": "thin", "direction": "unknown", "trend": "flat", "shape": "before"}
        self.row = FakeRow("private-row-id", "2026-03-10", None, [self.name, self.region], 8,
                           {"atom_q": {self.name.key: .7, self.region.key: .9}})

    @staticmethod
    def exact_support(anchor, query, _names):
        return anchor.key == query.key and anchor.value == query.value

    def test_feature_summary_is_invariant_to_atom_order(self):
        forward = summarize_rich_features(self.row, [self.name_trace, self.region_trace], {},
                                          self.exact_support)
        reversed_row = FakeRow(self.row.id, self.row.date, self.row.label,
                               [self.region, self.name], self.row.page, self.row.meta)
        reversed_summary = summarize_rich_features(reversed_row, [self.region_trace, self.name_trace], {},
                                                   self.exact_support)

        self.assertEqual(forward, reversed_summary)
        self.assertEqual(forward["rich__summary_atom_p_true_min"], .6)
        self.assertAlmostEqual(forward["rich__summary_atom_p_true_mean"], .7)
        self.assertEqual(forward["rich__summary_atom_p_true_max"], .8)

    def test_anchor_index_uses_only_true_exact_atomic_observations(self):
        same = FakeAtom("Hooly", "employees", {"kind": "quantity", "value": 10, "unit": "person"})
        rows = [FakeRow("true-id", "2026-03-01", "true", [same], 7),
                FakeRow("false-id", "2026-03-02", "false", [same], 8),
                FakeRow("unsure-id", "2026-03-03", "unsure", [same], 9),
                FakeRow("op-id", "2026-03-04", "true", [FakeAtom("Hooly", "employees",
                          {"kind": "quantity", "value": 10, "unit": "person", "op": "gt"})], 10)]

        anchors = build_true_anchor_index(rows)

        self.assertEqual(list(anchors), [same.key])
        self.assertEqual([(d, p) for d, p, _a in anchors[same.key]], [("2026-03-01", 7)])

    def test_training_excludes_own_page_anchors_and_features_contain_no_row_ids(self):
        own = FakeAtom("Hooly", "employees", dict(self.name.value))
        other = FakeAtom("Hooly", "employees", dict(self.name.value))
        anchors = {self.name.key: [("2026-03-09", self.row.page, own),
                                   ("2026-03-01", self.row.page + 1, other)]}

        training_features = summarize_rich_features(self.row, [self.name_trace], anchors,
                                                    self.exact_support, training=True)
        inference_features = summarize_rich_features(self.row, [self.name_trace], anchors,
                                                     self.exact_support, training=False)

        self.assertEqual(training_features["rich__support_anchor_observation_count"], 1.0)
        self.assertEqual(inference_features["rich__support_anchor_observation_count"], 2.0)
        self.assertEqual(training_features["rich__support_nearest_past_gap_days"], 9.0)
        self.assertNotIn("private-row-id", " ".join(training_features))
        self.assertTrue(all(key.startswith("rich__") for key in training_features))

    def test_histograms_and_parse_summary_are_value_derived(self):
        feats = summarize_rich_features(self.row, [self.name_trace, self.region_trace], {}, self.exact_support)

        self.assertEqual(feats["rich__summary_atom_kind_quantity"], 1.0)
        self.assertEqual(feats["rich__summary_atom_kind_date"], 1.0)
        self.assertEqual(feats["rich__summary_atom_unit_person"], 1.0)
        self.assertEqual(feats["rich__summary_date_precision_month"], 1.0)
        self.assertEqual(feats["rich__summary_parse_q_min"], .7)
        self.assertAlmostEqual(feats["rich__summary_parse_q_mean"], .8)
        self.assertEqual(feats["rich__summary_unique_subject_count"], 2.0)


if __name__ == "__main__":
    unittest.main()
