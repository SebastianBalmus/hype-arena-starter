import unittest
import warnings

import numpy as np

from experiments.daria_feature_variants import (
    CONFIGS,
    _aggregate_atoms,
    _rich_bags,
    _structured,
    make_feature_model,
)


class DariaFeatureVariantTests(unittest.TestCase):
    @staticmethod
    def _records(n=36):
        records = []
        labels = []
        for i in range(n):
            label = ("true", "false", "unsure")[i % 3]
            records.append({
                "id": f"row-{i}",
                "date": "2026-10-01",
                "__text__": f"Hooly product launch price customer account topic{i % 6} month {i % 4}",
                "native__windows": float(i % 5) / 4,
                "native__hmm": float(i % 3) / 2,
                "native__agree": float(i % 2),
                "expanded__n_atoms": float(i % 3),
                "expanded__parsed": 1.0,
                "expanded__year": 2026.0,
                "expanded__month": float((i % 12) + 1),
                "expanded__mean_parse_q": 0.7 + (i % 3) * 0.1,
                "expanded__atom0_state=core_match": float(i % 3 == 0),
                "rich__content_subject_hooly": float(i % 2),
                "rich__content_attribute_price": float(i % 3 == 0),
                "rich__content_type_money": float(i % 4 == 0),
                "rich__summary_parse_confidence": 0.8,
                "rich__unrelated_note": float(i % 5),
            })
            labels.append(label)
        return records, labels

    def test_aggregate_atom_features_are_order_invariant(self):
        left = {
            "native__windows": 0.25,
            "expanded__atom0_p_true": 0.2,
            "expanded__atom0_state=core_match": 1.0,
            "expanded__atom1_p_true": 0.8,
            "expanded__atom1_state=core_mismatch": 1.0,
        }
        swapped = {
            "native__windows": 0.25,
            "expanded__atom0_p_true": 0.8,
            "expanded__atom0_state=core_mismatch": 1.0,
            "expanded__atom1_p_true": 0.2,
            "expanded__atom1_state=core_match": 1.0,
        }

        self.assertEqual(_aggregate_atoms([left]), _aggregate_atoms([swapped]))

    def test_expanded_baseline_and_rich_content_selection_are_scoped(self):
        record = {
            "id": "secret-row-id",
            "date": "2026-10-01",
            "__text__": "raw sentence",
            "native__hmm": 0.3,
            "expanded__month": 10,
            "expanded__mean_parse_q": 0.8,
            "rich__content_subject_hooly": 1.0,
            "rich__content_attribute_price": 1.0,
            "rich__summary_parse_confidence": 0.9,
            "rich__unrelated_note": 1.0,
        }
        baseline = _structured([record], "expanded")[0]
        rich = _structured([record], "rich")[0]
        no_parse = _structured([record], "rich_no_parse")[0]
        bags = _rich_bags([record])[0]

        self.assertEqual(set(baseline), {"native__hmm", "expanded__month", "expanded__mean_parse_q"})
        self.assertIn("rich__summary_parse_confidence", rich)
        self.assertNotIn("rich__summary_parse_confidence", no_parse)
        self.assertNotIn("expanded__mean_parse_q", no_parse)
        self.assertIn("rich__content_subject_hooly", bags)
        self.assertIn("rich__content_attribute_price", bags)
        self.assertNotIn("rich__summary_parse_confidence", bags)
        self.assertNotIn("rich__unrelated_note", bags)
        for feature_set in (baseline, rich, no_parse, bags):
            self.assertNotIn("id", feature_set)
            self.assertNotIn("__text__", feature_set)

    def test_tfidf_vocabulary_is_learned_from_training_text_only(self):
        train, labels = self._records(30)
        held = {**train[0], "__text__": "raresignalheldouttoken never seen before"}
        config = next(c for c in CONFIGS if c["kind"] == "word_svd32")
        model = make_feature_model(config, {"n_estimators": 2, "max_depth": 1}, seed=42)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(train, labels)
        tfidf = model.named_steps["features"].transformer_list[1][1].named_steps["tfidf"]

        self.assertNotIn("raresignalheldouttoken", tfidf.vocabulary_)
        self.assertNotIn("never", tfidf.vocabulary_)
        transformed = model.predict_proba([held])
        self.assertEqual(transformed.shape, (1, 3))

    def test_two_fold_fit_predict_works_with_string_labels(self):
        rows, labels = self._records(36)
        config = next(c for c in CONFIGS if c["id"] == "word_binary_chi2_128")
        for fold in (0, 1):
            train_indices = [i for i in range(len(rows)) if i % 2 != fold]
            test_indices = [i for i in range(len(rows)) if i % 2 == fold]
            model = make_feature_model(config, {"n_estimators": 2, "max_depth": 1}, seed=42)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit([rows[i] for i in train_indices], [labels[i] for i in train_indices])
                probabilities = model.predict_proba([rows[i] for i in test_indices[:3]])
            self.assertEqual(model.classes_.tolist(), ["false", "true", "unsure"])
            self.assertEqual(probabilities.shape, (3, 3))
            np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(3))

    def test_all_declared_variants_fit_and_predict(self):
        rows, labels = self._records(36)
        self.assertEqual(len({config["id"] for config in CONFIGS}), len(CONFIGS))
        for config in CONFIGS:
            with self.subTest(variant=config["id"]), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = make_feature_model(config, {"n_estimators": 2, "max_depth": 1}, seed=42)
                model.fit(rows, labels)
                probabilities = model.predict_proba(rows[:2])
                self.assertEqual(probabilities.shape, (2, 3))
                np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(2), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
