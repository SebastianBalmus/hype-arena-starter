import unittest

import numpy as np

from experiments.daria_sklearn import CONFIGS, make_candidate, select_features
from integrations.daria_candidates import probabilities_to_rows


class DariaSklearnTests(unittest.TestCase):
    def test_configs_are_unique_and_probability_capable(self):
        ids = [config["id"] for config in CONFIGS]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertGreaterEqual(len(CONFIGS), 45)
        for config in CONFIGS:
            with self.subTest(candidate=config["id"]):
                self.assertTrue(hasattr(make_candidate(config, seed=42), "predict_proba"))

    def test_feature_selection_limits_text_and_excludes_identifiers(self):
        features = [{
            "id": "row-17",
            "date": "2026-10-01",
            "native__windows": 0.25,
            "expanded__n_atoms": 2,
            "__text__": "Workspace had 500 users",
        }]

        native = select_features(features, "native")[0]
        expanded = select_features(features, "expanded")[0]
        with_text = select_features(features, "expanded_word")[0]
        number_masked = select_features(features, "expanded_word_num")[0]

        self.assertEqual(native, {"native__windows": 0.25})
        self.assertEqual(expanded, {
            "native__windows": 0.25,
            "expanded__n_atoms": 2,
        })
        self.assertNotIn("__text__", native)
        self.assertNotIn("__text__", expanded)
        self.assertEqual(with_text["__text__"], features[0]["__text__"])
        self.assertEqual(number_masked["__text__"], features[0]["__text__"])
        for selected in (native, expanded, with_text, number_masked):
            self.assertNotIn("id", selected)
            self.assertNotIn("date", selected)

    def test_preprocessing_is_fitted_only_on_training_rows(self):
        config = next(c for c in CONFIGS if c["id"] == "native_standard_lr")
        train = [
            {"native__windows": float(i), "native__hmm": float(i % 3), "native__agree": 0.1}
            for i in range(18)
        ]
        labels = ["true", "false", "unsure"] * 6
        model = make_candidate(config, seed=42).fit(train, labels)
        scaler = model.named_steps["scale"]
        fitted_mean = scaler.mean_.copy()

        held_out = {"native__windows": 1_000_000.0, "native__hmm": -500_000.0,
                    "native__agree": 900_000.0}
        transformed = model[:-1].transform([held_out])

        np.testing.assert_array_equal(scaler.mean_, fitted_mean)
        self.assertGreater(float(np.max(np.abs(transformed))), 1_000.0)

    def test_probability_policy_preserves_unsure_and_exact_half(self):
        rows = probabilities_to_rows(
            ["native-unsure", "below-half", "exact-half"],
            [
                [0.20, 0.10, 0.70],
                [0.49, 0.48, 0.03],
                [0.50, 0.30, 0.20],
            ],
            ("true", "false", "unsure"),
        )

        self.assertEqual(rows[0]["label"], "unsure")
        self.assertEqual(rows[0]["decision_reason"], "model_unsure")
        self.assertIsNone(rows[0]["confidence"])

        self.assertEqual(rows[1]["label"], "unsure")
        self.assertEqual(rows[1]["decision_reason"], "low_confidence")
        self.assertIsNone(rows[1]["confidence"])

        self.assertEqual(rows[2]["label"], "true")
        self.assertEqual(rows[2]["decision_reason"], "decided")
        self.assertEqual(rows[2]["confidence"], 0.5)


if __name__ == "__main__":
    unittest.main()
