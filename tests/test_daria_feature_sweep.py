import unittest

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from evaluation.scoring import score_rows
from experiments.daria_feature_sweep import audit_cache, fit_predict
from experiments.daria_feature_variants import CONFIGS, make_feature_model
from experiments.daria_sklearn import select_features


def sample_cache(n_train=18, n_pending=3, constant=False):
    train_ids = [f"train-{i}" for i in range(n_train)]
    labels = [("true", "false", "unsure")[i % 3] for i in range(n_train)]

    def features(i):
        if constant:
            value = 0.0
        else:
            value = float(i % 3)
        return {
            "native__windows": value,
            "native__hmm": value / 2,
            "native__agree": value / 3,
            "expanded__n_atoms": float(i % 2),
            "expanded__parsed": 1.0,
            "expanded__year": 2026.0,
            "expanded__mean_parse_q": 0.9,
            "__text__": f"Hooly item type {i % 3}",
        }

    pending_ids = [f"pending-{i}" for i in range(n_pending)]
    return {
        "train_ids": train_ids,
        "train_labels": labels,
        "train_features": [features(i) for i in range(n_train)],
        "pending": [{"id": rid, "features": features(i)}
                    for i, rid in enumerate(pending_ids)],
        "audit": {
            "train_ids": train_ids.copy(),
            "pending_ids": pending_ids.copy(),
            "pending_labels_and_forms_absent": True,
        },
    }


class DariaFeatureSweepTests(unittest.TestCase):
    def test_cache_audit_rejects_training_pending_overlap(self):
        cache = sample_cache()
        overlap_id = cache["train_ids"][0]
        cache["pending"][0]["id"] = overlap_id
        cache["audit"]["pending_ids"][0] = overlap_id

        with self.assertRaisesRegex(ValueError, "unique and disjoint"):
            audit_cache(cache)

    def test_cache_audit_rejects_training_feature_alignment_length_mismatch(self):
        cache = sample_cache()
        cache["train_features"].pop()

        with self.assertRaisesRegex(ValueError, "align uniquely"):
            audit_cache(cache)

    def test_cache_audit_rejects_pending_audit_mismatch(self):
        cache = sample_cache()
        cache["audit"]["pending_ids"].pop()

        with self.assertRaisesRegex(ValueError, "audit IDs do not match"):
            audit_cache(cache)

    def test_expanded_baseline_matches_original_dictvectorizer_gb_pipeline(self):
        cache = sample_cache(n_train=45, n_pending=4)
        params = {"n_estimators": 7, "learning_rate": 0.08,
                  "max_depth": 1, "min_samples_leaf": 1}
        baseline_cfg = next(c for c in CONFIGS if c["id"] == "expanded_none")

        new_pipeline = make_feature_model(baseline_cfg, params, seed=42).fit(
            cache["train_features"], cache["train_labels"])
        original_pipeline = Pipeline([
            ("dict", DictVectorizer(sparse=False, sort=True)),
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("model", GradientBoostingClassifier(**params, random_state=42)),
        ]).fit(select_features(cache["train_features"], "expanded"), cache["train_labels"])
        pending = [row["features"] for row in cache["pending"]]

        np.testing.assert_array_equal(new_pipeline.classes_, original_pipeline.classes_)
        np.testing.assert_allclose(new_pipeline.predict_proba(pending),
                                   original_pipeline.predict_proba(select_features(pending, "expanded")),
                                   rtol=0.0, atol=0.0)

    def test_baseline_fit_predict_returns_valid_string_label_scores_and_null_unsure_confidence(self):
        cache = sample_cache(n_train=18, n_pending=3, constant=True)
        baseline_cfg = next(c for c in CONFIGS if c["id"] == "expanded_none")
        predictions, timing = fit_predict(
            baseline_cfg, cache, {"n_estimators": 3, "max_depth": 1})

        self.assertEqual(timing["status"], "ok")
        self.assertEqual([row["label"] for row in predictions], ["unsure"] * 3)
        self.assertTrue(all(row["confidence"] is None for row in predictions))
        self.assertTrue(all(row["decision_reason"] == "low_confidence" for row in predictions))
        for row in predictions:
            self.assertAlmostEqual(row["p_true"] + row["p_false"] + row["p_unsure"], 1.0)

        gold = [{"id": f"pending-{i}", "label": label}
                for i, label in enumerate(("true", "false", "unsure"))]
        metrics, audit, scored = score_rows(gold, predictions)
        self.assertTrue(audit["eligible_for_comparison"])
        self.assertEqual(metrics["rows"], 3)
        self.assertEqual(metrics["per_class"]["unsure"]["support"], 1)
        self.assertTrue(all(row["output_valid"] for row in scored))


if __name__ == "__main__":
    unittest.main()
