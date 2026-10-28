import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

from experiments.daria_gb_grid import (FoldCachedGradientBoosting, LABELS,
                                       fixed_050_macro_scorer, load_cache)
from evaluation.run_daria_gb_grid import make_folds


class FoldCachedGradientBoostingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="daria-gb-grid-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.train_ids = [f"train-{i}" for i in range(12)]
        self.labels = ["true", "false", "unsure"] * 4
        self.pending_ids = [f"validation-{i}" for i in range(4)]
        self.cache_path = self.root / "fold-cache.json"
        train_features = [self._features(i) for i in range(len(self.train_ids))]
        pending = [{"id": rid, "features": self._features(i + 12)}
                   for i, rid in enumerate(self.pending_ids)]
        cache = {
            "train_ids": self.train_ids,
            "train_labels": self.labels,
            "audit": {"train_ids": self.train_ids, "pending_ids": self.pending_ids},
            "train_features": train_features,
            "pending": pending,
        }
        self.cache_path.write_text(json.dumps(cache))
        self.manifest_path = self.root / "manifest.json"
        self.manifest_path.write_text(json.dumps({"caches": [{
            "train_ids": self.train_ids,
            "test_ids": self.pending_ids,
            "path": str(self.cache_path),
        }]}))
        load_cache.cache_clear()

    @staticmethod
    def _features(i):
        return {
            "native__windows": float(i % 5) / 4,
            "native__hmm": float(i % 3) / 2,
            "native__agree": float(i % 2),
            "expanded__n_atoms": float(i % 3),
            "expanded__parsed": 1.0,
        }

    def _estimator(self):
        return FoldCachedGradientBoosting(
            str(self.manifest_path), n_estimators=3, learning_rate=0.1,
            max_depth=1, min_samples_leaf=1, subsample=1.0, random_state=7)

    def test_fit_rejects_any_pool_without_an_exact_cache_match(self):
        estimator = self._estimator()
        changed_pool = self.train_ids[:-1] + ["different-row"]
        with self.assertRaisesRegex(ValueError, "exact training pool"):
            estimator.fit(np.asarray(changed_pool, dtype=object), self.labels)

    def test_fit_rejects_labels_that_disagree_with_cached_rows(self):
        labels = list(self.labels)
        labels[0] = "false"
        with self.assertRaisesRegex(ValueError, "labels disagree"):
            self._estimator().fit(np.asarray(self.train_ids, dtype=object), labels)

    def test_predictions_are_limited_to_this_folds_pending_ids(self):
        estimator = self._estimator().fit(np.asarray(self.train_ids, dtype=object), self.labels)
        probabilities = estimator.predict_proba(np.asarray(self.pending_ids[:2], dtype=object))
        self.assertEqual(probabilities.shape, (2, 3))
        with self.assertRaisesRegex(ValueError, "not in this fold"):
            estimator.predict_proba(np.asarray([self.train_ids[0]], dtype=object))
        with self.assertRaisesRegex(ValueError, "not in this fold"):
            estimator.predict_proba(np.asarray(["another-fold-row"], dtype=object))

    def test_fit_rejects_corrupted_pending_pool_audit(self):
        cache = json.loads(self.cache_path.read_text())
        cache["audit"]["pending_ids"] = self.pending_ids[:-1]
        self.cache_path.write_text(json.dumps(cache))
        load_cache.cache_clear()

        with self.assertRaisesRegex(ValueError, "pending-pool audit"):
            self._estimator().fit(np.asarray(self.train_ids, dtype=object), self.labels)

    def test_fit_rejects_duplicate_pending_rows(self):
        cache = json.loads(self.cache_path.read_text())
        cache["pending"].append(dict(cache["pending"][0]))
        self.cache_path.write_text(json.dumps(cache))
        load_cache.cache_clear()

        with self.assertRaisesRegex(ValueError, "pending-pool audit"):
            self._estimator().fit(np.asarray(self.train_ids, dtype=object), self.labels)

    def test_fixed050_keeps_native_unsure_and_macro_f1_includes_it(self):
        estimator = self._estimator().fit(np.asarray(self.train_ids, dtype=object), self.labels)
        # sklearn sorts string classes as false, true, unsure. Exercise a native unsure
        # and the exact 0.50 boundary without depending on fitted model scores.
        estimator.classes_ = np.asarray(["false", "true", "unsure"], dtype=object)
        fixed = np.asarray([[0.10, 0.20, 0.70],
                            [0.50, 0.30, 0.20],
                            [0.40, 0.35, 0.25]])
        estimator.model_.predict_proba = lambda _x: fixed
        predicted = estimator.predict(np.asarray(self.pending_ids[:3], dtype=object))
        self.assertEqual(predicted.tolist(), ["unsure", "false", "unsure"])

        gold = ["true", "false", "unsure"]
        macro = f1_score(gold, predicted, labels=LABELS, average="macro", zero_division=0)
        # Per-class F1 is 0, 1, 2/3; the unsure example remains in the macro average.
        self.assertAlmostEqual(macro, 5.0 / 9.0)

    def test_estimator_is_sklearn_cloneable_and_exposes_search_parameters(self):
        estimator = self._estimator()
        params = estimator.get_params(deep=False)
        self.assertEqual(params["cache_manifest"], str(self.manifest_path))
        self.assertEqual(params["n_estimators"], 3)
        self.assertEqual(params["random_state"], 7)
        copied = clone(estimator)
        self.assertIsNot(copied, estimator)
        self.assertEqual(copied.get_params(deep=False), params)

    def test_stratified_folds_cover_each_row_once(self):
        rows = [{"id": f"row-{i}", "label": ["true", "false", "unsure"][i % 3],
                 "date": f"2026-10-{i + 1:02d}"} for i in range(18)]
        folds = make_folds(rows, "stratified")

        self.assertEqual(len(folds), 3)
        test_indices = [int(i) for _, test in folds for i in test]
        self.assertEqual(sorted(test_indices), list(range(len(rows))))
        self.assertEqual(len(test_indices), len(set(test_indices)))

    def test_temporal_folds_keep_same_date_rows_together_and_forward_only(self):
        rows = [{"id": f"row-{day}-{j}", "label": ["true", "false", "unsure"][j % 3],
                 "date": f"2026-10-{day:02d}"}
                for day in range(1, 10) for j in range(2)]
        folds = make_folds(rows, "temporal")

        self.assertEqual(len(folds), 3)
        for train, test in folds:
            train_dates = {rows[int(i)]["date"] for i in train}
            test_dates = {rows[int(i)]["date"] for i in test}
            self.assertFalse(train_dates & test_dates)
            self.assertLess(max(train_dates), min(test_dates))
            for day in range(1, 10):
                indices = {i for i, row in enumerate(rows)
                           if row["date"] == f"2026-10-{day:02d}"}
                self.assertFalse(indices & set(train) and indices & set(test))

    def test_grid_search_scores_string_labels_with_fixed050_macro_f1(self):
        # Two complementary fold caches model the required feature rebuild for each
        # exact training subset. Every training and validation fold contains all labels.
        ids = np.asarray(self.train_ids, dtype=object)
        labels = np.asarray(self.labels, dtype=object)
        folds = [
            (np.asarray([0, 1, 2, 6, 7, 8]), np.asarray([3, 4, 5, 9, 10, 11])),
            (np.asarray([3, 4, 5, 9, 10, 11]), np.asarray([0, 1, 2, 6, 7, 8])),
        ]
        caches = []
        for train_indices, test_indices in folds:
            fold_train_ids = ids[train_indices].tolist()
            fold_pending_ids = ids[test_indices].tolist()
            cache_path = self.root / f"fold-{len(caches)}.json"
            fold_cache = {
                "train_ids": fold_train_ids,
                "train_labels": labels[train_indices].tolist(),
                "audit": {"train_ids": fold_train_ids,
                          "pending_ids": fold_pending_ids},
                "train_features": [self._features(int(i)) for i in train_indices],
                "pending": [{"id": str(ids[i]), "features": self._features(int(i))}
                            for i in test_indices],
            }
            cache_path.write_text(json.dumps(fold_cache))
            caches.append({"train_ids": fold_train_ids, "test_ids": fold_pending_ids,
                           "path": str(cache_path)})
        self.manifest_path.write_text(json.dumps({"caches": caches}))
        load_cache.cache_clear()

        search = GridSearchCV(
            FoldCachedGradientBoosting(str(self.manifest_path), random_state=11),
            {"n_estimators": [2]}, cv=folds, scoring=fixed_050_macro_scorer,
            refit=False, error_score="raise", n_jobs=1,
        )
        search.fit(ids, labels)

        self.assertEqual(search.best_params_, {"n_estimators": 2})
        self.assertEqual(len(search.cv_results_["mean_test_score"]), 1)
        self.assertTrue(np.isfinite(search.best_score_))


if __name__ == "__main__":
    unittest.main()
