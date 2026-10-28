"""Focused contract tests for classification scoring and input sanitization."""

import math
import csv
import tempfile
import unittest
from pathlib import Path

from evaluation.scoring import score_rows
from evaluation.run_text_experiments import sanitized_rows
from evaluation.inner_model_selection import score_predictions


def prediction(rid, label, confidence=None, reason="decided", **probs):
    return {"id": rid, "label": label, "confidence": confidence,
            "decision_reason": reason, **probs}


class ScoreRowsTests(unittest.TestCase):
    def test_three_class_metrics_coverage_and_briers_by_hand(self):
        gold = [{"id": "a", "label": "true"}, {"id": "b", "label": "false"},
                {"id": "c", "label": "unsure"}]
        preds = [prediction("a", "true", .8, p_true=.8, p_false=.1, p_unsure=.1),
                 prediction("b", "true", .6, p_true=.6, p_false=.3, p_unsure=.1),
                 prediction("c", "unsure", None, "model_unsure", p_true=.1, p_false=.2, p_unsure=.7)]
        m, errors, scored = score_rows(gold, preds)
        self.assertTrue(errors["eligible_for_comparison"])
        self.assertEqual(m["rows"], 3)
        self.assertAlmostEqual(m["accuracy"], 2 / 3)
        # true F1=2/3, false F1=0, unsure F1=1.
        self.assertAlmostEqual(m["macro_f1"], 5 / 9)
        self.assertAlmostEqual(m["decided_coverage"], 2 / 3)
        self.assertEqual((m["decided_correct"], m["decided_errors"]), (1, 1))
        self.assertAlmostEqual(m["correctness_brier"], ((.8 - 1) ** 2 + (.6 - 0) ** 2) / 2)
        # Multiclass squared errors: .06 + .86 + .14, averaged over three rows.
        self.assertAlmostEqual(m["multiclass_brier"], 1.06 / 3)
        self.assertEqual([row["correct"] for row in scored], [True, False, True])

    def test_duplicate_uses_first_and_missing_unknown_make_ineligible(self):
        gold = [{"id": "a", "label": "true"}, {"id": "b", "label": "false"}]
        preds = [prediction("a", "true", .8), prediction("a", "false", .9),
                 prediction("extra", "true", .8)]
        m, errors, scored = score_rows(gold, preds)
        self.assertFalse(errors["eligible_for_comparison"])
        self.assertEqual(errors["duplicate_count"], 1)
        self.assertEqual(errors["missing_count"], 1)
        self.assertEqual(errors["unknown_count"], 1)
        self.assertEqual(scored[0]["label"], "true")
        self.assertEqual(scored[1]["label"], "unsure")
        self.assertAlmostEqual(m["accuracy"], .5)

    def test_invalid_confidence_and_unsure_confidence_normalize_to_unsure(self):
        gold = [{"id": "nan", "label": "true"}, {"id": "abstain", "label": "false"}]
        preds = [prediction("nan", "true", math.nan), prediction("abstain", "unsure", .7, "model_unsure")]
        m, errors, scored = score_rows(gold, preds)
        self.assertFalse(errors["eligible_for_comparison"])
        self.assertEqual([r["label"] for r in scored], ["unsure", "unsure"])
        self.assertIn("invalid_confidence", scored[0]["error_codes"])
        self.assertIn("unsure_has_confidence", scored[1]["error_codes"])
        self.assertEqual(m["decided_count"], 0)

    def test_absent_class_uses_zero_division_metrics_and_flag(self):
        m, _, _ = score_rows([{"id": "a", "label": "true"}],
                             [prediction("a", "true", .8, p_true=1, p_false=0, p_unsure=0)])
        self.assertTrue(m["per_class"]["false"]["absent"])
        self.assertEqual(m["per_class"]["false"]["precision"], 0)
        self.assertEqual(m["per_class"]["false"]["recall"], 0)
        self.assertEqual(m["per_class"]["false"]["f1"], 0)

    def test_no_decisions_has_null_conditional_metrics(self):
        m, _, _ = score_rows([{"id": "a", "label": "unsure"}],
                             [prediction("a", "unsure", None, "model_unsure")])
        self.assertEqual(m["decided_coverage"], 0)
        self.assertIsNone(m["decided_accuracy"])
        self.assertIsNone(m["decided_risk"])
        self.assertIsNone(m["correctness_brier"])


class SanitizedRowsTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"id": "train", "text": "known", "date": "2026-01-01", "label": "true", "form": "secret"},
            {"id": "test", "text": "pending", "date": "2026-01-02", "label": "false", "form": "secret"},
        ]

    def test_runtime_shape_keeps_train_label_masks_test_and_drops_form(self):
        result = sanitized_rows(self.rows, ["train"], ["test"])
        self.assertEqual([set(row) for row in result], [
            {"id", "text", "date", "label"}, {"id", "text", "date", "label"}])
        self.assertEqual(result[0]["label"], "true")
        self.assertEqual(result[1]["label"], "")
        self.assertNotIn("form", result[0])

    def test_rejects_overlap_and_unknown_ids(self):
        with self.assertRaisesRegex(ValueError, "overlapping"):
            sanitized_rows(self.rows, ["train"], ["train"])
        with self.assertRaisesRegex(ValueError, "absent"):
            sanitized_rows(self.rows, ["train"], ["missing"])

    def test_validation_scoring_does_not_write_gold_to_method_input_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            prediction_path = directory / 'predictions.csv'
            with prediction_path.open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=['id', 'label', 'confidence', 'decision_reason'])
                writer.writeheader()
                writer.writerow({'id': 'test', 'label': 'true', 'confidence': 0.8,
                                 'decision_reason': 'decided'})
            metrics, errors, _ = score_predictions([{'id': 'test', 'label': 'true'}], prediction_path, None)
            self.assertTrue(errors['eligible_for_comparison'])
            self.assertEqual(metrics['accuracy'], 1.0)
            self.assertEqual({path.name for path in directory.iterdir()}, {'predictions.csv'})


if __name__ == "__main__":
    unittest.main()
