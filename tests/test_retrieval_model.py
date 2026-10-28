import csv
from collections import Counter
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "retrieval_predict.py"
sys.path.insert(0, str(ROOT))
from experiments.retrieval_predict import RetrievalClassifier


def rows():
    result = []
    for label, phrase in (("true", "confirmed launch"), ("false", "denied launch"), ("unsure", "unclear launch")):
        for i in range(6):
            result.append({"id": f"{label}-{i}", "text": f"Hooly {phrase} report item {i}.",
                           "date": f"2026-01-{i+1:02d}", "label": label})
    return result


class RetrievalClassifierTests(unittest.TestCase):
    def setUp(self):
        self.training = rows()
        self.lexicon = {"subjects": [{"name": "Hooly", "aliases": [], "attributes": ["launch"]}]}

    def test_oof_reference_sets_exclude_each_query_fold(self):
        model = RetrievalClassifier(self.lexicon).fit(self.training)
        self.assertEqual(Counter(t["query_id"] for t in model.oof_traces_), Counter(r["id"] for r in self.training))
        for fold in model.oof_folds_:
            self.assertFalse(set(fold["query_ids"]) & set(fold["reference_ids"]))
        for item in model.oof_traces_:
            fold = next(f for f in model.oof_folds_ if f["fold_index"] == item["fold_index"])
            self.assertNotIn(item["query_id"], fold["reference_ids"])
            self.assertEqual(item["fold_index"], next(f["fold_index"] for f in model.oof_folds_ if item["query_id"] in f["query_ids"]))

    def test_oof_query_payload_is_label_sanitized_and_neighbor_labels_are_reference_only(self):
        model = RetrievalClassifier(self.lexicon).fit(self.training)
        training_by_id = {r["id"]: r for r in self.training}
        for item in model.oof_traces_:
            self.assertIn(training_by_id[item["query_id"]]["label"], {"true", "false", "unsure"})
            fold = next(f for f in model.oof_folds_ if f["fold_index"] == item["fold_index"])
            self.assertNotIn(item["query_id"], fold["reference_ids"])
            # Every visible candidate label belongs to an admissible reference.
            ref_labels = {r["id"]: r["label"] for r in self.training if r["id"] in fold["reference_ids"]}
            for neighbors in item["neighbors"].values():
                for neighbor in neighbors:
                    self.assertEqual(neighbor["label"], ref_labels[neighbor["id"]])
            for view_classes in item.get("class_neighbors", {}).values():
                for label, neighbors in view_classes.items():
                    self.assertLessEqual(len(neighbors), 5)
                    for neighbor in neighbors:
                        self.assertIn(neighbor["id"], ref_labels)
                        self.assertEqual(neighbor["label"], label)

    def test_pending_predictions_have_full_probability_distributions_and_traces(self):
        model = RetrievalClassifier(self.lexicon, variant="raw_masked").fit(self.training)
        pending = [{"id": "pending", "text": "Hooly launch costs 123 euros monthly", "date": "2026-03-01", "label": "false"}]
        predictions, traces = model.predict_rows(pending)
        self.assertEqual(predictions[0]["id"], "pending")
        self.assertAlmostEqual(sum(predictions[0][f"p_{x}"] for x in ("true", "false", "unsure")), 1.0)
        self.assertIn(predictions[0]["label"], {"true", "false", "unsure"})
        if predictions[0]["label"] == "unsure":
            self.assertEqual(predictions[0]["confidence"], "")
        else:
            self.assertGreaterEqual(predictions[0]["confidence"], 0.5)
        self.assertEqual(traces[0]["route"], "retrieval_classifier")
        self.assertEqual(traces[0]["query_id"], "pending")
        changed = dict(pending[0], label="true")
        pred2, trace2 = model.predict_rows([changed])
        self.assertEqual(predictions, pred2)
        self.assertEqual(traces, trace2)

    def test_fit_sends_blank_labels_to_oof_retrieval_transform(self):
        import experiments.retrieval_predict as module
        original = module.RetrievalIndex
        seen = []
        class SpyIndex(original):
            def transform(self, queries):
                seen.extend(r["label"] for r in queries)
                return super().transform(queries)
        with patch.object(module, "RetrievalIndex", SpyIndex):
            RetrievalClassifier(self.lexicon).fit(self.training)
        self.assertTrue(seen)
        self.assertEqual(set(seen), {""})

    def test_cli_success_writes_predictions_and_all_oof_traces(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            inp, lex, out, arts = root / "in.csv", root / "lex.json", root / "out.csv", root / "arts"
            with inp.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["date", "label", "id", "text"])
                writer.writeheader()
                writer.writerows([{**r, "text": r["text"]} for r in self.training] + [
                    {"date": "2026-03-01", "label": "", "id": "pending", "text": "Hooly unclear launch report"}])
            lex.write_text(json.dumps(self.lexicon), encoding="utf-8")
            result = subprocess.run([sys.executable, str(SCRIPT), "--input", str(inp), "--lexicon", str(lex),
                                     "--output", str(out), "--artifacts", str(arts)], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            with out.open(encoding="utf-8", newline="") as f:
                predictions = list(csv.DictReader(f))
            self.assertEqual([p["id"] for p in predictions], ["pending"])
            self.assertIn(predictions[0]["decision_reason"], {"model_unsure", "decided", "low_confidence"})
            oof = [json.loads(line) for line in (arts / "oof_traces.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(Counter(t["query_id"] for t in oof), Counter(r["id"] for r in self.training))

    def test_cli_rejects_extra_gold_form_column(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            inp, lex, out, arts = root / "in.csv", root / "lex.json", root / "out.csv", root / "arts"
            with inp.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["id", "text", "date", "label", "form"])
                writer.writeheader(); writer.writerow({"id": "x", "text": "sentence", "date": "2026-01-01", "label": "true", "form": "{}"})
            lex.write_text(json.dumps(self.lexicon), encoding="utf-8")
            result = subprocess.run([sys.executable, str(SCRIPT), "--input", str(inp), "--lexicon", str(lex),
                                     "--output", str(out), "--artifacts", str(arts)], cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("expected exactly columns", result.stderr)
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
