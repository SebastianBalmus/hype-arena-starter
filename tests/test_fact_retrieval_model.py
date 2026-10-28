import json
import csv
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "fact_retrieval_predict.py"
sys.path.insert(0, str(ROOT))
from experiments.fact_retrieval_predict import FactRetrievalClassifier
from experiments.kb_predict import match_training_forms


def sample_rows():
    rows = []
    for label, phrase in (("true", "confirmed"), ("false", "denied"), ("unsure", "unclear")):
        for i in range(5):
            text = f"Hooly {phrase} launch report lists 17 customers, item {i}."
            rows.append({"id": f"{label}-{i}", "text": text, "date": f"2026-10-{20+i:02d}",
                         "label": label, "form": json.dumps([{"subject": "Hooly", "attribute": "launch",
                         "value": {"kind": "quantity", "value": 17, "unit": "customer"}}])})
    return rows


class FactRetrievalClassifierTests(unittest.TestCase):
    def setUp(self):
        self.lexicon = {"subjects": [{"name": "Hooly", "aliases": [], "attributes": ["launch"]}]}
        self.training = sample_rows()

    def test_fold_catalogues_only_use_fold_reference_ids(self):
        model = FactRetrievalClassifier(self.lexicon).fit(self.training)
        self.assertEqual([x["scope"] for x in model.claim_catalogues_],
                         ["fold-0", "fold-1", "fold-2", "fold-3", "fold-4", "full"])
        folds = {f["fold_index"]: f for f in model.oof_folds_}
        for record in model.claim_catalogues_[:-1]:
            fold_index = int(record["scope"].split("-")[1])
            allowed = set(folds[fold_index]["reference_ids"])
            self.assertEqual(set(record["reference_ids"]), allowed)
            for candidate in record["records"]:
                self.assertTrue(set(candidate["source_ids"]) <= allowed)
                self.assertFalse(set(candidate["source_ids"]) & set(folds[fold_index]["query_ids"]))
                self.assertTrue(all(next(r for r in self.training if r["id"] == rid)["label"] == "true"
                                    for rid in candidate["source_ids"]))

    def test_oof_traces_have_auxiliary_provenance_and_pending_label_is_ignored(self):
        model = FactRetrievalClassifier(self.lexicon, mode="observed").fit(self.training)
        self.assertEqual(Counter(t["query_id"] for t in model.oof_traces_), Counter(r["id"] for r in self.training))
        self.assertTrue(all("auxiliary_claims" in t for t in model.oof_traces_))
        pending = [{"id": "pending", "text": "Hooly confirmed launch report lists 17 customers.",
                    "date": "2026-10-29", "label": ""}]
        first = model.predict_rows(pending)
        pending[0]["label"] = "false"
        second = model.predict_rows(pending)
        self.assertEqual(first, second)

    def test_forms_are_only_on_reference_rows_and_catalogue_source_ids_are_labelled(self):
        model = FactRetrievalClassifier(self.lexicon, mode="generated").fit(self.training)
        refs = {r["id"] for r in self.training}
        for scope in model.claim_catalogues_:
            self.assertTrue(set(scope["reference_ids"]) <= refs)
            for candidate in scope["records"]:
                self.assertTrue(set(candidate["source_ids"]) <= set(scope["reference_ids"]))
        full = model.claim_catalogues_[-1]["records"]
        self.assertTrue(any(c["type"] == "fact" and c["form"]["value"].get("value") == 17 for c in full))

    def test_training_form_matcher_rejects_pending_or_mismatched_rows(self):
        runtime = [dict(r) for r in self.training]
        runtime.append({"id": "pending", "text": "Hooly launch?", "date": "2026-04-01", "label": ""})
        form_rows = [{**r, "form": r["form"]} for r in self.training]
        self.assertEqual(len(match_training_forms(runtime, form_rows)), len(self.training))
        with self.assertRaisesRegex(ValueError, "exactly the labelled runtime IDs"):
            match_training_forms(runtime, form_rows + [{**runtime[-1], "form": "[]"}])
        changed = [dict(r) for r in form_rows]
        changed[0]["text"] += " altered"
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            match_training_forms(runtime, changed)

    def test_cli_writes_predictions_and_fold_scoped_catalogue(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runtime, forms = root / "input.csv", root / "forms.csv"
            lexicon, output, artifacts = root / "lexicon.json", root / "predictions.csv", root / "artifacts"
            with runtime.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["id", "text", "date", "label"])
                writer.writeheader()
                writer.writerows([{k: r[k] for k in ("id", "text", "date", "label")} for r in self.training])
                writer.writerow({"id": "pending", "text": "Hooly confirmed launch report lists 22 customers.",
                                 "date": "2026-10-29", "label": ""})
            with forms.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["id", "text", "date", "label", "form"])
                writer.writeheader(); writer.writerows(self.training)
            lexicon.write_text(json.dumps(self.lexicon), encoding="utf-8")
            proc = subprocess.run([sys.executable, str(SCRIPT), "--input", str(runtime),
                                   "--training-forms", str(forms), "--lexicon", str(lexicon),
                                   "--output", str(output), "--artifacts", str(artifacts)],
                                  cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            with output.open(encoding="utf-8", newline="") as stream:
                predictions = list(csv.DictReader(stream))
            self.assertEqual([r["id"] for r in predictions], ["pending"])
            catalogue = artifacts / "claim_catalogues.json.gz"
            self.assertTrue(catalogue.is_file())
            self.assertTrue((artifacts / "claims_summary.json").is_file())
            with forms.open("a", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["id", "text", "date", "label", "form"])
                writer.writerow({"id": "pending", "text": "Hooly pending", "date": "2026-10-29",
                                 "label": "", "form": "[]"})
            rejected = subprocess.run([sys.executable, str(SCRIPT), "--input", str(runtime),
                                       "--training-forms", str(forms), "--lexicon", str(lexicon),
                                       "--output", str(root / "rejected.csv"), "--artifacts", str(root / "rejected")],
                                      cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("exactly the labelled runtime IDs", rejected.stderr)
            self.assertFalse((root / "rejected.csv").exists())


if __name__ == "__main__":
    unittest.main()
