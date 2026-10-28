"""Subprocess checks for split-input rejection before loading upstream model code."""

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "integrations" / "daria_features.py"


class DariaFeatureInputBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.runtime = self.path / "runtime.csv"
        self.forms = self.path / "forms.csv"

    def write_csv(self, path, fields, rows):
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def invoke(self):
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--source-dir", str(self.path / "missing-source"),
             "--input", str(self.runtime), "--forms", str(self.forms),
             "--lexicon", str(self.path / "missing-lexicon.json"),
             "--calibration-fit-ids", str(self.path / "missing-fit-ids.json"),
             "--output", str(self.path / "features.json")],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )

    def base_inputs(self, label="true", form_rows=None):
        runtime_row = {"id": "train-1", "text": "Hooly has 5 users", "date": "2026-10-01", "label": label}
        self.write_csv(self.runtime, ["id", "text", "date", "label"], [runtime_row])
        forms = form_rows if form_rows is not None else [runtime_row | {"form": "[]"}]
        self.write_csv(self.forms, ["id", "text", "date", "label", "form"], forms)

    def test_invalid_label_is_rejected_before_loading_source_snapshot(self):
        self.base_inputs(label="maybe")

        result = self.invoke()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Invalid runtime label", result.stderr)
        self.assertNotIn("ModuleNotFoundError", result.stderr)

    def test_forms_cannot_include_pending_rows_or_omit_training_rows(self):
        self.base_inputs(form_rows=[])

        result = self.invoke()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Forms must cover exactly the labelled training pool", result.stderr)
        self.assertNotIn("ModuleNotFoundError", result.stderr)

    def test_form_fingerprint_must_match_runtime_row(self):
        self.base_inputs(form_rows=[{"id": "train-1", "text": "different text",
                                    "date": "2026-10-01", "label": "true", "form": "[]"}])

        result = self.invoke()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Form fingerprint mismatch", result.stderr)
        self.assertNotIn("ModuleNotFoundError", result.stderr)

    def test_short_csv_row_is_rejected(self):
        self.runtime.write_text("id,text,date,label\ntrain-1,Hooly has 5 users,2026-10-01\n", encoding="utf-8")
        self.write_csv(self.forms, ["id", "text", "date", "label", "form"], [])

        result = self.invoke()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("row has missing columns", result.stderr)
        self.assertNotIn("ModuleNotFoundError", result.stderr)


if __name__ == "__main__":
    unittest.main()
