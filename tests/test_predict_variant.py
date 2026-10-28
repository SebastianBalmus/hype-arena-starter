"""CLI contract checks for experiments/predict.py."""

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "predict.py"
FIELDS = ["id", "type", "text", "form", "label", "confidence"]


def write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "text", "date", "label"])
        writer.writeheader()
        writer.writerows(rows)


class PredictVariantCliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.train = self.root / "train.csv"
        self.pending = self.root / "pending.csv"
        self.output = self.root / "output.csv"
        write_rows(self.train, [
            {"id": "t1", "text": "Hooly cloud costs twenty euros monthly.", "date": "2026-01-01", "label": "true"},
            {"id": "t2", "text": "Hooly cloud costs thirty euros monthly.", "date": "2026-01-02", "label": "false"},
            {"id": "t3", "text": "The report gives no cloud price.", "date": "2026-01-03", "label": "unsure"},
        ])

    def run_cli(self):
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--train", str(self.train), "--input", str(self.pending),
             "--output", str(self.output), "--model", "word_char_lr_balanced"],
            cwd=ROOT, capture_output=True, text=True)

    def test_trains_on_labelled_rows_and_outputs_only_blank_ids_in_contract_shape(self):
        write_rows(self.pending, [
            {"id": "t4", "text": "Hooly cloud costs forty euros monthly.", "date": "2026-01-04", "label": ""},
            {"id": "t5", "text": "No price is stated for Hooly cloud.", "date": "2026-01-05", "label": ""},
            {"id": "ignored", "text": "already labelled", "date": "2026-01-06", "label": "true"},
        ])
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        with self.output.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            self.assertEqual(reader.fieldnames, FIELDS)
        self.assertEqual([row["id"] for row in rows], ["t4", "t5"])
        self.assertTrue(all(row["label"] in {"true", "false", "unsure"} for row in rows))
        self.assertTrue(all(row["type"] == row["text"] == row["form"] == "" for row in rows))

    def test_rejects_blank_input_id_found_in_labelled_training_file(self):
        write_rows(self.pending, [
            {"id": "t1", "text": "Accidentally held out a training ID.", "date": "2026-01-07", "label": ""},
        ])
        result = self.run_cli()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Blank input IDs also appear", result.stderr)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
