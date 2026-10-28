import contextlib
from decimal import Decimal
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_annotations import run, signature
from entity_linking import EntityLinker, build_registry
from entity_linking.values import annotate, exact_json


class AnnotationAuditTests(unittest.TestCase):
    def test_independent_reference_comparison_respects_units_and_operators(self):
        self.assertEqual(signature(dict(kind="quantity", value=1, unit="GB")),
                         signature(dict(kind="quantity", value=1000, unit="MB")))
        self.assertEqual(signature(dict(kind="quantity", value=Decimal("6.4"), unit="in")),
                         signature(dict(kind="quantity", value=Decimal("6.4"), unit="inch")))
        self.assertNotEqual(signature(dict(kind="quantity", value=100, unit="customer", op="gt")),
                            signature(dict(kind="quantity", value=100, unit="customer")))

    def test_audit_rejects_corrupted_spans(self):
        lexicon = json.loads((ROOT / "data/lexicon.json").read_text())
        text = "Hooly AI Pro costs €49 per month."
        result = annotate(text, EntityLinker(build_registry(lexicon)))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source.csv"
            values = [{"subject": "Hooly AI Pro", "attribute": "price", "value": {
                "kind": "money", "value": 49, "currency": "EUR", "per": "month"}}]
            with source.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", "text", "date", "form"])
                writer.writeheader()
                writer.writerow(dict(id="a", text=text, date="2026-01-01", form=json.dumps(values)))
            annotations = root / "annotations"
            annotations.mkdir()
            record = {"id": "a", "text": text, **result}
            (annotations / "annotations.jsonl").write_text(exact_json(record) + "\n")
            with contextlib.redirect_stdout(io.StringIO()):
                run(source, annotations, root / "audit")
            summary = json.loads((root / "audit/summary.json").read_text())
            self.assertEqual(summary["reference_unique_values_found"], 1)
            record["values"][0]["start"] += 1
            (annotations / "annotations.jsonl").write_text(exact_json(record) + "\n")
            with self.assertRaises(AssertionError):
                run(source, annotations, root / "bad-audit")


if __name__ == "__main__":
    unittest.main()
