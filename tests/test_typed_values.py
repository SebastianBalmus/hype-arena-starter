import csv
from decimal import Decimal
import json
from pathlib import Path
import sys
import unittest
import contextlib
import importlib.util
import io
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from entity_linking import EntityLinker, build_registry
from entity_linking.values import annotate, exact_json, parse_values


class TypedValueTests(unittest.TestCase):
    def setUp(self):
        self.linker = EntityLinker(build_registry(json.loads((ROOT / "data/lexicon.json").read_text())))

    def test_money_preserves_currency_period_and_decimal_precision(self):
        values, _ = parse_values("It costs €49,99 per user per month, or USD 299.99.")
        self.assertEqual(len(values), 2)
        self.assertEqual(values[0]["value"], Decimal("49.99"))
        self.assertEqual(values[0]["currency"], "EUR")
        self.assertEqual(values[0]["per"], "month")
        self.assertEqual(values[0]["basis"], "user")
        self.assertIsNone(values[1]["per"])
        self.assertEqual(values[1]["currency"], "USD")

    def test_quantity_units_number_words_and_scales(self):
        values, _ = parse_values("forty languages, 1.2 million users, 128GB and 3 m.")
        self.assertEqual([(v["value"], v["unit"]) for v in values],
                         [(40, "language"), (1200000, "user"), (128, "GB"), (3, "m")])

    def test_percent_and_operator_are_not_rounded_or_lost(self):
        values, _ = parse_values("at least 99.999% uptime; no more than 40 customers")
        self.assertEqual(values[0]["value"], Decimal("99.999"))
        self.assertEqual(values[0]["op"], "ge")
        self.assertEqual(values[1]["op"], "le")
        self.assertFalse(values[1]["negated_prefix"])

    def test_dates_preserve_day_month_year_precision(self):
        values, _ = parse_values("12 March 2026; March 2026; founded in 2025.")
        self.assertEqual([(v["value"], v["precision"]) for v in values],
                         [("2026-03-12", "day"), ("2026-03", "month"), ("2025", "year")])

    def test_invalid_date_and_range_do_not_create_exact_scalar_endpoints(self):
        values, unresolved = parse_values("2026-02-31; 10-20 GB; between 10 and 20 customers")
        self.assertEqual(values, [])
        self.assertIn("invalid_calendar_date", {value["reason"] for value in unresolved})
        self.assertIn("range_or_choice_not_resolved", {value["reason"] for value in unresolved})

    def test_approximation_and_negation_are_recorded_without_discarding_literal(self):
        values, _ = parse_values("not €49 per month; around 100 customers")
        self.assertTrue(values[0]["negated_prefix"])
        self.assertTrue(values[1]["approximate"])

    def test_billing_period_is_not_duplicated_as_duration(self):
        values, _ = parse_values("€49 per month; under a second; 14-day trial")
        self.assertEqual([value["kind"] for value in values], ["money", "duration", "duration"])
        self.assertEqual(values[1]["op"], "lt")
        self.assertEqual(values[2]["unit"], "day")

    def test_explicit_and_entity_following_versions(self):
        result = annotate("Hooly Gemini 2.2, version 3.1 and v4.0.1.", self.linker)
        self.assertEqual([value["value"] for value in result["values"]], ["2.2", "3.1", "4.0.1"])
        self.assertTrue(all(value["kind"] == "version" for value in result["values"]))
        result = annotate("Hooly Gemini 2.2 brings improvements.", self.linker)
        self.assertEqual([(value["kind"], value["value"]) for value in result["values"]], [("version", "2.2")])

    def test_entity_embedded_years_are_protected_and_local_value_ids_are_separate(self):
        result = annotate("Hooly Cloud Next 2026 lasts 3 days on 12 June 2026.", self.linker)
        self.assertEqual([value["kind"] for value in result["values"]], ["duration", "date"])
        self.assertEqual([value["value_id"] for value in result["values"]], [1, 1])
        self.assertIn("<ENTITY:18>", result["normalized_text"])

    def test_word_fragments_are_not_number_words(self):
        values, _ = parse_values("someone is alone; wonderful progress")
        self.assertEqual(values, [])

    def test_exact_json_keeps_decimal_as_number(self):
        encoded = exact_json({"value": Decimal("0.1234567890123456789")})
        self.assertEqual(json.loads(encoded, parse_float=Decimal)["value"], Decimal("0.1234567890123456789"))
        self.assertNotIn('"0.123', encoded)

    def test_signed_values_and_plus_comparison_preserve_meaning(self):
        values, _ = parse_values("-12% change; -3 customers; 10,000+ customers")
        self.assertEqual([(value["value"], value["op"]) for value in values],
                         [(-12, "eq"), (-3, "eq"), (10000, "ge")])

    def test_tagging_is_idempotent_and_does_not_parse_token_ids_as_values(self):
        first = annotate("Hooly AI Pro costs €49 per month.", self.linker)["normalized_text"]
        second = annotate(first, self.linker)
        self.assertEqual(second["normalized_text"], first)
        self.assertEqual(second["values"], [])

    def test_long_decimal_scale_keeps_all_digits(self):
        values, _ = parse_values("1.12345678901234567890123456789 million users")
        self.assertEqual(values[0]["value"], Decimal("1123456.78901234567890123456789"))

    def test_audit_versions_are_not_inches_or_benchmark_counts(self):
        for text, expected in [("Hooly Gemini 1.5 in production", "1.5"),
                               ("Hooly Gemini 2.0 benchmark", "2.0"),
                               ("Hooly Cloud CLI 2.3 ships today", "2.3"),
                               ("Release 1.4 in use", "1.4")]:
            with self.subTest(text=text):
                result = annotate(text, self.linker)
                self.assertEqual([(v["kind"], v["value"]) for v in result["values"]], [("version", expected)])
        values, _ = parse_values("A 6.4 in screen")
        self.assertEqual(values[0]["unit"], "in")

    def test_audit_count_nouns_plural_and_compound_units(self):
        values, _ = parse_values("120,000 watches; 4 data centre sites; 3 storage tiers")
        self.assertEqual([(v["value"], v["unit"]) for v in values], [(120000, "watch"), (4, "site"), (3, "tier")])

    def test_audit_billing_basis_does_not_hide_monthly_period(self):
        values, _ = parse_values("€9 per terabyte each month, and €3 per terabyte a month.")
        self.assertEqual([(v["value"], v["per"], v["basis"]) for v in values],
                         [(9, "month", "terabyte"), (3, "month", "terabyte")])

    def test_audit_identifiers_idioms_and_pronouns_are_not_quantities(self):
        for text in ["SOC 2 Type II certified", "not SOC 2 certified", "net zero date", "one of them"]:
            with self.subTest(text=text):
                values, _ = parse_values(text)
                self.assertEqual(values, [])

    def test_audit_photos_in_a_count_are_not_a_product_link(self):
        result = annotate("Hooly Photos holds 4,900,000,000 photos.", self.linker)
        self.assertEqual(len(result["mentions"]), 1)
        self.assertEqual(result["values"][0]["unit"], "photo")
        self.assertEqual(result["values"][0]["value"], 4900000000)
        self.assertEqual(result["suppressed_mentions"][0]["reason"], "lowercase_counted_noun_context")

    def test_original_and_normalized_offsets_roundtrip_every_dataset_row(self):
        with (ROOT / "data/practice.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            result = annotate(row["text"], self.linker)
            restored = result["normalized_text"]
            spans = sorted(result["mentions"] + result["values"], key=lambda item: item["normalized_start"])
            for span in reversed(spans):
                token = span.get("token", f"<ENTITY:{span.get('entity_id')}>")
                self.assertEqual(restored[span["normalized_start"]:span["normalized_end"]], token, row["id"])
                self.assertEqual(row["text"][span["start"]:span["end"]], span["surface"], row["id"])
                restored = restored[:span["normalized_start"]] + span["surface"] + restored[span["normalized_end"]:]
            self.assertEqual(restored, row["text"], row["id"])

    def test_dataset_runner_ignores_gold_labels_and_forms_and_preserves_columns(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        spec = importlib.util.spec_from_file_location("annotate_values", ROOT / "scripts/annotate_values.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "input.csv"
            with source.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", "text", "date", "label", "form"])
                writer.writeheader()
                for rid, label, form in [("a", "true", "[]"), ("b", "false", "deliberately not JSON")]:
                    writer.writerow(dict(id=rid, text="Hooly AI Pro costs €49.99 per month.",
                                         date="2026-01-01", label=label, form=form))
            original = source.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                module.run(source, ROOT / "data/lexicon.json", ROOT / "data/entity_registry.json", root / "out")
            with (root / "out/annotated.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["normalized_text"], rows[1]["normalized_text"])
            self.assertEqual(rows[0]["values"], rows[1]["values"])
            self.assertEqual(rows[1]["form"], "deliberately not JSON")
            self.assertEqual(source.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
