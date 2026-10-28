import csv
import importlib.util
import json
from pathlib import Path
import sys
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from entity_linking import EntityLinker, build_registry


def lexicon(*subjects):
    return {"subjects": [{"name": name, "aliases": aliases, "attributes": ["price"]}
                          for name, aliases in subjects], "events": []}


class EntityLinkingTests(unittest.TestCase):
    def setUp(self):
        self.lexicon = lexicon(("Hooly", ["Hooly Inc."]),
                               ("Hooly AI Pro", ["AI Pro"]),
                               ("Hooly Cloud Next 2026", ["Hooly Cloud Next", "Cloud Next"]))
        self.registry = build_registry(self.lexicon)
        self.linker = EntityLinker(self.registry)
        self.ids = {entity["name"]: entity["id"] for entity in self.registry["entities"]}

    def test_all_event_aliases_share_one_id_and_preserve_external_values(self):
        token = f"<ENTITY:{self.ids['Hooly Cloud Next 2026']}>"
        for name in ["Hooly Cloud Next 2026", "Hooly Cloud Next", "Cloud Next"]:
            with self.subTest(name=name):
                original = name + " has 64 sessions on 12 June 2026, not 40."
                result = self.linker.normalize(original)
                self.assertEqual(result["normalized_text"], token + " has 64 sessions on 12 June 2026, not 40.")
                self.assertEqual(result["original_text"], original)
                self.assertEqual(len(result["mentions"]), 1)

    def test_longest_name_does_not_also_link_company(self):
        result = self.linker.normalize("Hooly AI Pro costs €49/month.")
        self.assertEqual([mention["canonical_name"] for mention in result["mentions"]], ["Hooly AI Pro"])
        self.assertEqual(result["normalized_text"], f"<ENTITY:{self.ids['Hooly AI Pro']}> costs €49/month.")

    def test_case_spacing_and_word_boundaries(self):
        for text in ["HOOLY AI PRO", "HoolyAIPro", "AI  Pro"]:
            self.assertEqual(self.linker.normalize(text)["normalized_text"], f"<ENTITY:{self.ids['Hooly AI Pro']}>")
        self.assertFalse(self.linker.normalize("SuperHooly and Hoolyish and Cloud Nextgen")["mentions"])

    def test_offsets_roundtrip_with_unicode_and_repeated_mentions(self):
        text = "🚀 Hooly AI Pro’s price is €49; Hooly Inc. has 100 customers."
        result = self.linker.normalize(text)
        restored = result["normalized_text"]
        for mention in reversed(result["mentions"]):
            self.assertEqual(text[mention["start"]:mention["end"]], mention["surface"])
            self.assertEqual(restored[mention["normalized_start"]:mention["normalized_end"]],
                             f"<ENTITY:{mention['entity_id']}>")
            restored = restored[:mention["normalized_start"]] + mention["surface"] + restored[mention["normalized_end"]:]
        self.assertEqual(restored, text)

    def test_apostrophe_alias_and_possessive_suffix(self):
        linker = EntityLinker(build_registry(lexicon(("Hooly Board", ["Hooly's board"]))))
        result = linker.normalize("Hooly’s board has 5 members.")
        self.assertEqual(result["normalized_text"], "<ENTITY:1> has 5 members.")
        self.assertEqual(self.linker.normalize("AI Pro's price")["normalized_text"],
                         f"<ENTITY:{self.ids['Hooly AI Pro']}>'s price")

    def test_ambiguous_alias_remains_unchanged_and_consumes_long_span(self):
        data = lexicon(("Alpha", ["Hooly AI Pro"]), ("Beta", ["Hooly AI Pro"]), ("Hooly", []))
        result = EntityLinker(build_registry(data)).normalize("Hooly AI Pro costs €49.")
        self.assertEqual(result["normalized_text"], "Hooly AI Pro costs €49.")
        self.assertEqual(result["mentions"], [])
        self.assertEqual(len(result["ambiguous_mentions"][0]["candidate_entity_ids"]), 2)

    def test_registry_is_stable_under_reordering_and_new_entities(self):
        previous = build_registry(lexicon(("Zebra", [])))
        current = build_registry(lexicon(("Aardvark", []), ("Zebra", ["Z"])), previous)
        ids = {entity["name"]: entity["id"] for entity in current["entities"]}
        self.assertEqual(ids, {"Zebra": 1, "Aardvark": 2})
        self.assertEqual(build_registry({**self.lexicon, "subjects": self.lexicon["subjects"][::-1]}), self.registry)

    def test_removed_entity_ids_are_never_reused(self):
        previous = build_registry(lexicon(("Alpha", [])))
        current = build_registry(lexicon(("Beta", [])), previous)
        self.assertFalse(current["entities"][0]["active"])
        self.assertEqual(current["entities"][1]["id"], 2)
        self.assertEqual(EntityLinker(current).normalize("Alpha and Beta")["normalized_text"], "Alpha and <ENTITY:2>")

    def test_duplicate_registry_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            build_registry(self.lexicon, {"schema_version": 1, "entities": [
                {"id": 1, "name": "Alpha"}, {"id": 1, "name": "Beta"}]})

    def test_pronouns_unknown_names_and_existing_tokens_stay_unchanged(self):
        text = "He met Gavin Bellson; it costs €49. <ENTITY:1>"
        self.assertEqual(self.linker.normalize(text)["normalized_text"], text)
        first = self.linker.normalize("Hooly and AI Pro")["normalized_text"]
        self.assertEqual(self.linker.normalize(first)["normalized_text"], first)

    def test_version_number_is_not_partially_consumed_by_numbered_entity(self):
        linker = EntityLinker(build_registry(lexicon(("Hooly Android", []), ("Hooly Android 16", []))))
        result = linker.normalize("Hooly Android 16.0 remains stable.")
        self.assertEqual(result["normalized_text"], "<ENTITY:1> 16.0 remains stable.")


class DatasetTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node is optional for inspection-script syntax checking")
    def test_inspection_javascript_has_valid_syntax(self):
        import re
        from entity_linking.report import render_report
        html = render_report([], {"entities": []}, {})
        scripts = re.findall(r'<script(?: [^>]*)?>(.*?)</script>', html, re.S)
        with tempfile.TemporaryDirectory() as folder:
            script = Path(folder) / "inspection.js"
            script.write_text(scripts[1])
            result = subprocess.run([shutil.which("node"), "--check", str(script)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_dataset_output_preserves_sources_and_roundtrips_all_spans(self):
        registry = build_registry(json.loads((ROOT / "data/lexicon.json").read_text()))
        linker = EntityLinker(registry)
        with (ROOT / "data/practice.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            result = linker.normalize(row["text"])
            restored = result["normalized_text"]
            for mention in reversed(result["mentions"]):
                self.assertEqual(row["text"][mention["start"]:mention["end"]], mention["surface"])
                self.assertEqual(restored[mention["normalized_start"]:mention["normalized_end"]],
                                 f"<ENTITY:{mention['entity_id']}>")
                restored = restored[:mention["normalized_start"]] + mention["surface"] + restored[mention["normalized_end"]:]
            self.assertEqual(restored, row["text"])

    def test_cli_preserves_metadata_and_emits_safe_self_contained_report(self):
        spec = importlib.util.spec_from_file_location("normalize_entities", ROOT / "scripts/normalize_entities.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            lex = root / "lexicon.json"
            source = root / "data.csv"
            lex.write_text(json.dumps(lexicon(("Hooly", []))))
            with source.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["id", "text", "date", "label", "form"])
                writer.writerow(["a", "Hooly </script><script>alert(1)</script>", "2026-01-01", "false", "[]"])
            before = source.read_bytes()
            module.run(source, lex, root / "registry.json", root / "output")
            with (root / "output/normalized.csv").open(newline="") as handle:
                result = list(csv.DictReader(handle))[0]
            self.assertEqual(result["label"], "false")
            self.assertEqual(result["form"], "[]")
            self.assertEqual(source.read_bytes(), before)
            report = (root / "output/inspection.html").read_text()
            self.assertNotIn("</script><script>alert", report)
            self.assertIn("\\u003c/script", report)
            with self.assertRaises(FileExistsError):
                module.run(source, lex, root / "registry.json", root / "output")


if __name__ == "__main__":
    unittest.main()
