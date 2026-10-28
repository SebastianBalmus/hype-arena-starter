"""Inspect typed values alongside the existing lexicon entity links."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from entity_linking import EntityLinker, build_registry
from entity_linking.report import render_report
from entity_linking.values import annotate, exact_json
from entity_linking.io import read_rows, write_rows


def run(input_path, lexicon_path, registry_path, output):
    if output.exists():
        raise FileExistsError(f"Use a fresh output directory: {output}")
    original = {"dataset": input_path.read_bytes(), "lexicon": lexicon_path.read_bytes()}
    registry = build_registry(json.loads(original["lexicon"]), json.loads(registry_path.read_text()))
    linker = EntityLinker(registry)
    rows, fields = read_rows(input_path)
    extra = ["entity_normalized_text", "normalized_text", "mentions", "ambiguous_mentions", "suppressed_mentions", "values", "unresolved_values"]
    if set(fields) & set(extra):
        raise ValueError("Input already contains annotation columns")
    records, annotated, unresolved, counts, examples = [], [], [], Counter(), {}
    for row in rows:
        result = annotate(row["text"], linker)
        record = {"id": row["id"], "date": row["date"], "text": row["text"], **result}
        records.append(record)
        annotated.append({**row, **{key: result[key] if key.endswith("text") else exact_json(result[key]) for key in extra}})
        for value in result["values"]:
            counts[value["kind"]] += 1
            bucket = examples.setdefault(value["kind"], [])
            if len(bucket) < 8:
                bucket.append({"id": row["id"], "text": row["text"], "tagged_text": result["normalized_text"], "annotation": value})
        for value in result["unresolved_values"]:
            unresolved.append({"id": row["id"], "text": row["text"], **value})
    linked = sum(bool(row["mentions"]) for row in records)
    summary = {
        "method": "lexicon_entity_links_plus_rule_based_typed_literals", "annotation_version": 2,
        "source_sha256": {key: hashlib.sha256(value).hexdigest() for key, value in original.items()},
        "code_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in [
            ROOT / "entity_linking/values.py", ROOT / "entity_linking/linker.py", ROOT / "entity_linking/report.py", Path(__file__)]},
        "rows": len(rows), "rows_with_links": linked, "rows_without_links": len(rows) - linked,
        "linked_mentions": sum(len(row["mentions"]) for row in records),
        "suppressed_entity_mentions": sum(len(row["suppressed_mentions"]) for row in records),
        "entities_seen": len({mention["entity_id"] for row in records for mention in row["mentions"]}),
        "value_mentions": sum(counts.values()), "values_by_kind": dict(sorted(counts.items())),
        "rows_with_values": sum(bool(row["values"]) for row in records),
        "rows_with_any_annotation": sum(bool(row["values"] or row["mentions"]) for row in records),
        "rows_without_any_annotation": sum(not row["values"] and not row["mentions"] for row in records),
        "approximate_values": sum(value["approximate"] for row in records for value in row["values"]),
        "values_with_negation_prefix": sum(value["negated_prefix"] for row in records for value in row["values"]),
        "comparison_values": sum(value["op"] != "eq" for row in records for value in row["values"]),
        "unresolved_values": len(unresolved),
        "unresolved_by_reason": dict(Counter(value["reason"] for value in unresolved)),
        "unique_entity_normalized_texts": len({row["entity_normalized_text"] for row in records}),
        "unique_fully_tagged_texts": len({row["normalized_text"] for row in records}),
        "uses_labels_or_forms_for_annotation": False,
        "value_ids": "local to row and kind; entity IDs remain global",
        "scope": "literal mentions only; no subject/attribute attribution or truth judgment",
    }
    output.mkdir(parents=True, exist_ok=False)
    write_rows(output / "annotated.csv", annotated, fields + extra)
    write_rows(output / "unresolved.csv", unresolved, ["id", "text", "start", "end", "surface", "reason"])
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "examples.json").write_text(exact_json(examples) + "\n")
    (output / "entity_registry.json").write_text(json.dumps(registry, indent=2) + "\n")
    with (output / "annotations.jsonl").open("w") as handle:
        for row in records:
            handle.write(exact_json(row) + "\n")
    (output / "inspection.html").write_text(render_report(records, registry, summary))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/practice.csv")
    parser.add_argument("--lexicon", type=Path, default=ROOT / "data/lexicon.json")
    parser.add_argument("--registry", type=Path, default=ROOT / "data/entity_registry.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/typed_values")
    args = parser.parse_args()
    run(args.input, args.lexicon, args.registry, args.output)
