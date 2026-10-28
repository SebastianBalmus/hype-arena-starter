"""Normalize declared entity aliases and write dataset inspection artifacts."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from entity_linking import EntityLinker, build_registry
from entity_linking.report import render_report


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if not {"id", "text", "date"} <= set(fields):
            raise ValueError("Dataset requires id, text and date")
        rows = list(reader)
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate input row IDs")
    return rows, fields


def write_rows(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(input_path, lexicon_path, registry_path, output):
    if output.exists():
        raise FileExistsError(f"Use a fresh output directory: {output}")
    lexicon_bytes, input_bytes = lexicon_path.read_bytes(), input_path.read_bytes()
    lexicon = json.loads(lexicon_bytes)
    previous = json.loads(registry_path.read_text()) if registry_path.exists() else None
    registry = build_registry(lexicon, previous)
    linker = EntityLinker(registry)
    rows, fields = read_rows(input_path)
    extra = ["normalized_text", "mentions", "ambiguous_mentions", "suppressed_mentions"]
    if set(fields) & set(extra):
        raise ValueError("Input already has normalization output columns")
    normalized, inspection = [], []
    mentions_per_entity, rows_per_entity = Counter(), Counter()
    normalized_groups = defaultdict(set)
    for row in rows:
        result = linker.normalize(row["text"])
        normalized.append({**row, "normalized_text": result["normalized_text"],
                           "mentions": json.dumps(result["mentions"], ensure_ascii=False),
                           "ambiguous_mentions": json.dumps(result["ambiguous_mentions"], ensure_ascii=False),
                           "suppressed_mentions": json.dumps(result["suppressed_mentions"], ensure_ascii=False)})
        inspection.append({"id": row["id"], "date": row["date"], "text": row["text"],
                           "normalized_text": result["normalized_text"], "mentions": result["mentions"],
                           "ambiguous_mentions": result["ambiguous_mentions"],
                           "suppressed_mentions": result["suppressed_mentions"]})
        mentions_per_entity.update(mention["entity_id"] for mention in result["mentions"])
        rows_per_entity.update({mention["entity_id"] for mention in result["mentions"]})
        normalized_groups[result["normalized_text"]].add(row["text"])
    linked_rows = sum(bool(row["mentions"]) for row in inspection)
    summary = {
        "method": "lexicon_only_entity_recognition_and_linking", "registry_schema_version": 1,
        "source_sha256": {"dataset": hashlib.sha256(input_bytes).hexdigest(),
                          "lexicon": hashlib.sha256(lexicon_bytes).hexdigest()},
        "rows": len(rows), "rows_with_links": linked_rows, "rows_without_links": len(rows) - linked_rows,
        "linked_row_coverage": linked_rows / len(rows) if rows else 0,
        "linked_mentions": sum(mentions_per_entity.values()), "entities_seen": len(mentions_per_entity),
        "active_entities": len(linker.entities),
        "ambiguous_mentions": sum(len(row["ambiguous_mentions"]) for row in inspection),
        "suppressed_mentions": sum(len(row["suppressed_mentions"]) for row in inspection),
        "rows_with_ambiguous_mentions": sum(bool(row["ambiguous_mentions"]) for row in inspection),
        "unique_original_texts": len({row["text"] for row in rows}),
        "unique_normalized_texts": len(normalized_groups),
        "normalized_groups_merging_distinct_texts": sum(len(texts) > 1 for texts in normalized_groups.values()),
        "uses_labels_or_forms_for_linking": False,
        "offset_convention": "zero-based Python Unicode character offsets; end exclusive",
    }
    output.mkdir(parents=True, exist_ok=False)
    registry_text = json.dumps(registry, ensure_ascii=False, indent=2) + "\n"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    if not registry_path.exists() or registry_path.read_text() != registry_text:
        registry_path.write_text(registry_text)
    (output / "entity_registry.json").write_text(registry_text)
    write_rows(output / "normalized.csv", normalized, fields + extra)
    write_rows(output / "entity_counts.csv", [
        {"entity_id": entity["id"], "canonical_name": entity["name"],
         "mentions": mentions_per_entity[entity["id"]], "rows": rows_per_entity[entity["id"]]}
        for entity in registry["entities"] if entity["active"]],
        ["entity_id", "canonical_name", "mentions", "rows"])
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "inspection.html").write_text(render_report(inspection, registry, summary))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/practice.csv")
    parser.add_argument("--lexicon", type=Path, default=ROOT / "data/lexicon.json")
    parser.add_argument("--registry", type=Path, default=ROOT / "data/entity_registry.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/entity_linking/v1")
    args = parser.parse_args()
    run(args.input, args.lexicon, args.registry, args.output)
