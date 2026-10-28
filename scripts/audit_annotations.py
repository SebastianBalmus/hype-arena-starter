"""Audit saved annotations; reference forms are diagnostics, not full NER gold."""

import argparse
from collections import Counter
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from entity_linking.values import exact_json
from entity_linking.io import read_rows, write_jsonl
from entity_linking.evaluation import signature


# Independent reference normalization, rather than calling the annotator's parser.


def run(dataset, annotations, output):
    if output.exists():
        raise FileExistsError("Use a fresh audit output directory")
    source, _ = read_rows(dataset)
    path = annotations / "annotations.jsonl"
    records = [json.loads(line, parse_float=Decimal) for line in path.read_text().splitlines()]
    if [row["id"] for row in source] != [row["id"] for row in records]:
        raise ValueError("Source and annotation rows do not align")
    missing, additional, unresolved, sample_pool, entity_samples = [], [], [], [], {}
    expected_count = covered_count = 0
    for row, result in zip(source, records):
        if row["text"] != result["original_text"] or row["text"] != result["text"]:
            raise AssertionError("Original text differs")
        spans = sorted(result["mentions"] + result["values"], key=lambda item: item["start"])
        for left, right in zip(spans, spans[1:]):
            if left["end"] > right["start"]:
                raise AssertionError("Overlapping entity/value spans")
        reconstructed = result["normalized_text"]
        for span in reversed(spans):
            token = span.get("token", f"<ENTITY:{span.get('entity_id')}>")
            if row["text"][span["start"]:span["end"]] != span["surface"]:
                raise AssertionError("Source offset mismatch")
            if reconstructed[span["normalized_start"]:span["normalized_end"]] != token:
                raise AssertionError("Normalized offset mismatch")
            reconstructed = reconstructed[:span["normalized_start"]] + span["surface"] + reconstructed[span["normalized_end"]:]
        if reconstructed != row["text"]:
            raise AssertionError("Round-trip did not preserve source")
        gold = [fact["value"] for fact in json.loads(row["form"], parse_float=Decimal)
                if fact["value"]["kind"] not in {"name", "boolean"}]
        reference = {signature(value): value for value in gold}
        predicted = {signature(value) for value in result["values"]}
        expected_count += len(reference)
        covered_count += len(set(reference) & predicted)
        for key in set(reference) - predicted:
            value = reference[key]
            embedded = value["kind"] == "date" and len(str(value["value"])) == 4 and any(
                str(value["value"]) in mention["surface"] for mention in result["mentions"])
            missing.append({"id": row["id"], "text": row["text"], "reference_value": value,
                            "status": "year_protected_inside_entity" if embedded else "needs_review"})
        for value in result["values"]:
            review = {"id": row["id"], "text": row["text"], "annotation": value}
            sample_pool.append(review)
            if signature(value) not in reference:
                additional.append(review)
        unresolved.extend({"id": row["id"], "text": row["text"], **value} for value in result["unresolved_values"])
        for mention in result["mentions"]:
            entity_samples.setdefault(mention["entity_id"], {"id": row["id"], "text": row["text"], "mention": mention})
    sample = []
    for kind in sorted({item["annotation"]["kind"] for item in sample_pool}):
        candidates = [item for item in sample_pool if item["annotation"]["kind"] == kind]
        sample.extend(sorted(candidates, key=lambda item: hashlib.sha256(
            f"{item['id']}:{item['annotation']['start']}:{kind}".encode()).hexdigest())[:8])
    summary = {"rows_checked": len(source), "all_source_and_offset_invariants_passed": True,
               "detected_value_mentions": sum(len(row["values"]) for row in records),
               "reference_unique_values": expected_count, "reference_unique_values_found": covered_count,
               "reference_values_missing": len(missing),
               "missing_by_status": dict(Counter(item["status"] for item in missing)),
               "additional_mentions_not_in_forms": len(additional),
               "unresolved_or_intentionally_untyped_mentions": len(unresolved),
               "stratified_review_samples": len(sample), "distinct_entity_review_samples": len(entity_samples),
               "limitations": "Reference forms describe assertions, not all literal mentions; no NER precision/recall or held-out generalization is claimed.",
               "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
               "annotations_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_jsonl(output / "reference_exceptions.jsonl", missing)
    write_jsonl(output / "additional_literals.jsonl", additional)
    write_jsonl(output / "unresolved.jsonl", unresolved)
    write_jsonl(output / "typed_review_sample.jsonl", sample)
    write_jsonl(output / "entity_review_sample.jsonl", list(entity_samples.values()))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/practice.csv")
    parser.add_argument("--annotations", type=Path, default=ROOT / "artifacts/typed_values")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/value_audit")
    args = parser.parse_args()
    run(args.input, args.annotations, args.output)
