"""Assemble isolated practice paraphrases; never modify the original corpus.

Run from any directory with Python 3. No model, package, or network is needed.
The reviewed label-specific generators live in work/<label>/generate.py.
"""

from __future__ import annotations

import collections
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
FIELDS = ["id", "text", "date", "label", "form"]
TARGETS = {"true": 900, "false": 850, "unsure": 250}
PINNED = {
    "data/practice.csv": "52743befaa0fa38b89b5c802c0f953d242ae027c86839c9c0f05a9e69a86d8e9",
    "data/lexicon.json": "d0fd5e9bdc6b33ad3ec160001f50e00e7f8bf2b25a92779022f893db7c3d5507",
    "evaluation/splits-v1.json": "9699327af7bf80813eabb66a582593d5152683ee98abf36090a252b0b2723860",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_key(text):
    # Punctuation-only and case-only variants do not count as new examples.
    return re.sub(r"[\W_]+", " ", text.casefold()).strip()


def fingerprint(row):
    canonical = json.dumps({k: row[k] for k in FIELDS}, ensure_ascii=False,
                           sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def numeric_literals(text):
    return {Decimal(token.replace(",", ""))
            for token in re.findall(r"\d[\d,]*(?:\.\d+)?", text)}


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    for relative, expected in PINNED.items():
        require(sha(REPO / relative) == expected, f"Source changed: {relative}")
    with (REPO / "data/practice.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == FIELDS, "Unexpected source columns")
        original = list(reader)
    sources = {r["id"]: r for r in original}
    lexicon = json.loads((REPO / "data/lexicon.json").read_text())
    attributes = {s["name"]: set(s["attributes"]) for s in lexicon["subjects"]}
    seen = {text_key(r["text"]) for r in original}
    candidates = []
    for label, target in TARGETS.items():
        builder = ROOT / "work" / label / "generate.py"
        subprocess.run([sys.executable, str(builder)], cwd=REPO, check=True)
        batch = [json.loads(line) for line in builder.with_name("candidates.jsonl").read_text().splitlines()]
        require(len(batch) == target, f"Wrong {label} count: {len(batch)}")
        for row in batch:
            source = sources[row["source_id"]]
            require(row["label"] == source["label"] == label, f"Label drift: {row}")
            require(row["date"] == source["date"], f"Date drift: {row}")
            require(row["form"] == json.loads(source["form"]), f"Form drift: {row}")
            require(not (numeric_literals(row["text"]) - numeric_literals(source["text"] + " " + source["form"])),
                    f"Introduced numeric literal: {row}")
            require(isinstance(row["text"], str) and 0 < len(row["text"]) <= 300,
                    f"Bad text length: {row}")
            require(row["text"] == row["text"].strip() and "\n" not in row["text"],
                    f"Unexpected text whitespace: {row}")
            key = text_key(row["text"])
            require(key not in seen, f"Duplicate text ignoring punctuation/case: {row}")
            seen.add(key)
            for atom in row["form"]:
                require(atom["attribute"] in attributes.get(atom["subject"], set()),
                        f"Unknown subject/attribute: {row}")
            require(bool(row.get("transformation")), f"Missing transformation: {row}")
            candidates.append(row)

    random.Random(42).shuffle(candidates)
    additional = []
    provenance = []
    for position, candidate in enumerate(candidates, 1):
        source = sources[candidate["source_id"]]
        row = {"id": f"syn-{position:04d}", "text": candidate["text"],
               "date": source["date"], "label": source["label"], "form": source["form"]}
        require(row["id"] not in sources, "Synthetic ID collides with source")
        additional.append(row)
        provenance.append({
            "id": row["id"], "source_id": source["id"],
            "source_fingerprint_sha256": fingerprint(source),
            "source_text": source["text"], "source_date": source["date"],
            "source_label": source["label"], "transformation": candidate["transformation"],
            "source_form_preserved": True,
            "label_basis": "inherited_from_source_under_semantic_paraphrase",
            "independent_world_truth_verification": "not_performed",
            "inherited_text_form_mismatch": "inherited_text_form_mismatch" in candidate["transformation"],
            "synthetic": True, "world_evidence": False,
        })
    write_csv(ROOT / "practice_additional_2000.csv", additional)
    write_csv(ROOT / "practice_combined_3897.csv", original + additional)
    with (ROOT / "provenance.jsonl").open("w", encoding="utf-8") as stream:
        for record in provenance:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    subject_counts = collections.Counter(a["subject"] for c in candidates for a in c["form"])
    pair_counts = collections.Counter((a["subject"], a["attribute"]) for c in candidates for a in c["form"])
    report = {
        "purpose": "source-grounded synthetic practice paraphrases, not new world evidence",
        "seed": 42, "source_sha256": PINNED,
        "original_rows": len(original), "additional_rows": len(additional),
        "combined_rows": len(original) + len(additional),
        "original_label_counts": dict(collections.Counter(r["label"] for r in original)),
        "additional_label_counts": dict(collections.Counter(r["label"] for r in additional)),
        "combined_label_counts": dict(collections.Counter(r["label"] for r in original + additional)),
        "unique_source_rows": len({c["source_id"] for c in candidates}),
        "unique_sources_per_label": {label: len({c["source_id"] for c in candidates if c["label"] == label}) for label in TARGETS},
        "subjects_covered": len(subject_counts), "subject_attribute_pairs_covered": len(pair_counts),
        "subject_counts": dict(sorted(subject_counts.items())),
        "max_text_characters": max(len(r["text"]) for r in additional),
        "checks": {
            "exact_row_counts": "passed", "ids_unique_across_combined": "passed",
            "texts_novel_and_unique_ignoring_case_and_punctuation": "passed",
            "source_dates_labels_and_complete_forms_preserved": "passed",
            "no_new_numeric_literals_beyond_source_text_and_form": "passed",
            "subjects_and_attributes_in_lexicon": "passed",
            "original_inputs_and_frozen_splits_unchanged": "passed",
        },
        "truth_verification": {"status": "not_independently_verified",
                               "basis": "original annotations and semantic paraphrase review; no new oracle"},
        "evaluation": {"status": "not_run", "reason": "dataset authoring only; full-practice descendants are not held-out test data"},
        "builder_sha256": {str(p.relative_to(ROOT)): sha(p) for p in [ROOT / "build_dataset.py"] + sorted((ROOT / "work").glob("*/generate.py"))},
        "output_sha256": {name: sha(ROOT / name) for name in ["practice_additional_2000.csv", "practice_combined_3897.csv", "provenance.jsonl"]},
    }
    (ROOT / "validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: report[k] for k in ["additional_rows", "combined_rows", "additional_label_counts", "unique_source_rows", "subjects_covered", "max_text_characters"]}, indent=2))


if __name__ == "__main__":
    main()
