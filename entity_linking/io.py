"""Shared source-table and prediction serialization; no model dependencies."""

import csv

from entity_linking.json_utils import exact_json


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


def write_jsonl(path, records):
    with path.open("w") as handle:
        for record in records:
            handle.write(exact_json(record) + "\n")
