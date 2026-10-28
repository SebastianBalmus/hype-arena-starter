import csv
import json
import os

from arena.model import Fact, Row

COLUMNS = ["id", "type", "text", "form", "label", "confidence"]


def _rows_from(records) -> list[Row]:
    rows, page, prev = [], -1, None
    for r in records:
        date = r["date"]
        if date != prev:
            page += 1
            prev = date
        rows.append(Row(r["id"], r["text"], date, (r.get("label") or "").strip().lower(), page))
    return rows


def read_universe(path: str) -> list[Row]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return _rows_from(csv.DictReader(f))


def load_practice(path: str) -> tuple[list[Row], dict[str, list[Fact]]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        records = list(csv.DictReader(f))
    gold = {
        r["id"]: [Fact(d["subject"], d["attribute"], d["value"]) for d in json.loads(r.get("form") or "[]")]
        for r in records
    }
    return _rows_from(records), gold


def write_output(out_dir: str, labels: list[tuple[str, str, float | None]], claims: list[dict]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    tmp, final = os.path.join(out_dir, "output.tmp"), os.path.join(out_dir, "output.csv")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for rid, label, conf in labels:
            w.writerow([rid, "", "", "", label, "" if conf is None else conf])
        for c in claims:
            w.writerow([c["id"], c["type"], c["text"], json.dumps(c["form"], ensure_ascii=False), "true", c["confidence"]])
    os.replace(tmp, final)
    return final
