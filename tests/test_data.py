import csv
import json

from arena.data import load_practice, read_universe, write_output


def test_read_universe_pages(tmp_path):
    p = tmp_path / "u.csv"
    p.write_text("﻿id,text,date,label\na,x,2026-01-01,true\nb,y,2026-01-01,\nc,z,2026-02-01,\nd,w,2026-01-01,false\n", encoding="utf-8")
    rows = read_universe(str(p))
    assert [r.page for r in rows] == [0, 0, 1, 2]
    assert rows[1].label == "" and rows[0].label == "true"


def test_load_practice_gold():
    rows, gold = load_practice("data/practice.csv")
    assert len(rows) == 1897 and len(gold) == 1897
    assert gold["p-1"][0].subject == "Holdan" and gold["p-1"][0].attribute == "joined"
    assert gold["p-1"][0].value == {"kind": "date", "value": "2026-01-12"}


def test_write_output_roundtrip_quoting(tmp_path):
    labels = [("x-1", "true", 0.9), ("x-2", "unsure", None)]
    claims = [{"id": "c-1", "type": "fact", "text": 'Says "hi", then\nleaves.', "form": {"a": "b,c"}, "confidence": 0.8}]
    path = write_output(str(tmp_path), labels, claims)
    rows = list(csv.DictReader(open(path, encoding="utf-8", newline="")))
    assert rows[1]["confidence"] == "" and rows[2]["text"] == 'Says "hi", then\nleaves.'
    assert json.loads(rows[2]["form"]) == {"a": "b,c"} and rows[2]["label"] == "true"
    assert not (tmp_path / "output.tmp").exists()
