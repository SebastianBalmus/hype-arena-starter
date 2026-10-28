"""Loading rows and the lexicon.

Practice rows carry gold `form` atoms; runtime rows (universe.csv) do not. Pages are not
given by the contract, so `page` is a hypothesis: a run of consecutive rows sharing a date.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field, replace

from .values import NameNormalizer

LABELS = ("true", "false", "unsure")


@dataclass(frozen=True)
class Atom:
    subject: str
    attribute: str
    value: dict

    @property
    def key(self) -> tuple[str, str]:
        return (self.subject, self.attribute)


@dataclass
class Row:
    id: str
    text: str
    date: str                       # page date, YYYY-MM-DD
    label: str | None               # "true" / "false" / "unsure", or None when unlabelled
    atoms: list[Atom] | None = None  # gold atoms (practice) or parsed atoms; None = not parsed
    index: int = 0                  # position in the input file
    page: int = 0                   # page hypothesis id
    meta: dict = field(default_factory=dict)

    def hidden(self) -> "Row":
        """The same row with its label removed (what a model sees for a test row)."""
        return replace(self, label=None)


def load_rows(path: str) -> list[Row]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for i, r in enumerate(reader):
            atoms = None
            if r.get("form"):
                atoms = [Atom(a["subject"], a["attribute"], a["value"]) for a in json.loads(r["form"])]
            label = (r.get("label") or "").strip().lower() or None
            rows.append(Row(r["id"], r["text"], r["date"], label, atoms, index=i))
    assign_pages(rows)
    return rows


def assign_pages(rows: list[Row]) -> None:
    """Consecutive rows (in file order) with the same date form one page hypothesis."""
    page, prev = -1, None
    for r in sorted(rows, key=lambda r: r.index):
        if r.date != prev:
            page, prev = page + 1, r.date
        r.page = page


def load_lexicon(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def name_values(rows: list[Row]) -> set[str]:
    return {a.value["value"] for r in rows if r.atoms for a in r.atoms if a.value["kind"] == "name"}


def build_names(lexicon: dict, rows: list[Row]) -> NameNormalizer:
    return NameNormalizer(lexicon, extra_names=name_values(rows))
