"""Page-grouped folds.

Random row splits leak: rows of one page share subjects, context and lies, and some
sentences repeat verbatim across pages. So whole pages move together, and pages that share
an exact sentence are merged into one group first.
"""
from __future__ import annotations

import random
from collections import defaultdict

from hooly.data import Row


def _norm_text(t: str) -> str:
    return " ".join(t.lower().split()).rstrip(".!? ")


def page_groups(rows: list[Row]) -> list[list[Row]]:
    parent: dict[int, int] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    first_page_of_text: dict[str, int] = {}
    for r in rows:
        find(r.page)
        t = _norm_text(r.text)
        if t in first_page_of_text:
            union(r.page, first_page_of_text[t])
        else:
            first_page_of_text[t] = r.page
    groups = defaultdict(list)
    for r in rows:
        groups[find(r.page)].append(r)
    return list(groups.values())


def grouped_folds(rows: list[Row], k: int = 5, seed: int = 0) -> list[set[str]]:
    """Return k sets of test row ids. Groups go to the currently smallest fold."""
    groups = page_groups([r for r in rows if r.label])
    random.Random(seed).shuffle(groups)
    groups.sort(key=len, reverse=True)  # stable: big groups first, random order among equals
    folds: list[set[str]] = [set() for _ in range(k)]
    for g in groups:
        smallest = min(range(k), key=lambda i: len(folds[i]))
        folds[smallest].update(r.id for r in g)
    return folds
