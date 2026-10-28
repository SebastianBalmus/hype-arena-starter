"""Per-attribute value kind, unit and boolean cue index learned from gold facts."""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from arena.model import Fact, Row


def unit_of(value: dict) -> str | None:
    kind = value.get("kind")
    if kind in ("quantity", "duration"):
        return value.get("unit")
    if kind == "money":
        cur = value.get("currency")
        return f"{cur}/{value['per']}" if value.get("per") else cur
    return None


def bigrams(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9']+", text.lower().replace("’", "'"))
    return {f"{a} {b}" for a, b in zip(words, words[1:])}


class KindIndex:
    def __init__(self, kinds: dict[str, str], units: dict[str, str], cues: dict[str, list[str]]):
        self._kinds, self._units, self._cues = kinds, units, cues

    @classmethod
    def from_gold(cls, gold: dict[str, list[Fact]], rows: list[Row] | None = None) -> "KindIndex":
        """Kinds and units come from ``gold``; boolean cues additionally need the source ``rows``."""
        kinds: dict[str, Counter] = defaultdict(Counter)
        units: dict[str, Counter] = defaultdict(Counter)
        for facts in gold.values():
            for f in facts:
                kinds[f.attribute][f.value.get("kind")] += 1
                u = unit_of(f.value)
                if u:
                    units[f.attribute][u] += 1
        kind_map = {a: c.most_common(1)[0][0] for a, c in kinds.items()}
        unit_map = {a: c.most_common(1)[0][0] for a, c in units.items()}
        cues: dict[str, list[str]] = {}
        if rows:
            by_id = {r.id: r for r in rows}
            grams = {rid: bigrams(by_id[rid].text) for rid in gold if rid in by_id}
            df: Counter = Counter()
            for g in grams.values():
                df.update(g)
            for attr, kind in kind_map.items():
                if kind != "boolean":
                    continue
                pos = [grams[rid] for rid, fs in gold.items() if rid in grams and any(f.attribute == attr for f in fs)]
                if not pos:
                    continue
                cnt: Counter = Counter()
                for g in pos:
                    cnt.update(g)
                scored = []
                for bg, c in cnt.items():
                    if c < 2 and len(pos) > 1:
                        continue
                    precision = c / df[bg]
                    if precision < 0.6:
                        continue
                    scored.append((precision * math.log(1 + c), bg))
                scored.sort(key=lambda t: (-t[0], t[1]))
                cues[attr] = [bg for _, bg in scored[:5]]
        return cls(kind_map, unit_map, cues)

    def kind(self, attribute: str) -> str | None:
        return self._kinds.get(attribute)

    def unit(self, attribute: str) -> str | None:
        return self._units.get(attribute)

    def cues(self, attribute: str) -> list[str]:
        return list(self._cues.get(attribute, []))
