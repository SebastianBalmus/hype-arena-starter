"""Entity resolution: lexicon names and aliases -> stable entity IDs.

Ported from Sebastian Balmus's `entity_linking/linker.py` (team branch `sebi`), unchanged in
behaviour except where noted. Every declared name or alias is matched as whole words, in any
case and spacing; the longest actual span wins, so a name inside a longer name names that one
only ("Hooly AI Pro" is the plan, not the company), exactly as `contracts/claim-forms.md`
reads names. A spelling shared by two entities is left unresolved instead of guessed. A name
ending in a number does not swallow the prefix of a version ("Hooly Android 16.0").

`normalize()` also returns the sentence with each entity replaced by `<ENTITY:id>`, which the
demo shows as the system's first step.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

COUNTED_NOUNS = {"photos", "docs", "sheets", "slides", "forms", "maps"}
_COUNT_BEFORE = re.compile(r"(?:\d[\d,.]*(?:\s*(?:million|billion|thousand|M|k))?|\b(?:one|two|three|four|five|six|"
                           r"seven|eight|nine|ten|twenty|hundred))\s+$", re.I)


def _key(name):
    return re.sub(r"\s+", "", name.replace("’", "'")).casefold()


def build_registry(lexicon, previous=None):
    """Assign IDs once (alphabetical by canonical name), keeping existing IDs when the lexicon grows."""
    old = (previous or {}).get("entities", [])
    entities = {item["name"]: {**item, "active": False} for item in old}
    next_id = max((item["id"] for item in old), default=0) + 1
    events = set(lexicon.get("events", []))
    for subject in sorted(lexicon.get("subjects", []), key=lambda s: (s["name"].casefold(), s["name"])):
        name = subject["name"]
        if name in entities:
            entity_id = entities[name]["id"]
        else:
            entity_id, next_id = next_id, next_id + 1
        entities[name] = {"id": entity_id, "name": name, "aliases": list(subject.get("aliases", [])),
                          "attributes": list(subject.get("attributes", [])), "is_event": name in events, "active": True}
    return {"schema_version": 1, "entities": sorted(entities.values(), key=lambda item: item["id"])}


def _pattern(spelling):
    parts = ["".join("['’]" if ch in "'’" else re.escape(ch) for ch in word) for word in spelling.split()]
    right = r"(?!\w|\.\d)" if spelling[-1].isdigit() else r"(?!\w)"
    return re.compile(r"(?<!\w)" + r"\s*".join(parts) + right, re.IGNORECASE)


@dataclass(frozen=True)
class Mention:
    entity_id: int
    canonical_name: str
    surface: str
    start: int
    end: int


class EntityLinker:
    def __init__(self, registry, suppress_counted_nouns: bool = False):
        # suppress_counted_nouns: Sebi's rule that "4,900,000,000 photos" is a noun, not Hooly
        # Photos. Off here: on practice pages such words name the row's subject 18 times in 21.
        self.suppress = suppress_counted_nouns
        self.entities = {e["id"]: e for e in registry["entities"] if e.get("active", True)}
        spellings = {}
        for entity in self.entities.values():
            for spelling in [entity["name"], *entity.get("aliases", [])]:
                entry = spellings.setdefault(_key(spelling), {"pattern": _pattern(spelling), "entities": set()})
                entry["entities"].add(entity["id"])
        self.patterns = list(spellings.values())
        self._cache = {}

    def normalize(self, text):
        hit = self._cache.get(text)
        if hit is not None:
            return hit
        candidates = {}
        for entry in self.patterns:
            for m in entry["pattern"].finditer(text):
                start, end = m.span()
                surface = text[start:end]
                if self.suppress and surface.islower() and surface in COUNTED_NOUNS and _COUNT_BEFORE.search(text[:start]):
                    continue
                candidates.setdefault((start, end), set()).update(entry["entities"])
        selected = []
        for start, end in sorted(candidates, key=lambda s: (-(s[1] - s[0]), s[0], s[1])):
            if not any(start < r and l < end for l, r in selected):
                selected.append((start, end))
        pieces, mentions, ambiguous, cursor = [], [], [], 0
        for start, end in sorted(selected):
            pieces.append(text[cursor:start])
            ids = candidates[(start, end)]
            if len(ids) > 1:
                pieces.append(text[start:end])
                ambiguous.append({"surface": text[start:end], "start": start, "end": end, "candidate_entity_ids": sorted(ids)})
            else:
                eid = next(iter(ids))
                mentions.append(Mention(eid, self.entities[eid]["name"], text[start:end], start, end))
                pieces.append(f"<ENTITY:{eid}>")
            cursor = end
        pieces.append(text[cursor:])
        out = {"normalized_text": "".join(pieces), "mentions": mentions, "ambiguous_mentions": ambiguous}
        if len(self._cache) < 200000:
            self._cache[text] = out
        return out

    def mentions(self, text) -> list[tuple[int, int, str]]:
        return [(m.start, m.end, m.canonical_name) for m in self.normalize(text)["mentions"]]

    def as_json(self, text):
        out = self.normalize(text)
        return {**out, "mentions": [asdict(m) for m in out["mentions"]]}
