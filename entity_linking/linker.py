"""Recognize declared names/aliases without rewriting other sentence content."""

from dataclasses import asdict, dataclass
import re


def _key(name):
    return re.sub(r"\s+", "", name.replace("’", "'")).casefold()


def build_registry(lexicon, previous=None):
    """Assign IDs once, preserving existing IDs when entities are added/reordered."""
    if previous is not None and previous.get("schema_version") != 1:
        raise ValueError("Unsupported registry version")
    old = (previous or {}).get("entities", [])
    ids = [item["id"] for item in old]
    names = [item["name"] for item in old]
    if any(type(value) is not int or value < 1 for value in ids):
        raise ValueError("Entity IDs must be positive integers")
    if len(set(ids)) != len(ids) or len(set(names)) != len(names):
        raise ValueError("Duplicate registry ID or canonical name")
    entities = {item["name"]: {**item, "active": False} for item in old}
    subjects = lexicon["subjects"]
    if len({subject["name"] for subject in subjects}) != len(subjects):
        raise ValueError("Duplicate canonical lexicon name")
    next_id = max(ids, default=0) + 1
    events = set(lexicon.get("events", []))
    for subject in sorted(subjects, key=lambda item: (item["name"].casefold(), item["name"])):
        name = subject["name"]
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Empty canonical name")
        entity_id = entities[name]["id"] if name in entities else next_id
        if name not in entities:
            next_id += 1
        aliases = list(subject.get("aliases", []))
        if any(not isinstance(alias, str) or not alias.strip() for alias in aliases):
            raise ValueError("Empty alias")
        entities[name] = {"id": entity_id, "name": name, "aliases": aliases,
                          "attributes": list(subject.get("attributes", [])),
                          "is_event": name in events, "active": True}
    return {"schema_version": 1, "entities": sorted(entities.values(), key=lambda item: item["id"])}


def _pattern(spelling):
    parts = ["".join("['’]" if char in "'’" else re.escape(char) for char in word)
             for word in spelling.split()]
    # Do not consume just the integer prefix of a version, e.g. Android 16
    # inside Android 16.0. A separately listed shorter platform may still link.
    right = r"(?!\w|\.\d)" if spelling[-1].isdigit() else r"(?!\w)"
    return re.compile(r"(?<!\w)" + r"\s*".join(parts) + right, re.IGNORECASE)


@dataclass(frozen=True)
class Mention:
    entity_id: int
    canonical_name: str
    surface: str
    start: int
    end: int
    normalized_start: int
    normalized_end: int
    lexicon_spellings: tuple[str, ...]


class EntityLinker:
    def __init__(self, registry):
        if registry.get("schema_version") != 1:
            raise ValueError("Unsupported registry version")
        self.entities = {item["id"]: item for item in registry["entities"] if item.get("active", True)}
        spellings = {}
        for entity in self.entities.values():
            for spelling in [entity["name"], *entity.get("aliases", [])]:
                key = _key(spelling)
                entry = spellings.setdefault(key, {"pattern": _pattern(spelling), "entities": set(), "spellings": set()})
                entry["entities"].add(entity["id"])
                entry["spellings"].add(spelling)
        self.patterns = list(spellings.values())

    def normalize(self, text):
        candidates = {}
        suppressed = []
        protected = [(match.start(), match.end()) for match in re.finditer(r"<ENTITY:\d+>", text)]
        for entry in self.patterns:
            for match in entry["pattern"].finditer(text):
                start, end = match.span()
                if any(start < right and left < end for left, right in protected):
                    continue
                # A lower-case plural following an explicit count is a noun,
                # not automatically a mention of the identically named product.
                if text[start:end].islower() and text[start:end].lower() in {
                        "photos", "docs", "sheets", "slides", "forms", "maps"} and re.search(
                        r"(?:\d[\d,.]*(?:\s*(?:million|billion|thousand|M|k))?|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|twenty|hundred))\s+$",
                        text[:start], re.I):
                    suppressed.append({"surface": text[start:end], "start": start, "end": end,
                                       "candidate_entity_ids": sorted(entry["entities"]),
                                       "reason": "lowercase_counted_noun_context"})
                    continue
                span = candidates.setdefault((start, end), {"entities": set(), "spellings": set()})
                span["entities"].update(entry["entities"])
                span["spellings"].update(entry["spellings"])
        # Prefer the longest actual span. Ties use the earliest original offset.
        selected = []
        for start, end in sorted(candidates, key=lambda span: (-(span[1] - span[0]), span[0], span[1])):
            if not any(start < right and left < end for left, right in selected):
                selected.append((start, end))
        pieces, mentions, ambiguous = [], [], []
        cursor, normalized_length = 0, 0
        for start, end in sorted(selected):
            prefix = text[cursor:start]
            pieces.append(prefix)
            normalized_length += len(prefix)
            candidate = candidates[(start, end)]
            if len(candidate["entities"]) > 1:
                pieces.append(text[start:end])
                ambiguous.append({"surface": text[start:end], "start": start, "end": end,
                                  "candidate_entity_ids": sorted(candidate["entities"]),
                                  "lexicon_spellings": sorted(candidate["spellings"])})
                normalized_length += end - start
            else:
                entity_id = next(iter(candidate["entities"]))
                token = f"<ENTITY:{entity_id}>"
                mentions.append(Mention(entity_id, self.entities[entity_id]["name"], text[start:end],
                                        start, end, normalized_length, normalized_length + len(token),
                                        tuple(sorted(candidate["spellings"]))))
                pieces.append(token)
                normalized_length += len(token)
            cursor = end
        pieces.append(text[cursor:])
        return {"original_text": text, "normalized_text": "".join(pieces),
                "mentions": [asdict(mention) for mention in mentions], "ambiguous_mentions": ambiguous,
                "suppressed_mentions": suppressed}
