"""Lexicon of subjects, mention finding, attribute words and a name index."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass


@dataclass
class Mention:
    start: int
    end: int
    subject: str
    surface: str


def _norm(name: str) -> str:
    return re.sub(r"\s+", " ", name.replace("’", "'")).strip().lower()


def _squash(name: str) -> str:
    return re.sub(r"\s+", "", name.replace("’", "'")).lower()


def _name_pattern(name: str) -> str:
    """Regex for a name: any case, spaces optional, apostrophes either kind."""
    parts = []
    for tok in name.split():
        parts.append("".join("['’]" if ch in "'’" else re.escape(ch) for ch in tok))
    return r"\s*".join(parts)


def _compile(patterns: list[str]) -> re.Pattern[str]:
    return re.compile(r"(?<!\w)(?:" + "|".join(patterns) + r")(?!\w)", re.IGNORECASE)


class Lexicon:
    def __init__(self, subjects: dict[str, list[str]], aliases: dict[str, list[str]], events: list[str]):
        self.subjects = subjects
        self.events = events
        self._by_key: dict[str, str] = {}
        self._by_squash: dict[str, str] = {}
        spellings: list[tuple[str, str]] = [(s, s) for s in subjects]
        for subject, als in aliases.items():
            spellings.extend((a, subject) for a in als)
        originals: dict[str, str] = {}
        for spelling, subject in spellings:
            self._by_key.setdefault(_norm(spelling), subject)
            self._by_squash.setdefault(_squash(spelling), subject)
            originals.setdefault(_norm(spelling), spelling)
        ordered = sorted(originals, key=len, reverse=True)
        self._regex = _compile([_name_pattern(originals[k]) for k in ordered])

    @classmethod
    def load(cls, path: str) -> "Lexicon":
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        subjects = {s["name"]: list(s.get("attributes", [])) for s in data["subjects"]}
        aliases = {s["name"]: list(s.get("aliases", [])) for s in data["subjects"]}
        return cls(subjects, aliases, list(data.get("events", [])))

    def canonical(self, name: str) -> str | None:
        return self._by_key.get(_norm(name)) or self._by_squash.get(_squash(name))

    def attributes(self, subject: str) -> list[str]:
        return list(self.subjects.get(subject, []))

    def mentions(self, text: str) -> list[Mention]:
        out = []
        for m in self._regex.finditer(text):
            subject = self.canonical(m.group(0))
            if subject is not None:
                out.append(Mention(m.start(), m.end(), subject, m.group(0)))
        return out


def attribute_words(attribute: str) -> str:
    return attribute.replace("_", " ")


def has_attribute_words(text: str, attribute: str) -> bool:
    tokens: list[str] = []
    for word in attribute.split("_"):
        tokens.extend(re.findall(r"[A-Za-z]+|[0-9]+", word))
    if not tokens:
        return False
    pattern = r"(?<![A-Za-z0-9])" + r"[\s-]*".join(re.escape(t) for t in tokens) + r"(?![A-Za-z0-9])"
    return re.search(pattern, text, re.IGNORECASE) is not None


class NameIndex:
    def __init__(self, names: list[str]):
        self.names = names
        self._regex = _compile([_name_pattern(n) for n in sorted(names, key=len, reverse=True)]) if names else None
        self._by_key = {_norm(n): n for n in names}
        self._by_squash = {_squash(n): n for n in names}

    @classmethod
    def from_values(cls, names: list[str], lexicon: Lexicon) -> "NameIndex":
        seen: dict[str, str] = {}
        for n in names:
            n = lexicon.canonical(n) or n.strip()
            seen.setdefault(_norm(n), n)
        return cls(list(seen.values()))

    def canonical(self, name: str) -> str:
        hit = self._by_key.get(_norm(name)) or self._by_squash.get(_squash(name))
        if hit:
            return hit
        key = _norm(name)
        longer = [n for n in self.names if _norm(n).startswith(key + " ")]
        return longer[0] if len(longer) == 1 else name

    def find(self, text: str) -> list[tuple[int, int, str]]:
        if self._regex is None:
            return []
        return [(m.start(), m.end(), self.canonical(m.group(0))) for m in self._regex.finditer(text)]
