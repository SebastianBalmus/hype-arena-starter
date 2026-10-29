"""Hype wording: generic English boast markers (absolutes, superlatives) and first person.

Fixed in advance from general English, not picked from this dataset. In the practice data,
sentences with them are lies about twice as often (26% vs 14%), and our verdicts on them were
right 78% of the time while claiming 88%. So their confidence is shrunk towards 0.5 by a factor
learned on held-out pages (`hype_shrink` in assets/calibration.json). Labels are unchanged.
"""
import re

HYPE_WORDS = re.compile(r"\b(never|ever|always|every time|forever|best|worst|perfect|flawless|incredible|amazing|"
                        r"unbelievable|insane|huge|massive|record|fastest|biggest|largest|greatest|unmatched|"
                        r"unbeatable|no one|nobody|more than (?:any|all|anyone|everyone)|by far|easily|literally|"
                        r"totally|absolutely)\b|!", re.I)
FIRST_PERSON = re.compile(r"\b(I|me|my|I'm|I am)\b")


def hype_spans(text: str) -> list[tuple[int, int]]:
    return [m.span() for m in HYPE_WORDS.finditer(text)] + [m.span() for m in FIRST_PERSON.finditer(text)]


def is_hyped(text: str) -> bool:
    return bool(HYPE_WORDS.search(text) or FIRST_PERSON.search(text))
