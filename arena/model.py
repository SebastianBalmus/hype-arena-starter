from dataclasses import dataclass

AS_OF = "2026-10-29"


@dataclass(frozen=True)
class Row:
    id: str
    text: str
    date: str
    label: str
    page: int


@dataclass
class Fact:
    subject: str
    attribute: str
    value: dict
    confidence: float = 1.0
