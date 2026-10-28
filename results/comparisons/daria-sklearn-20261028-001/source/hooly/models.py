"""The interface every row-labelling model implements (baselines and paper variants alike)."""
from __future__ import annotations

from dataclasses import dataclass, field

from .data import LABELS, Row
from .values import NameNormalizer


@dataclass
class Prediction:
    label: str                 # "true" / "false" / "unsure"
    probs: dict[str, float]    # probability of each label; sums to 1
    trace: dict = field(default_factory=dict)  # evidence for the demo and for error analysis

    @property
    def confidence(self) -> float | None:
        """What goes in the output's confidence column: 0.5-1, empty for unsure."""
        if self.label == "unsure":
            return None
        return min(1.0, max(0.5, self.probs.get(self.label, 0.5)))


class Model:
    """fit() sees every row: labelled rows are training evidence, rows with label=None are to
    be predicted. Atoms may be gold (oracle mode) or None (the model must parse the text)."""

    name = "model"

    def fit(self, rows: list[Row], lexicon: dict, names: NameNormalizer) -> "Model":
        raise NotImplementedError

    def predict(self, row: Row) -> Prediction:
        raise NotImplementedError


def smoothed(counts: dict[str, float], alpha: float = 1.0) -> dict[str, float]:
    total = sum(counts.get(l, 0) for l in LABELS) + alpha * len(LABELS)
    return {l: (counts.get(l, 0) + alpha) / total for l in LABELS}
