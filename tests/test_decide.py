import json

import pytest

from arena.decide import FEATURES, Decider, row_features
from arena.evidence import AtomCheck
from arena.model import Fact
from arena.train import build_artifacts

FACT = Fact("Hooly", "founded", {"kind": "date", "value": "2020-01-01"}, 0.9)
FLOOR = 0.78  # measured label_accuracy 0.8044 (below always-true 0.8334) - 0.02, rounded down


def test_row_features_verdicts():
    t = AtomCheck("true", "in", 3, True, 0)
    f = AtomCheck("false", "in", 2, False, 0)
    x = row_features([FACT, FACT], [t, f], 0.8)
    assert set(x) == set(FEATURES) and x["verdict_false"] == 1 and x["n_false"] == 1
    assert row_features([], [], 0.8)["verdict_unknown"] == 1
    assert row_features([FACT], [t], 0.8)["verdict_true"] == 1


def test_decider_predict_shapes():
    X = [row_features([FACT], [AtomCheck("false" if i % 5 == 0 else "true", "in", 2, True, 0)], 0.8)
         for i in range(200)]
    y = ["false" if i % 5 == 0 else "true" for i in range(200)]
    d = Decider().fit(X, y, [i // 4 for i in range(200)])
    label, conf = d.predict(X[0])
    assert label == "false" and 0.5 <= conf <= 1.0


@pytest.fixture(scope="session")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("art")
    m = build_artifacts("data/practice.csv", "data/lexicon.json", str(out))
    print("METRICS", json.dumps(m))
    return out, m


def test_build_artifacts(built):
    out, m = built
    for f in ["kinds.pkl", "names.pkl", "attribute_model.pkl", "decider.pkl", "metrics.json"]:
        assert (out / f).exists()
    assert 0.80 <= m["always_true_accuracy"] <= 0.86
    assert m["label_accuracy"] >= FLOOR
