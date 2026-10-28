"""Reusable sklearn candidates for Daria's Hooly HybridStack.

The snapshot's reader, parser, temporal shape model and HMM remain authoritative.  This
module only replaces the final binary stacker with a three-label estimator and exposes
its row features for fold-local model comparisons.
"""
from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier,
                              HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import PolynomialFeatures, RobustScaler, StandardScaler, QuantileTransformer
from sklearn.svm import LinearSVC, SVC

LABELS = ("true", "false", "unsure")


def _logit(p: float, eps: float = 1e-5) -> float:
    p = min(1.0 - eps, max(eps, float(p)))
    return float(np.log(p / (1.0 - p)))


def _feature_hybrid_class(default_config=None, default_seed=42):
    from hooly.hmm import HybridStack

    class FeatureHybrid(HybridStack):
        """Feature cache wrapper around snapshot ``HybridStack``.

        Instantiate this against the selected upstream snapshot (its ``hooly`` package must
        already be on ``sys.path``). ``feature_train`` and ``target_train`` align in order;
        ``feature_row`` emits the same schema for inference. Features use only the fitted
        split's rows and permitted unlabelled batch, never IDs or raw date strings.
        """

        def __init__(self, config=default_config, seed: int = default_seed):
            self.config = config
            self.seed = seed

        def fit(self, rows, lexicon, names):
            from dataclasses import replace
            from hooly.folds import grouped_folds
            from hooly.hmm import TimelineHMM
            from hooly.learn import LogReg
            from hooly.temporal import TemporalWindowsShape
            self.shape = TemporalWindowsShape().fit(rows, lexicon, names)
            labelled = [r for r in rows if r.label]
            self.training_ids = [r.id for r in labelled]
            # Rebuild grouped out-of-fold HMM features; a training row's own label is never
            # present in the HMM used to score that row. The snapshot uses page/text groups.
            oof = {}
            for ids in grouped_folds(labelled, k=5, seed=1):
                view = [replace(r, label=None) if r.id in ids else r for r in rows]
                h = TimelineHMM().fit(view, lexicon, names)
                for r in view:
                    if r.id in ids:
                        oof[r.id] = h.predict(r).probs
            # Extremely small folds can leave a row uncovered. The native stacker requires
            # OOF features too, so use a fully-hidden HMM for such rows (no label leakage).
            missing = [r for r in labelled if r.id not in oof]
            if missing:
                hidden = [replace(r, label=None) for r in rows]
                h = TimelineHMM().fit(hidden, lexicon, names)
                oof.update({r.id: h.predict(r).probs for r in missing})
            self.hmm = TimelineHMM().fit(rows, lexicon, names)
            self.native_train = []
            self.feature_train = []
            for r in labelled:
                base, _ = self.shape.row_probs(r, exclude_page=r.page)
                self.native_train.append(self._features(base, oof[r.id]))
                self.feature_train.append(self._make_features(r, oof[r.id], training=True))
            self.target_train = [r.label for r in labelled]
            self.stacker = LogReg(l2=1e-3, epochs=60).fit(
                self.native_train, [int(label == "false") for label in self.target_train])
            self.candidate = None
            if self.config is not None:
                self.candidate = make_candidate(self.config, self.seed)
                self.candidate.fit(select_features(self.feature_train, self.config["features"]), self.target_train)
            return self

        def _make_features(self, row, hp, training=False):
            base, atoms = self.shape.row_probs(row, exclude_page=row.page if training else None)
            a = _logit(base["false"] / max(1e-9, base["false"] + base["true"]))
            b = _logit(hp["false"] / max(1e-9, hp["false"] + hp["true"]))
            out = {"__text__": row.text, "native__windows": a, "native__hmm": b, "native__agree": a * b / 4,
                   "expanded__window_p_true": base["true"], "expanded__window_p_false": base["false"],
                   "expanded__window_p_unsure": base["unsure"], "expanded__hmm_p_true": hp["true"],
                   "expanded__hmm_p_false": hp["false"], "expanded__hmm_p_unsure": hp["unsure"],
                   "expanded__n_atoms": len(row.atoms or []), "expanded__parsed": float(bool(row.atoms)),
                   "expanded__year": int(row.date[:4]), "expanded__month": int(row.date[5:7]),
                   "expanded__day": int(row.date[8:10])}
            qs = row.meta.get("atom_q") or {}
            out["expanded__mean_parse_q"] = float(np.mean(list(qs.values()))) if qs else 1.0
            # State and shape labels are categorical dictionary features. Avoid row/page IDs.
            for i, atom in enumerate(atoms):
                for key in ("state", "stability", "direction", "trend", "shape"):
                    if atom.get(key) is not None:
                        out[f"expanded__atom{i}_{key}={atom[key]}"] = 1.0
                if "p_true" in atom:
                    out[f"expanded__atom{i}_p_true"] = atom["p_true"]
            return out

        def feature_row(self, row):
            hp = self.hmm.predict(row).probs
            return self._make_features(row, hp, training=False)

        def predict(self, row):
            if self.candidate is None:
                return super().predict(row)
            from hooly.models import Prediction
            feats = self.feature_row(row)
            dist = self.candidate.predict_proba(select_features([feats], self.config["features"]))[0]
            probs = {label: 0.0 for label in LABELS}
            for label, value in zip(self.candidate.classes_, dist):
                probs[str(label)] = float(value)
            label = max(LABELS, key=lambda key: probs[key])
            return Prediction(label, probs, {"rule": "sklearn_hybrid", "candidate": self.config["id"]})
    return FeatureHybrid


def build_truth_class(config=None, seed: int = 42):
    """Return the feature-cache HybridStack factory, matching the requested API."""
    return _feature_hybrid_class(config, seed)


def _model(name: str, seed: int):
    balanced = "balanced" in name
    base_name = name.replace("_balanced", "")
    if name.startswith("lr"):
        return LogisticRegression(C=0.1 if "strong" in name else 1.0, max_iter=500,
                                  class_weight="balanced" if balanced else None,
                                  solver="lbfgs", random_state=seed)
    if name == "rbf":
        return SVC(C=1.0, kernel="rbf", probability=True, class_weight=None, random_state=seed)
    if name == "rbf_balanced":
        return SVC(C=1.0, kernel="rbf", probability=True, class_weight="balanced", random_state=seed)
    if name == "histgb":
        return HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, l2_regularization=1.0, random_state=seed)
    if name == "rf":
        return RandomForestClassifier(n_estimators=140, min_samples_leaf=2, max_features="sqrt", class_weight="balanced" if balanced else None, n_jobs=2, random_state=seed)
    if name == "extratrees":
        return ExtraTreesClassifier(n_estimators=160, min_samples_leaf=2, max_features="sqrt", class_weight="balanced" if balanced else None, n_jobs=2, random_state=seed)
    if name == "gb":
        return GradientBoostingClassifier(n_estimators=80, max_depth=2, learning_rate=.05, random_state=seed)
    if name == "ada":
        return AdaBoostClassifier(n_estimators=80, learning_rate=.05, random_state=seed)
    if base_name == "linsvc":
        return CalibratedClassifierCV(LinearSVC(C=.5, class_weight="balanced" if balanced else None, random_state=seed), cv=3, method="sigmoid")
    if name == "ridge":
        return RidgeClassifier(class_weight="balanced" if balanced else None)
    if base_name == "sgd":
        return SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=500, class_weight="balanced" if balanced else None, random_state=seed)
    if name == "gnb":
        return GaussianNB()
    if name == "quantile_lr":
        return LogisticRegression(C=1.0, max_iter=500, solver="lbfgs", random_state=seed)
    if name == "quantile_knn":
        return KNeighborsClassifier(n_neighbors=11, weights="distance")
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=11, weights="distance")
    if name == "mlp":
        return MLPClassifier(hidden_layer_sizes=(32,), alpha=.01, max_iter=200, early_stopping=False, random_state=seed)
    raise KeyError(name)


CONFIGS = tuple(
    {"id": f"{features}_{prep}_{model}", "features": features, "preprocess": prep, "model": model}
    for features in ("native", "expanded")
    for prep in ("standard", "robust")
    for model in ("lr", "lr_balanced", "lr_strong", "linsvc", "ridge", "sgd", "knn", "mlp")
) + tuple({"id": f"expanded_{prep}_{model}", "features": "expanded", "preprocess": prep, "model": model}
          for prep in ("none",) for model in ("histgb", "rf", "extratrees", "gb", "ada", "gnb")) + tuple(
    {"id": f"expanded_{analyzer}_{model}", "features": f"expanded_{analyzer}",
     "preprocess": "standard", "model": model}
    for analyzer in ("word", "char") for model in ("lr", "lr_balanced", "linsvc")) + tuple(
    {"id": f"{features}_quantile_{model}", "features": features,
     "preprocess": "quantile", "model": model}
    for features in ("native", "expanded") for model in ("quantile_lr", "quantile_knn")) + tuple(
    {"id": f"expanded_{model}", "features": "expanded", "preprocess": "standard", "model": model}
    for model in ("rbf", "rbf_balanced")) + tuple(
    {"id": f"expanded_word_num_{model}", "features": f"expanded_word_num",
     "preprocess": "standard", "model": model}
    for model in ("lr", "lr_balanced")) + tuple(
    {"id": f"{features}_standard_lr_interactions", "features": features,
     "preprocess": "standard", "model": "lr", "interactions": True}
    for features in ("native", "expanded"))


def make_candidate(config: dict, seed: int = 42):
    """Create a fold-local sklearn pipeline accepting feature dictionaries."""
    cfg = dict(config)
    model = _model(cfg["model"], seed)
    # RidgeClassifier has no native probabilities: calibrated scores are learned solely
    # from the candidate's training fold. Avoid fold calibration for tiny datasets.
    if cfg["model"] == "ridge":
        model = CalibratedClassifierCV(model, cv=3, method="sigmoid")
    text_mode = cfg["features"].endswith(("_word", "_char", "_word_num"))
    if text_mode:
        analyzer = "char" if cfg["features"].endswith("_char") else "word"
        structured = Pipeline([("select", FunctionTransformer(
            lambda xs: [{k: v for k, v in x.items() if k.startswith(("native__", "expanded__"))} for x in xs],
            validate=False)), ("dict", DictVectorizer(sparse=True, sort=True)), ("scale", StandardScaler(with_mean=False))])
        def get_text(xs):
            import re
            texts = [x.get("__text__", "") for x in xs]
            if cfg["features"].endswith("_num"):
                texts = [re.sub(r"\b\d+(?:[.,:/-]\d+)*%?\b", " NUM ", text) for text in texts]
            return texts
        textual = Pipeline([("select", FunctionTransformer(get_text, validate=False)),
                            ("tfidf", TfidfVectorizer(analyzer=analyzer, ngram_range=(3, 5) if analyzer == "char" else (1, 2), min_df=2,
                                max_features=30000 if analyzer == "word" else 40000, sublinear_tf=True))])
        return Pipeline([("features", FeatureUnion([("structured", structured), ("text", textual)])),
                         ("model", model)])
    steps = [("dict", DictVectorizer(sparse=False, sort=True)),
             ("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    prep = cfg.get("preprocess", "standard")
    if prep == "standard":
        steps.append(("scale", StandardScaler()))
    elif prep == "robust":
        steps.append(("scale", RobustScaler()))
    elif prep == "quantile":
        steps.append(("scale", QuantileTransformer(output_distribution="normal", n_quantiles=100, random_state=seed)))
    elif prep != "none":
        raise ValueError(f"Unknown preprocessing: {prep}")
    if cfg.get("interactions"):
        steps.append(("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)))
    steps.append(("model", model))
    return Pipeline(steps)


def select_features(features, feature_set: str):
    base = feature_set.removesuffix("_word_num").removesuffix("_word").removesuffix("_char")
    prefix = {"native": ("native__",), "expanded": ("native__", "expanded__")}[base]
    return [{k: v for k, v in row.items() if k.startswith(prefix) or (feature_set != base and k == "__text__")} for row in features]

