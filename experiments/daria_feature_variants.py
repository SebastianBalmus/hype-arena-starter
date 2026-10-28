"""Fold-local feature representations for fixed-parameter Daria GB comparisons.

Every candidate is a sklearn Pipeline over feature-cache dictionaries. Text vectorizers,
feature selectors, scaling and dimensionality reduction are fitted only by ``Pipeline.fit``
on the current training fold. The module adds no parser, evidence rule or pretrained model.
"""
from __future__ import annotations

import re
from functools import partial
from collections import defaultdict

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from sklearn.feature_extraction.text import (CountVectorizer, HashingVectorizer,
                                             TfidfTransformer, TfidfVectorizer)
from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, Normalizer, StandardScaler


def _structured(records, mode="expanded"):
    """Select structured keys; never place IDs or raw text in the tabular branch."""
    prefixes = ("native__", "expanded__")
    if mode.startswith("rich"):
        prefixes += ("rich__",)
    drop_calendar = "no_calendar" in mode
    drop_parse = "no_parse" in mode
    result = []
    for record in records:
        current = {}
        for key, value in record.items():
            if not isinstance(key, str) or not key.startswith(prefixes):
                continue
            if drop_calendar and key in {"expanded__year", "expanded__month", "expanded__day"}:
                continue
            if drop_parse and "parse" in key.lower():
                continue
            current[key] = value
        result.append(current)
    return result


def _aggregate_atoms(records):
    """Replace per-position temporal atom keys with order-invariant summaries."""
    pattern = re.compile(r"^expanded__atom\d+_(.+)$")
    result = []
    for record in records:
        row = {k: v for k, v in record.items()
               if isinstance(k, str) and k.startswith(("native__", "expanded__"))
               and not pattern.match(k)}
        groups = defaultdict(list)
        for key, value in record.items():
            match = pattern.match(key) if isinstance(key, str) else None
            if match:
                groups[match.group(1)].append(value)
        for name, values in groups.items():
            numeric = []
            for value in values:
                try:
                    numeric.append(float(value))
                except (TypeError, ValueError):
                    continue
            row[f"aggregate__{name}__count"] = len(values)
            if numeric:
                row[f"aggregate__{name}__sum"] = float(sum(numeric))
                row[f"aggregate__{name}__mean"] = float(sum(numeric) / len(numeric))
                row[f"aggregate__{name}__max"] = float(max(numeric))
        result.append(row)
    return result


def _rich_bags(records):
    """Keep only rich entity/attribute/type bag features alongside base features."""
    return [{k: v for k, v in _structured([row], "rich")[0].items()
             if not k.startswith("rich__") or k.startswith("rich__content_")}
            for row in records]


def _texts(records, mode="word"):
    texts = [str(row.get("__text__", "")) for row in records]
    if mode == "word_num":
        # Numeric time/value cues are represented by structured metadata where available.
        texts = [re.sub(r"(?<!\w)\d+(?:[.,:/-]\d+)*(?:%|\w+)?", " NUM ", text)
                 for text in texts]
    return texts


def _dense(matrix):
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


class AdaptiveKBest(BaseEstimator, TransformerMixin):
    """Select at most k features, keeping small fold smoke fixtures usable."""
    def __init__(self, k=128, score_func=chi2):
        self.k = k
        self.score_func = score_func

    def fit(self, X, y):
        width = X.shape[1]
        self.selector_ = SelectKBest(score_func=self.score_func, k=min(int(self.k), width))
        self.selector_.fit(X, y)
        return self

    def transform(self, X):
        return self.selector_.transform(X)


class AdaptiveSVD(BaseEstimator, TransformerMixin):
    """Fit requested SVD dimensions, bounded by the current fold's matrix shape."""
    def __init__(self, n_components=32, random_state=42):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        limit = min(X.shape) - 1
        components = max(1, min(int(self.n_components), int(limit)))
        self.svd_ = TruncatedSVD(n_components=components, random_state=self.random_state)
        self.svd_.fit(X)
        return self

    def transform(self, X):
        return self.svd_.transform(X)


class AdaptivePCA(BaseEstimator, TransformerMixin):
    """PCA with an effective component count supported by the current fold."""
    def __init__(self, n_components=32, random_state=42):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        components = max(1, min(int(self.n_components), X.shape[0] - 1, X.shape[1]))
        self.pca_ = PCA(n_components=components, random_state=self.random_state)
        self.pca_.fit(X)
        return self

    def transform(self, X):
        return self.pca_.transform(X)


def _score_mi(X, y):
    return mutual_info_classif(X, y, random_state=42)


def _select_records(mode):
    if mode == "aggregate":
        return FunctionTransformer(_aggregate_atoms, validate=False)
    if mode == "rich_bags":
        return FunctionTransformer(_rich_bags, validate=False)
    return FunctionTransformer(partial(_structured, mode=mode), validate=False)


def _text_branch(config, seed):
    kind = config["kind"]
    if kind in {"word_binary_128", "word_binary_256", "word_num_binary_128"}:
        text_mode = "word_num" if kind.startswith("word_num") else "word"
        k = 256 if "256" in kind else 128
        return Pipeline([
            ("text", FunctionTransformer(partial(_texts, mode=text_mode), validate=False)),
            ("vectorize", CountVectorizer(binary=True, ngram_range=(1, 2), min_df=2, max_features=16000)),
            ("select", AdaptiveKBest(k=k, score_func=chi2)),
            ("dense", FunctionTransformer(_dense, validate=False)),
        ])
    if kind == "char_chi2_128":
        return Pipeline([
            ("text", FunctionTransformer(_texts, validate=False)),
            ("vectorize", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                          max_features=16000, sublinear_tf=True)),
            ("select", AdaptiveKBest(k=128, score_func=chi2)),
            ("dense", FunctionTransformer(_dense, validate=False)),
        ])
    if kind.startswith("word_svd"):
        dims = int(kind.removeprefix("word_svd"))
        return _svd_text_branch("word", dims, seed)
    if kind.startswith("char_svd"):
        dims = int(kind.removeprefix("char_svd"))
        return _svd_text_branch("char_wb", dims, seed)
    if kind == "word_char_svd64":
        return Pipeline([
            ("union", FeatureUnion([
                ("word", _tfidf_branch("word")),
                ("char", _tfidf_branch("char_wb")),
            ])),
            ("svd", AdaptiveSVD(64, seed)),
            ("norm", Normalizer()),
        ])
    if kind == "hash_tfidf_svd32":
        return Pipeline([
            ("text", FunctionTransformer(_texts, validate=False)),
            ("hash", HashingVectorizer(n_features=4096, alternate_sign=False,
                                       ngram_range=(1, 2), norm=None)),
            ("tfidf", TfidfTransformer()),
            ("svd", AdaptiveSVD(32, seed)),
            ("norm", Normalizer()),
        ])
    raise KeyError(kind)


def _tfidf_branch(analyzer):
    return Pipeline([
        ("text", FunctionTransformer(_texts, validate=False)),
        ("tfidf", TfidfVectorizer(analyzer=analyzer,
            ngram_range=(3, 5) if analyzer == "char_wb" else (1, 2),
            min_df=2, max_features=16000, sublinear_tf=True)),
    ])


def _svd_text_branch(analyzer, components, seed):
    return Pipeline([
        ("text", FunctionTransformer(_texts, validate=False)),
        ("tfidf", TfidfVectorizer(analyzer=analyzer,
            ngram_range=(3, 5) if analyzer == "char_wb" else (1, 2),
            min_df=2, max_features=16000, sublinear_tf=True)),
        ("svd", AdaptiveSVD(components, seed)),
        ("norm", Normalizer()),
    ])


def _structured_branch(config, seed):
    kind = config["kind"]
    mode = config.get("structured", "expanded")
    select = _select_records(mode)
    if kind == "feature_hasher128":
        return Pipeline([
            ("select_records", select),
            ("hash", FeatureHasher(n_features=128, input_type="dict", alternate_sign=False)),
            ("dense", FunctionTransformer(_dense, validate=False)),
        ])
    steps = [("select_records", select), ("dict", DictVectorizer(sparse=False, sort=True)),
             ("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if kind.startswith("select_percentile_"):
        method, pct = kind.removeprefix("select_percentile_").split("_")
        scorer = f_classif if method == "f" else _score_mi
        steps.append(("select", SelectPercentile(score_func=scorer, percentile=int(pct))))
    if kind == "pca32_scaled":
        steps.extend([("scale", StandardScaler()), ("pca", AdaptivePCA(32, seed))])
    return Pipeline(steps)


def _gb(config, gb_params, seed):
    params = dict(gb_params or {})
    params.setdefault("random_state", seed)
    return GradientBoostingClassifier(**params)


CONFIGS = (
    {"id": "expanded_none", "kind": "expanded_none", "structured": "expanded"},
    {"id": "expanded_order_aggregate", "kind": "aggregate", "structured": "aggregate"},
    {"id": "rich_structured", "kind": "rich_structured", "structured": "rich"},
    {"id": "rich_no_calendar", "kind": "rich_no_calendar", "structured": "rich_no_calendar"},
    {"id": "rich_no_parse_confidence", "kind": "rich_no_parse", "structured": "rich_no_parse"},
    {"id": "rich_content_bags", "kind": "rich_bags", "structured": "rich_bags"},
    {"id": "word_binary_chi2_128", "kind": "word_binary_128"},
    {"id": "word_binary_chi2_256", "kind": "word_binary_256"},
    {"id": "word_number_masked_chi2_128", "kind": "word_num_binary_128"},
    {"id": "char_wb_chi2_128", "kind": "char_chi2_128"},
    {"id": "word_tfidf_svd32", "kind": "word_svd32"},
    {"id": "word_tfidf_svd64", "kind": "word_svd64"},
    {"id": "char_wb_tfidf_svd32", "kind": "char_svd32"},
    {"id": "char_wb_tfidf_svd64", "kind": "char_svd64"},
    {"id": "word_char_tfidf_svd64", "kind": "word_char_svd64"},
    {"id": "hash_tfidf_svd32", "kind": "hash_tfidf_svd32"},
    {"id": "text_only_word_binary128", "kind": "word_binary_128", "text_only": True},
    {"id": "text_only_word_tfidf_svd32", "kind": "word_svd32", "text_only": True},
    {"id": "structured_feature_hasher128", "kind": "feature_hasher128", "structured": "rich"},
    {"id": "rich_select_f50", "kind": "select_percentile_f_50", "structured": "rich"},
    {"id": "rich_select_f75", "kind": "select_percentile_f_75", "structured": "rich"},
    {"id": "rich_select_mi50", "kind": "select_percentile_mi_50", "structured": "rich"},
    {"id": "rich_select_mi75", "kind": "select_percentile_mi_75", "structured": "rich"},
    {"id": "rich_pca32_scaled", "kind": "pca32_scaled", "structured": "rich"},
)


def make_feature_model(config, gb_params=None, seed=42):
    """Build a train-only sklearn pipeline over raw cache feature dictionaries."""
    cfg = dict(config)
    kind = cfg["kind"]
    gb = _gb(cfg, gb_params, seed)
    text_kinds = {"word_binary_128", "word_binary_256", "word_num_binary_128",
                "char_chi2_128", "word_svd32", "word_svd64", "char_svd32",
                "char_svd64", "word_char_svd64", "hash_tfidf_svd32"}
    if kind in text_kinds:
        text = _text_branch(cfg, seed)
        if cfg.get("text_only", False):
            features = text
        else:
            structured = _structured_branch({"kind": "expanded_none", "structured": "expanded"}, seed)
            features = FeatureUnion([("structured", structured), ("text", text)])
        return Pipeline([("features", features), ("model", gb)])
    return Pipeline([("features", _structured_branch(cfg, seed)), ("model", gb)])

