"""Fold-aware GradientBoosting estimator for leakage-safe GridSearchCV.

Each feature cache was built from its exact fold's labelled training pool and
unlabelled validation text. Cache lookup never reuses full-pool factual features
inside a smaller CV fold.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from joblib import parallel_backend
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.daria_sklearn import select_features
from integrations.daria_candidates import probabilities_to_rows
from evaluation.run_text_experiments import write_csv, write_json

LABELS = ['true', 'false', 'unsure']
PARAM_GRID = {'n_estimators': [40, 80, 160], 'learning_rate': [.03, .05, .1],
              'max_depth': [1, 2, 3], 'min_samples_leaf': [1, 8],
              'subsample': [.8, 1.]}
BASE_PARAMS = {'n_estimators': 80, 'learning_rate': .05, 'max_depth': 2,
               'min_samples_leaf': 1, 'subsample': 1.}


def fixed_050_macro_scorer(estimator, X, y):
    """Call predictions directly: macro F1 has no binary positive-label parameter."""
    return f1_score(y, estimator.predict(X), labels=LABELS, average='macro', zero_division=0)


@lru_cache(maxsize=24)
def load_cache(path):
    return json.loads(Path(path).read_text())


class FoldCachedGradientBoosting(ClassifierMixin, BaseEstimator):
    def __init__(self, cache_manifest, n_estimators=80, learning_rate=.05,
                 max_depth=2, min_samples_leaf=1, subsample=1., random_state=42):
        self.cache_manifest = cache_manifest
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.random_state = random_state

    def fit(self, X, y):
        ids = [str(x) for x in np.asarray(X).reshape(-1)]
        if len(ids) != len(set(ids)) or len(ids) != len(y):
            raise ValueError('Training IDs must be unique and align with labels')
        manifest = json.loads(Path(self.cache_manifest).read_text())
        matches = [item for item in manifest['caches'] if set(item['train_ids']) == set(ids)]
        if len(matches) != 1:
            raise ValueError('No unique feature cache for this exact training pool')
        entry = matches[0]
        cache = load_cache(entry['path'])
        if (set(cache['train_ids']) != set(ids) or len(cache['train_ids']) != len(ids)
                or set(cache['audit']['train_ids']) != set(ids)):
            raise ValueError('Feature cache training-pool audit failed')
        labels = dict(zip(cache['train_ids'], cache['train_labels']))
        if labels != dict(zip(ids, map(str, y))):
            raise ValueError('Supplied labels disagree with training cache')
        self.pending_ = {r['id']: r['features'] for r in cache['pending']}
        if (len(self.pending_) != len(cache['pending'])
                or set(self.pending_) != set(entry['test_ids'])
                or set(self.pending_) != set(cache['audit']['pending_ids'])):
            raise ValueError('Feature cache pending-pool audit failed')
        if set(ids) & self.pending_.keys():
            raise ValueError('Feature cache exposes training IDs as pending')
        self.cache_path_ = entry['path']
        self.model_ = Pipeline([
            ('dict', DictVectorizer(sparse=False, sort=True)),
            ('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
            ('model', GradientBoostingClassifier(n_estimators=self.n_estimators,
                learning_rate=self.learning_rate, max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf, subsample=self.subsample,
                random_state=self.random_state))])
        self.model_.fit(select_features(cache['train_features'], 'expanded'), cache['train_labels'])
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        ids = [str(x) for x in np.asarray(X).reshape(-1)]
        if any(rid not in self.pending_ for rid in ids):
            raise ValueError('Prediction ID is not in this fold’s sanitized pending batch')
        return self.model_.predict_proba(select_features([self.pending_[rid] for rid in ids], 'expanded'))

    def predict(self, X):
        probs = self.predict_proba(X)
        best = np.argmax(probs, axis=1)
        labels = np.asarray(self.classes_[best], dtype=object)
        labels[np.max(probs, axis=1) < .5] = 'unsure'
        return labels


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='mode', required=True)
    search = sub.add_parser('search')
    search.add_argument('--spec', required=True)
    search.add_argument('--output', required=True)
    predict = sub.add_parser('predict')
    predict.add_argument('--cache', required=True)
    predict.add_argument('--params', required=True)
    predict.add_argument('--output', required=True)
    predict.add_argument('--timings', required=True)
    args = ap.parse_args()
    with threadpool_limits(limits=1):
        if args.mode == 'search':
            spec = json.loads(Path(args.spec).read_text())
            folds = [(np.asarray(f['train_indices']), np.asarray(f['test_indices'])) for f in spec['folds']]
            grid = GridSearchCV(FoldCachedGradientBoosting(spec['cache_manifest']),
                PARAM_GRID, scoring=fixed_050_macro_scorer, cv=folds, n_jobs=2, pre_dispatch=2,
                refit=False, error_score='raise', return_train_score=False, verbose=1)
            start = time.perf_counter()
            with parallel_backend('threading', n_jobs=2):
                grid.fit(np.asarray(spec['ids'], dtype=object), np.asarray(spec['labels']))
            baseline_index = grid.cv_results_['params'].index(BASE_PARAMS)
            result = {'best_params': grid.best_params_, 'best_score': float(grid.best_score_),
                'best_index': int(grid.best_index_), 'grid': PARAM_GRID, 'folds': spec['folds'],
                'baseline_params': BASE_PARAMS,
                'baseline_cv_score': float(grid.cv_results_['mean_test_score'][baseline_index]),
                'search_seconds': time.perf_counter() - start,
                'cv_results': {k: v.tolist() if hasattr(v, 'tolist') else v
                               for k, v in grid.cv_results_.items()}}
            write_json(args.output, result)
            print(json.dumps({'best_params': grid.best_params_, 'best_score': grid.best_score_}), flush=True)
        else:
            cache = load_cache(args.cache)
            params = json.loads(Path(args.params).read_text())
            start = time.perf_counter()
            model = Pipeline([('dict', DictVectorizer(sparse=False, sort=True)),
                ('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
                ('model', GradientBoostingClassifier(**params, random_state=42))])
            model.fit(select_features(cache['train_features'], 'expanded'), cache['train_labels'])
            fit_seconds = time.perf_counter() - start
            start = time.perf_counter()
            ids = [r['id'] for r in cache['pending']]
            probs = model.predict_proba(select_features([r['features'] for r in cache['pending']], 'expanded'))
            predictions = probabilities_to_rows(ids, probs, model.classes_)
            predict_seconds = time.perf_counter() - start
            start = time.perf_counter()
            write_csv(args.output, predictions, ['id', 'label', 'confidence', 'decision_reason',
                                               'p_true', 'p_false', 'p_unsure'])
            write_json(args.timings, {'fit_seconds': fit_seconds, 'inference_seconds': predict_seconds,
                                     'serialization_seconds': time.perf_counter() - start})


if __name__ == '__main__':
    main()
