#!/usr/bin/env python3
"""Run split-scoped, text-only classification experiments on sanitized rows."""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import baseline

MODELS = (
    *baseline.MODELS,
    'char_lr_balanced', 'word_char_lr_balanced', 'word_char_lr_unweighted',
    'word_char_complement_nb', 'word_lr_unsure_boost',
)


def make_model(name, seed):
    if name in baseline.MODELS:
        return baseline.make_model(name, seed)
    char = TfidfVectorizer(analyzer='char', ngram_range=(3, 5),
                           max_features=30000, sublinear_tf=True)
    word = TfidfVectorizer(ngram_range=(1, 2), max_features=20000,
                           sublinear_tf=True, token_pattern=r'(?u)\b\w+\b')
    if name == 'char_lr_balanced':
        features = char
        classifier = LogisticRegression(C=1.0, max_iter=1000, random_state=seed,
                                        class_weight='balanced')
    elif name in ('word_char_lr_balanced', 'word_char_lr_unweighted'):
        features = FeatureUnion([('word', word), ('char', char)])
        classifier = LogisticRegression(
            C=1.0, max_iter=1000, random_state=seed,
            class_weight='balanced' if name.endswith('_balanced') else None)
    elif name == 'word_char_complement_nb':
        features = FeatureUnion([('word', word), ('char', char)])
        classifier = ComplementNB()
    elif name == 'word_lr_unsure_boost':
        features = word
        classifier = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
    else:
        raise ValueError(f'Unknown model {name!r}')
    return Pipeline([('tfidf', features), ('classifier', classifier)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True,
                        help='Sanitized CSV: labelled training rows, blank-label test rows')
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timings', type=Path,
                        help='Optional JSON timing sidecar path')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    rows = baseline.read_rows(args.input)
    train = [r for r in rows if r['label']]
    pending = [r for r in rows if not r['label']]
    if not train:
        parser.error('input has no labelled training rows')
    if not pending:
        parser.error('input has no blank-label rows to predict')
    model = make_model(args.model, args.seed)
    fit_params = {}
    if args.model == 'word_lr_unsure_boost':
        counts = {label: sum(r['label'] == label for r in train)
                  for label in baseline.LABELS}
        n, k = len(train), len(baseline.LABELS)
        weights = {label: n / (k * counts[label]) for label in baseline.LABELS}
        weights['unsure'] *= 2.0
        fit_params['classifier__sample_weight'] = np.asarray(
            [weights[r['label']] for r in train], dtype=float)
    with threadpool_limits(limits=2):
        fit_start = time.perf_counter()
        model.fit([r['text'] for r in train], [r['label'] for r in train], **fit_params)
        fit_seconds = time.perf_counter() - fit_start
        inference_start = time.perf_counter()
        predictions = baseline.predict_rows(model, pending, min_confidence=0.5)
        inference_seconds = time.perf_counter() - inference_start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialization_start = time.perf_counter()
    with args.output.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            'id', 'label', 'confidence', 'decision_reason',
            'p_true', 'p_false', 'p_unsure'])
        writer.writeheader()
        for prediction in predictions:
            writer.writerow({
                'id': prediction['id'], 'label': prediction['label'],
                'confidence': prediction['confidence'],
                'decision_reason': ('decided' if prediction['decision_reason'] == 'model_label'
                                    else prediction['decision_reason']),
                **{f'p_{label}': prediction[f'p_{label}'] for label in baseline.LABELS},
            })
    serialization_seconds = time.perf_counter() - serialization_start
    if args.timings:
        args.timings.parent.mkdir(parents=True, exist_ok=True)
        args.timings.write_text(json.dumps({
            'fit_seconds': fit_seconds,
            'inference_seconds': inference_seconds,
            'serialization_seconds': serialization_seconds,
            'generation_status': 'not_implemented',
            'audit_status': 'not_implemented',
        }, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
