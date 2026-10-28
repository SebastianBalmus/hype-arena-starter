#!/usr/bin/env python3
"""Small classification baselines. Run `python baseline.py --help`.

Only sentence text is used as a model feature. Dates define evaluation splits;
IDs identify rows; practice forms and the lexicon are intentionally unused.
This is an experiment runner, not a complete Arena submission.
"""

import argparse
import csv
import datetime as dt
import hashlib
import json
import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np
import scipy
import sklearn
from threadpoolctl import threadpool_limits
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parent
LABELS = ('true', 'false', 'unsure')
MODELS = ('always_true', 'tfidf_lr', 'tfidf_lr_balanced')
SPLITS = ('checker', 'stratified', 'temporal')
TEMPORAL_CUTOFF = '2026-09-01'


def read_rows(path):
    """Read the runtime columns only, deliberately discarding gold forms."""
    rows, ids = [], set()
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'id', 'text', 'date', 'label'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'{path}: requires columns {sorted(required)}')
        for line, original in enumerate(reader, 2):
            row = {key: (original.get(key) or '').strip() for key in required}
            row['label'] = row['label'].lower()
            if not row['id'] or row['id'] in ids:
                raise ValueError(f'{path}:{line}: missing or duplicate id {row["id"]!r}')
            if not row['text'] or row['label'] not in ('', *LABELS):
                raise ValueError(f'{path}:{line}: empty text or invalid label')
            try:
                dt.date.fromisoformat(row['date'])
            except ValueError as exc:
                raise ValueError(f'{path}:{line}: invalid ISO date {row["date"]!r}') from exc
            ids.add(row['id'])
            rows.append(row)
    if not rows:
        raise ValueError(f'{path}: no data rows')
    return rows


def split_rows(rows, name, seed=42):
    if any(row['label'] not in LABELS for row in rows):
        raise ValueError('Evaluation requires gold labels on every row')
    indices = list(range(len(rows)))
    if name == 'checker':
        train = [i for i in indices if i % 5 != 0]
        test = [i for i in indices if i % 5 == 0]
    elif name == 'stratified':
        train, test = train_test_split(
            indices, test_size=0.2, random_state=seed,
            stratify=[row['label'] for row in rows],
        )
    elif name == 'temporal':
        train = [i for i in indices if rows[i]['date'] < TEMPORAL_CUTOFF]
        test = [i for i in indices if rows[i]['date'] >= TEMPORAL_CUTOFF]
    else:
        raise ValueError(f'Unknown split {name!r}')
    if not train or not test:
        raise ValueError(f'{name}: both training and test partitions must be nonempty')
    return [rows[i] for i in sorted(train)], [rows[i] for i in sorted(test)]


def make_model(name, seed=42):
    if name == 'always_true':
        # Prior probabilities provide an honest prevalence-based confidence.
        # In this dataset true is the training majority in every supplied split.
        return DummyClassifier(strategy='prior', random_state=seed)
    if name not in MODELS:
        raise ValueError(f'Unknown model {name!r}')
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2), max_features=20000, sublinear_tf=True,
            token_pattern=r'(?u)\b\w+\b',
        )),
        ('classifier', LogisticRegression(
            C=1.0, max_iter=1000, random_state=seed,
            class_weight='balanced' if name.endswith('_balanced') else None,
        )),
    ])


def predict_rows(model, rows, min_confidence=0.5, force_true=False):
    """Apply the same decision policy in evaluation and CSV prediction.

    Uncalibrated model probabilities are diagnostic scores. If a true/false
    winner is below the contract's minimum confidence, emit unsure instead.
    Native unsure predictions also receive an empty output confidence.
    """
    if not rows:
        return []
    if not 0.5 <= min_confidence <= 1:
        raise ValueError('min_confidence must be between 0.5 and 1')
    probabilities = model.predict_proba([row['text'] for row in rows])
    predictions = []
    for row, distribution in zip(rows, probabilities):
        probs = {label: 0.0 for label in LABELS}
        probs.update(zip(model.classes_, map(float, distribution)))
        raw_label = 'true' if force_true else str(model.classes_[int(np.argmax(distribution))])
        raw_confidence = probs[raw_label]
        label = raw_label
        reason = 'model_unsure' if raw_label == 'unsure' else 'model_label'
        if label != 'unsure' and raw_confidence < min_confidence:
            label, reason = 'unsure', 'low_confidence'
        predictions.append({
            'id': row['id'], 'text': row['text'], 'date': row['date'],
            'gold': row['label'], 'label': label,
            'confidence': raw_confidence if label != 'unsure' else '',
            'raw_label': raw_label, 'raw_confidence': raw_confidence,
            'decision_reason': reason,
            **{f'p_{key}': value for key, value in probs.items()},
        })
    return predictions


def compute_metrics(rows, predictions):
    gold = [row['label'] for row in rows]
    predicted = [p['label'] for p in predictions]
    report = classification_report(
        gold, predicted, labels=list(LABELS), output_dict=True, zero_division=0,
    )
    decided = [(g, p) for g, p in zip(gold, predicted) if p != 'unsure']
    # log_loss expects columns in alphabetical order when labels are supplied.
    ordered_labels = sorted(LABELS)
    probabilities = [[p[f'p_{label}'] for label in ordered_labels] for p in predictions]
    return {
        'rows': len(rows), 'accuracy': float(accuracy_score(gold, predicted)),
        'macro_f1': report['macro avg']['f1-score'],
        'false_recall': report['false']['recall'],
        'decided_coverage': len(decided) / len(rows),
        'decided_precision': sum(g == p for g, p in decided) / len(decided) if decided else None,
        'low_confidence_abstentions': sum(p['decision_reason'] == 'low_confidence' for p in predictions),
        'log_loss_uncalibrated': float(log_loss(gold, probabilities, labels=ordered_labels)),
        'per_class': {label: report[label] for label in LABELS},
        'confusion_label_order': list(LABELS),
        'confusion_matrix_gold_rows_predicted_columns': confusion_matrix(
            gold, predicted, labels=list(LABELS),
        ).tolist(),
    }


def write_predictions(path, predictions, arena_format=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = (['id', 'type', 'text', 'form', 'label', 'confidence'] if arena_format else
               ['id', 'date', 'text', 'gold', 'label', 'confidence', 'raw_label',
                'raw_confidence', 'decision_reason', 'p_true', 'p_false', 'p_unsure'])
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for prediction in predictions:
            if arena_format:
                writer.writerow({
                    'id': prediction['id'], 'type': '', 'text': '', 'form': '',
                    'label': prediction['label'], 'confidence': prediction['confidence'],
                })
            else:
                writer.writerow(prediction)


def evaluate(args):
    rows = read_rows(args.data)
    results = {
        'dataset': str(args.data.resolve()),
        'dataset_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
        'seed': args.seed, 'min_confidence': args.min_confidence,
        'features': 'sentence text only; no dates, ids, forms or lexicon',
        'probabilities': 'uncalibrated; do not interpret as verified truth probabilities',
        'versions': {'python': platform.python_version(), 'sklearn': sklearn.__version__,
                     'numpy': np.__version__, 'scipy': scipy.__version__},
        'splits': {},
    }
    names = SPLITS if args.split == 'all' else (args.split,)
    print(f'{"split":12} {"model":21} {"accuracy":>9} {"macro F1":>9} {"false R":>9} {"coverage":>9}')
    for name in names:
        train, test = split_rows(rows, name, args.seed)
        overlap = {r['text'] for r in train} & {r['text'] for r in test}
        split = {
            'train_rows': len(train), 'test_rows': len(test),
            'train_label_counts': dict(Counter(r['label'] for r in train)),
            'test_label_counts': dict(Counter(r['label'] for r in test)),
            'train_ids': [r['id'] for r in train], 'test_ids': [r['id'] for r in test],
            'exact_text_overlap_count': len(overlap), 'models': {},
        }
        for model_name in args.models:
            model = make_model(model_name, args.seed)
            start = time.perf_counter()
            model.fit([r['text'] for r in train], [r['label'] for r in train])
            fitted = time.perf_counter()
            predictions = predict_rows(model, test, args.min_confidence,
                                       force_true=model_name == 'always_true')
            predicted = time.perf_counter()
            metrics = compute_metrics(test, predictions)
            metrics['fit_seconds'] = fitted - start
            metrics['predict_seconds'] = predicted - fitted
            split['models'][model_name] = metrics
            write_predictions(args.output_dir / 'predictions' / f'{name}-{model_name}.csv', predictions)
            print(f'{name:12} {model_name:21} {metrics["accuracy"]:9.3f} '
                  f'{metrics["macro_f1"]:9.3f} {metrics["false_recall"]:9.3f} '
                  f'{metrics["decided_coverage"]:9.3f}')
        results['splits'][name] = split
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / 'metrics.json'
    destination.write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    print(f'\nMetrics and fixed split IDs: {destination}')
    print(f'Row-level predictions: {args.output_dir / "predictions"}')


def predict(args):
    train = [r for r in read_rows(args.train) if r['label']]
    if not train:
        raise ValueError('No labelled training rows')
    pending = [r for r in read_rows(args.input) if not r['label']]
    model = make_model(args.model, args.seed)
    model.fit([r['text'] for r in train], [r['label'] for r in train])
    predictions = predict_rows(model, pending, args.min_confidence,
                               force_true=args.model == 'always_true')
    write_predictions(args.output, predictions, arena_format=True)
    print(f'Wrote {len(predictions)} classification rows to {args.output}')
    print('This file contains no generated claims; it is not a complete Arena submission.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    evaluation = commands.add_parser('evaluate', help='Compare models on fixed held-out rows')
    evaluation.add_argument('--data', type=Path, default=ROOT / 'data/practice.csv')
    evaluation.add_argument('--split', choices=(*SPLITS, 'all'), default='checker')
    evaluation.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    evaluation.add_argument('--output-dir', type=Path, default=ROOT / 'results/baseline')
    evaluation.set_defaults(action=evaluate)
    prediction = commands.add_parser('predict', help='Train and label only blank-label input rows')
    prediction.add_argument('--train', type=Path, default=ROOT / 'data/practice.csv')
    prediction.add_argument('--input', type=Path, required=True)
    prediction.add_argument('--model', choices=MODELS, default='tfidf_lr')
    prediction.add_argument('--output', type=Path, default=ROOT / 'outputs/labels.csv')
    prediction.set_defaults(action=predict)
    for command in (evaluation, prediction):
        command.add_argument('--seed', type=int, default=42)
        command.add_argument('--min-confidence', type=float, default=0.5)
    args = parser.parse_args()
    if not 0.5 <= args.min_confidence <= 1:
        parser.error('--min-confidence must be between 0.5 and 1')
    try:
        with threadpool_limits(limits=2):
            args.action(args)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
