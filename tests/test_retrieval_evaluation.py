"""Detect missing or contaminated cross-fit traces without fitting a model."""

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.run_retrieval_experiments import audit_retrieval_artifacts


class RetrievalBoundaryAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.train = [{'id': str(i), 'text': f'Source {i}', 'date': '2026-01-01',
                       'label': ('true', 'false', 'unsure')[i % 3]} for i in range(5)]
        self.folds, self.oof = [], []
        for i, row in enumerate(self.train):
            refs = [r for r in self.train if r['id'] != row['id']]
            self.folds.append({'fold_index': i, 'query_ids': [row['id']],
                               'reference_ids': [r['id'] for r in refs]})
            neighbor = {k: refs[0][k] for k in ('id', 'date', 'label')}
            self.oof.append({'query_id': row['id'], 'fold_index': i,
                             'neighbors': {'raw': [{**neighbor, 'similarity': .7}]}})
        neighbor = {k: self.train[0][k] for k in ('id', 'date', 'label')}
        self.pending = [{'query_id': 'pending', 'views': {'raw': [{**neighbor, 'similarity': .5}]}}]

    def write(self):
        (self.path / 'oof_folds.json').write_text(json.dumps(self.folds))
        for filename, rows in [('oof_traces.jsonl', self.oof), ('traces.jsonl', self.pending)]:
            (self.path / filename).write_text(''.join(json.dumps(r) + '\n' for r in rows))

    def test_valid_fold_partition_and_pending_neighbors(self):
        self.write()
        result = audit_retrieval_artifacts(self.path, self.train, ['pending'])
        self.assertEqual(result['oof_rows'], 5)
        self.assertEqual(result['self_or_excluded_fold_neighbors'], 0)

    def test_missing_oof_query_is_rejected(self):
        self.oof.pop()
        self.write()
        with self.assertRaisesRegex(ValueError, 'cover each training ID'):
            audit_retrieval_artifacts(self.path, self.train, ['pending'])

    def test_self_neighbor_is_rejected(self):
        self.oof[0]['neighbors']['raw'][0].update(id='0', label='true')
        self.write()
        with self.assertRaisesRegex(ValueError, 'retrieval audit'):
            audit_retrieval_artifacts(self.path, self.train, ['pending'])

    def test_wrong_reference_label_is_rejected(self):
        self.pending[0]['views']['raw'][0]['label'] = 'false'
        self.write()
        with self.assertRaisesRegex(ValueError, 'retrieval audit'):
            audit_retrieval_artifacts(self.path, self.train, ['pending'])

    def test_unpermitted_per_class_neighbor_is_rejected(self):
        self.oof[0]['class_neighbors'] = {'raw': {'true': [
            {'id': '0', 'date': '2026-01-01', 'label': 'true', 'similarity': 1.0}]}}
        self.write()
        with self.assertRaisesRegex(ValueError, 'Per-class neighbor'):
            audit_retrieval_artifacts(self.path, self.train, ['pending'])


if __name__ == '__main__':
    unittest.main()
