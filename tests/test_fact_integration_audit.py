"""Detect synthetic references crossing their permitted source-row boundary."""

import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from evaluation.run_fact_integration import audit_claims


class ClaimProvenanceAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)
        self.rows = [{'id': str(i), 'text': f'Observed value {i}', 'date': '2026-10-20',
                      'label': 'false' if i == 5 else 'true'} for i in range(6)]
        self.catalogues, self.folds, self.traces = {}, [], []
        for i in range(5):
            refs = [r['id'] for r in self.rows if r['id'] != str(i)]
            self.folds.append({'fold_index': i, 'query_ids': [str(i)], 'reference_ids': refs})
            source = next(r for r in self.rows if r['id'] in refs and r['label'] == 'true')
            record = self.record(f'candidate-{i}', source)
            self.catalogues[f'fold-{i}'] = {'reference_ids': refs, 'records': [record]}
            self.traces.append({'query_id': str(i), 'fold_index': i,
                                'auxiliary_claims': {'query_id': str(i), 'neighbors': [self.neighbor(record)]}})
        record = self.record('full-candidate', self.rows[0])
        self.catalogues['full'] = {'reference_ids': [r['id'] for r in self.rows], 'records': [record]}
        self.pending = [{'query_id': 'pending', 'auxiliary_claims':
                         {'query_id': 'pending', 'neighbors': [self.neighbor(record)]}}]

    @staticmethod
    def record(cid, row):
        fingerprint = hashlib.sha256('\0'.join(row[k] for k in ('id', 'text', 'date')).encode()).hexdigest()
        return {'candidate_id': cid, 'type': 'fact', 'source_ids': [row['id']],
                'source_fingerprints': {row['id']: fingerprint}, 'source_dates': [row['date']],
                'status': 'unverified_auxiliary'}

    @staticmethod
    def neighbor(record):
        return {k: record[k] for k in ('candidate_id', 'source_ids', 'source_dates')} | {'similarity': .8}

    def write(self):
        with gzip.open(self.path / 'claim_catalogues.json.gz', 'wt') as stream:
            json.dump(self.catalogues, stream)
        (self.path / 'oof_folds.json').write_text(json.dumps(self.folds))
        for filename, rows in [('oof_traces.jsonl', self.traces), ('traces.jsonl', self.pending)]:
            (self.path / filename).write_text(''.join(json.dumps(r) + '\n' for r in rows))

    def test_all_fold_and_full_catalogues_have_admissible_sources(self):
        self.write()
        audit = audit_claims(self.path, self.rows, ['pending'])
        self.assertEqual(audit['catalogue_scopes'], 6)
        self.assertEqual(audit['candidate_records_checked'], 6)
        self.assertEqual(audit['truth_verification'], 'not_verified')

    def test_false_row_cannot_be_source(self):
        self.catalogues['full']['records'] = [self.record('full-candidate', self.rows[5])]
        self.write()
        with self.assertRaisesRegex(ValueError, 'matching true permitted row'):
            audit_claims(self.path, self.rows, ['pending'])

    def test_training_query_form_cannot_enter_its_own_catalogue(self):
        self.catalogues['fold-0']['records'] = [self.record('candidate-0', self.rows[0])]
        self.write()
        with self.assertRaisesRegex(ValueError, 'outside its fold'):
            audit_claims(self.path, self.rows, ['pending'])

    def test_reused_source_id_with_changed_text_fails_fingerprint(self):
        self.rows[0]['text'] = 'Changed original observation'
        self.write()
        with self.assertRaisesRegex(ValueError, 'matching true permitted row'):
            audit_claims(self.path, self.rows, ['pending'])

    def test_oof_trace_cannot_retrieve_full_catalogue(self):
        self.traces[0]['auxiliary_claims']['neighbors'] = [self.neighbor(self.catalogues['full']['records'][0])]
        self.write()
        with self.assertRaisesRegex(ValueError, 'correct fold catalogue'):
            audit_claims(self.path, self.rows, ['pending'])


if __name__ == '__main__':
    unittest.main()
