from pathlib import Path
import tempfile
import unittest

import numpy as np
from entity_linking import EntityLinker, ExtractionPipeline, build_registry
from entity_linking.io import read_rows, write_rows
from entity_linking.relationships import LexiconRelationshipRanker


class Encoder:
    dim = 1
    def encode(self, texts):
        return np.ones((len(texts), 1), dtype=np.float32)


class PipelineTests(unittest.TestCase):
    def pipeline(self, batch_size=1):
        registry = build_registry({'subjects':[{'name':'Product','aliases':[], 'attributes':['price']}], 'events':[]})
        return ExtractionPipeline(EntityLinker(registry), LexiconRelationshipRanker(Encoder(),registry), batch_size=batch_size)

    def test_reusable_pipeline_streams_batches_without_source_mutation_or_state_leak(self):
        rows=[{'id':str(i),'text':f'Product costs €{i+1} per month.', 'date':'2026-01-01',
               'label':'false','form':'forbidden gold'} for i in range(3)]
        original=[dict(r) for r in rows]
        first=list(self.pipeline().extract(iter(rows)))
        pipeline=self.pipeline(batch_size=2)
        self.assertEqual(list(pipeline.extract(iter(rows))),first)
        self.assertEqual(list(pipeline.extract(iter(rows))),first)
        self.assertEqual(rows,original)
        self.assertEqual([r['predicted_form'][0]['value']['value'] for r in first],[1,2,3])
        self.assertEqual(list(pipeline.extract(iter(()))),[])

    def test_standalone_annotations_use_same_registry_as_full_extraction(self):
        pipeline=self.pipeline()
        annotations=pipeline.annotate('Product costs €49 per month.')
        self.assertEqual(annotations['mentions'][0]['entity_id'],next(iter(pipeline.ranker.entities)))
        self.assertEqual(annotations['values'][0]['kind'],'money')

    def test_invalid_batch_size_is_rejected(self):
        for size in [0,-1,True,1.5]:
            with self.assertRaises(ValueError):self.pipeline(size)

    def test_shared_csv_roundtrip_preserves_quoted_and_multiline_text(self):
        rows=[{'id':'one','text':'A comma, a "quote"\nand a line','date':'2026-01-01'}]
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'rows.csv'
            write_rows(path,rows,['id','text','date'])
            found,fields=read_rows(path)
            self.assertEqual(found,rows)
            self.assertEqual(fields,['id','text','date'])
