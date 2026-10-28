"""Reusable offline extraction pipeline; CLI-independent and model loaded once."""
from itertools import islice
import json
import os
from pathlib import Path


def configure_runtime():
    """Set offline CPU defaults before loading optional embedding dependencies."""
    for key, value in {'HF_HUB_OFFLINE':'1', 'TOKENIZERS_PARALLELISM':'false',
                       'OMP_NUM_THREADS':'2', 'OPENBLAS_NUM_THREADS':'2'}.items():
        os.environ.setdefault(key, value)


class ExtractionPipeline:
    def __init__(self, linker, ranker, *, batch_size=256):
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError('batch_size must be a positive integer')
        from entity_linking.forms import FormExtractor
        self.linker, self.ranker = linker, ranker
        self.extractor = FormExtractor(linker, ranker)
        self.batch_size = batch_size

    @classmethod
    def from_paths(cls, model_path, lexicon_path, registry_path, *, threshold=.4, margin=.02, batch_size=256):
        configure_runtime()
        from entity_linking import EntityLinker, build_registry
        from entity_linking.relationships import LexiconRelationshipRanker
        lexicon_path, registry_path = Path(lexicon_path), Path(registry_path)
        registry = build_registry(json.loads(lexicon_path.read_text()), json.loads(registry_path.read_text()))
        linker = EntityLinker(registry)
        ranker = LexiconRelationshipRanker.load(model_path, registry_path, threshold, margin, lexicon_path)
        return cls(linker, ranker, batch_size=batch_size)

    def annotate(self, text):
        """Entity and typed-value annotations without relationship extraction."""
        from entity_linking.values import annotate
        return annotate(text, self.linker)

    def extract(self, rows):
        """Yield full forms and traces from any iterable of source rows."""
        iterator = iter(rows)
        while batch := list(islice(iterator, self.batch_size)):
            yield from self.extractor.extract(batch)


def extract_rows(rows, model_path, lexicon_path, registry_path, threshold=.4, margin=.02):
    """Convenience API preserving the original extraction function signature."""
    pipeline = ExtractionPipeline.from_paths(model_path, lexicon_path, registry_path,
                                             threshold=threshold, margin=margin)
    yield from pipeline.extract(rows)
