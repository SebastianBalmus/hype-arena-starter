"""Lexicon-only entity recognition and linking with stable numeric IDs."""

from entity_linking.linker import EntityLinker, build_registry
from entity_linking.pipeline import ExtractionPipeline, extract_rows

__all__ = ["EntityLinker", "build_registry", "ExtractionPipeline", "extract_rows"]
