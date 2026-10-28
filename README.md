# Extraction and knowledge base

The restored extraction pipeline recognizes lexicon entities and aliases, tags
money, quantities, durations, percentages, dates, versions, names and Boolean
polarity, and binds extracted values to lexicon attributes using structural rules
and the local CPU POTION model. Classification and temporal experiments are absent.

```sh
uv sync

# Full tagging and relationship extraction on sentence text.
uv run python scripts/extract_forms.py --input data/practice.csv --output artifacts/extracted

# Optional stage inspections and extraction evaluation.
uv run python scripts/normalize_entities.py
uv run python scripts/annotate_values.py
uv run python scripts/evaluate_forms.py

# Trusted KB, built from provided forms on true rows.
uv run python scripts/build_knowledge_base.py
```

The extraction and KB commands are independent. The KB retains 1,998 assertions
from 1,581 true rows as multiline JSON at
`artifacts/knowledge_base/knowledge_base.json`. It does not automatically trust
model-predicted facts. All output directories must be fresh.

`models/potion-base-8M/` contains the preserved local model. Only if missing,
download it during development with `uv run python scripts/download_potion.py`.
Runtime extraction uses CPU and local resources without network access.

Outputs from `extract_forms.py` are `predictions.csv` and `traces.jsonl`; evaluation
and stage inspection commands also produce searchable HTML. Full extraction
reference performance is 994 correct out of 1,047 predicted assertions: 94.9%
precision and 41.4% recall. These are development measurements.

```sh
uv run python -m unittest discover -s tests -q
```

Implementation: `entity_linking/linker.py`, `values.py`, `relationships.py`,
`structural_rules.py`, `forms.py` and `knowledge_base.py`. The KB retains its
separate exact-number JSON serializer in `json_utils.py`.

## Reusing the library

Scripts are command-line entry points. Reusable code lives in `entity_linking/`;
library modules never import scripts.

| Module | Responsibility |
| --- | --- |
| `pipeline.py` | Load resources once, annotate text and extract forms from iterable batches |
| `linker.py` / `values.py` | Entity linking and typed literal recognition |
| `context.py` | Source spans and local wording windows |
| `candidates.py` | Typed, name and Boolean candidates, polarity and scope checks |
| `structural_rules.py` / `relationships.py` | Wording rules and lexicon-only similarity |
| `forms.py` | Compose candidates, bindings, forms and traces |
| `knowledge_base.py` | Validate, store and retrieve true-source dated assertions |
| `io.py` / `json_utils.py` | CSV, JSONL and exact-number JSON serialization |
| `evaluation.py` | Reference value normalization and extraction metrics |
| `report.py` | HTML rendering from supplied records |

```python
from pathlib import Path
from entity_linking import ExtractionPipeline
from entity_linking.io import read_rows

pipeline = ExtractionPipeline.from_paths(
    model_path=Path("models/potion-base-8M"),
    lexicon_path=Path("data/lexicon.json"),
    registry_path=Path("data/entity_registry.json"),
    batch_size=256,
)

annotations = pipeline.annotate("Hooly Gemini 1.5 supports 12 languages.")
rows, columns = read_rows(Path("data/practice.csv"))
predictions = list(pipeline.extract(rows))
# Reuse the same instance for subsequent requests without reloading the model.
```

`ExtractionPipeline(linker, ranker)` also accepts preconstructed components, for
testing or alternative ranking implementations. Extraction accepts generators as
well as lists. Existing CLI names and the original `extract_rows` function
signature are preserved. CPU/offline defaults are configured before model loading.
