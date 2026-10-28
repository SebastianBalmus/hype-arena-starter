# Typed-value annotation, version 1

This documents the initial v1 snapshot. The latest corrected output and inspection
findings are in `AUDIT.md` and `artifacts/typed_values/v2/`. The CLI now generates
annotation version 2; original v1 artifacts remain available for comparison.

This extends lexicon linking with text-only recognition of money, quantities,
percentages, dates, durations and versions. These are **literal mentions**, not
subject/attribute assertions or verified facts. No labels or practice forms are
used to create annotations. This stage is rule-based; general NER for new people,
organizations and locations has not been added.

## Representation

The original text and the entity-only normalized text are retained. A combined
view replaces entity spans and typed-value spans with categorical tokens:

```text
Hooly AI Pro costs €49 per month, and every plan is free for two weeks.
<ENTITY:9> costs <MONEY:1>, and every plan is free for <DURATION:1>.
```

For the money token, the sidecar annotation records kind money, value 49,
currency EUR, period month, source text, original and normalized offsets, rule,
operator and qualifier flags. Duration records value 2, unit week. Entity IDs
remain globally stable in the registry. Value IDs restart at one **per row and
kind**: `<MONEY:1>` in another sentence can refer to a different price. Always use
the row's sidecar annotation; matching token text does not establish equal values.

Values are parsed with Decimal arithmetic and emitted as exact JSON numbers in
the CSV/JSONL sidecars. The HTML report displays decimals as strings to avoid
browser floating-point rounding. Canonical unit spellings are recorded, but
values are not converted between units or currencies in this stage.

## Rules and limits

- Entity spans own their embedded years/product numbers, which are not retagged.
- Money retains amount, currency and billing period; a missing period is recorded
  as null. An explicitly written user/seat basis is retained separately. Periods
  included in money spans are not duplicated as durations.
- Quantities recognize physical units and a heuristic counted noun, including
  examples such as 4 engineers, 40 languages, 1.2 million users and 128 GB.
  Count-noun binding is heuristic and may still confuse other words with nouns.
- Dates retain year, month or day precision; invalid calendar dates are rejected.
  Month names without a year and relative phrases such as today are not resolved.
  A page's date is metadata, not automatically an effective date for a value.
- Explicit versions and decimal versions following linked entities are recognized.
  The latter uses contextual heuristics and does not establish that an entity
  has a particular version attribute. Version precision is kept as written.
- Operators such as over, under, up to and at least are recorded, as is a trailing
  numeric plus sign. Supported prefix comparison words may be inside the replaced
  span; their meaning remains in the annotation. This is not a complete operator
  grammar; unsupported suffixes and temporal wording remain visible in the text.
- Approximation and preceding-clause negation are flagged. Negated values are
  retained as mentions, not treated as positive evidence. The negation scope
  flag is a lexical heuristic, not full semantic scope resolution.
- Ranges/choices, ambiguous numeric spellings, unsupported standalone numbers and
  invalid dates are left for review, not converted into guessed exact values.
- Article durations are context-limited: under a second is supported; requests
  a day is not automatically interpreted as a one-day duration.
- The recognizer keeps no learned parameters and creates no new entity identities,
  relations, world facts, or classifications.

Existing tokens are protected. Every source span has zero-based Python Unicode
character offsets with the end excluded. All original characters can be restored
exactly from the combined normalized text and span annotations.

## Full-dataset inspection

| Type | Mentions |
| --- | ---: |
| Quantity | 1,040 |
| Money | 268 |
| Duration | 232 |
| Percent | 201 |
| Date | 167 |
| Version | 120 |
| **Total** | **2,028** |

There are typed values in 1,618 of 1,897 rows. Combining them with the unchanged
1,541 entity links gives at least one annotation in 1,836 rows (96.78%); 61 rows
have neither a linked entity nor a typed value. Twenty number mentions remain
unresolved. Thirty-eight values have recognized comparison operators and four
have a negation-prefix flag. No approximate values were encountered in this run.

These are occurrence/coverage counts, not precision, recall or truth scores.
Incidental values can be correctly detected without being part of an annotated
assertion. The recognizer has not been evaluated against gold mention spans.
Some count-noun and implicit-version assignments need manual inspection before
they can support later assertion/graph construction.

Entity-only normalization yielded 1,878 distinct texts; the combined view yields
1,831. This reduction shows some structural convergence but does not demonstrate
improved downstream classification.

## Inspect and reproduce

`artifacts/typed_values/v1/` contains:

- `inspection.html`: searchable original/tagged text with green entity spans,
  blue typed values and structured details. Filters include value kind, unresolved
  numbers, no entities, and no resolved annotations at all. Hover for the complete
  annotation; searching a rule name such as counted_noun_heuristic filters its rows.
- `annotated.csv`: all original columns, entity-only and combined normalized text,
  entity links, typed values and unresolved numeric mentions.
- `annotations.jsonl`: text, spans and structured annotations per row, without
  source labels/forms.
- `examples.json`: first examples of each type.
- `unresolved.csv`: unresolved numeric text and reasons.
- `summary.json`: counts and source/code hashes.
- `entity_registry.json`: snapshot of the stable entity registry.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 scripts/annotate_values.py \
  --output /tmp/hype-typed-values-inspection
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
```

Use a fresh output directory. The existing `data/entity_registry.json` from the
entity-linking stage supplies stable IDs. No additional dependencies are needed.
The Python API is `annotate(text, linker)` from `entity_linking.values`.

Thirty tests cover the combined implementation, including full-dataset offset
round-trips, label/form independence, exact numbers, qualifiers, units, dates,
version boundaries, token protection and report JavaScript syntax. Original data,
lexicon and registry files remain unchanged by this stage.
