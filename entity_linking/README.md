# Entity recognition and linking, version 1

The counts below describe the original v1 run. The later inspection found and
corrected counted-noun alias collisions; see `AUDIT.md` for the current v2 output.

This first stage replaces declared lexicon names/aliases with stable numeric
tokens, retaining the original text and exact mention offsets. It does not
extract assertions, determine truth, resolve pronouns, or construct a graph.
No labels/forms are used to recognize or link mentions, and there is no learned
model or additional dependency.

The next preprocessing stage is implemented separately; see `TYPED_VALUES.md`
for typed literal annotations and the combined entity/value inspection report.

## Representation

The persistent mapping is `data/entity_registry.json`. Initially, entities are
assigned positive integer IDs in canonical-name order. Subsequent builds retain
existing IDs and append new ones, even if the lexicon order changes. Removed
entities become inactive; their IDs are never reused. A canonical rename requires
an explicit migration rather than assuming it represents the same entity.

Each registry entry records its canonical name, aliases, attributes, event flag
and active status. IDs are categorical identifiers, not numeric features.

For this dataset:

| Entity | ID |
| --- | ---: |
| Hooly | 7 |
| Hooly AI Pro | 9 |
| Hooly Cloud Next 2026 | 18 |
| Hooly Gemini | 23 |

`Cloud Next`, `Hooly Cloud Next`, and `Hooly Cloud Next 2026` all map to
`<ENTITY:18>`. The sentence “Hooly Cloud Next closed after 3 days at Ringsend
Forum.” becomes “`<ENTITY:18> closed after 3 days at Ringsend Forum.`”

## Matching rules and limits

- Only canonical names and aliases declared in the lexicon are candidates.
- Matching accepts case and spacing variants and straight/curly apostrophes.
  It respects whole-word boundaries, with longest actual spans selected first.
- A product name suppresses shorter names inside it: Hooly AI Pro is one mention,
  without a second mention of Hooly.
- A name ending in a number cannot consume just the prefix of a decimal version.
  For example, Hooly Android 16.0 can link the listed platform Hooly Android while
  leaving 16.0 intact, rather than replacing Android 16 and leaving only .0.
- If a surface has multiple entity candidates, the whole span remains unchanged
  and candidate IDs are recorded. No shorter inner name is substituted instead.
- Pronouns, unlisted names and descriptive references remain unchanged. For
  instance, “the platform” is not assumed to mean a particular product.
- Every character outside selected entity spans is preserved, including numbers,
  units, operators, negation, dates and punctuation. A year inside a declared event
  name is part of that mention and replaced with it.
- Both original and normalized offsets are zero-based Python Unicode character
  offsets, with ends excluded. Each normalized span points at its exact token.
- Existing `<ENTITY:number>` tokens are protected from recognition inside them.

Matching a listed alias is not semantic disambiguation: a common product name
could appear in another sense. Likewise, zero ambiguous alias collisions does
not prove every match is contextually correct. Inspection is the purpose of this
version; no NER precision/recall is claimed without gold mention-span annotations.

## Dataset audit

The full 1,897-row practice CSV was processed, preserving every original column.

| Measure | Result |
| --- | ---: |
| Rows with at least one linked mention | 1,336 (70.43%) |
| Rows with no linked mention | 561 |
| Linked mentions | 1,541 |
| Distinct lexicon entities seen | 62 of 62 |
| Ambiguous alias collisions encountered | 0 |
| Unique original texts | 1,879 |
| Unique normalized texts | 1,878 |

One normalized group merged distinct source texts: “Gemini speaks 12 languages.”
and “Hooly Gemini speaks 12 languages.” both become
“`<ENTITY:23> speaks 12 languages.`” Most sentences remain distinct because their
other wording or values differ. This is linked-row coverage, not entity recall or
classification accuracy. Unmatched rows include pronouns, implied subjects and
sentences with no listed entity name; they are not automatically recognition errors.

## Inspect and reproduce

The saved artifacts are in `artifacts/entity_linking/v1/`:

- `inspection.html`: self-contained searchable report with original/normalized
  text, highlighted spans, entity names/IDs, and filters for unlinked/ambiguous
  rows and individual entities. Displays 200 matches at a time, with a load-more
  button. Open the HTML file directly in a browser; no server is needed.
- `normalized.csv`: all source columns plus normalized text, linked mentions and
  ambiguous mentions as JSON. Labels/forms are copied unchanged for inspection.
- `entity_registry.json`: snapshot of the registry used for that run.
- `entity_counts.csv`: mention and row counts for each active entity.
- `summary.json`: coverage counts and source dataset/lexicon SHA-256 hashes.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 scripts/normalize_entities.py \
  --output /tmp/hype-entity-linking-inspection
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
```

Use a fresh output directory; the command refuses to overwrite prior runs.
The default registry is reused so numeric IDs remain stable. Dataset and lexicon
sources are not modified. All paths can be overridden with CLI arguments.

The Python API is `EntityLinker(registry).normalize(text)`. It returns original
and normalized text, links with original/normalized offsets and matched lexicon
spellings, and unresolved ambiguous spans. Recognition has no dependency on row
IDs, dates, labels or structured assertion annotations.

Fourteen tests cover aliases, longest matches, boundaries, numeric versions,
ambiguity, stable IDs, source preservation, Unicode offsets, idempotence, and
HTML data escaping. Every dataset row was round-tripped using recorded offsets
to verify that original text can be reconstructed exactly.
