# Annotation inspection and corrections (v2)

The v1 output was not fully in order. This audit checked every saved entity/value
span and compared all detected typed values against the practice assertion forms,
then reviewed all discrepancies plus a deterministic sample of eight mentions per
type and one example of each of the 62 linked entities. Forms are used only by the
audit; the annotators still read text and the lexicon, not forms or truth labels.

## Corrected defects

| Defect | Correction / example |
| --- | --- |
| `1.5 in production` interpreted as inches | Context distinguishes version continuation from measurement; explicit inch/measurement contexts remain quantities |
| Missing CLI/release versions | `Hooly Cloud CLI 2.3` and `Release 1.4` now retain their version literals |
| `2.0 benchmark` interpreted as a count | Decimal version context after a linked entity takes precedence in this case |
| Plural noun corruption | `watches` becomes `watch`, rather than `watche` |
| Compound count units truncated | `4 data centre sites` and `3 storage tiers` retain `site` and `tier` |
| Billing period hidden by basis words | `€9 per terabyte each month` retains EUR 9, per month, basis terabyte |
| Identifier numbers treated as quantities | `SOC 2` / `SOC 2 Type II` remain identifiers, not counts |
| Idioms/pronouns treated as quantities | `net zero date` and `one of them` no longer invent numeric count units |
| Generic noun linked as product | Lower-case `photos` after a count is not linked to Hooly Photos; nine such candidates are retained as suppressed links with their reason |

The counted-noun suppression uses a conservative spelling/context heuristic; it
does not solve general entity ambiguity. Global entity IDs and the registry remain
unchanged. The original v1 artifacts are retained for comparison. Corrected
entity-only output is in `artifacts/entity_linking/v2/`; corrected combined output
is in `artifacts/typed_values/v2/`.

## Whole-dataset checks

- All 1,897 rows preserve original text and metadata.
- Original and normalized offsets are valid; entity/value replacement spans do not
  overlap; reversing replacements exactly reconstructs each original sentence.
- Corrected output has 1,532 linked entity mentions, including all 62 entities,
  and 2,032 typed value mentions: 1,036 quantities, 268 money, 232 durations,
  201 percentages, 167 dates and 128 versions.
- The practice forms contain 2,017 distinct numeric/date/version values when
  counted once per row. Independent matching, including unit equivalence and
  operators, finds 2,015 in the output. The two exceptions are the years in
  `Hooly I/O 2026` and `Hooly Cloud Next 2026`, deliberately protected inside
  entity names rather than asserted as separate calendar mentions.
- Fifteen additional literals not represented in the forms were reviewed against
  their source sentences. They are valid incidental mentions: four dateline
  dates, three durations and eight counts such as two tools or one person.
  Their presence does not make them facts about the nearest named entity.
- Twenty-five spans remain untyped or intentionally excluded: eighteen standalone
  or anaphoric numbers, five SOC identifiers and two net-zero idioms. For example,
  `Since 1 April` lacks a year, and `pay for three` lacks an explicit count noun.
- Forty-eight sampled typed mentions and 62 entity examples were inspected.
  Sample lists, every reference exception and every additional literal are saved
  in `artifacts/typed_values/audit_v2/`.

Reference-value coverage is **not NER accuracy or a held-out evaluation**. Practice
forms annotate assertions, not every literal or span. The fixes were developed
using this dataset, and no generalization score is claimed.

## Remaining limits

Some generic aliases still require contextual interpretation, and pronouns are
not linked. Count-noun and implicit-version recognition remain heuristic.
Negation scope, comparison suffixes such as `or later`, relative/incomplete dates,
and quoted or hypothetical statements need later semantic handling. Their source
wording is preserved; a typed mention is not proof of a positive assertion.

No subject/attribute binding or temporal graph has been created. The audit makes
the preprocessing more reliable for the next stage, without declaring every
future annotation correct.

## Reproduce

```sh
PYTHONDONTWRITEBYTECODE=1 python3 scripts/annotate_values.py \
  --output /tmp/hype-annotations-v2
PYTHONDONTWRITEBYTECODE=1 python3 scripts/audit_annotations.py \
  --annotations /tmp/hype-annotations-v2 --output /tmp/hype-audit-v2
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
```

All output directories must be fresh. Thirty-seven tests cover the implementation,
including the new regression cases and an independent audit failure test with a
deliberately corrupted span. The latest searchable report is
`artifacts/typed_values/v2/inspection.html`; it includes a suppressed-alias filter.
