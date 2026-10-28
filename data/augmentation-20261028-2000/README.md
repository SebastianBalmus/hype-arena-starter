# 2,000 additional Hooly practice examples

These are synthetic semantic paraphrases derived only from `data/practice.csv`
and the names/attributes in `data/lexicon.json`. They expand the ways existing
claims are expressed; they do not expand the fictional world's facts.

| File | Contents |
| --- | --- |
| `practice_additional_2000.csv` | Exactly 2,000 new rows, IDs `syn-0001`–`syn-2000`. |
| `practice_combined_3897.csv` | The 1,897 original rows, followed by the 2,000 new rows. |
| `provenance.jsonl` | One source ID, fingerprint, original text and transformation note per new row. |
| `validation.json` | Counts, coverage, source/output hashes and automated validation results. |

Both CSVs use the original columns: `id,text,date,label,form`.
The additional rows contain **900 true, 850 false and 250 unsure** examples.
The combined file contains **2,481 true, 1,127 false and 289 unsure** examples.
Row order in the new dataset is shuffled deterministically with seed 42.

## What was preserved

- Every new row retains its source's date, label and complete structured form,
  including all atoms, operators, date precision, units, currency and billing period.
- Existing numerical values and named values are reworded, not perturbed or invented.
- False conjunctions remain conjunctions: their individual atoms are not separately
  assumed false. No false row supplies a newly guessed correct value.
- Unsure examples inherit `unsure` from the supplied data. Rephrasing preserves
  missing subjects, conflicting time anchors, odd commentary and other ambiguity
  where present. Adding a hedge to a known fact does not create an unsure label.
- True/false sources with clear text/form mismatches or unsafe scope are excluded
  by the label-specific builders. Unclear source annotations are not silently fixed.
- The original files and frozen evaluation splits are unchanged. No runtime,
  model, benchmark result, or packaged evidence store is modified.

The new texts are unique against both the original corpus and each other after
case folding and punctuation/whitespace normalization. Semantic overlap is
intentional: these are additional expressions of existing claims.

## Interpretation and evaluation limits

Labels are inherited through paraphrase, not certified by a new truth oracle.
Automated checks establish structural consistency and source provenance; they
do not independently establish fictional-world truth. The dataset preserves
uncertainty and any annotation limitations inherited from the source corpus.
In particular, the six variants of `p-1038` retain the original Dinesh/Richard
text/form mismatch and are flagged `inherited_text_form_mismatch` in provenance;
they are deliberately unclear examples, not repaired gold extraction examples.
The 2,000 rows are **not** the competition's 1,000 generated true claims.

These files are derived from the **full practice corpus**, including rows used
as held-out answers in the frozen v1 evaluations. Do not add this full batch to
training for those evaluations or randomly split the combined file and call
the resulting test independent. For split-specific augmentation, rebuild using
only permitted training ancestors and keep descendants with their source rows;
related paraphrase families also require grouping. `provenance.jsonl` exposes
the ancestors for this purpose. Synthetic rows are not independent world evidence.

## Reproduce

From the repository root, with Python 3 and no external dependencies:

```sh
python3 data/augmentation-20261028-2000/build_dataset.py
```

The builder runs the three reviewed generators in `work/`, checks the frozen
source hashes, rejects label/date/form drift and duplicate text, then writes the
two CSVs, provenance and validation report. It does not train or score a model.
