# First dated knowledge-base experiment

Run: `kb-20261028-002`, protocol `arena-eval-v1`, seed 42, decision policy
`fixed_050`. Implementation and artifact instructions are in
[knowledge_base/README.md](../knowledge_base/README.md).

The evidence store and numeric parser work, but this first version does **not**
improve macro F1. The hybrid emits the same labels as its text-model fallback.
The main observed limitation is language coverage, followed by the deliberately
strict requirement for evidence on the exact query date.

## Results

All metrics use every canonical held-out ID, including unsure gold rows.

| Track | Balanced word+character LR | KB only macro F1 | Hybrid macro F1 | Complete numeric parses | KB T/F decisions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stratified | 0.431 | 0.028 | 0.431 | 4 / 380 | 2 / 380 |
| Temporal | 0.299 | 0.007 | 0.299 | 5 / 499 | 0 / 499 |
| Checker | 0.361 | 0.014 | 0.361 | 7 / 380 | 1 / 380 |

The three KB decisions are correct against held-out labels. Three decisions do
not establish general reliability. Every unresolved KB-only row is emitted as
unsure, so its low macro F1 reflects very low coverage rather than many incorrect
T/F decisions. The hybrid uses `word_char_lr_balanced` on unresolved rows;
its predicted labels match that control exactly on all three tracks.

Zero temporal KB decisions are expected: every permitted observation date precedes
every test date, and this method does not carry values across dates. This track
cannot measure the usefulness of exact-date lookup alone on later dates.

The parser learned 272 templates from 298 eligible stratified training rows, but
recognized only four test sentences. Whole-sentence template matching generalizes
numeric values and aliases while retaining the surrounding wording. That makes
this version a conservative starting point with little paraphrase coverage.

The stratified store indexed 1,164 numeric atoms. That number counts usable
observations and negative single-atom constraints, not distinct world facts and
not evidence that covers every test date. Stored false compounds and excluded
historical or planned observations remain inspectable in the source records.

## Data boundaries and confidence

Each track rebuilds its parser and evidence from the manifest's permitted rows.
Forms from all training labels teach extraction. Only true rows add positive
observations; false single-atom numeric rows may exclude their own predicate.
False compound rows never negate all components, and unsure rows add no positive
facts. Units, operators, currency, billing period, source IDs and observation dates
are preserved.

First, the evaluator supplies inner fit rows and their forms to the method, with
inner validation rows having blank labels and no forms. It groups raw T/F
validation decisions by predicted class and calculates `(correct + 1)/(count + 2)`.
The final method refits its parser, store and optional text fallback on the full
outer training pool. It receives only train forms and blank-label test rows.
The operating threshold stays at 0.50 throughout.

**There were zero raw KB validation decisions in every track.** Consequently all
KB confidence scores are the explicitly declared prior-only 0.50; they are not
empirically calibrated. Hybrid fallback probabilities also remain uncalibrated.
Refitting changes the evidence available to the final method, so even a nonempty
validation bucket would need its sample size and reliability assessed.

The evaluator alone retains held-out labels/forms and joins predictions by ID.
The method rejects forms for pending IDs and requires an exact ID/text/date/label
match for every labelled source. Subprocess input separation is logical; there
is no OS-enforced filesystem sandbox. Existing practice-based development means
these are development benchmarks, not untouched generalization estimates.

## What was implemented and checked

- A lexicon-aware numeric template parser with longest-name matching and
  rejection of ambiguous, compound, negated and temporal wording.
- A dated numeric evidence store with Decimal interval reasoning, incompatible
  unit/period checks, positive/negative conflict handling and source fingerprints.
- A prediction CLI, split-scoped evaluator, strict KB and predeclared hybrid modes.
- Per-row execution traces, exported source records, coverage diagnostics,
  validation confidence metadata, input/source/asset hashes and native timings.
- 45 passing unit and integration tests covering existing baselines plus evidence
  dates, false compounds, interval boundaries, provenance, pending-form rejection,
  reused IDs, parser ambiguity, calibration abstention and hybrid routing.

Traces have not passed an independent proof verifier. Native host timings are
supplementary; constrained Linux runtime and memory have not been measured.
Generation, frontend and submission packaging are not implemented by this method.

## Next experiment

Improve extraction before adding a larger storage system. Keep deterministic
numeric parsing, but learn entity-conditioned attribute cues from permitted
training forms so new phrasing can map to an existing attribute. Measure extraction
on the frozen inner validation rows, including missing subjects, pronouns,
association errors and incomplete compounds. A prediction with incomplete
extraction must remain unresolved.

After language coverage improves, handle explicit effective dates and historical
scopes. Do not replace exact-date lookup with unconditional nearest-value or
last-value carry-forward: those would add unsupported persistence assumptions.

## Reproduce and inspect

```sh
.venv/bin/python -m unittest discover -s tests
.venv/bin/python evaluation/run_kb_experiment.py --run-id NEW-UNIQUE-ID
```

Use a fresh run ID. Detailed results are in
[the strict KB report](../results/evaluations/kb_same_date/kb-20261028-002/report.md)
and [the hybrid report](../results/evaluations/kb_hybrid/kb-20261028-002/report.md).
The earlier `kb-20261028-001` run is retained as a preliminary run. The final run
clarifies prior-only confidence metadata and adds artifact hashes; its predicted
labels are unchanged.
