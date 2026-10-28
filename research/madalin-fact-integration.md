# Madalin claim-generation features integrated with retrieval

Run: `madalin-facts-20261028-001`; `arena-eval-v1`; seed 42; `fixed_050`.
Teammate source pinned at `madalin@5fcec3f46c63c6ddf07f3c4178cc46be0c59e1c1`.

The integration reuses Madalin's value renderer and claim generator as a separate
classifier feature source. Dated rendered observations improve all three outer
development scores against the fixed raw+number-masked retrieval control. The
generator's composite claims reduce those scores. Inner validation selected the
original control for stratified and the rendered-observation method for temporal
and checker; those selections were recorded before any test fit.

## Results and model selection

| Method | Stratified macro F1 | Temporal macro F1 | Checker macro F1 |
| --- | ---: | ---: | ---: |
| Fixed raw+masked retrieval control | 0.468 | 0.332 | 0.362 |
| + Dated rendered observation features | 0.505 | 0.365 | 0.403 |
| + Generated composite claim features | 0.427 | 0.282 | 0.349 |
| Selected using inner validation in this experiment | 0.468 | 0.365 | 0.403 |

**The 0.505 stratified result is an exploratory fixed-candidate result, not the
validation-selected result.** Promoting that candidate based on the test result
would be test-driven selection. The integration preserves the original control
for the stratified selected output.

| Track | Control validation macro F1 | Rendered observations validation | Composite claims validation | Frozen selection |
| --- | ---: | ---: | ---: | --- |
| Stratified | 0.442 | 0.406 | 0.363 | Retrieval control |
| Temporal | 0.312 | 0.324 | 0.218 | Rendered observations |
| Checker | 0.360 | 0.388 | 0.347 | Rendered observations |

This experiment's candidate pool contains the fixed raw+masked retrieval model
and two additions to that same model. The earlier retrieval-selection experiment
also considered the plain word/character classifier; it selected that classifier
on temporal and checker. These candidate pools differ. We do not claim this new
selection is the best of every method developed so far.

| Track / method | Accuracy | False precision / recall | Unsure F1 | T/F errors / count |
| --- | ---: | ---: | ---: | ---: |
| Stratified control | 0.776 | 0.420 / 0.382 | 0.133 | 65 / 358 |
| Stratified rendered observations | 0.789 | 0.431 / 0.400 | 0.222 | 64 / 361 |
| Temporal control | 0.671 | 0.190 / 0.128 | 0.038 | 117 / 451 |
| Temporal rendered observations | 0.707 | 0.245 / 0.140 | 0.089 | 108 / 459 |
| Checker control | 0.708 | 0.221 / 0.306 | 0.000 | 88 / 357 |
| Checker rendered observations | 0.745 | 0.268 / 0.306 | 0.069 | 76 / 358 |

The small unsure class makes macro F1 sensitive to a few examples. All scores are
development measurements on previously explored frozen splits; they do not
establish generalization to independent wording families or new world evidence.
Full confusion matrices, reliability bins, coverage, risk, Brier, controls and
five representative failures are in the per-method reports and metrics.

## What was reused

Nine unmodified source files were imported into `arena/`: the value and lexicon
utilities, kinds, model, evidence intervals, schema/grammar checker and generator.
`integrations/madalin-source.json` records the upstream repository, commit and
every file hash. The upstream classifier, training pipeline, packaged full-gold
cache, server and Docker environment were not imported.

The existing sklearn/NumPy environment remains pinned. JSON Schema and its
dependencies were added at exact versions so the generator's local referee can
run. Every comparable evaluation first verifies the upstream source hashes and
the existing practice, lexicon and split-manifest hashes.

There is no independently supplied file of 1,000 new facts in this snapshot.
Madalin's generator derives candidates from evidence, with freshness and interval
assumptions. Its candidates therefore do not supply independent information about
the world. Reusing their structured wording as classifier features is different
from asserting them as new trusted evidence.

## Two feature modes

`madalin_observed_features` renders exact atomic annotations from permitted true
training rows into canonical text, using Madalin's typed renderer and attribute
names. Each retains its source observation date. Duplicate observations on the
same date are grouped with all supporting source IDs. The index is deterministically
capped at 1,000 records; it never pads a shortfall. Historical/plan/target scope
cues are retained as uncertainty flags.

`madalin_generated_features` invokes the pinned `enumerate_candidates` and
`select_claims` over the true training annotations. It includes comparisons,
historical changes, event claims, relations and current facts. Candidate premises
must map back to permitted true source rows. Generated records retain the
upstream assumptions; they remain unverified auxiliary features.

| Outer training pool | Rendered observations | Generated candidates | Generated type counts |
| --- | ---: | ---: | --- |
| Stratified | 1,000 | 906 | 202 fact; 59 compare; 260 change; 363 at_event; 22 relation |
| Temporal | 1,000 | 616 | 194 change; 406 at_event; 16 relation |
| Checker | 1,000 | 848 | 198 fact; 64 compare; 252 change; 318 at_event; 16 relation |

These counts describe auxiliary classifier catalogues, not a submission with
exactly 1,000 valid claims. In particular, temporal training has no sufficiently
recent current-value candidates under the generator's freshness rules.

Each mode appends 46 features to the existing 76 retrieval features: per-type
similarity, entity overlap, typed value agreement under conservative attribute
cues, source-date distance, scope uncertainty and temporal compatibility. Numeric
agreement and similarity are model inputs; they are not truth proofs. Candidates
are never added to trusted world evidence or used as positive synthetic training
targets. The classifier still learns its targets only from original labels.

## Provenance and temporal boundaries

Each of the classifier's five training folds builds its own catalogue from only
its reference rows. The query fold's forms and labels are excluded. A separate
full training catalogue is used for pending inference. Inner validation fits use
only the frozen fit IDs, and final test fits use only the outer training IDs.
The method receives a runtime-shaped CSV and a training-only forms file. Forms
must match the labelled runtime row's ID, text, date and label exactly.

Every candidate contains its original source IDs, ID/text/date SHA-256 fingerprints,
source observation dates, type, form, status and assumptions. Every candidate
neighbor in pending and training traces is audited against the appropriate scope
catalogue. All sources must be permitted true-labelled rows. Source admissibility
does not prove semantic entailment or independent truth.

Current generated facts/comparisons are tied to 29 October 2026. Event dates use
the same latest observed event-date value used by the generator, including when a
later announcement moves an event earlier. Relation compatibility features use
actual source-date intersections. Historical observations can match their own
source dates; they are not silently promoted to the as-of date. Interval
compaction, gap continuity and relation extensions remain explicitly unverified
generator assumptions.

The evaluator alone retains held-out answers and joins predictions afterward.
Execution uses sanitized subprocess inputs without an OS filesystem sandbox.
The smaller OOF pools and final full index have different reference distributions.

## Verification and reproduction

All **89 unit and integration tests** pass. They cover ordinary retrieval
regressions, fold-specific claim generation, nonempty numeric/name candidate
generation, missing-premise rejection, original date semantics, event rescheduling,
units/currency/period checks, form fingerprint mismatches and leaked pending-form
rejection. Injected audit failures verify that false sources, self-fold sources,
reused IDs and wrong catalogue retrieval are rejected.

An independent artifact audit also found matching source and exported hashes,
permitted positive sources and fingerprints, complete query coverage, correct
fold/catalogue references and exact selected-output copies. It checked 267,339
auxiliary neighbor citations across the candidate validation and test artifacts.
This is a provenance audit, not an independent truth verifier.

```sh
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests
.venv/bin/python evaluation/run_fact_integration.py --run-id NEW-UNIQUE-ID
```

Use a fresh run ID. See [integrations/README.md](../integrations/README.md) for
the CLI and compressed catalogue layout. Reports:

- [Rendered observations](../results/evaluations/madalin_observed_features/madalin-facts-20261028-001/report.md)
- [Generated claims](../results/evaluations/madalin_generated_features/madalin-facts-20261028-001/report.md)
- [Validation-selected output](../results/evaluations/madalin_facts_selected/madalin-facts-20261028-001/report.md)
- [Matched retrieval control](../results/evaluations/retrieval_raw_masked/madalin-facts-20261028-001/report.md)

Outer candidate process times were roughly 11–12 seconds for rendered observations
and 14–15 seconds for composite claims on the native host, versus about nine
seconds for the retrieval control. Constrained Linux runtime, peak RAM, independent
truth/proof verification, submission claim output and frontend integration were
not measured or implemented by this experiment.
