# Retrieval examples as learned classification features

Run: `retrieval-20261028-001`; protocol `arena-eval-v1`; seed 42; `fixed_050`.

Original-plus-number-masked retrieval improves the primary stratified development
benchmark from **0.431 to 0.468 macro F1**, an absolute gain of 0.037. False recall
improves from 15/55 to 21/55. The unchanged baseline is selected by inner
validation on temporal and checker, so the selected procedure keeps those
baseline predictions. These are development results on previously explored outer
splits, not an untouched generalization estimate.

## Fixed candidates and results

All candidates were declared before these test scores were computed. Each is
trained separately on the same frozen permitted rows. Test metrics include all
canonical IDs and all three target classes, after the fixed confidence policy.

| Candidate | Stratified test macro F1 | Temporal test macro F1 | Checker test macro F1 |
| --- | ---: | ---: | ---: |
| Unchanged balanced word/character LR | 0.431 | 0.299 | 0.361 |
| Text + original-text retrieval | 0.429 | 0.340 | 0.377 |
| Text + original and number-masked retrieval | **0.468** | 0.332 | 0.362 |
| Candidate selected using inner validation | **0.468** | 0.299 | 0.361 |

The raw retrieval candidate scores higher on temporal and checker test rows than
the unchanged control, but its validation score was lower. Selecting that
candidate for those tracks using these test scores would be test-driven model
selection. The recorded selection retains the baseline.

| Track | Baseline validation macro F1 | Raw retrieval validation | Raw + masked validation | Frozen selection |
| --- | ---: | ---: | ---: | --- |
| Stratified | 0.385 | 0.400 | **0.442** | Raw + masked retrieval |
| Temporal | **0.355** | 0.302 | 0.312 | Baseline |
| Checker | **0.389** | 0.353 | 0.360 | Baseline |

On the stratified test rows, the selected retrieval model changes 101 labels
relative to the baseline. Among changed rows, 56 new labels are correct and 34
old labels were correct, a net gain of 22 correct classifications.

| Stratified metric | Baseline | Raw + masked retrieval |
| --- | ---: | ---: |
| Accuracy | 71.8% | 77.6% |
| False precision | 34.9% | 42.0% |
| False recall | 27.3% (15/55) | 38.2% (21/55) |
| False F1 | 0.306 | 0.400 |
| Unsure F1 | 0.154 | 0.133 |
| T/F coverage | 88.4% | 94.2% |
| Incorrect T/F decisions | 67 / 336 | 65 / 358 |

The unsure class has only eight stratified test cases. Its recall decreases from
4/8 to 2/8, so this change does not improve every class. Sparse unsure validation
and test samples limit confidence in small macro F1 differences across tracks.

## How retrieval is used

The text representation is unchanged: word unigrams/bigrams with 20,000 features
and character 3–5-grams with 30,000 features, both using sublinear TF-IDF. Balanced
logistic regression uses C=1 and the fixed 0.50 policy.

Each retrieval view contributes 38 additional numeric features, standardized
using the training out-of-fold feature matrix. Features summarize the five most
similar examples overall and by reference class. They include cosine similarity,
label composition, date gaps, lexicon entity matches, numeric-set equality and
overlap, currency and billing-period matches. The raw+masked candidate supplies
76 features. It retrieves a second time after replacing numeric surfaces with a
placeholder, helping similar wording remain retrievable when the asserted number
changes.

This is a statistical classifier. It does not parse every assertion or prove
that any retrieved sentence supports the query. In particular, numeric overlap
is an untyped heuristic, and date proximity does not establish persistence.
False/unsure references supply labelled comparison features; their asserted
values never become positive world facts. The factual knowledge-base engine is
unchanged.

## Training and evaluation boundary

1. For inner validation, fit only on the manifest's fit IDs. Split those rows
   into five stratified folds with seed 42.
2. For each fold, fit a retrieval vocabulary and index using only the other four
   folds. Produce query features with the held-out fold's labels blank. No row
   can retrieve itself or any other member of its query fold.
3. Fit the numeric scaler on these out-of-fold features. Append the original text
   features and fit the classifier on permitted training labels.
4. Build a full fit-only reference index for inner validation queries. Compare
   macro F1 for the baseline and both retrieval candidates. Freeze the winner
   before any outer-test fitting or scoring.
5. Refit every candidate on the complete outer training pool, including rebuilding
   its internal five-fold training features. Infer on blank-label test rows using
   the full outer-training reference index. Reuse the selected candidate's saved
   predictions for `retrieval_selected`.

All method inputs have exactly `id,text,date,label`. The lexicon is public input;
practice forms are unused. The evaluator keeps gold labels/forms outside method
inputs and scores outputs afterward. Method subprocesses have logical separation,
without an OS-enforced filesystem sandbox.

The full inference index is larger than each out-of-fold training index, and its
vocabulary/IDF differs. That training/inference distribution difference remains
a limitation. Frozen splits can also share related wording and facts; this run
does not establish generalization to entirely new template families.

## Verification and capabilities

All 66 unit and integration tests pass. They include numeric/currency/period
features, canonical aliases, self-reference rejection, query-label invariance,
complete out-of-fold trace coverage, permitted reference membership, strict
runtime input shape and CLI output checks.

The evaluator also audits every exposed global and per-class neighbor against
the permitted reference IDs, labels and dates; verifies that every training row
is held out exactly once; and checks complete pending predictions/traces. Source,
input and exported artifact hashes are retained in each `run.json`. The freshly
rerun baseline reproduces the earlier text experiment's scores.

Class probabilities are uncalibrated. Neighbor traces are not independent truth
proofs. Claim generation, frontend and submission packaging are not implemented
by this method. Native model-process times for outer stratified runs were about
1.0 seconds for the control, 5.9 seconds for raw retrieval and 9.1 seconds for
raw+masked retrieval. Those measurements do not replace the constrained offline
Linux benchmark or establish 20,000-row resource usage.

## Reproduce and inspect

```sh
.venv/bin/python -m unittest discover -s tests
.venv/bin/python evaluation/run_retrieval_experiments.py --run-id NEW-UNIQUE-ID
```

Use a fresh run ID. See [retrieval/README.md](../retrieval/README.md) for the method
CLI and artifact layout. Saved reports:

- [Original-text retrieval](../results/evaluations/retrieval_raw/retrieval-20261028-001/report.md)
- [Original + masked retrieval](../results/evaluations/retrieval_raw_masked/retrieval-20261028-001/report.md)
- [Validation-selected method](../results/evaluations/retrieval_selected/retrieval-20261028-001/report.md)
- [Fresh baseline control](../results/evaluations/word_char_lr_balanced/retrieval-20261028-001/report.md)

The next useful comparison would isolate retrieval feature groups and test their
weight on inner validation, while keeping the existing fixed test outputs as
development records. Explicit assertion extraction and temporal scopes remain a
separate route toward evidence-based claim generation.
