# Simple text model variations

Run `text-variants-20261028-001` compares eight predeclared classifiers using the frozen `arena-eval-v1` splits, `batch_unlabelled_v1` access and the `fixed_050` decision policy. This is a classification-only development experiment; all models use sentence text as their only feature. They do not infer facts or determine whether a claim is true from dated evidence.

## Macro F1

The stratified split is the primary development comparison; checker is a compatibility control, and temporal is a forward batch stress test. Higher is better.

| Model | Change | Stratified | Temporal | Checker |
| --- | --- | ---: | ---: | ---: |
| `always_true` | Majority control | 0.303 | 0.300 | 0.307 |
| `tfidf_lr` | Word TF-IDF, logistic regression | 0.303 | 0.300 | 0.334 |
| `tfidf_lr_balanced` | Same, balanced classes | 0.405 | 0.292 | 0.384 |
| `char_lr_balanced` | Character n-grams, balanced LR | 0.376 | 0.296 | 0.370 |
| `word_char_lr_balanced` | Word and character n-grams, balanced LR | **0.431** | **0.299** | 0.361 |
| `word_char_lr_unweighted` | Word and character n-grams, unweighted LR | 0.304 | 0.300 | 0.347 |
| `word_char_complement_nb` | Word and character n-grams, complement naive Bayes | 0.315 | 0.295 | 0.305 |
| `word_lr_unsure_boost` | Balanced word LR with double weight for `unsure` | 0.399 | 0.289 | 0.382 |

The combined balanced model increases stratified macro F1 by **0.026** over balanced word LR, but falls by **0.022** on checker. On the temporal track it scores 0.299, almost the same as the 0.300 always-true control. There is no clear winner across all tracks.

## The tradeoff on the primary split

| Measure | Balanced word LR | Balanced word + character LR |
| --- | ---: | ---: |
| Accuracy | 65.0% | 71.8% |
| Macro F1 | 0.405 | 0.431 |
| False precision | 36.6% | 34.9% |
| False recall | 27.3% (15 / 55) | 27.3% (15 / 55) |
| Unsure F1 | 0.114 | 0.154 |
| Decided coverage | 78.9% (300 / 380) | 88.4% (336 / 380) |
| Wrong decided labels | 58 / 300 | 67 / 336 |

The combined model catches the **same number** of false claims on this split. Its higher macro F1 comes with higher coverage and more incorrect true/false decisions. There are only eight gold `unsure` rows in this split, so small counts can move that class's F1 substantially. Raw model probabilities are uncalibrated and should not be interpreted as independently established truth.

## What some mistakes suggest

These examples were inspected after the run, so they are useful for future development rather than a new untouched evaluation:

- `p-498` says a Hooly Docs file allows 75 editors in May 2026. The gold label is `false`, but the combined model predicts `true`. The same attribute is supported at 75 editors in training row `p-30` in August. A text model that does not use dates cannot distinguish the two observations reliably.
- `p-203` reports that the Sustainability Report came out on 25 August; it is false in September. Training row `p-688` says in July that the report was *scheduled* for that date, while `p-838` later gives a different publication date. A parser needs to keep planned and actual dates distinct and respect when the observation was made.
- `p-22` contains three assertions about Hooly Jules and is labelled `true`, but the combined model predicts `false`. A sentence-level label gives no indication of which assertion was responsible; parsing the individual assertions would make such cases easier to investigate.

The full metrics, confusion matrices, reliability bins, sanitized predictions, scored predictions, model description, and five example errors are in the individual run directories, such as [word + character LR](../results/evaluations/word_char_lr_balanced/text-variants-20261028-001/report.md) and [balanced word LR](../results/evaluations/tfidf_lr_balanced/text-variants-20261028-001/report.md). Their `run.json` files contain exact input/source hashes, row IDs, dependency versions and timings.

## Selecting a model using training data

The follow-up [inner-selection run](../results/evaluations/inner_selected/inner-selected-20261028-002/report.md) used the frozen inner fit and validation IDs to select among these eight models by validation macro F1. The selected model was then refitted on the complete outer training set and evaluated on the outer test set, with the same fixed 0.5 policy. The selected model can differ between tracks.

| Track | Selected using inner validation | Inner validation macro F1 | Outer test macro F1 |
| --- | --- | ---: | ---: |
| Stratified | Balanced word + character LR | 0.385 | 0.431 |
| Temporal | Balanced word LR | 0.360 | 0.292 |
| Checker | Balanced word + character LR | 0.389 | 0.361 |

The selected method still misses many false claims and scores poorly on the temporal track. Earlier exploration had already used the outer development splits; inner selection reduces direct test-label selection but does not make those splits pristine. The initial selection draft `inner-selected-20261028-001` is retained and marked ineligible because its validation scoring briefly wrote gold labels into the method input directory. The corrected run uses a separate ID and keeps those labels out of that directory.

## Reproduce or extend

From the repository root, choose a new run ID:

```sh
.venv/bin/python evaluation/run_text_experiments.py --run-id my-next-run-001
.venv/bin/python evaluation/inner_model_selection.py --run-id my-selection-001
```

The script refuses to overwrite an existing run. It verifies the three v1 hashes, materializes each split in original row order, removes forms and hides test labels, then runs each classifier in a separate process. This is a logical input boundary: the host does not enforce OS-level filesystem isolation. The evaluator separately joins predictions to gold by ID, checks output completeness and validity, and computes the metrics for all test rows. No thresholds were selected or adjusted on the outer test rows.

All variants were specified before this run's outer-test results were inspected. The corpus has already informed previous development, related templates can cross splits, and the tracks are not pristine unseen tests. Further iterations can overfit these development tracks. Follow [EVALUATION.md](../EVALUATION.md) before selecting models or adding another benchmark.

To try a variation on genuinely new blank-label rows, after `sample/sample.csv` is available:

```sh
.venv/bin/python experiments/predict.py --input sample/sample.csv --model word_char_lr_balanced --output outputs/sample-variant.csv
```

This fits using the labelled practice rows by default and outputs only blank input rows, in the six-column Arena label format. Choose another model from `python experiments/predict.py --help`, or pass `--train` for another labelled source. A blank input ID that is already in training is rejected to avoid accidentally evaluating on a known answer. This command does not generate claims.

These results do not establish a complete Arena submission. Claim generation and proof traces are not implemented; truth verification and constrained Linux/amd64 benchmarking were not run. Native macOS timings are supplementary.

Development guidance: Kassis, Agarwal, He, Patel and Brueckner (2026), [Scientific Agent Skills](https://doi.org/10.48550/arXiv.2609.00065).
