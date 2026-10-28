# retrieval_raw_masked: retrieval-feature experiment

Frozen arena-eval-v1 development splits; fixed_050; seed 42. All candidates were declared before their outer-test scores were computed. Prior experiments have already informed development on these splits.

| Track | Validation macro F1 | Test macro F1 | False precision / recall | Unsure F1 | Coverage | T/F errors / decisions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| stratified | 0.442 | 0.468 | 0.420 / 0.382 | 0.133 | 0.942 | 65 / 358 |
| temporal | 0.312 | 0.332 | 0.190 / 0.128 | 0.038 | 0.904 | 117 / 451 |
| checker | 0.360 | 0.362 | 0.221 / 0.306 | 0.000 | 0.939 | 88 / 357 |

## Method and boundaries

The unchanged control uses balanced word/character TF-IDF logistic regression. Retrieval candidates append standardized neighbor features learned from five stratified folds within the permitted training pool. Each training query receives references and vocabulary fitted on its other four folds only. A full training index is used only for pending-row inference. Query labels and forms are never retrieval inputs. Numeric agreement and similarity are heuristics; neither establishes truth.

The raw candidate retrieves on original wording. The raw_masked candidate adds a second retrieval view with numeric surfaces replaced by a placeholder. Nearest-neighbor labels, date gaps, entity overlap, numeric overlap and currency/period matches are supplied to balanced logistic regression at C=1. Class probabilities are uncalibrated; the fixed 0.50 gate applies after prediction.

Inner fit/validation predictions select among the unchanged control and both retrieval candidates. Selection is recorded before test inference. Each candidate is then refitted on the complete outer training pool. The selected method reuses that candidate output rather than fitting it twice. Validation/test text are only pending queries; no pending labels or forms enter training.

Inputs are sanitized and method subprocesses are separate. Filesystem access is not OS-enforced. Related templates may cross the frozen partitions. Generation, accepted proofs, frontend and constrained runtime are unavailable.

## Validation selection

| Track | Selected candidate |
| --- | --- |
| stratified | retrieval_raw_masked |
| temporal | word_char_lr_balanced |
| checker | word_char_lr_balanced |

Neighbor traces, fold membership and feature names are stored with the retrieval candidates. The evaluator audits every exposed neighbor against the permitted reference IDs and labels. The selected method references those candidate artifacts through selection.json.

## Reproduce

```sh
.venv/bin/python evaluation/run_retrieval_experiments.py --run-id NEW-UNIQUE-ID
```
