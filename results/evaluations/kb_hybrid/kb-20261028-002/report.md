# kb_hybrid: arena-eval-v1

Same-date KB reasoning with the predeclared word_char_lr_balanced classifier for unresolved rows. Fallback classifier confidence is uncalibrated.

The confidence calibration is fixed before test inference: validation raw KB decisions are grouped by predicted true/false label, and Laplace-smoothed correctness is applied at the fixed 0.50 threshold. No test labels or forms reach the method process. These are development comparisons; the practice corpus and related templates have informed prior work.

| Split | Macro F1 | False precision / recall | Unsure F1 | Coverage | Errors / decided | Risk | Brier | Runtime | Peak RAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| stratified | 0.431 | 0.349 / 0.273 | 0.154 | 0.884 | 67 / 336 | 0.199 | 0.166 | 1.47s native | not measured |
| temporal | 0.299 | 0.176 / 0.070 | 0.000 | 0.842 | 97 / 420 | 0.231 | 0.172 | 1.42s native | not measured |
| checker | 0.361 | 0.259 / 0.143 | 0.042 | 0.892 | 59 / 339 | 0.174 | 0.149 | 1.39s native | not measured |

## Validation calibration counts

Laplace smoothing gives a 0.50 prior-only confidence for any predicted class with no raw validation KB decisions. Sparse calibration counts are descriptive point estimates; they do not establish reliability.

| Split | Raw KB validation T/F decisions | True count | False count | Calibration note |
| --- | ---: | ---: | ---: | --- |
| stratified | 0 | 0 | 0 | No raw KB validation decisions were observed; both classes use prior-only 0.50 confidence. |
| temporal | 0 | 0 | 0 | No raw KB validation decisions were observed; both classes use prior-only 0.50 confidence. |
| checker | 0 | 0 | 0 | No raw KB validation decisions were observed; both classes use prior-only 0.50 confidence. |

Confusion matrices and reliability bins are in each track’s `metrics.json`. Grounded yield and proof acceptance are not measured: traces are available, but no independent proof verifier was run. Generation is not implemented.

## Matched controls

Controls are reported only when fixed_050 artifacts from `text-variants-20261028-001` exist for the matching track. Their access and benchmark metadata are checked before comparison.

| Split | This method macro F1 | word_char_lr_balanced | tfidf_lr_balanced |
| --- | ---: | ---: | ---: |
| stratified | 0.431 | 0.431 | 0.405 |
| temporal | 0.299 | 0.299 | 0.292 |
| checker | 0.361 | 0.361 | 0.384 |

## Representative errors

The selected rows document emitted labels and confidence. A causal parsing/evidence diagnosis is not inferred from aggregate metrics; inspect the matching trace record for the executed rule and evidence references.

| ID | Gold | Predicted | Confidence | Reason |
| --- | --- | --- | ---: | --- |
| p-1009 | false | true | 0.721 | decided |
| p-1029 | unsure | true | 0.611 | decided |
| p-1030 | unsure | true | 0.709 | decided |
| p-1042 | false | true | 0.514 | decided |
| p-1056 | false | unsure | n/a | model_unsure |

## Limitations and reproduction

The same-date KB deliberately declines to carry observations across dates, so coverage is limited. The temporal track is a forward batch stress test. These are not untouched generalization test claims: the practice corpus and related templates have informed development. This is classification-only, not submission-ready. Native host timings are supplementary; constrained Linux resources, generation, frontend readiness and independent proof verification were not measured. No complete-practice form cache is available to inference.

```sh
.venv/bin/python evaluation/run_kb_experiment.py --run-id NEW-UNIQUE-ID
```

Test gold labels/forms remain in the evaluator and are joined by ID after subprocess inference. Validation gold is used only to calculate the documented confidence calibration. The operating threshold is fixed at 0.50; no threshold selection is performed.
