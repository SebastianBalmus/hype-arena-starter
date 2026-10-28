# kb_same_date: arena-eval-v1

Same-date KB reasoning only: split-scoped training forms and lexicon build a conservative evidence store. No time carry-forward is used.

The confidence calibration is fixed before test inference: validation raw KB decisions are grouped by predicted true/false label, and Laplace-smoothed correctness is applied at the fixed 0.50 threshold. No test labels or forms reach the method process. These are development comparisons; the practice corpus and related templates have informed prior work.

| Split | Macro F1 | False precision / recall | Unsure F1 | Coverage | Errors / decided | Risk | Brier | Runtime | Peak RAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| stratified | 0.028 | 1.000 / 0.018 | 0.041 | 0.005 | 0 / 2 | 0.000 | 0.250 | 0.51s native | not measured |
| temporal | 0.007 | 0.000 / 0.000 | 0.020 | 0.000 | 0 / 0 | n/a | n/a | 0.50s native | not measured |
| checker | 0.014 | 0.000 / 0.000 | 0.036 | 0.003 | 0 / 1 | 0.000 | 0.250 | 0.51s native | not measured |

Confusion matrices and reliability bins are in each track’s `metrics.json`. Grounded yield and proof acceptance are not measured: traces are available, but no independent proof verifier was run. Generation is not implemented.

## Matched controls

Controls are reported only when fixed_050 artifacts from `text-variants-20261028-001` exist for the matching track. Their access and benchmark metadata are checked before comparison.

| Split | This method macro F1 | word_char_lr_balanced | tfidf_lr_balanced |
| --- | ---: | ---: | ---: |
| stratified | 0.028 | 0.431 | 0.405 |
| temporal | 0.007 | 0.299 | 0.292 |
| checker | 0.014 | 0.361 | 0.384 |

## Representative errors

The selected rows document emitted labels and confidence. A causal parsing/evidence diagnosis is not inferred from aggregate metrics; inspect the matching trace record for the executed rule and evidence references.

| ID | Gold | Predicted | Confidence | Reason |
| --- | --- | --- | ---: | --- |
| p-1009 | false | unsure | n/a | unsupported |
| p-1042 | false | unsure | n/a | unsupported |
| p-1056 | false | unsure | n/a | unsupported |
| p-1057 | false | unsure | n/a | unsupported |
| p-1058 | false | unsure | n/a | unsupported |

## Limitations and reproduction

The same-date KB deliberately declines to carry observations across dates, so coverage is limited. The temporal track is a forward batch stress test. This is classification-only, not submission-ready. Native host timings are supplementary; constrained Linux resources, generation, frontend readiness and independent proof verification were not measured. No complete-practice form cache is available to inference.

```sh
.venv/bin/python evaluation/run_kb_experiment.py --run-id kb-20261028-001
```

Test gold labels/forms remain in the evaluator and are joined by ID after subprocess inference. Validation gold is used only to calculate the documented confidence calibration. The operating threshold is fixed at 0.50; no threshold selection is performed.
