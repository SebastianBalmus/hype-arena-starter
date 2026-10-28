# Inner-selected text classifier

Run: `inner-selected-20261028-002`; protocol `arena-eval-v1`; policy `fixed_050`.

Models were compared using frozen inner validation IDs. Macro F1 selected the model, with ties broken by the declared MODELS ordering. The selected model was refit on each full outer training pool and evaluated once on its test IDs. Test labels were not sent to the method process.

The practice corpus and outer development splits have already informed prior experiments. Inner selection reduces direct test-label selection, but these outer scores are not pristine generalization estimates. The temporal track is a forward batch stress test. Confidence is uncalibrated. Generation and proofs are not implemented; constrained resources were not measured.

| Split | Selected model | Validation macro F1 | Outer macro F1 | False P/R | Unsure F1 | Coverage | Errors/count | Risk | Brier |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| stratified | word_char_lr_balanced | 0.385 | 0.431 | 0.349/0.273 | 0.154 | 0.884 | 67/336 | 0.19940476190476192 | 0.1651681676358963 |
| temporal | tfidf_lr_balanced | 0.360 | 0.292 | 0.172/0.058 | 0.016 | 0.756 | 80/377 | 0.21220159151193635 | 0.18591168772245253 |
| checker | word_char_lr_balanced | 0.389 | 0.361 | 0.259/0.143 | 0.042 | 0.892 | 59/339 | 0.17404129793510326 | 0.1483487948071604 |

## Matched controls

The controls below use the same benchmark, split and fixed_050 policy. They trained on the complete outer training pool without inner model selection, so they are useful context but not fully paired with this selected-fit procedure.

| Split | Selected model | Always true | Word LR | Balanced word LR |
|---|---:|---:|---:|---:|
| stratified | 0.431 | 0.303 | 0.303 | 0.405 |
| temporal | 0.292 | 0.300 | 0.300 | 0.292 |
| checker | 0.361 | 0.307 | 0.334 | 0.384 |

Confusion matrices and reliability bins are included in each split's `metrics.json`. Grounded yield, verified generation yield, proof rate and constrained peak RAM are not measured.

## Representative stratified errors

The text classifiers do not implement parsing, evidence or temporal rules; the emitted reason is shown rather than a guessed causal explanation.

| ID | Gold | Predicted | Confidence | Reason |
|---|---|---|---:|---|
| p-1009 | false | true | 0.7213517870129345 | decided |
| p-1029 | unsure | true | 0.6110667056367379 | decided |
| p-1030 | unsure | true | 0.7094806257129886 | decided |
| p-1042 | false | true | 0.5144402549402707 | decided |
| p-1056 | false | unsure | n/a | model_unsure |

## Reproduction

```sh
.venv/bin/python evaluation/inner_model_selection.py --run-id NEW-UNIQUE-ID
```

Method code hashes and exact frozen ID arrays are recorded in `run.json` and `validation_selection.json`. The method receives sanitized runtime-shaped CSV in a separate process; filesystem isolation is logical, not OS-enforced.

Development guidance: Kassis, Agarwal, He, Patel and Brueckner (2026), [Scientific Agent Skills](https://doi.org/10.48550/arXiv.2609.00065).
