# word_char_lr_balanced: arena-eval-v1

Word plus character TF-IDF with class-balanced logistic regression.

Classification-only, text-only experiment. Fixed decision policy: `fixed_050`. All models were specified before looking at these results; no threshold or hyperparameter was selected using outer-test labels.

| Split | Macro F1 | False precision / recall | Unsure F1 | Coverage | Decided errors / count | Risk | Correctness Brier | Grounded yield | Generation / verified yield | Proof / rate | Runtime / peak RAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| stratified | 0.431 | 0.349 / 0.273 | 0.154 | 0.884 | 67 / 336 | 0.199 | 0.165 | n/a | not implemented / n/a | not implemented / n/a | 1.01s macOS / not measured |
| temporal | 0.299 | 0.176 / 0.070 | 0.000 | 0.842 | 97 / 420 | 0.231 | 0.172 | n/a | not implemented / n/a | not implemented / n/a | 0.99s macOS / not measured |
| checker | 0.361 | 0.259 / 0.143 | 0.042 | 0.892 | 59 / 339 | 0.174 | 0.148 | n/a | not implemented / n/a | not implemented / n/a | 0.95s macOS / not measured |

Confusion matrices and reliability bins are in the per-split [metrics files](fixed_050/stratified/metrics.json); the temporal and checker files are in their respective sibling directories. Native macOS timings are supplementary, not constrained Linux measurements.

## Matched controls

| Split | This macro F1 | Always true | Word LR | Balanced word LR | Difference vs balanced word LR |
| --- | ---: | ---: | ---: | ---: | ---: |
| stratified | 0.431 | 0.303 | 0.303 | 0.405 | +0.026 |
| temporal | 0.299 | 0.300 | 0.300 | 0.292 | +0.008 |
| checker | 0.361 | 0.307 | 0.334 | 0.384 | -0.022 |

## Representative errors

These classifiers do not parse assertions or consult evidence and dates. The table records the actual emitted reason; determining the factual cause would require a separate sentence-level review.

| ID | Gold | Predicted | Confidence | Decision reason |
| --- | --- | --- | ---: | --- |
| p-1009 | false | true | 0.721 | decided |
| p-1029 | unsure | true | 0.611 | decided |
| p-1030 | unsure | true | 0.709 | decided |
| p-1042 | false | true | 0.514 | decided |
| p-1056 | false | unsure | n/a | model_unsure |

## Limitations and reproduction

Practice rows have already informed development, and related sentence templates may cross partitions. These are development results, not pristine generalization estimates. The temporal track is a forward batch stress test. Confidence scores are uncalibrated. Models supply no evidence proofs, claim generation, Docker benchmark, or submission-ready system.

Inference inputs were produced with test labels blank and forms absent. The method is a separate process that reads the sanitized input; the host does not enforce a filesystem sandbox against other workspace files.

Run the following with a **new** run ID; original outputs are never overwritten:

```sh
.venv/bin/python evaluation/run_text_experiments.py --run-id text-variants-20261028-001
```

Development guidance: Kassis, Agarwal, He, Patel and Brueckner (2026), [Scientific Agent Skills](https://doi.org/10.48550/arXiv.2609.00065).
