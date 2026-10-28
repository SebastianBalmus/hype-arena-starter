# char_lr_balanced: arena-eval-v1

Character 3–5-gram TF-IDF with class-balanced logistic regression.

Classification-only, text-only experiment. Fixed decision policy: `fixed_050`. All models were specified before looking at these results; no threshold or hyperparameter was selected using outer-test labels.

| Split | Macro F1 | False precision / recall | Unsure F1 | Coverage | Decided errors / count | Risk | Correctness Brier | Grounded yield | Generation / verified yield | Proof / rate | Runtime / peak RAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| stratified | 0.376 | 0.351 / 0.236 | 0.082 | 0.763 | 59 / 290 | 0.203 | 0.189 | n/a | not implemented / n/a | not implemented / n/a | 0.95s macOS / not measured |
| temporal | 0.296 | 0.188 / 0.070 | 0.032 | 0.762 | 89 / 380 | 0.234 | 0.191 | n/a | not implemented / n/a | not implemented / n/a | 0.93s macOS / not measured |
| checker | 0.370 | 0.290 / 0.184 | 0.074 | 0.805 | 54 / 306 | 0.176 | 0.177 | n/a | not implemented / n/a | not implemented / n/a | 1.00s macOS / not measured |

Confusion matrices and reliability bins are in the per-split [metrics files](fixed_050/stratified/metrics.json); the temporal and checker files are in their respective sibling directories. Native macOS timings are supplementary, not constrained Linux measurements.

## Matched controls

| Split | This macro F1 | Always true | Word LR | Balanced word LR | Difference vs balanced word LR |
| --- | ---: | ---: | ---: | ---: | ---: |
| stratified | 0.376 | 0.303 | 0.303 | 0.405 | -0.028 |
| temporal | 0.296 | 0.300 | 0.300 | 0.292 | +0.004 |
| checker | 0.370 | 0.307 | 0.334 | 0.384 | -0.013 |

## Representative errors

These classifiers do not parse assertions or consult evidence and dates. The table records the actual emitted reason; determining the factual cause would require a separate sentence-level review.

| ID | Gold | Predicted | Confidence | Decision reason |
| --- | --- | --- | ---: | --- |
| p-1009 | false | true | 0.723 | decided |
| p-1030 | unsure | true | 0.588 | decided |
| p-1042 | false | true | 0.551 | decided |
| p-1056 | false | unsure | n/a | model_unsure |
| p-1057 | false | true | 0.518 | decided |

## Limitations and reproduction

Practice rows have already informed development, and related sentence templates may cross partitions. These are development results, not pristine generalization estimates. The temporal track is a forward batch stress test. Confidence scores are uncalibrated. Models supply no evidence proofs, claim generation, Docker benchmark, or submission-ready system.

Inference inputs were produced with test labels blank and forms absent. The method is a separate process that reads the sanitized input; the host does not enforce a filesystem sandbox against other workspace files.

Run the following with a **new** run ID; original outputs are never overwritten:

```sh
.venv/bin/python evaluation/run_text_experiments.py --run-id text-variants-20261028-001
```

Development guidance: Kassis, Agarwal, He, Patel and Brueckner (2026), [Scientific Agent Skills](https://doi.org/10.48550/arXiv.2609.00065).
