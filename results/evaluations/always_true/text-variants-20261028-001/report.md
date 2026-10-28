# always_true: arena-eval-v1

Majority-prior control, always true on these splits.

Classification-only, text-only experiment. Fixed decision policy: `fixed_050`. All models were specified before looking at these results; no threshold or hyperparameter was selected using outer-test labels.

| Split | Macro F1 | False precision / recall | Unsure F1 | Coverage | Decided errors / count | Risk | Correctness Brier | Grounded yield | Generation / verified yield | Proof / rate | Runtime / peak RAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| stratified | 0.303 | 0.000 / 0.000 | 0.000 | 1.000 | 63 / 380 | 0.166 | 0.138 | n/a | not implemented / n/a | not implemented / n/a | 0.81s macOS / not measured |
| temporal | 0.300 | 0.000 / 0.000 | 0.000 | 1.000 | 91 / 499 | 0.182 | 0.150 | n/a | not implemented / n/a | not implemented / n/a | 0.73s macOS / not measured |
| checker | 0.307 | 0.000 / 0.000 | 0.000 | 1.000 | 56 / 380 | 0.147 | 0.126 | n/a | not implemented / n/a | not implemented / n/a | 0.70s macOS / not measured |

Confusion matrices and reliability bins are in the per-split [metrics files](fixed_050/stratified/metrics.json); the temporal and checker files are in their respective sibling directories. Native macOS timings are supplementary, not constrained Linux measurements.

## Matched controls

| Split | This macro F1 | Always true | Word LR | Balanced word LR | Difference vs balanced word LR |
| --- | ---: | ---: | ---: | ---: | ---: |
| stratified | 0.303 | 0.303 | 0.303 | 0.405 | -0.102 |
| temporal | 0.300 | 0.300 | 0.300 | 0.292 | +0.008 |
| checker | 0.307 | 0.307 | 0.334 | 0.384 | -0.077 |

## Representative errors

These classifiers do not parse assertions or consult evidence and dates. The table records the actual emitted reason; determining the factual cause would require a separate sentence-level review.

| ID | Gold | Predicted | Confidence | Decision reason |
| --- | --- | --- | ---: | --- |
| p-1009 | false | true | 0.833 | decided |
| p-1029 | unsure | true | 0.833 | decided |
| p-1030 | unsure | true | 0.833 | decided |
| p-1041 | unsure | true | 0.833 | decided |
| p-1042 | false | true | 0.833 | decided |

## Limitations and reproduction

Practice rows have already informed development, and related sentence templates may cross partitions. These are development results, not pristine generalization estimates. The temporal track is a forward batch stress test. Confidence scores are uncalibrated. Models supply no evidence proofs, claim generation, Docker benchmark, or submission-ready system.

Inference inputs were produced with test labels blank and forms absent. The method is a separate process that reads the sanitized input; the host does not enforce a filesystem sandbox against other workspace files.

Run the following with a **new** run ID; original outputs are never overwritten:

```sh
.venv/bin/python evaluation/run_text_experiments.py --run-id text-variants-20261028-001
```

Development guidance: Kassis, Agarwal, He, Patel and Brueckner (2026), [Scientific Agent Skills](https://doi.org/10.48550/arXiv.2609.00065).
