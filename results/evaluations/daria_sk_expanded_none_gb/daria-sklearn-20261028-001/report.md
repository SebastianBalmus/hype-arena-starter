# daria_sk_expanded_none_gb

Run `daria-sklearn-20261028-001`, Daria `5943f61`, arena-eval-v1 / fixed_050.

Latest reader and temporal/HMM features with a declared sklearn decision pipeline. The native control retains Daria’s binary stacker; sklearn candidates learn all three labels. Selected method chooses only using frozen inner validation and refits on full outer train. All candidates were declared before this sweep. These are development splits already used by the team.

| Split | Candidate | Macro F1 | False P/R | Unsure F1 | Coverage | Errors/count | Risk | Brier | Native ΔF1 | Runtime / RAM |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| stratified | expanded_none_gb | 0.616 | 0.643/0.491 | 0.364 | 0.992 | 48/377 | 0.127 | 0.106 | +0.064 | shared features 10.0s + fit 0.78s / unmeasured |
| temporal | expanded_none_gb | 0.425 | 0.800/0.233 | 0.000 | 0.998 | 75/498 | 0.151 | 0.131 | +0.009 | shared features 9.7s + fit 0.63s / unmeasured |
| checker | expanded_none_gb | 0.500 | 0.583/0.571 | 0.000 | 0.987 | 47/375 | 0.125 | 0.098 | +0.002 | shared features 9.4s + fit 0.76s / unmeasured |

Generation, independent proof rate, grounded decision yield and constrained peak RAM are unavailable (null). Confusion matrices, class supports, reliability bins and full metrics are in each split’s `metrics.json`.

| Split | Always true F1 | Word LR F1 | Balanced word LR F1 | Δ vs balanced LR |
|---|---:|---:|---:|---:|
| stratified | 0.303 | 0.303 | 0.405 | +0.211 |
| temporal | 0.300 | 0.300 | 0.292 | +0.133 |
| checker | 0.307 | 0.334 | 0.384 | +0.116 |

Representative errors (diagnostic only; no post-test retuning):

| Split / ID | Gold | Predicted | Confidence | Reason |
|---|---|---|---:|---|
| stratified / p-1009 | false | true | 0.9176462216033493 | decided |
| stratified / p-1029 | unsure | true | 0.9093804016654518 | decided |
| stratified / p-1041 | unsure | true | 0.9176462216033493 | decided |
| stratified / p-1066 | false | true | 0.9297370150963273 | decided |
| stratified / p-1095 | false | true | 0.9846404211570332 | decided |

No independent sentence-level causal audit was performed; errors can arise from extraction, temporal assumptions or classifier generalization. Native HMM out-of-fold training hides labels while retaining permitted training forms; this is not a fully cross-fitted extraction pipeline. Gold-form training versus parsed inference remains a distribution mismatch. Timelines assume persistence and use noisy batch observations; scores supply no factual proof. Tiny unsure support and selection across many candidates increase variance. Native macOS timing is supplementary; no submission-readiness claim.

Reproduce with a new ID: `.venv/bin/python evaluation/run_daria_sklearn.py --run-id NEW-UNIQUE-ID --ref 5943f61d07c751f2a560125959ec1f8ba882bb0a`

<!-- artifact-links -->

Detailed classification artifacts: [stratified metrics, confusion matrix and reliability bins](fixed_050/stratified/metrics.json), [temporal metrics, confusion matrix and reliability bins](fixed_050/temporal/metrics.json), [checker metrics, confusion matrix and reliability bins](fixed_050/checker/metrics.json).
