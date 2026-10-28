# daria_sk_expanded_none_histgb

Run `daria-sklearn-20261028-001`, Daria `5943f61`, arena-eval-v1 / fixed_050.

Latest reader and temporal/HMM features with a declared sklearn decision pipeline. The native control retains Daria’s binary stacker; sklearn candidates learn all three labels. Selected method chooses only using frozen inner validation and refits on full outer train. All candidates were declared before this sweep. These are development splits already used by the team.

| Split | Candidate | Macro F1 | False P/R | Unsure F1 | Coverage | Errors/count | Risk | Brier | Native ΔF1 | Runtime / RAM |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| stratified | expanded_none_histgb | 0.603 | 0.609/0.509 | 0.333 | 0.989 | 50/376 | 0.133 | 0.117 | +0.052 | shared features 10.0s + fit 1.01s / unmeasured |
| temporal | expanded_none_histgb | 0.428 | 0.525/0.244 | 0.071 | 0.954 | 84/476 | 0.176 | 0.175 | +0.012 | shared features 9.7s + fit 0.82s / unmeasured |
| checker | expanded_none_histgb | 0.502 | 0.533/0.653 | 0.000 | 1.000 | 52/380 | 0.137 | 0.116 | +0.005 | shared features 9.4s + fit 1.00s / unmeasured |

Generation, independent proof rate, grounded decision yield and constrained peak RAM are unavailable (null). Confusion matrices, class supports, reliability bins and full metrics are in each split’s `metrics.json`.

| Split | Always true F1 | Word LR F1 | Balanced word LR F1 | Δ vs balanced LR |
|---|---:|---:|---:|---:|
| stratified | 0.303 | 0.303 | 0.405 | +0.198 |
| temporal | 0.300 | 0.300 | 0.292 | +0.136 |
| checker | 0.307 | 0.334 | 0.384 | +0.119 |

Representative errors (diagnostic only; no post-test retuning):

| Split / ID | Gold | Predicted | Confidence | Reason |
|---|---|---|---:|---|
| stratified / p-1009 | false | true | 0.8527514588795511 | decided |
| stratified / p-1029 | unsure | true | 0.8511259074217616 | decided |
| stratified / p-1041 | unsure | true | 0.9285429091674142 | decided |
| stratified / p-1066 | false | true | 0.9971156175042258 | decided |
| stratified / p-1095 | false | true | 0.9998265486147767 | decided |

No independent sentence-level causal audit was performed; errors can arise from extraction, temporal assumptions or classifier generalization. Native HMM out-of-fold training hides labels while retaining permitted training forms; this is not a fully cross-fitted extraction pipeline. Gold-form training versus parsed inference remains a distribution mismatch. Timelines assume persistence and use noisy batch observations; scores supply no factual proof. Tiny unsure support and selection across many candidates increase variance. Native macOS timing is supplementary; no submission-readiness claim.

Reproduce with a new ID: `.venv/bin/python evaluation/run_daria_sklearn.py --run-id NEW-UNIQUE-ID --ref 5943f61d07c751f2a560125959ec1f8ba882bb0a`

<!-- artifact-links -->

Detailed classification artifacts: [stratified metrics, confusion matrix and reliability bins](fixed_050/stratified/metrics.json), [temporal metrics, confusion matrix and reliability bins](fixed_050/temporal/metrics.json), [checker metrics, confusion matrix and reliability bins](fixed_050/checker/metrics.json).
