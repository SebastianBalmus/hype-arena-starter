# daria_sk_native_robust_mlp

Run `daria-sklearn-20261028-001`, Daria `5943f61`, arena-eval-v1 / fixed_050.

Latest reader and temporal/HMM features with a declared sklearn decision pipeline. The native control retains Daria’s binary stacker; sklearn candidates learn all three labels. Selected method chooses only using frozen inner validation and refits on full outer train. All candidates were declared before this sweep. These are development splits already used by the team.

| Split | Candidate | Macro F1 | False P/R | Unsure F1 | Coverage | Errors/count | Risk | Brier | Native ΔF1 | Runtime / RAM |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| stratified | native_robust_mlp | 0.497 | 0.651/0.509 | 0.000 | 0.995 | 50/378 | 0.132 | 0.105 | -0.054 | shared features 10.0s + fit 0.12s / unmeasured |
| temporal | native_robust_mlp | 0.416 | 0.731/0.221 | 0.000 | 0.996 | 79/497 | 0.159 | 0.135 | +0.000 | shared features 9.7s + fit 0.11s / unmeasured |
| checker | native_robust_mlp | 0.506 | 0.609/0.571 | 0.000 | 0.995 | 45/378 | 0.119 | 0.096 | +0.009 | shared features 9.4s + fit 0.11s / unmeasured |

Generation, independent proof rate, grounded decision yield and constrained peak RAM are unavailable (null). Confusion matrices, class supports, reliability bins and full metrics are in each split’s `metrics.json`.

| Split | Always true F1 | Word LR F1 | Balanced word LR F1 | Δ vs balanced LR |
|---|---:|---:|---:|---:|
| stratified | 0.303 | 0.303 | 0.405 | +0.092 |
| temporal | 0.300 | 0.300 | 0.292 | +0.124 |
| checker | 0.307 | 0.334 | 0.384 | +0.123 |

Representative errors (diagnostic only; no post-test retuning):

| Split / ID | Gold | Predicted | Confidence | Reason |
|---|---|---|---:|---|
| stratified / p-1009 | false | true | 0.8563921043997056 | decided |
| stratified / p-1029 | unsure | true | 0.8563921043997056 | decided |
| stratified / p-1030 | unsure | true | 0.6347356531784166 | decided |
| stratified / p-1041 | unsure | true | 0.8563921043997056 | decided |
| stratified / p-1054 | unsure | true | 0.9445120972576002 | decided |

No independent sentence-level causal audit was performed; errors can arise from extraction, temporal assumptions or classifier generalization. Native HMM out-of-fold training hides labels while retaining permitted training forms; this is not a fully cross-fitted extraction pipeline. Gold-form training versus parsed inference remains a distribution mismatch. Timelines assume persistence and use noisy batch observations; scores supply no factual proof. Tiny unsure support and selection across many candidates increase variance. Native macOS timing is supplementary; no submission-readiness claim.

Reproduce with a new ID: `.venv/bin/python evaluation/run_daria_sklearn.py --run-id NEW-UNIQUE-ID --ref 5943f61d07c751f2a560125959ec1f8ba882bb0a`

<!-- artifact-links -->

Detailed classification artifacts: [stratified metrics, confusion matrix and reliability bins](fixed_050/stratified/metrics.json), [temporal metrics, confusion matrix and reliability bins](fixed_050/temporal/metrics.json), [checker metrics, confusion matrix and reliability bins](fixed_050/checker/metrics.json).
