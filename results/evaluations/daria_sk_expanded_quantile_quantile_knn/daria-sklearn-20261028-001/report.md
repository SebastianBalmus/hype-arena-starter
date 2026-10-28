# daria_sk_expanded_quantile_quantile_knn

Run `daria-sklearn-20261028-001`, Daria `5943f61`, arena-eval-v1 / fixed_050.

Latest reader and temporal/HMM features with a declared sklearn decision pipeline. The native control retains Daria’s binary stacker; sklearn candidates learn all three labels. Selected method chooses only using frozen inner validation and refits on full outer train. All candidates were declared before this sweep. These are development splits already used by the team.

| Split | Candidate | Macro F1 | False P/R | Unsure F1 | Coverage | Errors/count | Risk | Brier | Native ΔF1 | Runtime / RAM |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| stratified | expanded_quantile_quantile_knn | 0.466 | 0.636/0.382 | 0.000 | 1.000 | 53/380 | 0.139 | 0.109 | -0.085 | shared features 10.0s + fit 0.08s / unmeasured |
| temporal | expanded_quantile_quantile_knn | 0.395 | 0.396/0.244 | 0.000 | 0.996 | 100/497 | 0.201 | 0.170 | -0.021 | shared features 9.7s + fit 0.07s / unmeasured |
| checker | expanded_quantile_quantile_knn | 0.427 | 0.484/0.306 | 0.000 | 0.976 | 55/371 | 0.148 | 0.109 | -0.070 | shared features 9.4s + fit 0.08s / unmeasured |

Generation, independent proof rate, grounded decision yield and constrained peak RAM are unavailable (null). Confusion matrices, class supports, reliability bins and full metrics are in each split’s `metrics.json`.

| Split | Always true F1 | Word LR F1 | Balanced word LR F1 | Δ vs balanced LR |
|---|---:|---:|---:|---:|
| stratified | 0.303 | 0.303 | 0.405 | +0.062 |
| temporal | 0.300 | 0.300 | 0.292 | +0.103 |
| checker | 0.307 | 0.334 | 0.384 | +0.044 |

Representative errors (diagnostic only; no post-test retuning):

| Split / ID | Gold | Predicted | Confidence | Reason |
|---|---|---|---:|---|
| stratified / p-1009 | false | true | 0.7159847679301585 | decided |
| stratified / p-1029 | unsure | true | 0.6312515511798809 | decided |
| stratified / p-1030 | unsure | true | 0.8125284035264585 | decided |
| stratified / p-1041 | unsure | true | 0.7264735304099614 | decided |
| stratified / p-1054 | unsure | true | 0.648878427769098 | decided |

No independent sentence-level causal audit was performed; errors can arise from extraction, temporal assumptions or classifier generalization. Native HMM out-of-fold training hides labels while retaining permitted training forms; this is not a fully cross-fitted extraction pipeline. Gold-form training versus parsed inference remains a distribution mismatch. Timelines assume persistence and use noisy batch observations; scores supply no factual proof. Tiny unsure support and selection across many candidates increase variance. Native macOS timing is supplementary; no submission-readiness claim.

Reproduce with a new ID: `.venv/bin/python evaluation/run_daria_sklearn.py --run-id NEW-UNIQUE-ID --ref 5943f61d07c751f2a560125959ec1f8ba882bb0a`

<!-- artifact-links -->

Detailed classification artifacts: [stratified metrics, confusion matrix and reliability bins](fixed_050/stratified/metrics.json), [temporal metrics, confusion matrix and reliability bins](fixed_050/temporal/metrics.json), [checker metrics, confusion matrix and reliability bins](fixed_050/checker/metrics.json).
