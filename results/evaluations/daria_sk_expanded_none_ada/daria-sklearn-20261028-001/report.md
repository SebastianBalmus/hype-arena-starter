# daria_sk_expanded_none_ada

Run `daria-sklearn-20261028-001`, Daria `5943f61`, arena-eval-v1 / fixed_050.

Latest reader and temporal/HMM features with a declared sklearn decision pipeline. The native control retains Daria’s binary stacker; sklearn candidates learn all three labels. Selected method chooses only using frozen inner validation and refits on full outer train. All candidates were declared before this sweep. These are development splits already used by the team.

| Split | Candidate | Macro F1 | False P/R | Unsure F1 | Coverage | Errors/count | Risk | Brier | Native ΔF1 | Runtime / RAM |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| stratified | expanded_none_ada | 0.160 | 0.000/0.000 | 0.054 | 0.237 | 3/90 | 0.033 | 0.244 | -0.391 | shared features 10.0s + fit 0.21s / unmeasured |
| temporal | expanded_none_ada | 0.055 | 0.000/0.000 | 0.021 | 0.072 | 4/36 | 0.111 | 0.247 | -0.361 | shared features 9.7s + fit 0.17s / unmeasured |
| checker | expanded_none_ada | 0.222 | 0.000/0.000 | 0.043 | 0.408 | 6/155 | 0.039 | 0.247 | -0.276 | shared features 9.4s + fit 0.21s / unmeasured |

Generation, independent proof rate, grounded decision yield and constrained peak RAM are unavailable (null). Confusion matrices, class supports, reliability bins and full metrics are in each split’s `metrics.json`.

| Split | Always true F1 | Word LR F1 | Balanced word LR F1 | Δ vs balanced LR |
|---|---:|---:|---:|---:|
| stratified | 0.303 | 0.303 | 0.405 | -0.244 |
| temporal | 0.300 | 0.300 | 0.292 | -0.237 |
| checker | 0.307 | 0.334 | 0.384 | -0.162 |

Representative errors (diagnostic only; no post-test retuning):

| Split / ID | Gold | Predicted | Confidence | Reason |
|---|---|---|---:|---|
| stratified / p-1009 | false | unsure | None | low_confidence |
| stratified / p-1042 | false | unsure | None | low_confidence |
| stratified / p-1056 | false | unsure | None | low_confidence |
| stratified / p-1057 | false | unsure | None | low_confidence |
| stratified / p-1058 | false | unsure | None | low_confidence |

No independent sentence-level causal audit was performed; errors can arise from extraction, temporal assumptions or classifier generalization. Native HMM out-of-fold training hides labels while retaining permitted training forms; this is not a fully cross-fitted extraction pipeline. Gold-form training versus parsed inference remains a distribution mismatch. Timelines assume persistence and use noisy batch observations; scores supply no factual proof. Tiny unsure support and selection across many candidates increase variance. Native macOS timing is supplementary; no submission-readiness claim.

Reproduce with a new ID: `.venv/bin/python evaluation/run_daria_sklearn.py --run-id NEW-UNIQUE-ID --ref 5943f61d07c751f2a560125959ec1f8ba882bb0a`

<!-- artifact-links -->

Detailed classification artifacts: [stratified metrics, confusion matrix and reliability bins](fixed_050/stratified/metrics.json), [temporal metrics, confusion matrix and reliability bins](fixed_050/temporal/metrics.json), [checker metrics, confusion matrix and reliability bins](fixed_050/checker/metrics.json).
