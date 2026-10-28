# daria_sk_expanded_rbf_balanced

Run `daria-sklearn-20261028-001`, Daria `5943f61`, arena-eval-v1 / fixed_050.

Latest reader and temporal/HMM features with a declared sklearn decision pipeline. The native control retains Daria’s binary stacker; sklearn candidates learn all three labels. Selected method chooses only using frozen inner validation and refits on full outer train. All candidates were declared before this sweep. These are development splits already used by the team.

| Split | Candidate | Macro F1 | False P/R | Unsure F1 | Coverage | Errors/count | Risk | Brier | Native ΔF1 | Runtime / RAM |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| stratified | expanded_rbf_balanced | 0.484 | 0.574/0.491 | 0.000 | 0.997 | 53/379 | 0.140 | 0.108 | -0.068 | shared features 10.0s + fit 0.77s / unmeasured |
| temporal | expanded_rbf_balanced | 0.395 | 0.682/0.174 | 0.000 | 0.996 | 82/497 | 0.165 | 0.136 | -0.021 | shared features 9.7s + fit 0.58s / unmeasured |
| checker | expanded_rbf_balanced | 0.496 | 0.625/0.510 | 0.000 | 0.995 | 46/378 | 0.122 | 0.095 | -0.001 | shared features 9.4s + fit 0.79s / unmeasured |

Generation, independent proof rate, grounded decision yield and constrained peak RAM are unavailable (null). Confusion matrices, class supports, reliability bins and full metrics are in each split’s `metrics.json`.

| Split | Always true F1 | Word LR F1 | Balanced word LR F1 | Δ vs balanced LR |
|---|---:|---:|---:|---:|
| stratified | 0.303 | 0.303 | 0.405 | +0.079 |
| temporal | 0.300 | 0.300 | 0.292 | +0.103 |
| checker | 0.307 | 0.334 | 0.384 | +0.113 |

Representative errors (diagnostic only; no post-test retuning):

| Split / ID | Gold | Predicted | Confidence | Reason |
|---|---|---|---:|---|
| stratified / p-1009 | false | true | 0.8521664085815653 | decided |
| stratified / p-1029 | unsure | true | 0.85214227142239 | decided |
| stratified / p-1030 | unsure | false | 0.5106542205409578 | decided |
| stratified / p-1041 | unsure | true | 0.8465208473459894 | decided |
| stratified / p-1054 | unsure | true | 0.7280508729523073 | decided |

No independent sentence-level causal audit was performed; errors can arise from extraction, temporal assumptions or classifier generalization. Native HMM out-of-fold training hides labels while retaining permitted training forms; this is not a fully cross-fitted extraction pipeline. Gold-form training versus parsed inference remains a distribution mismatch. Timelines assume persistence and use noisy batch observations; scores supply no factual proof. Tiny unsure support and selection across many candidates increase variance. Native macOS timing is supplementary; no submission-readiness claim.

Reproduce with a new ID: `.venv/bin/python evaluation/run_daria_sklearn.py --run-id NEW-UNIQUE-ID --ref 5943f61d07c751f2a560125959ec1f8ba882bb0a`

<!-- artifact-links -->

Detailed classification artifacts: [stratified metrics, confusion matrix and reliability bins](fixed_050/stratified/metrics.json), [temporal metrics, confusion matrix and reliability bins](fixed_050/temporal/metrics.json), [checker metrics, confusion matrix and reliability bins](fixed_050/checker/metrics.json).
