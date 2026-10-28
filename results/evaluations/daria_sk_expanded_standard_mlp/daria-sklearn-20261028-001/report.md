# daria_sk_expanded_standard_mlp

Run `daria-sklearn-20261028-001`, Daria `5943f61`, arena-eval-v1 / fixed_050.

Latest reader and temporal/HMM features with a declared sklearn decision pipeline. The native control retains Daria’s binary stacker; sklearn candidates learn all three labels. Selected method chooses only using frozen inner validation and refits on full outer train. All candidates were declared before this sweep. These are development splits already used by the team.

| Split | Candidate | Macro F1 | False P/R | Unsure F1 | Coverage | Errors/count | Risk | Brier | Native ΔF1 | Runtime / RAM |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| stratified | expanded_standard_mlp | 0.528 | 0.583/0.509 | 0.125 | 0.979 | 51/372 | 0.137 | 0.122 | -0.024 | shared features 10.0s + fit 0.34s / unmeasured |
| temporal | expanded_standard_mlp | 0.341 | 0.223/0.558 | 0.000 | 0.986 | 203/492 | 0.413 | 0.249 | -0.075 | shared features 9.7s + fit 0.28s / unmeasured |
| checker | expanded_standard_mlp | 0.504 | 0.577/0.612 | 0.000 | 0.982 | 47/373 | 0.126 | 0.117 | +0.007 | shared features 9.4s + fit 0.33s / unmeasured |

Generation, independent proof rate, grounded decision yield and constrained peak RAM are unavailable (null). Confusion matrices, class supports, reliability bins and full metrics are in each split’s `metrics.json`.

| Split | Always true F1 | Word LR F1 | Balanced word LR F1 | Δ vs balanced LR |
|---|---:|---:|---:|---:|
| stratified | 0.303 | 0.303 | 0.405 | +0.123 |
| temporal | 0.300 | 0.300 | 0.292 | +0.049 |
| checker | 0.307 | 0.334 | 0.384 | +0.120 |

Representative errors (diagnostic only; no post-test retuning):

| Split / ID | Gold | Predicted | Confidence | Reason |
|---|---|---|---:|---|
| stratified / p-1009 | false | unsure | None | low_confidence |
| stratified / p-1030 | unsure | true | 0.942208457539074 | decided |
| stratified / p-1041 | unsure | true | 0.5065302133201639 | decided |
| stratified / p-1054 | unsure | true | 0.5937483451096717 | decided |
| stratified / p-1066 | false | true | 0.9797869291390904 | decided |

No independent sentence-level causal audit was performed; errors can arise from extraction, temporal assumptions or classifier generalization. Native HMM out-of-fold training hides labels while retaining permitted training forms; this is not a fully cross-fitted extraction pipeline. Gold-form training versus parsed inference remains a distribution mismatch. Timelines assume persistence and use noisy batch observations; scores supply no factual proof. Tiny unsure support and selection across many candidates increase variance. Native macOS timing is supplementary; no submission-readiness claim.

Reproduce with a new ID: `.venv/bin/python evaluation/run_daria_sklearn.py --run-id NEW-UNIQUE-ID --ref 5943f61d07c751f2a560125959ec1f8ba882bb0a`

<!-- artifact-links -->

Detailed classification artifacts: [stratified metrics, confusion matrix and reliability bins](fixed_050/stratified/metrics.json), [temporal metrics, confusion matrix and reliability bins](fixed_050/temporal/metrics.json), [checker metrics, confusion matrix and reliability bins](fixed_050/checker/metrics.json).
