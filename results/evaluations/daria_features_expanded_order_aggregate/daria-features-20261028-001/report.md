# daria_features_expanded_order_aggregate

Feature representation experiment using fixed, previously CV-tuned GradientBoostingClassifier parameters per track. Full feature resources are rebuilt per fold; learned vectorizers, selection and reductions fit on training only. The selected method freezes its representation using mean three-fold CV macro F1 inside shared fit_ids. Shared validation is diagnostic only, and final assets use full outer train_ids. All outputs use fixed_050.

| Track | Feature candidate | CV F1 | Shared validation F1 | Test F1 | Δ vs base | False P/R | Unsure F1 | Coverage | Errors/count | Risk | Brier |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stratified | expanded_order_aggregate | 0.448 | 0.542 | 0.660 | +0.011 | 0.643/0.491 | 0.500 | 0.989 | 48/376 | 0.128 | 0.103 |
| temporal | expanded_order_aggregate | 0.380 | 0.378 | 0.427 | -0.004 | 0.622/0.267 | 0.000 | 1.000 | 81/499 | 0.162 | 0.154 |
| checker | expanded_order_aggregate | 0.480 | 0.501 | 0.499 | -0.001 | 0.583/0.571 | 0.000 | 0.984 | 47/374 | 0.126 | 0.102 |

Matched controls:

| Track | Always true F1 | Word LR F1 | Balanced word LR F1 | Native F1 | Tuned GB F1 |
|---|---:|---:|---:|---:|---:|
| stratified | 0.303 | 0.303 | 0.405 | 0.551 | 0.650 |
| temporal | 0.300 | 0.300 | 0.292 | 0.416 | 0.431 |
| checker | 0.307 | 0.334 | 0.384 | 0.497 | 0.500 |

Detailed metrics, full confusion matrices and reliability bins: [stratified](fixed_050/stratified/metrics.json), [temporal](fixed_050/temporal/metrics.json), [checker](fixed_050/checker/metrics.json).

Representative errors (diagnostic; no post-test changes):

| Track / ID | Gold | Predicted | Confidence | Reason |
|---|---|---|---:|---|
| stratified/p-1009 | false | true | 0.8359130604108659 | decided |
| stratified/p-1029 | unsure | true | 0.8330956510655622 | decided |
| stratified/p-1041 | unsure | true | 0.8536967662913529 | decided |
| stratified/p-1066 | false | true | 0.9640371155305753 | decided |
| stratified/p-1095 | false | true | 0.9898357357770946 | decided |

No independent sentence-level causal error audit was performed. Generated truth, proof rate, grounded yield and constrained peak RAM are unavailable/null. Fit/inference timings and effective feature counts are in the metric resource section; native macOS timings are supplementary. No submission-readiness claim.

Limitations: these repeatedly used development splits are not pristine. Classifier hyperparameters were selected previously on the same fit-pool CV folds, so current CV results are selection diagnostics. Few gold unsure examples make macro F1 variable. Gold-form training versus parsed inference and native internal HMM partial cross-fitting remain limitations. Anchor support/gaps are features, not proof of persistence. Feature classifiers do not change the reader or make unlabelled observations trusted facts. No new dependencies or pretrained external facts were added.

Reproduce with a new ID: `.venv/bin/python evaluation/run_daria_features.py --run-id NEW-UNIQUE-ID`

<!-- selection-audit -->

This report evaluates a predeclared fixed candidate. CV selected it on: **none of the tracks**. Its other test results are descriptive and were not used to choose a representation.

| Track / control | Control F1 | Candidate ΔF1 | Control T/F errors/count | Candidate T/F errors/count |
|---|---:|---:|---:|---:|
| stratified / tuned_gb | 0.650 | +0.011 | 47/375 | 48/376 |
| stratified / native | 0.551 | +0.109 | 47/375 | 48/376 |
| stratified / always_true | 0.303 | +0.357 | 63/380 | 48/376 |
| stratified / tfidf_lr | 0.303 | +0.357 | 63/380 | 48/376 |
| stratified / tfidf_lr_balanced | 0.405 | +0.256 | 58/300 | 48/376 |
| temporal / tuned_gb | 0.431 | -0.004 | 79/499 | 81/499 |
| temporal / native | 0.416 | +0.012 | 79/497 | 81/499 |
| temporal / always_true | 0.300 | +0.127 | 91/499 | 81/499 |
| temporal / tfidf_lr | 0.300 | +0.127 | 91/499 | 81/499 |
| temporal / tfidf_lr_balanced | 0.292 | +0.136 | 80/377 | 81/499 |
| checker / tuned_gb | 0.500 | -0.001 | 47/376 | 47/374 |
| checker / native | 0.497 | +0.002 | 46/377 | 47/374 |
| checker / always_true | 0.307 | +0.192 | 56/380 | 47/374 |
| checker / tfidf_lr | 0.334 | +0.165 | 54/380 | 47/374 |
| checker / tfidf_lr_balanced | 0.384 | +0.116 | 46/313 | 47/374 |
