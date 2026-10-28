# daria_features_rich_select_mi50

Feature representation experiment using fixed, previously CV-tuned GradientBoostingClassifier parameters per track. Full feature resources are rebuilt per fold; learned vectorizers, selection and reductions fit on training only. The selected method freezes its representation using mean three-fold CV macro F1 inside shared fit_ids. Shared validation is diagnostic only, and final assets use full outer train_ids. All outputs use fixed_050.

| Track | Feature candidate | CV F1 | Shared validation F1 | Test F1 | Δ vs base | False P/R | Unsure F1 | Coverage | Errors/count | Risk | Brier |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stratified | rich_select_mi50 | 0.432 | 0.516 | 0.535 | -0.115 | 0.659/0.491 | 0.176 | 0.842 | 35/320 | 0.109 | 0.091 |
| temporal | rich_select_mi50 | 0.355 | 0.302 | 0.409 | -0.022 | 0.773/0.198 | 0.000 | 1.000 | 79/499 | 0.158 | 0.173 |
| checker | rich_select_mi50 | 0.447 | 0.515 | 0.461 | -0.039 | 0.402/0.673 | 0.000 | 0.984 | 70/374 | 0.187 | 0.125 |

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
| stratified/p-1009 | false | unsure |  | low_confidence |
| stratified/p-1030 | unsure | true | 0.6108582236948161 | decided |
| stratified/p-1066 | false | true | 0.9489093802836248 | decided |
| stratified/p-1095 | false | true | 0.9882375052657133 | decided |
| stratified/p-1098 | false | unsure |  | low_confidence |

No independent sentence-level causal error audit was performed. Generated truth, proof rate, grounded yield and constrained peak RAM are unavailable/null. Fit/inference timings and effective feature counts are in the metric resource section; native macOS timings are supplementary. No submission-readiness claim.

Limitations: these repeatedly used development splits are not pristine. Classifier hyperparameters were selected previously on the same fit-pool CV folds, so current CV results are selection diagnostics. Few gold unsure examples make macro F1 variable. Gold-form training versus parsed inference and native internal HMM partial cross-fitting remain limitations. Anchor support/gaps are features, not proof of persistence. Feature classifiers do not change the reader or make unlabelled observations trusted facts. No new dependencies or pretrained external facts were added.

Reproduce with a new ID: `.venv/bin/python evaluation/run_daria_features.py --run-id NEW-UNIQUE-ID`

<!-- selection-audit -->

This report evaluates a predeclared fixed candidate. CV selected it on: **none of the tracks**. Its other test results are descriptive and were not used to choose a representation.

| Track / control | Control F1 | Candidate ΔF1 | Control T/F errors/count | Candidate T/F errors/count |
|---|---:|---:|---:|---:|
| stratified / tuned_gb | 0.650 | -0.115 | 47/375 | 35/320 |
| stratified / native | 0.551 | -0.017 | 47/375 | 35/320 |
| stratified / always_true | 0.303 | +0.232 | 63/380 | 35/320 |
| stratified / tfidf_lr | 0.303 | +0.232 | 63/380 | 35/320 |
| stratified / tfidf_lr_balanced | 0.405 | +0.130 | 58/300 | 35/320 |
| temporal / tuned_gb | 0.431 | -0.022 | 79/499 | 79/499 |
| temporal / native | 0.416 | -0.007 | 79/497 | 79/499 |
| temporal / always_true | 0.300 | +0.109 | 91/499 | 79/499 |
| temporal / tfidf_lr | 0.300 | +0.109 | 91/499 | 79/499 |
| temporal / tfidf_lr_balanced | 0.292 | +0.117 | 80/377 | 79/499 |
| checker / tuned_gb | 0.500 | -0.039 | 47/376 | 70/374 |
| checker / native | 0.497 | -0.036 | 46/377 | 70/374 |
| checker / always_true | 0.307 | +0.154 | 56/380 | 70/374 |
| checker / tfidf_lr | 0.334 | +0.127 | 54/380 | 70/374 |
| checker / tfidf_lr_balanced | 0.384 | +0.078 | 46/313 | 70/374 |
