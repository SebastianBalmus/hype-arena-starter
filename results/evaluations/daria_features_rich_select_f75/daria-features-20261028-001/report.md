# daria_features_rich_select_f75

Feature representation experiment using fixed, previously CV-tuned GradientBoostingClassifier parameters per track. Full feature resources are rebuilt per fold; learned vectorizers, selection and reductions fit on training only. The selected method freezes its representation using mean three-fold CV macro F1 inside shared fit_ids. Shared validation is diagnostic only, and final assets use full outer train_ids. All outputs use fixed_050.

| Track | Feature candidate | CV F1 | Shared validation F1 | Test F1 | Δ vs base | False P/R | Unsure F1 | Coverage | Errors/count | Risk | Brier |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stratified | rich_select_f75 | 0.431 | 0.559 | 0.586 | -0.063 | 0.365/0.636 | 0.429 | 0.984 | 81/374 | 0.217 | 0.117 |
| temporal | rich_select_f75 | 0.379 | 0.315 | 0.341 | -0.090 | 0.216/0.488 | 0.000 | 1.000 | 199/499 | 0.399 | 0.186 |
| checker | rich_select_f75 | 0.437 | 0.511 | 0.464 | -0.036 | 0.407/0.673 | 0.000 | 0.987 | 69/375 | 0.184 | 0.124 |

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
| stratified/p-1029 | unsure | false | 0.5485407863643362 | decided |
| stratified/p-1041 | unsure | false | 0.538493772083258 | decided |
| stratified/p-1066 | false | true | 0.9382832301084338 | decided |
| stratified/p-1095 | false | true | 0.9887829707278156 | decided |
| stratified/p-1107 | unsure | true | 0.8509886060814219 | decided |

No independent sentence-level causal error audit was performed. Generated truth, proof rate, grounded yield and constrained peak RAM are unavailable/null. Fit/inference timings and effective feature counts are in the metric resource section; native macOS timings are supplementary. No submission-readiness claim.

Limitations: these repeatedly used development splits are not pristine. Classifier hyperparameters were selected previously on the same fit-pool CV folds, so current CV results are selection diagnostics. Few gold unsure examples make macro F1 variable. Gold-form training versus parsed inference and native internal HMM partial cross-fitting remain limitations. Anchor support/gaps are features, not proof of persistence. Feature classifiers do not change the reader or make unlabelled observations trusted facts. No new dependencies or pretrained external facts were added.

Reproduce with a new ID: `.venv/bin/python evaluation/run_daria_features.py --run-id NEW-UNIQUE-ID`

<!-- selection-audit -->

This report evaluates a predeclared fixed candidate. CV selected it on: **none of the tracks**. Its other test results are descriptive and were not used to choose a representation.

| Track / control | Control F1 | Candidate ΔF1 | Control T/F errors/count | Candidate T/F errors/count |
|---|---:|---:|---:|---:|
| stratified / tuned_gb | 0.650 | -0.063 | 47/375 | 81/374 |
| stratified / native | 0.551 | +0.035 | 47/375 | 81/374 |
| stratified / always_true | 0.303 | +0.283 | 63/380 | 81/374 |
| stratified / tfidf_lr | 0.303 | +0.283 | 63/380 | 81/374 |
| stratified / tfidf_lr_balanced | 0.405 | +0.182 | 58/300 | 81/374 |
| temporal / tuned_gb | 0.431 | -0.090 | 79/499 | 199/499 |
| temporal / native | 0.416 | -0.074 | 79/497 | 199/499 |
| temporal / always_true | 0.300 | +0.041 | 91/499 | 199/499 |
| temporal / tfidf_lr | 0.300 | +0.041 | 91/499 | 199/499 |
| temporal / tfidf_lr_balanced | 0.292 | +0.050 | 80/377 | 199/499 |
| checker / tuned_gb | 0.500 | -0.036 | 47/376 | 69/375 |
| checker / native | 0.497 | -0.034 | 46/377 | 69/375 |
| checker / always_true | 0.307 | +0.157 | 56/380 | 69/375 |
| checker / tfidf_lr | 0.334 | +0.130 | 54/380 | 69/375 |
| checker / tfidf_lr_balanced | 0.384 | +0.080 | 46/313 | 69/375 |
