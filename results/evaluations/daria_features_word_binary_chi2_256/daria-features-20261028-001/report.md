# daria_features_word_binary_chi2_256

Feature representation experiment using fixed, previously CV-tuned GradientBoostingClassifier parameters per track. Full feature resources are rebuilt per fold; learned vectorizers, selection and reductions fit on training only. The selected method freezes its representation using mean three-fold CV macro F1 inside shared fit_ids. Shared validation is diagnostic only, and final assets use full outer train_ids. All outputs use fixed_050.

| Track | Feature candidate | CV F1 | Shared validation F1 | Test F1 | Δ vs base | False P/R | Unsure F1 | Coverage | Errors/count | Risk | Brier |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stratified | word_binary_chi2_256 | 0.488 | 0.533 | 0.664 | +0.014 | 0.659/0.491 | 0.500 | 0.989 | 46/376 | 0.122 | 0.105 |
| temporal | word_binary_chi2_256 | 0.374 | 0.382 | 0.427 | -0.004 | 0.688/0.256 | 0.000 | 0.994 | 78/496 | 0.157 | 0.135 |
| checker | word_binary_chi2_256 | 0.502 | 0.503 | 0.502 | +0.002 | 0.596/0.571 | 0.000 | 0.982 | 45/373 | 0.121 | 0.098 |

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
| stratified/p-1009 | false | true | 0.903586201444378 | decided |
| stratified/p-1029 | unsure | true | 0.8990229182993952 | decided |
| stratified/p-1041 | unsure | true | 0.8847354580138208 | decided |
| stratified/p-1066 | false | true | 0.926174620626229 | decided |
| stratified/p-1095 | false | true | 0.9893257334113497 | decided |

No independent sentence-level causal error audit was performed. Generated truth, proof rate, grounded yield and constrained peak RAM are unavailable/null. Fit/inference timings and effective feature counts are in the metric resource section; native macOS timings are supplementary. No submission-readiness claim.

Limitations: these repeatedly used development splits are not pristine. Classifier hyperparameters were selected previously on the same fit-pool CV folds, so current CV results are selection diagnostics. Few gold unsure examples make macro F1 variable. Gold-form training versus parsed inference and native internal HMM partial cross-fitting remain limitations. Anchor support/gaps are features, not proof of persistence. Feature classifiers do not change the reader or make unlabelled observations trusted facts. No new dependencies or pretrained external facts were added.

Reproduce with a new ID: `.venv/bin/python evaluation/run_daria_features.py --run-id NEW-UNIQUE-ID`

<!-- selection-audit -->

This report evaluates a predeclared fixed candidate. CV selected it on: **none of the tracks**. Its other test results are descriptive and were not used to choose a representation.

| Track / control | Control F1 | Candidate ΔF1 | Control T/F errors/count | Candidate T/F errors/count |
|---|---:|---:|---:|---:|
| stratified / tuned_gb | 0.650 | +0.014 | 47/375 | 46/376 |
| stratified / native | 0.551 | +0.113 | 47/375 | 46/376 |
| stratified / always_true | 0.303 | +0.361 | 63/380 | 46/376 |
| stratified / tfidf_lr | 0.303 | +0.361 | 63/380 | 46/376 |
| stratified / tfidf_lr_balanced | 0.405 | +0.259 | 58/300 | 46/376 |
| temporal / tuned_gb | 0.431 | -0.004 | 79/499 | 78/496 |
| temporal / native | 0.416 | +0.011 | 79/497 | 78/496 |
| temporal / always_true | 0.300 | +0.127 | 91/499 | 78/496 |
| temporal / tfidf_lr | 0.300 | +0.127 | 91/499 | 78/496 |
| temporal / tfidf_lr_balanced | 0.292 | +0.135 | 80/377 | 78/496 |
| checker / tuned_gb | 0.500 | +0.002 | 47/376 | 45/373 |
| checker / native | 0.497 | +0.005 | 46/377 | 45/373 |
| checker / always_true | 0.307 | +0.195 | 56/380 | 45/373 |
| checker / tfidf_lr | 0.334 | +0.168 | 54/380 | 45/373 |
| checker / tfidf_lr_balanced | 0.384 | +0.119 | 46/313 | 45/373 |
