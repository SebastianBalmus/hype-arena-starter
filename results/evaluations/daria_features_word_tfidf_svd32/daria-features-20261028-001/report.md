# daria_features_word_tfidf_svd32

Feature representation experiment using fixed, previously CV-tuned GradientBoostingClassifier parameters per track. Full feature resources are rebuilt per fold; learned vectorizers, selection and reductions fit on training only. The selected method freezes its representation using mean three-fold CV macro F1 inside shared fit_ids. Shared validation is diagnostic only, and final assets use full outer train_ids. All outputs use fixed_050.

| Track | Feature candidate | CV F1 | Shared validation F1 | Test F1 | Δ vs base | False P/R | Unsure F1 | Coverage | Errors/count | Risk | Brier |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stratified | word_tfidf_svd32 | 0.465 | 0.587 | 0.601 | -0.048 | 0.651/0.509 | 0.308 | 0.987 | 47/375 | 0.125 | 0.107 |
| temporal | word_tfidf_svd32 | 0.367 | 0.378 | 0.421 | -0.010 | 0.656/0.244 | 0.000 | 0.998 | 81/498 | 0.163 | 0.134 |
| checker | word_tfidf_svd32 | 0.475 | 0.490 | 0.504 | +0.004 | 0.592/0.592 | 0.000 | 0.989 | 47/376 | 0.125 | 0.101 |

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
| stratified/p-1009 | false | true | 0.9099683104192732 | decided |
| stratified/p-1029 | unsure | true | 0.9040000149672576 | decided |
| stratified/p-1041 | unsure | true | 0.8885490652438577 | decided |
| stratified/p-1066 | false | true | 0.9049451849985903 | decided |
| stratified/p-1095 | false | true | 0.988028753711958 | decided |

No independent sentence-level causal error audit was performed. Generated truth, proof rate, grounded yield and constrained peak RAM are unavailable/null. Fit/inference timings and effective feature counts are in the metric resource section; native macOS timings are supplementary. No submission-readiness claim.

Limitations: these repeatedly used development splits are not pristine. Classifier hyperparameters were selected previously on the same fit-pool CV folds, so current CV results are selection diagnostics. Few gold unsure examples make macro F1 variable. Gold-form training versus parsed inference and native internal HMM partial cross-fitting remain limitations. Anchor support/gaps are features, not proof of persistence. Feature classifiers do not change the reader or make unlabelled observations trusted facts. No new dependencies or pretrained external facts were added.

Reproduce with a new ID: `.venv/bin/python evaluation/run_daria_features.py --run-id NEW-UNIQUE-ID`

<!-- selection-audit -->

This report evaluates a predeclared fixed candidate. CV selected it on: **none of the tracks**. Its other test results are descriptive and were not used to choose a representation.

| Track / control | Control F1 | Candidate ΔF1 | Control T/F errors/count | Candidate T/F errors/count |
|---|---:|---:|---:|---:|
| stratified / tuned_gb | 0.650 | -0.048 | 47/375 | 47/375 |
| stratified / native | 0.551 | +0.050 | 47/375 | 47/375 |
| stratified / always_true | 0.303 | +0.298 | 63/380 | 47/375 |
| stratified / tfidf_lr | 0.303 | +0.298 | 63/380 | 47/375 |
| stratified / tfidf_lr_balanced | 0.405 | +0.196 | 58/300 | 47/375 |
| temporal / tuned_gb | 0.431 | -0.010 | 79/499 | 81/498 |
| temporal / native | 0.416 | +0.005 | 79/497 | 81/498 |
| temporal / always_true | 0.300 | +0.121 | 91/499 | 81/498 |
| temporal / tfidf_lr | 0.300 | +0.121 | 91/499 | 81/498 |
| temporal / tfidf_lr_balanced | 0.292 | +0.129 | 80/377 | 81/498 |
| checker / tuned_gb | 0.500 | +0.004 | 47/376 | 47/376 |
| checker / native | 0.497 | +0.007 | 46/377 | 47/376 |
| checker / always_true | 0.307 | +0.198 | 56/380 | 47/376 |
| checker / tfidf_lr | 0.334 | +0.171 | 54/380 | 47/376 |
| checker / tfidf_lr_balanced | 0.384 | +0.121 | 46/313 | 47/376 |
