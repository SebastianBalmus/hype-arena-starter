# daria_features_text_only_word_tfidf_svd32

Feature representation experiment using fixed, previously CV-tuned GradientBoostingClassifier parameters per track. Full feature resources are rebuilt per fold; learned vectorizers, selection and reductions fit on training only. The selected method freezes its representation using mean three-fold CV macro F1 inside shared fit_ids. Shared validation is diagnostic only, and final assets use full outer train_ids. All outputs use fixed_050.

| Track | Feature candidate | CV F1 | Shared validation F1 | Test F1 | Δ vs base | False P/R | Unsure F1 | Coverage | Errors/count | Risk | Brier |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stratified | text_only_word_tfidf_svd32 | 0.323 | 0.301 | 0.303 | -0.346 | 0.000/0.000 | 0.000 | 1.000 | 63/380 | 0.166 | 0.140 |
| temporal | text_only_word_tfidf_svd32 | 0.311 | 0.312 | 0.298 | -0.133 | 0.000/0.000 | 0.000 | 0.996 | 93/497 | 0.187 | 0.154 |
| checker | text_only_word_tfidf_svd32 | 0.305 | 0.302 | 0.307 | -0.193 | 0.000/0.000 | 0.000 | 1.000 | 56/380 | 0.147 | 0.127 |

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
| stratified/p-1009 | false | true | 0.8975209554840022 | decided |
| stratified/p-1029 | unsure | true | 0.8964199651203343 | decided |
| stratified/p-1030 | unsure | true | 0.8580159965356783 | decided |
| stratified/p-1041 | unsure | true | 0.7936356004882427 | decided |
| stratified/p-1042 | false | true | 0.8845694936078229 | decided |

No independent sentence-level causal error audit was performed. Generated truth, proof rate, grounded yield and constrained peak RAM are unavailable/null. Fit/inference timings and effective feature counts are in the metric resource section; native macOS timings are supplementary. No submission-readiness claim.

Limitations: these repeatedly used development splits are not pristine. Classifier hyperparameters were selected previously on the same fit-pool CV folds, so current CV results are selection diagnostics. Few gold unsure examples make macro F1 variable. Gold-form training versus parsed inference and native internal HMM partial cross-fitting remain limitations. Anchor support/gaps are features, not proof of persistence. Feature classifiers do not change the reader or make unlabelled observations trusted facts. No new dependencies or pretrained external facts were added.

Reproduce with a new ID: `.venv/bin/python evaluation/run_daria_features.py --run-id NEW-UNIQUE-ID`

<!-- selection-audit -->

This report evaluates a predeclared fixed candidate. CV selected it on: **none of the tracks**. Its other test results are descriptive and were not used to choose a representation.

| Track / control | Control F1 | Candidate ΔF1 | Control T/F errors/count | Candidate T/F errors/count |
|---|---:|---:|---:|---:|
| stratified / tuned_gb | 0.650 | -0.346 | 47/375 | 63/380 |
| stratified / native | 0.551 | -0.248 | 47/375 | 63/380 |
| stratified / always_true | 0.303 | +0.000 | 63/380 | 63/380 |
| stratified / tfidf_lr | 0.303 | +0.000 | 63/380 | 63/380 |
| stratified / tfidf_lr_balanced | 0.405 | -0.102 | 58/300 | 63/380 |
| temporal / tuned_gb | 0.431 | -0.133 | 79/499 | 93/497 |
| temporal / native | 0.416 | -0.117 | 79/497 | 93/497 |
| temporal / always_true | 0.300 | -0.002 | 91/499 | 93/497 |
| temporal / tfidf_lr | 0.300 | -0.002 | 91/499 | 93/497 |
| temporal / tfidf_lr_balanced | 0.292 | +0.007 | 80/377 | 93/497 |
| checker / tuned_gb | 0.500 | -0.193 | 47/376 | 56/380 |
| checker / native | 0.497 | -0.191 | 46/377 | 56/380 |
| checker / always_true | 0.307 | +0.000 | 56/380 | 56/380 |
| checker / tfidf_lr | 0.334 | -0.027 | 54/380 | 56/380 |
| checker / tfidf_lr_balanced | 0.384 | -0.077 | 46/313 | 56/380 |
