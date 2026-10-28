# GradientBoostingClassifier: fold-rebuilt GridSearchCV

The expanded temporal feature set is unchanged. The single tested change is choosing gradient-boosting hyperparameters using GridSearchCV on shared fit_ids. Each CV fold rebuilds the full reader, parse calibration, windows and HMM from its own labelled rows/forms; fold validation is unlabelled text with no forms. This avoids applying ordinary CV to features already learned from all tuning labels.

Run `daria-gb-grid-20261028-002`. 108 parameter combinations × 3 folds per track; 972 CV classifier fits total. Search criterion: mean macro F1 after fixed_050. Ties use declared ParameterGrid order. The grid includes the unchanged GB parameters (80 trees, learning rate .05, depth 2, leaf minimum 1, subsample 1). CV runs only within frozen fit_ids. Shared validation is scored once after selection; it never changes selected parameters. Final classifier is refitted on complete outer train_ids using verified split-scoped cached features.

CV is stratified three-fold with seed 42 on stratified/checker and expanding whole-date forward three-fold on temporal. These are internal training folds; the shared outer benchmark is unchanged. Parser calibration has a nested training-only 80/20 original-order holdout. Native internal HMM grouping retains seed 1 and permitted training forms; it is not a completely cross-fitted reader within its own training features.

| Track | Best CV F1 | Shared validation F1 | Untuned GB test F1 | Tuned test F1 | ΔF1 |
|---|---:|---:|---:|---:|---:|
| stratified | 0.504 | 0.545 | 0.616 | 0.650 | +0.034 |
| temporal | 0.379 | 0.383 | 0.425 | 0.431 | +0.006 |
| checker | 0.502 | 0.496 | 0.500 | 0.500 | +0.001 |

Selected parameters:

| Track | Trees | Learning rate | Max depth | Min samples per leaf | Subsample |
|---|---:|---:|---:|---:|---:|
| stratified | 160 | 0.03 | 2 | 8 | 0.8 |
| temporal | 160 | 0.1 | 1 | 8 | 0.8 |
| checker | 80 | 0.1 | 1 | 8 | 0.8 |

| Track | False precision / recall | Unsure F1 | Coverage | T/F errors / count | Risk | Brier | Search / feature build |
|---|---:|---:|---:|---:|---:|---:|---|
| stratified | 0.659/0.491 | 0.462 | 0.987 | 47/375 | 0.125 | 0.104 | 67.1s / 15.5s native |
| temporal | 0.676/0.267 | 0.000 | 1.000 | 79/499 | 0.158 | 0.135 | 43.9s / 14.5s native |
| checker | 0.583/0.571 | 0.000 | 0.989 | 47/376 | 0.125 | 0.101 | 66.7s / 15.5s native |

Matched controls (all were independently rescored from saved predictions with the same protocol, IDs, access and policy):

| Track / control | Control F1 | Tuned ΔF1 | Control T/F errors/count | Fixed errors | Introduced errors |
|---|---:|---:|---:|---:|---:|
| stratified / untuned_gb | 0.616 | +0.034 | 48/377 | 1 | 1 |
| stratified / native | 0.551 | +0.098 | 47/375 | 4 | 2 |
| stratified / always_true | 0.303 | +0.346 | 63/380 | 30 | 16 |
| stratified / tfidf_lr | 0.303 | +0.346 | 63/380 | 30 | 16 |
| stratified / tfidf_lr_balanced | 0.405 | +0.245 | 58/300 | 101 | 17 |
| temporal / untuned_gb | 0.425 | +0.006 | 75/498 | 3 | 6 |
| temporal / native | 0.416 | +0.015 | 79/497 | 6 | 4 |
| temporal / always_true | 0.300 | +0.131 | 91/499 | 23 | 11 |
| temporal / tfidf_lr | 0.300 | +0.131 | 91/499 | 23 | 11 |
| temporal / tfidf_lr_balanced | 0.292 | +0.139 | 80/377 | 133 | 11 |
| checker / untuned_gb | 0.500 | +0.001 | 47/375 | 3 | 2 |
| checker / native | 0.497 | +0.003 | 46/377 | 3 | 5 |
| checker / always_true | 0.307 | +0.193 | 56/380 | 28 | 23 |
| checker / tfidf_lr | 0.334 | +0.166 | 54/380 | 26 | 23 |
| checker / tfidf_lr_balanced | 0.384 | +0.117 | 46/313 | 83 | 24 |

Representative errors; no post-test retuning:

| Track / ID | Gold | Prediction | Confidence | Reason |
|---|---|---|---:|---|
| stratified/p-1009 | false | true | 0.8909335007184209 | decided |
| stratified/p-1029 | unsure | true | 0.8855396132215423 | decided |
| stratified/p-1041 | unsure | true | 0.9023614878277699 | decided |
| stratified/p-1066 | false | true | 0.9090019687122463 | decided |
| stratified/p-1095 | false | true | 0.9897261448768696 | decided |

No independent sentence-level causal audit was performed: these are observed classifier errors, not verified parser/evidence diagnoses. Probabilistic persistence, gold-form training versus parsed inference, scarce unsure examples and correlated rows remain limitations. Outer development results have already informed the choice of GB for this follow-up; the test sets are not pristine. CV-selected parameters are best within this finite grid and training protocol, not globally optimal.

Generation, independently checked proofs, grounded yield and constrained peak RAM are unmeasured/null. Native macOS timings are supplementary. No independent generated-truth or submission-readiness claim is made. Confidence is uncalibrated.

Full CV fold scores, timings, confusion matrices, per-class supports and reliability bins:

- [stratified full grid results](fixed_050/stratified/grid-results.json); [stratified outer metrics](fixed_050/stratified/metrics.json); [stratified shared-validation metrics](fixed_050/stratified/validation/metrics.json).
- [temporal full grid results](fixed_050/temporal/grid-results.json); [temporal outer metrics](fixed_050/temporal/metrics.json); [temporal shared-validation metrics](fixed_050/temporal/validation/metrics.json).
- [checker full grid results](fixed_050/checker/grid-results.json); [checker outer metrics](fixed_050/checker/metrics.json); [checker shared-validation metrics](fixed_050/checker/validation/metrics.json).

Reproduce with a new ID:

```sh
.venv/bin/python evaluation/run_daria_gb_grid.py --run-id NEW-UNIQUE-ID
```

Verification: all 107 repository tests passed. The untuned configuration reproduces its original predictions exactly on every track; see [baseline parity](baseline-parity/baseline-parity.json) and [verification](verification.json). The initial failed scorer invocation is preserved separately in run001 and is ineligible for comparison.
