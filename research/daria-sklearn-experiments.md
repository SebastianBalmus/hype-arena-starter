# Daria’s latest pipeline with sklearn alternatives

Run `daria-sklearn-20261028-001` builds on Daria commit `5943f61d07c751f2a560125959ec1f8ba882bb0a`. The branch was fetched and reviewed; the working branch and existing experiment artifacts were preserved.

The sweep tests **54 frozen configurations across 13 estimator families**. It preserves Daria’s learned reader, parse calibration, temporal windows and HMM, and replaces the final binary decision stacker with three-label sklearn classifiers. A native decision control is rebuilt from the same source and permitted evidence.

Model families: logistic regression, calibrated LinearSVC, calibrated Ridge, SGD log-loss, KNN, MLP, histogram gradient boosting, random forest, extra trees, gradient boosting, AdaBoost, Gaussian naive Bayes and RBF SVM. Feature variants include the original three stacker inputs; added timeline states, probability, parse-confidence, atom-count and calendar features; word/character TF-IDF; and number-masked text. Preprocessing includes standard/robust scaling, quantile transforms and second-order interactions. All learned transforms fit on permitted training data.

The native control and all configurations are scored on frozen inner validation first. Highest macro F1 wins, with ties resolved by declared candidate order. The choice is saved before outer test inference. The entire reader, calibrator, timelines and decision model are rebuilt for each inner and outer pool. All runs use arena-eval-v1, batch_unlabelled_v1, seed 42 and fixed_050. Daria’s internal HMM OOF grouping preserves her fixed seed 1.

| Track | Frozen validation choice | Validation F1 | Native test F1 | Selected test F1 | Δ |
|---|---|---:|---:|---:|---:|
| stratified | expanded_none_gb | 0.566 | 0.551 | 0.616 | +0.064 |
| temporal | native_robust_mlp | 0.420 | 0.416 | 0.416 | +0.000 |
| checker | expanded_none_gb | 0.568 | 0.497 | 0.500 | +0.002 |

The primary comparison is stratified; temporal is a forward batch stress test and checker is a compatibility control. The tracks are not averaged. These are previously used development splits, not pristine unseen tests.

| Track / method | Accuracy | False P/R | Unsure F1 | Coverage | Wrong T/F / decisions | Risk | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| stratified / native | 0.866 | 0.667/0.509 | 0.154 | 0.987 | 47/375 | 0.125 | 0.104 |
| stratified / selected | 0.871 | 0.643/0.491 | 0.364 | 0.992 | 48/377 | 0.127 | 0.106 |
| temporal / native | 0.838 | 0.731/0.221 | 0.000 | 0.996 | 79/497 | 0.159 | 0.132 |
| temporal / selected | 0.838 | 0.731/0.221 | 0.000 | 0.996 | 79/497 | 0.159 | 0.135 |
| checker / native | 0.871 | 0.605/0.531 | 0.000 | 0.992 | 46/377 | 0.122 | 0.098 |
| checker / selected | 0.863 | 0.583/0.571 | 0.000 | 0.987 | 47/375 | 0.125 | 0.098 |

Paired label changes:

| Track | Fixed native errors | Introduced errors | Labels changed |
|---|---:|---:|---:|
| stratified | 3 | 1 | 7 |
| temporal | 2 | 2 | 4 |
| checker | 2 | 5 | 10 |

On stratified, unsure F1 changes from 0.154 to 0.364, with only 8 gold unsure rows. That small denominator makes macro F1 sensitive to individual examples. Compare false precision/recall, risk and coverage before choosing an operating model; a macro-F1 gain does not establish a lower false-decision rate.

The full candidate table is saved as [candidate-results.csv](../results/comparisons/daria-sklearn-20261028-001/candidate-results.csv). Individual method directories retain all three metric files, predictions, evaluator-only scored predictions, five failure examples, confusion matrices and reliability bins. The selected-method report also includes all matching controls.

[Selected-method report](../results/evaluations/daria_sklearn_selected/daria-sklearn-20261028-001/report.md). [Run metadata and selections](../results/comparisons/daria-sklearn-20261028-001/run.json). [Latest source review](daria-latest-review-2026-10-28.md).

Native-control parity was checked by separately invoking the original Daria adapter on each sanitized split. [Parity results](../results/comparisons/daria-sklearn-20261028-001/native-parity/native-parity.json) retain commands, hashes and any differences.

All 97 repository tests passed, including eight new boundary and pipeline tests. Candidate warning messages and failures are retained in each phase’s `predictions/timings.json`. MLP and some optimization-heavy candidates can reach iteration limits; their warnings remain visible. Native timing is supplementary; concurrent host work, CPU allocation and peak memory are not controlled.

Limitations: gold-form training versus parsed inference remains a mismatch. Native internal HMM OOF hides labels while retaining permitted training forms, so extraction is not fully cross-fitted there. Temporal continuity and noisy unlabelled observations are modeling assumptions, not proofs. No validation/test gold forms reach method fitting. The large search and tiny unsure class can overfit inner validation. Output confidence remains uncalibrated except declared internal sigmoid calibration. This is a classification experiment: upstream generation, independent truth/proof verification and constrained Linux performance were not evaluated, and no submission-readiness claim is made.

Reproduce with a new run ID:

```sh
.venv/bin/python evaluation/run_daria_sklearn.py --run-id NEW-UNIQUE-ID --ref 5943f61d07c751f2a560125959ec1f8ba882bb0a
.venv/bin/python evaluation/summarize_daria_sklearn.py --run-id NEW-UNIQUE-ID
.venv/bin/python -m unittest discover -s tests
```

For new blank-label rows, use `integrations/daria_sklearn_predict.py --help`; choose a candidate explicitly. The prediction CLI shares the feature and candidate code, checks training-form fingerprints and publishes predictions atomically. It does not generate Arena claims.
