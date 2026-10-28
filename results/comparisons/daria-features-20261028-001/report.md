# Additional feature extraction experiments

24 declared feature pipelines with fixed per-track gradient-boosting hyperparameters. Methods include order-invariant assertion summaries, typed assertion/subject/attribute bags, true-anchor support/date gaps, binary word counts, word-boundary character TF-IDF, chi-square selection, latent text features via SVD, hashing, F/MI feature selection, and scaled PCA. Text-only controls distinguish stand-alone text signal from adding text to temporal features.

| Track | CV-selected representation | Previous GB F1 | Selected test F1 | Δ |
|---|---|---:|---:|---:|
| stratified | expanded_none | 0.6496 | 0.6496 | +0.0000 |
| temporal | rich_pca32_scaled | 0.4309 | 0.4158 | -0.0151 |
| checker | word_binary_chi2_128 | 0.5002 | 0.5030 | +0.0028 |

Failed phase/candidate fits: 0.

Selection uses full-pipeline CV on frozen fit_ids; shared validation and outer labels never select the representation. Existing GB hyperparameters were kept fixed. The baseline representation exactly preserves all original features and is checked against the previous GB predictions on each track. These are development results with no guarantee of unseen generalization. Shared-validation diagnostics, all candidate scores and uncertainty/risk tradeoffs remain visible.

[All candidate results](candidate-results.csv). [Selected-method detailed report](../../evaluations/daria_features_selected/daria-features-20261028-001/report.md).

Generation and independently verified proofs were not evaluated; native timings are supplementary.

```sh
.venv/bin/python evaluation/run_daria_features.py --run-id NEW-UNIQUE-ID
```

<!-- extended-results -->

All configurations: each cell shows **training CV F1 / outer test F1**. Only CV determined selection.

| Feature configuration | Stratified | Temporal | Checker |
|---|---:|---:|---:|
| expanded_none | 0.504 / 0.650 | 0.379 / 0.431 | 0.502 / 0.500 |
| expanded_order_aggregate | 0.448 / 0.660 | 0.380 / 0.427 | 0.480 / 0.499 |
| rich_structured | 0.421 / 0.586 | 0.376 / 0.341 | 0.437 / 0.463 |
| rich_no_calendar | 0.428 / 0.532 | 0.374 / 0.343 | 0.441 / 0.463 |
| rich_no_parse_confidence | 0.422 / 0.586 | 0.374 / 0.341 | 0.438 / 0.463 |
| rich_content_bags | 0.492 / 0.650 | 0.381 / 0.419 | 0.504 / 0.500 |
| word_binary_chi2_128 | 0.463 / 0.651 | 0.375 / 0.430 | 0.505 / 0.503 |
| word_binary_chi2_256 | 0.488 / 0.664 | 0.374 / 0.427 | 0.502 / 0.502 |
| word_number_masked_chi2_128 | 0.483 / 0.651 | 0.373 / 0.430 | 0.482 / 0.503 |
| char_wb_chi2_128 | 0.486 / 0.605 | 0.379 / 0.429 | 0.479 / 0.500 |
| word_tfidf_svd32 | 0.465 / 0.601 | 0.367 / 0.421 | 0.475 / 0.504 |
| word_tfidf_svd64 | 0.465 / 0.554 | 0.381 / 0.420 | 0.500 / 0.507 |
| char_wb_tfidf_svd32 | 0.466 / 0.618 | 0.370 / 0.413 | 0.499 / 0.497 |
| char_wb_tfidf_svd64 | 0.464 / 0.659 | 0.371 / 0.420 | 0.501 / 0.504 |
| word_char_tfidf_svd64 | 0.503 / 0.610 | 0.378 / 0.425 | 0.470 / 0.512 |
| hash_tfidf_svd32 | 0.462 / 0.618 | 0.385 / 0.422 | 0.477 / 0.504 |
| text_only_word_binary128 | 0.307 / 0.315 | 0.310 / 0.297 | 0.308 / 0.321 |
| text_only_word_tfidf_svd32 | 0.323 / 0.303 | 0.311 / 0.298 | 0.305 / 0.307 |
| structured_feature_hasher128 | 0.456 / 0.545 | 0.381 / 0.422 | 0.479 / 0.493 |
| rich_select_f50 | 0.427 / 0.586 | 0.378 / 0.340 | 0.438 / 0.461 |
| rich_select_f75 | 0.431 / 0.586 | 0.379 / 0.341 | 0.437 / 0.464 |
| rich_select_mi50 | 0.432 / 0.535 | 0.355 / 0.409 | 0.447 / 0.461 |
| rich_select_mi75 | 0.453 / 0.578 | 0.373 / 0.343 | 0.434 / 0.461 |
| rich_pca32_scaled | 0.426 / 0.492 | 0.402 / 0.416 | 0.423 / 0.498 |

The highest observed primary-test scores among new candidates were: `word_binary_chi2_256` (0.664), `expanded_order_aggregate` (0.660), `char_wb_tfidf_svd64` (0.659). These are **post-test descriptive rankings**, not CV-selected winners. The original feature representation was the primary CV winner.

Gradient-boosting hyperparameters were previously optimized for the original representation and stayed fixed here. This isolates representation changes at those settings; joint feature-specific hyperparameter tuning was not tested. The reader and assertion parser were unchanged.

All 121 repository tests passed. Every original baseline prediction matched exactly across all three tracks. All 24 configurations completed CV, shared-validation and outer prediction without failure.

Recreate this extended summary after the experiment with:

```sh
.venv/bin/python evaluation/summarize_daria_features.py --run-id daria-features-20261028-001
```
