# Synthetic-only training → original practice predictions

Training: 2,000 synthetic rows only. Test: all 1,897 original rows, supplied with empty labels and no forms. Seed 42; fixed 0.50 decision policy. Daria expanded features + gradient boosting, frozen settings selected in the prior primary-track CV comparison.

**This is a source-dependent transfer diagnostic, not independent confirmation that synthetic data are correct.** Generated training examples were built from these original test labels/forms/texts. All parser/calibration/temporal resources are rebuilt using synthetic training inputs only, but that indirect ancestry leakage remains.

Accuracy: 0.8998. Macro F1: 0.5474.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| true | 0.9194 | 0.9671 | 0.9427 | 1581 |
| false | 0.7672 | 0.6426 | 0.6994 | 277 |
| unsure | 0.0000 | 0.0000 | 0.0000 | 39 |

Confusion matrix: gold rows and prediction columns ordered true, false, unsure.

```json
{"labels": ["true", "false", "unsure"], "counts": [[1529, 51, 1], [98, 178, 1], [36, 3, 0]]}
```

| Control | Accuracy | Macro F1 |
|---|---:|---:|
| daria_expanded_gb | 0.8998 | 0.5474 |
| always_true | 0.0206 | 0.0134 |
| tfidf_lr | 0.3195 | 0.2794 |
| tfidf_lr_balanced | 0.1629 | 0.1721 |

The prior-only always_true control abstains under fixed_050 because synthetic true prevalence is 45%, below the decision threshold. It is not forced to emit an invalid confidence.

Per-class metrics, all-row predictions, errors, reliability diagnostics and direct-ancestor slices are saved beside this report. Neither slice is independent: all share the supplied fictional universe and full-corpus augmentation design. Native/macOS timing is supplementary; constrained image resources, generation and independent proof verification were not run. No baseline results, runtime assets, original data or shared splits were overwritten.

Reproduce from repository root: `.venv/bin/python evaluation/run_synthetic_transfer.py NEW-UNIQUE-RUN-ID`.

## Interpretation

The pipeline correctly labels 1,707/1,897 originals. True recall is 1,529/1,581; false recall is 178/277. It misses all 39 unsure rows (36 become true and three become false); its two unsure predictions are both errors. This run does not establish acceptable uncertainty behavior or certify annotation quality. High aggregate accuracy partly reflects the original corpus being 83.3% true. A cleaner augmentation-quality test would generate only from permitted training ancestors and test on originals whose answers were not used to generate training material.

## Representative errors

| ID | Gold | Prediction | Original text |
|---|---|---|---|
| p-121 | unsure | true | Hooly Android 17 supports 19 models in the beta. |
| p-179 | unsure | true | Hooly's board runs an audit committee, and finance questions go to the full table. |
| p-126 | false | true | Nelson Bighettie thanked every tester in a note to the group. |
| p-127 | false | true | Hooly Android 16.0 remains the stable release on 3,100,000,000 devices. |
| p-25 | true | false | Gemini still speaks 6 languages and connects to 9 integrations; the big expansion comes in the spring. |
| p-45 | true | false | Release v1.0 of the editor leans on Hooly Gemini 1.5 for suggestions. |

These errors identify observed label failures; no independent per-row causal proof audit was performed. Full readable predictions are in `original_predictions_with_labels.csv`.
