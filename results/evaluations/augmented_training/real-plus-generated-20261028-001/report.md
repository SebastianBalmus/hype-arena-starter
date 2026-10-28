# Original training split plus generated examples

Same Daria expanded-feature gradient boosting pipeline, primary-track frozen settings, seed 42 and fixed_050 for all runs. Real-only controls were rebuilt under identical settings and calibration policy. No model selection used these outcomes.

Only generated descendants of each frozen training pool are included. The calibration fitting pool likewise excludes descendants of its held source rows. All pending original labels/forms are hidden. This is still a development diagnostic: augmentation authoring/selection saw the full corpus, and the supplied synthetic batch was filtered rather than recreated from split-only information. Results cannot establish a clean v1 improvement or be directly compared to the previous 1,897-row transfer test.

| Track | Training | Original train | Generated train | Test | Accuracy | Macro F1 |
|---|---|---:|---:|---:|---:|---:|
| stratified | real_only | 1517 | 0 | 380 | 0.8711 | 0.6496 |
| stratified | real_plus_generated | 1517 | 1593 | 380 | 0.8500 | 0.4815 |
| temporal | real_only | 1398 | 0 | 499 | 0.8417 | 0.4165 |
| temporal | real_plus_generated | 1398 | 1443 | 499 | 0.8096 | 0.4076 |
| checker | real_only | 1517 | 0 | 380 | 0.8605 | 0.4952 |
| checker | real_plus_generated | 1517 | 1622 | 380 | 0.8684 | 0.4805 |

## stratified: per-class results

| Training | Class | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| real_only | true | 0.9012 | 0.9495 | 0.9247 | 317 |
| real_only | false | 0.6585 | 0.4909 | 0.5625 | 55 |
| real_only | unsure | 0.6000 | 0.3750 | 0.4615 | 8 |

real_only confusion matrix (gold rows / predicted columns true, false, unsure): `{"labels": ["true", "false", "unsure"], "counts": [[301, 14, 2], [28, 27, 0], [5, 0, 3]]}.`

| real_plus_generated | true | 0.8919 | 0.9369 | 0.9138 | 317 |
| real_plus_generated | false | 0.6047 | 0.4727 | 0.5306 | 55 |
| real_plus_generated | unsure | 0.0000 | 0.0000 | 0.0000 | 8 |

real_plus_generated confusion matrix (gold rows / predicted columns true, false, unsure): `{"labels": ["true", "false", "unsure"], "counts": [[297, 17, 3], [28, 26, 1], [8, 0, 0]]}.`

Representative augmented-training failures (observed errors, not a causal proof audit):

- p-22: true → false. Hooly Jules has 9,000 users, Hooly Jules users accept 38% of its suggestions, and Hooly Jules suggests code in 14 languages.
- p-120: true → false. The beta of Hooly Android 17 now has 480,000 testers.
- p-163: true → false. The Pixel 9 is not discontinued, so stock is not an issue.
- p-181: false → true. 9 roles stand open at the Hooly Hiring Programme.
- p-203: false → true. The Hooly Sustainability Report came out on August 25, 2026.

## temporal: per-class results

| Training | Class | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| real_only | true | 0.8478 | 0.9828 | 0.9103 | 408 |
| real_only | false | 0.7308 | 0.2209 | 0.3393 | 86 |
| real_only | unsure | 0.0000 | 0.0000 | 0.0000 | 5 |

real_only confusion matrix (gold rows / predicted columns true, false, unsure): `{"labels": ["true", "false", "unsure"], "counts": [[401, 7, 0], [67, 19, 0], [5, 0, 0]]}.`

| real_plus_generated | true | 0.8470 | 0.9363 | 0.8894 | 408 |
| real_plus_generated | false | 0.4783 | 0.2558 | 0.3333 | 86 |
| real_plus_generated | unsure | 0.0000 | 0.0000 | 0.0000 | 5 |

real_plus_generated confusion matrix (gold rows / predicted columns true, false, unsure): `{"labels": ["true", "false", "unsure"], "counts": [[382, 24, 2], [64, 22, 0], [5, 0, 0]]}.`

Representative augmented-training failures (observed errors, not a causal proof audit):

- p-197: false → true. Responses are not anonymous.
- p-200: false → true. 58% agree that they can grow at Hooly.
- p-203: false → true. The Hooly Sustainability Report came out on August 25, 2026.
- p-210: false → true. We set out to cut emissions by 40%.
- p-212: false → true. The Hooly Board hears the figures from Bertram Gilfoil.

## checker: per-class results

| Training | Class | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| real_only | true | 0.9172 | 0.9228 | 0.9200 | 324 |
| real_only | false | 0.5600 | 0.5714 | 0.5657 | 49 |
| real_only | unsure | 0.0000 | 0.0000 | 0.0000 | 7 |

real_only confusion matrix (gold rows / predicted columns true, false, unsure): `{"labels": ["true", "false", "unsure"], "counts": [[299, 22, 3], [20, 28, 1], [7, 0, 0]]}.`

| real_plus_generated | true | 0.9029 | 0.9475 | 0.9247 | 324 |
| real_plus_generated | false | 0.5750 | 0.4694 | 0.5169 | 49 |
| real_plus_generated | unsure | 0.0000 | 0.0000 | 0.0000 | 7 |

real_plus_generated confusion matrix (gold rows / predicted columns true, false, unsure): `{"labels": ["true", "false", "unsure"], "counts": [[307, 17, 0], [26, 23, 0], [7, 0, 0]]}.`

Representative augmented-training failures (observed errors, not a causal proof audit):

- p-21: true → false. "Jules is Gemini for code," said Richard Hendrix, our CTO.
- p-31: true → false. The template gallery in Hooly Docs grows to 80 templates.
- p-121: unsure → true. Hooly Android 17 supports 19 models in the beta.
- p-126: false → true. Nelson Bighettie thanked every tester in a note to the group.
- p-141: true → false. Made by Hooly has signed 6 sponsors for the show floor.

Per-class controls, risk/coverage, reliability bins, confusion matrices, source-scoped inputs and predictions are saved per condition. Generation, independent proof verification and constrained container benchmarks were not run. Prior reference results and shared splits are unchanged.

Reproduce: `.venv/bin/python evaluation/run_augmented_training.py NEW-UNIQUE-RUN-ID`.
