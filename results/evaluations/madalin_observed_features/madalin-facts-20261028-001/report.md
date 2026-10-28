# madalin_observed_features: Madalin claim-feature integration

arena-eval-v1, fixed_050, seed 42. Development results on previously explored frozen splits. Generator candidates are auxiliary model features, not independently verified observations.

| Track | Validation macro F1 | Test macro F1 | False P/R | Unsure F1 | Coverage | Errors/decided | Risk | Brier | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| stratified | 0.406 | 0.505 | 0.431/0.400 | 0.222 | 0.950 | 64/361 | 0.177 | 0.147 | 11.34s native |
| temporal | 0.324 | 0.365 | 0.245/0.140 | 0.089 | 0.920 | 108/459 | 0.235 | 0.190 | 11.18s native |
| checker | 0.388 | 0.403 | 0.268/0.306 | 0.069 | 0.942 | 76/358 | 0.212 | 0.162 | 11.65s native |

Confusion matrices and reliability bins are in each metrics.json. Grounded yield, independently verified generation yield and accepted proof rate are unavailable. Peak RAM and constrained Linux resources were not measured.

## Paired changes and reference controls

| Track | This macro F1 | Retrieval control | Always true | Word LR | Balanced word LR |
| --- | ---: | ---: | ---: | ---: | ---: |
| stratified | 0.505 | 0.468 | 0.303 | 0.303 | 0.405 |
| temporal | 0.365 | 0.332 | 0.300 | 0.300 | 0.292 |
| checker | 0.403 | 0.362 | 0.307 | 0.334 | 0.384 |

## Design and boundaries

The reference is unchanged raw+number-masked retrieval. The observed candidate renders dated atomic statements from permitted true training forms with the teammate value renderer. The generated candidate calls Madalin’s pinned enumerate_candidates/select_claims implementation on true training annotations, with no pending/unsure/false positive observations and at most 1,000 candidates per index. Each training fold regenerates its own assets from its other four folds. All query labels are blank and query forms are absent.

Claim-feature catalogues preserve source IDs, fingerprints, observation dates and temporal assumptions. Current fact/comparison candidates retain the contract as-of date; publication dates are not silently substituted for that date. Generator intervals, freshness and relation overlap remain assumptions. Candidates never enter the trusted world evidence store or become synthetic true-labelled training targets. All output probabilities remain uncalibrated.

Selection uses the frozen inner fit/validation rows; the winner is recorded before any test fit. Final refits use the complete outer training pool. All methods share the same fixed 0.50 decision policy. Method inputs are sanitized CSV files and split-permitted training forms; gold answers remain evaluator-side. No OS-enforced filesystem sandbox exists.

The selected method references its candidate’s saved artifact directory instead of refitting twice. Source and exported artifact hashes, upstream commit, exact row-ID pools and native timing are in run.json.

## Validation selection

| Track | Selected candidate |
| --- | --- |
| stratified | retrieval_raw_masked |
| temporal | madalin_observed_features |
| checker | madalin_observed_features |

## Representative failures

The rows below show the executed classifier outcome. Neighbor/claim-feature traces describe its inputs; no unsupported causal explanation is inferred.

| ID | Gold | Predicted | Confidence | Reason |
| --- | --- | --- | ---: | --- |
| p-1009 | false | true | 0.729 | decided |
| p-1029 | unsure | true | 0.632 | decided |
| p-1030 | unsure | true | 0.708 | decided |
| p-1056 | false | true | 0.983 | decided |
| p-1095 | false | true | 0.943 | decided |

## Reproduce

```sh
.venv/bin/python evaluation/run_fact_integration.py --run-id NEW-UNIQUE-ID
```

Submission claim generation, independent truth/proof verification, frontend integration and constrained-container readiness are not implemented by this adapter.
