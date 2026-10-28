# retrieval_raw_masked: Madalin claim-feature integration

arena-eval-v1, fixed_050, seed 42. Development results on previously explored frozen splits. Generator candidates are auxiliary model features, not independently verified observations.

| Track | Validation macro F1 | Test macro F1 | False P/R | Unsure F1 | Coverage | Errors/decided | Risk | Brier | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| stratified | 0.442 | 0.468 | 0.420/0.382 | 0.133 | 0.942 | 65/358 | 0.182 | 0.148 | 8.91s native |
| temporal | 0.312 | 0.332 | 0.190/0.128 | 0.038 | 0.904 | 117/451 | 0.259 | 0.198 | 8.62s native |
| checker | 0.360 | 0.362 | 0.221/0.306 | 0.000 | 0.939 | 88/357 | 0.246 | 0.161 | 9.23s native |

Confusion matrices and reliability bins are in each metrics.json. Grounded yield, independently verified generation yield and accepted proof rate are unavailable. Peak RAM and constrained Linux resources were not measured.

## Paired changes and reference controls

| Track | This macro F1 | Retrieval control | Always true | Word LR | Balanced word LR |
| --- | ---: | ---: | ---: | ---: | ---: |
| stratified | 0.468 | 0.468 | 0.303 | 0.303 | 0.405 |
| temporal | 0.332 | 0.332 | 0.300 | 0.300 | 0.292 |
| checker | 0.362 | 0.362 | 0.307 | 0.334 | 0.384 |

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
| p-1009 | false | true | 0.773 | decided |
| p-1029 | unsure | true | 0.754 | decided |
| p-1030 | unsure | true | 0.716 | decided |
| p-1041 | unsure | false | 0.578 | decided |
| p-1056 | false | true | 0.985 | decided |

## Reproduce

```sh
.venv/bin/python evaluation/run_fact_integration.py --run-id NEW-UNIQUE-ID
```

Submission claim generation, independent truth/proof verification, frontend integration and constrained-container readiness are not implemented by this adapter.
