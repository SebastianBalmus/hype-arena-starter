# Comparing future Hype Arena methods

Prepared 28 October 2026. This is a proposed local evaluation protocol, not the organizer's scoring formula. No new method was trained or benchmarked while preparing it.

Agents running experiments MUST follow [the shared evaluation instructions](../EVALUATION.md) and their frozen split manifest. This research note explains the rationale; the agent instructions define the versioned operational defaults.

The task has two outputs: dated `true` / `false` / `unsure` verdicts and exactly 1,000 new true claims. Compare methods with a fixed benchmark and a small scorecard covering both outputs, evidence, confidence and resources. A single weighted score would hide tradeoffs that the published contract does not resolve.

## 1. The minimum scorecard

| Dimension | Report | Better direction | Why it fits |
| --- | --- | --- | --- |
| Label quality | Three-class macro F1 | Higher | Gives `false` and `unsure` equal class weight despite the majority of `true` rows. |
| Contradiction detection | False precision and recall, with counts | Higher | Shows whether a method actually catches lies without indiscriminately calling things false. |
| Reliable coverage | Coverage and actual risk under a validation-selected error policy | More coverage at the same risk | Separates a useful conservative method from one that abstains almost everywhere. |
| Confidence | Correctness Brier loss plus a reliability table | Lower loss; confidence close to observed correctness | Tests the meaning of reported confidence, alongside coverage. |
| Grounded decisions | Correct, independently proof-verified T/F decisions divided by all test rows | Higher | Rewards answers that are correct and justified by admissible evidence. |
| Generation | Independently verified valid claims in the first 1,000, divided by 1,000 | Higher | Missing and invalid claims cannot disappear from the denominator. |
| Explainability | Independently replayable proof rate, separately for decisions and claims | Higher | Makes explanations inspectable and testable. |
| Operational cost | Full batch seconds, peak container RAM, p95 live-row latency | Lower, after quality requirements | Measures feasibility under the actual execution constraints. |

Always attach the full confusion matrix, per-class precision/recall/F1 and sample counts. These explain changes in the headline metrics. Raw accuracy is a useful reference, but predicting `true` on all 1,897 practice rows already gets 1,581 right, or 83.34%, while detecting no false rows.

## 2. Exact definitions

Let `N` be the number of held-out labelled rows, `y_i` the gold label and `h_i` the emitted label. The three labels are always included in evaluation. A missing prediction is treated as `unsure` for semantic metrics, while separately failing the output-completeness gate.

### Labels

For each class `k`, use one-versus-rest counts:

```text
precision_k = TP_k / (TP_k + FP_k)
recall_k    = TP_k / (TP_k + FN_k)
F1_k        = 2 TP_k / (2 TP_k + FP_k + FN_k)
macro_F1    = (F1_true + F1_false + F1_unsure) / 3
```

Report zero-denominator conventions. A class with gold support but no correct predictions has F1 zero. If an entire test split lacks a class, mark that class unsupported and do not silently compare its macro score to a split containing all three classes. Pool held-out predictions across folds for the main three-class result.

False precision answers “How many rows called false were false?” False recall answers “How many false rows did we catch?” Keep both; predicting everything false has perfect false recall but poor precision and macro F1.

### Selective decisions: coverage and risk

```text
D(tau) = {i: h_i is true or false, and confidence_i >= tau}
coverage(tau) = |D(tau)| / N
risk(tau) = number of incorrect decisions in D(tau) / |D(tau)|
decided_accuracy(tau) = 1 - risk(tau)
```

For rows below the threshold, emit `unsure` and recompute the external three-class metrics under that policy. If `D` is empty, risk is undefined, not zero.

Use a predeclared working point, for example a **5% empirical validation-error target**, as a local engineering preference. Choose the threshold on validation data, freeze it, and report the resulting **actual test risk and coverage**. A validation target is not a guarantee that test error stays below 5%. Report error counts and confidence intervals. If the target is not attained on validation, say so. If test risk exceeds it, display that failure.

Also plot risk against coverage or compare risk at common attainable coverage levels. Never compare decided accuracy alone. A method with 99% decided accuracy at 3% coverage may be less useful than one with 96% at 80% coverage. Methods without candidate T/F decisions for enough rows cannot be credited with unattainable coverage.

**Important distinction:** gold `unsure` means the dataset itself does not resolve the proposition. Model abstention means this method did not decide. They share an output label, but need separate internal reasons. Emitting T/F on gold `unsure` is an error in decided accuracy and risk. Record abstention reasons such as ambiguous parse, missing evidence, temporal conflict and low confidence. Report coverage by gold class to expose disproportionate rejection of false or unsure cases.

This coverage/error formulation follows reject-option classification research; the proposed working point is our choice, not an Arena rule. [Franc et al., 2023](https://www.jmlr.org/papers/v24/21-0048.html).

### Confidence

The CSV confidence is “probability that my selected label is right,” not necessarily `P(true)`. For decided rows, set `z_i = 1` if the chosen label equals the gold label, otherwise zero:

```text
correctness_Brier = mean over decided rows of (confidence_i - z_i)^2
```

A gold-`unsure` row given a T/F answer has `z_i = 0`. Lower is better. Compare the deployed policies alongside coverage; use a shared decided subset for a more direct comparison of their confidence estimates. Equal coverage alone does not make the selected rows equally difficult.

If every method supplies full normalized probabilities for all three classes, also use the all-row multiclass Brier score:

```text
multiclass_Brier = (1/N) sum_i sum_k (p_i,k - 1[y_i = k])^2
```

This convention has range 0 to 2 and no extra factor of one half. Do not invent the missing class distribution from a scalar output confidence.

For a directly explainable calibration check, publish bins with their sample count, mean confidence and observed fraction correct. For example, predictions near 0.9 should be correct about 90% of the time over enough cases. Fit calibration on validation data only. Brier measures probability quality, including discrimination; it is not a pure calibration measure. ECE can be supplementary with fixed bin definitions, but small bins make it unstable. [scikit-learn calibration guide](https://scikit-learn.org/stable/modules/calibration.html).

## 3. Make explainability measurable

For every T/F decision and generated claim, preserve a sidecar execution record containing:

```text
output ID; queried date or temporal condition
parsed assertions, alternatives, typed values and operators
source row IDs, source text fingerprints, source dates and supplied labels
normalized premises; inference rule IDs; intermediate results
explicit assumptions, conflicts and confidence inputs
```

Keep these records outside the submission's strict CSV form schema. Record the reasoning actually executed; a plausible explanation written after the decision is weaker evidence of how the method behaved.

An independent verifier must check that cited rows exist, were permitted in this split, were interpreted correctly and support the conclusion under its temporal semantics. Reusing the engine's own inference result only measures self-consistency.

```text
proof_replay_rate = accepted complete proofs / all emitted T/F decisions
claim_proof_replay_rate = accepted complete proofs / all emitted claims
grounded_decision_yield = correct T/F decisions with accepted proofs / N
assumption_bearing_rate = outputs requiring an unproved assumption / outputs audited
```

An assumption-free proof needs admissible premises and sound rules. A trace that replays only after assuming “the June value persists through October” must disclose that assumption and cannot pass an assumption-free support check. Report operational replay under declared assumptions separately from accepted evidentiary proofs.

For compound rows, a `true` verdict needs support for every required assertion; a `false` verdict needs at least one justified contradiction. A false compound row does not make each of its atoms false. An `unsure` explanation should expose the missing or conflicting premise, rather than claim a proof of global unknowability.

Test explanation sufficiency by rerunning the decision from only its cited premises. Where a necessary premise is known in a controlled test, remove it, rebuild derived assets and check that the original conclusion loses support. A conclusion surviving deletion is not automatically a failure: another complete proof may exist. These are local adaptations of evidence-aware verification and explanation-faithfulness ideas, not official FEVER or ERASER scores. [FEVER shared task](https://aclanthology.org/W18-5501/), [ERASER](https://aclanthology.org/2020.acl-main.408/).

## 4. Measure generation without confusing support and truth

For each of the first 1,000 submitted claim positions, independently check structure, allowed names and attributes, controlled text/form agreement, runtime-input text novelty and truth under that claim type's temporal meaning.

```text
verified_valid_yield_at_1000 =
  number of positions passing all checks, including independent truth verification / 1000
```

Missing positions and invalid entries contribute zero. For an audit that cannot resolve every position, report counts of verified-valid, contradicted, invalid and unverified claims. `verified_valid_yield_at_1000` is then a confirmed yield, not an estimate that all remaining claims are false. If sampling claims for manual audit, report a sampling design and interval; never substitute sample precision for an exact all-1,000 count.

Semantic uniqueness is a useful additional quality measure. Canonicalize subject aliases and typed values; normalize compare direction so `A > B` and `B < A` share one key; retain date precision, billing periods, event identity and relation structure. Report unique verified yield and redundant count separately. The contract explicitly requires text novelty; this stricter semantic deduplication is an internal anti-padding criterion, not a published disqualification rule.

Also report the number and validation status of candidates before selection, and selected yield by `fact`, `compare`, `change`, `at_event` and `relation`. Diversity is diagnostic; the contract does not publish a five-type quota.

**There is no complete public truth oracle.** Practice forms state what a row asserts, including false assertions. An old true observation does not certify the value on 29 October. The generator's proof store and confidence cannot independently establish its own truth accuracy.

Use two complementary evaluations:

1. **Real-corpus support audit:** independent review of admissible premises, parsing, temporal conditions and assumptions. Where truth remains unknown, report supported or unverified yield explicitly as a proxy.
2. **Controlled temporal-world benchmark:** create a separate fully specified world with known value intervals, event occurrence dates, relationship overlaps and missing evidence. Keep its answer store private to the evaluator and expose only the intended noisy/labelled observations to methods. An independent oracle can then score all five claim types exactly. Use worlds large enough to contain at least 1,000 legitimate novel claims for the full quota test; use smaller cases only as diagnostic tests. Synthetic-world success tests reasoning soundness and does not establish Hooly-world accuracy.

For generated claims, distinguish actual truth in the controlled world from whether that truth is justified by the observations the method received. A lucky unsupported assertion may pass a truth check but fail the evidence check.

Measure shared-premise exposure too:

```text
max_premise_exposure =
  maximum, over source premises, of selected claims depending on that premise / selected claims
```

Report it alongside premise count and a histogram of reuse. A single stale premise supporting hundreds of comparisons creates correlated risk. This exposure is a diagnostic, not a probability of failure or an official score.

## 5. Diagnose why a method improved

Use the same test rows and normalize only semantically equivalent representations.

| Component or failure slice | Measurement |
| --- | --- |
| Assertion extraction | Micro precision/recall/F1 of canonical `(subject, attribute, typed value, predicate)` tuples with one-to-one matching; exact full assertion-set match per row. |
| Subject and attribute resolution | Correct associations divided by gold assertions, with omissions and extra assertions exposed by extraction precision/recall. |
| Compound rows | Complete assertion-set match and end-to-end label metrics on multi-atom rows. |
| Time reasoning | End-to-end metrics on changing slots, large evidence-date gaps, scheduled-versus-held events and interval-overlap cases. |
| Context | Metrics on explicit-subject versus implicit-subject rows. |
| Value semantics | Metrics by value kind, bounds, negation, unit conversion, date precision and billing period. |
| Robustness | Paired correctness and label stability under verified equivalent aliases, units and paraphrases; date-counterfactual accuracy where an independent timeline determines the new answer. |

Report counts for every slice; do not average tiny slices into a new headline score. Gold forms can define evaluator-side slices but must not reach the deployed method for held-out rows. Gold-form reasoner experiments bypass extraction and belong in a separate privileged-diagnostics table.

For robustness, consistency alone is insufficient: an always-true classifier is perfectly invariant. Report accuracy on the transformed cases as well. Test shuffled input order as a separate no-order-assumption stress condition; the contract provides no document boundaries, so do not declare row order a guaranteed source of context. A date change or arbitrary number mutation does not automatically give a known new truth label.

Run component ablations, for example base rules, rules plus learned parsing, then learned parsing plus temporal inference. Keep everything else fixed and report paired metric changes. A component earns its complexity when it improves measured failures without unacceptable regressions.

## 6. Fix the evaluation conditions before comparing methods

Use one versioned manifest of row IDs and access policies for every method.

| Evaluation track | Purpose | What is hidden |
| --- | --- | --- |
| Grouped in-period holdout | Main generalization comparison with dated anchors across the year | Test labels, test forms and all assets derived from them. Keep duplicate/template families and synthetic descendants together. |
| Forward temporal stress test | Future-change and stale-evidence behavior | Training evidence dated on/after 1 September 2026; evaluate later rows. Keep separate from in-period results. |
| Controlled semantic tests | Known-answer temporal, compound, type and proof failures | Independent world state and oracle answers. |
| Full-practice integration | Container and output readiness | Not an independent generalization test when full practice assets are packaged. |

The temporal cutoff above matches the existing local baseline; the PDF's separate experiment used 3 September. Choose one frozen manifest and rerun methods on it. Do not compare historical numbers from different splits as if they were paired experiments.

For the main track, use fixed grouped folds, preferably five if class counts allow, and pool out-of-fold predictions. Form groups from exact/near-duplicate templates and attach every synthetic descendant to its source before splitting. There are no document IDs, so these groups approximate dependence rather than reconstruct known documents. Publish the grouping rules, group sizes and class counts.

Calibrators, thresholds and tuning require an inner training/validation split; outer held-out rows remain untouched. Rebuild parser resources, vocabularies, evidence stores, compiled timelines and synthetic data from the permitted training rows for each fold. Every curated factual asset needs provenance so prohibited held-out facts can be excluded. Hiding a label while retaining its gold form or a timeline compiled from it is leakage.

If a method can jointly inspect held-out raw text as unlabelled input, allow the same access for all methods and label the main track **transductive batch**. Unlabelled text is not a trusted anchor. For strict forward forecasting, do not expose future batches while preparing earlier predictions. Grouped and forward experiments test different questions and should remain separate.

A strict unseen-fact-key test is optional and harsher. Grouping only the entire set of keys in a compound row does not prevent individual keys appearing in both partitions. Enforce disjoint keys through connected groups or quarantine rows crossing the partition, and report the resulting sample loss. Do not silently score familiar keys as unseen.

Save the dataset hash, split manifest, source commit, artifact hashes, random seed, evidence policy, threshold, dependency versions and hardware. Report paired differences between methods with 95% intervals, resampling dependence groups rather than pretending every related row is independent. Use suitable binomial or group-bootstrap intervals for decision-error rates and show raw counts. There are only 39 gold-`unsure` practice rows, so its estimates will be noisy.

Practice data already used to design hand rules cannot become a pristine final test through a new random split. Label such results retrospective development estimates. Reserve genuinely unseen annotations or a freshly curated test set for the strongest future generalization claims. The split methodology follows standard guidance on grouped, temporal and nested evaluation. [scikit-learn cross-validation guide](https://scikit-learn.org/stable/modules/cross_validation.html).

## 7. Treat operational requirements as gates

Before calling a method submission-ready, require complete prediction IDs, exactly 1,000 claim slots, valid confidences, full schemas, controlled grammar and runtime-input novelty. Confirm the offline amd64 container runs with 2 CPUs, 6 GiB RAM, user 10001 and a read-only root, publishes a complete CSV atomically within 30 minutes, and respects image, process, temporary-storage and output-size limits.

Measure the whole approximately 20,000-row batch, including startup, extraction, reasoning, generation, verification and writing. Peak container memory should include child processes and charged temporary storage. For the live frontend report p50/p95 row latency and cold-start time, stating batch size and cache policy. Repeating rows is useful for throughput stress but does not simulate unseen semantic difficulty; label the load construction.

The supplied checker is necessary integration evidence, but it does not check all these properties. In particular, its printed classification counters exclude gold-`unsure` rows from right/wrong decisions, and it does not establish generated truth or complete text/form agreement. Existing `baseline.py` already computes macro F1, per-class metrics, a full confusion matrix and decided precision including gold-`unsure` errors; it remains a text-baseline control rather than a generation or proof benchmark.

## 8. How to select a future method

Keep always-true and the existing text baselines as controls. For each candidate, report one row per evaluation track:

```text
method, dataset/split version, contract status,
macro F1, false precision/recall, unsure F1,
coverage, actual selective risk, chosen threshold,
correctness Brier, grounded decision yield,
verified valid yield@1000 OR explicitly labelled supported/unverified yield,
decision/claim proof replay rates, assumption-bearing rate,
full-batch seconds, peak RAM, live p95 latency
```

Use a Pareto comparison: a method clearly improves another when it improves a relevant dimension without a material regression on the others under identical conditions. If classification improves while verified claim yield falls, retain the tradeoff explicitly. Agree operational preferences before selecting a winner. Add a weighted utility only after weights and mistake/abstention costs are deliberately chosen or published; do not label a locally chosen utility the official contest score.

The first implementation priority for evaluation is the shared split manifest and per-row output log. Next add label/risk/confidence metrics, independent generation checks and replayable proof auditing. These make future architecture comparisons reproducible and explain why a change helped.

## Sources

- User-supplied analysis: `/Users/bogdandura/Downloads/hype-arena-analysis.pdf`, especially pages 9 and 14-16.
- Local task and contracts: `README.md`, `contracts/README.md`, `contracts/claim-forms.md`, `contracts/value-forms.md`, their schemas and `scripts/check_output.py`.
- [FEVER shared task, 2018](https://aclanthology.org/W18-5501/): correct labels accompanied by complete evidence.
- [ERASER, 2020](https://aclanthology.org/2020.acl-main.408/): explanation alignment and faithfulness.
- [Optimal Strategies for Reject Option Classifiers, 2023](https://www.jmlr.org/papers/v24/21-0048.html): coverage and selective-error tradeoffs.
- [scikit-learn calibration guide](https://scikit-learn.org/stable/modules/calibration.html): reliability checks, calibration data and proper-score caveats.
- [scikit-learn cross-validation guide](https://scikit-learn.org/stable/modules/cross_validation.html): held-out, grouped and temporal evaluation.
