# Shared evaluation instructions for all agents

Protocol: **arena-eval-v1**. All agents implementing or comparing methods MUST follow this document. `MUST` denotes a requirement; `SHOULD` denotes a recommendation. This defines our local comparison procedure, not the unpublished competition scoring formula.

## 1. Read first and keep one shared benchmark

Read `README.md` and `contracts/README.md` for the task, runtime and output rules. Read the claim/value grammars and schemas when implementing generation. The methodological rationale is in [research/evaluation-framework.md](research/evaluation-framework.md); this document supplies the operational defaults.

Use [evaluation/splits-v1.json](evaluation/splits-v1.json) as the source of row IDs. Do not independently regenerate random splits, move difficult rows, or substitute the PDF's different temporal cutoff.

Verify these SHA-256 hashes before any comparable run:

| File | SHA-256 |
| --- | --- |
| `data/practice.csv` | `52743befaa0fa38b89b5c802c0f953d242ae027c86839c9c0f05a9e69a86d8e9` |
| `data/lexicon.json` | `d0fd5e9bdc6b33ad3ec160001f50e00e7f8bf2b25a92779022f893db7c3d5507` |
| `evaluation/splits-v1.json` | `9699327af7bf80813eabb66a582593d5152683ee98abf36090a252b0b2723860` |

A mismatch makes the run incompatible with v1. Create a new shared benchmark version if inputs, splits, metric conventions or policies must change, and rerun the controls. Do not silently change v1 or mix versions in a leaderboard.

Use seed **42** for the required run and label order **`["true", "false", "unsure"]`** everywhere. Record any remaining nondeterminism. Additional seeds are supplementary runs; never select only the best seed.

## 2. Required evaluation tracks

Run all three tracks for every method that predicts labels:

| Track | Outer train | Test | Purpose |
| --- | ---: | ---: | --- |
| `stratified` | 1,517 | 380 | Primary retrospective in-period development comparison. |
| `temporal` | 1,398 | 499 | Forward stress: training labels before `2026-09-01`, test rows on/after that date. |
| `checker` | 1,517 | 380 | Compatibility control matching every-fifth-row masking. |

The exact arrays in the manifest take precedence over descriptions. Preserve original CSV order when materializing files; join predictions by ID rather than position.

These are development benchmarks. Random/checker partitions can share wording and related facts. The practice corpus has already informed development, so these tracks MUST NOT be described as pristine unseen generalization tests. The temporal track is a stress test, not an estimate of the official run's sampling.

Grouped-template evaluation is not included in v1. If it is added, first freeze one shared grouping algorithm and manifest as a new version/track and rerun every control. Agents MUST NOT invent different template groups and compare the resulting scores directly.

## 3. Data and evidence access

The standard access profile is **`batch_unlabelled_v1`**:

1. A method may learn from the current split's permitted training `id,text,date,label,form` rows and the public lexicon. Forms teach what a sentence asserts; false forms do not become true facts.
2. At inference, the method receives permitted labelled training rows and all test rows together in runtime shape `id,text,date,label`. Test labels are empty; test forms are absent. Test text may be used jointly as unlabelled observations, never automatically as trusted truth.
3. Gold test labels and forms belong only to the evaluator. They MUST NOT reach model fitting, parser mining, factual timelines, lookup tables, calibration, confidence thresholds or claim selection.
4. Rebuild every learned or curated asset per split from permitted sources. A full-practice cache, ID lookup, synthetic descendant or timeline compiled using held-out answers is leakage even if the test CSV label is blank. Match known rows by checked ID/text/date, not an arbitrary reused ID.
5. LLMs may assist development as the task permits; split-specific training assets MUST obey the same data boundary. The submitted runtime cannot be an LLM. Do not import external facts about real companies/products into the fictional universe.
6. The method runner MUST use sanitized split files and split-scoped assets, not open the full gold practice CSV during inference. Keep evaluator answer files outside its input directory. Record any inability to enforce that separation.

The temporal v1 track also exposes its pending batch as unlabelled text. Call it a **forward batch stress test**, not strict online forecasting. A no-future-text or no-batch-access experiment is a separate access profile and cannot silently replace this one.

Methods have equal permission to use the allowed training sources; choosing not to use forms is a design decision. Declare added training material, human-written rules, pretrained assets and all factual assumptions in `run.json`. Unknown provenance or held-out-answer contamination makes the result ineligible for a clean improvement claim.

## 4. Decision policies

### Mandatory: `fixed_050`

Use the manifest's complete `train_ids` as the allowed training pool. All methods MUST report this policy before reporting a tuned policy.

- Preserve a native `unsure` decision.
- Emit a candidate `true` or `false` only if its correctness confidence is at least **0.50**.
- Otherwise emit `unsure` with empty CSV confidence / JSON `null`.
- Emitted T/F confidence MUST be finite and in `[0.5, 1.0]`.
- Do not clamp confidence upward, interpret cosine similarity as probability, or invent class probabilities from a scalar score. If an uncalibrated score is used as confidence, label it uncalibrated and do not claim probabilistic reliability.
- Any fitting, tuning or calibration must stay inside `train_ids`; use the shared inner partitions when holding out validation data. No test-driven threshold or hyperparameter selection is permitted.

The threshold controls the emitted label, so compute all metrics after applying it. This matches the existing baseline's decision policy.

### Optional: `validation_5pct`

This is a separate comparison lane. Use only manifest `fit_ids` for fitting parser/model/factual assets; `validation_ids` may be used for calibrating scores and selecting the threshold. Validation gold labels/forms MUST NOT enter the factual evidence store, learned parser or training augmentation. Validation text is pending unlabelled input during validation, and MUST NOT remain as extra runtime evidence in the final test run. Do not refit on validation rows after calibration.

This policy preserves native `unsure` decisions and applies its selected threshold to the candidate T/F correctness confidence after any calibration. Final test inference receives labelled evidence only from `fit_ids`, plus pending unlabelled test rows; the outer training pool is not an additional evidence source. All output-validity and abstention rules from `fixed_050` still apply.

| Track | Fit rows | Validation rows |
| --- | ---: | ---: |
| `stratified` | 1,212 | 305 |
| `temporal` | 1,279 | 119 |
| `checker` | 1,212 | 305 |

After any calibration is fitted on validation data, evaluate candidate thresholds `0.50, 0.51, ..., 1.00` on that same validation partition. This is model/operating-point selection, not an unbiased validation performance estimate. A separate test set measures the resulting policy.

A threshold qualifies when it gives at least **30** T/F validation decisions and empirical decision error at most **0.05**. Choose the qualifying threshold with the greatest decided count; break ties by lower error, then higher threshold. If none qualifies, report `target_not_attained`, emit all test labels as `unsure` for this policy and set the threshold to `null`.

Freeze the chosen threshold before test inference. Report actual test error and coverage even if test error exceeds 5%. The validation target and 30-decision minimum are local conventions, not statistical guarantees or organizer rules. The small temporal validation set limits confidence in this policy.

Rerun controls with the same fit/validation access for this lane. Historical controls trained on the complete outer training pool are not paired controls for it. Optional policy results MUST NOT replace `fixed_050` results.

## 5. Normalize outputs and expose failures

The evaluator MUST join against **every canonical test ID**, including gold `unsure` rows. Do not evaluate only rows a method answered.

- Missing IDs: fail completeness; use `unsure` only for diagnostic label metrics.
- Duplicate IDs: fail uniqueness. For diagnostic metrics, the first occurrence counts, matching the competition's first-row convention.
- Unknown IDs: fail the local output gate and record their count; exclude them from the canonical test denominator.
- Invalid labels, missing/nonfinite/out-of-range T/F confidence or nonempty `unsure` confidence: fail output validity. Normalize that row to `unsure` only for diagnostic metrics and record the error.
- Malformed CSV: fail output validity. Do not publish a comparable score from an unparseable table.
- Extra test-form, truth-label or source columns in method inputs: fail the data-boundary audit.

Diagnostic scores from invalid/incomplete output MUST carry `eligible_for_comparison: false`; they cannot establish an improvement. The official contract may ignore some errors that this local comparison flags.

In JSON use `null` for undefined or unmeasured quantities, never NaN/Infinity, an empty string, or a fabricated zero. Keep measurement status and reason alongside missing metrics.

## 6. Required label and confidence metrics

Let `N` be all test rows, `y` the gold label and `h` the emitted label. For each of the three classes calculate one-versus-rest TP/FP/FN.

```text
accuracy = correct emitted labels / N
precision_k = TP_k / (TP_k + FP_k)
recall_k = TP_k / (TP_k + FN_k)
F1_k = 2 TP_k / (2 TP_k + FP_k + FN_k)
macro_f1 = (F1_true + F1_false + F1_unsure) / 3
```

For v1, use zero when a precision/recall/F1 denominator is zero (`zero_division=0`), always retaining all three labels. Also publish each class's support and flag absent classes. All v1 outer tests contain all three classes.

Required fields:

- `rows`, `accuracy`, `macro_f1`.
- `per_class`: precision, recall, F1 and support for all three classes; highlight false precision/recall and unsure F1 in reports.
- `confusion_matrix`: gold rows, predicted columns, in fixed label order.
- `decided_count`, `decided_correct`, `decided_errors`, `decided_coverage`, `decided_accuracy`, `decided_risk`.
- Decided counts and coverage within each gold class; internal abstention-reason counts.
- `correctness_brier` and the reliability bins below.

```text
D = test rows emitted as true or false
decided_coverage = |D| / N
decided_risk = count(h != y within D) / |D|
decided_accuracy = 1 - decided_risk
correctness_brier = mean within D of (confidence - 1[h == y])^2
```

T/F on gold `unsure` is an error. Gold `unsure` is a target class; model abstention is an internal reason, not a fourth target class. If `D` is empty: coverage is zero; decided accuracy, risk and Brier are `null`.

Reliability bins are `[0.5,0.6)`, `[0.6,0.7)`, `[0.7,0.8)`, `[0.8,0.9)`, `[0.9,1.0]`. For each, store count, mean confidence and empirical correctness; empty bins have count zero and means `null`. Brier and reliability refer to the chosen label's correctness, not `P(true)`.

If genuine full probabilities are available for every test row, also report:

```text
multiclass_brier = (1/N) sum_i sum_k (p_i,k - 1[y_i == k])^2
```

Use no factor of one half; the range is 0 to 2. Probabilities must be finite, nonnegative, sum to one within `1e-6` and follow the declared class order. Missing distributions make this optional metric `null`, not a guessed distribution. Report any clipping convention for optional log loss.

Compare confidence metrics alongside coverage. A shared-decided-subset analysis is useful; differing selected populations prevent interpreting conditional Brier alone as a pure calibration ranking. Test risk-coverage curves are descriptive, not permission to retune on test labels.

## 7. Generation and independent truth verification

Run generation with the same permitted evidence as classification. Use the contract's temporal semantics: current facts/comparisons on **2026-10-29**, historical ordered changes, resolved event dates, and overlapping relationship intervals.

Check novelty against the full runtime input text (permitted labelled rows plus pending test rows), not merely the training file. In `validation_5pct`, validation-only evidence/text is not an additional runtime input. Keep proof metadata out of strict submitted forms.

For a complete-system run, audit the **first 1,000 claim positions**. Do not discard invalid claims and refill from later positions during scoring. Report actual generated count; counts other than 1,000 fail the complete-system count gate.

Audit each existing position independently for:

1. CSV/ID constraints, entry schema, lexicon membership and exact value types.
2. Grammar and text/form agreement.
3. Novelty against runtime input.
4. Truth under that type's temporal semantics.
5. Evidence/proof acceptance and disclosed assumptions.
6. Canonical semantic duplication, as an additional local diagnostic.

```text
verified_valid_yield_at_1000 =
  positions passing structure, text/form, novelty and independent truth checks / 1000
```

Missing and invalid positions earn zero. Report a mutually exclusive partition in this order: `missing`, `invalid` (structure/text/novelty failed), `contradicted` (otherwise valid but independently false), `verified_valid`, `unverified`. These five counts MUST sum to 1,000. Preserve all per-check flags so the partition does not hide multiple errors.

No public complete truth oracle exists. A generator checking its own store, finding no contradiction or emitting confidence 0.99 does not independently establish truth. If only some claims have independent verification, verified yield is a confirmed lower bound; the remaining `unverified` positions are not automatically false. If none has been independently audited, use `null` for verified yield and `status: "not_verified"`; expose any self-consistency/support proxy under a different field name.

Report verified yield by type and semantic unique yield separately. Canonicalization should resolve aliases and equivalent units, preserve date/version precision and billing periods, and treat `A > B` / `B < A` as one comparison. Semantic deduplication is our anti-padding diagnostic; the official contract explicitly requires text novelty and publishes no type-balance quota.

Controlled-world evaluation requires one shared versioned fixture suite and an independent truth oracle. Such a suite is **not supplied by this document**: mark it `not_run` until it exists. Agents MUST NOT compare different self-created worlds as the same benchmark or call synthetic success proven Hooly truth.

A classification-only method is allowed in the classification comparison. Set generation `status: "not_implemented"` and unmeasured generation metrics to `null`; label it classification-only and do not call it submission-ready. A complete-system run that actually produces no claims has 1,000 missing positions and yield zero.

## 8. Explanation and proof checks

Emit a proof sidecar for each decided row and generated claim when the method supports it. Each record MUST contain:

```text
output_id, output_kind, queried_date_or_temporal_condition,
parsed_assertions, source_row_ids_and_fingerprints, source_dates,
normalized_premises, inference_rule_ids, intermediate_results,
assumptions, conflicts, confidence_inputs
```

An independent checker must validate source admissibility, interpretation, types, temporal conditions and every inference. A readable explanation or valid source ID alone does not pass. A false compound needs at least one justified contradiction; a true compound needs every required assertion supported.

```text
decision_proof_rate = accepted complete proofs / all emitted T/F decisions
claim_proof_rate = accepted complete proofs / existing first-1000 claim positions
grounded_decision_yield = correct T/F decisions with accepted proofs / N
assumption_bearing_rate = audited outputs requiring unproved assumptions / outputs audited
```

Use `null` on zero denominators. Missing proofs remain in the denominator when proof support is implemented. If no independent proof checker was run, proof metrics are `null` with `status: "not_verified"`; if proof functionality is absent, use `not_implemented`.

Accepted evidentiary proofs cannot rely on unjustified persistence, guessed event occurrence dates or guessed interval overlap. Operational replay under disclosed assumptions may be reported separately and MUST NOT be counted as assumption-free evidentiary acceptance. Log the reasoning actually executed; do not attach a post-hoc story and call it an execution trace.

SHOULD test sufficiency by replaying from only the cited evidence. For a known necessary-premise case, delete the premise and rebuild derived assets. A surviving conclusion is not automatically a failure if another valid proof exists. An unsure explanation should identify the unresolved condition, not claim universal unknowability.

## 9. Required artifacts for every method/run

Write to a new directory:

```text
results/evaluations/<method-id>/<run-id>/
  run.json
  report.md
  <policy>/<split>/
    metrics.json
    predictions.csv
    scored_predictions.csv
    claims.csv             # when generation ran
    claim_audit.csv        # when generation audit ran
    proofs.jsonl           # when proof traces were produced
```

Do not overwrite another run or `results/baseline`. Omitted optional files require explicit capability status in metrics.

`predictions.csv` is method output with columns `id,label,confidence,decision_reason`. Keep gold labels out of this file. Use reason codes `decided`, `model_unsure`, `ambiguous_parse`, `missing_evidence`, `temporal_conflict`, `low_confidence`, `unsupported`, or `other` with an explanation in the run metadata. Optional additional columns must not replace these.

`scored_predictions.csv` is evaluator-only: add `gold`, `correct`, `output_valid` and error codes after inference. Preserve every canonical test row. `claims.csv` uses the six contract columns. `claim_audit.csv` records position, ID, type, each check's status, independent truth/proof verdicts, assumptions, semantic key and error codes.

`run.json` MUST record:

- Protocol version, all three pinned hashes, method ID, run ID, seed and split/policy names.
- Exact command(s), source commit and source-file hashes covering uncommitted/untracked method code; dependency versions and model/asset hashes.
- Training sources and permitted row-ID sets, evidence access profile, factual-asset provenance and leakage audit result.
- Confidence semantics, calibration procedure, threshold and any threshold-selection result.
- OS/CPU/architecture, CPU and memory limits, process/thread limits and timing method.
- Implemented capabilities and any unavailable tests/verifiers.

`metrics.json` MUST include `benchmark_version`, `method_id`, `run_id`, `split`, `policy`, `access_profile`, `eligible_for_comparison`, `output_errors`, and sections `classification`, `generation`, `proofs`, `resources`. Each section has `status` in `ok`, `not_implemented`, `not_run`, `not_verified`, `invalid`, or `failed`, plus `reason` when not `ok`. A measured zero is different from an unavailable measurement.

## 10. Runtime and submission-readiness checks

Report method build/fit time, inference time, generation time, audit time and serialization time separately. Report total runtime from process start through complete output publication. Distinguish an external evaluator's audit time from checks performed inside the submitted runtime.

Comparable batch resource measurements require the same workload, target architecture and limits: offline Linux amd64, **2 CPUs**, **6 GiB RAM**, non-root user **10001**, read-only root, and the contract's process/storage limits. Measure container peak memory including child processes and charged temporary storage. The submitted run must publish a complete CSV within **1,800 seconds** by atomic rename.

For approximately 20,000-row stress testing, freeze and hash one shared workload first. No full-size workload is supplied here; use `not_run` until available. Repeating practice rows measures throughput only and requires unique IDs; disclose it as synthetic load rather than unseen semantic accuracy. Native macOS timing is allowed as supplementary data but is not directly comparable to an amd64 container run.

For the live system, report cold start and p50/p95 row latency under identical cache/batch/concurrency settings. Use at least 100 measured requests for a reported p95; otherwise mark it preliminary and state the count.

Before calling a method submission-ready, verify complete classification output, exactly 1,000 claims, full schema/grammar/novelty checks, the actual container limits, image/output sizes, deadline handling, and the required frontend. Truth verification status must remain explicit. Passing `scripts/check.sh` alone does not establish semantic correctness, full output coverage, full schema compliance or generated truth. Its right/wrong counters also omit gold-`unsure` decisions.

## 11. Report and compare results consistently

`report.md` MUST contain:

1. Method description and the single change being tested, with declared assumptions.
2. One row per split/policy: macro F1, false precision/recall, unsure F1, coverage, decided errors/count, risk, correctness Brier, grounded decision yield, generation status/yield, proof status/rate, runtime and peak RAM.
3. Full confusion matrix and reliability-bin table, or links to their metric artifacts.
4. Comparison to `always_true`, `tfidf_lr` and `tfidf_lr_balanced` under the same protocol/access/policy. Rerun controls when access or training pools differ. Report unavailable controls explicitly rather than invent scores.
5. At least five representative failure cases when errors exist, with IDs, gold/predicted labels, confidence and the failed parsing/evidence/time rule. Use failures for future development; do not rerun a tuned version on the same test and claim it remained untouched.
6. Known limitations, unsupported capabilities, leakage risk and exact reproduction command(s).

Use `stratified` as the primary **development** comparison, `temporal` to expose forward regressions and `checker` as compatibility evidence. Do not average these different tracks into one headline number. Compare methods only when benchmark hash, split, access profile and policy match. Show paired metric differences and error counts; confidence intervals must disclose resampling units and dependence assumptions.

Do not choose a winner using accuracy alone, decided accuracy without coverage, Brier without its selected population, or generated volume without truth status. Do not create an arbitrary weighted total and call it the official score. State classification, generation, proof and cost tradeoffs directly.

The existing `baseline.py` is a legacy classification runner. It already provides several core label metrics and the v1 outer splits, but it does **not** yet implement this complete artifact contract, frozen inner-validation lane, generation audit, independent proofs or resource benchmark. Adapters/evaluator implementation are separate work; do not claim this document has implemented them.

To reproduce the legacy `fixed_050` control predictions without overwriting their reference artifacts:

```sh
.venv/bin/python baseline.py evaluate --split all --seed 42 --min-confidence 0.5 --output-dir results/evaluations/controls/<unique-run-id>/legacy
```

Replace `<unique-run-id>` before running. Verify its recorded outer IDs against the shared manifest, then adapt/rescore the predictions to the required per-method artifacts; this command alone is not the complete v1 evaluator.

## 12. Agent completion checklist

- [ ] Verified pinned inputs/manifest and used the shared row-ID arrays.
- [ ] Rebuilt assets with only the permitted rows; recorded a leakage audit.
- [ ] Ran `fixed_050` on all three tracks, or explicitly marked each unavailable run.
- [ ] Preserved all test rows and all three target labels in metric denominators.
- [ ] Recorded confidence policy and actual risk/coverage without test tuning.
- [ ] Saved immutable run metadata, row predictions, metrics and report.
- [ ] Reported generation/proof capabilities and verification status honestly.
- [ ] Compared to matching controls and described material regressions.
- [ ] Distinguished measured results, development estimates and submission readiness.
