# Madalin claim-generator integration

The imported `arena/` subset is byte-for-byte source from the team's `madalin`
branch at commit `5fcec3f46c63c6ddf07f3c4178cc46be0c59e1c1`. The repository,
commit and SHA-256 of every imported file are recorded in `madalin-source.json`.
The evaluation runner verifies these hashes before a comparable run.

Imported components are the typed value/lexicon utilities, evidence interval and
claim candidate models, local schema/grammar referee and generator. The teammate's
training, classifier, full-practice gold recovery, server, Docker configuration
and runtime pipeline were not imported. The original sklearn and NumPy versions
remain unchanged; pinned JSON Schema dependencies were added for the referee.

## Classification experiment

`experiments/fact_retrieval_predict.py` adds a separate candidate-feature index to
the existing original-plus-number-masked retrieval classifier. It compares two
predeclared approaches:

| Mode | Candidate construction | Temporal metadata |
| --- | --- | --- |
| `observed` | Render permitted true-row atomic annotations into canonical wording using teammate utilities | Original source observation dates, with scope uncertainty retained |
| `generated` | Run teammate `enumerate_candidates` and `select_claims` using permitted true annotations | Contract as-of date, event dates, transitions and relation assumptions |

Each index is capped at 1,000 candidates. The generated mode may contain fewer
than 1,000 because source support, grammar, deduplication and generator caps can
limit its output. Candidate volume is not a verification metric.

Generated references remain `unverified_auxiliary`. They are not independent facts,
positive labelled training targets, trusted observations or submission-ready
claims. They contribute numeric classifier features such as similarity by claim
type, entity overlap, typed value agreement, source-date gaps and temporal
compatibility indicators. Model probabilities remain uncalibrated.

## Provenance and leakage boundary

The evaluator supplies only permitted training forms. Each is accepted only after
matching the labelled runtime row's ID, text, date and label. Pending forms are
rejected. During the classifier's five-fold training procedure, each fold builds
its own candidate catalogue from its other four folds. Neither a training query's
form nor its label enters its candidate generator. The full training catalogue is
used only for pending inference.

Every candidate records source IDs, ID/text/date SHA-256 fingerprints, source
observation dates, type, form and assumptions. Only true labelled reference rows
can supply its candidate premises; false compounds are not negated into facts,
and unsure/unlabelled rows add no positive observations. The evaluator audits all
catalogues and every exposed candidate neighbor against its permitted source pool.

Current `fact` and `compare` candidates retain the generator's 29 October 2026
as-of semantics; they are not silently assigned to historical query dates. Event
dates are read from the same latest observed event-date value used by the
generator. Relations expose actual same-date premise intersections separately
from the generator's overlap assumptions. Observation source dates never prove
continuity over an unobserved interval.

The teammate generator uses interval compaction, recent-value as-of rules and
bounded relation interval extensions. These assumptions are retained in metadata
and do not count as independent truth verification. Our provenance audit checks
source admissibility, not whether the reconstructed world is true.

## Run and inspect

```sh
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests
.venv/bin/python evaluation/run_fact_integration.py --run-id NEW-UNIQUE-ID
```

The evaluator first compares the unchanged retrieval control and both candidate
modes on the frozen inner validation rows, then freezes selection before test
inference. All three are refitted on the outer training pool and evaluated on
the same stratified, temporal and checker test IDs under `fixed_050`.

For prepared split files, the method CLI is:

```sh
.venv/bin/python experiments/fact_retrieval_predict.py \
  --input /path/to/sanitized-runtime.csv \
  --training-forms /path/to/permitted-training-forms.csv \
  --lexicon data/lexicon.json \
  --mode generated \
  --output /tmp/fact-retrieval-predictions.csv \
  --artifacts /tmp/fact-retrieval-artifacts
```

The compressed `claim_catalogues.json.gz` artifact contains the five fold-specific
catalogues and the full-training catalogue. Each includes its exact reference ID
pool. `claims_summary.json` contains counts and assumptions. Pending and out-of-fold
traces retain candidate neighbor IDs; ordinary retrieval artifacts are preserved.

`scored_predictions.csv` and held-out gold belong to the evaluator. Input
separation uses subprocesses but has no OS-enforced filesystem sandbox.
Independent proof/truth verification, a submission output with exactly 1,000
accepted claims, frontend integration and constrained Linux benchmarking remain
unavailable in this experiment.

Results are documented in
[the integration experiment report](../research/madalin-fact-integration.md).

## Daria reader, timelines and sklearn alternatives

`evaluation/run_daria_sklearn.py` extracts a pinned source snapshot from Daria's
latest reviewed commit `5943f61`, preserving the current checkout and excluding
packaged calibration/practice assets. It rebuilds the reader, parse calibration,
temporal windows and HMM separately for each permitted training pool. The shared
inner validation IDs choose among the native stacker and 54 predeclared sklearn
feature/model configurations. All candidates are then fitted on the full outer
training pools and measured on all three frozen tracks under `fixed_050`.

```sh
.venv/bin/python evaluation/run_daria_sklearn.py --run-id NEW-UNIQUE-ID
```

The sweep includes three-label linear models, calibrated margin classifiers,
RBF SVM, forests, boosting, Gaussian naive Bayes, nearest neighbors and a small
MLP. Features range from the original three stacker inputs to richer temporal
states, parse confidence and atom counts, optionally combined with word or
character TF-IDF. Standard, robust and quantile preprocessing, numeric masking
and interaction features are declared before evaluation. All learned transforms
fit inside the current training pool.

For new runtime-shaped data, choose an already validated candidate explicitly:

```sh
.venv/bin/python integrations/daria_sklearn_predict.py \
  --source-dir results/comparisons/daria-sklearn-20261028-001/source \
  --input /path/to/runtime.csv \
  --forms /path/to/training-only-forms.csv \
  --candidate expanded_none_histgb \
  --output /tmp/daria-sklearn-predictions.csv
```

`--candidate` accepts the frozen configuration IDs; `native` runs the unchanged
decision stacker. The input has exactly `id,text,date,label` columns, with blank
labels for pending rows. The forms file has exactly `id,text,date,label,form`
columns and must cover only the labelled runtime rows, with matching source
fingerprints. The generic prediction CLI reserves the first 80% of labelled IDs
for parse-calibration fitting; the benchmark runner supplies its explicitly
recorded nested/shared partitions instead. These calibration policies differ.

All probabilities are estimates, not evidence proofs. Native internal HMM OOF
features retain permitted training forms while hiding their labels; reader
extraction is not fully cross-fitted inside that stage. Complete outer/inner
validation forms remain inaccessible to the method. This experiment evaluates
classification only; the upstream writer, generated truth and constrained
container performance are not evaluated.

## Additional feature representations

`evaluation/run_daria_features.py` compares 24 declared feature configurations
using the previously CV-tuned gradient-boosting parameters, held fixed per track.
It rebuilds the reader and rich feature caches for every original CV fold, selects
the representation using only the shared fit pool, then evaluates shared
validation and all three outer tracks. Original features and baseline predictions
are checked against the preceding runs before results are compared.

```sh
.venv/bin/python evaluation/run_daria_features.py --run-id NEW-UNIQUE-ID
```

The new representations include order-invariant assertion summaries, typed
assertion histograms, subject/attribute bags, true-anchor observation counts and
date gaps, binary word counts, word-boundary character TF-IDF, chi-square feature
selection, SVD latent text features, feature hashing, F/MI feature selection and
scaled PCA. Most text candidates add their features to the original temporal
features; two text-only controls measure text signal separately. Learned text
vocabularies, selectors and reductions fit only on the current training pool.

These experiments change classifier inputs, while Daria's reader and temporal
models remain the same. Support counts and observation gaps are probabilistic
features, not evidence proofs or persistence guarantees. Feature-specific
classifier hyperparameters are not retuned in this comparison.
