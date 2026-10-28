# Text classification baseline

This starter trains three small classifiers on the labelled practice sentences: a majority-class baseline (`always_true`), TF-IDF plus logistic regression (`tfidf_lr`), and class-weighted TF-IDF plus logistic regression (`tfidf_lr_balanced`). It is a classification-only experiment, not a complete Arena submission.

These are legacy reference results recorded before [the shared evaluation protocol](EVALUATION.md). The saved outer split IDs match [the frozen manifest](evaluation/splits-v1.json), but this runner does not implement the complete `arena-eval-v1` artifact contract, sanitized inference boundary, validation policy, generation audit, proofs or resource benchmark. Follow `EVALUATION.md` for future method comparisons.

## Set up and run

Use Python 3.12 and the pinned dependencies in `requirements.txt`:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python baseline.py evaluate --output-dir results/evaluations/controls/checker-rerun-001/legacy
```

The default evaluation uses the checker split. Evaluate all three splits with:

```sh
.venv/bin/python baseline.py evaluate --split all --output-dir results/evaluations/controls/all-splits-rerun-001/legacy
```

The original metrics, split IDs and nine row-level prediction files are preserved in `results/baseline/`. Do not overwrite them. Always set `--output-dir` to a new, unused experiment directory; replace the example run IDs above on subsequent runs. The legacy default, when `--output-dir` is omitted, overwrites `results/baseline/`.

Within the chosen output directory, the runner writes `metrics.json` and a `predictions/` subdirectory. Those legacy prediction files include gold labels for scoring and are not sanitized method outputs. For a separate experiment:

```sh
.venv/bin/python baseline.py evaluate --split all --output-dir results/evaluations/controls/my-experiment-001/legacy
```

To run the unit tests:

```sh
.venv/bin/python -m unittest discover -s tests
```

## Baseline results

Recorded in `results/baseline/metrics.json` with seed 42 and minimum confidence 0.5. The checker split holds out every fifth row; stratified uses an 80/20 stratified split with seed 42; temporal trains on dates before September 1, 2026 and tests on that date or later.

| Split | Model | Accuracy | Macro F1 | False recall |
|---|---|---:|---:|---:|
| Checker | always_true | 0.853 | 0.307 | 0.000 |
| Checker | tfidf_lr | 0.858 | 0.334 | 0.041 |
| Checker | tfidf_lr_balanced | 0.711 | 0.384 | 0.163 |
| Stratified | always_true | 0.834 | 0.303 | 0.000 |
| Stratified | tfidf_lr | 0.834 | 0.303 | 0.000 |
| Stratified | tfidf_lr_balanced | 0.650 | 0.405 | 0.273 |
| Temporal | always_true | 0.818 | 0.300 | 0.000 |
| Temporal | tfidf_lr | 0.818 | 0.300 | 0.000 |
| Temporal | tfidf_lr_balanced | 0.597 | 0.292 | 0.058 |

`tfidf_lr_balanced` catches more false rows on checker and stratified splits, while lowering accuracy. Ordinary logistic regression nearly always predicts `true` and matches the majority baseline on the stratified and temporal splits. These results show that the text-only model is learning lexical patterns, not factual reasoning about the fictional universe.

The pipeline uses sentence text as its only model feature. IDs identify rows, dates define the temporal split, and forms and the lexicon are excluded as features. Reported probability scores are uncalibrated and are not verified truth probabilities. The fixed 0.5 confidence policy leaves native `unsure` predictions as `unsure` and abstains to `unsure` when a `true`/`false` winner is below 0.5; do not treat this as calibrated confidence.

## Predict unlabeled rows

When `sample/sample.csv` is released, predict its blank-label rows using the labelled practice data:

```sh
.venv/bin/python baseline.py predict --input sample/sample.csv --output outputs/sample-labels.csv
```

For another CSV, pass its path with `--input`; it must have `id`, `text`, `date`, and `label` columns. The command trains on labelled rows from `--train` (default `data/practice.csv`) and emits classifications only for rows with blank labels. Its default model is `tfidf_lr`; pass `--model tfidf_lr_balanced` to try class balancing. The output uses Arena label columns but contains no generated claims.

Use `evaluate` to measure performance. Running `predict` on masked practice rows while training on the full practice file would expose their labels during training. The fixed splits are development benchmarks, not a final untouched test set; the metrics record identical-text overlap between partitions.

## Scope and skill references

This baseline does not generate the required 1,000 claims, provide a frontend, or include the Docker runner; those parts of the Arena system remain incomplete. For modeling reference, see the [K-Dense scikit-learn skill](https://github.com/K-Dense-AI/scientific-agent-skills/blob/main/skills/scikit-learn/SKILL.md). The [official OpenAI Jupyter Notebook skill](https://github.com/openai/skills/blob/main/skills/.curated/jupyter-notebook/SKILL.md) is relevant if this experiment is later presented as a notebook. Neither skill was installed while building this baseline; a notebook is not needed to run this script. The branch now includes team-provided development skills under `.agents/skills/`.

The first set of text-model variations and their shared-protocol results are documented in [research/text-model-experiments.md](research/text-model-experiments.md).
