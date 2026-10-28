# Experiments

Each paper idea is a hypothesis. We write the prediction down **before** running it,
then record what happened. Unless noted, runs use the practice data, 5 page-grouped folds
(seed 0), `--mode gold` (gold atoms, labels hidden: the truth model alone, parsing assumed
perfect). Held-out rows stay in the input as unlabelled rows.

Run: `python3 -m evaluation.run [--models ...] [--mode gold|parsed]`
Tests: `python3 -m unittest tests.test_values`

## E0 Baselines (2026-10-28)

| model | acc | 95% CI | macro F1 | F1 false | Brier | ECE | acc on top 50% conf |
|---|---|---|---|---|---|---|---|
| always_true | 0.833 | 0.811-0.853 | 0.303 | 0.000 | 0.284 | 0.001 | 0.828 |
| latest_before | 0.729 | 0.696-0.765 | 0.438 | 0.492 | 0.240 | 0.061 | 0.949 |
| nearest | 0.780 | 0.750-0.812 | 0.473 | 0.560 | 0.219 | 0.034 | 0.967 |

Matches an independent measurement (latest-before ~77%, nearest ~79% with all other rows
as anchors), so the harness is trustworthy.

**Where they fail (nearest):**

| branch | rows | gold true | gold false | gold unsure |
|---|---|---|---|---|
| all atoms match an anchor | 1095 | 1061 | 22 | 12 |
| some atom mismatches | 612 | 350 | 249 | 13 |
| no anchor for some atom | 190 | 170 | 6 | 14 |

- A match is reliable (97% true).
- A mismatch is wrong more often than right. Typical case: a January page says
  "Hooly Docs 1.0", the nearest anchor is from August and says 2.0. The value changed in
  between; the row is true. A mismatch against a distant anchor means "changed", not "lie".
- No evidence: 89% true, so the prior should decide.

## E1 Four-date windows (2026-10-28)

`hooly/temporal.py`, `TemporalWindows`. Per fact, labelled-true values are grouped into
runs; a run's first..last sighting is its certain core, the gap between runs is where the
change happened. Each atom gets a state (in a core / in a gap, matching old, new or neither
value / before the first or after the last sighting, by distance). Label probabilities per
state are measured on training rows with their own page's anchors left out; rare states back
off to coarser ones. No hand-tuned numbers.

| model | acc | 95% CI | macro F1 | F1 false | Brier | ECE | acc on top 50% conf |
|---|---|---|---|---|---|---|---|
| always_true | 0.833 | 0.811-0.853 | 0.303 | 0.000 | 0.284 | 0.001 | 0.828 |
| nearest | 0.780 | 0.750-0.812 | 0.473 | 0.560 | 0.219 | 0.034 | 0.967 |
| **windows (E1)** | **0.889** | 0.872-0.907 | 0.515 | 0.607 | 0.173 | 0.020 | 0.968 |
| windows_plus (E1b) | 0.893 | 0.877-0.909 | 0.535 | 0.667 | 0.170 | 0.019 | 0.976 |

**Prediction held:** E1 is the first variant to beat always_true, outside the error bars.
Inside a core the states are clean: match 299/299 true, mismatch 48/48 false.

Remaining errors (E1): 144 missed lies, 27 false alarms, 39 unsure never predicted. Missed
lies sit almost entirely in mismatches outside the observed range (before the first / after
the last sighting); false alarms mostly in values outside both neighbours on metrics that
bounce around.

**E1b** adds two learned modifiers: stability of the fact (stable / volatile / thin) and
whether the claimed value fits the attribute's usual direction of change (learned per
attribute from the anchors: users go up, latency goes down; versions always up).

| state outside the cores (single-atom test rows) | n | % false |
|---|---|---|
| before the first sighting, against the trend | 20 | 0.80 |
| after the last sighting, against the trend | 10 | 1.00 |
| before the first sighting, fits the trend | 26 | 0.12 |
| after the last sighting, fits the trend | 20 | 0.35 |
| stable fact, mismatch before the first sighting | 16 | 0.25 |
| thin fact, no trend, before / after | 76 / 58 | 0.21 / 0.26 |

- **Direction: prediction held.** Against-trend values are inflated or stale lies.
- **Stability: prediction failed.** I expected mismatches on stable facts (specs, launch
  dates) to be lies; most are true. Kept only as a modifier the data can ignore.
- The largest unresolved bucket is thinly observed facts with no trend (134 rows, ~23%
  false): the timeline alone cannot decide them. That is the job of the unlabelled rows and
  credibility (E3-E5).
- The overall gain of E1b is +0.4 points, inside the error bars: "probably better, not proven".

## Planned experiments and predictions

| id | idea | source | prediction |
|---|---|---|---|
| E1 | Four-date windows | TAC KBP temporal slot filling | first variant to beat always_true. **Held: 0.889** |
| E2 | Gradual-change smoothing | DynaTD (KDD 2015) | worse than E1: Hooly changes in steps |
| E3 | Iterative source/truth voting with unlabelled rows | TruthFinder-style | small gain at best: no source ids |
| E4 | Page exactness + freshness | Dong et al. (VLDB 2009) | freshness catches stale lies exactness misses |
| E5 | Credibility from text features, trained on labels | SLiMFast (2017) | best trust signal: labels exist, source ids don't |
| E6 | Rule outputs combined by a label model vs supervised logistic regression | Snorkel / FlyingSquid | supervised LR >= label model |
| E7 | Verdict policy: unsure-unless-proved vs most-likely class | evidence policy vs proposal | most-likely class wins on accuracy |
