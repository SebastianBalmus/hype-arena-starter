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

## E5 Credibility from text style (2026-10-28)

`hooly/credibility.py`, `hooly/combined.py`. Logistic regression P(lie | text), numbers
masked (style, not values): words, bigrams, first person, superlatives, "!", round/huge
numbers, bound words, fragments. Combined with E1b by a small stacker trained on
out-of-fold scores.

| model | acc | 95% CI | F1 false | Brier | ECE |
|---|---|---|---|---|---|
| windows_plus (E1b) | 0.893 | 0.877-0.909 | 0.667 | 0.170 | 0.019 |
| credibility_only | 0.776 | 0.752-0.799 | 0.169 | 0.365 | 0.115 |
| windows_cred (E1b + E5) | 0.892 | 0.875-0.908 | 0.677 | 0.173 | 0.023 |

Text alone, page-grouped CV, true vs false rows:

| features | AUC | log loss (base rate 0.421) |
|---|---|---|
| style cues only | 0.53 | 0.418-0.421 |
| unigrams + cues | 0.55-0.57 | 0.47-0.76 |
| unigrams + bigrams + cues | 0.55-0.56 | 0.49-0.70 |

**Prediction failed.** Style carries almost no signal (AUC ~0.53-0.57, 0.5 = chance);
more regularisation only removes the overconfidence. Hooly's lies are written in the same
voice as its truths: a lie can only be caught by checking it against the record.

Lies cluster by page only weakly: P(false) is 0.117 when no other row of the page is false,
0.145 / 0.199 / 0.164 with 1 / 2 / 3+ false neighbours (base rate 0.146). So page-level
trust (E4) has little room either. Evidence has to come from other rows stating the same fact.

## E3 Unlabelled rows as votes (2026-10-28)

`hooly/evidence.py`. Rows of other pages stating the same fact within 45 days vote for or
against an atom's value (counted per page); single-fact rows labelled false stating the same
value within 60 days mark it as a repeated known lie.

| model | acc | 95% CI | F1 false | Brier | ECE |
|---|---|---|---|---|---|
| windows_plus (E1b) | 0.893 | 0.877-0.909 | 0.667 | 0.170 | 0.019 |
| windows_evidence (E3: votes as extra state modifiers) | 0.885 | 0.868-0.902 | 0.655 | 0.177 | 0.023 |
| **windows_votes (E3b: votes weighed by a stacker)** | **0.894** | 0.878-0.909 | **0.683** | 0.172 | 0.033 |

The signal is real (single-atom test rows): other pages disagree -> 34% false; agree -> 6%
false; repeats a known lie -> 44-73% false. But 72% of rows get no vote here: each held-out
row has ~1 unlabelled peer for its fact, while the real run has ~10x more unlabelled rows.
Splitting states by votes (E3) spreads training rows too thin and costs accuracy; weighing
the votes in a stacker (E3b) is on par with E1b with the best F1 on lies so far.
**This simulation understates E3**: expect more benefit at runtime. Re-measure once the
Reader parses real rows.

## E7 Verdict policy (2026-10-28)

`ConservativeWindows`: the evidence-policy rule on E1 timelines (an atom is proved inside a
core; false if any atom proved false, true if all proved true, otherwise unsure).

| model | acc | 95% CI | predicted true / false / unsure |
|---|---|---|---|
| windows (most likely label) | 0.889 | 0.872-0.907 | 1736 / 161 / 0 |
| conservative (unsure unless proved) | 0.259 | 0.226-0.292 | 383 / 70 / 1444 |

**Prediction held, decisively.** The conservative rule is never wrong when it commits, but
commits on 24% of rows; only 2% of rows are actually unsure. Proposal for the team: keep the
evidence policy's *evidence* rules, replace its *verdict* rule with the most likely label
plus calibrated confidence.

## Reader: sentence -> facts (2026-10-28)

`hooly/extract.py` (value grammar), `hooly/reader.py` (subjects, attributes, yes/no),
`hooly/pipeline.py` (Reader + truth model). `python3 -m evaluation.extract_recall`,
`python3 -m evaluation.reader_eval`, `python3 -m evaluation.run --mode parsed`.

**Value extraction**: 2,225 / 2,228 gold non-boolean values recovered (99.9%; the three
misses are annotation quirks). Tuned on the same sentences, so expect less on unseen rows.

**Fact assignment** (held-out pages, gold forms as reference; threshold 0.02):

| | precision | recall |
|---|---|---|
| all atoms | 0.904 | 0.736 |
| quantity | 0.955 | 0.818 |
| money | 1.000 | 0.784 |
| duration | 0.935 | 0.760 |
| percent | 0.938 | 0.756 |
| version | 0.945 | 0.812 |
| date | 0.904 | 0.630 |
| name | 0.686 | 0.742 |
| boolean | 0.422 | 0.155 |

Threshold sweep (atom F1): 0.0 -> 0.790 (P .79, R .79), 0.02 -> 0.811, 0.05 -> 0.790,
0.2 -> 0.760. Weak spots: yes/no attributes and name values.

**End to end** (`--mode parsed`: held-out rows lose labels *and* forms; the Reader parses them):

| model | gold forms | **parsed** | parsed macro F1 | parsed F1 false |
|---|---|---|---|---|
| always_true | 0.833 | 0.833 | 0.303 | 0.000 |
| nearest | 0.780 | 0.804 | 0.465 | 0.516 |
| windows (E1) | 0.889 | 0.869 | 0.471 | 0.488 |
| **windows_plus (E1b)** | 0.893 | **0.873** (0.857-0.890) | 0.497 | 0.563 |
| windows_votes (E3b) | 0.894 | 0.870 | 0.498 | 0.567 |

Parsing costs ~2 points. At runtime all practice rows are anchors (here 80%) and there are
~10x more unlabelled rows, which should help, but new subjects/attributes in the hidden data
will hurt; the sample rows (29 Oct) are the first check.

## Team benchmark arena-eval-v1 (2026-10-28)

`python3 -m evaluation.arena_v1`. The team's shared protocol (EVALUATION.md on the `bogdan`
branch): frozen splits `evaluation/splits-v1.json` (hashes verified), access profile
batch_unlabelled_v1 (labelled training rows + unlabelled test text, no test forms: our
`--mode parsed`), seed 42, policy fixed_050 (true/false only if its probability >= 0.50,
otherwise unsure). Bogdan's numbers are from BASELINE.md (text-only TF-IDF + LR).

| track | model | accuracy | macro F1 | F1 false | false P / R | decided coverage |
|---|---|---|---|---|---|---|
| stratified | always_true | 0.834 | 0.303 | 0.000 | - | 1.000 |
| stratified | tfidf_lr_balanced (Bogdan) | 0.650 | 0.405 | | - / 0.27 | |
| stratified | **windows_plus** | **0.866** | **0.496** | 0.565 | 0.70 / 0.47 | 0.992 |
| stratified | windows_votes | 0.853 | 0.494 | 0.566 | 0.64 / 0.51 | 0.984 |
| temporal | always_true | 0.818 | 0.300 | 0.000 | - | 1.000 |
| temporal | tfidf_lr_balanced (Bogdan) | 0.597 | 0.292 | | - / 0.06 | |
| temporal | **windows_plus** | **0.828** | **0.393** | 0.275 | 0.65 / 0.17 | 0.992 |
| checker | always_true | 0.853 | 0.307 | 0.000 | - | 1.000 |
| checker | tfidf_lr_balanced (Bogdan) | 0.711 | 0.384 | | - / 0.16 | |
| checker | **windows_plus** | **0.868** | **0.485** | 0.529 | 0.61 / 0.47 | 0.992 |

Ahead on every track on accuracy and macro F1 at once (the text-only model buys macro F1
by giving up accuracy). The temporal track is the weak one: test rows after 1 Sept only
have earlier anchors, so changes in Sept/Oct are invisible and lies there are hard to catch.
The same limitation applies to `fact` claims "as of 29 Oct" in generation.

## Reader fix: self-inclusion leak in parser training (2026-10-28)

Error analysis of parsed vs gold verdicts: parsing lost 67 rows and won 29. Cause found in
the parser's own training: its statistics (known values of a fact, cue words, attribute
frequencies, unit associations) included the training row itself, so "the value matches a
known value of this fact" was always on for correct pairs and got weight 6.9, dwarfing
everything else. On unseen rows it misled ("Hooly Docs 1.0" parsed as Hooly Gemini's version
because Gemini once had 1.0). Also the fallback "company" subject was Hooly Gemini (most
frequent) instead of Hooly (most attributes). Fix: every statistic for a training row leaves
its own page out, as at inference.

| | parser P / R / F1 | page-grouped acc | arena stratified | temporal | checker |
|---|---|---|---|---|---|
| before (threshold 0.02) | 0.904 / 0.736 / 0.811 | 0.873 | 0.866 | 0.828 | 0.868 |
| fixed, threshold 0.1 | 0.890 / 0.794 / 0.839 | 0.868 | 0.861 | 0.814 | 0.839 |
| **fixed, threshold 0.3** | higher P, lower R | **0.871** | **0.871** | **0.836** | 0.866 |

Lesson: a better parser by F1 did not mean better labels. More parsed facts at ~0.89
precision add wrong facts that turn true rows into false alarms; for labelling, parser
precision matters more than recall. Kept: the leak-free parser at threshold 0.3. Yes/no facts
stay weak (recall 0.18-0.32 depending on threshold). Accuracy tuning stops here; remaining
effort goes to claims, runtime, demo and self-evaluation.

## Round 2: parse-aware verdicts, lie shapes, HMM, calibration (2026-10-28)

Scoring is a black box comparing our labels with the organisers' truth, so the target is
plain agreement (accuracy); our own numbers feed the self-evaluation.

**#4 Parse-aware verdicts** (`hooly/pipeline.py`). Inside the training data, a second Reader
trained without a slice of pages parses that slice: an isotonic map turns a parse score into
"how often a parse like this is right". A parsed atom's evidence is pulled towards the prior
by that amount. Parsed, page-grouped: 0.873 vs 0.871 without; Brier 0.215 vs 0.218, ECE
0.035 vs 0.040. Within noise, safe, kept.

**#3 Lie shapes** (`TemporalWindowsShape`). For a value that disagrees with the timeline: does
it equal a value the fact held at another time; ratio to the reference value; is it rounder.
Gold: 0.900 vs 0.893, F1 false 0.695 vs 0.667. Parsed: **0.877**. Striking pattern on
single-fact held-out rows: *much bigger and rounder than the true value* -> 93% false (26/28),
e.g. "over 5,000 attendees" vs a true 2,600. Small or downward deviations are mostly real
changes (11-21% false).

**#1 Probabilistic timelines** (`hooly/hmm.py`, `TimelineHMM`). Per fact, a hidden Markov
model over which value is true at each date; labelled rows pin or exclude values, other rows
tell the truth with probability alpha; EM learns alpha and the time between changes (on the
full practice data: alpha 0.57 for the rows it is unsure about, ~230 days between changes).
No hand-made states. Gold: 0.885, F1 false 0.687, best ECE (0.023).

**Hybrid** (`HybridStack`): windows_shape + HMM weighed by a stacker; HMM scores for the
stacker's training rows are out-of-fold (their labels hidden from the HMM that scores them).

| model | gold acc | gold F1 false | parsed acc | parsed F1 false |
|---|---|---|---|---|
| windows_plus | 0.893 | 0.667 | 0.873 | 0.535 |
| windows_shape | 0.900 | 0.695 | 0.877 | 0.554 |
| hmm | 0.885 | 0.687 | - | - |
| **hybrid** | **0.905** | **0.727** | 0.876 | **0.572** |

Team benchmark (arena-eval-v1, parsed, fixed_050):

| track | windows_shape acc / macro F1 / F1 false | **hybrid** acc / macro F1 / F1 false |
|---|---|---|
| stratified | 0.868 / 0.487 / 0.535 | 0.868 / **0.502** / **0.581** |
| temporal | 0.838 / 0.399 / 0.286 | 0.838 / **0.409** / **0.318** |
| checker | 0.868 / 0.481 / 0.517 | 0.861 / 0.477 / 0.511 |

Same accuracy, better at catching lies on most tracks (including the hard temporal one): the
hybrid is the main model.

**#5 Calibration** (`evaluation/calibrate_report.py`, `hooly/calibrate.py`). Cross-fitted
isotonic maps per label ("when we said false with p, how often were we right?"); a false
call whose calibrated chance drops below 0.5 flips to true. Runtime map fitted on all folds:
`hooly/assets/calibration.json`.

| hybrid, parsed | acc (95% CI) | F1 false | Brier | ECE |
|---|---|---|---|---|
| raw | 0.876 (0.860-0.893) | 0.572 | 0.213 | 0.045 |
| **calibrated** | **0.878** (0.861-0.895) | 0.549 | 0.216 | **0.021** |

Calibrated reliability: said 0.78 -> right 0.83 (306 rows), 0.82 -> 0.82 (348), 0.92 -> 0.92
(789), 0.97 -> 0.95 (341).

## Self-evaluation: what we expect on the hidden data (first estimate, 2026-10-28)

*Updated after round 4: page-held-out CV 0.886 (0.869-0.903); expected on the hidden set
about 0.88, plausibly 0.85-0.90. The runtime also prints its own estimate (self-check).*

Final label system: Reader -> hybrid (timeline states + lie shapes + HMM) -> calibration.

| evidence | accuracy |
|---|---|
| our page-held-out CV (whole pages unseen, text only) | 0.878 (0.861-0.895) |
| team benchmark, stratified / checker | 0.868 / 0.861 |
| team benchmark, temporal (only older anchors) | 0.838 |
| always-true on the same data | 0.818-0.853 |

Expected on the hidden set: **about 0.87, plausibly 0.84-0.89** (on rows about subjects the
practice data never shows, closer to 0.82-0.84; see Robustness). Pushing up: at runtime all
practice rows are anchors and there are ~10x more unlabelled rows (the HMM uses them).
Pushing down: subjects, attributes and phrasings the practice data never shows (unknown
share; the 20 sample rows on 29 Oct are the first check, and we will revise this estimate
after seeing them). Confidence is calibrated on unseen pages (ECE ~0.02). Claims: ~100% of generated claims survive held-out evidence; expected precision on the
hidden truth ~0.95+, the main risk being sentences our Referee accepts but theirs reads differently.

## Claims: Writer + Referee (2026-10-28)

`hooly/writer.py`. Truth only from labelled-true rows; candidates in order of safety:
change (two labelled-true values in time order), compare between past dates, relation,
at_event (value whose certain core covers the event's final date), compare between recent
amounts with a >= 1.5x gap, fact (immutable past dates or values seen within 30 days).
Conservative templates in the value forms of value-forms.md; our Referee re-reads every
sentence (subject named, value read back exactly once, no competing value/date/time
word/negation, compare without any number or number word, one sentence, new, lexicon
subject and attribute). At most 6 claims per fact, relaxed to 10/20 only if short; a second
tier uses rows the system labelled true with >= 95% confidence (claim confidence x0.9).

Referee spot test on the contract's own examples: 17/18 as expected after adding number
words to the compare check (the 18th was a test mistake: that sentence is an input row).

**Do the claims survive evidence the generator did not see?** Claims generated with a fold's
labels hidden, checked against timelines from all practice rows (`python3 -m evaluation.claims_eval`):

| type | claims (5 folds) | confirmed | contradicted | unverifiable |
|---|---|---|---|---|
| at_event | 1475 | 1387 | 0 | 88 |
| change | 1536 | 1535 | 1 | 0 |
| compare, amounts | 847 | 847 | 0 | 0 |
| compare, dates | 429 | 424 | 0 | 5 |
| fact | 426 | 426 | 0 | 0 |
| relation | 67 | 59 | 0 | 8 |

The one contradiction: a rescheduled event date whose sightings came out of order. Harder
version (claims generated only from pages before 1 Sept, checked against everything): 622
claims, 0 contradicted; the generator declines "as of 29 Oct" amounts without recent
evidence (only 6 facts). The residual risk is wording, not truth: our Referee re-implements
the referee's reading, so templates are kept deliberately plain.

**Runtime** (`python -m hooly.run`, `Dockerfile`): practice universe in check.sh shape ->
380/380 labelled, 1,000 claims, organisers' format checker passes, 13 s. Simulated 19k-row
universe: 47 s, 0.7 MB output (limits: 30 min, 16 MiB).

## Robustness: unseen subjects and 10% labelled (2026-10-28)

`python3 -m evaluation.robustness --split subjects|sparse [--gold]`. *subjects*: subjects in 5
groups; every row about a held-out subject loses label and gold facts (other rows mentioning
them are hidden too, unscored), the Reader never sees their forms. *sparse*: only 10% of pages
keep labels (2 seeds), the rest is scored. `--gold`: hidden rows keep gold facts (parser out
of the picture).

| setting | facts | always_true | windows_shape | hmm | hybrid |
|---|---|---|---|---|---|
| unseen subjects | parsed | 0.833 | 0.824 (F1 false 0.06) | 0.807 (F1 false 0.23) | 0.823 |
| unseen subjects | gold | 0.833 | 0.834 | **0.853** (cal. 0.867, F1 false 0.54) | 0.834 |
| 10% labelled | parsed | 0.829 | 0.828 | - | 0.823 (F1 false 0.17) |
| 10% labelled | gold | 0.829 | 0.826 | **0.869** (F1 false 0.58) | 0.866 |

1. **Without anchors the timeline model has nothing to say** and falls back to the prior. The
   HMM, which uses unlabelled rows as evidence, still works: 0.869 with 10% labels (VERIDYCT
   reports 0.851 there), 0.853-0.867 on unseen subjects, with gold facts.
2. **The hybrid's stacker ignores the HMM when anchors are missing**: it learned its weights
   where anchors almost always exist. Needs regime-aware weighting (anchor coverage as input).
3. **The parser is the bottleneck under shift**: gold vs parsed is 0.853 vs 0.807 (unseen
   subjects) and 0.866 vs 0.823 (10% labels). New subjects and attributes are parsed poorly.
   Self-training the Reader on the unlabelled rows is warranted.
4. Calibration degrades under shift (ECE 0.04-0.06 instead of 0.02). Self-evaluation: on rows
   about subjects the practice data never shows, expect roughly always-true level (~0.82-0.84)
   until points 2-3 are fixed.

## Round 3: ensemble of views, regime gate, cue dropout (2026-10-28)

**Regime-aware ensemble** (`hooly/ensemble.py`): five views (timeline, HMM, cross-page votes,
identical sentence elsewhere, numeric outlier), regime inputs (anchor coverage, anchor counts,
parse confidence) with gated copies, *anchor dropout* twins so the meta-model sees the
no-anchor case, greedy forward selection on inner CV, isotonic on top (the 2026 ensembling
paper's advice: greedy selection, recalibrate after stacking).

| setting | hybrid | ensemble |
|---|---|---|
| normal, gold facts | 0.905 | 0.906 |
| unseen subjects, gold | 0.834 | 0.846 |
| 10% labelled, gold | 0.866 | 0.858 |
| **normal, parsed** | **0.876** | **0.844** |

**Negative end to end.** The meta-model learned from clean gold facts but sees parsed ones;
a misparsed fact looks exactly like "a fact with no anchors", so the gating sends it to the
HMM and votes, which call it false (258 false calls vs 181). Fixing that needs the meta-model
trained on parsed versions of the training rows; not pursued. Greedy selection kept only
timeline + HMM: votes, identical sentences and outliers added nothing on inner CV.

**Simple regime gate** (`GatedHybrid`): the hybrid, except when no labelled anchor backs any of
the row's facts, the parse is confident (>= 0.8), and the HMM says false with >= 0.8 backed by
observations on >= 4 dates of that fact; then the HMM decides.

**Cue dropout in the parser** (`Reader`, cue_dropout=0.35): on unseen subjects the parser's
recall was 0.22-0.28; 52% of their facts use attribute names never seen in training, and the
scorer leaned on cue words those attributes lack. During training each row now treats ~35% of
all attribute names (gold or not) as unseen (no cues, known values or prior), and for unseen
attributes the guessed value type is a soft signal instead of a filter.

| parser on unseen subjects | precision | recall | F1 |
|---|---|---|---|
| without cue dropout | 0.709 | 0.223 | 0.339 |
| **with cue dropout** | **0.780** | **0.420** | **0.546** |

Normal setting with cue dropout: parser P 0.920 / R 0.753 at threshold 0.3.

**End to end, final configuration (cue dropout + strict gate):**

| setting | hybrid (before) | **gated + cue dropout** |
|---|---|---|
| normal, parsed, calibrated | 0.878 | **0.876** (same; ECE 0.021) |
| unseen subjects, parsed | 0.823 | **0.825** (F1 false 0.13 vs 0.06) |
| unseen subjects, gold facts | 0.834 | **0.866** |
| runtime on check.sh-shaped input | 332/380 | **334/380** |

Neutral where we were already good, better on unseen subjects, with more headroom as
parsing improves (gold-facts ceiling +3.2). This is the runtime model; the calibration map in
`hooly/assets/calibration.json` was refitted for it.

## Round 4: where the errors come from, and parser fixes (2026-10-28)

**Error map** (`evaluation/error_map.py`): every held-out prediction (gated, parsed,
page-grouped CV) sorted by what the parser did with the row.

| parse outcome | rows | errors before | errors after | of which missed lies (after) | false alarms (after) |
|---|---|---|---|---|---|
| facts right | 1,313 -> 1,440 | 110 | 128 | 67 | 33 |
| a fact missing | 64 -> 54 | 15 | 9 | 9 | 0 |
| a wrong fact parsed | 139 -> 112 | 47 | 33 | 7 | 24 |
| nothing parsed | 381 -> 291 | 67 | 51 | 42 | 0 |
| **total** | 1,897 | **241** | **221** | | |

Before: 54% of the errors came from parsing. Why gold facts were missed
(`evaluation/parse_misses.py`): subject not among the candidates (30%), yes/no fact below
threshold (24%), pair score below threshold (33%), lost to another pair (10%), value not
extracted (3%).

**Fixes, all in the Reader:**

1. **Yes/no polarity bug.** The negation was read from the last punctuation mark onwards, so
   for any sentence ending in a full stop it was never seen: 66 of the 174 annotated yes/no
   facts got the wrong value (every "is not SOC 2 certified" became *yes*). Now: a negation
   anywhere in the clause holding the attribute's words, a bare answer after a colon
   ("Desktop mode: no", "AI editing: not live"), the whole sentence if no attribute word is
   found. Wrong polarity 66 -> 2 of 174 (`evaluation/bool_threshold.py`).
2. **Subjects.** "Hooly Android 16.0" names Hooly Android (version 16.0), not Hooly Android 16;
   shorter names inside a longer mention become candidates; the whole page is context, not
   only the 6 rows before (spec cards); later rows of the page are weaker candidates (a
   card's title can come last); distinctive name words ("Berlin" -> Hooly Berlin Office,
   "the Plus plan" -> Hooly Photos Plus). The scorer learns how far to trust each source.
3. **Interview speaker tags** ("Gavin: ...") were read as values ("Hooly's CEO is Gavin"):
   10 of 27 sampled name false positives. Tags are no longer values (3 of 163 tagged rows
   have the speaker as a fact value, and each also writes the full name later).
4. Crude stems in attribute-word matching (encrypts / encrypted, weighs / weight) and hyphens
   split ("trade-in").
5. Yes/no threshold 0.5 -> 0.4: with polarity fixed, precision 0.82 -> 0.84, recall 0.40 -> 0.46.

| parser, page-grouped CV | precision | recall | F1 | yes/no P / R | name P |
|---|---|---|---|---|---|
| round 3 | 0.920 | 0.753 | 0.829 | ~0.48 / 0.22 | 0.73 |
| **round 4** | **0.940** | **0.809** | **0.870** | **0.84 / 0.43** | **0.81** |

**End to end** (gated, parsed):

| test | round 3 | **round 4** |
|---|---|---|
| page-grouped CV, raw | 0.873 | **0.884** |
| page-grouped CV, calibrated | 0.876, F1 false 0.555 | **0.886** (0.869-0.903), F1 false **0.620**, ECE 0.025 |
| unseen subjects | 0.825, F1 false 0.13 | 0.818, F1 false 0.14 (0.824 before the speaker-tag fix) |
| team benchmark stratified / temporal / checker (raw) | 0.868 / 0.838 / 0.861 | 0.861 / 0.834 / **0.874** |
| runtime on check.sh-shaped input | 334/380 | 333/380 |

The benchmark tracks have 380 rows each (95% CI about +-3.5 points): net flat there, +1 point
on the larger CV. Calibration map refitted (`hooly/assets/calibration.json`).

**Negative results:**

- *Lower parse threshold (0.15)*, on the idea that shaky parses now count less anyway: same
  in distribution (0.877 vs 0.876 calibrated), worse on unseen subjects (0.818 vs 0.825). Kept 0.3.
- *#1 Self-training the Reader* (`Pipeline(self_train=0.9)`, `evaluation/self_train_cv.py`):
  unlabelled rows whose every parse has calibrated probability >= 0.9 become training
  sentences for a second Reader; rows of those pages are re-parsed with their own page's
  statistics left out. 0.880 vs 0.884 raw, F1 false 0.58 vs 0.62: it reinforces its own
  mistakes. The CV pool is small (380 rows) while the runtime has ~18k unlabelled rows, a
  setting we cannot test, so it stays off.

**#0 Runtime self-check** (`hooly/run.py`): the log reports lexicon facts with no practice
evidence, how many rows fall in each regime (nothing parsed / parsed without labelled anchor /
parsed and anchored), and the **expected accuracy**: the mean calibrated chance that each
label is right. On the check.sh-shaped input it predicted 0.881; observed 0.876.

Where the remaining 221 errors are: 128 on rows parsed right (mostly a value that disagrees
with the timeline before its first or after its last labelled anchor: a change or a lie?),
51 on rows with nothing parsed, 33 with a wrong fact parsed.

## Round 5: teammates' ideas and change rates per attribute (2026-10-28)

**Bogdan: 54 sklearn final classifiers** (his branch, `research/daria-sklearn-experiments.md`).
He kept this pipeline up to the stacker and replaced the 3-input logistic stacker with
three-label classifiers on richer inputs (timeline and HMM probabilities, states, lie
shapes, parse confidence, date). Tuned gradient boosting: macro F1 0.551 -> 0.650 on the
stratified track, accuracy 0.866 -> 0.871; temporal and checker flat. The macro-F1 gain is
3 of the track's 8 *unsure* rows; lie precision/recall barely move. Labels are scored by
agreement, so we keep our stacker: the decision layer is not the bottleneck.

**Sebi: delexicalisation** (his branch, `entity_linking/`): lexicon names and typed values
replaced by stable tokens (`<ENTITY:9> costs <MONEY:1>`), with an audited inspection report.
Our Reader already does the linking and value grammar; the transferable idea is to learn
cue words from the sentence with names blanked out (`Reader.delex`).

| Reader.delex | parser P / R (pages held out) | parser F1, unseen subjects | end to end, raw |
|---|---|---|---|
| off | 0.940 / 0.809 | 0.583 | 0.884, F1 false 0.623 |
| on | 0.941 / 0.808 | 0.583 | 0.886, F1 false 0.629 |

Neutral (2 rows of 1,897): leave-own-page-out statistics and cue dropout already keep the
scorer from leaning on product names. Off. His rule of not linking lower-case generic words
("4,900,000,000 photos") would hurt us: 18 of our 21 lower-case mentions are the row's subject.

**Change rates per attribute** (`TimelineHMM(attr_prior=m)`). The HMM used one change rate
for all facts, but 55 of 233 attributes never change in the labelled data while others
change 5-9 times a year (fact duration prediction, Zhang & Choi, EMNLP 2023; evolving truth,
Li et al., KDD 2015). Each attribute now can get its own rate under a Gamma prior worth m
pseudo-changes (learned taus: stable_version ~110 days, launched ~890 days).

| test (HMM alone unless noted) | one rate | per attribute (m=0.5) |
|---|---|---|
| pages held out, gold facts: HMM / gated | 0.885 / 0.905 | 0.889 / 0.904 |
| 10% labelled, gold facts: HMM | 0.869 | **0.879** |
| unseen subjects, gold facts: HMM / gated | 0.853 / 0.866 | **0.869** / 0.866 |
| **pages held out, real parsing: gated** | 0.884 | 0.884 |
| **unseen subjects, real parsing: gated** | 0.818 | 0.818 |
| ... and letting the HMM decide more often (gate at p >= 0.5, 2 dates): held out / unseen subjects | | 0.883 / 0.804 |

The HMM itself gets better where labels are scarce (+1.0 / +1.6 with perfect parsing), but
the shipped system does not: in the normal setting the stacker already has the timeline's
stable/volatile flag, and without anchors misparsed facts dominate. Loosening the gate to
use the better HMM more often hurts (more false alarms on misparsed facts). Off; parsing
remains the limit wherever labels are scarce.

**Observations, not used:** speaker tags are rare (9% of rows); "Gavin:" (the CEO) rows are
false 29% of the time vs a 15% base rate, but n=31. Lies cluster mildly by page: a row is
false 16.6% of the time when another row of its page is false, 11.1% otherwise.

## Round 6: fresh-wording test (2026-10-28)

**Test set** (development only, not in the repository): 63 practice rows from whole pages
were held out of everything and **rewritten in new words** (same facts, date and real label;
6 lies, 1 unsure), plus **86 new sentences** about facts the labelled data pins down, with a
date inside a certain core (70 true, 16 lies built the way the data's lies look: stale,
premature, inflated). Written with an LLM's help during development; the new sentences'
labels are constructed, so they test reading, not truth. The universe is the rest of
practice, labelled, as at runtime.

| held-out practice rows, real labels | accuracy | lies caught | false alarms | parser recall / precision |
|---|---|---|---|---|
| original wording | 0.921 (58/63) | 3/6 | 1 | 0.77 / 0.97 |
| **rewritten** | **0.905** (57/63) | 3/6 | 2 | 0.74 / 0.94 |
| new sentences (constructed labels, each names its subject) | 0.977 -> 0.988 with fix C | 15/16 | 1 -> 0 | 0.98 |

New wording costs one verdict in 63 and ~3 points of parser recall. Failures found: two
durations swapped between trial and refund in one sentence; "7 days free" linked to Hooly
AI Free through the lower-case word "free"; "should be generally available" read as *yes*;
"lets you edit offline" missed (5-letter stem "editi"); "cannot speak its answers yet" missed.

| fix | held out, calibrated | unseen subjects | new wording | kept |
|---|---|---|---|---|
| none | 0.886 | 0.818 | 0.905 / 0.977 | |
| A: distinctive name words only when capitalised | 0.881 | | | no |
| B: suffix stemmer + 5-letter prefix | 0.883 | | | no |
| **C: no yes/no fact from a plan or hedge** ("should", "will", "plans to") | 0.886 (identical) | 0.818 (identical) | 0.905 / **0.988** | **yes** |
| A+B+C | 0.884 | 0.821 | 0.905 / 0.988 | |

C changes nothing on practice and removes a false alarm type the practice data does not
show; A and B cost 1-3 rows on held-out pages. Runs are deterministic (identical across
hash seeds). Self-evaluation: on wordings the practice data never shows, expect about
1-2 points lower accuracy than on held-out practice pages.

## Planned experiments and predictions

| id | idea | source | prediction |
|---|---|---|---|
| E1 | Four-date windows | TAC KBP temporal slot filling | first variant to beat always_true. **Held: 0.889** |
| E2 | Gradual-change smoothing | DynaTD (KDD 2015) | worse than E1: Hooly changes in steps. *Not run: superseded by the HMM, which learns the change rate* |
| E3 | Unlabelled rows as votes | TruthFinder-style | small gain at best. **Held (in simulation): on par, best F1 false** |
| E4 | Page exactness + freshness | Dong et al. (VLDB 2009) | freshness catches stale lies exactness misses. *Deprioritised: lies cluster by page only weakly* |
| E5 | Credibility from text features, trained on labels | SLiMFast (2017) | best trust signal. **Failed: style AUC ~0.55** |
| E6 | Rule outputs combined by a label model vs supervised logistic regression | Snorkel / FlyingSquid | supervised LR >= label model |
| E7 | Verdict policy: unsure-unless-proved vs most-likely class | evidence policy vs proposal | most-likely class wins. **Held: 0.889 vs 0.259** |
