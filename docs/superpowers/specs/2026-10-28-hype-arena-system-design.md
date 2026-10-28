# Hype Arena system — design

Source: the architecture proposal doc (claude.ai, "Hype Arena: Architecture Proposal and Build Plan", 28 Oct 2026),
condensed here so builders can work from the repo alone. Contract files in `contracts/` are authoritative.

## Intent

Build the system the challenge asks for (`README.md`, `contracts/README.md`): read `/universe/universe.csv` and
`/universe/lexicon.json`, label every unlabelled row `true`/`false`/`unsure` with a confidence, and write up to
1,000 new claims that are all true, into `/output/output.csv`. Plus a frontend page that shows the architecture
and runs the same pipeline live on pasted rows.

## Goals

- Labels beat the always-true floor (83% on practice) on page-grouped holdouts, with honest confidences.
- Every generated claim passes a local copy of the referee rules (`contracts/claim-forms.md`,
  `contracts/value-forms.md`) and `contracts/entry.schema.json`, names only lexicon subjects/attributes,
  and is backed by the reconstructed fact histories.
- A full run on 20,000 rows finishes well inside 30 minutes on 2 CPUs / 6 GiB.

## Non-goals

- No LLM, no pretrained language model, no network at run time.
- No pushing, no PR, no clone: all work stays in this local repo (branch `hype-arena-system`).
- No padding: if fewer than 1,000 claims are valid, output fewer (a missing slot scores nothing; an invalid one
  also scores nothing, but a false one may cost more).

## Anti-goals (failure even if it "works")

- An image that only scores well because it contains `data/practice.csv` labels (leakage in evaluation).
- Claims written in free style that the deterministic referee rejects.
- A run that leaves no `/output/output.csv` at the deadline.

## Constraints (verbatim from the contract)

- `docker build --platform linux/amd64 .`, image ≤ 3 GiB unpacked, Dockerfile ends with `USER 10001`.
- Run: offline, 2 CPUs, 6 GiB, ≤ 256 processes/threads, read-only root; `/tmp` writable ≤ 128 MB; `/output` ≤ 256 MB;
  no home directory; 30 minutes; `ARENA_DEADLINE` holds the deadline in Unix seconds; SIGTERM then kill after 10 s.
- Output columns `id,type,text,form,label,confidence`; label rows have empty `type,text,form`; `unsure` has empty
  confidence; confidences in [0.5, 1]; claim text ≤ 300 chars; claim id `^[A-Za-z0-9_.-]{1,64}$`; output ≤ 16 MiB;
  write to a temp name in `/output` and rename.

## Data facts the design relies on (measured on practice)

- 1,897 rows; labels 1,581 true / 277 false / 39 unsure; 2,402 facts; 430 rows hold ≥ 2 facts.
- Truth is per row date. Values change over time (AI Pro price €49 → €59 on 2026-04-01; CTO Richard Hendrix →
  Bertram Gilfoil on 2026-07-01). Statements made too early are false (Gemini "12 languages" on 2025-12-10).
- Some pages lie: 11 date-blocks are entirely false. Rows are grouped by page as runs of consecutive rows with the
  same date (291 blocks) but the file is not sorted by date.
- 35% of rows omit a gold subject (pronouns, "the platform") → page context needed.
- Every attribute has exactly one value kind across practice; 51 attributes are boolean.
- 24 of 35 distinct name values are not lexicon subjects (Gavin Bellson, Nimbus Cloud, cities); short forms
  Gavin / Nimbus / Raviqa occur.
- No key ever has two different true values on one date (attributes are functional).
- Unsure mostly = the date lies outside the span covered by trusted evidence for that fact (35 of 44 facts).
- Label-free consensus that answers unsure unless proved scores 12–53% overall: abstaining is costly.

## Decided design (the how)

Pipeline, one Python package `arena/` used by both batch CLI and live server:

1. **Extract** typed facts per row: value parser implementing value-forms.md; lexicon mention matching (longest
   first, whole words, any case); page-context subject resolution; attribute ranker (TF-IDF + logistic
   regression trained on practice forms, restricted to the subject's lexicon attributes of a compatible kind,
   name-overlap fallback for unseen attributes); boolean facts from attribute words + negation polarity.
   Practice rows found in the input (same id, text and date) reuse their gold forms.
2. **Evidence store**: per (subject, attribute) observations; weighted by page reliability (TruthFinder-style
   iteration) and labels (true labels = trusted anchors; false single-fact labels = refutations); segmented into
   value intervals with lone dissenting values dropped as suspected lies; time queries `value_at`,
   `intervals`, `current_value`.
3. **Decide labels**: per-fact states proved_true / proved_false / unknown; row verdict = any false → false, all
   true → true, else unknown; a gradient-boosting classifier over verdict + evidence features picks the output
   label; isotonic calibration gives the confidence. `unsure` is a model output, expected to be rare.
4. **Generate claims**: enumerate change / compare / relation / at_event / fact candidates from trusted
   intervals, render with fixed templates, keep only those that pass the local referee + schema + novelty,
   rank by a proof-strength score, cap per subject, take up to 1,000.
5. **Outputs**: batch writes a safe table within 60 s (all unlabelled rows `true` at 0.6), then the full result,
   always via temp file + `os.replace`; live server returns JSON with traces.

## Calls made by the builder (Claude), not by the user

- Python 3.12; numpy, scikit-learn, jsonschema, fastapi, uvicorn; scikit-learn `HistGradientBoostingClassifier`
  instead of LightGBM.
- Models are trained during `docker build` (`python -m arena.train`) so pickles match the image's library versions;
  the image contains `data/practice.csv` and `data/lexicon.json`.
- Frontend: static HTML/CSS/JS in `web/`, served by FastAPI; no build step.
- Fewer than 1,000 valid claims → output fewer rather than pad.
- Local git: branch `hype-arena-system`, local commits only.

## Left to the builder

Exact feature sets, templates per attribute family, thresholds, and visual design of the page, within the
constraints above.
