"""The runtime: what the organisers run inside the container.

    python -m hooly.run [--universe /universe] [--output /output] [--practice data/practice.csv]

1. Read universe.csv and the runtime lexicon.
2. Re-attach packaged practice forms to rows that match by id + text + date (an id alone is
   not trusted). Every other row is parsed from its text.
3. Write a safe fallback table at once (every unlabelled row `true` at the base rate), so a
   crash or a timeout still hands in a valid table.
4. Reader -> hybrid truth model -> calibration -> labels; Writer -> 1,000 claims.
5. Write the table under a temporary name and rename it into place. On SIGTERM (deadline),
   the last complete table stays in place.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import sys
import time

from .calibrate import Isotonic
from .data import Atom, build_names, load_lexicon, load_rows
from .hmm import GatedHybrid
from .hype import is_hyped
from .pipeline import Pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
COLUMNS = ["id", "type", "text", "form", "label", "confidence"]


def log(msg):
    print(f"[hooly {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def write_table(out_dir, label_rows, claims):
    """Atomic write: a reader of /output/output.csv only ever sees a complete table."""
    tmp = os.path.join(out_dir, ".output.csv.tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for rid, label, conf in label_rows:
            w.writerow([rid, "", "", "", label, "" if conf is None else f"{conf:.4f}"])
        for c in claims:
            w.writerow([c["id"], c["type"], c["text"], json.dumps(c["form"], ensure_ascii=False), "true",
                        f"{c['confidence']:.4f}"])
    os.replace(tmp, os.path.join(out_dir, "output.csv"))


def attach_practice_forms(rows, practice_path):
    """Gold forms for rows that are practice rows (same id, text and date)."""
    if not os.path.exists(practice_path):
        return 0
    known = {(r.id, r.text, r.date): r.atoms for r in load_rows(practice_path)}
    n = 0
    for r in rows:
        atoms = known.get((r.id, r.text, r.date))
        if atoms is not None and r.label:
            r.atoms = [Atom(a.subject, a.attribute, a.value) for a in atoms]
            n += 1
    return n


def load_calibration(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        tables = json.load(f)
    maps = {}
    for lab, pts in tables.items():
        if not isinstance(pts, list):  # scalar settings, e.g. hype_shrink
            maps[lab] = float(pts)
            continue
        m = Isotonic()
        m.xs, m.ys = [p[0] for p in pts], [p[1] for p in pts]
        maps[lab] = m
    return maps


def calibrated(pred, maps, text=""):
    lab = pred.label
    if not maps or lab not in maps:
        return lab, pred.confidence
    c = maps[lab](pred.probs[lab])
    if lab == "false" and c < 0.5:
        lab, c = "true", 1 - c
    c = min(1.0, max(0.5, c))
    s = maps.get("hype_shrink")
    if s and is_hyped(text):  # boastful wording: we are less often right, so say so
        c = 0.5 + s * (c - 0.5)
    return lab, c


def regime(row, pred) -> str:
    """Which evidence regime a prediction came from (for the self-check in the log)."""
    atoms = [a for a in pred.trace.get("atoms", []) if "subject" in a]
    if not atoms:
        return "nothing parsed"
    if all(a.get("state", "").startswith("no_anchor") for a in atoms):
        return "parsed, no labelled anchor"
    return "parsed, anchored"


def diagnostics(rows, todo, preds, lexicon) -> dict:
    """What this universe looks like next to the practice data, and how sure we are.

    expected accuracy = mean calibrated chance that each label is right (the confidence
    column; for unsure rows the model's unsure probability). Our cross-validation showed the
    calibrated confidences match observed accuracy within ~2 points (ECE ~0.02)."""
    seen = {a.key for r in rows if r.label and r.atoms for a in r.atoms}
    lex = {(s["name"], a) for s in lexicon.get("subjects", []) for a in s.get("attributes", [])}
    lex_subj = {s["name"] for s in lexicon.get("subjects", [])}
    seen_subj = {k[0] for k in seen}
    out = {"rows_to_label": len(todo),
           "lexicon_facts": len(lex), "lexicon_facts_without_practice_evidence": len(lex - seen),
           "lexicon_subjects_without_practice_evidence": sorted(lex_subj - seen_subj)[:20]}
    by = {}
    for r, (lab, conf, pred) in zip(todo, preds):
        reg = regime(r, pred)
        p_right = conf if conf is not None else pred.probs.get("unsure", 0.0)
        b = by.setdefault(reg, [0, 0.0])
        b[0] += 1
        b[1] += p_right
    n = max(1, len(todo))
    out["regimes"] = {k: {"rows": v[0], "share": round(v[0] / n, 3), "expected_accuracy": round(v[1] / max(1, v[0]), 3)}
                      for k, v in sorted(by.items())}
    out["expected_accuracy"] = round(sum(v[1] for v in by.values()) / n, 3)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="/universe")
    ap.add_argument("--output", default="/output")
    ap.add_argument("--practice", default=os.path.join(HERE, "..", "data", "practice.csv"))
    ap.add_argument("--calibration", default=os.path.join(HERE, "assets", "calibration.json"))
    args = ap.parse_args(argv)
    t0 = time.time()
    deadline = float(os.environ.get("ARENA_DEADLINE", t0 + 30 * 60))
    log(f"start; {int(deadline - t0)} s until the deadline")

    rows = load_rows(os.path.join(args.universe, "universe.csv"))
    lexicon = load_lexicon(os.path.join(args.universe, "lexicon.json"))
    todo = [r for r in rows if not r.label]
    labelled = [r for r in rows if r.label]
    base_true = sum(r.label == "true" for r in labelled) / max(1, len(labelled)) if labelled else 0.83
    log(f"{len(rows)} rows ({len(todo)} to label), {len(lexicon.get('subjects', []))} lexicon subjects")

    state = {"labels": [(r.id, "true", max(0.5, base_true)) for r in todo], "claims": []}
    write_table(args.output, state["labels"], state["claims"])
    log("fallback table written")

    def on_term(signum, frame):  # the deadline: keep the last complete table, exit cleanly
        log("SIGTERM: keeping the last complete table")
        sys.exit(0)
    signal.signal(signal.SIGTERM, on_term)

    n_forms = attach_practice_forms(rows, args.practice)
    log(f"practice forms re-attached to {n_forms} labelled rows")
    names = build_names(lexicon, [r for r in rows if r.atoms])
    model = Pipeline(GatedHybrid).fit(rows, lexicon, names)
    log(f"pipeline fitted ({time.time() - t0:.0f} s)")
    maps = load_calibration(args.calibration)
    labels, preds = [], []
    for r in todo:
        pred = model.predict(r)
        lab, conf = calibrated(pred, maps, r.text)
        labels.append((r.id, lab, None if lab == "unsure" else conf))
        preds.append((lab, None if lab == "unsure" else conf, pred))
    state["labels"] = labels
    write_table(args.output, state["labels"], state["claims"])
    log(f"labels written ({time.time() - t0:.0f} s): " +
        ", ".join(f"{l} {sum(1 for _, x, _ in labels if x == l)}" for l in ("true", "false", "unsure")))
    try:
        log("self-check: " + json.dumps(diagnostics(rows, todo, preds, lexicon)))
    except Exception as e:  # diagnostics must never cost us the table
        log(f"self-check failed: {e!r}")

    try:
        from .writer import generate_claims
        trusted = {rid for rid, lab, conf in labels if lab == "true" and conf is not None and conf >= 0.95}
        rep = {}
        state["claims"] = generate_claims(rows, lexicon, names, model, n=1000, deadline=deadline,
                                          extra_true=trusted, report=rep)
        log(f"claims: {rep}")
        write_table(args.output, state["labels"], state["claims"])
        log(f"{len(state['claims'])} claims written ({time.time() - t0:.0f} s)")
    except ImportError:
        log("no writer yet: table holds labels only")
    log(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
