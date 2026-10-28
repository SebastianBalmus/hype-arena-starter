"""Batch entry point: safe table first, then labels, then claims."""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time
import traceback

from arena.data import read_universe, write_output
from arena.lexicon import Lexicon
from arena.pipeline import SAFE_LABEL, load_artifacts, run

MARGIN = 120


_state = {"safe": False, "term": False}


def _on_term(signum, frame):
    # output.csv is only ever replaced atomically, so exiting once a table exists leaves a complete one.
    # Before the first safe table is written, only note the signal; main() exits right after writing it.
    _state["term"] = True
    if _state["safe"]:
        sys.exit(0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="/universe")
    ap.add_argument("--output", default="/output")
    ap.add_argument("--artifacts", default="/app/artifacts")
    ap.add_argument("--practice", default="/app/data/practice.csv")
    a = ap.parse_args(argv)
    _state.update(safe=False, term=False)
    try:
        signal.signal(signal.SIGTERM, _on_term)
    except ValueError:  # not the main thread
        pass
    try:
        deadline = float(os.environ["ARENA_DEADLINE"])
    except (KeyError, ValueError):
        deadline = time.time() + 30 * 60

    rows = read_universe(os.path.join(a.universe, "universe.csv"))
    write_output(a.output, [(r.id, *SAFE_LABEL) for r in rows if r.label == ""], [])
    _state["safe"] = True
    if _state["term"]:
        return 0
    if deadline - time.time() < MARGIN:
        return 0
    try:
        lexicon = Lexicon.load(os.path.join(a.universe, "lexicon.json"))
        art = load_artifacts(a.artifacts, a.practice)

        def save_labels(res):
            write_output(a.output, res.labels, [])

        res = run(rows, lexicon, art, generate=True, stop_at=deadline - 30, on_labels=save_labels,
                  claims_after=deadline - MARGIN)
        if res.claims:
            write_output(a.output, res.labels, res.claims)
    except Exception:  # the last complete table stays in place
        traceback.print_exc()
    return 0


if __name__ == "__main__":
    sys.exit(main())
