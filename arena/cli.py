"""Batch entry point: safe table first, then labels, then claims."""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time

from arena.data import read_universe, write_output
from arena.lexicon import Lexicon
from arena.pipeline import SAFE_LABEL, load_artifacts, run

MARGIN = 120


def _on_term(signum, frame):
    # output.csv is only ever replaced atomically, so exiting here leaves a complete table.
    sys.exit(0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="/universe")
    ap.add_argument("--output", default="/output")
    ap.add_argument("--artifacts", default="/app/artifacts")
    ap.add_argument("--practice", default="/app/data/practice.csv")
    a = ap.parse_args(argv)
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
    if deadline - time.time() < MARGIN:
        return 0
    try:
        lexicon = Lexicon.load(os.path.join(a.universe, "lexicon.json"))
        art = load_artifacts(a.artifacts, a.practice)

        def save_labels(res):
            write_output(a.output, res.labels, [])

        res = run(rows, lexicon, art, generate=True, stop_at=deadline - MARGIN, on_labels=save_labels)
        if res.claims:
            write_output(a.output, res.labels, res.claims)
    except Exception as e:  # the last complete table stays in place
        print(f"pipeline failed, keeping last table: {e!r}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
