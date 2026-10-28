"""Demo API for the frontend: the container's code path behind HTTP (not part of the image).

    python3 -m demo.server [--port 8000] [--sample sample/sample.csv] [--snapshot frontend/src/lib/snapshot.ts]

At startup the Reader is trained once on the labelled practice rows (it only ever learns
from labelled rows, so it never needs retraining). Each run then does exactly what
`hooly.run` does in the container: the uploaded rows join the practice universe as
unlabelled rows, are parsed, the truth model is fitted on the whole universe, labels are
calibrated, the 1,000 claims are written, and `output.csv` is written in the contract format.

Sample: by default 20 practice rows (whole pages) are held out of everything, so the demo
can show how many verdicts agree with their real labels. `--sample` uses a CSV instead
(id,text,date[,label]); a filled label there is only shown for comparison, never used.

Endpoints: POST /api/run {rows: [{id, text, date}]}, GET /api/output.csv, GET /api/claims,
GET /api/metrics, GET /api/sample, GET /api/health.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import re
import tempfile
import threading
import time
from dataclasses import replace
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from hooly.data import Row, assign_pages, build_names, load_lexicon, load_rows
from hooly.hmm import GatedHybrid
from hooly.pipeline import Pipeline
from hooly.reader import attr_tokens, stem
from hooly.run import calibrated, diagnostics, load_calibration, write_table
from hooly.values import satisfies
from hooly.writer import generate_claims, render

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CLAIM_TYPES = {"fact": "FACT", "compare": "COMPARE", "change": "CHANGE", "at_event": "AT EVENT", "relation": "RELATION"}


def _exact(v):
    return {k: x for k, x in v.items() if k != "op"}


def value_text(v) -> str:
    if v.get("kind") == "boolean":
        return "yes" if v.get("value") else "no"
    t = render(_exact(v)) if v.get("kind") else None
    if t is None:
        t = str(v.get("value"))
        if v.get("unit"):
            t += " " + str(v["unit"])
    op = {"gt": "more than ", "ge": "at least ", "lt": "less than ", "le": "at most "}.get(v.get("op", ""), "")
    return op + t


def _d(s: str) -> date:
    return date.fromisoformat(s[:10])


class Demo:
    def __init__(self, practice, lexicon_path, calibration, sample=None, n_holdout=20, seed=11):
        t0 = time.time()
        self.lock = threading.Lock()
        self.lexicon = load_lexicon(lexicon_path)
        practice_rows = load_rows(practice)
        self.sample_gold = {}
        if sample:
            with open(sample, encoding="utf-8") as f:
                rs = list(csv.DictReader(f))
            self.sample = [{"id": r["id"], "text": r["text"], "date": r["date"]} for r in rs]
            self.sample_gold = {r["id"]: r["label"] for r in rs if r.get("label")}
            held = set()
        else:
            held = self._holdout(practice_rows, n_holdout, seed)
            self.sample = [{"id": r.id, "text": r.text, "date": r.date} for r in practice_rows if r.id in held]
            self.sample_gold = {r.id: r.label for r in practice_rows if r.id in held}
        self.train = [r for r in practice_rows if r.id not in held and r.label]
        self.names = build_names(self.lexicon, self.train)
        self.pipeline = Pipeline(GatedHybrid)
        self.pipeline.parse_cal = self.pipeline._parse_calibration(self.train, self.lexicon, self.names)
        self.pipeline.reader = self.pipeline._reader().fit(self.train, self.lexicon, self.names)
        self.maps = load_calibration(calibration)
        self.out_dir = tempfile.mkdtemp(prefix="hooly-demo-")
        with open(os.path.join(HERE, "metrics.json"), encoding="utf-8") as f:
            self.metrics = json.load(f)
        self.last = None
        print(f"[demo] reader trained on {len(self.train)} labelled rows in {time.time() - t0:.0f} s; "
              f"sample: {len(self.sample)} rows", flush=True)

    @staticmethod
    def _holdout(rows, n, seed):
        """Whole pages of practice rows, with a few lies among them, kept out of everything."""
        pages = {}
        for r in rows:
            pages.setdefault(r.page, []).append(r)
        for s in range(seed, seed + 200):
            order = [p for p in pages.values() if 2 <= len(p) <= 8]
            random.Random(s).shuffle(order)
            pick = []
            for p in order:
                if len(pick) + len(p) <= n:
                    pick += p
                if len(pick) >= n - 1:
                    break
            if sum(r.label == "false" for r in pick) >= 3:
                return {r.id for r in pick}
        return {r.id for r in rows[:n]}

    # ------------------------------------------------------------------ one run
    def run(self, inputs):
        t0 = time.time()
        with self.lock:
            return self._run(inputs, t0)

    def _run(self, inputs, t0):
        ids = {x["id"] for x in inputs}
        gold = dict(self.sample_gold)
        universe = []
        for r in self.train:
            if r.id in ids:  # an uploaded practice row: its label is hidden for this run
                gold.setdefault(r.id, r.label)
                continue
            universe.append(replace(r, meta={}))
        base = max((r.index for r in universe), default=0) + 1
        new = [Row(x["id"], x["text"], x.get("date") or "2026-10-29", None, None, index=base + i)
               for i, x in enumerate(inputs)]
        assign_pages(new)
        top = max((r.page for r in universe), default=0) + 1
        for r in new:
            r.page += top
        universe += new
        p = self.pipeline
        p.reader.parse_all(universe)
        for r in new:
            if r.meta.get("parsed"):
                r.meta["atom_q"] = {k: p.parse_cal(s) for k, s in r.meta["parse_scores"].items()}
        p.truth = GatedHybrid().fit(universe, self.lexicon, self.names)
        self._uni = {r.id: r for r in universe}
        labels, preds, out_rows = [], [], []
        for r in new:
            pred = p.predict(r)
            lab, conf = calibrated(pred, self.maps)
            conf = None if lab == "unsure" else conf
            labels.append((r.id, lab, conf))
            preds.append((lab, conf, pred))
            out_rows.append(self._result(r, lab, conf, pred, gold.get(r.id)))
        trusted = {rid for rid, lab, conf in labels if lab == "true" and conf is not None and conf >= 0.95}
        rep = {}
        claims = generate_claims(universe, self.lexicon, self.names, p, n=1000, extra_true=trusted, report=rep)
        write_table(self.out_dir, labels, claims)
        diag = diagnostics(universe, new, preds, self.lexicon)
        known = [r for r in out_rows if r.get("gold")]
        agree = sum(r["gold"] == r["label"] for r in known)
        meta = {"runtime_ms": int((time.time() - t0) * 1000), "model": "Reader -> gated hybrid -> calibration",
                "csv_url": "/api/output.csv", "rows": len(new), "claims": len(claims),
                "expected_accuracy": diag["expected_accuracy"], "regimes": diag["regimes"],
                "gold_rows": len(known), "gold_agree": agree}
        self.last = {"universe": universe, "claims": claims, "labels": dict((rid, lab) for rid, lab, _ in labels)}
        return {"rows": out_rows, "meta": meta}

    # ------------------------------------------------------------------ explanations
    def _timeline(self, key):
        tl = self.pipeline.truth.shape._timeline(key, None)
        return tl

    def _span(self, text, atom):
        v = atom["value"]
        if v["kind"] == "boolean":
            low = text.lower()
            best = None
            for t in attr_tokens(atom["attribute"]):
                m = re.search(r"\b" + re.escape(stem(t)) + r"[a-z]*", low)
                if m and len(t) > 2:
                    best = m if best is None or m.start() < best.start() else best
            return [best.start(), best.end()] if best else [0, 0]
        for f in self.pipeline.reader._find(text):
            if f.kind == v["kind"] and satisfies(_exact(v), _exact(f.value), self.names):
                return [f.start, f.end]
        return [0, 0]

    def _fact(self, row, atom, by_id):
        key = (atom["subject"], atom["attribute"])
        claimed = atom["value"]
        tl = self._timeline(key)
        runs = [] if tl.multi else tl.runs
        d = _d(row.date)
        values = [{"value_text": value_text(r.values[0]), "from": r.start.isoformat(), "to": r.end.isoformat()} for r in runs]
        sup, con = [], []
        for r in runs:
            v = r.verdict(claimed, self.names)
            dist = min(abs((r.start - d).days), abs((r.end - d).days)) if not (r.start <= d <= r.end) else 0
            for rid in r.rows:
                x = by_id.get(rid)
                if x is None:
                    continue
                e = (dist, {"id": x.id, "date": x.date, "text": x.text})
                (sup if v else con if v is False else []).append(e)
        sup = [e for _, e in sorted(sup, key=lambda t: t[0])[:3]]
        con = [e for _, e in sorted(con, key=lambda t: t[0])[:3]]
        q = (row.meta.get("atom_q") or {}).get(key, atom.get("parse_confidence", 1.0))
        return {"subject": key[0], "attribute": key[1], "value_text": value_text(claimed),
                "span": self._span(row.text, atom), "parse_confidence": round(float(q), 3),
                "state": atom.get("state", "no_anchor").split("|")[0], "p_true": atom.get("p_true", 0.5),
                "timeline": {"row_date": row.date, "values": values[-6:]},
                "evidence": {"supporting": sup, "contradicting": con}}, runs

    def _lie_type(self, atom, runs, d):
        parts = atom.get("state", "").split("|")
        shape = atom.get("shape") or (parts[3] if len(parts) > 3 else "")
        if shape == "seen":
            for r in runs:
                if r.verdict(atom["value"], self.names):
                    return "STALE" if r.end < d else "PREMATURE"
            return "STALE"
        if shape.startswith("up"):
            return "INFLATED"
        if shape.startswith("down"):
            return "DEFLATED"
        if shape.startswith("near"):
            return "NEAR MISS"
        return "CONTRADICTED"

    def _result(self, row, lab, conf, pred, gold):
        tr = pred.trace
        by_id = self._uni
        atoms = [a for a in tr.get("atoms", []) if "subject" in a]
        facts, worst, worst_runs = [], None, []
        for a in atoms:
            f, runs = self._fact(row, a, by_id)
            facts.append(f)
            if worst is None or a.get("p_true", 1) < worst.get("p_true", 1):
                worst, worst_runs = a, runs
        d = _d(row.date)
        lie = self._lie_type(worst, worst_runs, d) if lab == "false" and worst else None
        pf_w = tr.get("p_false_windows", pred.probs.get("false", 0))
        pf_h = tr.get("p_false_hmm", pred.probs.get("false", 0))
        pick = (lambda pf: pf) if lab == "false" else (lambda pf: 1 - pf)
        return {"id": row.id, "text": row.text, "date": row.date, "label": lab,
                "confidence": None if conf is None else round(conf, 4), "lie_type": lie,
                "explanation": self._explain(row, lab, worst, worst_runs, tr, lie),
                "p_timeline": round(pick(pf_w), 3), "p_hmm": round(pick(pf_h), 3),
                "facts": facts, **({"gold": gold} if gold else {})}

    def _explain(self, row, lab, atom, runs, tr, lie):
        if atom is None:
            return ("No fact recognised in this sentence, so it gets the base rate: most sentences about "
                    "Hooly are true.")
        what = f"{atom['subject']} · {atom['attribute'].replace('_', ' ')}"
        said = value_text(atom["value"])
        d = _d(row.date)
        if tr.get("gate"):
            return (f"Says {what} = {said}. No labelled sentence covers this fact, but the sentences about it "
                    f"on other dates disagree, so the probabilistic model calls it false.")
        cover = next((r for r in runs if r.start <= d <= r.end), None)
        prev = max((r for r in runs if r.end < d), key=lambda r: r.end, default=None)
        nxt = min((r for r in runs if r.start > d), key=lambda r: r.start, default=None)
        ref = cover or prev or nxt
        if ref is None:
            return (f"Says {what} = {said}. No labelled sentence covers this fact yet, so the verdict leans on "
                    f"how often sentences like this are true.")
        rv = value_text(ref.values[0])
        span = f"{rv} from {ref.start.isoformat()} to {ref.end.isoformat()} ({len(ref.rows)} labelled rows)"
        where = "on that date" if ref is cover else "before that date" if ref is prev else "after that date"
        if ref.verdict(atom["value"], self.names):
            s = f"Says {what} = {said}. The record agrees {where}: {span}."
            if ref is not cover:
                s += " The value could have changed since, which lowers the confidence."
            return s
        s = f"Says {what} = {said}; the record {where} says {span}."
        if lie == "STALE":
            s += " That was the value earlier, not any more."
        elif lie == "PREMATURE":
            s += " That value only became true later."
        elif lie == "INFLATED":
            s += " The claim is bigger than the record."
        elif lie == "DEFLATED":
            s += " The claim is smaller than the record."
        elif lab == "true":
            s += " The value may simply have changed in between, so it is not called a lie."
        return s

    # ------------------------------------------------------------------ claims
    def claims(self, limit=1000):
        if not self.last:
            return []
        uni = {r.id: r for r in self.last["universe"]}
        trusted = {rid for rid, lab in self.last["labels"].items() if lab == "true"}
        by_key = {}
        for r in uni.values():
            if r.atoms and (r.label == "true" or r.id in trusted):
                for a in r.atoms:
                    by_key.setdefault(a.key, []).append((r, a.value))
        out = []
        for c in self.last["claims"][:limit]:
            out.append({"id": c["id"], "type": CLAIM_TYPES.get(c["type"], c["type"].upper()), "text": c["text"],
                        "confidence": c["confidence"], "form": c["form"], "proof": self._proof(c["form"], by_key)})
        return out

    def _proof(self, form, by_key):
        pairs = []
        if "left" in form:
            pairs += [((form["left"]["subject"], form["left"]["attribute"]), []),
                      ((form["right"]["subject"], form["right"]["attribute"]), [])]
        elif "link" in form:
            pairs += [((form["subject"], form["link"]), [{"kind": "name", "value": form["target"]}]),
                      ((form["target"], form["attribute"]), [form["value"]])]
        else:
            pairs.append(((form["subject"], form["attribute"]), [form[k] for k in ("value", "from", "to") if k in form]))
        proof, seen = [], set()
        for key, vals in pairs:
            cands = by_key.get(key, [])
            hits = [r for r, v in cands if not vals or any(satisfies(_exact(w), _exact(v), self.names) for w in vals)]
            for r in sorted(hits, key=lambda r: r.date)[:2 if len(pairs) > 1 else 3]:
                if r.id not in seen:
                    seen.add(r.id)
                    proof.append({"id": r.id, "date": r.date, "text": r.text})
        return proof

    def sample_csv(self):
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(["id", "text", "date", "label"])
        for x in self.sample:
            w.writerow([x["id"], x["text"], x["date"], ""])
        return buf.getvalue()


def make_handler(demo: Demo):
    class H(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="application/json", extra=None):
            data = body if isinstance(body, bytes) else (json.dumps(body, ensure_ascii=False, default=str).encode()
                                                          if ctype == "application/json" else body.encode())
            self.send_response(code)
            self.send_header("Content-Type", ctype + "; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)

        def do_OPTIONS(self):
            self._send(204, b"")

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/api/health":
                return self._send(200, {"ok": True, "sample_rows": len(demo.sample)})
            if u.path == "/api/sample":
                return self._send(200, {"csv": demo.sample_csv()})
            if u.path == "/api/metrics":
                return self._send(200, demo.metrics)
            if u.path == "/api/claims":
                limit = int(parse_qs(u.query).get("limit", ["1000"])[0])
                return self._send(200, {"claims": demo.claims(limit)})
            if u.path == "/api/output.csv":
                path = os.path.join(demo.out_dir, "output.csv")
                if not os.path.exists(path):
                    return self._send(404, {"error": "no run yet"})
                with open(path, "rb") as f:
                    return self._send(200, f.read(), "text/csv",
                                      {"Content-Disposition": 'attachment; filename="output.csv"'})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if urlparse(self.path).path != "/api/run":
                return self._send(404, {"error": "not found"})
            try:
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                rows = [{"id": str(x["id"]).strip(), "text": str(x["text"]).strip(), "date": str(x.get("date", "")).strip()}
                        for x in body.get("rows", []) if x.get("id") and x.get("text")]
                for x in rows:
                    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", x["date"]):
                        x["date"] = "2026-10-29"
                if not rows:
                    return self._send(400, {"error": "no rows"})
                self._send(200, demo.run(rows))
            except Exception as e:  # keep the demo alive; the frontend falls back
                import traceback
                traceback.print_exc()
                self._send(500, {"error": repr(e)})

        def log_message(self, fmt, *args):
            print("[demo]", self.address_string(), fmt % args, flush=True)

    return H


def write_snapshot(demo: Demo, path: str):
    """A real run, saved for the frontend's offline fallback."""
    res = demo.run(demo.sample)
    snap = {"sample_csv": demo.sample_csv(), "run": res, "claims": demo.claims(60), "metrics": demo.metrics}
    with open(path, "w", encoding="utf-8") as f:
        f.write("// Generated by `python3 -m demo.server --snapshot`: a real run of the system on the demo\n"
                "// sample, shown when the API is not reachable. Do not edit by hand.\n"
                "import type { Snapshot } from \"./types\";\n\n"
                "export const SNAPSHOT: Snapshot = " + json.dumps(snap, ensure_ascii=False, indent=1, default=str)
                + " as unknown as Snapshot;\n")
    print(f"[demo] snapshot written: {path}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--practice", default=os.path.join(ROOT, "data", "practice.csv"))
    ap.add_argument("--lexicon", default=os.path.join(ROOT, "data", "lexicon.json"))
    ap.add_argument("--calibration", default=os.path.join(ROOT, "hooly", "assets", "calibration.json"))
    ap.add_argument("--sample", default=None, help="CSV id,text,date[,label] to offer as the sample")
    ap.add_argument("--snapshot", default=None, help="write a frontend fallback snapshot here and exit")
    a = ap.parse_args(argv)
    demo = Demo(a.practice, a.lexicon, a.calibration, a.sample)
    if a.snapshot:
        return write_snapshot(demo, a.snapshot)
    r = demo.run(demo.sample)  # warm up: claims and output.csv exist before the first click
    print(f"[demo] warm-up run: {r['meta']['runtime_ms']} ms, {r['meta']['gold_agree']}/{r['meta']['gold_rows']} "
          f"agree with known labels", flush=True)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(demo))
    print(f"[demo] API on http://{a.host}:{a.port}  (Ctrl+C to stop)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
