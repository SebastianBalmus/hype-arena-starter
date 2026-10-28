// Offline fallback: a real, saved run of the system (src/lib/snapshot.ts, written by
// `python3 -m demo.server --snapshot`), used only when the API cannot be reached.
import { SNAPSHOT } from "./snapshot";
import type { Claim, InputRow, Metrics, ResultRow, RunResponse } from "./types";

export const SAMPLE_CSV = SNAPSHOT.sample_csv;

/** Minimal CSV reader for id,text,date[,label]: quoted fields, doubled quotes, header optional. */
export function parseCsv(text: string): InputRow[] {
  const records: string[][] = [];
  let cells: string[] = [], cur = "", q = false;
  const src = text.trim();
  for (let i = 0; i < src.length; i++) {
    const c = src[i];
    if (q) {
      if (c === '"' && src[i + 1] === '"') { cur += '"'; i++; }
      else if (c === '"') q = false;
      else cur += c;
    } else if (c === '"') q = true;
    else if (c === ",") { cells.push(cur); cur = ""; }
    else if (c === "\n" || c === "\r") {
      if (c === "\r" && src[i + 1] === "\n") i++;
      cells.push(cur); records.push(cells); cells = []; cur = "";
    } else cur += c;
  }
  cells.push(cur); records.push(cells);
  const out: InputRow[] = [];
  const start = /^id$/i.test((records[0]?.[0] ?? "").trim()) ? 1 : 0;
  for (const r of records.slice(start)) {
    const [id, txt, date] = [r[0]?.trim() ?? "", r[1]?.trim() ?? "", r[2]?.trim() ?? ""];
    if (id && txt) out.push({ id, text: txt, date });
  }
  return out;
}

const q = (s: string) => (/[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s);

/** The label part of output.csv (contract format), built in the browser when offline. */
export function toOutputCsv(rows: ResultRow[]): string {
  const lines = ["id,type,text,form,label,confidence"];
  for (const r of rows) lines.push(`${q(r.id)},,,,${r.label},${r.confidence == null ? "" : r.confidence.toFixed(4)}`);
  return lines.join("\n") + "\n";
}

export function mockRun(rows: InputRow[]): RunResponse {
  const known = new Map(SNAPSHOT.run.rows.map((r) => [r.text, r]));
  const out = rows.map((r): ResultRow => {
    const hit = known.get(r.text);
    if (hit) return { ...hit, id: r.id, date: r.date || hit.date };
    return {
      id: r.id, text: r.text, date: r.date, label: "unsure", confidence: null, lie_type: null,
      explanation: "The API is offline and this sentence is not in the saved run. Start the API (python3 -m demo.server) to check it live.",
      p_timeline: 0.5, p_hmm: 0.5, facts: [],
    };
  });
  return { rows: out, meta: { ...SNAPSHOT.run.meta, model: `${SNAPSHOT.run.meta.model} (saved run)`, csv_url: undefined } };
}

export const MOCK_CLAIMS: Claim[] = SNAPSHOT.claims;
export const MOCK_METRICS: Metrics = SNAPSHOT.metrics;
