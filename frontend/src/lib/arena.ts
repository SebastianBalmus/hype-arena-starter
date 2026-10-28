export type Label = "true" | "false" | "unsure";
export interface InputRow { id: string; text: string; date: string; label?: string }
export interface LabelResult { id: string; label: Label; confidence: number | null; evidence?: string }

export interface Stage { id: string; name: string; detail: string }

/** Pipeline stages shown in the diagram. Edit to match the real system. */
export const STAGES: Stage[] = [
  { id: "input", name: "Input rows", detail: "universe.csv: id, text, date, label" },
  { id: "entities", name: "Entity linking", detail: "Match subjects & aliases from lexicon.json" },
  { id: "extract", name: "Fact extraction", detail: "Sentence → (subject, attribute, value)" },
  { id: "timeline", name: "Fact timeline", detail: "Labelled rows build a dated fact store" },
  { id: "verify", name: "Verification", detail: "Compare extracted value with timeline on date" },
  { id: "score", name: "Confidence", detail: "Calibrated score 0.5 – 1" },
  { id: "label", name: "Labels", detail: "true / false / unsure per row" },
  { id: "claims", name: "Claim generator", detail: "1,000 new true claims from the timeline" },
];

/** Edges between stages (index pairs). */
export const EDGES: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [2, 4], [3, 4], [4, 5], [5, 6], [3, 7],
];

/** Expected API: POST {apiUrl}/label  body {rows: InputRow[]} → {results: LabelResult[]} */
export async function labelViaApi(apiUrl: string, rows: InputRow[]): Promise<LabelResult[]> {
  const res = await fetch(`${apiUrl.replace(/\/$/, "")}/label`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ rows }),
  });
  if (!res.ok) throw new Error(`API answered ${res.status}`);
  const json = await res.json();
  return json.results as LabelResult[];
}

/** Demo stand-in so the page works without the Python API. Deterministic. */
export function labelDemo(rows: InputRow[]): LabelResult[] {
  return rows.map((r) => {
    let h = 0;
    for (const ch of r.id + r.text) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
    const n = h % 100;
    const hasNumber = /\d/.test(r.text);
    const label: Label = n < 55 ? "true" : n < 80 ? "false" : "unsure";
    const confidence = label === "unsure" ? null : Math.round((0.55 + (h % 45) / 100) * 100) / 100;
    return {
      id: r.id,
      label,
      confidence,
      evidence: hasNumber ? "Demo: numeric value compared with timeline" : "Demo: entity matched in lexicon",
    };
  });
}

export function parseRowsCsv(text: string, parse: (t: string) => Record<string, string>[]): InputRow[] {
  return parse(text)
    .filter((r) => r["id"] && r["text"])
    .map((r) => ({ id: r["id"]!, text: r["text"]!, date: r["date"] ?? "", label: r["label"] ?? "" }));
}
