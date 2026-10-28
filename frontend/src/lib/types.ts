export type Verdict = "true" | "false" | "unsure";

export interface EvidenceRow { id: string; date: string; text: string }
export interface Fact {
  subject: string;
  attribute: string;
  value_text: string;
  span: [number, number];
  parse_confidence: number;
  state: string;
  p_true: number;
  timeline: { row_date: string; values: { value_text: string; from: string; to: string }[] };
  evidence: { supporting: EvidenceRow[]; contradicting: EvidenceRow[] };
}
export interface ResultRow {
  id: string;
  text: string;
  date: string;
  label: Verdict;
  confidence: number | null;
  lie_type: string | null;
  explanation: string;
  p_timeline: number;
  p_hmm: number;
  facts: Fact[];
  /** the row's real label, when the server knows it (held-out practice rows): shown, never used */
  gold?: Verdict | undefined;
}
export interface RunMeta {
  runtime_ms: number;
  model: string;
  /** path of the run's output.csv on the API (contract format: labels + 1,000 claims) */
  csv_url?: string | undefined;
  rows?: number | undefined;
  claims?: number | undefined;
  expected_accuracy?: number | undefined;
  gold_rows?: number | undefined;
  gold_agree?: number | undefined;
  regimes?: Record<string, { rows: number; share: number; expected_accuracy: number }> | undefined;
}
export interface RunResponse { rows: ResultRow[]; meta: RunMeta }
export interface InputRow { id: string; text: string; date: string }

export type ClaimType = "CHANGE" | "COMPARE" | "AT EVENT" | "RELATION" | "FACT";
export interface Claim {
  id: string;
  type: ClaimType;
  text: string;
  confidence: number;
  form: Record<string, unknown>;
  proof: EvidenceRow[];
}
export interface Metrics {
  ladder: { name: string; accuracy: number }[];
  reliability: { confidence: number; accuracy: number; n: number }[];
  expected: { point: number; low: number; high: number };
  experiments: { id: string; idea: string; source: string; prediction: string; outcome: string; held: boolean }[];
}
export interface Snapshot { sample_csv: string; run: RunResponse; claims: Claim[]; metrics: Metrics }
