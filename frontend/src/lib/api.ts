import { create } from "./store";
import { MOCK_CLAIMS, MOCK_METRICS, SAMPLE_CSV, mockRun } from "./mock";
import type { Claim, InputRow, Metrics, RunResponse } from "./types";

export const API_BASE = (import.meta.env['VITE_API_URL'] as string | undefined) ?? "http://localhost:8000";

/** true while we show the saved run because the API is unreachable */
export const demoStore = create(false);
/** bumped after every live run, so the claims section reloads that run's claims */
export const runVersion = create(0);

async function call<T>(path: string, init: RequestInit | undefined, fallback: () => T, timeoutMs = 3500): Promise<T> {
  try {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), timeoutMs);
    const res = await fetch(API_BASE + path, { ...init, signal: ctrl.signal, headers: { "Content-Type": "application/json" } });
    clearTimeout(t);
    if (!res.ok) throw new Error(String(res.status));
    const body = (await res.json()) as T;
    demoStore.set(false);
    return body;
  } catch {
    demoStore.set(true);
    return fallback();
  }
}

export const runRows = async (rows: InputRow[]) => {
  const res = await call<RunResponse>("/api/run", { method: "POST", body: JSON.stringify({ rows }) }, () => mockRun(rows), 120000);
  if (!demoStore.get()) runVersion.set(runVersion.get() + 1);
  return res;
};

export const getClaims = () =>
  call<{ claims: Claim[] }>("/api/claims?limit=1000", undefined, () => ({ claims: MOCK_CLAIMS }), 10000);

export const getMetrics = () => call<Metrics>("/api/metrics", undefined, () => MOCK_METRICS);

export const getSample = () => call<{ csv: string }>("/api/sample", undefined, () => ({ csv: SAMPLE_CSV }));

/** URL of the last live run's output.csv; the cache-buster makes sure it is this run's file */
export const outputCsvUrl = (path: string) => `${API_BASE}${path}?t=${Date.now()}`;
