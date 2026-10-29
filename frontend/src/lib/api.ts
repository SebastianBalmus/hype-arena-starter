import { create } from "./store";
import { MOCK_CLAIMS, MOCK_METRICS, SAMPLE_CSV, mockRun } from "./mock";
import type { Claim, InputRow, Metrics, RunMode, RunResponse } from "./types";
import { applyPipelineEvent, pipelineStore, readRunStream } from "./pipeline";

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

export const runRows = async (rows: InputRow[], mode: RunMode = "full") => {
  pipelineStore.set({ status: "connecting", nodes: {} });
  const ctrl = new AbortController();
  const timeout = setTimeout(() => ctrl.abort(), 120000);
  let streaming = false;
  let responded = false;
  try {
    const response = await fetch(API_BASE + "/api/run?stream=1", {
      method: "POST", body: JSON.stringify({ rows, mode }), signal: ctrl.signal,
      headers: { "Content-Type": "application/json" },
    });
    responded = true;
    if (!response.ok) throw new Error(`API returned ${response.status}`);
    streaming = response.headers.get("Content-Type")?.includes("application/x-ndjson") ?? false;
    if (streaming && !response.body) throw new Error("The API returned no progress stream.");
    const result = streaming
      ? await readRunStream(response.body!, applyPipelineEvent)
      : await response.json() as RunResponse;
    if (mode !== "full" && result.meta.mode !== mode) throw new Error("Restart the Python API to enable separate classification and generation.");
    pipelineStore.set({ ...pipelineStore.get(), status: streaming ? "complete" : "legacy" });
    demoStore.set(false);
    if (mode !== "classify") runVersion.set(runVersion.get() + 1);
    return result;
  } catch (error) {
    if (streaming || responded) {
      const state = pipelineStore.get();
      const message = error instanceof Error ? error.message : String(error);
      const nodes = { ...state.nodes };
      for (const key of Object.keys(nodes) as (keyof typeof nodes)[]) {
        const node = nodes[key];
        if (node?.status === "running") nodes[key] = { ...node, status: "error" };
      }
      pipelineStore.set({ ...state, nodes, status: "error", error: message });
      throw error;
    }
    demoStore.set(true);
    pipelineStore.set({ status: "offline", nodes: {} });
    if (mode !== "classify") runVersion.set(runVersion.get() + 1);
    return mockRun(rows, mode);
  } finally {
    clearTimeout(timeout);
    ctrl.abort();
  }
};

export const getClaims = () =>
  call<{ claims: Claim[] }>("/api/claims?limit=1000", undefined, () => ({ claims: MOCK_CLAIMS }), 10000);

export const getMetrics = () => call<Metrics>("/api/metrics", undefined, () => MOCK_METRICS);

export const getSample = () => call<{ csv: string }>("/api/sample", undefined, () => ({ csv: SAMPLE_CSV }));

/** URL of the last live run's output.csv; the cache-buster makes sure it is this run's file */
export const outputCsvUrl = (path: string) => `${API_BASE}${path}?t=${Date.now()}`;
