import { create } from "./store";
import type { RunResponse } from "./types";

export const STAGE_NAMES = {
  input: "Input rows", lexicon: "Lexicon", practice: "Practice forms",
  entities: "Entities", reader: "Reader", facts: "Structured facts",
  truth: "Truth model", calib: "Calibration", labels: "Labels",
  writer: "Writer", referee: "Referee", claims: "Claims", output: "output.csv",
} as const;
export type StageId = keyof typeof STAGE_NAMES;
export interface StageSnapshot {
  status: "running" | "complete" | "error";
  started_ms: number;
  elapsed_ms: number;
  duration_ms?: number;
  output?: unknown;
}
export interface PipelineState {
  status: "idle" | "connecting" | "queued" | "running" | "complete" | "offline" | "legacy" | "error";
  nodes: Partial<Record<StageId, StageSnapshot>>;
  error?: string;
}
export type PipelineEvent =
  | { type: "queued" }
  | { type: "stage"; node: StageId; status: "running" | "complete"; elapsed_ms: number; output?: unknown }
  | { type: "error"; error: string }
  | { type: "result"; result: RunResponse };

export const pipelineStore = create<PipelineState>({ status: "idle", nodes: {} });

export function applyPipelineEvent(event: PipelineEvent) {
  const state = pipelineStore.get();
  if (event.type === "queued") pipelineStore.set({ ...state, status: "queued" });
  if (event.type === "stage") {
    if (!(event.node in STAGE_NAMES)) return;
    const started = state.nodes[event.node]?.started_ms ?? event.elapsed_ms;
    pipelineStore.set({ status: "running", nodes: { ...state.nodes, [event.node]: {
      status: event.status, started_ms: started, elapsed_ms: event.elapsed_ms,
      ...(event.status === "complete" ? { duration_ms: event.elapsed_ms - started } : {}),
      ...(event.output !== undefined ? { output: event.output } : {}),
    } } });
  }
}

/** Decode incremental NDJSON without losing lines or splitting UTF-8 characters. */
export async function readRunStream(stream: ReadableStream<Uint8Array>, onEvent: (event: PipelineEvent) => void): Promise<RunResponse> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let pending = "";
  let result: RunResponse | undefined;
  const consume = (line: string) => {
    if (!line.trim()) return;
    const event = JSON.parse(line) as PipelineEvent;
    if (event.type === "error") throw new Error(event.error);
    if (event.type === "result") result = event.result;
    onEvent(event);
  };
  try {
    while (true) {
      const { value, done } = await reader.read();
      pending += decoder.decode(value, { stream: !done });
      const lines = pending.split("\n");
      pending = lines.pop() ?? "";
      lines.forEach(consume);
      if (done) break;
    }
    consume(pending);
    if (!result) throw new Error("The API disconnected before the run finished.");
    return result;
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}
