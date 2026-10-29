import { afterEach, describe, expect, it, vi } from "vitest";
import { applyPipelineEvent, pipelineStore, readRunStream, type PipelineEvent } from "./pipeline";
import { demoStore, runRows } from "./api";

const result = { rows: [], meta: { model: "test", runtime_ms: 1 } };
function chunks(text: string, size = 7) {
  const bytes = new TextEncoder().encode(text);
  return new ReadableStream<Uint8Array>({ start(controller) {
    for (let i = 0; i < bytes.length; i += size) controller.enqueue(bytes.slice(i, i + size));
    controller.close();
  } });
}
afterEach(() => { vi.unstubAllGlobals(); pipelineStore.set({ status: "idle", nodes: {} }); });

describe("Pipeline stream", () => {
  it("preserves split JSON and multi-byte values and publishes outputs before the result", async () => {
    const events: PipelineEvent[] = [
      { type: "queued" },
      { type: "stage", node: "reader", status: "running", elapsed_ms: 5 },
      { type: "stage", node: "reader", status: "complete", elapsed_ms: 15, output: { value: "€59 · monthly" } },
      { type: "result", result },
    ];
    const seen: PipelineEvent[] = [];
    expect(await readRunStream(chunks(events.map((e) => JSON.stringify(e)).join("\n"), 1), (e) => {
      seen.push(e); applyPipelineEvent(e);
    })).toEqual(result);
    expect(seen).toEqual(events);
    expect(pipelineStore.get().nodes.reader).toMatchObject({ status: "complete", duration_ms: 10, output: { value: "€59 · monthly" } });
  });

  it("rejects a stream interrupted before its result", async () => {
    await expect(readRunStream(chunks('{"type":"queued"}\n'), () => {})).rejects.toThrow("disconnected");
  });

  it("shows a stage error instead of substituting a saved successful run", async () => {
    const lines = [
      { type: "stage", node: "truth", status: "running", elapsed_ms: 0 },
      { type: "error", error: "model failed" },
    ];
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(chunks(lines.map((e) => JSON.stringify(e)).join("\n")), { headers: { "Content-Type": "application/x-ndjson" } })));
    await expect(runRows([])).rejects.toThrow("model failed");
    expect(pipelineStore.get()).toMatchObject({ status: "error", nodes: { truth: { status: "error" } } });
  });

  it("clears previous traces and marks unreachable-API results as saved data", async () => {
    applyPipelineEvent({ type: "stage", node: "reader", status: "complete", elapsed_ms: 0, output: { old: true } });
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("offline")));
    await runRows([{ id: "unknown", text: "New sentence", date: "2026-10-29" }]);
    expect(pipelineStore.get()).toEqual({ status: "offline", nodes: {} });
    expect(demoStore.get()).toBe(true);
  });

  it("supports older JSON APIs without fabricating stage progress", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify(result), { headers: { "Content-Type": "application/json" } })));
    expect(await runRows([])).toEqual(result);
    expect(pipelineStore.get()).toEqual({ status: "legacy", nodes: {} });
  });

  it.each(["classify", "generate", "full"] as const)("sends the %s mode to the API", async (mode) => {
    const response = { ...result, meta: { ...result.meta, mode } };
    const fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify(response), { headers: { "Content-Type": "application/json" } }));
    vi.stubGlobal("fetch", fetch);
    expect(await runRows([], mode)).toEqual(response);
    expect(JSON.parse(fetch.mock.calls[0]![1].body)).toEqual({ rows: [], mode });
  });

  it("rejects an older API that ignores a partial run mode", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify(result), { headers: { "Content-Type": "application/json" } })));
    await expect(runRows([], "generate")).rejects.toThrow("Restart the Python API");
    expect(pipelineStore.get().status).toBe("error");
  });

  it("does not replace a server rejection with a successful saved run", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("invalid mode", { status: 400 })));
    await expect(runRows([], "classify")).rejects.toThrow("400");
    expect(pipelineStore.get().status).toBe("error");
  });
});
