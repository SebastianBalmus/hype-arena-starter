import { act } from "react";
import { createRoot } from "react-dom/client";
import { describe, expect, it, vi } from "vitest";
import { Architecture } from "./Architecture";
import { pipelineStore } from "@/lib/pipeline";

vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} });
vi.stubGlobal("IntersectionObserver", class { observe() {} unobserve() {} disconnect() {} });
vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);

describe("Interactive architecture", () => {
  it("highlights the actual running component and exposes its completed output", async () => {
    const host = document.createElement("div");
    document.body.appendChild(host);
    const root = createRoot(host);
    try {
      pipelineStore.set({ status: "running", nodes: { truth: { status: "running", started_ms: 0, elapsed_ms: 0 } } });
      await act(async () => root.render(<Architecture />));
      const running = host.querySelector('[aria-label="Truth model, running: open details"]');
      expect(running?.className).toContain("ring-fit");
      await act(async () => pipelineStore.set({ status: "complete", nodes: { truth: { status: "complete", started_ms: 0, elapsed_ms: 12, duration_ms: 12, output: { rows: [{ id: "r1", probabilities: { true: 0.9 } }] } } } }));
      await act(async () => host.querySelector<HTMLButtonElement>('[aria-label="Truth model, complete: open details"]')!.click());
      const dialog = host.querySelector('[role="dialog"]')!;
      const technical = Array.from(dialog.querySelectorAll('button')).find((button) => button.textContent === "Technical details")!;
      await act(async () => technical.click());
      const json = host.querySelector('[aria-label="Component output JSON"]');
      expect(json).toHaveTextContent('"r1"');
      expect(json).toHaveTextContent('0.9');
    } finally {
      await act(async () => root.unmount());
      host.remove();
      pipelineStore.set({ status: "idle", nodes: {} });
    }
  });
});
