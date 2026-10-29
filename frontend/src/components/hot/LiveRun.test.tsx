import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { LiveRun } from "./LiveRun";
import { demoStore, runRows } from "@/lib/api";

vi.mock("@/lib/api", async (original) => ({
  ...await original<typeof import("@/lib/api")>(),
  runRows: vi.fn(),
}));
vi.stubGlobal("IntersectionObserver", class { observe() {} unobserve() {} disconnect() {} });
HTMLElement.prototype.scrollIntoView = vi.fn();

vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
let host: HTMLDivElement;
let root: Root;
const button = (name: string) => Array.from(host.querySelectorAll("button")).find((b) => b.textContent?.replace(/^\[\s*|\s*\]$/g, "").trim() === name)!;
async function renderRun() { await act(async () => root.render(<LiveRun />)); }
async function click(name: string) { await act(async () => button(name).click()); }

beforeEach(() => {
  host = document.createElement("div");
  document.body.appendChild(host);
  root = createRoot(host);
  demoStore.set(false);
  vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
});
afterEach(async () => { await act(async () => root.unmount()); host.remove(); vi.restoreAllMocks(); });

describe("Separate pipeline actions", () => {
  it.each([
    ["Classify rows", "classify", "classifications.csv"],
    ["Generate claims", "generate", "claims.csv"],
    ["Run full pipeline + CSV", "full", "output.csv"],
  ] as const)("%s requests the correct flow and download", async (label, mode, filename) => {
    vi.mocked(runRows).mockResolvedValue({ rows: [], meta: {
      mode, runtime_ms: 1, model: "test", claims: mode === "classify" ? 0 : 1000,
      csv_url: `/api/runs/${mode}/output.csv`,
    } });
    await renderRun();
    await act(async () => {
      const textarea = host.querySelector("textarea")!;
      Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")!.set!.call(textarea, "id,text,date,label\ns-01,Example claim,2026-10-29,");
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
    });
    await click(label);
    expect(button(`Download ${filename}`)).toBeEnabled();
    expect(runRows).toHaveBeenLastCalledWith([{ id: "s-01", text: "Example claim", date: "2026-10-29" }], mode);
    expect(HTMLAnchorElement.prototype.click).toHaveBeenCalledTimes(mode === "full" ? 1 : 0);
    await click(`Download ${filename}`);
    expect(HTMLAnchorElement.prototype.click).toHaveBeenCalledTimes(mode === "full" ? 2 : 1);
    if (mode === "generate") expect(host.querySelector('a[href="#claims"]')).toBeInTheDocument();
  });

  it("allows generation without uploaded rows", async () => {
    vi.mocked(runRows).mockResolvedValue({ rows: [], meta: { mode: "generate", runtime_ms: 1, model: "test", claims: 1000, csv_url: "/api/runs/a/output.csv" } });
    await renderRun();
    await click("Generate claims");
    expect(runRows).toHaveBeenLastCalledWith([], "generate");
    expect(host.querySelector('a[href="#claims"]')).toBeInTheDocument();
  });

  it("requires rows for a full run", async () => {
    vi.mocked(runRows).mockClear();
    await renderRun();
    await click("Run full pipeline + CSV");
    expect(host).toHaveTextContent("Add at least one row");
    expect(runRows).not.toHaveBeenCalled();
  });
});
