import { AnimatePresence, motion } from "framer-motion";
import { useMemo, useState } from "react";
import { BracketButton, ConfBar, Reveal, SectionHead, VerdictChip } from "./ui";
import { WhyDrawer } from "./WhyDrawer";
import { demoStore, getSample, outputCsvUrl, runRows } from "@/lib/api";
import { parseCsv, toOutputCsv } from "@/lib/mock";
import type { ResultRow, RunMeta, RunMode, Verdict } from "@/lib/types";
import { cn } from "@/lib/utils";
import { pipelineStore, STAGE_NAMES, type StageId } from "@/lib/pipeline";
import { useStore } from "@/lib/store";

const STAGES: Record<RunMode, StageId[]> = {
  classify: ["reader", "truth", "calib", "labels", "output"],
  generate: ["reader", "truth", "writer", "referee", "claims", "output"],
  full: ["reader", "truth", "calib", "labels", "writer", "referee", "claims", "output"],
};
const CSV_NAMES: Record<RunMode, string> = { classify: "classifications.csv", generate: "claims.csv", full: "output.csv" };

function download(url: string, name = "output.csv") {
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
}

export function LiveRun() {
  const [csv, setCsv] = useState("");
  const [running, setRunning] = useState(false);
  const [runMode, setRunMode] = useState<RunMode>("full");
  const pipeline = useStore(pipelineStore);
  const [rows, setRows] = useState<ResultRow[]>([]);
  const [meta, setMeta] = useState<RunMeta | null>(null);
  const [live, setLive] = useState(false);
  const [filter, setFilter] = useState<Verdict | "all">("all");
  const [sortDesc, setSortDesc] = useState(true);
  const [open, setOpen] = useState<ResultRow | null>(null);
  const [drag, setDrag] = useState(false);
  const [err, setErr] = useState("");

  const run = async (mode: RunMode) => {
    const input = parseCsv(csv);
    if (!input.length && (mode !== "generate" || csv.trim())) { setErr("Add at least one row: id,text,date,label"); return; }
    setRunMode(mode);
    setErr(""); setRunning(true); setRows([]); setMeta(null); setOpen(null);
    const request = runRows(input, mode);
    document.getElementById("architecture")?.scrollIntoView({ block: "start", behavior: "instant" });
    try {
      const res = await request;
      setRows(res.rows); setMeta(res.meta); setLive(!demoStore.get() && !!res.meta.csv_url);
      if (mode === "full" && !demoStore.get() && res.meta.csv_url) download(outputCsvUrl(res.meta.csv_url));
    } catch (error) {
      setErr(error instanceof Error ? error.message : String(error));
    } finally {
      setRunning(false);
    }
  };

  const onFile = async (f?: File) => { if (f) setCsv(await f.text()); };

  const saveCsv = () => {
    if (live && meta?.csv_url) return download(outputCsvUrl(meta.csv_url), CSV_NAMES[runMode]);
    const url = URL.createObjectURL(new Blob([toOutputCsv(rows)], { type: "text/csv" }));
    download(url, "classifications.csv");
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  const counts = useMemo(() => ({
    true: rows.filter((r) => r.label === "true").length,
    false: rows.filter((r) => r.label === "false").length,
    unsure: rows.filter((r) => r.label === "unsure").length,
  }), [rows]);

  const shown = useMemo(() => rows
    .filter((r) => filter === "all" || r.label === filter)
    .sort((a, b) => ((a.confidence ?? 0) - (b.confidence ?? 0)) * (sortDesc ? -1 : 1)), [rows, filter, sortDesc]);

  const metaLine = meta ? [
    `${meta.runtime_ms} ms`,
    runMode !== "generate" ? `${rows.length} rows classified` : null,
    runMode !== "generate" ? (meta.gold_rows ? `${meta.gold_agree}/${meta.gold_rows} match the real labels` : "Test labels unavailable · accuracy not measured") : null,
    live && meta.claims ? `${meta.claims.toLocaleString()} new claims written` : null,
    !live ? "Saved preview · start the API to generate a CSV" : null,
  ].filter(Boolean).join(" · ") : "";

  return (
    <section id="live" className="scroll-mt-24 px-6 py-10 md:px-14 md:py-12">
      <SectionHead eyebrow="02 · Live run" a="Check claims. Write claims." b="Or run both." />
      <div className="grid gap-6 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
        <Reveal>
          <div className={cn("glass-card p-6 transition-colors", drag && "ring-2 ring-hype")}
            onDragOver={(e) => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)}
            onDrop={(e) => { e.preventDefault(); setDrag(false); onFile(e.dataTransfer.files[0]); }}>
            <div className="mb-3 flex items-center justify-between">
              <label htmlFor="csv" className="label text-muted-foreground">CSV · id,text,date,label</label>
              <label className="label cursor-pointer text-muted-foreground hover:text-foreground">
                Upload<input type="file" accept=".csv,text/csv" className="sr-only" onChange={(e) => onFile(e.target.files?.[0])} />
              </label>
            </div>
            <p className="mb-3 text-xs text-muted-foreground">Sample: datasets/sample.csv · unlabelled test rows.</p>
            <textarea id="csv" value={csv} onChange={(e) => setCsv(e.target.value)} spellCheck={false}
              placeholder={"Drop a CSV here, or paste rows:\nr1,\"Hooly AI Pro costs €59 per month.\",2026-03-14,"}
              className="h-72 w-full resize-none rounded-2xl border border-border bg-popover/60 p-4 font-mono text-xs leading-relaxed outline-none focus:border-foreground/40" />
            {err && <p className="mt-2 text-sm text-hype">{err}</p>}
            <div className="mt-4 flex flex-wrap gap-3">
              <BracketButton disabled={running} onClick={async () => setCsv((await getSample()).csv)}>Load sample</BracketButton>
              <BracketButton onClick={() => run("classify")} disabled={running}>{running && runMode === "classify" ? "Classifying…" : "Classify rows"}</BracketButton>
              <BracketButton onClick={() => run("generate")} disabled={running}>{running && runMode === "generate" ? "Generating…" : "Generate claims"}</BracketButton>
              <BracketButton onClick={() => run("full")} disabled={running} className="bg-primary text-primary-foreground hover:bg-primary/90 hover:border-primary">{running && runMode === "full" ? "Running full pipeline…" : "Run full pipeline + CSV"}</BracketButton>
              <BracketButton onClick={saveCsv} disabled={running || !meta || (!live && (runMode !== "classify" || !rows.length))}>Download {CSV_NAMES[runMode]}</BracketButton>
            </div>
            <p className="mt-3 text-xs leading-relaxed text-muted-foreground">Classify rows checks the CSV. Generate claims writes 1,000 claims from the evidence; input rows are optional context. The full pipeline downloads one CSV with both classifications and claims.</p>
            <div className="mt-6 flex flex-wrap items-center gap-2" aria-label="Pipeline progress">
              {STAGES[runMode].map((id) => {
                const state = pipeline.nodes[id]?.status;
                return <span key={id} className={cn("rounded-full border px-3 py-2 text-[10px] transition-colors",
                  state === "running" ? "border-fit bg-accent text-primary motion-safe:animate-pulse" : state === "complete" ? "border-primary bg-primary text-primary-foreground" : state === "error" ? "border-destructive text-destructive" : "border-border text-muted-foreground")}>{STAGE_NAMES[id]}{state === "complete" ? " ✓" : state === "running" ? "…" : state === "error" ? " !" : ""}</span>;
              })}
            </div>
          </div>
        </Reveal>

        <Reveal delay={0.1}>
          <div className="glass-card min-h-[28rem] p-6">
            <div className="mb-4 flex flex-wrap items-center gap-2">
              {(["all", "true", "false", "unsure"] as const).map((f) => (
                <button key={f} onClick={() => setFilter(f)} aria-pressed={filter === f}
                  className={cn("label rounded-full border px-3 py-1.5 transition-colors", filter === f ? "border-primary bg-primary text-primary-foreground" : "border-border text-muted-foreground hover:text-foreground")}>
                  {f}{f !== "all" && rows.length ? ` · ${counts[f]}` : ""}
                </button>
              ))}
              <button onClick={() => setSortDesc(!sortDesc)} className="label ml-auto text-muted-foreground hover:text-foreground">Confidence {sortDesc ? "↓" : "↑"}</button>
            </div>

            {rows.length > 0 && (
              <div className="mb-4 flex h-2 overflow-hidden rounded-full" aria-label={`${counts.true} true, ${counts.false} false, ${counts.unsure} unsure`}>
                <motion.div className="bg-fit" initial={{ width: 0 }} animate={{ width: `${(counts.true / rows.length) * 100}%` }} />
                <motion.div className="bg-hype" initial={{ width: 0 }} animate={{ width: `${(counts.false / rows.length) * 100}%` }} />
                <motion.div className="bg-muted-foreground/40" initial={{ width: 0 }} animate={{ width: `${(counts.unsure / rows.length) * 100}%` }} />
              </div>
            )}

            {rows.length === 0 ? (
              <div className="flex h-80 flex-col items-center justify-center text-center">
                <p className="font-display text-2xl font-light text-muted-foreground">{running ? (runMode === "generate" ? "Generating claims…" : "Checking…") : meta && runMode === "generate" ? (live ? `${meta.claims?.toLocaleString()} claims generated.` : "Saved claims preview · API unavailable.") : "Load a sample or upload a CSV."}</p>
                {meta && runMode === "generate" && <a href="#claims" className="mt-4 text-sm font-medium text-primary">Explore claims and evidence ↓</a>}
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-sm">
                  <thead><tr className="label text-muted-foreground">
                    <th className="py-2 pr-3 font-normal">Id</th><th className="py-2 pr-3 font-normal">Sentence</th><th className="py-2 pr-3 font-normal">Date</th>
                    <th className="py-2 pr-3 font-normal">Verdict</th><th className="py-2 pr-3 font-normal">Confidence</th><th className="py-2 font-normal">Type</th>
                  </tr></thead>
                  <tbody>
                    <AnimatePresence>
                      {shown.map((r, i) => (
                        <motion.tr key={r.id} layout initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ delay: i * 0.05 }}
                          tabIndex={0} onClick={() => setOpen(r)} onKeyDown={(e) => e.key === "Enter" && setOpen(r)}
                          className="cursor-pointer border-t border-border outline-none transition-colors hover:bg-popover/70 focus-visible:bg-popover/70">
                          <td className="py-3 pr-3 font-label text-[10px] text-muted-foreground">{r.id}</td>
                          <td className="max-w-xs py-3 pr-3">{r.text}</td>
                          <td className="whitespace-nowrap py-3 pr-3 text-xs text-muted-foreground">{r.date}</td>
                          <td className="py-3 pr-3">
                            <VerdictChip label={r.label} />
                            {r.gold && (
                              <div className={cn("label mt-1 text-[8px]", r.gold === r.label ? "text-fit" : "text-hype")}
                                title="the row's real label, held out from the system">
                                {r.gold === r.label ? "✓ real label" : `✗ real: ${r.gold}`}
                              </div>
                            )}
                          </td>
                          <td className="py-3 pr-3"><ConfBar value={r.confidence} /></td>
                          <td className="py-3">{r.lie_type && <span className="label rounded-full border border-hype/50 px-2 py-1 text-[8px] text-hype">{r.lie_type}</span>}</td>
                        </motion.tr>
                      ))}
                    </AnimatePresence>
                  </tbody>
                </table>
                <p className="mt-4 text-xs text-muted-foreground">Select a row for evidence.</p>
              </div>
            )}
            {meta && <p role="status" className="mt-4 text-xs text-muted-foreground">{metaLine}</p>}
          </div>
        </Reveal>
      </div>
      <WhyDrawer row={open} onClose={() => setOpen(null)} />
    </section>
  );
}
