import { AnimatePresence, motion } from "framer-motion";
import { useMemo, useState } from "react";
import { BracketButton, ConfBar, Reveal, SectionHead, VerdictChip } from "./ui";
import { WhyDrawer } from "./WhyDrawer";
import { demoStore, getSample, outputCsvUrl, runRows } from "@/lib/api";
import { parseCsv, toOutputCsv } from "@/lib/mock";
import type { ResultRow, RunMeta, Verdict } from "@/lib/types";
import { cn } from "@/lib/utils";

const STAGES = ["Reader", "Truth", "Calibration"];

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
  const [stage, setStage] = useState(-1);
  const [rows, setRows] = useState<ResultRow[]>([]);
  const [meta, setMeta] = useState<RunMeta | null>(null);
  const [live, setLive] = useState(false);
  const [filter, setFilter] = useState<Verdict | "all">("all");
  const [sortDesc, setSortDesc] = useState(true);
  const [open, setOpen] = useState<ResultRow | null>(null);
  const [drag, setDrag] = useState(false);
  const [err, setErr] = useState("");

  const run = async () => {
    const input = parseCsv(csv);
    if (!input.length) { setErr("Add at least one row: id,text,date,label"); return; }
    setErr(""); setRunning(true); setRows([]); setMeta(null);
    const p = runRows(input);
    for (let i = 0; i < STAGES.length; i++) { setStage(i); await new Promise((r) => setTimeout(r, 520)); }
    const res = await p;
    setRows(res.rows); setMeta(res.meta); setLive(!demoStore.get() && !!res.meta.csv_url);
    setStage(3); setRunning(false);
  };

  const onFile = async (f?: File) => { if (f) setCsv(await f.text()); };

  const saveCsv = () => {
    if (live && meta?.csv_url) return download(outputCsvUrl(meta.csv_url));
    const url = URL.createObjectURL(new Blob([toOutputCsv(rows)], { type: "text/csv" }));
    download(url);
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
    `${meta.model} · ${meta.runtime_ms} ms`,
    meta.gold_rows ? `${meta.gold_agree}/${meta.gold_rows} match the real labels` : null,
    meta.expected_accuracy ? `expected accuracy ${Math.round(meta.expected_accuracy * 100)}%` : null,
    live && meta.claims ? `${meta.claims.toLocaleString()} new claims written` : null,
  ].filter(Boolean).join(" · ") : "";

  return (
    <section id="live" className="px-6 py-32 md:px-14">
      <SectionHead eyebrow="02 · Live run" a="Paste sentences." b="Watch them get checked." />
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
            <textarea id="csv" value={csv} onChange={(e) => setCsv(e.target.value)} spellCheck={false}
              placeholder={"Drop a CSV here, or paste rows:\nr1,\"Hooly AI Pro costs €59 per month.\",2026-03-14,"}
              className="h-72 w-full resize-none rounded-2xl border border-border bg-popover/60 p-4 font-mono text-xs leading-relaxed outline-none focus:border-foreground/40" />
            {err && <p className="mt-2 text-sm text-hype">{err}</p>}
            <div className="mt-4 flex flex-wrap gap-3">
              <BracketButton onClick={async () => setCsv((await getSample()).csv)}>Load sample</BracketButton>
              <BracketButton onClick={run} disabled={running} className="bg-primary text-primary-foreground hover:bg-hype hover:border-hype">{running ? "Running…" : "Run"}</BracketButton>
              <BracketButton onClick={saveCsv} disabled={running || !rows.length} title={live ? "output.csv of this run: labels + 1,000 claims" : "labels only (offline)"}>Download output.csv</BracketButton>
            </div>
            <div className="mt-8 flex items-center gap-2" aria-live="polite">
              {STAGES.map((s, i) => (
                <div key={s} className="flex flex-1 items-center gap-2">
                  <div className={cn("label flex-1 rounded-full border px-3 py-2 text-center text-[9px] transition-all duration-500",
                    stage === i && running ? "border-hype bg-hype text-primary-foreground" : stage > i ? "border-primary bg-primary text-primary-foreground" : "border-border text-muted-foreground")}>{s}</div>
                  {i < 2 && <div className={cn("h-px w-4 transition-colors", stage > i ? "bg-primary" : "bg-border")} />}
                </div>
              ))}
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
                <motion.div className="bg-truth" initial={{ width: 0 }} animate={{ width: `${(counts.true / rows.length) * 100}%` }} />
                <motion.div className="bg-hype" initial={{ width: 0 }} animate={{ width: `${(counts.false / rows.length) * 100}%` }} />
                <motion.div className="bg-muted-foreground/40" initial={{ width: 0 }} animate={{ width: `${(counts.unsure / rows.length) * 100}%` }} />
              </div>
            )}

            {rows.length === 0 ? (
              <div className="flex h-80 flex-col items-center justify-center text-center">
                <p className="font-display text-2xl font-light text-muted-foreground">{running ? "Checking against the record…" : "No results yet."}</p>
                {!running && <p className="label mt-3 text-muted-foreground">Load the sample and press run</p>}
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
                <p className="label mt-4 text-muted-foreground">{metaLine} · click a row for why</p>
              </div>
            )}
          </div>
        </Reveal>
      </div>
      <WhyDrawer row={open} onClose={() => setOpen(null)} />
    </section>
  );
}
