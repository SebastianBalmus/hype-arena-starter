import { AnimatePresence, motion } from "framer-motion";
import { useState } from "react";
import type { Fact, ResultRow } from "@/lib/types";
import { VerdictChip } from "./ui";
import { cn } from "@/lib/utils";

const ts = (d: string) => new Date(d + "T00:00:00Z").getTime();

function Highlighted({ row }: { row: ResultRow }) {
  const facts = [...row.facts].sort((a, b) => a.span[0] - b.span[0]);
  const parts: React.ReactNode[] = [];
  let cur = 0;
  facts.forEach((f, i) => {
    if (f.span[1] <= cur) return;
    parts.push(row.text.slice(cur, f.span[0]));
    parts.push(
      <span key={i} tabIndex={0} className={cn("group relative rounded-md px-1 outline-none", f.p_true >= 0.5 ? "bg-truth/40" : "bg-hype/25")}>
        {row.text.slice(f.span[0], f.span[1])}
        <span className="glass-pill pointer-events-none absolute bottom-full left-0 z-10 mb-2 hidden whitespace-nowrap bg-popover px-3 py-1.5 font-label text-[9px] uppercase tracking-wider group-hover:block group-focus:block">
          {f.subject} · {f.attribute} · {f.value_text} · parse {f.parse_confidence.toFixed(2)}
        </span>
      </span>,
    );
    cur = f.span[1];
  });
  parts.push(row.text.slice(cur));
  return <p className="font-display text-2xl font-light leading-snug">{parts}</p>;
}

function Timeline({ fact }: { fact: Fact }) {
  const vals = fact.timeline.values;
  const dates = [...vals.flatMap((v) => [ts(v.from), ts(v.to)]), ts(fact.timeline.row_date)];
  const pad = 1000 * 60 * 60 * 24 * 60;
  const min = Math.min(...dates) - pad, max = Math.max(...dates) + pad;
  const W = 560, X = (t: number) => ((t - min) / (max - min)) * W;
  const rowX = X(ts(fact.timeline.row_date));
  const fits = fact.p_true >= 0.5;
  const years: number[] = [];
  for (let y = new Date(min).getUTCFullYear() + 1; y <= new Date(max).getUTCFullYear(); y++) years.push(y);
  return (
    <svg viewBox={`0 0 ${W} ${40 + vals.length * 34}`} className="w-full" role="img" aria-label={`Timeline for ${fact.attribute}`}>
      <defs>
        <linearGradient id="fuzz" x1="0" x2="1"><stop offset="0" stopColor="var(--fit)" stopOpacity="0.5" /><stop offset="1" stopColor="var(--fit)" stopOpacity="0" /></linearGradient>
      </defs>
      {years.map((y) => {
        const x = X(Date.UTC(y, 0, 1));
        return <g key={y}><line x1={x} x2={x} y1={0} y2={30 + vals.length * 34} stroke="var(--border)" /><text x={x + 3} y={10} fontSize={8} fontFamily="var(--font-label)" fill="var(--muted-foreground)">{y}</text></g>;
      })}
      {vals.map((v, i) => {
        const y = 22 + i * 34, x1 = X(ts(v.from)), x2 = X(ts(v.to));
        const next = vals[i + 1];
        return (
          <g key={i}>
            <motion.rect x={x1} y={y} height={14} rx={7} fill="var(--surface)" initial={{ width: 0 }} animate={{ width: Math.max(4, x2 - x1) }} transition={{ duration: 0.8, delay: i * 0.15 }} />
            {next && <rect x={x2} y={y} width={Math.max(0, X(ts(next.from)) - x2)} height={14} fill="url(#fuzz)" />}
            <text x={x1} y={y + 26} fontSize={9} fontFamily="var(--font-label)" fill="var(--foreground)">{v.value_text}</text>
          </g>
        );
      })}
      <motion.line x1={rowX} x2={rowX} y1={14} y2={30 + vals.length * 34} stroke="var(--foreground)" strokeDasharray="3 3" initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ delay: 0.5 }} />
      <motion.g initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.9 }}>
        <circle cx={rowX} cy={14} r={6} fill={fits ? "var(--fit)" : "var(--hype)"} />
        <text x={rowX > W * 0.6 ? rowX - 10 : rowX + 10} textAnchor={rowX > W * 0.6 ? "end" : "start"} y={17} fontSize={9} fontFamily="var(--font-label)" fill={fits ? "var(--fit)" : "var(--hype)"}>claims {fact.value_text}</text>
      </motion.g>
    </svg>
  );
}

function Gauge({ label, value, accent }: { label: string; value: number; accent?: boolean }) {
  const r = 30, c = Math.PI * r;
  return (
    <div className="flex flex-col items-center">
      <svg viewBox="0 0 80 46" className="w-24">
        <path d="M10,40 A30,30 0 0 1 70,40" fill="none" stroke="var(--muted)" strokeWidth={7} strokeLinecap="round" />
        <motion.path d="M10,40 A30,30 0 0 1 70,40" fill="none" stroke={accent ? "var(--hype)" : "var(--surface)"} strokeWidth={7} strokeLinecap="round"
          strokeDasharray={c} initial={{ strokeDashoffset: c }} animate={{ strokeDashoffset: c * (1 - value) }} transition={{ duration: 1 }} />
      </svg>
      <div className="-mt-3 font-display text-xl">{value.toFixed(2)}</div>
      <div className="label mt-1 text-center text-[8px] text-muted-foreground">{label}</div>
    </div>
  );
}

function Evidence({ fact }: { fact: Fact }) {
  const [open, setOpen] = useState(false);
  const all = [...fact.evidence.supporting.map((e) => ({ ...e, s: true })), ...fact.evidence.contradicting.map((e) => ({ ...e, s: false }))];
  const list = open ? all : all.slice(0, 2);
  return (
    <div>
      <ul className="space-y-2">
        {list.map((e) => (
          <li key={e.id + e.s} className="flex gap-3 rounded-2xl border border-border bg-popover/60 p-3 text-sm">
            <span className={cn("mt-1.5 h-2 w-2 shrink-0 rounded-full", e.s ? "bg-fit" : "bg-hype")} aria-label={e.s ? "supports" : "contradicts"} />
            <div><div className="label text-[9px] text-muted-foreground">{e.date} · {e.s ? "supports" : "contradicts"}</div><div>{e.text}</div></div>
          </li>
        ))}
      </ul>
      {all.length > 2 && <button className="label mt-3 text-muted-foreground hover:text-foreground" onClick={() => setOpen(!open)}>[ {open ? "Less" : `View evidence · ${all.length}`} ]</button>}
    </div>
  );
}

export function WhyDrawer({ row, onClose }: { row: ResultRow | null; onClose: () => void }) {
  return (
    <AnimatePresence>
      {row && (
        <>
          <motion.div className="fixed inset-0 z-[55] bg-foreground/15" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={onClose} />
          <motion.aside role="dialog" aria-label={`Why ${row.id}`} onKeyDown={(e) => e.key === "Escape" && onClose()}
            className="fixed bottom-0 right-0 top-0 z-[56] w-full max-w-2xl overflow-y-auto border-l border-border bg-popover p-8 md:p-10"
            initial={{ x: "100%" }} animate={{ x: 0 }} exit={{ x: "100%" }} transition={{ type: "spring", damping: 30, stiffness: 260 }}>
            <div className="flex items-center justify-between">
              <span className="label text-muted-foreground">Why · {row.id} · {row.date}</span>
              <button className="label text-muted-foreground hover:text-foreground" onClick={onClose} autoFocus>[ Close ]</button>
            </div>
            <div className="mt-6 flex items-center gap-3"><VerdictChip label={row.label} />{row.lie_type && <span className="label text-hype">{row.lie_type}</span>}</div>
            <div className="mt-6"><Highlighted row={row} /></div>
            {row.entities && row.entities.mentions.length > 0 && (
              <div className="mt-6 rounded-3xl border border-border p-4">
                <p className="label mb-2 text-[9px] text-muted-foreground">Step 1 · entity resolution</p>
                <p className="font-mono text-xs leading-relaxed">{row.entities.normalized_text}</p>
                <div className="mt-3 flex flex-wrap gap-2">
                  {[...new Map(row.entities.mentions.map((m) => [m.entity_id, m])).values()].map((m) => (
                    <span key={m.entity_id} className="label rounded-full border border-border px-2 py-1 text-[8px]">
                      ENTITY:{m.entity_id} = {m.canonical_name}
                    </span>
                  ))}
                </div>
              </div>
            )}
            <p className="mt-6 leading-relaxed text-foreground/80">{row.explanation}</p>

            <div className="mt-8 grid grid-cols-3 gap-2 rounded-3xl border border-border p-5">
              <Gauge label="Timeline model" value={row.p_timeline} />
              <Gauge label="Probabilistic" value={row.p_hmm} />
              <Gauge label="Final, calibrated" value={row.confidence ?? 0.5} accent />
            </div>

            {row.facts.map((f, i) => (
              <div key={i} className="mt-10">
                <p className="label mb-3 text-muted-foreground">Fact {i + 1} · {f.subject} · {f.attribute}</p>
                <div className="rounded-3xl border border-border p-4"><Timeline fact={f} /></div>
                <p className="label mb-3 mt-6 text-muted-foreground">Evidence</p>
                <Evidence fact={f} />
              </div>
            ))}
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
}
