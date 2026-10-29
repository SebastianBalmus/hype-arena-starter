import { motion } from "framer-motion";
import { useEffect, useState } from "react";
import { getMetrics } from "@/lib/api";
import type { Metrics } from "@/lib/types";
import { Reveal, SectionHead } from "./ui";
import { cn } from "@/lib/utils";

export function Evaluation() {
  const [m, setM] = useState<Metrics | null>(null);
  useEffect(() => { getMetrics().then(setM); }, []);
  if (!m) return <section id="evaluation" className="min-h-32" />;
  const lo = 0.7, hi = 0.92;
  const maxN = Math.max(...m.reliability.map((r) => r.n));
  const best = Math.max(...m.ladder.map((l) => l.accuracy));
  return (
    <section id="evaluation" className="px-6 py-10 md:px-14 md:py-12">
      <SectionHead eyebrow="04 · Evaluation" a="How good is it," b="honestly?" />
      <div className="grid gap-6 lg:grid-cols-3">
        <Reveal className="lg:col-span-2">
          <div className="glass-card p-8">
            <p className="label mb-8 text-muted-foreground">Accuracy ladder · unseen pages</p>
            <div className="space-y-6">
              {m.ladder.map((l, i) => (
                <div key={l.name}>
                  <div className="mb-2 flex justify-between"><span className="font-display">{l.name}</span><span className="font-label text-xs">{(l.accuracy * 100).toFixed(1)}%</span></div>
                  <div className="h-3 overflow-hidden rounded-full bg-muted">
                    <motion.div className={cn("h-full rounded-full", i === m.ladder.length - 1 ? "bg-primary" : l.accuracy === best ? "bg-primary" : "bg-ghost")}
                      initial={{ width: 0 }} whileInView={{ width: `${((l.accuracy - lo) / (hi - lo)) * 100}%` }} viewport={{ once: true }} transition={{ duration: 1.1, delay: i * 0.15 }} />
                  </div>
                </div>
              ))}
            </div>
            <p className="label mt-6 text-muted-foreground">Axis 70% – 92% · last bar is the shipped system</p>
          </div>
        </Reveal>
        <Reveal delay={0.1}>
          <div className="glass-card flex h-full flex-col justify-between p-8">
            <p className="label text-muted-foreground">Expected on hidden data</p>
            <div>
              <div className="font-display text-7xl font-light">~{Math.round(m.expected.point * 100)}%</div>
              <div className="label mt-3 text-muted-foreground">{Math.round(m.expected.low * 100)}–{Math.round(m.expected.high * 100)}% · plausible range</div>
              <div className="relative mt-6 h-2 rounded-full bg-muted">
                <div className="absolute h-full rounded-full bg-ghost" style={{ left: `${(m.expected.low - 0.8) / 0.15 * 100}%`, width: `${(m.expected.high - m.expected.low) / 0.15 * 100}%` }} />
                <div className="absolute -top-1 h-4 w-1 rounded bg-primary" style={{ left: `${(m.expected.point - 0.8) / 0.15 * 100}%` }} />
              </div>
            </div>
          </div>
        </Reveal>
        <Reveal>
          <div className="glass-card p-8">
            <p className="label mb-4 text-muted-foreground">Reliability · confidence vs accuracy</p>
            <svg viewBox="0 0 220 220" className="w-full" role="img" aria-label="Reliability diagram">
              {[0.5, 0.75, 1].map((t) => { const p = 200 - (t - 0.5) * 2 * 180; return <g key={t}><line x1={20} x2={200} y1={p} y2={p} stroke="var(--border)" /><text x={2} y={p + 3} fontSize={7} fill="var(--muted-foreground)">{t}</text></g>; })}
              <line x1={20} y1={200} x2={200} y2={20} stroke="var(--ghost)" strokeDasharray="4 4" />
              {m.reliability.map((r, i) => (
                <motion.circle key={i} cx={20 + (r.confidence - 0.5) * 2 * 180} cy={200 - (r.accuracy - 0.5) * 2 * 180} fill="var(--fit)" fillOpacity={0.6} stroke="var(--fit)"
                  initial={{ r: 0 }} whileInView={{ r: 4 + (r.n / maxN) * 12 }} viewport={{ once: true }} transition={{ delay: i * 0.1 }} />
              ))}
            </svg>
          </div>
        </Reveal>
        <Reveal className="lg:col-span-2" delay={0.1}>
          <details className="glass-card p-8">
            <summary className="label cursor-pointer text-muted-foreground">Experiments</summary>
            <ol className="relative mt-6 space-y-6 border-l border-border pl-6">
              {m.experiments.map((x) => (
                <li key={x.id} className="relative">
                  <span className={cn("absolute -left-[29px] top-1.5 h-2.5 w-2.5 rounded-full", x.held ? "bg-primary" : "bg-hype")} />
                  <div className="flex flex-wrap items-center gap-3">
                    <span className="font-display text-lg">{x.idea}</span>
                    <span className={cn("label rounded-full border px-2 py-0.5 text-[8px]", x.held ? "border-primary" : "border-hype text-hype")}>{x.held ? "Held" : "Failed"}</span>
                    <span className="label text-[8px] text-muted-foreground">{x.source}</span>
                  </div>
                  <p className="mt-1 text-sm text-muted-foreground">Predicted: {x.prediction} → <span className="text-foreground">{x.outcome}</span></p>
                </li>
              ))}
            </ol>
          </details>
        </Reveal>
      </div>
    </section>
  );
}

export function Footer() {
  return (
    <footer className="bg-footer relative overflow-hidden px-6 pb-8 pt-10 md:px-14">
      <div aria-hidden className="pointer-events-none select-none whitespace-nowrap text-center font-display text-[15vw] font-semibold leading-none tracking-tighter text-foreground/[0.08]">TRACEBACK</div>
      <div className="relative mt-6 flex flex-wrap items-center justify-between gap-6">
        <div>
          <p className="label text-muted-foreground">Built by</p>
          <p className="mt-2 font-display">Team Cercetasii</p>
        </div>
        <a href="https://github.com" target="_blank" rel="noreferrer" className="btn-bracket">[ GitHub ]</a>
      </div>
    </footer>
  );
}
