import { lazy, Suspense, useEffect, useState } from "react";
import { useReducedMotion, motion } from "framer-motion";
import { BracketButton, Counter } from "./ui";
import { useStore } from "@/lib/store";
import { demoStore } from "@/lib/api";
import type { HoverInfo } from "./ParticleField";

const ParticleField = lazy(() => import("./ParticleField"));

const TRUE_LABELS = ["€59 per month · TRUE 0.94", "HQ Hamburg · TRUE 0.98", "CEO Mara Lindqvist · TRUE 0.89", "2,600 attendees · TRUE 0.95", "Series B €42M · TRUE 0.90"];
const HYPE_LABELS = ["5,000 attendees · HYPE ×1.9", "400+ employees · HYPE ×1.2", "€49 per month · STALE", "“launched” 2023 · PREMATURE", "#1 in Europe · INVENTED"];

export function Nav() {
  const demo = useStore(demoStore);
  const links = [["Architecture", "#architecture"], ["Live run", "#live"]] as const;
  const links2 = [["Claims", "#claims"], ["Evaluation", "#evaluation"]] as const;
  return (
    <nav className="glass-pill fixed left-1/2 top-4 z-50 flex -translate-x-1/2 items-center gap-3 px-4 py-2.5 md:gap-7 md:px-7" aria-label="Main">
      {links.map(([l, h]) => <a key={h} href={h} className="label hidden whitespace-nowrap text-muted-foreground transition-colors hover:text-foreground sm:block">{l}</a>)}
      <a href="#top" aria-label="Traceback home" className="flex shrink-0 items-center gap-2 font-display text-sm font-semibold tracking-tight">
        <img src="/traceback.svg" alt="" width={28} height={28} />
        Traceback
      </a>
      {links2.map(([l, h]) => <a key={h} href={h} className="label hidden whitespace-nowrap text-muted-foreground transition-colors hover:text-foreground sm:block">{l}</a>)}
      {demo && <span className="label rounded-full border border-hype px-2 py-0.5 text-[8px] text-hype" title="API unreachable — showing bundled demo data">Demo data</span>}
    </nav>
  );
}

export function Hero() {
  const reduce = useReducedMotion();
  const [mounted, setMounted] = useState(false);
  const [hover, setHover] = useState<HoverInfo | null>(null);
  useEffect(() => setMounted(true), []);
  const label = hover ? (hover.hype ? HYPE_LABELS : TRUE_LABELS)[hover.index % 5] : null;

  return (
    <header id="top" className="bg-hero relative flex flex-col justify-end overflow-hidden px-6 pb-8 pt-28 md:px-14 md:pt-32">
      {mounted && !reduce && (
        <Suspense fallback={null}>
          <ParticleField onHover={setHover} />
        </Suspense>
      )}
      {label && hover && (
        <div className="glass-pill pointer-events-none absolute z-10 px-3 py-1.5 font-label text-[9px] uppercase tracking-widest" style={{ left: hover.x + 14, top: hover.y - 30 }}>
          <span className={hover.hype ? "text-hype" : "text-foreground"}>{label}</span>
        </div>
      )}
      <div className="pointer-events-none relative z-10 max-w-5xl">
        <motion.h1 initial={reduce ? false : { opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.9 }} className="text-6xl font-light md:text-[9rem]">
          <span className="block">Less hype.</span>
          <span className="block text-primary">More truth.</span>
        </motion.h1>
        <p className="mt-5 max-w-xl text-base leading-relaxed text-foreground/75 md:text-lg">
          Check Hooly claims against the record, with evidence and confidence.
        </p>
        <div className="pointer-events-auto mt-5 flex flex-wrap gap-3">
          <a href="#live"><BracketButton tabIndex={-1}>Run it live</BracketButton></a>
          <a href="#architecture"><BracketButton tabIndex={-1}>Architecture</BracketButton></a>
        </div>
      </div>

      <div className="relative z-10 mt-8 grid gap-6 border-t border-border pt-6 sm:grid-cols-3">
        {[
          [<Counter key="a" to={88.8} decimals={1} suffix="%" />, "accurate on unseen pages"],
          [<Counter key="b" to={1000} />, "target new claims per run"],
          [<><Counter key="c" to={47} /> s</>, "for 19,000 rows"],
        ].map(([n, l], i) => (
          <div key={i}>
            <div className="font-display text-4xl font-light md:text-5xl">{n}</div>
            <div className="label mt-2 text-muted-foreground">{l as string}</div>
          </div>
        ))}
      </div>
    </header>
  );
}
