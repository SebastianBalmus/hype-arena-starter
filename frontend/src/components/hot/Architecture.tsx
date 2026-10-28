import { AnimatePresence, motion } from "framer-motion";
import { useState } from "react";
import { SectionHead } from "./ui";
import { cn } from "@/lib/utils";

type Node = { id: string; title: string; sub: string; x: number; y: number; stat: string; statLabel: string; judge: string; engineer: string };

export const NODES: Node[] = [
  { id: "input", title: "Input rows", sub: "id · text · date", x: 70, y: 150, stat: "19,000", statLabel: "rows in 47 s",
    judge: "Every sentence about the company comes in with the date it was written. Labels only for the practice rows.",
    engineer: "universe.csv (id, text, date, label) and the runtime lexicon. Practice rows get their annotated facts back only when id, text and date all match; every other row is read from its text. Consecutive rows with the same date form a page." },
  { id: "reader", title: "Reader", sub: "value grammar · subjects · attribute scorer", x: 250, y: 150, stat: "0.94 / 0.81", statLabel: "fact precision / recall, unseen pages",
    judge: "The Reader picks out every concrete fact in a sentence — a price, a headcount, a name, a yes or no — and works out what it is about.",
    engineer: "A regex value grammar (money, counts, durations, dates with precision, versions, names, 'more than') finds 2,225 of 2,228 annotated values. Subjects come from lexicon names, distinctive name words and the rest of the page. A logistic-regression scorer ranks (subject, attribute) pairs from cue words, attribute words, counted nouns and closeness to known values, trained with each row's own page left out and 35% cue dropout. Yes/no facts: attribute words plus clause-level negation. An isotonic map turns the score into a parse confidence." },
  { id: "truth", title: "Truth model", sub: "timelines · HMM · hybrid stacker", x: 450, y: 150, stat: "88.8%", statLabel: "accuracy on unseen pages, real parsing",
    judge: "For each fact, it builds a timeline of what the record says and when. It then asks: does this sentence fit what was true on that date?",
    engineer: "Per (subject, attribute): runs of labelled-true values with certain cores and fuzzy edges (four-date windows, as in TAC KBP), states learned with back-off and leave-own-page-out, plus lie shapes (a value held at another time, bigger, smaller, rounder). An HMM fitted by EM over every sentence, labelled or not, adds cross-row consistency. A logistic stacker blends both; where no labelled anchor exists, a confident HMM decides." },
  { id: "calib", title: "Calibration", sub: "isotonic", x: 640, y: 90, stat: "92% → 92%", statLabel: "says 92%, right 92%",
    judge: "When it says it's 92% sure, it's right 92% of the time. Confidence you can actually lean on.",
    engineer: "Cross-fitted isotonic maps per label on held-out pages; ECE 0.025. A false call whose calibrated chance of being right drops below 0.5 becomes true. Unsure only when it is the most likely label." },
  { id: "labels", title: "Labels", sub: "true · false · unsure", x: 820, y: 90, stat: "5", statLabel: "lie types: stale · premature · inflated · deflated · contradicted",
    judge: "Each sentence gets a verdict, a confidence, and — if it's wrong — the kind of wrong.",
    engineer: "Label + calibrated confidence (0.5-1, empty for unsure). The lie type comes from the timeline: a value the fact held earlier (stale) or later (premature), bigger or smaller than the record, or simply contradicted." },
  { id: "writer", title: "Writer", sub: "templates from timelines", x: 640, y: 220, stat: "5", statLabel: "claim types",
    judge: "The Writer turns what the record proves into new, plain sentences: changes, comparisons, facts at events.",
    engineer: "Templates over labelled-true anchors only: changes between consecutive values, date and amount comparisons (at least 1.5x apart), relations through a link attribute, values at an event, plain facts. At most 6 claims per fact, relaxed before a second tier built from rows the system itself labelled true with confidence 0.95+." },
  { id: "referee", title: "Referee", sub: "contract rules", x: 820, y: 220, stat: "100%", statLabel: "claims re-read",
    judge: "The Referee re-reads every claim with the contract's rules and throws out anything it can't fully back up.",
    engineer: "Re-reads each claim with its own extractor: one sentence under 300 characters, every subject named, lexicon attributes only, the stated values match the form, comparisons state no numbers, no copy of an input row." },
  { id: "claims", title: "1,000 claims", sub: "new and true", x: 990, y: 220, stat: "0", statLabel: "contradicted by held-out evidence",
    judge: "What's left: a thousand new statements, each with its proof attached.",
    engineer: "Safest first, deduplicated, written to output.csv with their structured form. Checked against held-out practice rows: none contradicted." },
];

const EDGES: [string, string][] = [["input", "reader"], ["reader", "truth"], ["truth", "calib"], ["calib", "labels"], ["truth", "writer"], ["writer", "referee"], ["referee", "claims"]];
const byId = Object.fromEntries(NODES.map((n) => [n.id, n]));

export function Architecture() {
  const [sel, setSel] = useState<Node | null>(null);
  const [mode, setMode] = useState<"judge" | "engineer">("judge");
  return (
    <section id="architecture" className="relative px-6 py-32 md:px-14">
      <div className="flex flex-wrap items-end justify-between gap-6">
        <SectionHead eyebrow="01 · Architecture" a="Read. Check." b="Then say how sure." className="mb-0" />
        <div role="radiogroup" aria-label="Explanation depth" className="glass-pill flex p-1">
          {(["judge", "engineer"] as const).map((m) => (
            <button key={m} role="radio" aria-checked={mode === m} onClick={() => setMode(m)}
              className={cn("label rounded-full px-4 py-2 transition-colors", mode === m ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground")}>
              Explain like {m === "judge" ? "a judge" : "an engineer"}
            </button>
          ))}
        </div>
      </div>

      <div className="glass-card mt-12 overflow-x-auto p-4 md:p-8">
        <svg viewBox="0 0 1080 300" className="min-w-[860px]" role="group" aria-label="Pipeline diagram">
          {EDGES.map(([a, b], i) => {
            const A = byId[a]!, B = byId[b]!;
            const d = `M${A.x + 60},${A.y} C${(A.x + B.x) / 2 + 30},${A.y} ${(A.x + B.x) / 2 - 30},${B.y} ${B.x - 60},${B.y}`;
            return (
              <g key={i}>
                <path d={d} fill="none" stroke="var(--border)" strokeWidth={6} />
                <path d={d} fill="none" stroke="var(--ghost)" strokeWidth={1.4} className="flow-line" />
                {[0, 1, 2].map((k) => (
                  <circle key={k} r={3} fill={b === "labels" || b === "calib" ? "var(--hype)" : "var(--lilac)"}>
                    <animateMotion dur={`${2.4 + i * 0.2}s`} begin={`${k * 0.8}s`} repeatCount="indefinite" path={d} />
                  </circle>
                ))}
              </g>
            );
          })}
          {NODES.map((n) => (
            <g key={n.id} transform={`translate(${n.x - 60},${n.y - 30})`} className="cursor-pointer" tabIndex={0} role="button"
              aria-label={`${n.title}: open details`} onClick={() => setSel(n)} onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), setSel(n))}>
              <rect width={120} height={60} rx={18} fill={sel?.id === n.id ? "var(--surface)" : "var(--popover)"} fillOpacity={sel?.id === n.id ? 1 : 0.8} stroke="var(--border)" className="transition-all" />
              <text x={60} y={27} textAnchor="middle" fontFamily="var(--font-display)" fontSize={14} fill={sel?.id === n.id ? "var(--primary-foreground)" : "var(--foreground)"}>{n.title}</text>
              <text x={60} y={44} textAnchor="middle" fontFamily="var(--font-label)" fontSize={6} letterSpacing={0.8} fill={sel?.id === n.id ? "var(--primary-foreground)" : "var(--muted-foreground)"}>{n.sub.length > 26 ? n.sub.slice(0, 24) + "…" : n.sub.toUpperCase()}</text>
            </g>
          ))}
        </svg>
        <p className="label mt-4 text-muted-foreground">Click any node</p>
      </div>

      <AnimatePresence>
        {sel && (
          <>
            <motion.div className="fixed inset-0 z-[55] bg-foreground/10" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={() => setSel(null)} />
            <motion.aside role="dialog" aria-label={sel.title} className="glass-card fixed bottom-4 right-4 top-4 z-[56] w-[min(420px,calc(100vw-2rem))] overflow-y-auto bg-popover p-8"
              initial={{ x: 40, opacity: 0 }} animate={{ x: 0, opacity: 1 }} exit={{ x: 40, opacity: 0 }} transition={{ duration: 0.35 }}>
              <button className="label absolute right-6 top-6 text-muted-foreground hover:text-foreground" onClick={() => setSel(null)} autoFocus>[ Close ]</button>
              <p className="label text-muted-foreground">{sel.sub}</p>
              <h3 className="mt-3 text-4xl font-light">{sel.title}</h3>
              <AnimatePresence mode="wait">
                <motion.p key={mode} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} className="mt-6 leading-relaxed text-foreground/80">
                  {mode === "judge" ? sel.judge : sel.engineer}
                </motion.p>
              </AnimatePresence>
              <div className="mt-10 border-t border-border pt-6">
                <div className="font-display text-5xl font-light">{sel.stat}</div>
                <div className="label mt-2 text-muted-foreground">{sel.statLabel}</div>
              </div>
            </motion.aside>
          </>
        )}
      </AnimatePresence>
    </section>
  );
}
