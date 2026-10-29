import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useId, useRef, useState } from "react";
import { SectionHead } from "./ui";
import { cn } from "@/lib/utils";
import { pipelineStore, STAGE_NAMES, type StageId } from "@/lib/pipeline";
import { useStore } from "@/lib/store";
import { StageOutput } from "./StageOutput";

type Node = { id: StageId; title: string; sub: string; stat: string; statLabel: string; judge: string; engineer: string };

export const NODES: Node[] = [
  { id: "lexicon", title: "Lexicon", sub: "names · aliases · attributes", stat: "Runtime", statLabel: "supplied with the input",
    judge: "The organisers supply the products, people and events the system can recognise, with their aliases and attributes.",
    engineer: "lexicon.json builds the entity registry and constrains subject/attribute candidates and generated claims. It supplies identities, not unstated fact values." },
  { id: "practice", title: "Practice forms", sub: "labelled examples", stat: "3 fields", statLabel: "ID, text and date must match",
    judge: "The practice data supplies annotated examples for learning how to read sentences. Its true rows also anchor the factual record.",
    engineer: "Batch: packaged forms are re-attached only to labelled runtime rows matching ID, text and date. Demo: labelled practice rows are loaded directly. The Reader learns extraction from annotated rows across labels; false compound rows do not make every extracted fact false. Consecutive same-date rows supply a page hypothesis for context." },
  { id: "input", title: "Input rows", sub: "id · text · date", stat: "19,000", statLabel: "rows in 47 s",
    judge: "Every sentence about the company comes in with the date it was written. Labels only for the practice rows.",
    engineer: "Batch reads universe.csv (id, text, date, label) plus lexicon.json. The live API accepts uploaded ID/text/date rows and combines them with labelled practice rows. Only unlabelled rows receive predictions. Consecutive same-date rows are treated as a page hypothesis, not a supplied page identity. The 47-second figure is a recorded simulated batch benchmark." },
  { id: "entities", title: "Entities", sub: "registry · stable IDs", stat: "62", statLabel: "entities, 0 ambiguous names on practice",
    judge: "Before reading any number, the system works out which product, person or event each sentence is about, using the organisers' list of names and aliases.",
    engineer: "Entity resolution by Sebastian Balmus (hooly/entities.py): every lexicon name and alias matched as whole words in any case and spacing, longest span first, so a name inside a longer name belongs to the longer one, exactly as the claim referee reads names. Each entity gets a stable ID (<ENTITY:23>); a spelling shared by two entities stays unresolved instead of guessed; 'Hooly Android 16.0' is Hooly Android at version 16.0." },
  { id: "reader", title: "Reader", sub: "value grammar · subjects · attribute scorer", stat: "0.94 / 0.81", statLabel: "fact precision / recall, unseen pages",
    judge: "The Reader looks for concrete facts — a price, a headcount, a name, a yes or no — and estimates which subject and attribute each describes. It can miss or misread facts.",
    engineer: "A regex value grammar (money, counts, durations, dates with precision, versions, names, 'more than') finds 2,225 of 2,228 annotated values. Subjects come from the entity registry (previous step), distinctive name words and the rest of the page. A logistic-regression scorer ranks (subject, attribute) pairs from cue words, attribute words, counted nouns and closeness to known values, trained with each row's own page left out and 35% cue dropout. Yes/no facts: attribute words plus clause-level negation. An isotonic map turns the score into a parse confidence." },
  { id: "facts", title: "Structured facts", sub: "subject · attribute · value", stat: "Shared", statLabel: "input to both branches",
    judge: "Both branches use the extracted facts and their source rows, dates and supplied labels. Classification estimates whether uploaded sentences are true; generation starts from facts in labelled-true rows.",
    engineer: "Rows carry extracted or supplied atoms, dates, labels and parse confidence. GatedHybrid fits classification models from these rows. Separately, Writer constructs its own timelines from exact values in labelled-true rows; it does not consume the classifier’s timeline or HMM state. The runtime lexicon also constrains generation and the Referee." },
  { id: "truth", title: "Truth model", sub: "timelines · HMM · hybrid stacker", stat: "88.8%", statLabel: "accuracy on unseen pages, real parsing",
    judge: "For each fact, it builds a timeline of what the record says and when. It then asks: does this sentence fit what was true on that date?",
    engineer: "Two parallel models: TemporalWindowsShape builds runs of labelled-true values, assumes continuity inside their cores, and learns timeline states and lie shapes with each row's page excluded. TimelineHMM uses EM over labelled and unlabelled observations to estimate changing values. A logistic stacker blends their probabilities using out-of-fold HMM training scores. GatedHybrid overrides only to false when all facts lack true anchors, every parse has confidence >= 0.8, HMM false probability >= 0.8, and each fact has observations on at least four dates. These are statistical estimates, not proofs." },
  { id: "calib", title: "Calibration", sub: "isotonic · hype adjustment", stat: "0.025", statLabel: "recorded held-out calibration error",
    judge: "Confidence is adjusted using how often similar predictions were right on held-out pages. Boastful or first-person wording lowers confidence. Future accuracy can differ.",
    engineer: "Per-label isotonic maps ship in hooly/assets/calibration.json. A false prediction with calibrated confidence below 0.5 flips to true; confidence is clamped to 0.5–1, then shrunk towards 0.5 for hype or first-person wording. Unsure is selected upstream and exported with empty confidence. The reported ECE is a held-out measurement, not a live guarantee." },
  { id: "labels", title: "Labels", sub: "true · false · unsure", stat: "5", statLabel: "lie types: stale · premature · inflated · deflated · contradicted",
    judge: "Each unlabelled sentence gets a verdict and confidence. The live demo adds an explanation and, when available, a description of how a false value differs from the record.",
    engineer: "The CSV exports label and calibrated confidence (0.5–1, empty for unsure). The demo API separately adds extracted facts, entity mentions, model scores, timelines and evidence excerpts. Lie types are explanatory timeline heuristics, not a separate trained classifier." },
  { id: "writer", title: "Writer", sub: "own timelines · templates", stat: "5", statLabel: "claim types",
    judge: "The Writer turns dated observations into candidate changes, comparisons, facts, relations and facts at events. It starts with labelled-true evidence and can use highly confident predictions if needed.",
    engineer: "Writer builds its own timelines from exact values in labelled-true rows. It enumerates five template families; fact and compare use 29 October 2026, while change, relation and at_event have their own temporal checks. The per-fact cap starts at 6, then relaxes to 10 and 20. A second tier can include rows predicted true at confidence >= 0.95, with claim confidence scaled by 0.9. Temporal freshness and stability rules are assumptions, not guarantees." },
  { id: "referee", title: "Referee", sub: "contract rules", stat: "100%", statLabel: "claims re-read",
    judge: "The Referee checks that each sentence matches its structured claim and passes the implemented wording, naming and novelty rules. It does not independently establish world truth.",
    engineer: "Re-reads each claim with its own extractor: one sentence under 300 characters, every subject named, lexicon attributes only, the stated values match the form, comparisons state no numbers, no copy of an input row." },
  { id: "claims", title: "Claims", sub: "target · 1,000 new claims", stat: "1,000", statLabel: "target; fewer if candidates run out",
    judge: "The system selects up to a thousand new statements, safest first. The demo shows related evidence rows and structured forms; these excerpts are not a complete proof.",
    engineer: "Candidates are deduplicated by structured form and selected in evidence tiers. generate_claims can return fewer than 1,000 if accepted candidates run out; exactly 1,000 is the submission requirement. The 29 October claims audit records one contradicted and 166 unverifiable claims across five held-out folds (5,000 claims). Demo evidence excerpts are assembled separately from generation checks." },
  { id: "output", title: "output.csv", sub: "labels + structured claims", stat: "Atomic", statLabel: "temporary file, then rename",
    judge: "One downloadable table holds the predicted labels and generated claims. The live page adds explanations alongside those results.",
    engineer: "write_table emits id,type,text,form,label,confidence, replacing output.csv atomically. Batch writes an initial fallback, then predictions, then claims under /output. The demo writes to a temporary directory and serves the latest run at GET /api/output.csv. Evidence excerpts and lie types appear in API responses, not the submission CSV." },
];

const EDGES: [string, string][] = [["input", "entities"], ["lexicon", "entities"], ["practice", "reader"], ["entities", "reader"], ["reader", "facts"], ["facts", "truth"], ["truth", "calib"], ["calib", "labels"], ["facts", "writer"], ["labels", "writer"], ["writer", "referee"], ["referee", "claims"], ["labels", "output"], ["claims", "output"]];
const byId = Object.fromEntries(NODES.map((n) => [n.id, n]));

export function Architecture() {
  const run = useStore(pipelineStore);
  const diagramRef = useRef<HTMLDivElement>(null);
  const nodeRefs = useRef(new Map<string, HTMLButtonElement>());
  const [paths, setPaths] = useState<{ id: string; d: string; optional: boolean }[]>([]);
  const markerId = useId().replace(/:/g, "");
  const [sel, setSel] = useState<Node | null>(null);
  const [mode, setMode] = useState<"judge" | "engineer">("judge");
  useEffect(() => {
    const diagram = diagramRef.current;
    if (!diagram) return;
    let frame = 0;
    let active = true;
    const measure = () => {
      const bounds = diagram.getBoundingClientRect();
      setPaths(EDGES.flatMap(([from, to]) => {
        const source = nodeRefs.current.get(from)?.getBoundingClientRect();
        const target = nodeRefs.current.get(to)?.getBoundingClientRect();
        if (!source || !target) return [];
        const a = { left: source.left - bounds.left, right: source.right - bounds.left, top: source.top - bounds.top, bottom: source.bottom - bounds.top, x: (source.left + source.right) / 2 - bounds.left, y: (source.top + source.bottom) / 2 - bounds.top };
        const b = { left: target.left - bounds.left, right: target.right - bounds.left, top: target.top - bounds.top, bottom: target.bottom - bounds.top, x: (target.left + target.right) / 2 - bounds.left, y: (target.top + target.bottom) / 2 - bounds.top };
        const optional = from === "labels" && to === "writer";
        let d: string;
        if (optional && b.left > a.right) {
          const mid = (a.right + b.left) / 2;
          d = `M${a.right},${a.y} C${mid},${a.y} ${mid},${b.y} ${b.left},${b.y}`;
        } else if (optional) {
          // On phones the branches stack; keep the fallback in the outside gutter.
          d = `M${a.left},${a.y} H8 V${b.y} H${b.left}`;
        } else if (Math.abs(a.y - b.y) < 4 && b.left > a.right) {
          d = `M${a.right},${a.y} H${b.left}`;
        } else {
          const mid = (a.bottom + b.top) / 2;
          d = `M${a.x},${a.bottom} C${a.x},${mid} ${b.x},${mid} ${b.x},${b.top}`;
        }
        return [{ id: `${from}-${to}`, d, optional }];
      }));
    };
    const schedule = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(measure);
    };
    const observer = new ResizeObserver(schedule);
    observer.observe(diagram);
    nodeRefs.current.forEach((node) => observer.observe(node));
    void document.fonts?.ready.then(() => { if (active) schedule(); });
    schedule();
    return () => { active = false; observer.disconnect(); cancelAnimationFrame(frame); };
  }, []);

  const renderNode = (id: StageId) => {
    const node = byId[id]!;
    const stage = run.nodes[id];
    return (
      <button key={id} type="button" ref={(element) => { if (element) nodeRefs.current.set(id, element); else nodeRefs.current.delete(id); }}
        aria-label={`${node.title}${stage ? `, ${stage.status}` : ""}: open details`} aria-haspopup="dialog" onClick={() => setSel(node)}
        className={cn("relative z-10 flex min-h-24 min-w-0 w-full flex-col items-center justify-center gap-2 rounded-2xl border border-border px-3 py-4 text-center transition-colors focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-ring", sel?.id === id ? "bg-primary text-primary-foreground" : "bg-popover text-foreground hover:border-foreground/30",
          stage?.status === "running" && "border-fit ring-2 ring-fit/50 shadow-lg shadow-fit/15",
          stage?.status === "complete" && "border-fit/50",
          stage?.status === "error" && "border-destructive ring-2 ring-destructive/30")}>
        <span className="w-full break-words font-display text-sm font-medium leading-snug">{node.title}</span>
        <span className={cn("w-full break-words text-[11px] leading-relaxed", sel?.id === id ? "text-primary-foreground/80" : "text-muted-foreground")}>{node.sub}</span>
        {stage && <span className={cn("flex items-center gap-1.5 text-[10px]", sel?.id === id ? "text-primary-foreground" : stage.status === "error" ? "text-destructive" : "text-fit")}>
          <span className={cn("h-1.5 w-1.5 rounded-full bg-current", stage.status === "running" && "motion-safe:animate-pulse")} />
          {stage.status === "running" ? "Running" : stage.status === "error" ? "Failed" : `Done · ${stage.duration_ms ?? 0} ms`}
        </span>}
      </button>
    );
  };

  return (
    <section id="architecture" className="relative scroll-mt-24 px-6 py-10 md:px-14 md:py-12">
      <div className="flex flex-wrap items-end justify-between gap-6">
        <SectionHead eyebrow="01 · Architecture" a="One shared Reader." b="Two branches." className="mb-0" />
        <div role="radiogroup" aria-label="Explanation depth" className="glass-pill grid w-full min-w-0 grid-cols-2 p-1 sm:w-auto">
          {(["judge", "engineer"] as const).map((m) => (
            <button key={m} role="radio" aria-checked={mode === m} onClick={() => setMode(m)}
              className={cn("label min-w-0 rounded-full px-3 py-2 text-[9px] transition-colors sm:px-4", mode === m ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground")}>
              {m === "judge" ? "Overview" : "Technical details"}
            </button>
          ))}
        </div>
      </div>

      <div className="glass-card mt-6 min-w-0">
        {run.status !== "idle" && <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border px-5 py-3 text-sm sm:px-8">
          <p role="status" aria-live="polite" className={run.status === "error" ? "text-destructive" : "text-muted-foreground"}>
            {run.status === "connecting" ? "Connecting to the API…"
              : run.status === "queued" ? "Waiting for the pipeline…"
              : run.status === "running" ? `Running: ${Object.entries(run.nodes).filter(([, stage]) => stage.status === "running").map(([id]) => STAGE_NAMES[id as StageId]).join(", ")}`
              : run.status === "complete" ? "Run complete. Select a component to inspect its output."
              : run.status === "offline" ? "Saved demo data · live pipeline unavailable."
              : run.status === "legacy" ? "Restart the updated Python API to enable live component traces."
              : `Run failed: ${run.error ?? "unknown error"}`}
          </p>
          {(run.status === "complete" || run.status === "offline" || run.status === "legacy") && <a className="font-medium text-primary" href="#live">View results ↓</a>}
        </div>}
        <div ref={diagramRef} className="relative isolate p-5 sm:p-8">
          <svg className="pointer-events-none absolute inset-0 h-full w-full overflow-visible" aria-hidden="true">
            <defs>
              <marker id={markerId} viewBox="0 0 8 8" refX={7} refY={4} markerWidth={6} markerHeight={6} orient="auto-start-reverse">
                <path d="M1,1 L7,4 L1,7" fill="none" stroke="var(--ghost)" strokeWidth={1.5} />
              </marker>
            </defs>
            {paths.map(({ id, d, optional }) => (
              <path key={id} d={d} fill="none" stroke="var(--ghost)" strokeWidth={1.5}
                strokeDasharray={optional ? "5 5" : undefined} markerEnd={`url(#${markerId})`} />
            ))}
          </svg>

          <div role="group" aria-label="Inputs" className="grid min-w-0 grid-cols-1 gap-8 sm:grid-cols-3">
            {(["input", "lexicon", "practice"] as const).map(renderNode)}
          </div>

          <div role="group" aria-label="Shared extraction" className="mt-12 grid min-w-0 grid-cols-1 gap-8 sm:grid-cols-3">
            {(["entities", "reader", "facts"] as const).map(renderNode)}
          </div>

          <div className="mt-14 grid min-w-0 grid-cols-1 items-start gap-14 sm:grid-cols-2 sm:gap-10">
            <div role="group" aria-labelledby="classification-heading" className="min-w-0 rounded-3xl border border-border bg-secondary p-4 sm:p-5">
              <h3 id="classification-heading" className="relative z-10 mb-5 w-fit rounded bg-popover px-2 py-1 text-xs font-medium tracking-wide">01 · Classification</h3>
              <div className="grid min-w-0 grid-cols-1 gap-8 xl:grid-cols-3">
                {(["truth", "calib", "labels"] as const).map(renderNode)}
              </div>
            </div>
            <div role="group" aria-labelledby="generation-heading" className="min-w-0 rounded-3xl border border-border bg-accent/50 p-4 sm:p-5">
              <h3 id="generation-heading" className="relative z-10 mb-5 w-fit rounded bg-popover px-2 py-1 text-xs font-medium tracking-wide">02 · Generation</h3>
              <div className="grid min-w-0 grid-cols-1 gap-8 xl:grid-cols-3">
                {(["writer", "referee", "claims"] as const).map(renderNode)}
              </div>
            </div>
          </div>

          <div className="mx-auto mt-12 w-full max-w-sm">{renderNode("output")}</div>
        </div>
        <p className="px-5 pb-5 text-xs leading-relaxed text-muted-foreground sm:px-8 sm:pb-8">Dashed line: optional generation evidence from predictions marked true at ≥ 95% confidence.</p>
      </div>

      <AnimatePresence>
        {sel && (
          <>
            <motion.div className="fixed inset-0 z-[55] bg-foreground/10" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={() => setSel(null)} />
            <motion.aside role="dialog" aria-label={sel.title} onKeyDown={(e) => e.key === "Escape" && setSel(null)} className="glass-card fixed bottom-4 right-4 top-4 z-[56] w-[min(600px,calc(100vw-2rem))] overflow-y-auto bg-popover p-6 sm:p-8"
              initial={{ x: 40, opacity: 0 }} animate={{ x: 0, opacity: 1 }} exit={{ x: 40, opacity: 0 }} transition={{ duration: 0.35 }}>
              <button className="label absolute right-6 top-6 text-muted-foreground hover:text-foreground" onClick={() => setSel(null)} autoFocus>[ Close ]</button>
              <p className="label text-muted-foreground">{sel.sub}</p>
              <h3 className="mt-3 text-4xl font-light">{sel.title}</h3>
              <div className="mt-5 flex flex-wrap gap-2" role="radiogroup" aria-label="Component explanation depth">
                {(["judge", "engineer"] as const).map((value) => <button key={value} role="radio" aria-checked={mode === value} onClick={() => setMode(value)} className={cn("rounded-full border border-border px-3 py-2 text-xs", mode === value && "bg-primary text-primary-foreground")}>{value === "judge" ? "Overview" : "Technical details"}</button>)}
              </div>
              {mode === "engineer" && <StageOutput key={sel.id} id={sel.id} run={run} />}
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
