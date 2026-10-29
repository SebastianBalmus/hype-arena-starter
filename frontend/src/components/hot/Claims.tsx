import { useEffect, useMemo, useRef, useState } from "react";
import { getClaims, runVersion } from "@/lib/api";
import { useStore } from "@/lib/store";
import type { Claim, ClaimType } from "@/lib/types";
import { Reveal, SectionHead } from "./ui";
import { cn } from "@/lib/utils";

const TYPES: (ClaimType | "ALL")[] = ["ALL", "CHANGE", "COMPARE", "AT EVENT", "RELATION", "FACT"];

function ClaimCard({ c }: { c: Claim }) {
  const [flip, setFlip] = useState(false);
  return (
    <div className="mb-5 break-inside-avoid [perspective:1200px]">
      <button onClick={() => setFlip(!flip)} aria-pressed={flip} aria-label={`${c.text} — ${flip ? "show claim" : "show evidence"}`}
        className="flip-inner grid w-full text-left" style={{ transform: flip ? "rotateY(180deg)" : undefined }}>
        <div className="glass-card backface-hidden col-start-1 row-start-1 p-6 transition-transform hover:-translate-y-0.5">
          <div className="flex items-center justify-between"><span className="label text-muted-foreground">{c.type}</span><span className="font-label text-[10px] text-muted-foreground">{c.id}</span></div>
          <p className="mt-5 font-display text-xl font-light leading-snug">{c.text}</p>
          <div className="mt-6 flex items-center gap-3">
            <div className="h-1 flex-1 overflow-hidden rounded-full bg-muted"><div className="h-full bg-primary" style={{ width: `${c.confidence * 100}%` }} /></div>
            <span className="font-label text-[10px]">{c.confidence.toFixed(2)}</span>
          </div>
        </div>
        <div className="backface-hidden col-start-1 row-start-1 overflow-hidden rounded-3xl bg-primary p-5 text-primary-foreground [transform:rotateY(180deg)]">
          <p className="label text-[9px] opacity-60">Related evidence rows</p>
          <ul className="mt-2 space-y-2 text-xs">{c.proof.map((p) => <li key={p.id}><span className="opacity-60">{p.date} · </span>{p.text}</li>)}</ul>
          <p className="label mt-4 text-[9px] opacity-60">Structured form</p>
          <pre className="mt-2 overflow-x-auto text-[10px] leading-relaxed text-truth">{JSON.stringify(c.form, null, 2)}</pre>
        </div>
      </button>
    </div>
  );
}

export function Claims() {
  const [claims, setClaims] = useState<Claim[]>([]);
  const [type, setType] = useState<(typeof TYPES)[number]>("ALL");
  const [q, setQ] = useState("");
  const [minConf, setMinConf] = useState(0.6);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(6);
  const resultsRef = useRef<HTMLDivElement>(null);
  const version = useStore(runVersion);
  useEffect(() => {
    let active = true;
    getClaims().then((r) => { if (active) { setClaims(r.claims); setPage(1); } });
    return () => { active = false; };
  }, [version]);
  const shown = useMemo(() => claims.filter((c) => (type === "ALL" || c.type === type) && c.confidence >= minConf && c.text.toLowerCase().includes(q.toLowerCase())), [claims, type, q, minConf]);
  const pageCount = Math.max(1, Math.ceil(shown.length / pageSize));
  const currentPage = Math.min(page, pageCount);
  const start = (currentPage - 1) * pageSize;
  const visible = shown.slice(start, start + pageSize);
  const goToPage = (next: number) => {
    setPage(Math.max(1, Math.min(next, pageCount)));
    resultsRef.current?.scrollIntoView({ block: "start", behavior: "instant" });
  };

  return (
    <section id="claims" className="px-6 py-10 md:px-14 md:py-12">
      <SectionHead eyebrow="03 · Claims" a="New sentences." b="Explore their evidence." />
      <p className="-mt-2 mb-5 text-sm text-muted-foreground">Select a claim to see its evidence.</p>
      <Reveal className="mb-6 flex flex-wrap items-center gap-3">
        {TYPES.map((t) => (
          <button key={t} onClick={() => { setType(t); setPage(1); }} aria-pressed={type === t}
            className={cn("label rounded-full border px-3 py-1.5 transition-colors", type === t ? "border-primary bg-primary text-primary-foreground" : "border-border text-muted-foreground hover:text-foreground")}>{t}</button>
        ))}
        <input value={q} onChange={(e) => { setQ(e.target.value); setPage(1); }} placeholder="Search claims" aria-label="Search claims"
          className="glass-pill ml-auto w-full px-4 py-2 text-sm outline-none focus:border-foreground/40 sm:w-56" />
        <label className="label flex items-center gap-2 text-muted-foreground">
          Min {minConf.toFixed(2)}
          <input type="range" min={0.5} max={0.99} step={0.01} value={minConf} onChange={(e) => { setMinConf(+e.target.value); setPage(1); }} className="accent-[var(--hype)]" />
        </label>
      </Reveal>
      <div ref={resultsRef} className="mb-4 flex scroll-mt-24 flex-wrap items-center justify-between gap-3 text-sm text-muted-foreground">
        <p role="status">{shown.length ? `${start + 1}–${Math.min(start + pageSize, shown.length)} of ${shown.length.toLocaleString()} claims` : "0 claims"}</p>
        <label className="flex items-center gap-2">
          Per page
          <select value={pageSize} onChange={(e) => { setPageSize(Number(e.target.value)); setPage(1); }} className="rounded-lg border border-border bg-popover px-2 py-1 text-foreground">
            {[6, 12, 24].map((size) => <option key={size} value={size}>{size}</option>)}
          </select>
        </label>
      </div>
      <div className="columns-1 gap-5 sm:columns-2 lg:columns-3">
        {visible.map((c) => <ClaimCard key={c.id} c={c} />)}
      </div>
      {!shown.length && <p className="font-display text-xl text-muted-foreground">No claims match.</p>}
      {pageCount > 1 && (
        <nav aria-label="Claims pagination" className="mt-4 flex flex-wrap items-center justify-center gap-3 border-t border-border pt-5 text-sm">
          <button type="button" disabled={currentPage === 1} onClick={() => goToPage(currentPage - 1)} className="rounded-full border border-border px-4 py-2 transition-colors hover:bg-popover disabled:cursor-not-allowed disabled:opacity-40">Previous</button>
          <label className="flex items-center gap-2 text-muted-foreground">
            Page
            <select aria-label="Claims page" value={currentPage} onChange={(e) => goToPage(Number(e.target.value))} className="rounded-lg border border-border bg-popover px-2 py-1 text-foreground">
              {Array.from({ length: pageCount }, (_, i) => <option key={i + 1} value={i + 1}>{i + 1}</option>)}
            </select>
            of {pageCount}
          </label>
          <button type="button" disabled={currentPage === pageCount} onClick={() => goToPage(currentPage + 1)} className="rounded-full border border-border px-4 py-2 transition-colors hover:bg-popover disabled:cursor-not-allowed disabled:opacity-40">Next</button>
        </nav>
      )}
    </section>
  );
}
