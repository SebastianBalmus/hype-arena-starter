import { useEffect, useMemo, useState } from "react";
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
      <button onClick={() => setFlip(!flip)} aria-pressed={flip} aria-label={`${c.text} — ${flip ? "show claim" : "show proof"}`}
        className="flip-inner grid w-full text-left" style={{ transform: flip ? "rotateY(180deg)" : undefined }}>
        <div className="glass-card backface-hidden col-start-1 row-start-1 p-6 transition-transform hover:-translate-y-0.5">
          <div className="flex items-center justify-between"><span className="label text-muted-foreground">{c.type}</span><span className="font-label text-[10px] text-muted-foreground">{c.id}</span></div>
          <p className="mt-5 font-display text-xl font-light leading-snug">{c.text}</p>
          <div className="mt-6 flex items-center gap-3">
            <div className="h-1 flex-1 overflow-hidden rounded-full bg-muted"><div className="h-full bg-primary" style={{ width: `${c.confidence * 100}%` }} /></div>
            <span className="font-label text-[10px]">{c.confidence.toFixed(2)}</span>
          </div>
          <p className="label mt-4 text-[8px] text-muted-foreground">Click for proof</p>
        </div>
        <div className="backface-hidden col-start-1 row-start-1 overflow-hidden rounded-3xl bg-primary p-5 text-primary-foreground [transform:rotateY(180deg)]">
          <p className="label text-[9px] opacity-60">Proof rows</p>
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
  const version = useStore(runVersion);
  useEffect(() => { getClaims().then((r) => setClaims(r.claims)); }, [version]);
  const shown = useMemo(() => claims.filter((c) => (type === "ALL" || c.type === type) && c.confidence >= minConf && c.text.toLowerCase().includes(q.toLowerCase())), [claims, type, q, minConf]);

  return (
    <section id="claims" className="px-6 py-32 md:px-14">
      <SectionHead eyebrow="03 · Claims" a="1,000 new sentences." b="Each with its proof." />
      <Reveal className="mb-10 flex flex-wrap items-center gap-3">
        {TYPES.map((t) => (
          <button key={t} onClick={() => setType(t)} aria-pressed={type === t}
            className={cn("label rounded-full border px-3 py-1.5 transition-colors", type === t ? "border-primary bg-primary text-primary-foreground" : "border-border text-muted-foreground hover:text-foreground")}>{t}</button>
        ))}
        <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search claims" aria-label="Search claims"
          className="glass-pill ml-auto w-full px-4 py-2 text-sm outline-none focus:border-foreground/40 sm:w-56" />
        <label className="label flex items-center gap-2 text-muted-foreground">
          Min {minConf.toFixed(2)}
          <input type="range" min={0.5} max={0.99} step={0.01} value={minConf} onChange={(e) => setMinConf(+e.target.value)} className="accent-[var(--hype)]" />
        </label>
      </Reveal>
      <div className="columns-1 gap-5 sm:columns-2 lg:columns-3">
        {shown.map((c) => <ClaimCard key={c.id} c={c} />)}
      </div>
      {!shown.length && <p className="font-display text-xl text-muted-foreground">No claims match.</p>}
    </section>
  );
}
