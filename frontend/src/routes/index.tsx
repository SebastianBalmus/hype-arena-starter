import { createFileRoute, ClientOnly } from "@tanstack/react-router";
import { Fragment, lazy, Suspense, useState } from "react";
import Papa from "papaparse";
import { STAGES, labelDemo, labelViaApi, parseRowsCsv, type InputRow, type LabelResult } from "@/lib/arena";
import { EXAMPLE_ROWS } from "@/data/example-rows";

const Diagram3D = lazy(() => import("@/components/arena/Diagram3D"));

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Hype Arena — Fact Checker Console" },
      { name: "description", content: "Architecture diagram and live run console for our Hooly fact-checking system." },
      { property: "og:title", content: "Hype Arena — Fact Checker Console" },
      { property: "og:description", content: "See the pipeline in 3D and label rows live." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary" },
    ],
  }),
  component: Index,
});

const parse = (t: string) => Papa.parse<Record<string, string>>(t, { header: true, skipEmptyLines: true }).data;

function Index() {
  const [rows, setRows] = useState<InputRow[]>(EXAMPLE_ROWS);
  const [results, setResults] = useState<Record<string, LabelResult>>({});
  const [apiUrl, setApiUrl] = useState("http://localhost:8000");
  const [mode, setMode] = useState<"demo" | "api">("demo");
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [openRow, setOpenRow] = useState<string | null>(null);

  async function run() {
    setRunning(true); setError(null); setResults({});
    try {
      const started = Date.now();
      const out = mode === "api" ? await labelViaApi(apiUrl, rows) : labelDemo(rows);
      await new Promise((r) => setTimeout(r, Math.max(0, 1500 - (Date.now() - started))));
      setResults(Object.fromEntries(out.map((r) => [r.id, r])));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Run failed");
    } finally { setRunning(false); }
  }

  function onFile(f: File) {
    f.text().then((t) => { setRows(parseRowsCsv(t, parse)); setResults({}); });
  }

  const counts = Object.values(results).reduce<Record<string, number>>((a, r) => ({ ...a, [r.label]: (a[r.label] ?? 0) + 1 }), {});

  return (
    <div className="min-h-screen bg-background text-foreground">
      <header className="flex items-center justify-between border-b border-border px-6 py-4">
        <div>
          <p className="font-mono text-xs uppercase tracking-[0.3em] text-primary">Hype Arena · Hooly</p>
          <h1 className="font-display text-2xl">Fact Checker Console</h1>
        </div>
        <div className="flex items-center gap-2 font-mono text-xs">
          <span className={`status-dot ${running ? "status-dot-live" : ""}`} />
          {running ? "RUNNING" : "IDLE"}
        </div>
      </header>

      <main className="grid gap-6 p-6 lg:grid-cols-[3fr_2fr]">
        <section className="panel relative h-[520px] overflow-hidden">
          <div className="absolute left-4 top-3 z-10 font-mono text-xs uppercase tracking-widest text-muted-foreground">System architecture · drag to orbit · click a node</div>
          <ClientOnly fallback={<div className="grid h-full place-items-center text-muted-foreground">Loading diagram…</div>}>
            <Suspense fallback={<div className="grid h-full place-items-center text-muted-foreground">Loading diagram…</div>}>
              <Diagram3D running={running} selected={selected} onSelect={setSelected} />
            </Suspense>
          </ClientOnly>
          {selected !== null && (
            <div className="panel absolute bottom-4 left-4 right-4 z-10 p-4">
              <div className="flex justify-between">
                <h3 className="font-display text-lg text-primary">{STAGES[selected]?.name}</h3>
                <button className="font-mono text-xs text-muted-foreground" onClick={() => setSelected(null)}>close</button>
              </div>
              <p className="text-sm text-muted-foreground">{STAGES[selected]?.detail}</p>
            </div>
          )}
        </section>

        <section className="panel flex flex-col gap-4 p-5">
          <div className="flex gap-2 font-mono text-xs">
            {(["demo", "api"] as const).map((m) => (
              <button key={m} onClick={() => setMode(m)} className={`chip ${mode === m ? "chip-on" : ""}`}>
                {m === "demo" ? "Demo mode" : "Live API"}
              </button>
            ))}
          </div>
          {mode === "api" && (
            <label className="text-xs text-muted-foreground">
              System address (POST /label)
              <input value={apiUrl} onChange={(e) => setApiUrl(e.target.value)} className="field mt-1" />
            </label>
          )}
          <label className="text-xs text-muted-foreground">
            Rows CSV (id, text, date, label) — e.g. sample/sample.csv
            <input type="file" accept=".csv" onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])} className="field mt-1" />
          </label>
          <button onClick={run} disabled={running || !rows.length} className="run-btn">
            {running ? "Running…" : `Run on ${rows.length} rows`}
          </button>
          {error && <p className="text-sm text-destructive">{error}</p>}
          <div className="grid grid-cols-3 gap-2 text-center font-mono">
            {(["true", "false", "unsure"] as const).map((l) => (
              <div key={l} className={`stat stat-${l}`}><div className="text-2xl">{counts[l] ?? 0}</div><div className="text-xs uppercase">{l}</div></div>
            ))}
          </div>
        </section>

        <section className="panel lg:col-span-2">
          <table className="w-full text-sm">
            <thead className="font-mono text-xs uppercase text-muted-foreground">
              <tr className="border-b border-border"><th className="p-3 text-left">id</th><th className="p-3 text-left">text</th><th className="p-3 text-left">date</th><th className="p-3 text-left">label</th><th className="p-3 text-left w-40">confidence</th></tr>
            </thead>
            <tbody>
              {rows.map((r) => {
                const res = results[r.id];
                return (
                  <Fragment key={r.id}>
                    <tr key={r.id} onClick={() => setOpenRow(openRow === r.id ? null : r.id)} className="cursor-pointer border-b border-border hover:bg-muted">
                      <td className="p-3 font-mono text-xs">{r.id}</td>
                      <td className="p-3">{r.text}</td>
                      <td className="p-3 font-mono text-xs whitespace-nowrap">{r.date}</td>
                      <td className="p-3">{res ? <span className={`tag tag-${res.label}`}>{res.label}</span> : <span className="text-muted-foreground">—</span>}</td>
                      <td className="p-3">
                        {res?.confidence != null && (
                          <div className="flex items-center gap-2"><div className="bar"><div className="bar-fill" style={{ width: `${((res.confidence - 0.5) / 0.5) * 100}%` }} /></div><span className="font-mono text-xs">{res.confidence.toFixed(2)}</span></div>
                        )}
                      </td>
                    </tr>
                    {openRow === r.id && res?.evidence && (
                      <tr key={r.id + "-ev"} className="border-b border-border bg-muted"><td /><td colSpan={4} className="p-3 font-mono text-xs text-muted-foreground">{res.evidence}</td></tr>
                    )}
                  </Fragment>
                );
              })}
            </tbody>
          </table>
        </section>
      </main>
    </div>
  );
}
