import { useEffect, useState } from "react";
import { outputCsvUrl } from "@/lib/api";
import type { PipelineState, StageId } from "@/lib/pipeline";

export function StageOutput({ id, run }: { id: StageId; run: PipelineState }) {
  const stage = run.nodes[id];
  const [rowIndex, setRowIndex] = useState("");
  useEffect(() => setRowIndex(""), [stage?.output]);
  const output = stage?.output;
  const object = output && typeof output === "object" ? output as Record<string, unknown> : null;
  const rows = Array.isArray(object?.['rows']) ? object['rows'] as Record<string, unknown>[] : [];
  const selected = rowIndex === "" ? output : { ...object, rows: [rows[Number(rowIndex)]] };
  const json = JSON.stringify(selected, null, 2);
  const download = () => {
    const url = URL.createObjectURL(new Blob([JSON.stringify(output, null, 2)], { type: "application/json" }));
    const a = document.createElement("a");
    a.href = url; a.download = `traceback-${id}.json`; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  const empty = run.status === "offline" ? "Saved demo data has no live component trace. Start the API and run again."
    : run.status === "legacy" ? "This API does not report component outputs. Restart the updated Python API and run again."
    : stage?.status === "error" ? "This component failed before producing output."
    : stage?.status === "running" ? "Running. Output will appear here when this component finishes."
    : run.status === "idle" ? "Run some rows to inspect this component’s output."
    : "This component has not run yet.";
  return (
    <div className="mt-8 border-t border-border pt-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h4 className="text-lg font-medium">Run output</h4>
        {stage?.duration_ms !== undefined && <span className="text-xs text-muted-foreground">{stage.duration_ms} ms</span>}
      </div>
      {output === undefined ? <p className="mt-3 text-sm text-muted-foreground" role="status">{empty}</p> : (
        <>
          <div className="mt-4 flex flex-wrap items-center gap-3 text-xs">
            {!!rows.length && <label className="flex min-w-0 items-center gap-2">Row
              <select aria-label="Inspect output row" className="min-w-0 max-w-64 rounded-lg border border-border bg-popover p-2" value={rowIndex} onChange={(event) => setRowIndex(event.target.value)}>
                <option value="">All preview rows</option>
                {rows.map((row, i) => <option key={i} value={i}>{String(row['id'] ?? i + 1)}</option>)}
              </select>
            </label>}
            <button type="button" className="rounded-lg border border-border px-3 py-2 hover:bg-secondary" onClick={download}>Download JSON</button>
            {id === "output" && typeof object?.['csv_url'] === "string" && <a className="rounded-lg bg-primary px-3 py-2 text-primary-foreground" href={outputCsvUrl(object['csv_url'])} download="output.csv">Download CSV</a>}
          </div>
          {typeof object?.['preview_limit'] === "number" && <p className="mt-3 text-xs text-muted-foreground">Preview limited to {object['preview_limit']} items. Totals cover the full stage.</p>}
          <pre className="mt-4 max-h-[55vh] overflow-y-auto whitespace-pre-wrap break-words rounded-xl border border-border bg-secondary/60 p-4 font-mono text-[11px] leading-relaxed" tabIndex={0} aria-label="Component output JSON">{json}</pre>
        </>
      )}
    </div>
  );
}
