# Traceback frontend

The architecture follows actual events from the Python API. Run rows, choose
**Technical details**, then select a node to inspect its output, timing and JSON.
Row previews are capped at 50 items; practice examples at 8, and claim previews at
20. Totals cover the full stage. The CSV download contains the full run output.

Start the backend from the repository root:

```sh
python3 -m demo.server
```

**Load sample** uses `datasets/sample.csv`, an unlabelled test set. Results are
predictions with confidence; test accuracy cannot be measured without labels.
The offline fallback is a saved run on the same sample. Refresh it with
`python3 -m demo.server --snapshot frontend/src/lib/snapshot.ts`.

Choose one of three actions:

- **Classify rows** checks the supplied CSV and exports `classifications.csv`.
- **Generate claims** writes exactly 1,000 claims and exports `claims.csv`.
  Rows are optional unlabelled context; with no rows, generation uses practice evidence.
- **Run full pipeline + CSV** classifies the rows, generates 1,000 claims, and
  downloads the combined `output.csv` in the submission format.

Each live run has an immutable CSV URL. A later run cannot replace a previous
download. Offline mode shows saved previews; full and claims CSV exports require
the live API. Separate modes require restarting an older backend.
The API trains the reader at startup and waits for an action before running either branch.

In another terminal, start the frontend:

```sh
cd frontend
npm install --legacy-peer-deps --no-package-lock
VITE_API_URL=http://127.0.0.1:8000 npm run dev -- --host 127.0.0.1 --port 5173
```

Restart the Python process after backend changes. `POST /api/run?stream=1` emits
NDJSON stage events followed by the final result; ordinary `POST /api/run` keeps
the JSON response. Older APIs still run rows but cannot show stage outputs.
Both endpoints accept `{ "rows": [...], "mode": "classify" | "generate" | "full" }`;
omitting `mode` runs the full pipeline. Classification skips writer/referee stages;
generation skips classification/calibration stages. Both share extraction and model fitting.
If the API is unreachable, the page displays saved data without live stage traces.
A failed or interrupted live stream is shown as an error.

## Build with Lovable

Open your project in the [Lovable editor](https://lovable.dev) and keep building.

- **Ship faster**: describe what you want to build and Lovable handles the code.
- **Stay in sync**: connect the project to GitHub and every change made in Lovable is committed straight to your repository.
- **Full ownership**: this code is yours. Push to your repository and your changes sync back into Lovable, ready for your next prompt.

## Built with

- TanStack Start
- TypeScript
- React
- Tailwind CSS
