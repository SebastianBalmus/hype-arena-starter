# Hype Arena Spotlight UI proposal

Initial UI proposal imported from `hype-arena-spotlight`. This app runs separately
from the Python system in the repository root.

```sh
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite. Use `npm run build` to build, `npm test`
to run the existing tests, and `npm run lint` to check lint rules.
The original Bun dependency lockfile is included for Bun users.

The page starts in **Demo mode**, which returns deterministic placeholder labels,
not real model predictions. **Live API** mode sends `POST /label` to the configured
API URL (default `http://localhost:8000`) with `{ rows: InputRow[] }` and expects
`{ results: LabelResult[] }`; see `src/lib/arena.ts` for types. Connecting this
endpoint to the Python system is future work. The architecture diagram is also
a proposal and must be updated to match the final system.

## Original project notes

This project was built with [Lovable](https://lovable.dev).

## Build with Lovable

Open your project in the [Lovable editor](https://lovable.dev) and keep building.

- **Ship faster**: describe what you want to build and Lovable handles the code.
- **Stay in sync**: connect the project to GitHub and every change made in Lovable is committed straight to your repository.
- **Full ownership**: this code is yours. Push to your repository and your changes sync back into Lovable, ready for your next prompt.

## Development

Prefer working locally? You need Node.js and npm — [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating).

```sh
git clone <this-repository-url>
cd <repository-name>
npm i
npm run dev
```

## Built with

- TanStack Start
- TypeScript
- React
- Tailwind CSS
