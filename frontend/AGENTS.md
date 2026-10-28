<!-- LOVABLE:BEGIN -->
> [!IMPORTANT]
> This project is connected to [Lovable](https://lovable.dev). Avoid rewriting
> published git history — force pushing, or rebasing/amending/squashing commits
> that are already pushed — as it rewrites history on Lovable's side and the
> user will likely lose their project history.
>
> Commits you push to the connected branch sync back to Lovable and show up in
> the editor, so keep the branch in a working state.
<!-- LOVABLE:END -->

## Project rules
- All API calls go through src/lib/api.ts, which falls back to src/lib/mock.ts and flips the DEMO DATA badge; never block UI on the API.
- Hero particles live in a lazy-loaded canvas (ParticleField) skipped under prefers-reduced-motion; all text stays in React.
