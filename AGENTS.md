# AGENTS.md

## Cursor Cloud specific instructions

This is a single-package, client-side-only TypeScript/Vite project. There are no backend services, databases, or Docker dependencies.

### Key commands

See `package.json` scripts for the full list. The important ones:

- **Dev server**: `npm run dev` — Vite on port 5173 at path `/g9jax/`
- **Type check**: `npm run check` — runs `tsc --noEmit`
- **Offline tests**: `npm run test:offline` — headless regression tests via `tsx` (no browser needed)
- **Build**: `npm run build` — production build to `dist/`

### Non-obvious notes

- The app requires **Cross-Origin Isolation** headers (`Cross-Origin-Opener-Policy` and `Cross-Origin-Embedder-Policy`) for `SharedArrayBuffer` support, which the `@jax-js/jax` WASM backend needs. Vite is already configured to serve these headers in `vite.config.js`.
- A `coi-serviceworker.js` in the root provides cross-origin isolation for environments that don't support custom headers (e.g., GitHub Pages).
- The Vite dev server base path is `/g9jax/`, so the app is served at `http://localhost:5173/g9jax/`, not the root.
- The `@jax-js/jax` dependency is pinned to `latest` in `package.json`. Running `npm install` may pull a newer version.
- Node.js 22 is used (per CI workflow).
- There is no linter (ESLint) configured; the closest lint-like check is `npm run check` (TypeScript).
