## Agent skills

### Issue tracker

Local markdown under `.scratch/` (gitignored). See `docs/agents/issue-tracker.md`.

### Domain docs

Single-context: `GLOSSARY.md` and `docs/adr/` at the repo root, created lazily. See `docs/agents/domain.md`.

## Verification

- `npm test` runs the deterministic unit and integration suite covering DOM tweet parsing, multilingual heuristics (EN, ZH, JA), 256d Matryoshka vector truncation, relational feature extraction, thread-clone clustering, online ridge/prototype classification, and MV3 message routing.
- `npm run build` compiles TypeScript (`tsc --noEmit`) and bundles the Chrome Manifest V3 extension (`content.js`, `background.js`, `offscreen.js`, `manifest.json`, `offscreen.html`) into `dist/`.
