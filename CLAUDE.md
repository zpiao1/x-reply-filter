## Agent skills

### Issue tracker

Local markdown under `.scratch/` (gitignored). See `docs/agents/issue-tracker.md`.

### Domain docs

Single-context: `GLOSSARY.md` and `docs/adr/` at the repo root, created lazily. See `docs/agents/domain.md`.

## Verification

- `npm test` runs the deterministic unit and DOM suite covering 256d Matryoshka vector truncation, relational feature extraction, thread-clone clustering, online Passive-Aggressive updates, MV3 RPC routing, multilingual cutoffs (EN, ZH-Hans, ZH-Hant, JA), and in-place DOM dimming.
- `npm run build` compiles TypeScript (`tsc --noEmit`) and bundles `dist/` (`content.js`, `background.js`, `offscreen.js`, `manifest.json`, `offscreen.html`, and local `ort-wasm-simd-threaded.jsep.{mjs,wasm}`).
- Live browser check via `chrome-devtools` MCP is mandatory for any DOM, CSS, or lifecycle change: run `evaluate_script` with `sourcePath: "C:/Users/zpiao/x-reply-filter/scripts/verify-live-dom.js"` and `take_screenshot` on an open `x.com/<user>/status/<id>` tab, asserting `ok: true` (`xrfMutationsInWindow === 0`, `layoutViolations.length === 0`, and `scoredBelowCutoff === 0`).
- Every bug discovered on live `x.com` must first be reproduced as a failing test in `tests/dom-adapter.test.ts` or `tests/domain.test.ts` before changing production code.
