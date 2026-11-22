# X Bot & AI Slop Reply Filter

A local, privacy-preserving Chrome Manifest V3 extension that filters AI-generated slop, engagement-bait reply guys, bot clones, and spam on X (`/status/<id>` threads) across **English**, **Simplified/Traditional Chinese**, and **Japanese**.

All inference, scoring, and online feedback learning run **100% locally inside your browser** using `EmbeddingGemma-300M` on WebGPU (with WASM fallback) and an optional opt-in Chrome Built-in AI (`LanguageModel` / Gemini Nano) Stage-2 refiner—no third-party servers, telemetry, or external API keys.

## Key Features

- **256d Semantic + 58-Dim Calibrated Scoring (`COMPACT_V2`)**: Combines `onnx-community/embeddinggemma-300m-ONNX` (256d Matryoshka embeddings) with OP-relational novelty, multilingual AI-slop stylometrics (contrastive cadence like *"not just X, it's Y"*, sycophantic openers, bait closers, Latin-only em-dash heuristics), cross-language mismatch signals, and 36 disjoint multi-domain EN/ZH/JA prototypes.
- **Precision-First 3-Tier Verdicts (`dim` / `suspect` / `keep`)**:
  - `dimmed`: collapses high-confidence bots/slop into a 1-line bar without shifting X's virtualized scroll position.
  - `suspect`: marks borderline replies with a subtle amber pill and thin inset left accent while keeping tweet text 100% readable.
  - `clean`: leaves human replies untouched (hover reveals a quick **Mark bot** action and explainable signal tooltip).
- **Rule Table v2 & Trust Shields (`R0–R12`, `C1–C5`)**:
  - Protects OP follow-up replies, accounts you follow (`following: true`), and conversation context ancestors above the focal tweet.
  - Preserves casual unverified human reactions (`lol`, `草`, `哈哈`, `wtf`) while catching unfollowed blue-check short praise, Twemoji spam, zero-text media/GIF reply guys, and `@grok` bot summons (without collapsing substantive Grok discussions).
  - Caps org-verified accounts, `followedBy` mutuals, and high-like (`>= 100` likes) replies at `suspect`.
- **Passive MAIN-World GraphQL Tap + DOM Fallback**: Passively taps X's `TweetDetail` / `TweetResultByRestId` responses in `world: "MAIN"` (zero extra network requests) to inspect verified type, follow status, account age, bio, and engagement ratios—degrading gracefully to DOM-only heuristics when metadata is absent. Also extracts Twemoji `<img alt>` icons and isolates nested quote tweets.
- **Regularized Online Learning & Popup Controls**:
  - `Conservative`, `Balanced` (default), and `Strict` sensitivity presets plus 6 policy toggles synced via `chrome.storage.sync`.
  - 512-entry topic-balanced replay buffer regularized toward the calibrated baseline `θ₀` with an anchor-set precision rollback gate, plus **Undo**, **Export/Import JSON**, and **Reset** controls.
  - Optional opt-in **Chrome Built-in AI (`LanguageModel`)** Stage-2 judge for borderline `suspect` replies on Chrome 138+ desktop.

## Benchmark Highlights

Evaluated on a 200-case multilingual dataset (`benchmarks/dataset.json`, 100 human / 100 bot across EN, ZH-Hans, ZH-Hant, JA) using the production `ThreadFilterEngine` (`benchmarks/verify-prod-parity.ts` & `benchmarks/benchmark-report.md`):

| Mode | P@dim | R@dim | F1@dim | Human FPR | Suspect Capture | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Baseline (v1) | 94.9% | 37.0% | 53.2% | 2.0% | 37.0% | 0.818 |
| **v2 Balanced (default)** | **100.0%** | **93.0%** | **96.4%** | **0.0%** | **99.0%** | **0.995** |
| **v2 Conservative** | **100.0%** | 66.0% | 79.5% | **0.0%** | 98.0% | 0.995 |
| **v2 Strict** | 99.0% | **96.0%** | **97.5%** | 1.0% | **99.0%** | 0.995 |
| **v2 Balanced (100% DOM-only)** | **100.0%** | 74.0% | 85.1% | **0.0%** | 97.0% | 0.987 |

## Quick Start

```bash
npm install
npm run build
```

1. Open `chrome://extensions` and enable **Developer mode**.
2. Click **Load unpacked** and select the generated `dist/` folder.
3. Open any X thread (`https://x.com/<user>/status/<id>`). On first run, `EmbeddingGemma-300M` downloads locally from Hugging Face and caches in browser storage.

## Development & Verification

```bash
npm run typecheck                  # TypeScript strict check
npm test                           # 52 unit, DOM, RPC, and MV3 lifecycle tests
npm run build                      # Compiles TS + bundles dist/ via esbuild
npx tsx benchmarks/verify-prod-parity.ts   # Production engine parity benchmark
```

See [docs/architecture.md](docs/architecture.md) and [benchmarks/benchmark-report.md](benchmarks/benchmark-report.md) for architecture and empirical details.
