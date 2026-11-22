# X Reply Filter v2 Architecture

## Problem

X thread pages stream replies through a virtualized DOM scroller where removing nodes or altering outer cell heights causes scroll jumps, while modern reply spam spans English (`en`), Simplified Chinese (`zh-Hans`), Traditional Chinese (`zh-Hant`), and Japanese (`ja`). Chrome Manifest V3 forbids persistent WebGPU execution inside the transient service worker, content scripts cannot share one warm `EmbeddingGemma-300M` instance across tabs, and v1 heuristics suffered from four structural accuracy gaps:

1. **Unweighted relational head & sparse prototypes**: The v1 778-dim head had zeroed embedding weights, only 9 seed prototypes, and skipped `prototypeMargin` when either cosine was negative.
2. **DOM extraction blind spots**: `textContent` dropped Twemoji `<img alt="🔥">` emojis (collapsing verified emoji-only spam to 0 tokens), quote-tweet nodes polluted media/badge checks, zero-text media/GIF replies from unfollowed blue checks escaped, and thread context ancestors above the focal tweet were scored as replies.
3. **Rule ordering & lack of trust shields**: Running `thread_clone` before `short_human_exempt` falsely collapsed short human reactions (`"lol"`, `"草"`, `"哈哈"`, `"wtf"`), OP follow-ups and followed accounts had no trust shields, and uncalibrated probabilities showed false confidence on clean replies.
4. **Topic drift in online updates**: Passive-aggressive updates wrote directly into raw 256-dim embedding dimensions without regularization toward shipped weights or a regression guard.

## Pipeline Overview

```
  [x.com MAIN world: graphql-tap.ts]
    passive fetch/XHR tap (TweetDetail, TweetResultByRestId)
    bounded 256-entry replay ring -> window.postMessage("xrf-graphql-tap")
          |
          v
  [x.com isolated world: content/index.ts + dom-adapter.ts]
    Twemoji-aware text, quote-isolated DOM signals, context-ancestor tagging,
    Discover-more cutoff, LRU 1000-entry GraphqlTweetMeta cache, tap_ready handshake
          |
          |  OffscreenRpcRequestV2 (evaluate_batch, record_feedback, settings, ...)
          v
  [background/index.ts]  <-- threadId -> tabId relay for stage2_verdict_update & settings
          |
          v
  [offscreen/index.ts + gpu-host.ts]
    Stage-1: EmbeddingGemma-300M (q8, 256d Matryoshka) + prefixTokenOffset
          |
          v
  [domain/engine.ts + stylometrics.ts + trained-head.ts]
    58d COMPACT_V2 L2-LR + Rule Table v2 (R0-R12, Caps C1-C5) -> dim | suspect | keep
    Replay-buffer online learner (512 cap) + 200-item ANCHOR_SET_V2 rollback gate
          |
          |  (optional opt-in async queue on non-empty "suspect" replies only)
          v
  [offscreen/stage2-judge.ts]
    Chrome Built-in AI (LanguageModel / Gemini Nano), concurrency=1, 1500ms timeout
    Asymmetric guard: human >= 0.85 -> keep; bot >= 0.92 -> dim (no R0/R1/C1-C4 bypass)
```

## Core Components

### 1. Passive MAIN-World GraphQL Tap & Twemoji-Aware DOM Adapter

- **`src/content/graphql-tap.ts`** (`dist/graphql-tap.js`, registered in `public/manifest.json` with `"run_at": "document_start"` and `"world": "MAIN"`):
  - Idempotently hooks `window.fetch` and `XMLHttpRequest.prototype.{open,send}` (tracking per-XHR state in a `WeakMap` with `{ once: true }` `"load"` listeners) solely for `/\/graphql\/[^/]+\/(TweetDetail|TweetResultByRestId)\b/`.
  - Clones `fetch` responses synchronously before X reads the body and walks JSON recursively (`maxDepth = 32`, `WeakSet` cycle guard), unwrapping `TweetWithVisibilityResults` and skipping `TweetUnavailable` / `TweetTombstone` / `promotedMetadata` without overwriting an outer tweet with nested `quoted_status_result`.
  - Extracts lean `GraphqlTweetMeta` (`tweetId`, `authorRestId`, `screenName`, `displayName`, `authorCreatedAtMs`, `tweetCreatedAtMs`, `description`, `isBlueVerified`, `verifiedType`, strict tri-state `following: boolean | null`, `followedBy: boolean | null`, counts, `defaultProfileImage`, `mediaTypes`) into a bounded `MAX_TAP_BUFFER = 256` replay buffer and posts `{ source: "xrf-graphql-tap", v: 1, items }` to `window.location.origin`.
  - Replays buffered metadata when isolated `content.js` (`"run_at": "document_idle"`) sends `{ source: "xrf-content", type: "tap_ready" }`.
- **`src/content/dom-adapter.ts` & `src/content/index.ts`**:
  - Validates incoming `postMessage` events (`event.source === window`, allowed origin, `parseGraphqlTweetMeta`) into a 1,000-entry LRU map and triggers one re-evaluation pass if GraphQL metadata arrives after an initial DOM-only pass.
  - Imports `extractDomSignalsFromText` from `src/domain/stylometrics.ts` (never `calibrated-data.ts`, keeping content bundle size minimal).
  - Extracts Twemoji `<img alt="...">` emojis and `<br>` line breaks via `extractTweetTextWithTwemoji`, scopes media/link/badge/action-group queries outside `[data-testid="quoteTweet"]` and `[data-testid="card.wrapper"]`, clears stale attributes on recycled `<article>` elements, tags tweets preceding the focal `threadId` as `isContextAncestor`, and halts scanning at `Discover more` cutoff headings.
  - Renders 3-tier UI states in place without outer cell height jitter on `clean`/`suspect`:
    - `dimmed` (`revealed` when toggled via `Show`): collapses the card to a 1-line `.xrf-control-bar` (`<= 36px`, never a direct child of `<article>`).
    - `suspect`: amber badge + thin `2px` left border keeping reply text and media 100% readable, with 1-click `Mark bot` and `Not spam` buttons.
    - `clean`: no persistent noise; reveals subtle hover-only `Mark bot` button and explainable diagnostics tooltip.

### 2. Stage-1 Feature Set (`COMPACT_V2`, 58d) & Calibrated L2-LR

- **Backbone (`src/offscreen/gpu-host.ts`)**: `onnx-community/embeddinggemma-300m-ONNX` (`q8` default; `q4` verified drop-in compatible on `q8` weights), truncated and L2-normalized to 256d (`UnitVec256`), subtracting `prefixTokenOffset` from tokenizer attention masks so CJK and short replies receive accurate `TokenCount`, with 10-minute idle unload (`OFFSCREEN_IDLE_TIMEOUT_MS = 600_000`) and `device.lost` recovery.
- **58-dim `COMPACT_V2` Vector (`src/domain/stylometrics.ts`)**:
  - **Embedding (`e_*`, 8d)**: OP cosine, anchor presence, top-3 slop/human cosine means and `e_proto_margin` over 36 disjoint EN/ZH/JA prototypes (`CALIBRATED_WEIGHTS_V2.prototypes`), `e_generic` cosine to a generic-praise centroid, `e_clone_max`, `e_clone_cluster`.
  - **OP Overlap & Specificity (`l_*`, 5d)**: lexical novelty, Jaccard, char-3gram containment, specificity hits (numbers/handles/code/dates), first-person experience markers.
  - **Multilingual Stylometrics (`s_*`, 19d)**: contrast templates (`"not just X, it's Y"` / `"不仅是…更是"` / `"不僅是…更是"` / `"だけではなく"`), sycophantic openers, bait closers, slop lexicon, Latin-scoped `s_em_dash` (explicitly excluding Chinese `——`), colon reveals, listicles, sentence-length uniformity, Twemoji density/emoji-only/trailing emoji runs, hashtags, exclamations, caps, punctuation, human casual/profanity/informal markers, and `log(tokens)`.
  - **Cross-Language & Bot Summons (`x_*`, `b_*`, 4d)**: `x_lang_mismatch` and `x_mismatch_generic` (treating `zh-Hans` ↔ `zh-Hant` and `cjk-ambiguous` as matching), plus `@grok` / summon-handle detection (`b_summon_pure`, `b_summon_any`) guarded against substantive discussions mentioning `"Grok"` without `@` or with `>= 12` remaining tokens.
  - **GraphQL Account, Author Memory & Context (`m_*`, `a_*`, `c_*`, 22d)**: verification type (`blue`, `business`, `government`, `none`), strict tri-state `m_unfollowed` (`following === false`), `m_blue_unfollowed_short`, `m_followed_by`, account age, followers, friend/follower ratio, tweet rate, default avatar, strong/weak bio spam, display-name spam, handle digits, relative reply likes, decayed author prior (`30`-day half-life, `0.15` auto-flag weight), and primary media/link/quote flags.
- **Additive Explainable Scoring (`src/domain/trained-head.ts`)**: `scoreCompactV2` standardizes `COMPACT_V2` features, applies language-specific biases (`en`, `zh-Hans`, `zh-Hant`, `ja`, `other`) plus frozen `fixedCoefs` (`protoMargin`, `generic`), and decomposes `logit - baseHumanLogit` into exact additive per-feature `SignalContribution` records (`top <= 5`, `counter <= 3`, `corroboratingFamilies` where `familyLogits[fam] >= 0.5`).
- **5-Fold CV Benchmark (`benchmarks/benchmark-report.md`)**: On the 200-case multilingual evaluation set (Balanced preset with `τ_dim = 0.65` safety floor), Stage-1 achieves `100.0%` Precision@dim, `93.0%` Recall@dim, `96.4%` F1, `0.0%` Human FPR, `99.0%` Suspect Capture, and `0.989` ROC-AUC (`100.0%` P@dim / `78.0%` R@dim under 100% DOM-only fallback).

### 3. Rule Table v2 (`R0–R12`, Caps `C1–C5`) & Zero-GPU Re-evaluation

`evaluateVerdictRulesV2` in `src/domain/engine.ts` evaluates rules in strict precedence:

1. **`R0 user_override`**: Explicit per-tweet user mark (`dim` or `keep`).
2. **`R1` Shields → `keep`**: `context_ancestor`, `op_author`, `followed` (`following === true`), `trusted_author` (`humanCount >= 1 && humanCount >= spamCount`), or `self`.
3. **`R2 known_bot_author` → `dim`**: Triggered only by explicit user `spamCount` marks (never auto-flags).
4. **`R3 keyword_spam` / `bio_scam` → `dim`**: Hard rules (never capped); bare `t.me/` links require a bait verb.
5. **`R4 bot_answer` / `bot_summon`**: Controlled by `collapseBotSummons` (`suspect` on `conservative`, `dim` otherwise).
6. **`R5 blue_short_praise`**: `<= 12` tokens, zero specifics, non-casual, non-`followed_by`, with praise/slop/emoji/generic signal (`dim` when confirmed unfollowed blue + `collapseBlueCheckReplyGuys` + not `conservative`, else `suspect`).
7. **`R6 media_only_blue`**: Zero-text primary media by confirmed unfollowed blue check (`dim` when `collapseMediaOnlyReplies` + not `conservative`, else `suspect`); unverified media replies fall through.
8. **`R7 short_human_exempt` → `keep`**: Runs **before** clone detection so short human banter (`"lol"`, `"草"`, `"哈哈"`, `"wtf"`) is never clone-dimmed.
9. **`R8 thread_clone` → `dim`**: Requires `>= 8` tokens (or `>= 5` tokens in clusters of `>= 3`), retroactively promoting an unshielded prior occurrence.
10. **`R9` / `R9b op_paraphrase`**: Requires `opTokenCount >= 12`, `contentTokenCount >= 8`, `opCosSim < 0.985`, `specificityHits === 0`, and novelty guards (`novelty <= 0.5` with `>= 1` corroborating family or `0.5 < novelty <= 0.85` with `>= 2` families → `dim`; softer matches up to `novelty <= 0.90` → `suspect`).
11. **`R10 siamese_bot` → `dim`**: `probability >= thr.dim` with `>= thr.nf` corroborating families.
12. **`R11 model_suspect` → `suspect`**: `probability >= thr.suspect`.
13. **`R12 clean` → `keep`**.

**Caps (`C1–C5`)**: `C1_org_verified`, `C2_high_likes` (`>= 100` likes or high relative likes), `C3_followed_by`, and `C4_unsupported_lang` (`replyLang === "other"` on `siamese_bot`) cap `dim` verdicts down to `suspect` (never overriding `R0–R3`). `C5_hide_suspect` hides `suspect` badges to `keep` when `showSuspectBadges` is disabled.

**Caching**: `ThreadFilterEngine` caches per-thread embedding slabs (`MAX_CACHED_THREADS = 16`) plus a global 2,000-entry `featureCacheByTweetId` keyed by `(tweetId, settingsRevision, weightsRevision)` so changing sensitivity presets (`conservative` / `balanced` / `strict`), toggling any of the 6 policies, or applying online feedback/undo re-evaluates cached threads in `O(1)` without touching WebGPU.

### 4. Regularized Replay-Buffer Learner & Lossless IndexedDB v2

- **Replay Learner (`src/domain/trained-head.ts`)**: Stores up to `MAX_REPLAY_BUFFER_ENTRIES = 512` user-labeled `(z, fixed, y, lang, topicHash, tsMs)` items in `feedback_replay_v2`, enforcing an 8% cap per 8-bit `topicHash` bucket and 60-day age decay. Each feedback action runs 16 class-balanced AdaGrad steps with L2 anchoring toward `theta0` (`lambda = max(0.05, 50 / (50 + N))`), `FEATURE_SIGNS` projection, `||theta - theta0||_inf <= 1.5` drift clipping, and evaluation against the 200-item `ANCHOR_SET_V2`: if precision drops `> 2` pts or human FPR rises `> 2` pts, weights roll back and `rollbackCount` increments.
- **Auto-Flag Hygiene**: Only hard evidence (`keyword_spam`, `bio_scam`, `thread_clone`) increments `autoFlagCount`, deduplicated by `tweetId` (capped at `MAX_AUTO_FLAGGED_TWEET_IDS = 5000`).
- **Portability & Undo**: Keeps a 32-entry `UndoRecord` stack (`undo_feedback`) and supports JSON backup (`export_feedback` / `import_feedback`) plus `reset_feedback`.
- **Lossless v1→v2 Storage (`src/offscreen/gpu-host.ts`)**: Upgrades IndexedDB to v2 (`classifier_weights`, `feedback_replay_v2`) while preserving `primary_v1` and dual-writing `authorStats` to both `primary_v1` and `primary_v2`.

### 5. Optional Zero-VRAM Stage-2 Cascade (`LanguageModel` / Gemini Nano)

- Disabled by default (`enableStage2: false`). Local 0.3B–1B ONNX Stage-2 models were empirically rejected (`benchmarks/benchmark-report.md` §5) after degrading Precision@dim (`100% → 92.2–98.9%`) and adding 323–955 MB.
- When opted in on Chrome 138+ desktop, `src/offscreen/stage2-judge.ts` asynchronously inspects non-empty `suspect` replies at concurrency 1 (`1500ms` timeout, `session.clone()`, `responseConstraint` JSON schema, `child.destroy()` in `finally`).
- **Asymmetric Safety Policy**: Never overrides `R0` overrides, `R1` shields, or `C1–C4` caps. Only `human >= 0.85` clears `suspect → keep`, and only `bot >= 0.92` with `>= 1` corroborating family and `p >= thr.suspect` promotes `suspect → dim`. Updates push asynchronously via `stage2_verdict_update` through `background` to matching X tabs.
- Popup exposes a `"Prepare Gemini Nano"` button whenever `enableStage2` is enabled or `LanguageModel.availability()` is `"downloadable"`, running `LanguageModel.create()` inside the user click gesture with download progress feedback.

### 6. Build Artifacts & Interactive Diagnostics Popup

- **`scripts/build.mjs`** produces 5 bundled scripts plus runtime assets into `dist/`:
  - `dist/graphql-tap.js` (IIFE, `MAIN` world at `document_start`)
  - `dist/content.js` (IIFE, isolated world at `document_idle`)
  - `dist/background.js` (ESM service worker)
  - `dist/offscreen.js` (ESM offscreen worker)
  - `dist/popup.js` (ESM diagnostics & controls UI)
  - `manifest.json`, `offscreen.html`, `popup.html`, `icons/*`, and ONNX Runtime WASM JSEP assets.
- **`src/popup/index.ts` & `public/popup.html`**: Displays Stage-1 GPU & Stage-2 status pills, a 5-stat grid (`Scored`, `Dimmed`, `Suspect`, `Kept`, `Precision`), `Conservative` / `Balanced` / `Strict` segmented preset buttons, 6 policy toggles + `"Prepare Gemini Nano"`, feedback/learner telemetry (`Feedback updates`, `Replay buffer size`, `Anchor guard rollbacks`) with `Undo Last` / `Export JSON` / `Import JSON` / confirmed `Reset Learned`, per-language (`en`, `zh-Hans`, `zh-Hant`, `ja`, `other`) and signal/reason breakdowns, account reputation table, and color-coded recent verdicts showing `↑`/`↓` signal contributions, `GQL`/`DOM` source, and Stage-2 badges.
- **`scripts/verify-live-dom.js`**: Automated live DOM invariant check verifying zero post-settle `data-xrf-*` / `.xrf-control-bar` mutations, zero `Discover more` cutoff leaks (`scoredBelowCutoff === 0`), zero single-line bar overflows (`rect.height <= 36`, `!isDirectChildOfArticle`) across `dimmed`, `suspect`, `clean`, and `revealed` states, plus tier and GraphQL enrichment counters.
