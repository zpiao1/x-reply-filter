# Architecture rationale

## Problem

X thread pages stream replies through a virtualized DOM scroller where removing nodes or changing outer cell heights causes scroll jumps, and where bot replies span English, Chinese, and Japanese. Chrome Manifest V3 forbids persistent WebGPU execution inside the transient service worker, while content scripts cannot share one warm `EmbeddingGemma-300M` model across tabs or block the main thread during 80ms scroll ticks. Single-tweet classification also misses context-dependent bots that paraphrase the original post or flood a thread with near-duplicate replies from distinct accounts, while character-length heuristics misclassify short CJK reactions unless token gates read subword counts from the tokenizer.

## Usage (caller's view)

```ts
// 1. Content script scroll batch and 1-click user feedback on x.com
const snapshot = domAdapter.extractThreadSnapshot(document, window.location.pathname);
if (snapshot && snapshot.candidates.length > 0) {
  const res = await sendRpc({
    kind: "evaluate_batch",
    threadId: snapshot.threadId,
    candidates: snapshot.candidates,
  });
  if (res.kind === "evaluate_batch_ok") {
    const updates = [...res.result.verdicts, ...res.result.retroactiveUpdates];
    domAdapter.applyVerdictsInPlace(document, updates, (tweetId, label) => {
      void sendRpc({ kind: "record_feedback", threadId: snapshot.threadId, tweetId, label });
    });
  }
}

// 2. Offscreen batch evaluation with idempotent remount filtering
const { cached, toEmbed } = engine.filterUnseenCandidates(threadId, candidates);
const embedded = toEmbed.length > 0 ? await gpuHost.embedCandidates(toEmbed) : [];
const weights = await weightStore.getWeights();
const fresh = engine.ingestEmbeddedBatch(threadId, embedded, weights);

// 3. Offscreen 1-click online weight update without re-embedding
const update = engine.applyUserFeedback(threadId, tweetId, "spam", await weightStore.getWeights());
if (update) {
  await weightStore.saveWeights(update.nextWeights);
}
```

## Shape

The design centers on a deep domain engine, `ThreadFilterEngine`, which owns an LRU cache of per-thread vector slabs, a declarative verdict rule table, and 3-in-1 scoring behind three domain methods (`filterUnseenCandidates`, `ingestEmbeddedBatch`, `applyUserFeedback`), per `principle-model-the-domain` and `principle-minimize-reader-load`.

- `UnitVec256` brands a 256-float Matryoshka slice that `truncateAndNormalize256` L2-normalizes at creation, per `principle-type-system-discipline`. Every downstream function computes cosine similarity via a plain dot product without re-checking vector length or norm, per `principle-boundary-discipline`.
- `TokenCount` is computed inside `WebGpuEmbeddingHost.embedCandidates` by summing the tokenizer attention mask and subtracting `prefixTokenOffset` for `"task: classification | text: "`. This gives Chinese and Japanese replies accurate content token counts for the `< 6` token human-reaction gate.
- `ThreadAnchorState` is a discriminated union with `"unanchored"`, `"media_only"`, and `"text_anchored"` variants, per `principle-model-the-domain` and `principle-encode-lessons-in-structure`. When an OP has `< 5` tokens, the engine transitions to `"media_only"`, which omits the `embedding` field. The compiler prevents OP paraphrase checks and zeroes the `[256..768)` relational blocks of `SiameseFeatureVec`.
- `ThreadSlabState` packs seen reply embeddings into one contiguous `Float32Array(maxCapacity * 256)` alongside `verdictsByTweetId` and `featuresByTweetId` maps, per `principle-foundational-thinking`. During each scroll batch, `findMaxCloneSimilarity` scans the contiguous slab for `V_replies @ V_replies.T` matches across distinct authors and emits `retroactiveUpdates` when a second clone arrives and flips the first occurrence in the thread to `"dim"`.
- `VERDICT_RULE_TABLE` encodes rule precedence as a declarative array (`clone_swarm`, `keyword_spam`, `short_human_exempt`, `op_paraphrase` with `exactQuoteCeiling`, `siamese_bot`, and `clean` fallback), per `principle-model-the-domain`.
- Caching `VerdictRecord` and `SiameseFeatureVec` by `TweetId` makes scroll passes idempotent, per `principle-make-operations-idempotent`. Virtual scroller remounts hit `filterUnseenCandidates` in O(1), and 1-click user feedback updates the 774-float passive-aggressive weight vector without re-running WebGPU.
- `OffscreenFilterService` is the sole writer for `ThreadFilterEngine`, `WebGpuEmbeddingHost`, and `IndexedDbWeightStore`, per `principle-separate-before-serializing-shared-state`. `content.ts` holds no model state and mutates only `data-xrf-state` attributes in place so tweet cell heights never change, per `principle-experience-first`.

## Synthesis decision

Candidate 1 became the base because `ThreadFilterEngine` hides slab management, pre-GPU remount deduplication, retroactive clone promotion, and 1-click passive-aggressive weight updates behind three methods without temporal decomposition or split state ownership. Five grafts from Candidate 2 were folded into the base:
1. Subtracting `prefixTokenOffset` from tokenizer `attention_mask` sums so the prompt prefix does not inflate CJK and short-reply token counts.
2. Declarative `VERDICT_RULE_TABLE` with `exactQuoteCeiling` (`0.985`) and branded `UnitCosine` and `Probability` scalars.
3. Bounded multi-thread LRU cache (`MAX_CACHED_THREADS = 16`) inside `ThreadFilterEngine`.
4. Batch-capable `truncateAndNormalize256` signature for direct unit testing of 768d-to-256d Matryoshka truncation.
5. `mountThreadObserver` lifecycle handle with dataset idempotency guards in `XThreadDomAdapter`.

## Tradeoffs accepted

- We accept holding up to 512 reply embeddings and 774-float Siamese feature vectors per cached thread in offscreen RAM in exchange for O(1) virtual-scroll remount lookups, instant 1-click passive-aggressive weight updates without re-embedding, and contiguous stride-256 clone detection.
- We accept serializing all batch evaluations through the single MV3 Offscreen Document in exchange for having one WebGPU context and zero cross-tab write races on `OnlineClassifierWeights`.
- We accept dimming flagged replies in place via CSS opacity and blur instead of collapsing their height to zero in exchange for eliminating X virtual scroller jumpiness.

## Alternatives considered

- **Stateless offscreen embedder with scoring and state in `content.ts`.** Rejected because `content.ts` would have to receive raw `Float32Array` embeddings over Chrome IPC on every scroll batch, duplicate thread state across tabs, and coordinate concurrent IndexedDB weight updates across multiple `x.com` tabs.
- **Separate pipeline functions for OP paraphrase, clone clustering, and Siamese classification called sequentially by the host.** Rejected as temporal decomposition that forces callers to thread intermediate buffers across stages and repeat token-count guards.

## Open questions and risks

- When WebGPU `device.lost` fires twice within one minute on an unstable driver, should `WebGpuEmbeddingHost` fall back to the WASM execution provider for the rest of the browser session?

## Next implementation step

Implement `src/domain/types.ts`, `src/domain/seeds.ts`, and `src/domain/engine.ts` with deterministic unit tests covering Matryoshka normalization, `ThreadAnchorState` transitions, retroactive clone detection, declarative rule precedence, and passive-aggressive online updates.
