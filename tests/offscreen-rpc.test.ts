import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  BackgroundMessageRouter,
  type BackgroundScriptingApi,
  type BackgroundTabsApi,
  forwardRpcToOffscreen,
  reinjectOpenXTabs,
} from "../src/background/index.ts";
import { CALIBRATED_WEIGHTS_V2, ThreadFilterEngine } from "../src/domain/engine.ts";
import {
  createInitialClassifierWeights,
  extractDomSignalsFromText,
  recordAuthorOutcome,
} from "../src/domain/seeds.ts";
import {
  type AuthorId,
  type CandidateTweet,
  DEFAULT_FILTER_SETTINGS,
  EMBEDDING_DIM,
  type FilterSettings,
  type GraphqlTweetMeta,
  SIAMESE_TOTAL_DIM,
  type ThreadId,
  type TweetId,
  type UnitVec256,
  asTokenCount,
  parseAuthorId,
  parseTweetId,
  truncateAndNormalize256,
} from "../src/domain/types.ts";
import {
  DB_VERSION,
  type EmbeddingBackend,
  IndexedDbWeightStore,
  OffscreenFilterService,
  REPLAY_STORE_NAME,
  SETTINGS_KEY_V2,
  SETTINGS_STORE_NAME,
  STORE_NAME,
  WEIGHTS_KEY_V1,
  WEIGHTS_KEY_V2,
  WebGpuEmbeddingHost,
} from "../src/offscreen/gpu-host.ts";
import {
  type LanguageModelCreateMonitor,
  type LanguageModelCreateOptions,
  type LanguageModelProvider,
  type LanguageModelSession,
  type RawLanguageModelAvailability,
  Stage2JudgeService,
} from "../src/offscreen/stage2-judge.ts";
import {
  type OffscreenRpcRequestV2,
  type Stage2VerdictUpdateMessage,
  ensureOffscreenDocument,
  parseRpcRequest,
  parseRpcRequestV2,
  parseStage2VerdictUpdate,
} from "../src/shared/rpc.ts";

function makeCandidate(
  id: string,
  author: string,
  text: string,
  options?: {
    readonly isOp?: boolean;
    readonly hasVerifiedBadge?: boolean;
    readonly hasMediaOrLink?: boolean;
    readonly hasPrimaryMedia?: boolean;
    readonly graphqlMeta?: GraphqlTweetMeta | null;
  },
): CandidateTweet {
  return {
    tweetId: parseTweetId(id)!,
    authorId: parseAuthorId(author)!,
    rawText: text,
    normalizedText: text.trim(),
    isOp: options?.isOp ?? false,
    domSignals: extractDomSignalsFromText(
      text,
      options?.hasVerifiedBadge ?? false,
      options?.hasMediaOrLink ?? false,
    ),
    ...(options?.hasPrimaryMedia !== undefined ? { hasPrimaryMedia: options.hasPrimaryMedia } : {}),
    ...(options?.graphqlMeta !== undefined ? { graphqlMeta: options.graphqlMeta } : {}),
  };
}

function makeGraphqlMeta(overrides?: Partial<GraphqlTweetMeta>): GraphqlTweetMeta {
  return {
    displayName: "User",
    authorCreatedAtMs: Date.now() - 400 * 86_400_000,
    tweetCreatedAtMs: Date.now() - 60_000,
    inReplyToStatusId: null,
    conversationId: null,
    description: null,
    isBlueVerified: false,
    verifiedType: "none",
    following: false,
    followedBy: false,
    followersCount: 120,
    friendsCount: 150,
    statusesCount: 400,
    defaultProfileImage: false,
    favoriteCount: 1,
    replyCount: 0,
    retweetCount: 0,
    mediaTypes: [],
    ...overrides,
  };
}

function makeUnitBasis(dim: number): UnitVec256 {
  const raw = new Float32Array(EMBEDDING_DIM);
  raw[dim % EMBEDDING_DIM] = 1;
  return truncateAndNormalize256(raw, 0, EMBEDDING_DIM);
}

function mixUnitVecs(a: UnitVec256, b: UnitVec256, cosTarget: number): UnitVec256 {
  const sinPart = Math.sqrt(Math.max(0, 1 - cosTarget * cosTarget));
  const raw = new Float32Array(EMBEDDING_DIM);
  for (let i = 0; i < EMBEDDING_DIM; i++) {
    raw[i] = cosTarget * (a[i] ?? 0) + sinPart * (b[i] ?? 0);
  }
  return truncateAndNormalize256(raw, 0, EMBEDDING_DIM);
}

function createMockBackend(options?: {
  failFirstInfer?: boolean;
  embeddingByText?: ReadonlyMap<string, UnitVec256>;
}): {
  readonly backendFactory: () => Promise<EmbeddingBackend>;
  readonly getInferCalls: () => number;
  readonly getDisposeCalls: () => number;
  readonly getChunkSizes: () => ReadonlyArray<number>;
} {
  let inferCalls = 0;
  let disposeCalls = 0;
  let shouldFailOnce = options?.failFirstInfer ?? false;
  const chunkSizes: number[] = [];

  const backendFactory = async (): Promise<EmbeddingBackend> => ({
    async inferPrompts(prompts, rawTexts) {
      inferCalls += 1;
      chunkSizes.push(prompts.length);
      if (shouldFailOnce) {
        shouldFailOnce = false;
        throw new Error("Simulated WebGPU device.lost");
      }
      const embeddings256 = rawTexts.map((rawText, idx) => {
        const custom = options?.embeddingByText?.get(rawText);
        if (custom !== undefined) {
          return custom;
        }
        return makeUnitBasis(idx + inferCalls * 17);
      });
      const contentTokenCounts = rawTexts.map((t) => {
        const trimmed = t.trim();
        if (trimmed.length === 0) return asTokenCount(0);
        const words = trimmed.split(/\s+/).filter(Boolean).length;
        return asTokenCount(Math.max(words, Math.ceil(Array.from(trimmed).length / 4)));
      });
      return { embeddings256, contentTokenCounts };
    },
    async dispose() {
      disposeCalls += 1;
    },
  });

  return {
    backendFactory,
    getInferCalls: () => inferCalls,
    getDisposeCalls: () => disposeCalls,
    getChunkSizes: () => chunkSizes,
  };
}

/**
 * Minimal deterministic in-memory IDBFactory supporting versioned open,
 * onupgradeneeded, multi-store readonly/readwrite transactions, get, put,
 * getAll, clear, and atomic tx.oncomplete.
 */
interface MockIdbDatabaseState {
  version: number;
  readonly stores: Map<string, Map<string, unknown>>;
}

function createMockIdbFactory(): {
  readonly factory: IDBFactory;
  readonly databases: Map<string, MockIdbDatabaseState>;
} {
  const databases = new Map<string, MockIdbDatabaseState>();

  const makeStoreNamesList = (stores: Map<string, Map<string, unknown>>) => ({
    contains: (name: string) => stores.has(name),
    get length() {
      return stores.size;
    },
  });

  const makeRequest = <T>(exec: () => T): IDBRequest<T> => {
    const req: Record<string, unknown> = {
      result: undefined,
      error: null,
      onsuccess: null,
      onerror: null,
    };
    queueMicrotask(() => {
      try {
        req.result = exec();
        if (typeof req.onsuccess === "function") {
          (req.onsuccess as (ev: unknown) => void)({ target: req });
        }
      } catch (err) {
        req.error = err;
        if (typeof req.onerror === "function") {
          (req.onerror as (ev: unknown) => void)({ target: req });
        }
      }
    });
    return req as unknown as IDBRequest<T>;
  };

  const factory = {
    open(name: string, version?: number): IDBOpenDBRequest {
      const req: Record<string, unknown> = {
        result: undefined,
        error: null,
        onupgradeneeded: null,
        onsuccess: null,
        onerror: null,
      };

      queueMicrotask(() => {
        let dbState = databases.get(name);
        const isNew = dbState === undefined;
        if (!dbState) {
          dbState = { version: 0, stores: new Map() };
          databases.set(name, dbState);
        }
        const targetVersion = version ?? (isNew ? 1 : dbState.version);

        const dbHandle = {
          name,
          get version() {
            return dbState!.version;
          },
          objectStoreNames: makeStoreNamesList(dbState.stores),
          createObjectStore(storeName: string) {
            if (!dbState!.stores.has(storeName)) {
              dbState!.stores.set(storeName, new Map());
            }
            return {};
          },
          transaction(storeNames: string | string[], _mode?: IDBTransactionMode) {
            const names = Array.isArray(storeNames) ? storeNames : [storeNames];
            for (const s of names) {
              if (!dbState!.stores.has(s)) {
                throw new Error(`Object store not found: ${s}`);
              }
            }
            const tx: Record<string, unknown> = {
              error: null,
              oncomplete: null,
              onerror: null,
              onabort: null,
              objectStore(storeName: string) {
                const map = dbState!.stores.get(storeName);
                if (!map) throw new Error(`Missing store ${storeName}`);
                return {
                  get(key: IDBValidKey) {
                    return makeRequest(() => map.get(String(key)));
                  },
                  put(value: unknown, key?: IDBValidKey) {
                    const k = String(key ?? "");
                    return makeRequest(() => {
                      map.set(k, value);
                      return k;
                    });
                  },
                  getAll() {
                    return makeRequest(() => Array.from(map.values()));
                  },
                  clear() {
                    return makeRequest(() => {
                      map.clear();
                      return undefined;
                    });
                  },
                };
              },
            };
            // Fire tx.oncomplete after queued store operation microtasks settle
            queueMicrotask(() => {
              queueMicrotask(() => {
                if (typeof tx.oncomplete === "function") {
                  (tx.oncomplete as (ev: unknown) => void)({ target: tx });
                }
              });
            });
            return tx as unknown as IDBTransaction;
          },
          close() {},
        };

        req.result = dbHandle;
        if (targetVersion > dbState.version) {
          dbState.version = targetVersion;
          if (typeof req.onupgradeneeded === "function") {
            (req.onupgradeneeded as (ev: unknown) => void)({ target: req });
          }
        }
        if (typeof req.onsuccess === "function") {
          (req.onsuccess as (ev: unknown) => void)({ target: req });
        }
      });

      return req as unknown as IDBOpenDBRequest;
    },
  };

  return {
    factory: factory as unknown as IDBFactory,
    databases,
  };
}

describe("RPC v2 boundary parsing and single-flight offscreen creation", () => {
  it("validates all 11 v2 RPC request kinds and Stage-2 update messages", () => {
    const validHealth = parseRpcRequest({ kind: "get_health", target: "offscreen" });
    assert.deepEqual(validHealth, { kind: "get_health", target: "offscreen" });

    const validFeedback = parseRpcRequestV2({
      kind: "record_feedback",
      threadId: "/user/status/998877",
      tweetId: "998878",
      label: "spam",
    });
    assert.equal(validFeedback?.kind, "record_feedback");
    if (validFeedback?.kind === "record_feedback") {
      assert.equal(validFeedback.threadId, "998877");
      assert.equal(validFeedback.tweetId, "998878");
    }

    assert.equal(
      parseRpcRequestV2({
        kind: "record_feedback",
        threadId: "bad-thread",
        tweetId: "998878",
        label: "spam",
      }),
      null,
    );

    assert.deepEqual(parseRpcRequestV2({ kind: "get_settings" }), { kind: "get_settings" });
    assert.deepEqual(parseRpcRequestV2({ kind: "undo_feedback" }), { kind: "undo_feedback" });
    assert.deepEqual(parseRpcRequestV2({ kind: "export_feedback" }), { kind: "export_feedback" });
    assert.deepEqual(parseRpcRequestV2({ kind: "reset_feedback" }), { kind: "reset_feedback" });
    assert.deepEqual(
      parseRpcRequestV2({ kind: "trigger_stage2_download", action: "download" }),
      { kind: "trigger_stage2_download", action: "download" },
    );

    const parsedUpdate = parseRpcRequestV2({
      kind: "update_settings",
      settings: {
        ...DEFAULT_FILTER_SETTINGS,
        preset: "strict",
        revision: 2,
      },
    });
    assert.equal(parsedUpdate?.kind, "update_settings");
    assert.equal(
      parseRpcRequestV2({ kind: "update_settings", settings: { preset: "bogus" } }),
      null,
    );

    const parsedImport = parseRpcRequestV2({
      kind: "import_feedback",
      payloadJson: '{"schemaVersion":2,"replayBuffer":[]}',
    });
    assert.equal(parsedImport?.kind, "import_feedback");
    assert.equal(parseRpcRequestV2({ kind: "import_feedback", payloadJson: "" }), null);

    const validStage2 = parseStage2VerdictUpdate({
      kind: "stage2_verdict_update",
      threadId: "12345",
      tweetId: "67890",
      target: "content",
      verdict: {
        tweetId: "67890",
        action: "keep",
        score: 0.45,
        tokenCount: 10,
        reason: { kind: "clean", action: "keep" },
      },
    });
    assert.notEqual(validStage2, null);
    assert.equal(validStage2?.tweetId, "67890");

    // Reject mismatched verdict.tweetId
    assert.equal(
      parseStage2VerdictUpdate({
        kind: "stage2_verdict_update",
        threadId: "12345",
        tweetId: "67890",
        verdict: {
          tweetId: "99999",
          action: "keep",
          score: 0.45,
          tokenCount: 10,
          reason: { kind: "clean", action: "keep" },
        },
      }),
      null,
    );
  });

  it("coalesces concurrent ensureOffscreenDocument calls into one createDocument call", async () => {
    let hasDoc = false;
    let createCount = 0;

    const mockApi = {
      async hasDocument() {
        return hasDoc;
      },
      async createDocument() {
        createCount += 1;
        await new Promise((r) => setTimeout(r, 10));
        hasDoc = true;
      },
    };

    await Promise.all([
      ensureOffscreenDocument("offscreen.html", mockApi),
      ensureOffscreenDocument("offscreen.html", mockApi),
      ensureOffscreenDocument("offscreen.html", mockApi),
    ]);

    assert.equal(createCount, 1);
  });
});

describe("WebGpuEmbeddingHost lifecycle and chunk yielding", () => {
  it("recovers from WebGPU device loss and unloads after 10 idle minutes", async () => {
    let nowMs = 1_000;
    const mock = createMockBackend({ failFirstInfer: true });
    const host = new WebGpuEmbeddingHost(mock.backendFactory, () => nowMs, 600_000);

    assert.equal(host.getState().status, "unloaded");

    const candidates = [makeCandidate("101", "alice", "Testing recovery after GPU reset")];
    const embedded = await host.embedCandidates(candidates);

    assert.equal(embedded.length, 1);
    assert.equal(host.getState().status, "ready");
    assert.equal(host.getState().deviceEpoch, 2, "incremented epoch after device.lost recovery");
    assert.equal(mock.getDisposeCalls(), 1);

    nowMs += 300_000;
    assert.equal(await host.unloadIfIdle(nowMs), false);

    nowMs += 300_001;
    assert.equal(await host.unloadIfIdle(nowMs), true);
    assert.equal(host.getState().status, "unloaded");
    assert.equal(mock.getDisposeCalls(), 2);
  });

  it("chunks large scroll batches and yields microtasks between chunks", async () => {
    let yieldCalls = 0;
    const mock = createMockBackend();
    const host = new WebGpuEmbeddingHost(
      mock.backendFactory,
      () => Date.now(),
      600_000,
      4,
      async () => {
        yieldCalls += 1;
        await Promise.resolve();
      },
    );

    const texts = Array.from({ length: 10 }, (_, i) => `Reply text number ${i}`);
    const out = await host.embedTexts(texts);
    assert.equal(out.embeddings256.length, 10);
    assert.equal(out.contentTokenCounts.length, 10);
    assert.deepEqual(mock.getChunkSizes(), [4, 4, 2]);
    assert.equal(yieldCalls, 2, "yields between chunk 1->2 and chunk 2->3");
  });
});

describe("OffscreenFilterService v2 RPCs, pre-embedded seeds, and auto-flag hygiene", () => {
  it("skips GPU seed embedding on cold start and handles all v2 settings/feedback/undo/export/import/reset RPCs", async () => {
    const mock = createMockBackend();
    const engine = new ThreadFilterEngine();
    const gpuHost = new WebGpuEmbeddingHost(mock.backendFactory);
    const weightStore = new IndexedDbWeightStore("test-v2-rpcs", null);
    const service = new OffscreenFilterService(engine, gpuHost, weightStore);

    let hasDoc = true;
    const mockOffscreenApi = {
      async hasDocument() {
        return hasDoc;
      },
      async createDocument() {
        hasDoc = true;
      },
    };
    const send = (req: OffscreenRpcRequestV2) =>
      forwardRpcToOffscreen(req, mockOffscreenApi, (r) => service.handleRequest(r));

    const op = makeCandidate("7000", "op_user", "Full end to end RPC test on thread", {
      isOp: true,
    });
    const reply = makeCandidate("7001", "replier", "Look at my pinned profile link in bio!");

    const batchRes = await send({
      kind: "evaluate_batch",
      threadId: "7000" as ThreadId,
      candidates: [op, reply],
    });
    assert(batchRes !== null && batchRes.kind === "evaluate_batch_ok");
    assert.equal(batchRes.result.verdicts.length, 1);
    assert.equal(batchRes.result.verdicts[0]?.action, "dim");
    // Pre-embedded calibrated prototypes mean ZERO seed GPU pass: only 1 inferPrompts call!
    assert.equal(mock.getInferCalls(), 1, "calibrated 36 prototypes avoid cold-start GPU pass");

    // Remount hits cache without GPU
    const remountRes = await send({
      kind: "evaluate_batch",
      threadId: "7000" as ThreadId,
      candidates: [op, reply],
    });
    assert(remountRes !== null && remountRes.kind === "evaluate_batch_ok");
    assert.equal(mock.getInferCalls(), 1);

    // get_settings & update_settings (zero-GPU reevaluation)
    const getSetRes = await send({ kind: "get_settings" });
    assert(getSetRes !== null && getSetRes.kind === "settings_ok");
    assert.equal(getSetRes.settings.preset, "balanced");

    const updatedSettings: FilterSettings = {
      ...getSetRes.settings,
      preset: "conservative",
      toggles: { ...getSetRes.settings.toggles, showSuspectBadges: false },
      revision: getSetRes.settings.revision + 1,
    };
    const updSetRes = await send({ kind: "update_settings", settings: updatedSettings });
    assert(updSetRes !== null && updSetRes.kind === "settings_ok");
    assert.equal(updSetRes.settings.preset, "conservative");
    assert.equal(updSetRes.reevaluatedThreads, 1);
    assert.equal(mock.getInferCalls(), 1, "settings update re-evaluates with zero GPU calls");

    // record_feedback -> undo_feedback
    const fbRes = await send({
      kind: "record_feedback",
      threadId: "7000" as ThreadId,
      tweetId: "7001" as TweetId,
      label: "human",
    });
    assert(fbRes !== null && fbRes.kind === "record_feedback_ok");
    assert.equal(fbRes.updatedVerdict?.action, "keep");
    assert.equal(fbRes.updateCount, 1);

    const undoRes = await send({ kind: "undo_feedback" });
    assert(undoRes !== null && undoRes.kind === "undo_feedback_ok");
    assert.equal(undoRes.undone, true);
    assert.equal(undoRes.undoneTweetId, "7001");
    assert.equal(undoRes.restoredVerdict?.action, "dim");
    assert.equal(undoRes.updateCount, 0);

    // Second undo on empty stack returns undone: false
    const undoEmptyRes = await send({ kind: "undo_feedback" });
    assert(undoEmptyRes !== null && undoEmptyRes.kind === "undo_feedback_ok");
    assert.equal(undoEmptyRes.undone, false);

    // Record feedback again, export JSON, reset, then import back
    await send({
      kind: "record_feedback",
      threadId: "7000" as ThreadId,
      tweetId: "7001" as TweetId,
      label: "human",
    });

    const exportRes = await send({ kind: "export_feedback" });
    assert(exportRes !== null && exportRes.kind === "export_feedback_ok");
    assert(exportRes.payloadJson.includes('"schemaVersion": 2'));

    const resetRes = await send({ kind: "reset_feedback" });
    assert(resetRes !== null && resetRes.kind === "reset_feedback_ok");
    assert.equal(resetRes.updateCount, 0);

    const diagAfterReset = await send({ kind: "get_diagnostics" });
    assert(diagAfterReset !== null && diagAfterReset.kind === "diagnostics_ok");
    assert.equal(diagAfterReset.summary.replayBufferSize, 0);
    assert.equal(diagAfterReset.summary.topFlaggedAuthors.length, 0);
    assert.equal(diagAfterReset.stage2Status, "disabled");

    // Malformed import fails safely
    const badImport = await send({ kind: "import_feedback", payloadJson: "{not valid json" });
    assert(badImport !== null && badImport.kind === "rpc_error");

    // Valid import restores replay buffer & author memory
    const importRes = await send({ kind: "import_feedback", payloadJson: exportRes.payloadJson });
    assert(importRes !== null && importRes.kind === "import_feedback_ok");
    assert.equal(importRes.importedReplayCount, 1);
    assert.equal(importRes.importedAuthorCount, 1);

    const healthRes = await send({ kind: "get_health" });
    assert(healthRes !== null && healthRes.kind === "health_ok");
    assert.equal(healthRes.stage2Status, "disabled");
  });

  it("records auto_flag ONLY for hard evidence (keyword_spam, bio_scam, thread_clone + retroactive), deduplicated per tweetId, never paraphrase/model", async () => {
    const opVec = makeUnitBasis(1);
    const orthoVec = makeUnitBasis(2);
    const paraphraseVec = mixUnitVecs(opVec, orthoVec, 0.9);
    const humanProto = CALIBRATED_WEIGHTS_V2.prototypes.find((p) => p.label === -1)!;
    const cloneVec = truncateAndNormalize256(
      new Float32Array(humanProto.embedding),
      0,
      EMBEDDING_DIM,
    );
    const slopProtoVec = truncateAndNormalize256(
      new Float32Array(CALIBRATED_WEIGHTS_V2.prototypes[0]!.embedding),
      0,
      EMBEDDING_DIM,
    );

    const opText =
      "Distributed consensus protocols require bounded network delays and careful leader election timeouts across regions.";
    const kwText = "Great thread! Check my pinned profile link in bio right now!";
    const bioScamText = "Spot on breakdown of consensus design, appreciate this post!";
    const cloneText1 =
      "Distributed consensus systems benefit immensely when engineers tune quorum timeouts carefully across regions.";
    const cloneText2 =
      "Distributed consensus architectures work reliably when engineers configure quorum timeouts carefully across zones.";
    const paraphraseText =
      "Spot on! Distributed consensus protocols need bounded network delays and careful leader election timeouts across regions.";
    const modelSlopText =
      "Spot on analysis — this isn't just about consensus, it's a masterclass in systems thinking! Truly a game changer that reshapes the entire landscape. What do you think?";

    const embMap = new Map<string, UnitVec256>([
      [opText, opVec],
      [kwText, makeUnitBasis(20)],
      [bioScamText, makeUnitBasis(21)],
      [cloneText1, cloneVec],
      [cloneText2, cloneVec],
      [paraphraseText, paraphraseVec],
      [modelSlopText, slopProtoVec],
    ]);

    const mock = createMockBackend({ embeddingByText: embMap });
    const gpuHost = new WebGpuEmbeddingHost(mock.backendFactory);
    const weightStore = new IndexedDbWeightStore("test-autoflag", null);
    const engine1 = new ThreadFilterEngine();
    const service1 = new OffscreenFilterService(engine1, gpuHost, weightStore);

    const candidates: CandidateTweet[] = [
      makeCandidate("9000", "op_author", opText, { isOp: true }),
      makeCandidate("9001", "kw_bot", kwText),
      makeCandidate("9002", "bio_bot", bioScamText, {
        graphqlMeta: makeGraphqlMeta({
          description: "DM for promo 100x crypto signal group",
          following: false,
        }),
      }),
      makeCandidate("9003", "clone_bot_1", cloneText1),
      makeCandidate("9004", "clone_bot_2", cloneText2),
      makeCandidate("9005", "para_bot", paraphraseText),
      makeCandidate("9006", "slop_bot", modelSlopText, {
        hasVerifiedBadge: true,
        graphqlMeta: makeGraphqlMeta({
          isBlueVerified: true,
          verifiedType: "blue",
          following: false,
        }),
      }),
    ];

    const res = await service1.handleRequest({
      kind: "evaluate_batch",
      threadId: "9000" as ThreadId,
      candidates,
    });
    assert.equal(res.kind, "evaluate_batch_ok");
    if (res.kind !== "evaluate_batch_ok") return;

    const byId = new Map(res.result.verdicts.map((v) => [v.tweetId, v]));
    assert.equal(byId.get("9001" as TweetId)?.reason.kind, "keyword_spam");
    assert.equal(byId.get("9002" as TweetId)?.reason.kind, "bio_scam");
    assert.equal(byId.get("9004" as TweetId)?.reason.kind, "thread_clone");
    assert.equal(
      res.result.retroactiveUpdates.some(
        (u) => u.tweetId === "9003" && u.reason.kind === "thread_clone",
      ),
      true,
      "first clone peer retroactively promoted to thread_clone",
    );
    assert.equal(byId.get("9005" as TweetId)?.reason.kind, "op_paraphrase");
    assert.equal(byId.get("9006" as TweetId)?.reason.kind, "siamese_bot");

    const weightsAfterFirst = await weightStore.getWeights();
    // Hard evidence -> autoFlagCount === 1 (including retroactive clone promotion!)
    assert.equal(weightsAfterFirst.authorStats.get("kw_bot" as AuthorId)?.autoFlagCount, 1);
    assert.equal(weightsAfterFirst.authorStats.get("bio_bot" as AuthorId)?.autoFlagCount, 1);
    assert.equal(weightsAfterFirst.authorStats.get("clone_bot_1" as AuthorId)?.autoFlagCount, 1);
    assert.equal(weightsAfterFirst.authorStats.get("clone_bot_2" as AuthorId)?.autoFlagCount, 1);
    // Soft/model evidence -> NEVER auto-flagged
    assert.equal(weightsAfterFirst.authorStats.get("para_bot" as AuthorId), undefined);
    assert.equal(weightsAfterFirst.authorStats.get("slop_bot" as AuthorId), undefined);

    // Re-evaluate the exact same tweetIds on a fresh engine (simulating reload/slab eviction):
    // recordedAutoFlagIds MUST deduplicate every tweetId so counts stay at 1!
    const engine2 = new ThreadFilterEngine();
    const service2 = new OffscreenFilterService(engine2, gpuHost, weightStore);
    const res2 = await service2.handleRequest({
      kind: "evaluate_batch",
      threadId: "9000" as ThreadId,
      candidates,
    });
    assert.equal(res2.kind, "evaluate_batch_ok");

    const weightsAfterRevisit = await weightStore.getWeights();
    assert.equal(weightsAfterRevisit.authorStats.get("kw_bot" as AuthorId)?.autoFlagCount, 1);
    assert.equal(weightsAfterRevisit.authorStats.get("bio_bot" as AuthorId)?.autoFlagCount, 1);
    assert.equal(weightsAfterRevisit.authorStats.get("clone_bot_1" as AuthorId)?.autoFlagCount, 1);
    assert.equal(weightsAfterRevisit.authorStats.get("clone_bot_2" as AuthorId)?.autoFlagCount, 1);
  });
});

describe("IndexedDbWeightStore lossless v1 -> v2 migration", () => {
  it("preserves primary_v1 without deleting it, dual-writes authorStats, persists replay & settings, and resets cleanly", async () => {
    const { factory, databases } = createMockIdbFactory();
    const dbName = "x-reply-filter-migration-test";

    // Seed a legacy v1 IndexedDB state
    const legacyWeights = new Float32Array(SIAMESE_TOTAL_DIM);
    legacyWeights[0] = 0.42;
    const v1StoreMap = new Map<string, unknown>();
    v1StoreMap.set(WEIGHTS_KEY_V1, {
      weights: legacyWeights,
      bias: -0.25,
      updateCount: 4,
      authorStatsEntries: [
        ["legacy_spammer", { spamCount: 3, humanCount: 0, autoFlagCount: 2 }],
        ["legacy_friend", { spamCount: 0, humanCount: 2, autoFlagCount: 0 }],
      ],
    });
    databases.set(dbName, {
      version: 1,
      stores: new Map([[STORE_NAME, v1StoreMap]]),
    });

    // 1. Open in v2 store: upgrades DB_VERSION to 2 and loads primary_v1 losslessly
    const storeA = new IndexedDbWeightStore(dbName, factory);
    const loaded = await storeA.getWeights();

    const dbState = databases.get(dbName)!;
    assert.equal(dbState.version, DB_VERSION);
    assert.equal(dbState.stores.has(STORE_NAME), true);
    assert.equal(dbState.stores.has(REPLAY_STORE_NAME), true);
    assert.equal(dbState.stores.has(SETTINGS_STORE_NAME), true);
    assert.equal(dbState.stores.get(STORE_NAME)!.has(WEIGHTS_KEY_V1), true, "primary_v1 never deleted");

    assert.equal(loaded.updateCount, 4);
    assert.equal(loaded.authorStats.get("legacy_spammer" as AuthorId)?.spamCount, 3);
    assert.equal(loaded.authorStats.get("legacy_friend" as AuthorId)?.humanCount, 2);

    // 2. Mutate v2 weights (authorStats, recordedAutoFlagIds, replayBuffer, theta) and save
    let mutated = recordAuthorOutcome(loaded, "new_bot" as AuthorId, "user_spam", {
      tweetId: "55501" as TweetId,
      nowMs: 1_700_000_000_000,
    });
    mutated = recordAuthorOutcome(mutated, "new_bot" as AuthorId, "auto_flag", {
      tweetId: "55502" as TweetId,
      nowMs: 1_700_000_000_000,
    });
    const customTheta = new Float32Array(mutated.theta!);
    customTheta[0] = (customTheta[0] ?? 0) + 0.125;
    mutated = {
      ...mutated,
      theta: customTheta,
      updateCount: 5,
      rollbackCount: 1,
      replayBuffer: [
        {
          tweetId: "55501" as TweetId,
          authorId: "new_bot" as AuthorId,
          z: Array.from({ length: 58 }, () => 0.1),
          fixed: 0.2,
          y: 1,
          lang: "en",
          tsMs: 1_700_000_000_000,
          topicHash: 7,
          source: "user_spam",
        },
      ],
    };

    await storeA.saveWeights(mutated);
    const customSettings: FilterSettings = {
      preset: "strict",
      toggles: { ...DEFAULT_FILTER_SETTINGS.toggles, enableStage2: true },
      revision: 3,
    };
    await storeA.saveSettings(customSettings);

    // Verify dual-write to BOTH primary_v1 and primary_v2 (legacy 778d weights untouched!)
    const rawV1After = dbState.stores.get(STORE_NAME)!.get(WEIGHTS_KEY_V1) as {
      weights: Float32Array;
      bias: number;
      authorStatsEntries: Array<[string, { spamCount: number }]>;
    };
    const rawV2After = dbState.stores.get(STORE_NAME)!.get(WEIGHTS_KEY_V2) as {
      schemaVersion: number;
      recordedAutoFlagIds: string[];
    };
    assert.ok(rawV1After, "primary_v1 still present after v2 save");
    assert.equal(rawV1After.weights[0], Math.fround(0.42), "legacy 778d weights left untouched in primary_v1");
    assert.equal(
      rawV1After.authorStatsEntries.some(([id, s]) => id === "new_bot" && s.spamCount === 1),
      true,
      "authorStats dual-written to primary_v1 for downgrade compatibility",
    );
    assert.equal(rawV2After.schemaVersion, 2);
    assert.deepEqual(rawV2After.recordedAutoFlagIds, ["55502"]);
    assert.equal(dbState.stores.get(REPLAY_STORE_NAME)!.size, 1);
    assert.ok(dbState.stores.get(SETTINGS_STORE_NAME)!.has(SETTINGS_KEY_V2));

    // 3. Reload in a brand-new IndexedDbWeightStore instance (simulating restart)
    const storeB = new IndexedDbWeightStore(dbName, factory);
    const reloadedWeights = await storeB.getWeights();
    const reloadedSettings = await storeB.getSettings();

    assert.equal(reloadedWeights.updateCount, 5);
    assert.equal(reloadedWeights.rollbackCount, 1);
    assert.equal(reloadedWeights.replayBuffer?.length, 1);
    assert.equal(reloadedWeights.recordedAutoFlagIds?.has("55502" as TweetId), true);
    assert.equal(reloadedWeights.authorStats.get("legacy_spammer" as AuthorId)?.spamCount, 3);
    assert.equal(reloadedWeights.authorStats.get("new_bot" as AuthorId)?.spamCount, 1);
    assert.equal(reloadedWeights.authorStats.get("new_bot" as AuthorId)?.autoFlagCount, 1);
    assert(Math.abs((reloadedWeights.theta?.[0] ?? 0) - customTheta[0]!) < 1e-6);
    assert.equal(reloadedSettings.preset, "strict");
    assert.equal(reloadedSettings.toggles.enableStage2, true);
    assert.equal(reloadedSettings.revision, 3);

    // 4. Reset persisted feedback cleanly (still never deleting primary_v1!)
    await storeB.resetPersistedFeedback();
    assert.equal(dbState.stores.get(STORE_NAME)!.has(WEIGHTS_KEY_V1), true, "primary_v1 survives reset");
    assert.equal(dbState.stores.get(REPLAY_STORE_NAME)!.size, 0);

    const storeC = new IndexedDbWeightStore(dbName, factory);
    const afterReset = await storeC.getWeights();
    assert.equal(afterReset.updateCount, 0);
    assert.equal(afterReset.rollbackCount, 0);
    assert.equal(afterReset.authorStats.size, 0);
    assert.equal(afterReset.replayBuffer?.length ?? 0, 0);
    assert.equal(afterReset.recordedAutoFlagIds?.size ?? 0, 0);
  });
});

describe("Stage2JudgeService and OffscreenFilterService Stage-2 async integration", () => {
  it("enforces concurrency=1, max 3 pending per thread, skips empty/unsupported locale, handles timeout, and applies asymmetric guards via push callback", async () => {
    let activePrompts = 0;
    let maxObservedConcurrency = 0;
    let childCreated = 0;
    let childDestroyed = 0;
    const promptedInputs: string[] = [];

    const mockProvider: LanguageModelProvider = {
      async availability() {
        return "available";
      },
      async create() {
        const base: LanguageModelSession = {
          async prompt() {
            throw new Error("Base session should never be prompted directly; must clone()");
          },
          async clone() {
            childCreated += 1;
            const child: LanguageModelSession = {
              async prompt(input, opts) {
                activePrompts += 1;
                maxObservedConcurrency = Math.max(maxObservedConcurrency, activePrompts);
                promptedInputs.push(input);
                try {
                  // Simulate timeout on specifically tagged reply
                  if (input.includes("TIMEOUT_CASE")) {
                    await new Promise<void>((resolve, reject) => {
                      const t = setTimeout(resolve, 250);
                      opts?.signal?.addEventListener(
                        "abort",
                        () => {
                          clearTimeout(t);
                          reject(new Error("aborted"));
                        },
                        { once: true },
                      );
                    });
                  } else {
                    await new Promise((r) => setTimeout(r, 8));
                  }

                  if (input.includes("HUMAN_CLEAR")) {
                    return JSON.stringify({
                      verdict: "human",
                      confidence: 0.91,
                      reason: "Genuine reaction",
                    });
                  }
                  if (input.includes("BOT_PROMOTE") || input.includes("ORG_CAPPED")) {
                    return JSON.stringify({
                      verdict: "bot",
                      confidence: 0.96,
                      reason: "Generic engagement bait",
                    });
                  }
                  return JSON.stringify({
                    verdict: "uncertain",
                    confidence: 0.5,
                    reason: "Ambiguous",
                  });
                } finally {
                  activePrompts -= 1;
                }
              },
              async clone() {
                return child;
              },
              destroy() {
                childDestroyed += 1;
              },
            };
            return child;
          },
          destroy() {},
        };
        return base;
      },
    };

    const stage2Updates: Stage2VerdictUpdateMessage[] = [];
    const stage2 = new Stage2JudgeService({
      provider: mockProvider,
      timeoutMs: 35,
      maxPendingPerThread: 3,
    });

    const slopProtoVec = truncateAndNormalize256(
      new Float32Array(CALIBRATED_WEIGHTS_V2.prototypes[0]!.embedding),
      0,
      EMBEDDING_DIM,
    );
    const opText = "Benchmarking browser WebGPU embedding latency on desktop GPUs.";
    const humanClearText = "Spot on HUMAN_CLEAR take!";
    const botPromoteText = "Spot on BOT_PROMOTE — truly a game changer! What do you think?";
    const timeoutText = "Spot on TIMEOUT_CASE take!";
    const fourthDroppedText = "Spot on FOURTH_ITEM take!";
    const otherLangText = "Привет мир отличный пост 🔥🔥🔥";
    const orgCappedText =
      "Spot on ORG_CAPPED — this isn't just a benchmark, it's a paradigm shift! Truly a game changer for the whole ecosystem. What do you think?";

    const embMap = new Map<string, UnitVec256>([
      [opText, makeUnitBasis(5)],
      [humanClearText, slopProtoVec],
      [botPromoteText, slopProtoVec],
      [timeoutText, slopProtoVec],
      [fourthDroppedText, slopProtoVec],
      [otherLangText, slopProtoVec],
      [orgCappedText, slopProtoVec],
    ]);

    const mockBackend = createMockBackend({ embeddingByText: embMap });
    const engine = new ThreadFilterEngine();
    const gpuHost = new WebGpuEmbeddingHost(mockBackend.backendFactory);
    const weightStore = new IndexedDbWeightStore("test-stage2", null);
    const service = new OffscreenFilterService(engine, gpuHost, weightStore, {
      stage2Judge: stage2,
      onStage2Update: (msg) => {
        stage2Updates.push(msg);
      },
    });

    // Enable Stage-2 via update_settings
    const enableRes = await service.handleRequest({
      kind: "update_settings",
      settings: {
        ...DEFAULT_FILTER_SETTINGS,
        toggles: { ...DEFAULT_FILTER_SETTINGS.toggles, enableStage2: true },
        revision: 2,
      },
    });
    assert.equal(enableRes.kind, "settings_ok");

    // Batch 1 (thread 8100): empty media-only suspect, unsupported locale suspect, + 4 EN suspects
    // Expected: empty & "other" skipped; max 3 EN suspects queued (4th dropped by maxPendingPerThread=3);
    // Stage-1 returns IMMEDIATELY before Stage-2 finishes!
    const batch1 = await service.handleRequest({
      kind: "evaluate_batch",
      threadId: "8100" as ThreadId,
      candidates: [
        makeCandidate("8100", "op1", opText, { isOp: true }),
        makeCandidate("8101", "empty_blue", "", {
          hasVerifiedBadge: true,
          hasPrimaryMedia: true,
          graphqlMeta: null,
        }),
        makeCandidate("8102", "other_blue", otherLangText, {
          hasVerifiedBadge: true,
          graphqlMeta: null,
        }),
        makeCandidate("8103", "u_human", humanClearText, {
          hasVerifiedBadge: true,
          graphqlMeta: null,
        }),
        makeCandidate("8104", "u_bot", botPromoteText, {
          hasVerifiedBadge: true,
          graphqlMeta: null,
        }),
        makeCandidate("8105", "u_timeout", timeoutText, {
          hasVerifiedBadge: true,
          graphqlMeta: null,
        }),
        makeCandidate("8106", "u_fourth", fourthDroppedText, {
          hasVerifiedBadge: true,
          graphqlMeta: null,
        }),
      ],
    });
    assert.equal(batch1.kind, "evaluate_batch_ok");
    // Stage-1 MUST return before Stage-2 pushes any update
    assert.equal(stage2Updates.length, 0, "Stage-1 never blocks on Stage-2");

    await service.waitForStage2Idle();

    assert.equal(maxObservedConcurrency, 1, "strict concurrency = 1");
    assert.equal(promptedInputs.length, 3, "empty + unsupported locale skipped, 4th capped by maxPendingPerThread=3");
    assert.equal(childCreated, 3);
    assert.equal(childDestroyed, 3, "child.destroy() called in finally even on timeout");

    const updByTweet = new Map(stage2Updates.map((m) => [m.tweetId, m]));
    // 8103 cleared suspect -> keep (human 0.91 >= 0.85)
    assert.equal(updByTweet.get("8103" as TweetId)?.verdict.action, "keep");
    assert.equal(
      updByTweet.get("8103" as TweetId)?.verdict.explanation?.stage2?.provider,
      "chrome_gemini_nano",
    );
    // 8104 promoted suspect -> dim (bot 0.96 >= 0.92, >=1 corroborating family, p >= tau_suspect)
    assert.equal(updByTweet.get("8104" as TweetId)?.verdict.action, "dim");
    // 8105 timed out -> Stage-1 suspect untouched, no Stage-2 update pushed
    assert.equal(updByTweet.has("8105" as TweetId), false);
    assert.equal(engine.getCachedContext("8105" as TweetId)?.verdict.action, "suspect");
    assert.equal(engine.getCachedContext("8105" as TweetId)?.verdict.explanation?.stage2, undefined);

    // Batch 2 (thread 8200): C1_org_verified capped suspect MUST NOT be promoted to dim even at bot=0.96!
    const batch2 = await service.handleRequest({
      kind: "evaluate_batch",
      threadId: "8200" as ThreadId,
      candidates: [
        makeCandidate("8200", "op2", opText, { isOp: true }),
        makeCandidate("8201", "org_acct", orgCappedText, {
          hasVerifiedBadge: true,
          graphqlMeta: makeGraphqlMeta({
            verifiedType: "business",
            isBlueVerified: false,
            following: false,
          }),
        }),
      ],
    });
    assert.equal(batch2.kind, "evaluate_batch_ok");
    if (batch2.kind === "evaluate_batch_ok") {
      assert.equal(batch2.result.verdicts[0]?.action, "suspect");
      assert.equal(
        batch2.result.verdicts[0]?.explanation?.caps.includes("C1_org_verified"),
        true,
      );
    }

    await service.waitForStage2Idle();
    const orgVerdict = engine.getCachedContext("8201" as TweetId)?.verdict;
    assert.equal(orgVerdict?.action, "suspect", "Stage-2 never bypasses C1_org_verified cap");
    assert.equal(orgVerdict?.explanation?.stage2?.verdict, "bot");
  });

  it("never downloads in background when downloadable, supports explicit trigger_stage2_download, and destroys idle session", async () => {
    let nowMs = 10_000;
    let availabilityState: RawLanguageModelAvailability = "downloadable";
    let createCalls = 0;
    let baseDestroyCalls = 0;

    const mockProvider: LanguageModelProvider = {
      async availability() {
        return availabilityState;
      },
      async create(opts?: LanguageModelCreateOptions) {
        createCalls += 1;
        if (opts?.monitor) {
          const listeners: Array<(ev: { loaded?: number; total?: number }) => void> = [];
          const monitor: LanguageModelCreateMonitor = {
            addEventListener(_type, fn) {
              listeners.push(fn);
            },
          };
          opts.monitor(monitor);
          for (const fn of listeners) {
            fn({ loaded: 50, total: 100 });
          }
        }
        availabilityState = "available";
        const session: LanguageModelSession = {
          async prompt() {
            return '{"verdict":"uncertain","confidence":0.5,"reason":"ok"}';
          },
          async clone() {
            return session;
          },
          destroy() {
            baseDestroyCalls += 1;
          },
        };
        return session;
      },
    };

    const stage2 = new Stage2JudgeService({
      provider: mockProvider,
      nowFn: () => nowMs,
      idleTimeoutMs: 60_000,
    });
    const engine = new ThreadFilterEngine();
    const gpuHost = new WebGpuEmbeddingHost(createMockBackend().backendFactory);
    const weightStore = new IndexedDbWeightStore("test-stage2-dl", null);
    const service = new OffscreenFilterService(engine, gpuHost, weightStore, {
      stage2Judge: stage2,
    });

    // Background enqueue while "downloadable" MUST NOT trigger create()
    stage2.enqueue({
      threadId: "9900" as ThreadId,
      tweetId: "9901" as TweetId,
      replyText: "Spot on take!",
      replyLang: "en",
    });
    await stage2.waitForIdle();
    assert.equal(createCalls, 0, "never triggers download in background");

    // Status check reports "downloadable" without calling create()
    const statusRes = await service.handleRequest({
      kind: "trigger_stage2_download",
      action: "status",
    });
    assert.equal(statusRes.kind, "stage2_status_ok");
    if (statusRes.kind === "stage2_status_ok") {
      assert.equal(statusRes.status, "downloadable");
    }
    assert.equal(createCalls, 0);

    // Explicit user-gesture download invokes create() and caches base session
    const dlRes = await service.handleRequest({
      kind: "trigger_stage2_download",
      action: "download",
    });
    assert.equal(dlRes.kind, "stage2_status_ok");
    if (dlRes.kind === "stage2_status_ok") {
      assert.equal(dlRes.status, "available");
      assert.equal(dlRes.downloadProgress, 1);
    }
    assert.equal(createCalls, 1);
    assert.equal(stage2.hasActiveSession(), true);

    // Idle unload destroys base session after inactivity
    nowMs += 30_000;
    assert.equal(await stage2.unloadIfIdle(nowMs), false);
    nowMs += 30_001;
    assert.equal(await stage2.unloadIfIdle(nowMs), true);
    assert.equal(baseDestroyCalls, 1);
    assert.equal(stage2.hasActiveSession(), false);
  });
});

describe("BackgroundMessageRouter Stage-2 relay, settings broadcast, and onInstalled reinjection", () => {
  it("tracks threadId -> tabs, relays stage2_verdict_update only to matching tabs, broadcasts settings changes, and reinjects graphql-tap + content.js", async () => {
    const sentToTabs: Array<{ readonly tabId: number; readonly message: unknown }> = [];
    const failingTabs = new Set<number>();
    const openTabs = [
      { id: 42, url: "https://x.com/alice/status/8000" },
      { id: 99, url: "https://twitter.com/bob/status/8001" },
    ];

    const tabsApi: BackgroundTabsApi = {
      async query() {
        return openTabs;
      },
      async sendMessage(tabId, message) {
        if (failingTabs.has(tabId)) {
          throw new Error("Receiving end does not exist");
        }
        sentToTabs.push({ tabId, message });
      },
    };

    const offscreenApi = {
      async hasDocument() {
        return true;
      },
      async createDocument() {},
    };

    const service = new OffscreenFilterService(
      new ThreadFilterEngine(),
      new WebGpuEmbeddingHost(createMockBackend().backendFactory),
      new IndexedDbWeightStore("test-bg-router", null),
    );
    const router = new BackgroundMessageRouter(
      offscreenApi,
      (req) => service.handleRequest(req),
      tabsApi,
    );

    // Tab 42 evaluates thread 8000; Tab 99 evaluates thread 8001
    await router.handleMessage(
      {
        kind: "evaluate_batch",
        threadId: "8000",
        candidates: [makeCandidate("8000", "alice", "Hello thread 8000", { isOp: true })],
      },
      42,
    );
    await router.handleMessage(
      {
        kind: "evaluate_batch",
        threadId: "8001",
        candidates: [makeCandidate("8001", "bob", "Hello thread 8001", { isOp: true })],
      },
      99,
    );

    assert.deepEqual(router.getTrackedTabs("8000" as ThreadId), [42]);
    assert.deepEqual(router.getTrackedTabs("8001" as ThreadId), [99]);

    // Stage-2 update for thread 8000 from offscreen (senderTabId === undefined) -> ONLY tab 42
    const stage2Msg: Stage2VerdictUpdateMessage = {
      kind: "stage2_verdict_update",
      threadId: "8000" as ThreadId,
      tweetId: "8010" as TweetId,
      verdict: {
        tweetId: "8010" as TweetId,
        action: "keep",
        score: 0.4,
        tokenCount: asTokenCount(8),
        reason: { kind: "clean", action: "keep" },
      },
      target: "content",
    };
    await router.handleMessage(stage2Msg, undefined);

    assert.equal(sentToTabs.length, 1);
    assert.equal(sentToTabs[0]?.tabId, 42);
    assert.equal((sentToTabs[0]?.message as Stage2VerdictUpdateMessage).tweetId, "8010");

    // Settings update broadcasts { kind: "xrf_settings_changed" } to ALL open X tabs
    sentToTabs.length = 0;
    const updRes = await router.handleMessage({
      kind: "update_settings",
      settings: { ...DEFAULT_FILTER_SETTINGS, preset: "strict", revision: 2 },
    });
    assert.equal(updRes?.kind, "settings_ok");
    assert.equal(sentToTabs.length, 2);
    assert.deepEqual(
      sentToTabs.map((e) => [e.tabId, e.message]),
      [
        [42, { kind: "xrf_settings_changed" }],
        [99, { kind: "xrf_settings_changed" }],
      ],
    );

    // undo_feedback when stack is empty (undone: false) does NOT broadcast
    sentToTabs.length = 0;
    const emptyUndo = await router.handleMessage({ kind: "undo_feedback" });
    assert.equal(emptyUndo?.kind, "undo_feedback_ok");
    assert.equal(sentToTabs.length, 0);

    // reset_feedback broadcasts { kind: "xrf_settings_changed" }
    const resetRes = await router.handleMessage({ kind: "reset_feedback" });
    assert.equal(resetRes?.kind, "reset_feedback_ok");
    assert.equal(sentToTabs.length, 2);

    // Dead tab gets pruned on failed Stage-2 delivery
    failingTabs.add(42);
    await router.handleMessage(stage2Msg, undefined);
    assert.deepEqual(router.getTrackedTabs("8000" as ThreadId), []);

    // Fallback when no tracked tabs remain: matches open tab URL containing /status/8000
    failingTabs.delete(42);
    sentToTabs.length = 0;
    await router.handleMessage(stage2Msg, undefined);
    assert.equal(sentToTabs.length, 1);
    assert.equal(sentToTabs[0]?.tabId, 42);

    // onInstalled reinjection executes graphql-tap.js (MAIN) and content.js on open X tabs
    const injections: Array<{
      readonly tabId: number;
      readonly files: string[];
      readonly world?: "MAIN" | "ISOLATED";
    }> = [];
    const scriptingApi: BackgroundScriptingApi = {
      async executeScript(inj) {
        injections.push({
          tabId: inj.target.tabId,
          files: inj.files,
          ...(inj.world !== undefined ? { world: inj.world } : {}),
        });
      },
    };

    const injectedCount = await reinjectOpenXTabs(tabsApi, scriptingApi);
    assert.equal(injectedCount, 2);
    assert.equal(injections.length, 4);
    assert.deepEqual(
      injections.filter((i) => i.tabId === 42),
      [
        { tabId: 42, files: ["graphql-tap.js"], world: "MAIN" },
        { tabId: 42, files: ["content.js"] },
      ],
    );
  });
});
