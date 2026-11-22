import type { ThreadFilterEngine } from "../domain/engine.ts";
import {
  type OnlineClassifierWeights,
  SEED_CORPUS,
  type SeedPrototype,
  createInitialClassifierWeights,
  recordAuthorOutcome,
} from "../domain/seeds.ts";
import { insertIntoReplayBuffer, trainFromReplayBuffer } from "../domain/trained-head.ts";
import {
  type AuthorId,
  type AuthorStats,
  CLASSIFICATION_TASK_PREFIX,
  COMPACT_V2_DIM,
  type CandidateTweet,
  DEFAULT_FILTER_SETTINGS,
  type EmbeddedTweet,
  type FeedbackReplayEntry,
  type FilterSettings,
  type LangBiasKey,
  MAX_AUTO_FLAGGED_TWEET_IDS,
  MAX_CACHED_THREADS,
  MAX_REPLAY_BUFFER_ENTRIES,
  OFFSCREEN_IDLE_TIMEOUT_MS,
  RAW_EMBEDDING_DIM,
  SIAMESE_TOTAL_DIM,
  type Stage2StatusTag,
  type ThreadId,
  type TokenCount,
  type TweetId,
  type UndoRecord,
  type UnitVec256,
  type VerdictRecord,
  asTokenCount,
  formatClassificationPrompt,
  parseAuthorId,
  parseTweetId,
  truncateAndNormalizeBatch256,
} from "../domain/types.ts";
import {
  type OffscreenRpcRequestV2,
  type OffscreenRpcResponse,
  type Stage2VerdictUpdateMessage,
  parseFilterSettings,
} from "../shared/rpc.ts";
import type { Stage2Completion, Stage2JudgeService } from "./stage2-judge.ts";

export const MODEL_ID = "onnx-community/embeddinggemma-300m-ONNX" as const;
export const MODEL_DTYPE = "q8" as const;
export const MAX_EMBED_CHUNK_SIZE = 16 as const;

export function yieldBetweenChunks(): Promise<void> {
  return new Promise<void>((resolve) => {
    queueMicrotask(resolve);
  });
}

export interface RawBatchInferenceOutput {
  readonly embeddings256: ReadonlyArray<UnitVec256>;
  readonly contentTokenCounts: ReadonlyArray<TokenCount>;
}

export interface EmbeddingBackend {
  readonly inferPrompts: (
    prompts: ReadonlyArray<string>,
    rawTexts: ReadonlyArray<string>,
  ) => Promise<RawBatchInferenceOutput>;
  readonly dispose: () => Promise<void>;
}

export type GpuHostState =
  | { readonly status: "unloaded"; readonly deviceEpoch: number }
  | {
      readonly status: "loading";
      readonly deviceEpoch: number;
      readonly readyPromise: Promise<EmbeddingBackend>;
    }
  | {
      readonly status: "ready";
      readonly deviceEpoch: number;
      readonly lastUsedAtMs: number;
      readonly backend: EmbeddingBackend;
    }
  | {
      readonly status: "recovering";
      readonly deviceEpoch: number;
      readonly recoveryPromise: Promise<EmbeddingBackend>;
    };

function countMaskTokens(
  maskData: ArrayLike<number | bigint>,
  rowIndex: number,
  seqLen: number,
): number {
  const offset = rowIndex * seqLen;
  let sum = 0;
  for (let i = 0; i < seqLen; i++) {
    const val = Number(maskData[offset + i] ?? 0);
    if (val > 0) {
      sum += 1;
    }
  }
  return sum;
}

export async function createTransformersWebGpuBackend(): Promise<EmbeddingBackend> {
  const { AutoModel, AutoTokenizer, env } = await import("@huggingface/transformers");
  if (env.backends.onnx?.wasm) {
    env.backends.onnx.wasm.wasmPaths =
      typeof chrome !== "undefined" && chrome.runtime?.getURL
        ? chrome.runtime.getURL("")
        : "./";
  }

  const hasWebGpu = typeof navigator !== "undefined" && "gpu" in navigator;
  const device = hasWebGpu ? "webgpu" : "wasm";

  const [tokenizer, model] = await Promise.all([
    AutoTokenizer.from_pretrained(MODEL_ID),
    AutoModel.from_pretrained(MODEL_ID, {
      device,
      dtype: MODEL_DTYPE,
    }),
  ]);

  const prefixEncoded = await tokenizer(CLASSIFICATION_TASK_PREFIX, {
    padding: false,
    truncation: true,
  });
  const prefixMask = prefixEncoded.attention_mask?.data as ArrayLike<number | bigint> | undefined;
  const prefixTokenOffset = prefixMask
    ? Math.max(0, countMaskTokens(prefixMask, 0, prefixMask.length) - 1)
    : 6;

  return {
    async inferPrompts(
      prompts: ReadonlyArray<string>,
      rawTexts: ReadonlyArray<string>,
    ): Promise<RawBatchInferenceOutput> {
      if (prompts.length === 0) {
        return { embeddings256: [], contentTokenCounts: [] };
      }

      const promptList = Array.from(prompts);
      const tokenized = await tokenizer(promptList, {
        padding: true,
        truncation: true,
      });

      const maskTensor = tokenized.attention_mask;
      const maskDims = maskTensor?.dims ?? [promptList.length, 0];
      const seqLen = Number(maskDims[maskDims.length - 1] ?? 0);
      const maskData = (maskTensor?.data ?? []) as ArrayLike<number | bigint>;

      const contentTokenCounts: TokenCount[] = [];
      for (let r = 0; r < promptList.length; r++) {
        const rawTrimmed = (rawTexts[r] ?? "").trim();
        if (rawTrimmed.length === 0) {
          contentTokenCounts.push(asTokenCount(0));
          continue;
        }
        const totalTokens = seqLen > 0 ? countMaskTokens(maskData, r, seqLen) : 1;
        const contentTokens = Math.max(1, totalTokens - prefixTokenOffset);
        contentTokenCounts.push(asTokenCount(contentTokens));
      }

      const modelOut = (await model(tokenized)) as {
        readonly sentence_embedding?: { readonly data: Float32Array; readonly dims: readonly number[] };
        readonly last_hidden_state?: { readonly data: Float32Array; readonly dims: readonly number[] };
      };
      const tensorOut = modelOut.sentence_embedding ?? modelOut.last_hidden_state;
      if (!tensorOut) {
        throw new Error("EmbeddingGemma returned no sentence_embedding tensor");
      }

      const rawData = tensorOut.data;
      const outDims = tensorOut.dims;
      const hiddenDim = Number(outDims[outDims.length - 1] ?? RAW_EMBEDDING_DIM);
      const embeddings256 = truncateAndNormalizeBatch256(rawData, promptList.length, hiddenDim);

      return { embeddings256, contentTokenCounts };
    },
    async dispose(): Promise<void> {
      await model.dispose();
    },
  };
}

export class WebGpuEmbeddingHost {
  private state: GpuHostState = { status: "unloaded", deviceEpoch: 0 };
  private readonly backendFactory: () => Promise<EmbeddingBackend>;
  private readonly nowFn: () => number;
  private readonly idleTimeoutMs: number;
  private readonly maxChunkSize: number;
  private readonly yieldFn: () => Promise<void>;

  constructor(
    backendFactory: () => Promise<EmbeddingBackend> = createTransformersWebGpuBackend,
    nowFn: () => number = () => Date.now(),
    idleTimeoutMs: number = OFFSCREEN_IDLE_TIMEOUT_MS,
    maxChunkSize: number = MAX_EMBED_CHUNK_SIZE,
    yieldFn: () => Promise<void> = yieldBetweenChunks,
  ) {
    this.backendFactory = backendFactory;
    this.nowFn = nowFn;
    this.idleTimeoutMs = idleTimeoutMs;
    this.maxChunkSize = Math.max(1, Math.floor(maxChunkSize));
    this.yieldFn = yieldFn;
  }

  public getState(): { readonly status: GpuHostState["status"]; readonly deviceEpoch: number } {
    return {
      status: this.state.status,
      deviceEpoch: this.state.deviceEpoch,
    };
  }

  private async acquireBackend(): Promise<EmbeddingBackend> {
    if (this.state.status === "ready") {
      const backend = this.state.backend;
      this.state = {
        status: "ready",
        deviceEpoch: this.state.deviceEpoch,
        lastUsedAtMs: this.nowFn(),
        backend,
      };
      return backend;
    }

    if (this.state.status === "loading") {
      return this.state.readyPromise;
    }

    if (this.state.status === "recovering") {
      return this.state.recoveryPromise;
    }

    const epoch = this.state.deviceEpoch + 1;
    const readyPromise = this.backendFactory().then(
      (backend) => {
        this.state = {
          status: "ready",
          deviceEpoch: epoch,
          lastUsedAtMs: this.nowFn(),
          backend,
        };
        return backend;
      },
      (err) => {
        this.state = { status: "unloaded", deviceEpoch: epoch };
        throw err;
      },
    );

    this.state = {
      status: "loading",
      deviceEpoch: epoch,
      readyPromise,
    };

    return readyPromise;
  }

  private async recoverBackendAfterLoss(failedBackend: EmbeddingBackend): Promise<EmbeddingBackend> {
    if (this.state.status === "recovering") {
      return this.state.recoveryPromise;
    }

    const nextEpoch = this.state.deviceEpoch + 1;
    const recoveryPromise = (async () => {
      try {
        await failedBackend.dispose();
      } catch {}
      try {
        const fresh = await this.backendFactory();
        this.state = {
          status: "ready",
          deviceEpoch: nextEpoch,
          lastUsedAtMs: this.nowFn(),
          backend: fresh,
        };
        return fresh;
      } catch (err) {
        this.state = { status: "unloaded", deviceEpoch: nextEpoch };
        throw err;
      }
    })();

    this.state = {
      status: "recovering",
      deviceEpoch: nextEpoch,
      recoveryPromise,
    };

    return recoveryPromise;
  }

  private async inferChunkWithRecovery(
    prompts: ReadonlyArray<string>,
    rawTexts: ReadonlyArray<string>,
  ): Promise<RawBatchInferenceOutput> {
    const backend = await this.acquireBackend();
    try {
      const out = await backend.inferPrompts(prompts, rawTexts);
      if (this.state.status === "ready") {
        this.state = {
          ...this.state,
          lastUsedAtMs: this.nowFn(),
        };
      }
      return out;
    } catch {
      const recovered = await this.recoverBackendAfterLoss(backend);
      const out = await recovered.inferPrompts(prompts, rawTexts);
      if (this.state.status === "ready") {
        this.state = {
          ...this.state,
          lastUsedAtMs: this.nowFn(),
        };
      }
      return out;
    }
  }

  public async embedTexts(rawTexts: ReadonlyArray<string>): Promise<RawBatchInferenceOutput> {
    if (rawTexts.length === 0) {
      return { embeddings256: [], contentTokenCounts: [] };
    }
    const prompts = rawTexts.map((t) => formatClassificationPrompt(t));

    if (rawTexts.length <= this.maxChunkSize) {
      return this.inferChunkWithRecovery(prompts, rawTexts);
    }

    const embeddings256: UnitVec256[] = [];
    const contentTokenCounts: TokenCount[] = [];
    for (let offset = 0; offset < rawTexts.length; offset += this.maxChunkSize) {
      const end = Math.min(rawTexts.length, offset + this.maxChunkSize);
      const chunkOut = await this.inferChunkWithRecovery(
        prompts.slice(offset, end),
        rawTexts.slice(offset, end),
      );
      embeddings256.push(...chunkOut.embeddings256);
      contentTokenCounts.push(...chunkOut.contentTokenCounts);
      if (end < rawTexts.length) {
        await this.yieldFn();
      }
    }

    return { embeddings256, contentTokenCounts };
  }

  public async embedCandidates(
    candidates: ReadonlyArray<CandidateTweet>,
  ): Promise<ReadonlyArray<EmbeddedTweet>> {
    if (candidates.length === 0) {
      return [];
    }
    const rawTexts = candidates.map((c) => c.normalizedText);
    const { embeddings256, contentTokenCounts } = await this.embedTexts(rawTexts);

    return candidates.map((candidate, idx) => ({
      candidate,
      embedding: embeddings256[idx]!,
      tokenCount: contentTokenCounts[idx]!,
    }));
  }

  public async unloadIfIdle(nowMs: number = this.nowFn()): Promise<boolean> {
    if (this.state.status !== "ready") {
      return false;
    }
    if (nowMs - this.state.lastUsedAtMs < this.idleTimeoutMs) {
      return false;
    }

    const backend = this.state.backend;
    const epoch = this.state.deviceEpoch;
    this.state = { status: "unloaded", deviceEpoch: epoch };
    await backend.dispose();
    return true;
  }
}

export const DB_NAME = "x-reply-filter-db" as const;
export const DB_VERSION = 2 as const;
export const STORE_NAME = "classifier_weights" as const;
export const REPLAY_STORE_NAME = "feedback_replay_v2" as const;
export const SETTINGS_STORE_NAME = "settings_v2" as const;
export const WEIGHTS_KEY_V1 = "primary_v1" as const;
export const WEIGHTS_KEY_V2 = "primary_v2" as const;
export const SETTINGS_KEY_V2 = "current_v2" as const;
export const IDB_RECOVERY_COOLDOWN_MS = 5_000 as const;

const LANG_BIAS_KEYS: ReadonlyArray<LangBiasKey> = ["en", "zh-Hans", "zh-Hant", "ja", "other"];
const MAX_UNDO_HISTORY_ENTRIES = 32 as const;

interface StoredWeightsPayloadV1 {
  readonly weights: Float32Array;
  readonly bias: number;
  readonly updateCount: number;
  readonly authorStatsEntries?: ReadonlyArray<readonly [string, AuthorStats]>;
}

interface StoredWeightsPayloadV2 {
  readonly schemaVersion: 2;
  readonly theta: Float32Array;
  readonly langBias: Readonly<Record<LangBiasKey, number>>;
  readonly theta0Hash: string;
  readonly updateCount: number;
  readonly rollbackCount: number;
  readonly authorStatsEntries: ReadonlyArray<readonly [string, AuthorStats]>;
  readonly recordedAutoFlagIds: ReadonlyArray<TweetId>;
  readonly undoHistory: ReadonlyArray<UndoRecord>;
  readonly weightsRevision?: number | undefined;
  readonly dimsReviewedCount?: number | undefined;
  readonly dimsOverriddenHumanCount?: number | undefined;
}

export interface PersistenceHealth {
  readonly degraded: boolean;
  readonly settingsDegraded: boolean;
  readonly lastError: string | null;
}

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

function deserializeAuthorStats(
  entries: ReadonlyArray<readonly [string, AuthorStats]> | undefined,
): Map<AuthorId, AuthorStats> {
  const map = new Map<AuthorId, AuthorStats>();
  if (!entries) {
    return map;
  }
  for (const [rawId, stats] of entries) {
    const id = parseAuthorId(rawId);
    if (id !== null && stats && typeof stats.spamCount === "number") {
      map.set(id, {
        spamCount: Math.max(0, Math.floor(stats.spamCount)),
        humanCount: Math.max(0, Math.floor(stats.humanCount ?? 0)),
        autoFlagCount: Math.max(0, Math.floor(stats.autoFlagCount ?? 0)),
        ...(typeof stats.lastUpdatedAtMs === "number" && Number.isFinite(stats.lastUpdatedAtMs)
          ? { lastUpdatedAtMs: stats.lastUpdatedAtMs }
          : {}),
        ...(typeof stats.autoFlagDecayed === "number" && Number.isFinite(stats.autoFlagDecayed)
          ? { autoFlagDecayed: Math.max(0, stats.autoFlagDecayed) }
          : {}),
      });
    }
  }
  return map;
}

function deserializeReplayEntries(rawList: ReadonlyArray<unknown>): FeedbackReplayEntry[] {
  const validLangs = new Set<string>(LANG_BIAS_KEYS);
  const out: FeedbackReplayEntry[] = [];
  for (const raw of rawList) {
    if (typeof raw !== "object" || raw === null) continue;
    const e = raw as Record<string, unknown>;
    const tweetId = typeof e.tweetId === "string" ? parseTweetId(e.tweetId) : null;
    const rawZ = e.z;
    const zArr =
      rawZ instanceof Float32Array
        ? Array.from(rawZ)
        : Array.isArray(rawZ)
          ? rawZ
          : null;
    if (
      tweetId === null ||
      zArr === null ||
      zArr.length !== COMPACT_V2_DIM ||
      !zArr.every((v) => typeof v === "number" && Number.isFinite(v)) ||
      typeof e.fixed !== "number" ||
      !Number.isFinite(e.fixed) ||
      (e.y !== 0 && e.y !== 1) ||
      typeof e.lang !== "string" ||
      !validLangs.has(e.lang) ||
      typeof e.tsMs !== "number" ||
      !Number.isFinite(e.tsMs) ||
      typeof e.topicHash !== "number" ||
      !Number.isFinite(e.topicHash) ||
      (e.source !== "user_spam" && e.source !== "user_human")
    ) {
      continue;
    }
    const authorId =
      typeof e.authorId === "string" ? (parseAuthorId(e.authorId) ?? undefined) : undefined;
    out.push({
      tweetId,
      ...(authorId !== undefined ? { authorId } : {}),
      z: zArr as number[],
      fixed: e.fixed,
      y: e.y,
      lang: e.lang as LangBiasKey,
      tsMs: e.tsMs,
      topicHash: Math.floor(e.topicHash) & 0xff,
      source: e.source,
    });
  }
  out.sort((a, b) => a.tsMs - b.tsMs);
  return out.slice(-MAX_REPLAY_BUFFER_ENTRIES);
}

function deserializeLangBias(
  raw: unknown,
  fallback: Readonly<Record<LangBiasKey, number>>,
): Readonly<Record<LangBiasKey, number>> {
  if (typeof raw !== "object" || raw === null) {
    return { ...fallback };
  }
  const obj = raw as Record<string, unknown>;
  const next: Record<LangBiasKey, number> = {
    en: typeof obj.en === "number" && Number.isFinite(obj.en) ? obj.en : fallback.en,
    "zh-Hans":
      typeof obj["zh-Hans"] === "number" && Number.isFinite(obj["zh-Hans"])
        ? obj["zh-Hans"]
        : fallback["zh-Hans"],
    "zh-Hant":
      typeof obj["zh-Hant"] === "number" && Number.isFinite(obj["zh-Hant"])
        ? obj["zh-Hant"]
        : fallback["zh-Hant"],
    ja: typeof obj.ja === "number" && Number.isFinite(obj.ja) ? obj.ja : fallback.ja,
    other:
      typeof obj.other === "number" && Number.isFinite(obj.other) ? obj.other : fallback.other,
  };
  return next;
}

function deserializeAutoFlagIds(raw: unknown): Set<TweetId> {
  const out = new Set<TweetId>();
  if (!Array.isArray(raw)) {
    return out;
  }
  for (const item of raw) {
    if (typeof item === "string") {
      const id = parseTweetId(item);
      if (id !== null) {
        out.add(id);
      }
    }
  }
  if (out.size > MAX_AUTO_FLAGGED_TWEET_IDS) {
    const trimmed = Array.from(out).slice(-MAX_AUTO_FLAGGED_TWEET_IDS);
    return new Set<TweetId>(trimmed);
  }
  return out;
}

function toFloat32Vector(raw: unknown, expectedLen: number): Float32Array | null {
  if (raw instanceof Float32Array && raw.length === expectedLen) {
    return new Float32Array(raw);
  }
  if (Array.isArray(raw) && raw.length === expectedLen) {
    const arr = new Float32Array(expectedLen);
    for (let i = 0; i < expectedLen; i++) {
      const v = raw[i];
      if (typeof v !== "number" || !Number.isFinite(v)) {
        return null;
      }
      arr[i] = v;
    }
    return arr;
  }
  return null;
}

function areFilterSettingsEqual(a: FilterSettings, b: FilterSettings): boolean {
  return (
    a.preset === b.preset &&
    a.revision === b.revision &&
    a.toggles.collapseBlueCheckReplyGuys === b.toggles.collapseBlueCheckReplyGuys &&
    a.toggles.collapseMediaOnlyReplies === b.toggles.collapseMediaOnlyReplies &&
    a.toggles.collapseBotSummons === b.toggles.collapseBotSummons &&
    a.toggles.showSuspectBadges === b.toggles.showSuspectBadges &&
    a.toggles.learnFromFeedback === b.toggles.learnFromFeedback &&
    a.toggles.enableStage2 === b.toggles.enableStage2
  );
}

function mergeDegradedWeightsIntoLoaded(
  loaded: OnlineClassifierWeights,
  degraded: OnlineClassifierWeights,
): OnlineClassifierWeights {
  const mergedAuthors = new Map<AuthorId, AuthorStats>(loaded.authorStats);
  for (const [id, mem] of degraded.authorStats.entries()) {
    const prev = mergedAuthors.get(id);
    if (prev === undefined) {
      mergedAuthors.set(id, mem);
    } else {
      const mergedLastUpdated =
        prev.lastUpdatedAtMs !== undefined && mem.lastUpdatedAtMs !== undefined
          ? Math.max(prev.lastUpdatedAtMs, mem.lastUpdatedAtMs)
          : (mem.lastUpdatedAtMs ?? prev.lastUpdatedAtMs);
      const mergedDecayed =
        (prev.autoFlagDecayed ?? prev.autoFlagCount) +
        (mem.autoFlagDecayed ?? mem.autoFlagCount);
      mergedAuthors.set(id, {
        spamCount: prev.spamCount + mem.spamCount,
        humanCount: prev.humanCount + mem.humanCount,
        autoFlagCount: prev.autoFlagCount + mem.autoFlagCount,
        ...(mergedLastUpdated !== undefined ? { lastUpdatedAtMs: mergedLastUpdated } : {}),
        autoFlagDecayed: mergedDecayed,
      });
    }
  }

  let mergedReplay: FeedbackReplayEntry[] = [...(loaded.replayBuffer ?? [])];
  const degradedReplay = degraded.replayBuffer ?? [];
  for (const entry of degradedReplay) {
    mergedReplay = insertIntoReplayBuffer(mergedReplay, entry, MAX_REPLAY_BUFFER_ENTRIES);
  }

  const mergedAutoFlags = new Set<TweetId>(loaded.recordedAutoFlagIds ?? []);
  for (const tid of degraded.recordedAutoFlagIds ?? []) {
    if (mergedAutoFlags.size >= MAX_AUTO_FLAGGED_TWEET_IDS && !mergedAutoFlags.has(tid)) {
      const oldest = mergedAutoFlags.values().next().value;
      if (oldest !== undefined) mergedAutoFlags.delete(oldest);
    }
    mergedAutoFlags.add(tid);
  }

  const mergedUndo = [
    ...(loaded.undoHistory ?? []),
    ...(degraded.undoHistory ?? []),
  ].slice(-MAX_UNDO_HISTORY_ENTRIES);

  const dimsReviewedCount =
    (loaded.dimsReviewedCount ?? 0) + (degraded.dimsReviewedCount ?? 0);
  const dimsOverriddenHumanCount =
    (loaded.dimsOverriddenHumanCount ?? 0) + (degraded.dimsOverriddenHumanCount ?? 0);
  const updateCount = loaded.updateCount + degraded.updateCount;
  const weightsRevision = Math.max(loaded.weightsRevision ?? 1, degraded.weightsRevision ?? 1) + 1;

  const baseMerged: OnlineClassifierWeights = {
    ...loaded,
    authorStats: mergedAuthors,
    replayBuffer: mergedReplay,
    recordedAutoFlagIds: mergedAutoFlags,
    undoHistory: mergedUndo,
    updateCount,
    rollbackCount: (loaded.rollbackCount ?? 0) + (degraded.rollbackCount ?? 0),
    weightsRevision,
    dimsReviewedCount,
    dimsOverriddenHumanCount,
  };

  if (degradedReplay.length > 0 && mergedReplay.length > 0) {
    const retrained = trainFromReplayBuffer(baseMerged, mergedReplay);
    return {
      ...retrained.nextWeights,
      authorStats: mergedAuthors,
      recordedAutoFlagIds: mergedAutoFlags,
      undoHistory: mergedUndo,
      updateCount,
      weightsRevision,
      dimsReviewedCount,
      dimsOverriddenHumanCount,
    };
  }

  return baseMerged;
}

export class IndexedDbWeightStore {
  private memoryFallback: OnlineClassifierWeights | null = null;
  private weightsLoaded = false;
  private weightsLoadPromise: Promise<OnlineClassifierWeights> | null = null;
  private weightsMutationSeq = 0;
  private weightsResetEpoch = 0;
  private weightsPersistenceDegraded = false;
  private nextWeightsRetryAtMs = 0;
  private degradedWeightsMutated = false;
  private lastPersistedReplayRef: ReadonlyArray<FeedbackReplayEntry> | null = null;

  private settingsFallback: FilterSettings | null = null;
  private settingsLoaded = false;
  private settingsLoadPromise: Promise<FilterSettings> | null = null;
  private settingsMutationSeq = 0;
  private settingsPersistenceDegraded = false;
  private nextSettingsRetryAtMs = 0;
  private degradedSettingsMutated = false;

  private lastPersistenceError: string | null = null;
  private preservedV1Legacy: { readonly weights: Float32Array; readonly bias: number } | null =
    null;
  private readonly dbName: string;
  private readonly idbFactory: IDBFactory | null;
  private readonly nowFn: () => number;
  private readonly retryCooldownMs: number;
  private writeChain: Promise<void> = Promise.resolve();

  constructor(
    dbName: string = DB_NAME,
    idbFactory?: IDBFactory | null,
    nowFn: () => number = () => Date.now(),
    retryCooldownMs: number = IDB_RECOVERY_COOLDOWN_MS,
  ) {
    this.dbName = dbName;
    if (idbFactory !== undefined) {
      this.idbFactory = idbFactory;
    } else {
      this.idbFactory = typeof indexedDB !== "undefined" ? indexedDB : null;
    }
    this.nowFn = nowFn;
    this.retryCooldownMs = Math.max(0, retryCooldownMs);
  }

  public getPersistenceHealth(): PersistenceHealth {
    return {
      degraded: this.weightsPersistenceDegraded,
      settingsDegraded: this.settingsPersistenceDegraded,
      lastError: this.lastPersistenceError,
    };
  }

  private openDatabase(): Promise<IDBDatabase> {
    if (this.idbFactory === null) {
      return Promise.reject(new Error("IndexedDB unavailable"));
    }
    const factory = this.idbFactory;
    return new Promise((resolve, reject) => {
      const req = factory.open(this.dbName, DB_VERSION);
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains(STORE_NAME)) {
          db.createObjectStore(STORE_NAME);
        }
        if (!db.objectStoreNames.contains(REPLAY_STORE_NAME)) {
          db.createObjectStore(REPLAY_STORE_NAME);
        }
        if (!db.objectStoreNames.contains(SETTINGS_STORE_NAME)) {
          db.createObjectStore(SETTINGS_STORE_NAME);
        }
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error ?? new Error("IndexedDB open failed"));
    });
  }

  private async readWeightsFromIdbOnce(): Promise<OnlineClassifierWeights> {
    const db = await this.openDatabase();
    let v1Payload: StoredWeightsPayloadV1 | undefined;
    let v2Payload: StoredWeightsPayloadV2 | undefined;
    let rawReplay: ReadonlyArray<unknown> = [];

    try {
      const res = await new Promise<{
        readonly v1Payload: StoredWeightsPayloadV1 | undefined;
        readonly v2Payload: StoredWeightsPayloadV2 | undefined;
        readonly rawReplay: ReadonlyArray<unknown>;
      }>((resolve, reject) => {
        const storeNames = db.objectStoreNames.contains(REPLAY_STORE_NAME)
          ? [STORE_NAME, REPLAY_STORE_NAME]
          : [STORE_NAME];
        const tx = db.transaction(storeNames, "readonly");
        const wStore = tx.objectStore(STORE_NAME);
        const v1Req = wStore.get(WEIGHTS_KEY_V1);
        const v2Req = wStore.get(WEIGHTS_KEY_V2);
        const replayReq = db.objectStoreNames.contains(REPLAY_STORE_NAME)
          ? tx.objectStore(REPLAY_STORE_NAME).getAll()
          : null;

        let v1Res: StoredWeightsPayloadV1 | undefined;
        let v2Res: StoredWeightsPayloadV2 | undefined;
        let replayRes: ReadonlyArray<unknown> = [];

        v1Req.onsuccess = () => {
          v1Res = v1Req.result as StoredWeightsPayloadV1 | undefined;
        };
        v1Req.onerror = () => reject(v1Req.error ?? new Error("primary_v1 read failed"));
        v2Req.onsuccess = () => {
          v2Res = v2Req.result as StoredWeightsPayloadV2 | undefined;
        };
        v2Req.onerror = () => reject(v2Req.error ?? new Error("primary_v2 read failed"));
        if (replayReq !== null) {
          replayReq.onsuccess = () => {
            replayRes = Array.isArray(replayReq.result) ? replayReq.result : [];
          };
          replayReq.onerror = () => reject(replayReq.error ?? new Error("replay read failed"));
        }

        tx.oncomplete = () =>
          resolve({
            v1Payload: v1Res,
            v2Payload: v2Res,
            rawReplay: replayRes,
          });
        tx.onerror = () => reject(tx.error ?? new Error("IndexedDB read error"));
        tx.onabort = () => reject(tx.error ?? new Error("IndexedDB read aborted"));
      });
      v1Payload = res.v1Payload;
      v2Payload = res.v2Payload;
      rawReplay = res.rawReplay;
    } finally {
      db.close();
    }

    const defaults = createInitialClassifierWeights();

    // Preserve untouched v1 778d weights for future dual-write compatibility (NEVER delete primary_v1)
    let legacyWeights = defaults.weights;
    let legacyBias = defaults.bias;
    if (v1Payload && v1Payload.weights instanceof Float32Array) {
      const migratedLegacy = new Float32Array(defaults.weights);
      const copyLen = Math.min(v1Payload.weights.length, SIAMESE_TOTAL_DIM);
      for (let i = 0; i < copyLen; i++) {
        migratedLegacy[i] = v1Payload.weights[i]!;
      }
      legacyWeights = migratedLegacy;
      legacyBias = typeof v1Payload.bias === "number" ? v1Payload.bias : defaults.bias;
      this.preservedV1Legacy = {
        weights: new Float32Array(v1Payload.weights),
        bias: legacyBias,
      };
    }

    // Lossless authorStats merge: start with primary_v1, overlay primary_v2
    // (preserving higher counts if primary_v1 advanced during a v1 downgrade session)
    const mergedAuthors = deserializeAuthorStats(v1Payload?.authorStatsEntries);
    if (v2Payload && v2Payload.schemaVersion === 2) {
      const v2Authors = deserializeAuthorStats(v2Payload.authorStatsEntries);
      for (const [id, stats] of v2Authors.entries()) {
        const prevV1 = mergedAuthors.get(id);
        if (
          prevV1 !== undefined &&
          (prevV1.spamCount > stats.spamCount ||
            prevV1.humanCount > stats.humanCount ||
            prevV1.autoFlagCount > stats.autoFlagCount)
        ) {
          mergedAuthors.set(id, {
            ...stats,
            spamCount: Math.max(prevV1.spamCount, stats.spamCount),
            humanCount: Math.max(prevV1.humanCount, stats.humanCount),
            autoFlagCount: Math.max(prevV1.autoFlagCount, stats.autoFlagCount),
          });
        } else {
          mergedAuthors.set(id, stats);
        }
      }
    }

    const replayBuffer = deserializeReplayEntries(rawReplay);
    this.lastPersistedReplayRef = replayBuffer;

    if (v2Payload && v2Payload.schemaVersion === 2) {
      const recordedAutoFlagIds = deserializeAutoFlagIds(v2Payload.recordedAutoFlagIds);
      const undoHistory = Array.isArray(v2Payload.undoHistory)
        ? (v2Payload.undoHistory.slice(-MAX_UNDO_HISTORY_ENTRIES) as UndoRecord[])
        : [];
      const updateCount = Math.max(
        0,
        Math.floor(v2Payload.updateCount ?? v1Payload?.updateCount ?? 0),
      );
      const rollbackCount = Math.max(0, Math.floor(v2Payload.rollbackCount ?? 0));
      const weightsRevision = Math.max(1, Math.floor(v2Payload.weightsRevision ?? 1));
      const dimsReviewedCount =
        typeof v2Payload.dimsReviewedCount === "number" &&
        Number.isFinite(v2Payload.dimsReviewedCount)
          ? Math.max(0, Math.floor(v2Payload.dimsReviewedCount))
          : 0;
      const dimsOverriddenHumanCount =
        typeof v2Payload.dimsOverriddenHumanCount === "number" &&
        Number.isFinite(v2Payload.dimsOverriddenHumanCount)
          ? Math.max(0, Math.floor(v2Payload.dimsOverriddenHumanCount))
          : 0;

      const parsedTheta = toFloat32Vector(v2Payload.theta, COMPACT_V2_DIM);
      const parsedLangBias = deserializeLangBias(v2Payload.langBias, defaults.langBias!);

      if (v2Payload.theta0Hash === defaults.theta0Hash && parsedTheta !== null) {
        return {
          ...defaults,
          weights: legacyWeights,
          bias: legacyBias,
          authorStats: mergedAuthors,
          theta: parsedTheta,
          langBias: parsedLangBias,
          replayBuffer,
          updateCount,
          rollbackCount,
          weightsRevision,
          recordedAutoFlagIds,
          undoHistory,
          dimsReviewedCount,
          dimsOverriddenHumanCount,
        };
      }

      // Shipped theta0 changed: re-anchor cleanly on defaults.theta0 and replay stored user labels
      const reanchoredBase: OnlineClassifierWeights = {
        ...defaults,
        weights: legacyWeights,
        bias: legacyBias,
        authorStats: mergedAuthors,
        replayBuffer,
        updateCount,
        rollbackCount,
        weightsRevision,
        recordedAutoFlagIds,
        undoHistory,
        dimsReviewedCount,
        dimsOverriddenHumanCount,
      };
      if (replayBuffer.length > 0) {
        const retrained = trainFromReplayBuffer(reanchoredBase, replayBuffer);
        return {
          ...retrained.nextWeights,
          updateCount,
          dimsReviewedCount,
          dimsOverriddenHumanCount,
        };
      }
      return reanchoredBase;
    }

    if (v1Payload) {
      // First upgrade from v1 -> v2: keep all authorStats + updateCount from primary_v1,
      // score with calibrated v2 head (never topic-contaminated v1 778d weights).
      return {
        ...defaults,
        weights: legacyWeights,
        bias: legacyBias,
        updateCount: Math.max(0, Math.floor(v1Payload.updateCount ?? 0)),
        authorStats: mergedAuthors,
        replayBuffer,
      };
    }

    return defaults;
  }

  private enqueuePersistWeights(
    weights: OnlineClassifierWeights,
    expectedSeq: number,
    options?: {
      readonly forceReplayWrite?: boolean;
      readonly allowWhileDegraded?: boolean;
      readonly clearDegradedOnSuccess?: boolean;
    },
  ): Promise<void> {
    const run = async (): Promise<void> => {
      if (this.weightsMutationSeq !== expectedSeq) {
        return;
      }
      if (this.weightsPersistenceDegraded && options?.allowWhileDegraded !== true) {
        return;
      }
      try {
        await this.persistWeightsToIdb(weights, options);
        if (options?.clearDegradedOnSuccess && this.weightsMutationSeq === expectedSeq) {
          this.weightsPersistenceDegraded = false;
          this.nextWeightsRetryAtMs = 0;
          this.degradedWeightsMutated = false;
          this.weightsLoaded = true;
        }
      } catch (err) {
        this.lastPersistenceError = `IndexedDB weights write failed: ${errorMessage(err)}`;
        console.warn("[x-reply-filter] IndexedDB weights write failed:", err);
      }
    };

    const next = this.writeChain.then(run, run);
    this.writeChain = next;
    return next;
  }

  public async getWeights(): Promise<OnlineClassifierWeights> {
    if (this.weightsLoaded && this.memoryFallback !== null) {
      return this.memoryFallback;
    }
    if (this.weightsLoadPromise !== null) {
      return this.weightsLoadPromise;
    }

    if (this.idbFactory === null) {
      this.memoryFallback = this.memoryFallback ?? createInitialClassifierWeights();
      this.weightsLoaded = true;
      return this.memoryFallback;
    }

    // Gate recovery retries while degraded with a cooldown so broken IDB isn't hammered
    if (this.weightsPersistenceDegraded && this.nowFn() < this.nextWeightsRetryAtMs) {
      this.memoryFallback = this.memoryFallback ?? createInitialClassifierWeights();
      return this.memoryFallback;
    }

    const wasDegraded = this.weightsPersistenceDegraded;
    const seqBefore = this.weightsMutationSeq;
    const resetEpochBefore = this.weightsResetEpoch;
    const maxAttempts = wasDegraded ? 1 : 2;

    const loadPromise = (async (): Promise<OnlineClassifierWeights> => {
      let lastErr: unknown;
      for (let attempt = 0; attempt < maxAttempts; attempt++) {
        try {
          const loaded = await this.readWeightsFromIdbOnce();
          // If an explicit reset occurred while reading, keep reset state untouched
          if (this.weightsResetEpoch !== resetEpochBefore && this.memoryFallback !== null) {
            return this.memoryFallback;
          }

          this.weightsPersistenceDegraded = false;
          this.nextWeightsRetryAtMs = 0;
          this.weightsLoaded = true;

          if (wasDegraded && this.degradedWeightsMutated && this.memoryFallback !== null) {
            // Reconcile session mutations accumulated while degraded into recovered disk state
            const merged = mergeDegradedWeightsIntoLoaded(loaded, this.memoryFallback);
            this.degradedWeightsMutated = false;
            const mergedSeq = ++this.weightsMutationSeq;
            this.memoryFallback = merged;
            await this.enqueuePersistWeights(merged, mergedSeq, { forceReplayWrite: true });
            return this.memoryFallback;
          }

          this.degradedWeightsMutated = false;
          if (!wasDegraded && (this.weightsMutationSeq > 0 || this.weightsMutationSeq !== seqBefore) && this.memoryFallback !== null) {
            // Caller already wrote newer weights before/during cold load; keep them while preserving preservedV1Legacy
            return this.memoryFallback;
          }

          this.memoryFallback = loaded;
          return this.memoryFallback;
        } catch (err) {
          lastErr = err;
        }
      }

      // If an explicit reset succeeded while this read was failing, do not re-enter degraded mode
      if (this.weightsResetEpoch !== resetEpochBefore && this.memoryFallback !== null) {
        return this.memoryFallback;
      }

      // CRITICAL: Never treat fallback defaults as persisted state. Keep weightsLoaded = false
      // with a recovery cooldown so writes stay blocked until a real read succeeds, while
      // allowing automatic recovery once IndexedDB becomes available again.
      this.weightsPersistenceDegraded = true;
      this.nextWeightsRetryAtMs = this.nowFn() + this.retryCooldownMs;
      this.lastPersistenceError = `IndexedDB weights read failed: ${errorMessage(lastErr)}`;
      console.warn(
        "[x-reply-filter] IndexedDB weights read failed; disabling persistence writes to protect stored author memory:",
        lastErr,
      );
      if (this.memoryFallback === null) {
        this.memoryFallback = createInitialClassifierWeights();
      }
      return this.memoryFallback;
    })().finally(() => {
      this.weightsLoadPromise = null;
    });

    this.weightsLoadPromise = loadPromise;
    return loadPromise;
  }

  public async setSeedPrototypes(prototypes: ReadonlyArray<SeedPrototype>): Promise<void> {
    const current = await this.getWeights();
    this.weightsMutationSeq += 1;
    this.memoryFallback = {
      ...current,
      seedPrototypes: prototypes,
    };
  }

  private async persistWeightsToIdb(
    weights: OnlineClassifierWeights,
    options?: { readonly forceReplayWrite?: boolean },
  ): Promise<boolean> {
    const db = await this.openDatabase();
    try {
      const defaults = createInitialClassifierWeights();
      const authorStatsEntries = Array.from(weights.authorStats.entries());
      const theta =
        weights.theta && weights.theta.length === COMPACT_V2_DIM
          ? new Float32Array(weights.theta)
          : new Float32Array(defaults.theta0!);
      const lb = weights.langBias ?? defaults.langBias!;
      const langBias: Record<LangBiasKey, number> = {
        en: lb.en,
        "zh-Hans": lb["zh-Hans"],
        "zh-Hant": lb["zh-Hant"],
        ja: lb.ja,
        other: lb.other,
      };

      const v1Payload: StoredWeightsPayloadV1 = {
        weights: this.preservedV1Legacy?.weights ?? weights.weights,
        bias: this.preservedV1Legacy?.bias ?? weights.bias,
        updateCount: weights.updateCount,
        authorStatsEntries,
      };

      const v2Payload: StoredWeightsPayloadV2 = {
        schemaVersion: 2,
        theta,
        langBias,
        theta0Hash: weights.theta0Hash ?? defaults.theta0Hash!,
        updateCount: weights.updateCount,
        rollbackCount: weights.rollbackCount ?? 0,
        authorStatsEntries,
        recordedAutoFlagIds: Array.from(weights.recordedAutoFlagIds ?? []).slice(
          -MAX_AUTO_FLAGGED_TWEET_IDS,
        ),
        undoHistory: (weights.undoHistory ?? []).slice(-MAX_UNDO_HISTORY_ENTRIES),
        weightsRevision: weights.weightsRevision ?? 1,
        ...(weights.dimsReviewedCount !== undefined
          ? { dimsReviewedCount: weights.dimsReviewedCount }
          : {}),
        ...(weights.dimsOverriddenHumanCount !== undefined
          ? { dimsOverriddenHumanCount: weights.dimsOverriddenHumanCount }
          : {}),
      };

      const currentReplay = weights.replayBuffer ?? [];
      const shouldWriteReplay =
        options?.forceReplayWrite === true || currentReplay !== this.lastPersistedReplayRef;
      const replaySlice = shouldWriteReplay
        ? currentReplay.slice(-MAX_REPLAY_BUFFER_ENTRIES)
        : [];

      await new Promise<void>((resolve, reject) => {
        const hasReplayStore = db.objectStoreNames.contains(REPLAY_STORE_NAME);
        const storeNames =
          shouldWriteReplay && hasReplayStore ? [STORE_NAME, REPLAY_STORE_NAME] : [STORE_NAME];
        const tx = db.transaction(storeNames, "readwrite");
        const wStore = tx.objectStore(STORE_NAME);
        // Dual-write primary_v1 (compat, never deleted) AND primary_v2
        wStore.put(v1Payload, WEIGHTS_KEY_V1);
        wStore.put(v2Payload, WEIGHTS_KEY_V2);

        if (shouldWriteReplay && hasReplayStore) {
          const rStore = tx.objectStore(REPLAY_STORE_NAME);
          rStore.clear();
          for (const entry of replaySlice) {
            rStore.put(
              {
                ...entry,
                z: Array.from(entry.z),
              },
              entry.tweetId,
            );
          }
        }

        tx.oncomplete = () => resolve();
        tx.onerror = () => reject(tx.error ?? new Error("IndexedDB write error"));
        tx.onabort = () => reject(tx.error ?? new Error("IndexedDB write aborted"));
      });

      if (shouldWriteReplay) {
        this.lastPersistedReplayRef = currentReplay;
      }
      return true;
    } finally {
      db.close();
    }
  }

  public async saveWeights(weights: OnlineClassifierWeights): Promise<void> {
    const mySeq = ++this.weightsMutationSeq;
    this.memoryFallback = weights;
    if (this.weightsPersistenceDegraded) {
      this.degradedWeightsMutated = true;
    }
    if (this.idbFactory === null) {
      return;
    }

    // Ensure initial load (or post-cooldown recovery) runs first so preservedV1Legacy & degraded guard are accurate
    if (!this.weightsLoaded) {
      if (this.weightsLoadPromise !== null) {
        await this.weightsLoadPromise;
      } else if (!this.weightsPersistenceDegraded || this.nowFn() >= this.nextWeightsRetryAtMs) {
        await this.getWeights();
      }
    }

    // Bail if superseded by a newer mutation, explicit reset, or recovery merge
    if (this.weightsMutationSeq !== mySeq) {
      return;
    }

    if (this.weightsPersistenceDegraded) {
      this.degradedWeightsMutated = true;
      return;
    }

    await this.enqueuePersistWeights(weights, mySeq);
  }

  private async readSettingsFromIdbOnce(): Promise<FilterSettings> {
    const db = await this.openDatabase();
    try {
      const raw = await new Promise<unknown>((resolve, reject) => {
        if (!db.objectStoreNames.contains(SETTINGS_STORE_NAME)) {
          resolve(undefined);
          return;
        }
        const tx = db.transaction(SETTINGS_STORE_NAME, "readonly");
        const store = tx.objectStore(SETTINGS_STORE_NAME);
        const getReq = store.get(SETTINGS_KEY_V2);
        getReq.onsuccess = () => resolve(getReq.result);
        getReq.onerror = () => reject(getReq.error ?? new Error("settings read failed"));
        tx.onerror = () => reject(tx.error ?? new Error("settings tx error"));
        tx.onabort = () => reject(tx.error ?? new Error("settings tx aborted"));
      });
      return parseFilterSettings(raw) ?? DEFAULT_FILTER_SETTINGS;
    } finally {
      db.close();
    }
  }

  private enqueuePersistSettings(settings: FilterSettings, expectedSeq: number): Promise<void> {
    const run = async (): Promise<void> => {
      if (this.settingsMutationSeq !== expectedSeq || this.settingsPersistenceDegraded) {
        return;
      }
      try {
        const db = await this.openDatabase();
        try {
          await new Promise<void>((resolve, reject) => {
            const tx = db.transaction(SETTINGS_STORE_NAME, "readwrite");
            const store = tx.objectStore(SETTINGS_STORE_NAME);
            store.put(settings, SETTINGS_KEY_V2);
            tx.oncomplete = () => resolve();
            tx.onerror = () => reject(tx.error ?? new Error("settings write error"));
            tx.onabort = () => reject(tx.error ?? new Error("IndexedDB settings write aborted"));
          });
        } finally {
          db.close();
        }
      } catch (err) {
        this.lastPersistenceError = `IndexedDB settings write failed: ${errorMessage(err)}`;
        console.warn("[x-reply-filter] IndexedDB settings write failed:", err);
      }
    };

    const next = this.writeChain.then(run, run);
    this.writeChain = next;
    return next;
  }

  public async getSettings(): Promise<FilterSettings> {
    if (this.settingsLoaded && this.settingsFallback !== null) {
      return this.settingsFallback;
    }
    if (this.settingsLoadPromise !== null) {
      return this.settingsLoadPromise;
    }

    if (this.idbFactory === null) {
      this.settingsFallback = this.settingsFallback ?? DEFAULT_FILTER_SETTINGS;
      this.settingsLoaded = true;
      return this.settingsFallback;
    }

    if (this.settingsPersistenceDegraded && this.nowFn() < this.nextSettingsRetryAtMs) {
      this.settingsFallback = this.settingsFallback ?? DEFAULT_FILTER_SETTINGS;
      return this.settingsFallback;
    }

    const wasDegraded = this.settingsPersistenceDegraded;
    const seqBefore = this.settingsMutationSeq;
    const maxAttempts = wasDegraded ? 1 : 2;

    const loadPromise = (async (): Promise<FilterSettings> => {
      let lastErr: unknown;
      for (let attempt = 0; attempt < maxAttempts; attempt++) {
        try {
          const loaded = await this.readSettingsFromIdbOnce();
          this.settingsPersistenceDegraded = false;
          this.nextSettingsRetryAtMs = 0;
          this.settingsLoaded = true;

          if (wasDegraded && this.degradedSettingsMutated && this.settingsFallback !== null) {
            this.degradedSettingsMutated = false;
            if (this.settingsFallback.revision > loaded.revision) {
              const keepSeq = ++this.settingsMutationSeq;
              await this.enqueuePersistSettings(this.settingsFallback, keepSeq);
              return this.settingsFallback;
            }
            this.settingsFallback = loaded;
            return this.settingsFallback;
          }

          this.degradedSettingsMutated = false;
          if (
            !wasDegraded &&
            (this.settingsMutationSeq > 0 || this.settingsMutationSeq !== seqBefore) &&
            this.settingsFallback !== null
          ) {
            return this.settingsFallback;
          }

          this.settingsFallback = loaded;
          return this.settingsFallback;
        } catch (err) {
          lastErr = err;
        }
      }

      // Leave settingsLoaded = false with a cooldown so recovery succeeds once IDB returns
      this.settingsPersistenceDegraded = true;
      this.nextSettingsRetryAtMs = this.nowFn() + this.retryCooldownMs;
      this.lastPersistenceError = `IndexedDB settings read failed: ${errorMessage(lastErr)}`;
      console.warn(
        "[x-reply-filter] IndexedDB settings read failed; keeping defaults in memory without overwriting disk:",
        lastErr,
      );
      if (this.settingsFallback === null) {
        this.settingsFallback = DEFAULT_FILTER_SETTINGS;
      }
      return this.settingsFallback;
    })().finally(() => {
      this.settingsLoadPromise = null;
    });

    this.settingsLoadPromise = loadPromise;
    return loadPromise;
  }

  public async saveSettings(settings: FilterSettings): Promise<void> {
    const mySeq = ++this.settingsMutationSeq;
    this.settingsFallback = settings;
    if (this.settingsPersistenceDegraded) {
      this.degradedSettingsMutated = true;
    }
    if (this.idbFactory === null) {
      this.settingsLoaded = true;
      return;
    }

    if (!this.settingsLoaded) {
      if (this.settingsLoadPromise !== null) {
        await this.settingsLoadPromise;
      } else if (!this.settingsPersistenceDegraded || this.nowFn() >= this.nextSettingsRetryAtMs) {
        await this.getSettings();
      }
    }

    if (this.settingsMutationSeq !== mySeq) {
      return;
    }

    if (this.settingsPersistenceDegraded) {
      this.degradedSettingsMutated = true;
      return;
    }

    await this.enqueuePersistSettings(settings, mySeq);
  }

  public async resetPersistedFeedback(
    resetWeights?: OnlineClassifierWeights,
  ): Promise<OnlineClassifierWeights> {
    const base = createInitialClassifierWeights(
      resetWeights?.seedPrototypes ?? this.memoryFallback?.seedPrototypes,
      new Map(),
    );
    const cleaned: OnlineClassifierWeights = {
      ...base,
      weightsRevision:
        resetWeights?.weightsRevision ?? (this.memoryFallback?.weightsRevision ?? 1) + 1,
      updateCount: 0,
      rollbackCount: 0,
      replayBuffer: [],
      undoHistory: [],
      recordedAutoFlagIds: new Set<TweetId>(),
      dimsReviewedCount: 0,
      dimsOverriddenHumanCount: 0,
    };

    const mySeq = ++this.weightsMutationSeq;
    this.weightsResetEpoch += 1;
    this.memoryFallback = cleaned;
    this.degradedWeightsMutated = false;

    if (this.idbFactory === null) {
      this.weightsLoaded = true;
      return cleaned;
    }

    // Wait for any in-flight cold load so stale reads/saves can never interleave after reset
    if (this.weightsLoadPromise !== null) {
      await this.weightsLoadPromise;
    }
    if (this.weightsMutationSeq !== mySeq) {
      return this.memoryFallback!;
    }

    await this.enqueuePersistWeights(cleaned, mySeq, {
      forceReplayWrite: true,
      allowWhileDegraded: true,
      clearDegradedOnSuccess: true,
    });
    return this.memoryFallback!;
  }
}

export interface OffscreenFilterServiceOptions {
  readonly stage2Judge?: Stage2JudgeService | undefined;
  readonly onStage2Update?: ((msg: Stage2VerdictUpdateMessage) => void | Promise<void>) | undefined;
}

function isHardEvidenceAutoFlag(verdict: VerdictRecord): boolean {
  return (
    verdict.authorId !== undefined &&
    verdict.action === "dim" &&
    (verdict.reason.kind === "keyword_spam" ||
      verdict.reason.kind === "bio_scam" ||
      verdict.reason.kind === "thread_clone")
  );
}

function hasCalibratedSeedPrototypes(weights: OnlineClassifierWeights): boolean {
  if (weights.seedPrototypes.length === 0) {
    return false;
  }
  const first = weights.seedPrototypes[0]!;
  const last = weights.seedPrototypes[weights.seedPrototypes.length - 1]!;
  let normFirst = 0;
  let normLast = 0;
  for (let i = 0; i < first.embedding.length; i++) {
    const a = first.embedding[i] ?? 0;
    const b = last.embedding[i] ?? 0;
    normFirst += a * a;
    normLast += b * b;
  }
  return normFirst > 1e-6 && normLast > 1e-6;
}

export class OffscreenFilterService {
  private readonly engine: ThreadFilterEngine;
  private readonly gpuHost: WebGpuEmbeddingHost;
  private readonly weightStore: IndexedDbWeightStore;
  private readonly stage2Judge: Stage2JudgeService | undefined;
  private onStage2Update: ((msg: Stage2VerdictUpdateMessage) => void | Promise<void>) | undefined;
  private currentSettings: FilterSettings = DEFAULT_FILTER_SETTINGS;
  private settingsLoaded = false;
  private settingsInitPromise: Promise<FilterSettings> | null = null;
  private seedsEmbedded = false;
  private readonly opTextByThread = new Map<ThreadId, string>();

  constructor(
    engine: ThreadFilterEngine,
    gpuHost: WebGpuEmbeddingHost,
    weightStore: IndexedDbWeightStore,
    options?: OffscreenFilterServiceOptions,
  ) {
    this.engine = engine;
    this.gpuHost = gpuHost;
    this.weightStore = weightStore;
    this.stage2Judge = options?.stage2Judge;
    this.onStage2Update = options?.onStage2Update;

    if (this.stage2Judge !== undefined) {
      this.stage2Judge.setOnResult((completion) => this.handleStage2Completion(completion));
    }
  }

  public setOnStage2Update(
    cb: ((msg: Stage2VerdictUpdateMessage) => void | Promise<void>) | undefined,
  ): void {
    this.onStage2Update = cb;
  }

  public getPersistenceHealth(): PersistenceHealth {
    return this.weightStore.getPersistenceHealth();
  }

  public async waitForStage2Idle(): Promise<void> {
    if (this.stage2Judge !== undefined) {
      await this.stage2Judge.waitForIdle();
    }
  }

  private async ensureSettingsLoaded(): Promise<FilterSettings> {
    if (this.settingsLoaded) {
      return this.currentSettings;
    }
    if (this.settingsInitPromise !== null) {
      return this.settingsInitPromise;
    }

    this.settingsInitPromise = this.weightStore
      .getSettings()
      .then((loaded) => {
        if (!this.settingsLoaded) {
          this.currentSettings = loaded;
          this.engine.setSettings(loaded);
          if (!this.weightStore.getPersistenceHealth().settingsDegraded) {
            this.settingsLoaded = true;
          }
        }
        return this.currentSettings;
      })
      .finally(() => {
        this.settingsInitPromise = null;
      });

    return this.settingsInitPromise;
  }

  private async getStage2Status(settings: FilterSettings): Promise<Stage2StatusTag> {
    if (!settings.toggles.enableStage2) {
      return "disabled";
    }
    if (this.stage2Judge === undefined) {
      return "unavailable";
    }
    return this.stage2Judge.getStatus(true);
  }

  private rememberOpText(threadId: ThreadId, rawText: string): void {
    const trimmed = rawText.trim();
    if (trimmed.length === 0) return;
    if (this.opTextByThread.has(threadId)) {
      this.opTextByThread.delete(threadId);
    } else if (this.opTextByThread.size >= MAX_CACHED_THREADS * 2) {
      const oldest = this.opTextByThread.keys().next().value;
      if (oldest !== undefined) {
        this.opTextByThread.delete(oldest);
      }
    }
    this.opTextByThread.set(threadId, trimmed);
  }

  private async ensureLiveSeedPrototypes(weights: OnlineClassifierWeights): Promise<void> {
    if (this.seedsEmbedded || hasCalibratedSeedPrototypes(weights)) {
      this.seedsEmbedded = true;
      return;
    }
    this.seedsEmbedded = true;
    try {
      const texts = SEED_CORPUS.map((s) => s.text);
      const { embeddings256 } = await this.gpuHost.embedTexts(texts);
      const prototypes: SeedPrototype[] = SEED_CORPUS.map((seed, idx) => ({
        lang: seed.lang,
        label: seed.label,
        embedding: embeddings256[idx]!,
      }));
      await this.weightStore.setSeedPrototypes(prototypes);
    } catch {
      this.seedsEmbedded = false;
    }
  }

  private async handleStage2Completion(completion: Stage2Completion): Promise<void> {
    const settings = await this.ensureSettingsLoaded();
    if (!settings.toggles.enableStage2) {
      return;
    }

    const applied = this.engine.applyStage2Verdict(
      completion.task.threadId,
      completion.task.tweetId,
      completion.verdict,
      settings,
    );
    if (applied === null) {
      return;
    }

    const telemetryUpdated = applied.updatedVerdict.explanation?.stage2 === completion.verdict;
    if (!applied.changedTier && !telemetryUpdated) {
      return;
    }

    if (this.onStage2Update !== undefined) {
      const msg: Stage2VerdictUpdateMessage = {
        kind: "stage2_verdict_update",
        threadId: completion.task.threadId,
        tweetId: completion.task.tweetId,
        verdict: applied.updatedVerdict,
        target: "content",
      };
      await this.onStage2Update(msg);
    }
  }

  private enqueueStage2Suspects(
    threadId: ThreadId,
    embedded: ReadonlyArray<EmbeddedTweet>,
    verdicts: ReadonlyArray<VerdictRecord>,
    settings: FilterSettings,
  ): void {
    if (!settings.toggles.enableStage2 || this.stage2Judge === undefined) {
      return;
    }

    const candidateById = new Map<TweetId, CandidateTweet>();
    for (const item of embedded) {
      candidateById.set(item.candidate.tweetId, item.candidate);
    }
    const opText = this.opTextByThread.get(threadId);

    for (const v of verdicts) {
      if (v.action !== "suspect") continue;
      if (v.reason.kind === "user_override") continue;
      if ((v.explanation?.shields.length ?? 0) > 0) continue;

      const cand = candidateById.get(v.tweetId);
      if (!cand || cand.isOp || cand.rawText.trim().length === 0) continue;

      const replyLang = v.explanation?.lang.reply ?? "other";
      const signalSummary = v.explanation?.top
        .slice(0, 3)
        .map((s) => (s.evidence ? `${s.id}(${s.evidence})` : s.id))
        .join(", ");

      this.stage2Judge.enqueue({
        threadId,
        tweetId: v.tweetId,
        replyText: cand.rawText,
        ...(opText !== undefined ? { opText } : {}),
        replyLang,
        ...(signalSummary !== undefined && signalSummary.length > 0 ? { signalSummary } : {}),
      });
    }
  }

  public async handleRequest(req: OffscreenRpcRequestV2): Promise<OffscreenRpcResponse> {
    try {
      const settings = await this.ensureSettingsLoaded();

      if (req.kind === "get_health") {
        const state = this.gpuHost.getState();
        const stage2Status = await this.getStage2Status(settings);
        return {
          kind: "health_ok",
          gpuStatus: state.status,
          deviceEpoch: state.deviceEpoch,
          stage2Status,
        };
      }

      if (req.kind === "get_diagnostics") {
        const state = this.gpuHost.getState();
        const weights = await this.weightStore.getWeights();
        const summary = this.engine.getDiagnosticsSummary(weights, settings);
        const stage2Status = await this.getStage2Status(settings);
        return {
          kind: "diagnostics_ok",
          gpuStatus: state.status,
          deviceEpoch: state.deviceEpoch,
          summary,
          stage2Status,
        };
      }

      if (req.kind === "get_settings") {
        return {
          kind: "settings_ok",
          settings,
        };
      }

      if (req.kind === "update_settings") {
        // Always read live this.currentSettings (never a stale local snapshot across awaits)
        const cur = this.currentSettings;
        if (areFilterSettingsEqual(req.settings, cur)) {
          return {
            kind: "settings_ok",
            settings: cur,
            reevaluatedThreads: 0,
          };
        }
        const nextRevision = Math.max(req.settings.revision, cur.revision + 1);
        const nextSettings: FilterSettings = {
          preset: req.settings.preset,
          toggles: { ...req.settings.toggles },
          revision: nextRevision,
        };
        this.currentSettings = nextSettings;
        this.settingsLoaded = true;
        this.engine.setSettings(nextSettings);
        await this.weightStore.saveSettings(nextSettings);

        if (!this.currentSettings.toggles.enableStage2 && this.stage2Judge !== undefined) {
          this.stage2Judge.clearPending();
        }

        const weights = await this.weightStore.getWeights();
        const latestSettings = this.currentSettings;
        const reevaluated = this.engine.reevaluateAllThreads(weights, latestSettings);
        return {
          kind: "settings_ok",
          settings: latestSettings,
          reevaluatedThreads: reevaluated.size,
        };
      }

      if (req.kind === "record_feedback") {
        const currentWeights = await this.weightStore.getWeights();
        const feedback = this.engine.applyUserFeedback(
          req.threadId,
          req.tweetId,
          req.label,
          currentWeights,
          settings,
        );
        if (!feedback) {
          return {
            kind: "record_feedback_ok",
            updatedVerdict: null,
            retroactiveUpdates: [],
            updateCount: currentWeights.updateCount,
            rolledBack: false,
            rollbackCount: currentWeights.rollbackCount ?? 0,
          };
        }
        await this.weightStore.saveWeights(feedback.nextWeights);
        return {
          kind: "record_feedback_ok",
          updatedVerdict: feedback.updatedVerdict,
          retroactiveUpdates: feedback.retroactiveUpdates,
          updateCount: feedback.nextWeights.updateCount,
          rolledBack: feedback.rolledBack ?? false,
          rollbackCount: feedback.nextWeights.rollbackCount ?? 0,
        };
      }

      if (req.kind === "undo_feedback") {
        const currentWeights = await this.weightStore.getWeights();
        const undone = this.engine.undoLastFeedback(currentWeights, settings);
        if (!undone) {
          return {
            kind: "undo_feedback_ok",
            undone: false,
            restoredVerdict: null,
            updateCount: currentWeights.updateCount,
          };
        }
        await this.weightStore.saveWeights(undone.nextWeights);
        this.engine.reevaluateAllThreads(undone.nextWeights, settings);
        return {
          kind: "undo_feedback_ok",
          undone: true,
          restoredVerdict: undone.restoredVerdict,
          undoneTweetId: undone.undoneTweetId,
          updateCount: undone.nextWeights.updateCount,
        };
      }

      if (req.kind === "export_feedback") {
        const currentWeights = await this.weightStore.getWeights();
        return {
          kind: "export_feedback_ok",
          payloadJson: this.engine.exportFeedbackJson(currentWeights),
        };
      }

      if (req.kind === "import_feedback") {
        const currentWeights = await this.weightStore.getWeights();
        const imported = this.engine.importFeedbackJson(req.payloadJson, currentWeights);
        if (imported === null) {
          return {
            kind: "rpc_error",
            message: "Invalid feedback export JSON payload",
          };
        }
        await this.weightStore.saveWeights(imported.nextWeights);
        this.engine.reevaluateAllThreads(imported.nextWeights, settings);
        return {
          kind: "import_feedback_ok",
          importedReplayCount: imported.importedReplayCount,
          importedAuthorCount: imported.importedAuthorCount,
          rolledBack: imported.rolledBack,
        };
      }

      if (req.kind === "reset_feedback") {
        const currentWeights = await this.weightStore.getWeights();
        const resetWeights = this.engine.resetFeedback(currentWeights);
        const persisted = await this.weightStore.resetPersistedFeedback(resetWeights);
        this.engine.reevaluateAllThreads(persisted, settings);
        return {
          kind: "reset_feedback_ok",
          updateCount: persisted.updateCount,
        };
      }

      if (req.kind === "trigger_stage2_download") {
        if (this.stage2Judge === undefined) {
          return {
            kind: "stage2_status_ok",
            status: "unavailable",
          };
        }
        const result =
          req.action === "status"
            ? await this.stage2Judge.checkAvailability("en")
            : await this.stage2Judge.triggerDownload("en");
        return {
          kind: "stage2_status_ok",
          status: result.status,
          ...(result.downloadProgress !== undefined
            ? { downloadProgress: result.downloadProgress }
            : {}),
        };
      }

      // req.kind === "evaluate_batch"
      for (const cand of req.candidates) {
        if (cand.isOp && cand.rawText.trim().length > 0) {
          this.rememberOpText(req.threadId, cand.rawText);
        }
      }

      let weights = await this.weightStore.getWeights();
      const { cached, toEmbed } = this.engine.filterUnseenCandidates(
        req.threadId,
        req.candidates,
        weights,
        settings,
        req.focalAuthorId,
      );

      if (toEmbed.length === 0) {
        return {
          kind: "evaluate_batch_ok",
          result: {
            threadId: req.threadId,
            anchorKind: this.engine.getAnchorKind(req.threadId),
            verdicts: cached,
            retroactiveUpdates: [],
          },
        };
      }

      await this.ensureLiveSeedPrototypes(weights);
      const embedded = await this.gpuHost.embedCandidates(toEmbed);
      // Re-read both weights and currentSettings after GPU await so concurrent updates take effect
      weights = await this.weightStore.getWeights();
      const activeSettings = this.currentSettings;
      const fresh = this.engine.ingestEmbeddedBatch(
        req.threadId,
        embedded,
        weights,
        activeSettings,
        undefined,
        req.focalAuthorId,
      );

      // Auto-flag hygiene: ONLY record auto_flag when learnFromFeedback is enabled (spec §4d.5)
      // and ONLY for hard evidence (keyword_spam, bio_scam, thread_clone + retroactive), deduplicated per tweetId.
      if (activeSettings.toggles.learnFromFeedback) {
        let authorStatsChanged = false;
        for (const v of [...fresh.verdicts, ...fresh.retroactiveUpdates]) {
          if (isHardEvidenceAutoFlag(v)) {
            const next = recordAuthorOutcome(weights, v.authorId!, "auto_flag", {
              tweetId: v.tweetId,
            });
            if (next !== weights) {
              weights = next;
              authorStatsChanged = true;
            }
          }
        }
        if (authorStatsChanged) {
          await this.weightStore.saveWeights(weights);
        }
      }

      const response: OffscreenRpcResponse = {
        kind: "evaluate_batch_ok",
        result: {
          threadId: req.threadId,
          anchorKind: fresh.anchorKind,
          verdicts: [...cached, ...fresh.verdicts],
          retroactiveUpdates: fresh.retroactiveUpdates,
        },
      };

      // Enqueue Stage-2 suspects asynchronously AFTER constructing Stage-1 response (non-blocking)
      this.enqueueStage2Suspects(req.threadId, embedded, fresh.verdicts, activeSettings);

      return response;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      return {
        kind: "rpc_error",
        message,
      };
    }
  }
}
