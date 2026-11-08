import type { ThreadFilterEngine } from "../domain/engine.ts";
import {
  type OnlineClassifierWeights,
  SEED_CORPUS,
  type SeedPrototype,
  createInitialClassifierWeights,
  recordAuthorOutcome,
} from "../domain/seeds.ts";
import {
  type AuthorId,
  type AuthorStats,
  CLASSIFICATION_TASK_PREFIX,
  type CandidateTweet,
  type EmbeddedTweet,
  OFFSCREEN_IDLE_TIMEOUT_MS,
  RAW_EMBEDDING_DIM,
  SIAMESE_TOTAL_DIM,
  type TokenCount,
  type UnitVec256,
  asTokenCount,
  formatClassificationPrompt,
  parseAuthorId,
  truncateAndNormalizeBatch256,
} from "../domain/types.ts";
import type { OffscreenRpcRequest, OffscreenRpcResponse } from "../shared/rpc.ts";

export const MODEL_ID = "onnx-community/embeddinggemma-300m-ONNX" as const;
export const MODEL_DTYPE = "q8" as const;

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

  constructor(
    backendFactory: () => Promise<EmbeddingBackend> = createTransformersWebGpuBackend,
    nowFn: () => number = () => Date.now(),
    idleTimeoutMs: number = OFFSCREEN_IDLE_TIMEOUT_MS,
  ) {
    this.backendFactory = backendFactory;
    this.nowFn = nowFn;
    this.idleTimeoutMs = idleTimeoutMs;
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

  public async embedTexts(rawTexts: ReadonlyArray<string>): Promise<RawBatchInferenceOutput> {
    if (rawTexts.length === 0) {
      return { embeddings256: [], contentTokenCounts: [] };
    }
    const prompts = rawTexts.map((t) => formatClassificationPrompt(t));
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

const DB_NAME = "x-reply-filter-db";
const STORE_NAME = "classifier_weights";
const WEIGHTS_KEY = "primary_v1";

interface StoredWeightsPayload {
  readonly weights: Float32Array;
  readonly bias: number;
  readonly updateCount: number;
  readonly authorStatsEntries?: ReadonlyArray<readonly [string, AuthorStats]>;
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
        spamCount: stats.spamCount,
        humanCount: stats.humanCount ?? 0,
        autoFlagCount: stats.autoFlagCount ?? 0,
      });
    }
  }
  return map;
}

export class IndexedDbWeightStore {
  private memoryFallback: OnlineClassifierWeights | null = null;
  private readonly dbName: string;

  constructor(dbName: string = DB_NAME) {
    this.dbName = dbName;
  }

  private openDatabase(): Promise<IDBDatabase> {
    return new Promise((resolve, reject) => {
      const req = indexedDB.open(this.dbName, 1);
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains(STORE_NAME)) {
          db.createObjectStore(STORE_NAME);
        }
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  }

  public async getWeights(): Promise<OnlineClassifierWeights> {
    if (this.memoryFallback !== null) {
      return this.memoryFallback;
    }

    if (typeof indexedDB === "undefined") {
      this.memoryFallback = createInitialClassifierWeights();
      return this.memoryFallback;
    }

    try {
      const db = await this.openDatabase();
      const payload = await new Promise<StoredWeightsPayload | undefined>((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, "readonly");
        const store = tx.objectStore(STORE_NAME);
        const getReq = store.get(WEIGHTS_KEY);
        getReq.onsuccess = () => resolve(getReq.result as StoredWeightsPayload | undefined);
        getReq.onerror = () => reject(getReq.error);
      });
      db.close();

      const defaults = createInitialClassifierWeights();
      if (payload && payload.weights instanceof Float32Array) {
        const migratedWeights = new Float32Array(defaults.weights);
        const copyLen = Math.min(payload.weights.length, SIAMESE_TOTAL_DIM);
        for (let i = 0; i < copyLen; i++) {
          migratedWeights[i] = payload.weights[i]!;
        }
        this.memoryFallback = {
          weights: migratedWeights,
          bias: payload.bias,
          seedPrototypes: defaults.seedPrototypes,
          updateCount: payload.updateCount,
          authorStats: deserializeAuthorStats(payload.authorStatsEntries),
        };
      } else {
        this.memoryFallback = defaults;
      }
      return this.memoryFallback;
    } catch {
      this.memoryFallback = createInitialClassifierWeights();
      return this.memoryFallback;
    }
  }

  public async setSeedPrototypes(prototypes: ReadonlyArray<SeedPrototype>): Promise<void> {
    const current = await this.getWeights();
    this.memoryFallback = {
      ...current,
      seedPrototypes: prototypes,
    };
  }

  public async saveWeights(weights: OnlineClassifierWeights): Promise<void> {
    this.memoryFallback = weights;
    if (typeof indexedDB === "undefined") {
      return;
    }

    try {
      const db = await this.openDatabase();
      await new Promise<void>((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, "readwrite");
        const store = tx.objectStore(STORE_NAME);
        const payload: StoredWeightsPayload = {
          weights: weights.weights,
          bias: weights.bias,
          updateCount: weights.updateCount,
          authorStatsEntries: Array.from(weights.authorStats.entries()),
        };
        const putReq = store.put(payload, WEIGHTS_KEY);
        putReq.onsuccess = () => resolve();
        putReq.onerror = () => reject(putReq.error);
      });
      db.close();
    } catch {}
  }
}

export class OffscreenFilterService {
  private readonly engine: ThreadFilterEngine;
  private readonly gpuHost: WebGpuEmbeddingHost;
  private readonly weightStore: IndexedDbWeightStore;
  private seedsEmbedded = false;

  constructor(
    engine: ThreadFilterEngine,
    gpuHost: WebGpuEmbeddingHost,
    weightStore: IndexedDbWeightStore,
  ) {
    this.engine = engine;
    this.gpuHost = gpuHost;
    this.weightStore = weightStore;
  }

  private async ensureLiveSeedPrototypes(): Promise<void> {
    if (this.seedsEmbedded) {
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

  public async handleRequest(req: OffscreenRpcRequest): Promise<OffscreenRpcResponse> {
    try {
      if (req.kind === "get_health") {
        const state = this.gpuHost.getState();
        return {
          kind: "health_ok",
          gpuStatus: state.status,
          deviceEpoch: state.deviceEpoch,
        };
      }

      if (req.kind === "get_diagnostics") {
        const state = this.gpuHost.getState();
        const weights = await this.weightStore.getWeights();
        const summary = this.engine.getDiagnosticsSummary(weights);
        return {
          kind: "diagnostics_ok",
          gpuStatus: state.status,
          deviceEpoch: state.deviceEpoch,
          summary,
        };
      }

      if (req.kind === "record_feedback") {
        const currentWeights = await this.weightStore.getWeights();
        const feedback = this.engine.applyUserFeedback(
          req.threadId,
          req.tweetId,
          req.label,
          currentWeights,
        );
        if (!feedback) {
          return {
            kind: "record_feedback_ok",
            updatedVerdict: null,
            retroactiveUpdates: [],
            updateCount: currentWeights.updateCount,
          };
        }
        await this.weightStore.saveWeights(feedback.nextWeights);
        return {
          kind: "record_feedback_ok",
          updatedVerdict: feedback.updatedVerdict,
          retroactiveUpdates: feedback.retroactiveUpdates,
          updateCount: feedback.nextWeights.updateCount,
        };
      }

      const { cached, toEmbed } = this.engine.filterUnseenCandidates(
        req.threadId,
        req.candidates,
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

      await this.ensureLiveSeedPrototypes();
      const embedded = await this.gpuHost.embedCandidates(toEmbed);
      let weights = await this.weightStore.getWeights();
      const fresh = this.engine.ingestEmbeddedBatch(req.threadId, embedded, weights);

      let authorStatsChanged = false;
      for (const v of fresh.verdicts) {
        if (
          v.authorId !== undefined &&
          (v.reason.kind === "keyword_spam" ||
            v.reason.kind === "thread_clone" ||
            v.reason.kind === "op_paraphrase")
        ) {
          weights = recordAuthorOutcome(weights, v.authorId, "auto_flag");
          authorStatsChanged = true;
        }
      }
      if (authorStatsChanged) {
        await this.weightStore.saveWeights(weights);
      }

      return {
        kind: "evaluate_batch_ok",
        result: {
          threadId: req.threadId,
          anchorKind: fresh.anchorKind,
          verdicts: [...cached, ...fresh.verdicts],
          retroactiveUpdates: fresh.retroactiveUpdates,
        },
      };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      return {
        kind: "rpc_error",
        message,
      };
    }
  }
}
