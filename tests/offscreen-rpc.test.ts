import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { forwardRpcToOffscreen } from "../src/background/index.ts";
import { ThreadFilterEngine } from "../src/domain/engine.ts";
import { extractDomSignalsFromText } from "../src/domain/seeds.ts";
import {
  type CandidateTweet,
  EMBEDDING_DIM,
  type ThreadId,
  type TweetId,
  asTokenCount,
  parseAuthorId,
  parseTweetId,
  truncateAndNormalize256,
} from "../src/domain/types.ts";
import {
  type EmbeddingBackend,
  IndexedDbWeightStore,
  OffscreenFilterService,
  WebGpuEmbeddingHost,
} from "../src/offscreen/gpu-host.ts";
import { ensureOffscreenDocument, parseRpcRequest } from "../src/shared/rpc.ts";

function makeCandidate(id: string, author: string, text: string, isOp = false): CandidateTweet {
  return {
    tweetId: parseTweetId(id)!,
    authorId: parseAuthorId(author)!,
    rawText: text,
    normalizedText: text.trim(),
    isOp,
    domSignals: extractDomSignalsFromText(text, false, false),
  };
}

function createMockBackend(options?: { failFirstInfer?: boolean }): {
  readonly backendFactory: () => Promise<EmbeddingBackend>;
  readonly getInferCalls: () => number;
  readonly getDisposeCalls: () => number;
} {
  let inferCalls = 0;
  let disposeCalls = 0;
  let shouldFailOnce = options?.failFirstInfer ?? false;

  const backendFactory = async (): Promise<EmbeddingBackend> => ({
    async inferPrompts(prompts, rawTexts) {
      inferCalls += 1;
      if (shouldFailOnce) {
        shouldFailOnce = false;
        throw new Error("Simulated WebGPU device.lost");
      }
      const embeddings256 = prompts.map((_, idx) => {
        const raw = new Float32Array(EMBEDDING_DIM);
        raw[idx % EMBEDDING_DIM] = 1;
        return truncateAndNormalize256(raw, 0, EMBEDDING_DIM);
      });
      const contentTokenCounts = rawTexts.map((t) =>
        asTokenCount(Math.max(1, Array.from(t.trim()).length)),
      );
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
  };
}

describe("RPC boundary parsing and single-flight offscreen creation", () => {
  it("validates and brands RPC payloads at the boundary", () => {
    const validHealth = parseRpcRequest({ kind: "get_health", target: "offscreen" });
    assert.deepEqual(validHealth, { kind: "get_health", target: "offscreen" });

    const validFeedback = parseRpcRequest({
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
      parseRpcRequest({
        kind: "record_feedback",
        threadId: "bad-thread",
        tweetId: "998878",
        label: "spam",
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

describe("WebGpuEmbeddingHost lifecycle and OffscreenFilterService routing", () => {
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

  it("routes evaluate_batch and record_feedback through background and offscreen service", async () => {
    const mock = createMockBackend();
    const engine = new ThreadFilterEngine();
    const gpuHost = new WebGpuEmbeddingHost(mock.backendFactory);
    const weightStore = new IndexedDbWeightStore();
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

    const op = makeCandidate("7000", "op_user", "Full end to end RPC test on thread", true);
    const reply = makeCandidate("7001", "replier", "Look at my pinned profile link in bio!", false);

    const batchRes = await forwardRpcToOffscreen(
      {
        kind: "evaluate_batch",
        threadId: "7000" as ThreadId,
        candidates: [op, reply],
      },
      mockOffscreenApi,
      (req) => service.handleRequest(req),
    );

    assert(batchRes !== null && batchRes.kind === "evaluate_batch_ok");
    assert.equal(batchRes.result.verdicts.length, 1);
    assert.equal(batchRes.result.verdicts[0]?.action, "dim");
    const inferCountAfterFirstBatch = mock.getInferCalls();

    const remountRes = await forwardRpcToOffscreen(
      {
        kind: "evaluate_batch",
        threadId: "7000" as ThreadId,
        candidates: [op, reply],
      },
      mockOffscreenApi,
      (req) => service.handleRequest(req),
    );
    assert(remountRes !== null && remountRes.kind === "evaluate_batch_ok");
    assert.equal(
      mock.getInferCalls(),
      inferCountAfterFirstBatch,
      "remounted batch hits engine cache without touching GPU",
    );

    const feedbackRes = await forwardRpcToOffscreen(
      {
        kind: "record_feedback",
        threadId: "7000" as ThreadId,
        tweetId: "7001" as TweetId,
        label: "human",
      },
      mockOffscreenApi,
      (req) => service.handleRequest(req),
    );
    assert(feedbackRes !== null && feedbackRes.kind === "record_feedback_ok");
    assert.equal(feedbackRes.updatedVerdict?.action, "keep");
    assert.equal(feedbackRes.updateCount, 1);

    const diagRes = await forwardRpcToOffscreen(
      {
        kind: "get_diagnostics",
      },
      mockOffscreenApi,
      (req) => service.handleRequest(req),
    );
    assert(diagRes !== null && diagRes.kind === "diagnostics_ok");
    assert.equal(diagRes.summary.totalScoredReplies, 1);
    assert.equal(diagRes.summary.updateCount, 1);
    assert.equal(diagRes.summary.topFlaggedAuthors.length, 1);
    assert.equal(diagRes.summary.topFlaggedAuthors[0]?.authorId, "replier");
  });
});
