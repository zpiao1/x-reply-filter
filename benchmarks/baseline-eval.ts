import { ThreadFilterEngine } from "../src/domain/engine.ts";
import {
  SEED_CORPUS,
  type SeedPrototype,
  createInitialClassifierWeights,
  extractDomSignalsFromText,
} from "../src/domain/seeds.ts";
import {
  type AuthorId,
  type CandidateTweet,
  type EmbeddedTweet,
  type ThreadId,
  type TweetId,
  asTokenCount,
  truncateAndNormalize256,
} from "../src/domain/types.ts";
import type { BenchmarkCase } from "./dataset-humans.ts";
import { type EvalMetrics, computeMetricsFromVerdicts } from "./cv-engine.ts";
import { estimateContentTokenCount } from "./features-v2.ts";

export interface BaselineEvalResult {
  readonly metrics: EvalMetrics;
  readonly byLang: Record<string, EvalMetrics>;
  readonly byReason: Record<string, { bot: number; human: number }>;
  readonly verdicts: Array<{
    readonly id: string;
    readonly tier: "dim" | "suspect" | "keep";
    readonly probability: number;
    readonly reasonKind: string;
  }>;
}

const EMOJI_STRIP_RE = /\p{Extended_Pictographic}|\p{Regional_Indicator}/gu;

/**
 * Evaluates the EXACT unmodified baseline in src/domain/engine.ts & src/domain/seeds.ts
 * using live EmbeddingGemma-300M (q8, 256d) embeddings.
 * When simulateTwemojiBug = true, reproduces the live DOM bug where textContent drops Twemoji <img>.
 */
export function evaluateBaselineEngine(args: {
  readonly cases: ReadonlyArray<BenchmarkCase>;
  readonly seedEmbeddings: ReadonlyArray<Float32Array>;
  readonly opEmbeddings: ReadonlyArray<Float32Array>;
  readonly replyEmbeddings: ReadonlyArray<Float32Array>;
  readonly replyEmbeddingsNoEmoji?: ReadonlyArray<Float32Array>;
  readonly simulateTwemojiBug: boolean;
}): BaselineEvalResult {
  const { cases, seedEmbeddings, opEmbeddings, replyEmbeddings, simulateTwemojiBug } = args;

  const seedPrototypes: SeedPrototype[] = SEED_CORPUS.map((s, idx) => ({
    lang: s.lang,
    label: s.label,
    embedding: truncateAndNormalize256(seedEmbeddings[idx]!, 0, seedEmbeddings[idx]!.length),
  }));

  const classifier = createInitialClassifierWeights(seedPrototypes, new Map());

  // Group cases by threadId to faithfully test ThreadFilterEngine including thread_clone
  const threadGroups = new Map<string, number[]>();
  for (let i = 0; i < cases.length; i++) {
    const tid = cases[i]!.threadId;
    const list = threadGroups.get(tid) ?? [];
    list.push(i);
    threadGroups.set(tid, list);
  }

  const verdictByIndex = new Array<{
    readonly id: string;
    readonly tier: "dim" | "suspect" | "keep";
    readonly probability: number;
    readonly reasonKind: string;
  }>(cases.length);

  for (const [threadId, indices] of threadGroups.entries()) {
    const engine = new ThreadFilterEngine();
    const first = cases[indices[0]!]!;
    const opEmb256 = truncateAndNormalize256(
      opEmbeddings[indices[0]!]!,
      0,
      opEmbeddings[indices[0]!]!.length,
    );
    const opCandidate: CandidateTweet = {
      tweetId: threadId as TweetId,
      authorId: first.opAuthorId as AuthorId,
      rawText: first.opText,
      normalizedText: first.opText,
      isOp: true,
      domSignals: extractDomSignalsFromText(first.opText, false, false),
    };
    const batch: EmbeddedTweet[] = [
      {
        candidate: opCandidate,
        embedding: opEmb256,
        tokenCount: asTokenCount(estimateContentTokenCount(first.opText)),
      },
    ];

    for (let k = 0; k < indices.length; k++) {
      const idx = indices[k]!;
      const c = cases[idx]!;
      const text = simulateTwemojiBug
        ? c.replyText.replace(EMOJI_STRIP_RE, "").trim()
        : c.replyText.trim();
      const rawVec =
        simulateTwemojiBug && args.replyEmbeddingsNoEmoji
          ? args.replyEmbeddingsNoEmoji[idx]!
          : replyEmbeddings[idx]!;
      const emb256 = truncateAndNormalize256(rawVec, 0, rawVec.length);
      const replyTweetId = `${threadId}${String(k + 1).padStart(2, "0")}` as TweetId;
      const domSignals = extractDomSignalsFromText(
        text,
        c.domMeta.hasVerifiedBadge,
        c.domMeta.hasPrimaryMedia || c.domMeta.hasExternalLink || c.domMeta.hasQuote,
      );
      batch.push({
        candidate: {
          tweetId: replyTweetId,
          authorId: c.replyAuthorId as AuthorId,
          rawText: text,
          normalizedText: text,
          isOp: false,
          domSignals,
        },
        embedding: emb256,
        tokenCount: asTokenCount(estimateContentTokenCount(text)),
      });
    }

    const res = engine.ingestEmbeddedBatch(threadId as ThreadId, batch, classifier);
    // Apply retroactiveUpdates over initial verdicts
    const finalByTweet = new Map(res.verdicts.map((v) => [v.tweetId, v]));
    for (const retro of res.retroactiveUpdates) {
      finalByTweet.set(retro.tweetId, retro);
    }

    for (let k = 0; k < indices.length; k++) {
      const idx = indices[k]!;
      const replyTweetId = `${threadId}${String(k + 1).padStart(2, "0")}` as TweetId;
      const v = finalByTweet.get(replyTweetId)!;
      verdictByIndex[idx] = {
        id: cases[idx]!.id,
        tier: v.action === "dim" ? "dim" : "keep",
        probability: v.diagnostics?.siameseProbability ?? v.score,
        reasonKind: v.reason.kind,
      };
    }
  }

  const labels = cases.map((c) => (c.expectedLabel === "bot" ? 1 : 0));
  const metrics = computeMetricsFromVerdicts(verdictByIndex, labels);

  const byLang: Record<string, EvalMetrics> = {};
  for (const lang of ["en", "zh-hans", "zh-hant", "ja"] as const) {
    const subV: typeof verdictByIndex = [];
    const subY: number[] = [];
    for (let i = 0; i < cases.length; i++) {
      if (cases[i]!.lang === lang) {
        subV.push(verdictByIndex[i]!);
        subY.push(labels[i]!);
      }
    }
    byLang[lang] = computeMetricsFromVerdicts(subV, subY);
  }

  const byReason: Record<string, { bot: number; human: number }> = {};
  for (let i = 0; i < cases.length; i++) {
    const rk = verdictByIndex[i]!.reasonKind;
    const entry = byReason[rk] ?? { bot: 0, human: 0 };
    if (labels[i] === 1) entry.bot += 1;
    else entry.human += 1;
    byReason[rk] = entry;
  }

  return { metrics, byLang, byReason, verdicts: verdictByIndex };
}
