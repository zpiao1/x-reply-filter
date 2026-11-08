import {
  type OnlineClassifierWeights,
  detectDominantLanguage,
  recordAuthorOutcome,
} from "./seeds.ts";
import {
  type AuthorId,
  type AuthorStats,
  type CandidateTweet,
  DEFAULT_SLAB_CAPACITY,
  DEFAULT_THRESHOLDS,
  type DomSignals,
  EMBEDDING_DIM,
  type EmbeddedTweet,
  type FilterReason,
  MAX_CACHED_THREADS,
  type Probability,
  SIAMESE_RELATIONAL_DIM,
  SIAMESE_TOTAL_DIM,
  type SiameseFeatureVec,
  type SupportedLanguage,
  type ThreadAnchorState,
  type ThreadId,
  type ThresholdConfig,
  type TokenCount,
  type TweetId,
  type UnitCosine,
  type UnitVec256,
  type VerdictDiagnostics,
  type VerdictRecord,
  asProbability,
  asUnitCosine,
  computeAuthorSpamPrior,
} from "./types.ts";

export interface BatchEvaluationResult {
  readonly threadId: ThreadId;
  readonly anchorKind: ThreadAnchorState["kind"];
  readonly verdicts: ReadonlyArray<VerdictRecord>;
  readonly retroactiveUpdates: ReadonlyArray<VerdictRecord>;
}

export interface ReplyEvaluationContext {
  readonly candidate: CandidateTweet;
  readonly tokenCount: TokenCount;
  readonly anchor: ThreadAnchorState;
  readonly opCosSim: UnitCosine;
  readonly maxCloneSim: UnitCosine;
  readonly canonicalCloneTweetId: TweetId | null;
  readonly cloneClusterSize: number;
  readonly siameseProbability: Probability;
  readonly dominantLanguage: SupportedLanguage;
  readonly authorStats: AuthorStats | undefined;
  readonly authorSpamPrior: number;
}

export interface VerdictRule {
  readonly id: FilterReason["kind"];
  readonly match: (ctx: ReplyEvaluationContext, cfg: ThresholdConfig) => FilterReason | null;
}

export const VERDICT_RULE_TABLE: ReadonlyArray<VerdictRule> = [
  {
    id: "known_bot_author",
    match: (ctx, cfg) =>
      ctx.authorStats !== undefined &&
      ctx.authorStats.spamCount >= cfg.authorKnownBotMinSpam &&
      ctx.authorStats.spamCount > ctx.authorStats.humanCount * 2
        ? {
            kind: "known_bot_author",
            action: "dim",
            authorId: ctx.candidate.authorId,
            spamCount: ctx.authorStats.spamCount,
          }
        : null,
  },
  {
    id: "thread_clone",
    match: (ctx, cfg) =>
      ctx.maxCloneSim >= cfg.cloneCosineMin && ctx.canonicalCloneTweetId !== null
        ? {
            kind: "thread_clone",
            action: "dim",
            similarityToPeer: ctx.maxCloneSim,
            canonicalTweetId: ctx.canonicalCloneTweetId,
            clusterSize: ctx.cloneClusterSize,
          }
        : null,
  },
  {
    id: "keyword_spam",
    match: (ctx) =>
      ctx.candidate.domSignals.matchedSpamKeyword !== null
        ? {
            kind: "keyword_spam",
            action: "dim",
            matchedPattern: ctx.candidate.domSignals.matchedSpamKeyword,
          }
        : null,
  },
  {
    id: "short_human_exempt",
    match: (ctx, cfg) =>
      ctx.tokenCount < cfg.minReplyTokens &&
      ctx.authorSpamPrior <= 0 &&
      !(ctx.candidate.domSignals.hasVerifiedBadge && ctx.candidate.domSignals.emojiToCharRatio >= 0.8) &&
      ctx.siameseProbability < cfg.siameseScoreThreshold
        ? {
            kind: "short_human_exempt",
            action: "keep",
            tokenCount: ctx.tokenCount,
          }
        : null,
  },
  {
    id: "op_paraphrase",
    match: (ctx, cfg) =>
      ctx.anchor.kind === "text_anchored" &&
      ctx.candidate.authorId !== ctx.anchor.authorId &&
      ctx.opCosSim >= cfg.paraphraseCosineMin &&
      ctx.opCosSim < cfg.exactQuoteCeiling
        ? {
            kind: "op_paraphrase",
            action: "dim",
            similarityToOp: ctx.opCosSim,
          }
        : null,
  },
  {
    id: "siamese_bot",
    match: (ctx, cfg) =>
      ctx.siameseProbability >= cfg.siameseScoreThreshold
        ? {
            kind: "siamese_bot",
            action: "dim",
            probability: ctx.siameseProbability,
            dominantLanguage: ctx.dominantLanguage,
            opFallback: ctx.anchor.kind !== "text_anchored",
          }
        : null,
  },
];

interface ThreadSlabState {
  readonly threadId: ThreadId;
  anchor: ThreadAnchorState;
  readonly maxCapacity: number;
  count: number;
  readonly replyMatrix: Float32Array;
  readonly replyTweetIds: TweetId[];
  readonly replyAuthorIds: AuthorId[];
  readonly replySnippets: string[];
  readonly replyTokenCounts: Uint16Array;
  readonly cloneClusterCounts: Uint16Array;
  readonly slotByTweetId: Map<TweetId, number>;
  readonly verdictsByTweetId: Map<TweetId, VerdictRecord>;
  readonly featuresByTweetId: Map<TweetId, SiameseFeatureVec>;
}

export interface AuthorReputationEntry {
  readonly authorId: AuthorId;
  readonly spamCount: number;
  readonly humanCount: number;
  readonly autoFlagCount: number;
  readonly prior: number;
}

export interface DiagnosticsSummary {
  readonly cachedThreadCount: number;
  readonly totalScoredReplies: number;
  readonly dimmedReplies: number;
  readonly keptReplies: number;
  readonly updateCount: number;
  readonly countsByReason: Readonly<Record<string, number>>;
  readonly topFlaggedAuthors: ReadonlyArray<AuthorReputationEntry>;
  readonly recentVerdicts: ReadonlyArray<VerdictRecord>;
}

function createSlab(threadId: ThreadId, maxCapacity: number): ThreadSlabState {
  return {
    threadId,
    anchor: { kind: "unanchored" },
    maxCapacity,
    count: 0,
    replyMatrix: new Float32Array(maxCapacity * EMBEDDING_DIM),
    replyTweetIds: [],
    replyAuthorIds: [],
    replySnippets: [],
    replyTokenCounts: new Uint16Array(maxCapacity),
    cloneClusterCounts: new Uint16Array(maxCapacity),
    slotByTweetId: new Map(),
    verdictsByTweetId: new Map(),
    featuresByTweetId: new Map(),
  };
}

export function dotProduct256(a: UnitVec256, b: UnitVec256): UnitCosine {
  let sum = 0;
  for (let i = 0; i < EMBEDDING_DIM; i++) {
    sum += (a[i] ?? 0) * (b[i] ?? 0);
  }
  return asUnitCosine(sum);
}

export function findMaxCloneSimilarity(
  replyVec: UnitVec256,
  replyAuthorId: AuthorId,
  anchorAuthorId: AuthorId | null,
  slabMatrix: Float32Array,
  slabAuthorIds: ReadonlyArray<AuthorId>,
  priorCount: number,
): { readonly maxSim: UnitCosine; readonly matchSlot: number } {
  let bestSim = -1;
  let bestSlot = -1;

  for (let slot = 0; slot < priorCount; slot++) {
    const slotAuthor = slabAuthorIds[slot];
    if (slotAuthor === replyAuthorId || (anchorAuthorId !== null && slotAuthor === anchorAuthorId)) {
      continue;
    }
    const offset = slot * EMBEDDING_DIM;
    let sum = 0;
    for (let i = 0; i < EMBEDDING_DIM; i++) {
      sum += (slabMatrix[offset + i] ?? 0) * (replyVec[i] ?? 0);
    }
    if (sum > bestSim) {
      bestSim = sum;
      bestSlot = slot;
    }
  }

  return {
    maxSim: asUnitCosine(bestSim < 0 ? 0 : bestSim),
    matchSlot: bestSlot,
  };
}

export function buildSiameseFeatures(
  replyVec: UnitVec256,
  anchor: ThreadAnchorState,
  signals: DomSignals,
): SiameseFeatureVec {
  const out = new Float32Array(SIAMESE_TOTAL_DIM);
  out.set(replyVec, 0);

  if (anchor.kind === "text_anchored") {
    const opVec = anchor.embedding;
    for (let i = 0; i < EMBEDDING_DIM; i++) {
      const u = opVec[i] ?? 0;
      const v = replyVec[i] ?? 0;
      out[EMBEDDING_DIM + i] = Math.abs(u - v);
      out[EMBEDDING_DIM * 2 + i] = u * v;
    }
  }

  out[SIAMESE_RELATIONAL_DIM + 0] = signals.hasVerifiedBadge ? 1 : 0;
  out[SIAMESE_RELATIONAL_DIM + 1] = signals.hasMediaOrExternalLink ? 1 : 0;
  out[SIAMESE_RELATIONAL_DIM + 2] = signals.endsWithBaitQuestion ? 1 : 0;
  out[SIAMESE_RELATIONAL_DIM + 3] = signals.emojiToCharRatio;
  out[SIAMESE_RELATIONAL_DIM + 4] = signals.cjkCharRatio;
  out[SIAMESE_RELATIONAL_DIM + 5] = signals.matchedSpamKeyword !== null ? 1 : 0;

  return out as SiameseFeatureVec;
}

export function scoreSiameseHead(
  features: SiameseFeatureVec,
  replyVec: UnitVec256,
  rawText: string,
  classifier: OnlineClassifierWeights,
  authorSpamPrior = 0,
): { readonly probability: Probability; readonly dominantLanguage: SupportedLanguage } {
  let linearMargin = classifier.bias;
  for (let i = 0; i < SIAMESE_TOTAL_DIM; i++) {
    linearMargin += (classifier.weights[i] ?? 0) * (features[i] ?? 0);
  }

  let bestPosSim = -1;
  let bestNegSim = -1;
  let bestPosLang: SupportedLanguage | null = null;

  for (const proto of classifier.seedPrototypes) {
    const sim = dotProduct256(replyVec, proto.embedding);
    if (proto.label === 1 && sim > bestPosSim) {
      bestPosSim = sim;
      bestPosLang = proto.lang;
    } else if (proto.label === -1 && sim > bestNegSim) {
      bestNegSim = sim;
    }
  }

  const prototypeMargin =
    bestPosSim >= 0 && bestNegSim >= 0 ? (bestPosSim - bestNegSim) * 1.5 : 0;
  const authorMargin = authorSpamPrior * 1.35;
  const logit = linearMargin + prototypeMargin + authorMargin;
  const prob = 1 / (1 + Math.exp(-logit));
  const dominantLanguage = bestPosLang ?? detectDominantLanguage(rawText);

  return {
    probability: asProbability(prob),
    dominantLanguage,
  };
}

export function updateOnlineWeightsPassiveAggressive(
  current: OnlineClassifierWeights,
  features: SiameseFeatureVec,
  target: 1 | -1,
  aggressivenessC = DEFAULT_THRESHOLDS.passiveAggressiveC,
): OnlineClassifierWeights {
  let margin = current.bias;
  let normSq = 1;
  for (let i = 0; i < SIAMESE_TOTAL_DIM; i++) {
    const x = features[i] ?? 0;
    margin += (current.weights[i] ?? 0) * x;
    normSq += x * x;
  }

  const hingeLoss = Math.max(0, 1 - target * margin);
  if (hingeLoss === 0) {
    return {
      ...current,
      updateCount: current.updateCount + 1,
    };
  }

  const tau = Math.min(aggressivenessC, hingeLoss / normSq);
  const step = tau * target;
  const nextWeights = new Float32Array(current.weights);
  for (let i = 0; i < SIAMESE_TOTAL_DIM; i++) {
    nextWeights[i] = (nextWeights[i] ?? 0) + step * (features[i] ?? 0);
  }

  return {
    weights: nextWeights,
    bias: current.bias + step,
    seedPrototypes: current.seedPrototypes,
    updateCount: current.updateCount + 1,
    authorStats: current.authorStats,
  };
}

export function evaluateReplyRules(
  ctx: ReplyEvaluationContext,
  cfg: ThresholdConfig = DEFAULT_THRESHOLDS,
): VerdictRecord {
  const diagnostics: VerdictDiagnostics = {
    opCosSim: ctx.opCosSim,
    maxCloneSim: ctx.maxCloneSim,
    siameseProbability: ctx.siameseProbability,
    authorSpamPrior: ctx.authorSpamPrior,
    authorSpamCount: ctx.authorStats?.spamCount ?? 0,
    authorHumanCount: ctx.authorStats?.humanCount ?? 0,
  };
  const snippet = ctx.candidate.rawText.slice(0, 80);

  for (const rule of VERDICT_RULE_TABLE) {
    const reason = rule.match(ctx, cfg);
    if (reason !== null) {
      const score =
        reason.kind === "thread_clone"
          ? reason.similarityToPeer
          : reason.kind === "op_paraphrase"
            ? reason.similarityToOp
            : reason.kind === "keyword_spam" || reason.kind === "known_bot_author"
              ? 1
              : ctx.siameseProbability;
      return {
        tweetId: ctx.candidate.tweetId,
        authorId: ctx.candidate.authorId,
        snippet,
        action: reason.action,
        score,
        tokenCount: ctx.tokenCount,
        reason,
        diagnostics,
      };
    }
  }

  return {
    tweetId: ctx.candidate.tweetId,
    authorId: ctx.candidate.authorId,
    snippet,
    action: "keep",
    score: ctx.siameseProbability,
    tokenCount: ctx.tokenCount,
    reason: { kind: "clean", action: "keep" },
    diagnostics,
  };
}

export class ThreadFilterEngine {
  private readonly slabs = new Map<ThreadId, ThreadSlabState>();
  private readonly maxRepliesPerThread: number;
  private readonly maxCachedThreads: number;
  private readonly thresholds: ThresholdConfig;

  constructor(
    maxRepliesPerThread: number = DEFAULT_SLAB_CAPACITY,
    thresholds: ThresholdConfig = DEFAULT_THRESHOLDS,
    maxCachedThreads: number = MAX_CACHED_THREADS,
  ) {
    this.maxRepliesPerThread = maxRepliesPerThread;
    this.thresholds = thresholds;
    this.maxCachedThreads = maxCachedThreads;
  }

  private getOrCreateSlab(threadId: ThreadId): ThreadSlabState {
    const existing = this.slabs.get(threadId);
    if (existing) {
      this.slabs.delete(threadId);
      this.slabs.set(threadId, existing);
      return existing;
    }

    if (this.slabs.size >= this.maxCachedThreads) {
      const oldestKey = this.slabs.keys().next().value;
      if (oldestKey !== undefined) {
        this.slabs.delete(oldestKey);
      }
    }

    const slab = createSlab(threadId, this.maxRepliesPerThread);
    this.slabs.set(threadId, slab);
    return slab;
  }

  public getAnchorKind(threadId: ThreadId): ThreadAnchorState["kind"] {
    const slab = this.slabs.get(threadId);
    return slab ? slab.anchor.kind : "unanchored";
  }

  public getCachedThreadCount(): number {
    return this.slabs.size;
  }

  public getDiagnosticsSummary(classifier: OnlineClassifierWeights): DiagnosticsSummary {
    let totalScoredReplies = 0;
    let dimmedReplies = 0;
    let keptReplies = 0;
    const countsByReason: Record<string, number> = {};
    const recentVerdicts: VerdictRecord[] = [];

    for (const slab of this.slabs.values()) {
      for (const verdict of slab.verdictsByTweetId.values()) {
        totalScoredReplies += 1;
        if (verdict.action === "dim") {
          dimmedReplies += 1;
        } else {
          keptReplies += 1;
        }
        countsByReason[verdict.reason.kind] = (countsByReason[verdict.reason.kind] ?? 0) + 1;
        recentVerdicts.push(verdict);
      }
    }

    const topFlaggedAuthors: AuthorReputationEntry[] = [];
    for (const [authorId, stats] of classifier.authorStats.entries()) {
      if (stats.spamCount > 0 || stats.autoFlagCount > 0 || stats.humanCount > 0) {
        topFlaggedAuthors.push({
          authorId,
          spamCount: stats.spamCount,
          humanCount: stats.humanCount,
          autoFlagCount: stats.autoFlagCount,
          prior: computeAuthorSpamPrior(stats),
        });
      }
    }
    topFlaggedAuthors.sort((a, b) => b.spamCount - a.spamCount || b.prior - a.prior);

    return {
      cachedThreadCount: this.slabs.size,
      totalScoredReplies,
      dimmedReplies,
      keptReplies,
      updateCount: classifier.updateCount,
      countsByReason,
      topFlaggedAuthors: topFlaggedAuthors.slice(0, 15),
      recentVerdicts: recentVerdicts.slice(-25).reverse(),
    };
  }

  public filterUnseenCandidates(
    threadId: ThreadId,
    candidates: ReadonlyArray<CandidateTweet>,
  ): {
    readonly cached: ReadonlyArray<VerdictRecord>;
    readonly toEmbed: ReadonlyArray<CandidateTweet>;
  } {
    const slab = this.getOrCreateSlab(threadId);
    const cached: VerdictRecord[] = [];
    const toEmbed: CandidateTweet[] = [];
    const seenInBatch = new Set<TweetId>();

    for (const candidate of candidates) {
      if (seenInBatch.has(candidate.tweetId)) {
        continue;
      }
      seenInBatch.add(candidate.tweetId);

      if (
        candidate.isOp &&
        slab.anchor.kind !== "unanchored" &&
        slab.anchor.tweetId === candidate.tweetId
      ) {
        continue;
      }

      const existingVerdict = slab.verdictsByTweetId.get(candidate.tweetId);
      if (existingVerdict !== undefined) {
        cached.push(existingVerdict);
      } else {
        toEmbed.push(candidate);
      }
    }

    return { cached, toEmbed };
  }

  public ingestEmbeddedBatch(
    threadId: ThreadId,
    embeddedBatch: ReadonlyArray<EmbeddedTweet>,
    classifier: OnlineClassifierWeights,
  ): BatchEvaluationResult {
    const slab = this.getOrCreateSlab(threadId);
    const verdicts: VerdictRecord[] = [];
    const retroactiveUpdates: VerdictRecord[] = [];

    for (const item of embeddedBatch) {
      if (!item.candidate.isOp) {
        continue;
      }
      if (item.tokenCount < this.thresholds.minOpTokens) {
        slab.anchor = {
          kind: "media_only",
          tweetId: item.candidate.tweetId,
          authorId: item.candidate.authorId,
          tokenCount: item.tokenCount,
        };
      } else {
        slab.anchor = {
          kind: "text_anchored",
          tweetId: item.candidate.tweetId,
          authorId: item.candidate.authorId,
          tokenCount: item.tokenCount,
          embedding: item.embedding,
          cjkCharRatio: item.candidate.domSignals.cjkCharRatio,
        };
      }
    }

    for (const item of embeddedBatch) {
      if (item.candidate.isOp) {
        continue;
      }

      const existingVerdict = slab.verdictsByTweetId.get(item.candidate.tweetId);
      if (existingVerdict !== undefined) {
        verdicts.push(existingVerdict);
        continue;
      }

      const priorCount = slab.count;
      const anchorAuthorId =
        slab.anchor.kind !== "unanchored" ? slab.anchor.authorId : null;
      const { maxSim, matchSlot } = findMaxCloneSimilarity(
        item.embedding,
        item.candidate.authorId,
        anchorAuthorId,
        slab.replyMatrix,
        slab.replyAuthorIds,
        priorCount,
      );

      let canonicalCloneTweetId: TweetId | null = null;
      let cloneClusterSize = 1;

      if (matchSlot >= 0 && maxSim >= this.thresholds.cloneCosineMin) {
        canonicalCloneTweetId = slab.replyTweetIds[matchSlot] ?? null;
        const nextClusterSize = (slab.cloneClusterCounts[matchSlot] ?? 1) + 1;
        slab.cloneClusterCounts[matchSlot] = nextClusterSize;
        cloneClusterSize = nextClusterSize;

        if (canonicalCloneTweetId !== null) {
          const peerVerdict = slab.verdictsByTweetId.get(canonicalCloneTweetId);
          if (peerVerdict && peerVerdict.action === "keep") {
            const promotedVerdict: VerdictRecord = {
              ...peerVerdict,
              action: "dim",
              score: maxSim,
              reason: {
                kind: "thread_clone",
                action: "dim",
                similarityToPeer: maxSim,
                canonicalTweetId: canonicalCloneTweetId,
                clusterSize: nextClusterSize,
              },
            };
            slab.verdictsByTweetId.set(canonicalCloneTweetId, promotedVerdict);
            retroactiveUpdates.push(promotedVerdict);
          }
        }
      }

      if (slab.count < slab.maxCapacity) {
        const slot = slab.count;
        slab.count += 1;
        slab.replyMatrix.set(item.embedding, slot * EMBEDDING_DIM);
        slab.replyTweetIds[slot] = item.candidate.tweetId;
        slab.replyAuthorIds[slot] = item.candidate.authorId;
        slab.replySnippets[slot] = item.candidate.rawText.slice(0, 80);
        slab.replyTokenCounts[slot] = item.tokenCount;
        slab.cloneClusterCounts[slot] = cloneClusterSize;
        slab.slotByTweetId.set(item.candidate.tweetId, slot);
      }

      const opCosSim =
        slab.anchor.kind === "text_anchored"
          ? dotProduct256(slab.anchor.embedding, item.embedding)
          : asUnitCosine(0);

      const features = buildSiameseFeatures(
        item.embedding,
        slab.anchor,
        item.candidate.domSignals,
      );
      slab.featuresByTweetId.set(item.candidate.tweetId, features);

      const authorStats = classifier.authorStats.get(item.candidate.authorId);
      const authorSpamPrior = computeAuthorSpamPrior(authorStats);

      const { probability, dominantLanguage } = scoreSiameseHead(
        features,
        item.embedding,
        item.candidate.rawText,
        classifier,
        authorSpamPrior,
      );

      const verdict = evaluateReplyRules(
        {
          candidate: item.candidate,
          tokenCount: item.tokenCount,
          anchor: slab.anchor,
          opCosSim,
          maxCloneSim: maxSim,
          canonicalCloneTweetId,
          cloneClusterSize,
          siameseProbability: probability,
          dominantLanguage,
          authorStats,
          authorSpamPrior,
        },
        this.thresholds,
      );

      slab.verdictsByTweetId.set(item.candidate.tweetId, verdict);
      verdicts.push(verdict);
    }

    return {
      threadId,
      anchorKind: slab.anchor.kind,
      verdicts,
      retroactiveUpdates,
    };
  }

  public applyUserFeedback(
    threadId: ThreadId,
    tweetId: TweetId,
    label: "spam" | "human",
    currentWeights: OnlineClassifierWeights,
  ): {
    readonly updatedVerdict: VerdictRecord;
    readonly retroactiveUpdates: ReadonlyArray<VerdictRecord>;
    readonly nextWeights: OnlineClassifierWeights;
  } | null {
    const slab = this.slabs.get(threadId);
    if (!slab) {
      return null;
    }

    const features = slab.featuresByTweetId.get(tweetId);
    const priorVerdict = slab.verdictsByTweetId.get(tweetId);
    if (!features || !priorVerdict) {
      return null;
    }

    const slot = slab.slotByTweetId.get(tweetId);
    const authorId = slot !== undefined ? slab.replyAuthorIds[slot] : priorVerdict.authorId;

    const target: 1 | -1 = label === "spam" ? 1 : -1;
    let nextWeights = updateOnlineWeightsPassiveAggressive(
      currentWeights,
      features,
      target,
      this.thresholds.passiveAggressiveC,
    );

    if (authorId !== undefined) {
      nextWeights = recordAuthorOutcome(
        nextWeights,
        authorId,
        label === "spam" ? "user_spam" : "user_human",
      );
    }

    const updatedStats = authorId !== undefined ? nextWeights.authorStats.get(authorId) : undefined;
    const updatedPrior = computeAuthorSpamPrior(updatedStats);

    const updatedVerdict: VerdictRecord = {
      ...priorVerdict,
      action: label === "spam" ? "dim" : "keep",
      score: label === "spam" ? 1 : 0,
      reason: {
        kind: "user_override",
        action: label === "spam" ? "dim" : "keep",
        label,
      },
      diagnostics: priorVerdict.diagnostics
        ? {
            ...priorVerdict.diagnostics,
            authorSpamPrior: updatedPrior,
            authorSpamCount: updatedStats?.spamCount ?? 0,
            authorHumanCount: updatedStats?.humanCount ?? 0,
          }
        : undefined,
    };

    slab.verdictsByTweetId.set(tweetId, updatedVerdict);

    const retroactiveUpdates: VerdictRecord[] = [];
    if (
      label === "spam" &&
      authorId !== undefined &&
      updatedStats !== undefined &&
      updatedStats.spamCount >= this.thresholds.authorKnownBotMinSpam &&
      updatedStats.spamCount > updatedStats.humanCount * 2
    ) {
      for (let s = 0; s < slab.count; s++) {
        const peerTweetId = slab.replyTweetIds[s];
        if (peerTweetId === undefined || peerTweetId === tweetId) {
          continue;
        }
        if (slab.replyAuthorIds[s] !== authorId) {
          continue;
        }
        const peerVerdict = slab.verdictsByTweetId.get(peerTweetId);
        if (peerVerdict && peerVerdict.action === "keep") {
          const promoted: VerdictRecord = {
            ...peerVerdict,
            action: "dim",
            score: 1,
            reason: {
              kind: "known_bot_author",
              action: "dim",
              authorId,
              spamCount: updatedStats.spamCount,
            },
            diagnostics: peerVerdict.diagnostics
              ? {
                  ...peerVerdict.diagnostics,
                  authorSpamPrior: updatedPrior,
                  authorSpamCount: updatedStats.spamCount,
                  authorHumanCount: updatedStats.humanCount,
                }
              : undefined,
          };
          slab.verdictsByTweetId.set(peerTweetId, promoted);
          retroactiveUpdates.push(promoted);
        }
      }
    }

    return { updatedVerdict, retroactiveUpdates, nextWeights };
  }
}
