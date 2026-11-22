import { CALIBRATED_WEIGHTS_V2 } from "./calibrated-data.ts";
import {
  type OnlineClassifierWeights,
  createInitialClassifierWeights,
  detectDominantLanguage,
  recordAuthorOutcome,
} from "./seeds.ts";
import {
  type CompactFeatureExtraction,
  computeTopicHash8,
  extractCompactFeaturesV2,
  toLangBiasKey,
  toSupportedLanguage,
} from "./stylometrics.ts";
import {
  type AnchorEvaluation,
  type CompactV2ScoreResult,
  getEffectiveV2Params,
  getPresetThresholds,
  insertIntoReplayBuffer,
  parseFeedbackImport,
  scoreCompactV2,
  serializeFeedbackExport,
  trainFromReplayBuffer,
  updateOnlineWeightsReplay,
} from "./trained-head.ts";
import {
  type AuthorId,
  type AuthorStats,
  type CandidateTweet,
  DEFAULT_FILTER_SETTINGS,
  DEFAULT_SLAB_CAPACITY,
  DEFAULT_THRESHOLDS,
  DOM_SIGNAL_DIM,
  type DomSignals,
  EMBEDDING_DIM,
  type EmbeddedTweet,
  type FeedbackReplayEntry,
  type FilterReason,
  type FilterSettings,
  MAX_CACHED_THREADS,
  MAX_FEATURE_CACHE_ENTRIES,
  type PresetThresholds,
  type Probability,
  SIAMESE_RELATIONAL_DIM,
  SIAMESE_TOTAL_DIM,
  type SiameseFeatureVec,
  type SignalContribution,
  type SignalFamily,
  type Stage2Verdict,
  type SupportedLanguage,
  type ThreadAnchorState,
  type ThreadId,
  type ThresholdConfig,
  type TokenCount,
  type TweetId,
  type UndoRecord,
  type UnitCosine,
  type UnitVec256,
  type VerdictDiagnostics,
  type VerdictExplanation,
  type VerdictRecord,
  type VerdictTier,
  asProbability,
  asUnitCosine,
  computeAuthorSpamPrior,
  extractAuthorModelFeatures,
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

// Legacy v1 rule table retained for backward compatibility with direct unit callers
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
  // v2 diagnostics fields (optional for backward compatibility)
  readonly suspectReplies?: number | undefined;
  readonly countsByLang?:
    | Readonly<Record<string, { readonly dim: number; readonly suspect: number; readonly keep: number }>>
    | undefined;
  readonly topSignalHistogram?: Readonly<Record<string, number>> | undefined;
  readonly livePrecisionEstimate?: number | null | undefined;
  readonly rollbackCount?: number | undefined;
  readonly replayBufferSize?: number | undefined;
  readonly settings?: FilterSettings | undefined;
}

interface CachedTweetContext {
  readonly threadId: ThreadId;
  candidate: CandidateTweet;
  readonly embedding: UnitVec256;
  readonly tokenCount: TokenCount;
  anchorSnapshot: ThreadAnchorState;
  focalAuthorId?: AuthorId | undefined;
  maxCloneSim: UnitCosine;
  canonicalCloneTweetId: TweetId | null;
  canonicalTokenCount: number;
  cloneClusterSize: number;
  lastFeatures: CompactFeatureExtraction;
  lastScore: CompactV2ScoreResult;
  lastVerdict: VerdictRecord;
  evaluatedSettingsRevision: number;
  evaluatedWeightsRevision: number;
  userOverride?: "spam" | "human" | undefined;
  stage2Verdict?: Stage2Verdict | undefined;
}

interface ThreadSlabState {
  readonly threadId: ThreadId;
  anchor: ThreadAnchorState;
  focalAuthorId?: AuthorId | undefined;
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
  readonly contextsByTweetId: Map<TweetId, CachedTweetContext>;
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
    contextsByTweetId: new Map(),
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
  excludedSlots?: ReadonlySet<number>,
): { readonly maxSim: UnitCosine; readonly matchSlot: number } {
  let bestSim = -1;
  let bestSlot = -1;

  for (let slot = 0; slot < priorCount; slot++) {
    if (excludedSlots?.has(slot)) continue;
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
  authorId: AuthorId = "anonymous" as AuthorId,
  authorStats?: AuthorStats | undefined,
): SiameseFeatureVec {
  const out = new Float32Array(SIAMESE_TOTAL_DIM);
  out.set(replyVec, 0);

  if (anchor.kind === "text_anchored") {
    const opVec = anchor.embedding;
    const cos = dotProduct256(opVec, replyVec);
    for (let i = 0; i < EMBEDDING_DIM; i++) {
      const u = opVec[i] ?? 0;
      const v = replyVec[i] ?? 0;
      out[EMBEDDING_DIM + i] = v - cos * u;
      out[EMBEDDING_DIM * 2 + i] = u * v;
    }
  }

  out[SIAMESE_RELATIONAL_DIM + 0] = signals.hasVerifiedBadge ? 1 : 0;
  out[SIAMESE_RELATIONAL_DIM + 1] = signals.hasMediaOrExternalLink ? 1 : 0;
  out[SIAMESE_RELATIONAL_DIM + 2] = signals.endsWithBaitQuestion ? 1 : 0;
  out[SIAMESE_RELATIONAL_DIM + 3] = signals.emojiToCharRatio;
  out[SIAMESE_RELATIONAL_DIM + 4] = signals.cjkCharRatio;
  out[SIAMESE_RELATIONAL_DIM + 5] = signals.matchedSpamKeyword !== null ? 1 : 0;

  const authorFeatures = extractAuthorModelFeatures(authorId, authorStats);
  const authorBase = SIAMESE_RELATIONAL_DIM + DOM_SIGNAL_DIM;
  out[authorBase + 0] = authorFeatures.spamRate;
  out[authorBase + 1] = authorFeatures.spamVolume;
  out[authorBase + 2] = authorFeatures.humanRate;
  out[authorBase + 3] = authorFeatures.handleBotScore;

  return out as SiameseFeatureVec;
}

export function scoreSiameseHead(
  features: SiameseFeatureVec,
  replyVec: UnitVec256,
  rawText: string,
  classifier: OnlineClassifierWeights,
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
  const logit = linearMargin + prototypeMargin;
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
    ...current,
    weights: nextWeights,
    bias: current.bias + step * 0.1,
    updateCount: current.updateCount + 1,
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
        tier: reason.action,
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
    tier: "keep",
    score: ctx.siameseProbability,
    tokenCount: ctx.tokenCount,
    reason: { kind: "clean", action: "keep" },
    diagnostics,
  };
}

// ================= Rule Table v2 (R0–R12 + Caps C1–C5 + #16/#17 R9 caveat) =================
const BIO_SCAM_REPLY_CTA_RE =
  /\b(dm\s+me|send\s+(?:me\s+)?a?\s*dm|check\s+my\s+(?:bio|pinned|profile)|link\s+in\s+(?:my\s+)?bio|pinned\s+tweet|join\s+(?:my|our|the)\s+(?:group|channel|telegram|discord|vip|signal)|telegram|whatsapp|t\.me\/|onlyfans)\b|(看主页|看主頁|点我头像|點我頭像|置顶|置頂|私信|电报|電報|進群|进群|加群|プロフ|固ツイ|LINE(追加|登録)?)/i;

export interface EvaluateRulesV2Input {
  readonly candidate: CandidateTweet;
  readonly tokenCount: TokenCount;
  readonly anchor: ThreadAnchorState;
  readonly feat: CompactFeatureExtraction;
  readonly scoreResult: CompactV2ScoreResult;
  readonly canonicalCloneTweetId: TweetId | null;
  readonly canonicalTokenCount: number;
  readonly cloneClusterSize: number;
  readonly authorStats: AuthorStats | undefined;
  readonly settings: FilterSettings;
  readonly thresholds?: PresetThresholds | undefined;
  readonly focalAuthorId?: AuthorId | undefined;
  readonly viewerAuthorId?: AuthorId | undefined;
  readonly userOverride?: "spam" | "human" | undefined;
  readonly nowMs?: number | undefined;
}

function buildRuleContribution(
  ruleId: string,
  evidence?: string | undefined,
): SignalContribution {
  return evidence !== undefined
    ? { id: ruleId, family: "rule", value: 1, contribution: 2.0, evidence }
    : { id: ruleId, family: "rule", value: 1, contribution: 2.0 };
}

function withTierReason(reason: FilterReason, tier: VerdictTier): FilterReason {
  if (tier === "keep") {
    if (reason.action === "keep") return reason;
    return { kind: "clean", action: "keep" };
  }
  switch (reason.kind) {
    case "bot_summon":
    case "bot_answer":
    case "blue_short_praise":
    case "media_only_blue":
    case "op_paraphrase":
    case "thread_clone":
    case "siamese_bot":
      return { ...reason, action: tier };
    case "model_suspect":
      return reason;
    case "known_bot_author":
    case "keyword_spam":
    case "bio_scam":
    case "user_override":
    case "short_human_exempt":
    case "clean":
      return reason;
  }
}

export function evaluateVerdictRulesV2(input: EvaluateRulesV2Input): VerdictRecord {
  const {
    candidate,
    tokenCount,
    anchor,
    feat,
    scoreResult,
    canonicalCloneTweetId,
    canonicalTokenCount,
    cloneClusterSize,
    authorStats,
    settings,
  } = input;

  const thr = input.thresholds ?? getPresetThresholds(settings.preset);
  const toggles = settings.toggles;
  const meta = candidate.graphqlMeta ?? null;
  const v = feat.vector;
  const nowMs = input.nowMs ?? Date.now();
  const authorSpamPrior = computeAuthorSpamPrior(authorStats, nowMs);
  const opCosSim = asUnitCosine(v[0] ?? 0);
  const maxCloneSim = asUnitCosine(v[6] ?? 0);
  const dominantLanguage = toSupportedLanguage(feat.replyLang);
  const snippet = candidate.rawText.slice(0, 80);
  const metaSource: "graphql" | "dom" = meta !== null ? "graphql" : "dom";

  const diagnostics: VerdictDiagnostics = {
    opCosSim,
    maxCloneSim,
    siameseProbability: scoreResult.probability,
    authorSpamPrior,
    authorSpamCount: authorStats?.spamCount ?? 0,
    authorHumanCount: authorStats?.humanCount ?? 0,
  };

  const makeVerdict = (
    tier: VerdictTier,
    ruleId: string,
    reason: FilterReason,
    score: number,
    shields: ReadonlyArray<string>,
    caps: ReadonlyArray<string>,
    ruleEvidence?: string | undefined,
  ): VerdictRecord => {
    const topWithRule =
      ruleId !== "clean" &&
      ruleId !== "short_human_exempt" &&
      ruleId !== "siamese_bot" &&
      ruleId !== "model_suspect" &&
      !ruleId.startsWith("shield_")
        ? [buildRuleContribution(ruleId, ruleEvidence), ...scoreResult.top].slice(0, 5)
        : scoreResult.top;

    const explanation: VerdictExplanation = {
      schemaVersion: 2,
      tier,
      ruleId,
      probability: scoreResult.probability,
      logit: scoreResult.logit,
      thresholds: { dim: thr.dim, suspect: thr.suspect },
      preset: settings.preset,
      top: topWithRule,
      counter: scoreResult.counter,
      families: scoreResult.corroboratingFamilies,
      shields,
      caps,
      metaSource,
      lang: { reply: feat.replyLang, op: feat.opLang },
    };

    return {
      tweetId: candidate.tweetId,
      authorId: candidate.authorId,
      snippet,
      action: tier,
      tier,
      score,
      tokenCount,
      reason,
      diagnostics,
      explanation,
    };
  };

  // R0: Explicit per-tweet user override
  if (input.userOverride !== undefined) {
    const label = input.userOverride;
    const tier: VerdictTier = label === "spam" ? "dim" : "keep";
    return makeVerdict(
      tier,
      "user_override",
      { kind: "user_override", action: tier, label },
      label === "spam" ? 1 : 0,
      [],
      [],
      label,
    );
  }

  // R1: Shields -> keep (OP author, followed, trusted author, self, context ancestor)
  const shields: string[] = [];
  if (candidate.isContextAncestor === true) {
    shields.push("context_ancestor");
  }
  if (
    (anchor.kind !== "unanchored" && candidate.authorId === anchor.authorId) ||
    (input.focalAuthorId !== undefined && candidate.authorId === input.focalAuthorId)
  ) {
    shields.push("op_author");
  }
  if (meta?.following === true) {
    shields.push("followed");
  }
  if (
    authorStats !== undefined &&
    authorStats.humanCount >= 1 &&
    authorStats.humanCount >= authorStats.spamCount
  ) {
    shields.push("trusted_author");
  }
  if (input.viewerAuthorId !== undefined && candidate.authorId === input.viewerAuthorId) {
    shields.push("self");
  }
  if (shields.length > 0) {
    return makeVerdict(
      "keep",
      `shield_${shields[0]}`,
      { kind: "clean", action: "keep" },
      scoreResult.probability,
      shields,
      [],
    );
  }

  // R2: known_bot_author (user marks ONLY — auto-flags excluded)
  if (
    authorStats !== undefined &&
    authorStats.spamCount >= thr.kb &&
    authorStats.spamCount > authorStats.humanCount * 2
  ) {
    return makeVerdict(
      "dim",
      "known_bot_author",
      {
        kind: "known_bot_author",
        action: "dim",
        authorId: candidate.authorId,
        spamCount: authorStats.spamCount,
      },
      1,
      [],
      [],
      `@${candidate.authorId} (${authorStats.spamCount}x)`,
    );
  }

  // R3: keyword_spam / bio_scam (hard rules, never capped)
  if (feat.matchedSpamKeyword !== null) {
    return makeVerdict(
      "dim",
      "keyword_spam",
      {
        kind: "keyword_spam",
        action: "dim",
        matchedPattern: feat.matchedSpamKeyword,
      },
      1,
      [],
      [],
      feat.matchedSpamKeyword,
    );
  }
  // #33 MAJOR: bio_scam requires strong scam-only bio hit AND a link, explicit CTA, or keyword in reply text (never s_syco_open)
  if (
    (v[47] ?? 0) >= 1 &&
    ((v[56] ?? 0) > 0 ||
      /https?:\/\/\S+/i.test(candidate.rawText) ||
      BIO_SCAM_REPLY_CTA_RE.test(candidate.rawText)) &&
    meta?.following !== true
  ) {
    return makeVerdict(
      "dim",
      "bio_scam",
      {
        kind: "bio_scam",
        action: "dim",
        matchedPattern: "bio_spam_strong",
      },
      1,
      [],
      [],
      feat.evidence.m_bio_spam ?? "bio_spam",
    );
  }

  let ruleId = "clean";
  let tier: VerdictTier = "keep";
  let reason: FilterReason = { kind: "clean", action: "keep" };
  let score: number = scoreResult.probability;
  let ruleEvidence: string | undefined;

  const isConfirmedUnfollowedBlue =
    meta !== null && meta.verifiedType === "blue" && meta.following === false;
  const isDomFallbackVerified = meta === null && candidate.domSignals.hasVerifiedBadge;

  // R4: bot_summon / bot_answer
  if (feat.isBotAuthor) {
    ruleId = "bot_answer";
    tier = toggles.collapseBotSummons && settings.preset !== "conservative" ? "dim" : "suspect";
    reason = { kind: "bot_answer", action: tier, authorId: candidate.authorId };
    score = 1;
    ruleEvidence = `@${candidate.authorId}`;
  } else if (feat.isPureSummon) {
    ruleId = "bot_summon";
    tier = toggles.collapseBotSummons && settings.preset !== "conservative" ? "dim" : "suspect";
    const handle = feat.summonHandle ?? "@grok";
    reason = { kind: "bot_summon", action: tier, handle };
    score = 1;
    ruleEvidence = handle;
  }
  // R5: blue_short_praise
  else if (
    (isConfirmedUnfollowedBlue || isDomFallbackVerified) &&
    feat.contentTokenCount <= 12 &&
    feat.specificityHits === 0 &&
    (v[28] ?? 0) === 0 && // !s_casual
    (v[41] ?? 0) === 0 && // !m_followed_by
    ((v[14] ?? 0) > 0 || // s_syco_open
      (v[16] ?? 0) > 0 || // s_slop_lex
      (v[22] ?? 0) > 0 || // s_emoji_only
      ((v[23] ?? 0) > 0 && feat.contentTokenCount <= 4) || // s_emoji_tail & short
      (v[5] ?? 0) >= thr.g_thr)
  ) {
    ruleId = "blue_short_praise";
    tier =
      isConfirmedUnfollowedBlue &&
      toggles.collapseBlueCheckReplyGuys &&
      settings.preset !== "conservative"
        ? "dim"
        : "suspect";
    reason = {
      kind: "blue_short_praise",
      action: tier,
      confirmedUnfollowed: isConfirmedUnfollowedBlue,
    };
    score = Math.max(scoreResult.probability, isConfirmedUnfollowedBlue ? 0.9 : 0.7);
    ruleEvidence = feat.evidence.s_syco_open ?? feat.evidence.s_emoji_only ?? "blue short praise";
  }
  // R6: media_only_blue (unverified zero-text media falls through!)
  else if ((v[54] ?? 0) > 0 && (isConfirmedUnfollowedBlue || isDomFallbackVerified)) {
    ruleId = "media_only_blue";
    tier =
      isConfirmedUnfollowedBlue &&
      toggles.collapseMediaOnlyReplies &&
      settings.preset !== "conservative"
        ? "dim"
        : "suspect";
    reason = {
      kind: "media_only_blue",
      action: tier,
      confirmedUnfollowed: isConfirmedUnfollowedBlue,
    };
    score = isConfirmedUnfollowedBlue ? 0.9 : 0.65;
    ruleEvidence = "media-only blue";
  }
  // R7: short_human_reaction (runs BEFORE R8 thread_clone to fix FP1!)
  // #33 MAJOR: casual clause requires !s_syco_open && s_slop_lex === 0
  else if (
    (feat.contentTokenCount < thr.sg ||
      ((v[28] ?? 0) > 0 &&
        feat.contentTokenCount <= 12 &&
        (v[14] ?? 0) === 0 &&
        (v[16] ?? 0) === 0)) &&
    !((v[37] ?? 0) >= 1 && (v[39] ?? 0) >= 1 && (v[28] ?? 0) === 0)
  ) {
    ruleId = "short_human_exempt";
    tier = "keep";
    reason = { kind: "short_human_exempt", action: "keep", tokenCount };
    score = scoreResult.probability;
  }
  // R8: thread_clone_v2
  else if (
    canonicalCloneTweetId !== null &&
    ((maxCloneSim >= thr.cc &&
      feat.contentTokenCount >= thr.ct &&
      canonicalTokenCount >= thr.ct) ||
      (maxCloneSim >= thr.cc - 0.02 &&
        cloneClusterSize >= 3 &&
        feat.contentTokenCount >= 5 &&
        canonicalTokenCount >= 5))
  ) {
    ruleId = "thread_clone";
    tier = "dim";
    reason = {
      kind: "thread_clone",
      action: "dim",
      similarityToPeer: maxCloneSim,
      canonicalTweetId: canonicalCloneTweetId,
      clusterSize: cloneClusterSize,
    };
    score = maxCloneSim;
    ruleEvidence = `clone ${Math.round(maxCloneSim * 100)}% (${cloneClusterSize}x)`;
  }
  // R9 & R9b: op_paraphrase_v2 with #16/#17 caveat:
  //   - l_novelty <= 0.5: >= 1 corroborating family (style, meta, or e_generic >= g_thr) -> dim
  //   - 0.5 < l_novelty <= 0.85: >= 2 corroborating families (style, meta, embedding / e_generic >= g_thr) -> dim
  //   - otherwise falls through to R9b -> suspect
  else {
    const novelty = v[8] ?? 1;
    const styleCorr = (scoreResult.familyLogits.style ?? 0) >= 0.5;
    const metaCorr = (scoreResult.familyLogits.meta ?? 0) >= 0.5;
    const embCorr =
      (scoreResult.familyLogits.embedding ?? 0) >= 0.5 || (v[5] ?? 0) >= thr.g_thr;
    const corrCount = (styleCorr ? 1 : 0) + (metaCorr ? 1 : 0) + (embCorr ? 1 : 0);

    const paraphraseBase =
      anchor.kind === "text_anchored" &&
      feat.opTokenCount >= 12 &&
      feat.contentTokenCount >= 8 &&
      opCosSim < 0.985 &&
      feat.specificityHits === 0;

    const r9LowNovHit =
      paraphraseBase &&
      opCosSim >= thr.pc &&
      novelty <= 0.5 &&
      (styleCorr || metaCorr || (v[5] ?? 0) >= thr.g_thr);
    const r9SynonymHit =
      paraphraseBase && opCosSim >= thr.pc && novelty > 0.5 && novelty <= 0.85 && corrCount >= 2;
    const r9bSoftHit =
      paraphraseBase && opCosSim >= thr.pc - 0.04 && novelty <= 0.9;

    // #30 & #33 MAJOR: all dim rules (R9, R10) run BEFORE suspect fallbacks (R9b, R11)
    if (r9LowNovHit || r9SynonymHit) {
      ruleId = "op_paraphrase";
      tier = "dim";
      reason = { kind: "op_paraphrase", action: "dim", similarityToOp: opCosSim };
      score = opCosSim;
      ruleEvidence = `OP sim ${Math.round(opCosSim * 100)}%`;
    }
    // R10: model_slop (siamese_bot)
    else if (
      scoreResult.probability >= thr.dim &&
      (scoreResult.corroboratingFamilies.length >= thr.nf ||
        (settings.preset === "strict" &&
          scoreResult.probability >= thr.dim + 0.05 &&
          scoreResult.corroboratingFamilies.length >= 1))
    ) {
      ruleId = "siamese_bot";
      tier = "dim";
      reason = {
        kind: "siamese_bot",
        action: "dim",
        probability: scoreResult.probability,
        dominantLanguage,
        opFallback: anchor.kind !== "text_anchored",
      };
      score = scoreResult.probability;
    }
    // R9b: soft OP paraphrase -> suspect
    else if (r9bSoftHit) {
      ruleId = "op_paraphrase_soft";
      tier = "suspect";
      reason = { kind: "op_paraphrase", action: "suspect", similarityToOp: opCosSim };
      score = opCosSim;
      ruleEvidence = `OP sim ${Math.round(opCosSim * 100)}%`;
    }
    // R11: model_suspect
    else if (scoreResult.probability >= thr.suspect || scoreResult.probability >= thr.dim) {
      ruleId = "model_suspect";
      tier = "suspect";
      reason = {
        kind: "model_suspect",
        action: "suspect",
        probability: scoreResult.probability,
        dominantLanguage,
      };
      score = scoreResult.probability;
    }
  }

  // Caps C1–C5 (never applied to R0–R3, which already returned above)
  const caps: string[] = [];
  if (tier === "dim") {
    if ((v[38] ?? 0) > 0) {
      caps.push("C1_org_verified");
      tier = "suspect";
    }
    if (
      (ruleId === "thread_clone" || ruleId === "op_paraphrase" || ruleId === "siamese_bot") &&
      ((v[50] ?? 0) >= 0.5 || (meta?.favoriteCount ?? 0) >= 100)
    ) {
      caps.push("C2_high_likes");
      tier = "suspect";
    }
    if (meta?.followedBy === true) {
      caps.push("C3_followed_by");
      tier = "suspect";
    }
    if (ruleId === "siamese_bot" && feat.replyLang === "other") {
      caps.push("C4_unsupported_lang");
      tier = "suspect";
    }
  }

  if (!toggles.showSuspectBadges && tier === "suspect") {
    caps.push("C5_hide_suspect");
    tier = "keep";
  }

  reason = withTierReason(reason, tier);
  return makeVerdict(tier, ruleId, reason, score, [], caps, ruleEvidence);
}

export interface FeedbackActionResult {
  readonly updatedVerdict: VerdictRecord;
  readonly retroactiveUpdates: ReadonlyArray<VerdictRecord>;
  readonly nextWeights: OnlineClassifierWeights;
  readonly rolledBack?: boolean | undefined;
  readonly anchorEval?: AnchorEvaluation | undefined;
}

export interface UndoFeedbackResult {
  readonly restoredVerdict: VerdictRecord | null;
  readonly nextWeights: OnlineClassifierWeights;
  readonly undoneLabel: "spam" | "human";
  readonly undoneTweetId: TweetId;
}

export interface Stage2UpdateResult {
  readonly updatedVerdict: VerdictRecord;
  readonly changedTier: boolean;
}

export class ThreadFilterEngine {
  private readonly slabs = new Map<ThreadId, ThreadSlabState>();
  private readonly featureCacheByTweetId = new Map<TweetId, CachedTweetContext>();
  private readonly maxRepliesPerThread: number;
  private readonly maxCachedThreads: number;
  private readonly thresholds: ThresholdConfig;
  private settings: FilterSettings = DEFAULT_FILTER_SETTINGS;
  private dimsReviewedCount = 0;
  private dimsOverriddenHumanCount = 0;

  constructor(
    maxRepliesPerThread: number = DEFAULT_SLAB_CAPACITY,
    thresholds: ThresholdConfig = DEFAULT_THRESHOLDS,
    maxCachedThreads: number = MAX_CACHED_THREADS,
    initialSettings: FilterSettings = DEFAULT_FILTER_SETTINGS,
  ) {
    this.maxRepliesPerThread = maxRepliesPerThread;
    this.thresholds = thresholds;
    this.maxCachedThreads = maxCachedThreads;
    this.settings = initialSettings;
  }

  public getSettings(): FilterSettings {
    return this.settings;
  }

  public setSettings(next: FilterSettings): void {
    this.settings = next;
  }

  public setThreadFocalAuthor(threadId: ThreadId, focalAuthorId: AuthorId): void {
    const slab = this.getOrCreateSlab(threadId);
    slab.focalAuthorId = focalAuthorId;
  }

  private putFeatureCache(tweetId: TweetId, ctx: CachedTweetContext): void {
    if (this.featureCacheByTweetId.has(tweetId)) {
      this.featureCacheByTweetId.delete(tweetId);
    } else if (this.featureCacheByTweetId.size >= MAX_FEATURE_CACHE_ENTRIES) {
      const oldest = this.featureCacheByTweetId.keys().next().value;
      if (oldest !== undefined) {
        this.featureCacheByTweetId.delete(oldest);
      }
    }
    this.featureCacheByTweetId.set(tweetId, ctx);
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

  public getCachedFeatureCount(): number {
    return this.featureCacheByTweetId.size;
  }

  public getCachedContext(tweetId: TweetId): {
    readonly threadId: ThreadId;
    readonly candidate: CandidateTweet;
    readonly verdict: VerdictRecord;
  } | null {
    const ctx = this.featureCacheByTweetId.get(tweetId);
    if (!ctx) return null;
    return {
      threadId: ctx.threadId,
      candidate: ctx.candidate,
      verdict: ctx.lastVerdict,
    };
  }

  public getDiagnosticsSummary(
    classifier: OnlineClassifierWeights,
    settings: FilterSettings = this.settings,
  ): DiagnosticsSummary {
    let totalScoredReplies = 0;
    let dimmedReplies = 0;
    let suspectReplies = 0;
    let keptReplies = 0;
    const countsByReason: Record<string, number> = {};
    const countsByLang: Record<string, { dim: number; suspect: number; keep: number }> = {};
    const topSignalHistogram: Record<string, number> = {};
    const recentVerdicts: VerdictRecord[] = [];

    for (const slab of this.slabs.values()) {
      for (const verdict of slab.verdictsByTweetId.values()) {
        totalScoredReplies += 1;
        if (verdict.action === "dim") {
          dimmedReplies += 1;
        } else if (verdict.action === "suspect") {
          suspectReplies += 1;
        } else {
          keptReplies += 1;
        }
        countsByReason[verdict.reason.kind] = (countsByReason[verdict.reason.kind] ?? 0) + 1;

        const langKey = verdict.explanation?.lang.reply ?? "other";
        const lc = countsByLang[langKey] ?? { dim: 0, suspect: 0, keep: 0 };
        if (verdict.action === "dim") lc.dim += 1;
        else if (verdict.action === "suspect") lc.suspect += 1;
        else lc.keep += 1;
        countsByLang[langKey] = lc;

        if (verdict.explanation) {
          for (const sig of verdict.explanation.top.slice(0, 3)) {
            topSignalHistogram[sig.id] = (topSignalHistogram[sig.id] ?? 0) + 1;
          }
        }
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

    const reviewed = classifier.dimsReviewedCount ?? this.dimsReviewedCount;
    const overridden = classifier.dimsOverriddenHumanCount ?? this.dimsOverriddenHumanCount;
    const livePrecisionEstimate =
      reviewed > 0 ? Math.max(0, 1 - overridden / reviewed) : null;

    return {
      cachedThreadCount: this.slabs.size,
      totalScoredReplies,
      dimmedReplies,
      suspectReplies,
      keptReplies,
      updateCount: classifier.updateCount,
      countsByReason,
      countsByLang,
      topSignalHistogram,
      livePrecisionEstimate,
      rollbackCount: classifier.rollbackCount ?? 0,
      replayBufferSize: classifier.replayBuffer?.length ?? 0,
      settings,
      topFlaggedAuthors: topFlaggedAuthors.slice(0, 15),
      recentVerdicts: recentVerdicts.slice(-25).reverse(),
    };
  }

  private evaluateCachedContext(
    ctx: CachedTweetContext,
    classifier: OnlineClassifierWeights,
    settings: FilterSettings,
    nowMs: number = Date.now(),
  ): VerdictRecord {
    const params = getEffectiveV2Params(classifier);
    const authorStats = classifier.authorStats.get(ctx.candidate.authorId);
    const feat = extractCompactFeaturesV2({
      candidate: ctx.candidate,
      replyVec: ctx.embedding,
      anchor: ctx.anchorSnapshot,
      protoBank: params.protoBank,
      cloneCtx: {
        maxCloneSim: ctx.maxCloneSim,
        clusterSize: ctx.cloneClusterSize,
      },
      authorStats,
      focalAuthorId: ctx.focalAuthorId,
      nowMs,
    });
    const scoreResult = scoreCompactV2(feat, classifier);
    let verdict = evaluateVerdictRulesV2({
      candidate: ctx.candidate,
      tokenCount: ctx.tokenCount,
      anchor: ctx.anchorSnapshot,
      feat,
      scoreResult,
      canonicalCloneTweetId: ctx.canonicalCloneTweetId,
      canonicalTokenCount: ctx.canonicalTokenCount,
      cloneClusterSize: ctx.cloneClusterSize,
      authorStats,
      settings,
      focalAuthorId: ctx.focalAuthorId,
      userOverride: ctx.userOverride,
      nowMs,
    });

    // Re-apply Stage-2 verdict if already computed and settings allow it
    if (ctx.stage2Verdict !== undefined && settings.toggles.enableStage2) {
      verdict = this.applyStage2ToVerdict(verdict, ctx.stage2Verdict, settings).updatedVerdict;
    }

    ctx.lastFeatures = feat;
    ctx.lastScore = scoreResult;
    ctx.lastVerdict = verdict;
    ctx.evaluatedSettingsRevision = settings.revision;
    ctx.evaluatedWeightsRevision = classifier.weightsRevision ?? 1;
    return verdict;
  }

  /**
   * Re-evaluates all replies in a cached thread using stored features & embeddings with ZERO GPU cost.
   */
  public reevaluateThread(
    threadId: ThreadId,
    classifier: OnlineClassifierWeights,
    settings: FilterSettings = this.settings,
    nowMs: number = Date.now(),
  ): ReadonlyArray<VerdictRecord> {
    this.settings = settings;
    const slab = this.slabs.get(threadId);
    if (!slab) return [];
    const updated: VerdictRecord[] = [];
    for (const [tweetId, ctx] of slab.contextsByTweetId.entries()) {
      const verdict = this.evaluateCachedContext(ctx, classifier, settings, nowMs);
      slab.verdictsByTweetId.set(tweetId, verdict);
      updated.push(verdict);
    }
    return updated;
  }

  public reevaluateAllThreads(
    classifier: OnlineClassifierWeights,
    settings: FilterSettings = this.settings,
    nowMs: number = Date.now(),
  ): ReadonlyMap<ThreadId, ReadonlyArray<VerdictRecord>> {
    this.settings = settings;
    const out = new Map<ThreadId, ReadonlyArray<VerdictRecord>>();
    for (const threadId of this.slabs.keys()) {
      out.set(threadId, this.reevaluateThread(threadId, classifier, settings, nowMs));
    }
    return out;
  }

  public filterUnseenCandidates(
    threadId: ThreadId,
    candidates: ReadonlyArray<CandidateTweet>,
    classifier?: OnlineClassifierWeights,
    settings: FilterSettings = this.settings,
    focalAuthorId?: AuthorId | undefined,
  ): {
    readonly cached: ReadonlyArray<VerdictRecord>;
    readonly toEmbed: ReadonlyArray<CandidateTweet>;
  } {
    const slab = this.getOrCreateSlab(threadId);
    if (focalAuthorId !== undefined) {
      slab.focalAuthorId = focalAuthorId;
    }
    const cached: VerdictRecord[] = [];
    const toEmbed: CandidateTweet[] = [];
    const seenInBatch = new Set<TweetId>();
    const weightsRev = classifier?.weightsRevision ?? 1;

    for (const candidate of candidates) {
      if (seenInBatch.has(candidate.tweetId)) {
        continue;
      }
      seenInBatch.add(candidate.tweetId);

      if (candidate.isOp) {
        if (slab.anchor.kind !== "unanchored" && slab.anchor.tweetId === candidate.tweetId) {
          // Update OP favoriteCount / rawText if late GraphQL enriched the OP
          if (
            candidate.graphqlMeta?.favoriteCount !== undefined &&
            slab.anchor.favoriteCount !== candidate.graphqlMeta.favoriteCount
          ) {
            slab.anchor = {
              ...slab.anchor,
              favoriteCount: candidate.graphqlMeta.favoriteCount,
              rawText: candidate.rawText,
            };
          }
          continue;
        }
        toEmbed.push(candidate);
        continue;
      }

      const existingCtx = slab.contextsByTweetId.get(candidate.tweetId);
      const existingVerdict = slab.verdictsByTweetId.get(candidate.tweetId);
      if (existingCtx !== undefined) {
        const metaEnriched =
          existingCtx.candidate.graphqlMeta == null && candidate.graphqlMeta != null;
        const focalUpdated =
          slab.focalAuthorId !== undefined && existingCtx.focalAuthorId !== slab.focalAuthorId;
        const revChanged =
          existingCtx.evaluatedSettingsRevision !== settings.revision ||
          (classifier !== undefined && existingCtx.evaluatedWeightsRevision !== weightsRev);

        if ((metaEnriched || focalUpdated || revChanged) && classifier !== undefined) {
          if (metaEnriched) {
            existingCtx.candidate = candidate;
            existingCtx.anchorSnapshot = slab.anchor;
          }
          if (focalUpdated) {
            existingCtx.focalAuthorId = slab.focalAuthorId;
          }
          const refreshed = this.evaluateCachedContext(existingCtx, classifier, settings);
          slab.verdictsByTweetId.set(candidate.tweetId, refreshed);
          cached.push(refreshed);
        } else {
          cached.push(existingCtx.lastVerdict);
        }
      } else if (existingVerdict !== undefined) {
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
    settings: FilterSettings = this.settings,
    nowMs: number = Date.now(),
    focalAuthorId?: AuthorId | undefined,
  ): BatchEvaluationResult {
    this.settings = settings;
    const slab = this.getOrCreateSlab(threadId);
    if (focalAuthorId !== undefined) {
      slab.focalAuthorId = focalAuthorId;
    }
    const verdicts: VerdictRecord[] = [];
    const retroactiveUpdates: VerdictRecord[] = [];
    const params = getEffectiveV2Params(classifier);
    const thr = getPresetThresholds(settings.preset);

    // 1. Anchor OP first
    for (const item of embeddedBatch) {
      if (!item.candidate.isOp) {
        continue;
      }
      const opLikes =
        item.candidate.graphqlMeta?.favoriteCount ?? item.candidate.opFavoriteCount ?? null;
      if (item.tokenCount < this.thresholds.minOpTokens) {
        slab.anchor = {
          kind: "media_only",
          tweetId: item.candidate.tweetId,
          authorId: item.candidate.authorId,
          tokenCount: item.tokenCount,
          rawText: item.candidate.rawText,
          favoriteCount: opLikes,
        };
      } else {
        slab.anchor = {
          kind: "text_anchored",
          tweetId: item.candidate.tweetId,
          authorId: item.candidate.authorId,
          tokenCount: item.tokenCount,
          embedding: item.embedding,
          cjkCharRatio: item.candidate.domSignals.cjkCharRatio,
          rawText: item.candidate.rawText,
          favoriteCount: opLikes,
        };
      }
    }

    // 2. Score replies
    for (const item of embeddedBatch) {
      if (item.candidate.isOp) {
        continue;
      }

      const existingCtx = slab.contextsByTweetId.get(item.candidate.tweetId);
      if (existingCtx !== undefined) {
        verdicts.push(existingCtx.lastVerdict);
        continue;
      }

      const priorCount = slab.count;
      const anchorAuthorId =
        slab.anchor.kind !== "unanchored" ? slab.anchor.authorId : (slab.focalAuthorId ?? null);

      // Exclude shielded (followed / OP / ancestor) prior slots from clone matching
      const excludedSlots = new Set<number>();
      for (let s = 0; s < priorCount; s++) {
        const peerTid = slab.replyTweetIds[s];
        const peerCtx = peerTid !== undefined ? slab.contextsByTweetId.get(peerTid) : undefined;
        if (
          peerCtx &&
          (peerCtx.candidate.graphqlMeta?.following === true ||
            peerCtx.candidate.isContextAncestor === true ||
            peerCtx.lastVerdict.explanation?.shields.length)
        ) {
          excludedSlots.add(s);
        }
      }

      const { maxSim, matchSlot } = findMaxCloneSimilarity(
        item.embedding,
        item.candidate.authorId,
        anchorAuthorId,
        slab.replyMatrix,
        slab.replyAuthorIds,
        priorCount,
        excludedSlots,
      );

      let canonicalCloneTweetId: TweetId | null = null;
      let canonicalTokenCount = 0;
      let cloneClusterSize = 1;

      if (matchSlot >= 0) {
        canonicalCloneTweetId = slab.replyTweetIds[matchSlot] ?? null;
        canonicalTokenCount = slab.replyTokenCounts[matchSlot] ?? 0;
        if (maxSim >= thr.cc - 0.02) {
          const nextClusterSize = (slab.cloneClusterCounts[matchSlot] ?? 1) + 1;
          cloneClusterSize = nextClusterSize;
        }
      }

      const authorStats = classifier.authorStats.get(item.candidate.authorId);

      // Extract COMPACT_V2 features & score via Stage-1 calibrated head
      const feat = extractCompactFeaturesV2({
        candidate: item.candidate,
        replyVec: item.embedding,
        anchor: slab.anchor,
        protoBank: params.protoBank,
        cloneCtx: { maxCloneSim: maxSim, clusterSize: cloneClusterSize },
        authorStats,
        focalAuthorId: slab.focalAuthorId,
        nowMs,
      });

      const scoreResult = scoreCompactV2(feat, classifier);

      const verdict = evaluateVerdictRulesV2({
        candidate: item.candidate,
        tokenCount: item.tokenCount,
        anchor: slab.anchor,
        feat,
        scoreResult,
        canonicalCloneTweetId,
        canonicalTokenCount,
        cloneClusterSize,
        authorStats,
        settings,
        thresholds: thr,
        focalAuthorId: slab.focalAuthorId,
        nowMs,
      });

      // Retroactive clone promotion ONLY if this reply triggered thread_clone AND canonical passes guards
      // #30 & #33 MAJOR:
      //  (a) Persist peerCtx.canonicalCloneTweetId & canonicalTokenCount so re-evaluations never flip-flop
      //  (b) Promote suspect peers as well (action !== "dim"), not just action === "keep"
      if (
        verdict.reason.kind === "thread_clone" &&
        verdict.action === "dim" &&
        matchSlot >= 0 &&
        canonicalCloneTweetId !== null
      ) {
        slab.cloneClusterCounts[matchSlot] = cloneClusterSize;
        const peerCtx = slab.contextsByTweetId.get(canonicalCloneTweetId);
        const peerVerdict = slab.verdictsByTweetId.get(canonicalCloneTweetId);
        const peerShielded = (peerVerdict?.explanation?.shields.length ?? 0) > 0;
        const peerShortExempt = peerVerdict?.reason.kind === "short_human_exempt";
        const peerOverridden = peerCtx?.userOverride !== undefined;

        if (
          peerVerdict &&
          peerVerdict.action !== "dim" &&
          !peerShielded &&
          !peerShortExempt &&
          !peerOverridden &&
          peerCtx
        ) {
          const peerMeta = peerCtx.candidate.graphqlMeta ?? null;
          const peerCapped =
            (peerCtx.lastFeatures.vector[38] ?? 0) > 0 ||
            (peerCtx.lastFeatures.vector[50] ?? 0) >= 0.5 ||
            (peerMeta?.favoriteCount ?? 0) >= 100 ||
            peerMeta?.followedBy === true;
          const promotedTier: VerdictTier = peerCapped
            ? settings.toggles.showSuspectBadges
              ? "suspect"
              : "keep"
            : "dim";

          peerCtx.canonicalCloneTweetId = item.candidate.tweetId;
          peerCtx.canonicalTokenCount = feat.contentTokenCount;
          peerCtx.maxCloneSim = maxSim;
          peerCtx.cloneClusterSize = cloneClusterSize;

          if (
            promotedTier !== "keep" &&
            (promotedTier !== peerVerdict.action || peerVerdict.reason.kind !== "thread_clone")
          ) {
            const promotedVerdict: VerdictRecord = {
              ...peerVerdict,
              action: promotedTier,
              tier: promotedTier,
              score: maxSim,
              reason: {
                kind: "thread_clone",
                action: promotedTier,
                similarityToPeer: maxSim,
                canonicalTweetId: item.candidate.tweetId,
                clusterSize: cloneClusterSize,
              },
              explanation: peerVerdict.explanation
                ? {
                    ...peerVerdict.explanation,
                    tier: promotedTier,
                    ruleId: "thread_clone",
                  }
                : undefined,
            };
            slab.verdictsByTweetId.set(canonicalCloneTweetId, promotedVerdict);
            peerCtx.lastVerdict = promotedVerdict;
            retroactiveUpdates.push(promotedVerdict);
          }
        }
      }

      // Store in slab & bounded 2000-entry global feature cache
      if (slab.count < slab.maxCapacity) {
        const slot = slab.count;
        slab.count += 1;
        slab.replyMatrix.set(item.embedding, slot * EMBEDDING_DIM);
        slab.replyTweetIds[slot] = item.candidate.tweetId;
        slab.replyAuthorIds[slot] = item.candidate.authorId;
        slab.replySnippets[slot] = item.candidate.rawText.slice(0, 80);
        slab.replyTokenCounts[slot] = feat.contentTokenCount;
        slab.cloneClusterCounts[slot] = cloneClusterSize;
        slab.slotByTweetId.set(item.candidate.tweetId, slot);
      }

      const legacyFeatures = buildSiameseFeatures(
        item.embedding,
        slab.anchor,
        item.candidate.domSignals,
        item.candidate.authorId,
        authorStats,
      );
      slab.featuresByTweetId.set(item.candidate.tweetId, legacyFeatures);
      slab.verdictsByTweetId.set(item.candidate.tweetId, verdict);

      const cachedCtx: CachedTweetContext = {
        threadId,
        candidate: item.candidate,
        embedding: item.embedding,
        tokenCount: item.tokenCount,
        anchorSnapshot: slab.anchor,
        focalAuthorId: slab.focalAuthorId,
        maxCloneSim: maxSim,
        canonicalCloneTweetId,
        canonicalTokenCount,
        cloneClusterSize,
        lastFeatures: feat,
        lastScore: scoreResult,
        lastVerdict: verdict,
        evaluatedSettingsRevision: settings.revision,
        evaluatedWeightsRevision: classifier.weightsRevision ?? 1,
      };
      slab.contextsByTweetId.set(item.candidate.tweetId, cachedCtx);
      this.putFeatureCache(item.candidate.tweetId, cachedCtx);

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
    settings: FilterSettings = this.settings,
    nowMs: number = Date.now(),
  ): FeedbackActionResult | null {
    const slab = this.slabs.get(threadId);
    // Look up in slab first, then bounded 2000-entry cache surviving LRU slab eviction!
    const ctx = slab?.contextsByTweetId.get(tweetId) ?? this.featureCacheByTweetId.get(tweetId);
    const priorVerdict = slab?.verdictsByTweetId.get(tweetId) ?? ctx?.lastVerdict;
    if (!ctx || !priorVerdict) {
      return null;
    }

    const authorId = ctx.candidate.authorId;
    const prevParams = getEffectiveV2Params(currentWeights);
    const prevAuthorStats = currentWeights.authorStats.get(authorId);
    const prevReplayEntry = currentWeights.replayBuffer?.find((e) => e.tweetId === tweetId);

    // Track live precision metrics on reviewed dims (persisted on weights + mirrored in memory)
    const prevReviewed = currentWeights.dimsReviewedCount ?? this.dimsReviewedCount;
    const prevOverridden = currentWeights.dimsOverriddenHumanCount ?? this.dimsOverriddenHumanCount;
    let nextReviewed = prevReviewed;
    let nextOverridden = prevOverridden;
    if (priorVerdict.action === "dim" && priorVerdict.reason.kind !== "user_override") {
      nextReviewed += 1;
      if (label === "human") {
        nextOverridden += 1;
      }
    }
    this.dimsReviewedCount = nextReviewed;
    this.dimsOverriddenHumanCount = nextOverridden;

    let nextWeights = currentWeights;
    let rolledBack = false;
    let anchorEval: AnchorEvaluation | undefined;

    if (settings.toggles.learnFromFeedback) {
      const replayEntry: FeedbackReplayEntry = {
        tweetId,
        authorId,
        z: Array.from(ctx.lastScore.z),
        fixed: ctx.lastScore.fixedTerm,
        y: label === "spam" ? 1 : 0,
        lang: toLangBiasKey(ctx.lastFeatures.replyLang),
        tsMs: nowMs,
        topicHash: computeTopicHash8(ctx.embedding),
        source: label === "spam" ? "user_spam" : "user_human",
      };

      const replayRes = updateOnlineWeightsReplay(currentWeights, replayEntry, { nowMs });
      nextWeights = replayRes.nextWeights;
      rolledBack = replayRes.rolledBack;
      anchorEval = replayRes.anchorEval;

      // Also keep legacy 778d weights updated cleanly if any v1 caller inspects scoreSiameseHead
      const legacyFeat = slab?.featuresByTweetId.get(tweetId);
      if (legacyFeat) {
        const pa = updateOnlineWeightsPassiveAggressive(
          nextWeights,
          legacyFeat,
          label === "spam" ? 1 : -1,
          this.thresholds.passiveAggressiveC,
        );
        nextWeights = {
          ...nextWeights,
          weights: pa.weights,
          bias: pa.bias,
        };
      }
    } else {
      nextWeights = {
        ...currentWeights,
        updateCount: currentWeights.updateCount + 1,
      };
    }

    nextWeights = recordAuthorOutcome(
      nextWeights,
      authorId,
      label === "spam" ? "user_spam" : "user_human",
      { tweetId, nowMs },
    );

    // Record Undo snapshot (capped at 32)
    const undoRecord: UndoRecord = {
      threadId,
      tweetId,
      authorId,
      label,
      prevAuthorStats,
      prevReplayEntry,
      prevTheta: Array.from(prevParams.theta),
      prevLangBias: { ...prevParams.langBias },
      prevUpdateCount: currentWeights.updateCount,
      prevRollbackCount: currentWeights.rollbackCount ?? 0,
      prevDimsReviewedCount: prevReviewed,
      prevDimsOverriddenHumanCount: prevOverridden,
      prevVerdict: priorVerdict,
      tsMs: nowMs,
    };
    const nextUndo = [...(nextWeights.undoHistory ?? []), undoRecord].slice(-32);
    nextWeights = {
      ...nextWeights,
      undoHistory: nextUndo,
      dimsReviewedCount: nextReviewed,
      dimsOverriddenHumanCount: nextOverridden,
    };

    const updatedStats = nextWeights.authorStats.get(authorId);
    const updatedPrior = computeAuthorSpamPrior(updatedStats, nowMs);

    ctx.userOverride = label;

    const updatedTier: VerdictTier = label === "spam" ? "dim" : "keep";
    const updatedVerdict: VerdictRecord = {
      ...priorVerdict,
      action: updatedTier,
      tier: updatedTier,
      score: label === "spam" ? 1 : 0,
      reason: {
        kind: "user_override",
        action: updatedTier,
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
      explanation: priorVerdict.explanation
        ? {
            ...priorVerdict.explanation,
            tier: updatedTier,
            ruleId: "user_override",
          }
        : undefined,
    };

    ctx.lastVerdict = updatedVerdict;
    slab?.verdictsByTweetId.set(tweetId, updatedVerdict);

    // Retroactively dim/shield visible peers from same author in this thread
    const retroactiveUpdates: VerdictRecord[] = [];
    const thr = getPresetThresholds(settings.preset);
    if (slab && updatedStats !== undefined) {
      const isKnownBot =
        label === "spam" &&
        updatedStats.spamCount >= thr.kb &&
        updatedStats.spamCount > updatedStats.humanCount * 2;

      for (let s = 0; s < slab.count; s++) {
        const peerTweetId = slab.replyTweetIds[s];
        if (peerTweetId === undefined || peerTweetId === tweetId) {
          continue;
        }
        if (slab.replyAuthorIds[s] !== authorId) {
          continue;
        }
        const peerCtx = slab.contextsByTweetId.get(peerTweetId);
        if (!peerCtx || peerCtx.userOverride !== undefined) {
          continue;
        }
        const prevPeerAction = peerCtx.lastVerdict.action;

        if (isKnownBot && prevPeerAction !== "dim") {
          const refreshed = this.evaluateCachedContext(peerCtx, nextWeights, settings, nowMs);
          slab.verdictsByTweetId.set(peerTweetId, refreshed);
          if (refreshed.action === "dim") {
            retroactiveUpdates.push(refreshed);
          }
        } else if (label === "human" && prevPeerAction !== "keep") {
          const refreshed = this.evaluateCachedContext(peerCtx, nextWeights, settings, nowMs);
          slab.verdictsByTweetId.set(peerTweetId, refreshed);
          if (refreshed.action === "keep") {
            retroactiveUpdates.push(refreshed);
          }
        }
      }
    }

    return {
      updatedVerdict,
      retroactiveUpdates,
      nextWeights,
      rolledBack,
      anchorEval,
    };
  }

  public undoLastFeedback(
    currentWeights: OnlineClassifierWeights,
    settings: FilterSettings = this.settings,
  ): UndoFeedbackResult | null {
    const history = currentWeights.undoHistory ?? [];
    if (history.length === 0) {
      return null;
    }
    const last = history[history.length - 1]!;
    const remainingHistory = history.slice(0, -1);

    // Restore authorStats
    const nextAuthors = new Map(currentWeights.authorStats);
    if (last.authorId !== undefined) {
      if (last.prevAuthorStats !== undefined) {
        nextAuthors.set(last.authorId, last.prevAuthorStats);
      } else {
        nextAuthors.delete(last.authorId);
      }
    }

    // Restore replayBuffer entry
    let nextReplay = (currentWeights.replayBuffer ?? []).filter(
      (e) => e.tweetId !== last.tweetId,
    );
    if (last.prevReplayEntry !== undefined) {
      nextReplay = [...nextReplay, last.prevReplayEntry];
    }

    const restoredReviewed =
      last.prevDimsReviewedCount ?? currentWeights.dimsReviewedCount ?? this.dimsReviewedCount;
    const restoredOverridden =
      last.prevDimsOverriddenHumanCount ??
      currentWeights.dimsOverriddenHumanCount ??
      this.dimsOverriddenHumanCount;
    this.dimsReviewedCount = restoredReviewed;
    this.dimsOverriddenHumanCount = restoredOverridden;

    const nextWeights: OnlineClassifierWeights = {
      ...currentWeights,
      theta: new Float32Array(last.prevTheta),
      langBias: { ...last.prevLangBias },
      updateCount: last.prevUpdateCount,
      rollbackCount: last.prevRollbackCount,
      dimsReviewedCount: restoredReviewed,
      dimsOverriddenHumanCount: restoredOverridden,
      authorStats: nextAuthors,
      replayBuffer: nextReplay,
      undoHistory: remainingHistory,
      weightsRevision: (currentWeights.weightsRevision ?? 1) + 1,
    };

    // Restore cached verdict & clear userOverride
    const slab = this.slabs.get(last.threadId);
    const ctx = slab?.contextsByTweetId.get(last.tweetId) ?? this.featureCacheByTweetId.get(last.tweetId);
    let restoredVerdict: VerdictRecord | null = last.prevVerdict ?? null;
    if (ctx) {
      ctx.userOverride = undefined;
      restoredVerdict = this.evaluateCachedContext(ctx, nextWeights, settings);
      slab?.verdictsByTweetId.set(last.tweetId, restoredVerdict);
    }

    return {
      restoredVerdict,
      nextWeights,
      undoneLabel: last.label,
      undoneTweetId: last.tweetId,
    };
  }

  public exportFeedbackJson(weights: OnlineClassifierWeights): string {
    return serializeFeedbackExport(weights);
  }

  public importFeedbackJson(
    rawJson: string,
    currentWeights: OnlineClassifierWeights,
    nowMs: number = Date.now(),
  ): {
    readonly nextWeights: OnlineClassifierWeights;
    readonly importedReplayCount: number;
    readonly importedAuthorCount: number;
    readonly rolledBack: boolean;
  } | null {
    const parsed = parseFeedbackImport(rawJson);
    if (parsed === null) {
      return null;
    }

    const mergedAuthors = new Map(currentWeights.authorStats);
    for (const [id, stats] of parsed.authorStats.entries()) {
      mergedAuthors.set(id, stats);
    }

    // Merge imported replay entries into existing replayBuffer (deduping by tweetId, enforcing bucket/size caps)
    let mergedReplay: FeedbackReplayEntry[] = [...(currentWeights.replayBuffer ?? [])];
    for (const entry of parsed.replayBuffer) {
      mergedReplay = insertIntoReplayBuffer(mergedReplay, entry);
    }

    const baseFresh: OnlineClassifierWeights = {
      ...createInitialClassifierWeights(currentWeights.seedPrototypes, mergedAuthors),
      weightsRevision: currentWeights.weightsRevision ?? 1,
      rollbackCount: currentWeights.rollbackCount ?? 0,
      dimsReviewedCount: currentWeights.dimsReviewedCount ?? this.dimsReviewedCount,
      dimsOverriddenHumanCount:
        currentWeights.dimsOverriddenHumanCount ?? this.dimsOverriddenHumanCount,
      recordedAutoFlagIds: currentWeights.recordedAutoFlagIds,
    };

    const trained = trainFromReplayBuffer(baseFresh, mergedReplay, { nowMs });
    return {
      nextWeights: {
        ...trained.nextWeights,
        authorStats: mergedAuthors,
        updateCount: Math.max(currentWeights.updateCount, mergedReplay.length),
      },
      importedReplayCount: parsed.replayBuffer.length,
      importedAuthorCount: parsed.authorStats.size,
      rolledBack: trained.rolledBack,
    };
  }

  public resetFeedback(currentWeights: OnlineClassifierWeights): OnlineClassifierWeights {
    for (const ctx of this.featureCacheByTweetId.values()) {
      ctx.userOverride = undefined;
    }
    this.dimsReviewedCount = 0;
    this.dimsOverriddenHumanCount = 0;
    const fresh = createInitialClassifierWeights(currentWeights.seedPrototypes, new Map());
    return {
      ...fresh,
      dimsReviewedCount: 0,
      dimsOverriddenHumanCount: 0,
      weightsRevision: (currentWeights.weightsRevision ?? 1) + 1,
    };
  }

  private applyStage2ToVerdict(
    current: VerdictRecord,
    stage2: Stage2Verdict,
    settings: FilterSettings,
  ): Stage2UpdateResult {
    const exp = current.explanation;
    // Never modify R0 user_override or R1 shielded tweets
    if (
      current.reason.kind === "user_override" ||
      (exp !== undefined && exp.shields.length > 0)
    ) {
      return { updatedVerdict: current, changedTier: false };
    }

    const thr = getPresetThresholds(settings.preset);
    const hasBlockingCap =
      exp !== undefined &&
      exp.caps.some(
        (c) =>
          c === "C1_org_verified" ||
          c === "C2_high_likes" ||
          c === "C3_followed_by" ||
          c === "C4_unsupported_lang",
      );
    // #33 minor: restrict Stage-2 bot promotion strictly to model_suspect and op_paraphrase_soft
    // so Stage-2 never overrides Conservative or DOM-only (following=null) demotions.
    const canPromoteWithStage2 =
      exp !== undefined &&
      (exp.ruleId === "model_suspect" || exp.ruleId === "op_paraphrase_soft");

    let nextTier: VerdictTier = current.action;
    let nextReason: FilterReason = current.reason;
    let changedTier = false;

    if (current.action === "suspect") {
      // Asymmetric safety policy from benchmarks/benchmark-report.md §6:
      // - human >= 0.85 clears suspect -> keep
      // - bot >= 0.92 promotes model_suspect/op_paraphrase_soft -> dim ONLY with >=1 corroborating family, p >= tau_suspect, and no C1-C4 caps
      if (stage2.verdict === "human" && stage2.confidence >= 0.85) {
        nextTier = "keep";
        nextReason = { kind: "clean", action: "keep" };
        changedTier = true;
      } else if (
        stage2.verdict === "bot" &&
        stage2.confidence >= 0.92 &&
        canPromoteWithStage2 &&
        !hasBlockingCap &&
        exp !== undefined &&
        exp.families.length >= 1 &&
        exp.probability >= thr.suspect
      ) {
        nextTier = "dim";
        nextReason =
          current.reason.kind === "model_suspect"
            ? {
                kind: "siamese_bot",
                action: "dim",
                probability: asProbability(Math.max(exp.probability, stage2.confidence)),
                dominantLanguage: current.reason.dominantLanguage,
                opFallback: false,
              }
            : withTierReason(current.reason, "dim");
        changedTier = true;
      }
    }

    const updatedVerdict: VerdictRecord = {
      ...current,
      action: nextTier,
      tier: nextTier,
      reason: nextReason,
      explanation: exp
        ? {
            ...exp,
            tier: nextTier,
            stage2,
          }
        : undefined,
    };

    return { updatedVerdict, changedTier };
  }

  public applyStage2Verdict(
    threadId: ThreadId,
    tweetId: TweetId,
    stage2Verdict: Stage2Verdict,
    settings: FilterSettings = this.settings,
  ): Stage2UpdateResult | null {
    const slab = this.slabs.get(threadId);
    const ctx = slab?.contextsByTweetId.get(tweetId) ?? this.featureCacheByTweetId.get(tweetId);
    if (!ctx) return null;

    ctx.stage2Verdict = stage2Verdict;
    const res = this.applyStage2ToVerdict(ctx.lastVerdict, stage2Verdict, settings);
    ctx.lastVerdict = res.updatedVerdict;
    slab?.verdictsByTweetId.set(tweetId, res.updatedVerdict);
    return res;
  }
}

export { CALIBRATED_WEIGHTS_V2 };
