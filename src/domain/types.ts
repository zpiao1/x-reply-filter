export type TweetId = string & { readonly __brand: "TweetId" };
export type AuthorId = string & { readonly __brand: "AuthorId" };
export type ThreadId = string & { readonly __brand: "ThreadId" };
export type TokenCount = number & { readonly __brand: "TokenCount" };
export type UnitCosine = number & { readonly __brand: "UnitCosine" };
export type Probability = number & { readonly __brand: "Probability" };
export type PrototypeRevision = number & { readonly __brand: "PrototypeRevision" };

export type UnitVec256 = Float32Array & {
  readonly __brand: "UnitVec256";
  readonly length: 256;
};

export type SiameseFeatureVec = Float32Array & {
  readonly __brand: "SiameseFeatureVec";
  readonly length: 778;
};

export type CompactFeatureVec = Float32Array & {
  readonly __brand: "CompactFeatureVec";
  readonly length: 58;
};

export const RAW_EMBEDDING_DIM = 768 as const;
export const EMBEDDING_DIM = 256 as const;
export const SIAMESE_RELATIONAL_DIM = 768 as const;
export const DOM_SIGNAL_DIM = 6 as const;
export const AUTHOR_FEATURE_DIM = 4 as const;
export const SIAMESE_TOTAL_DIM = 778 as const;
export const COMPACT_V2_DIM = 58 as const;

export const OP_MIN_TOKENS = 5 as const;
export const REPLY_MIN_TOKENS = 6 as const;
export const OP_PARAPHRASE_MIN = 0.82 as const;
export const EXACT_QUOTE_CEILING = 0.985 as const;
export const THREAD_CLONE_MIN = 0.92 as const;
export const SIAMESE_BOT_THRESHOLD = 0.65 as const;
export const AUTHOR_KNOWN_BOT_MIN_SPAM = 2 as const;
export const MAX_CACHED_THREADS = 16 as const;
export const MAX_FEATURE_CACHE_ENTRIES = 2000 as const;
export const MAX_REPLAY_BUFFER_ENTRIES = 512 as const;
export const MAX_AUTO_FLAGGED_TWEET_IDS = 5000 as const;
export const DEFAULT_SLAB_CAPACITY = 512 as const;
export const OFFSCREEN_IDLE_TIMEOUT_MS = 600_000 as const;
export const SCROLL_DEBOUNCE_MS = 80 as const;
export const CLASSIFICATION_TASK_PREFIX = "task: classification | query: " as const;

export type SupportedLanguage = "en" | "zh" | "ja";
export type LangV2 = "en" | "zh-Hans" | "zh-Hant" | "ja" | "cjk-ambiguous" | "other";
export type LangBiasKey = "en" | "zh-Hans" | "zh-Hant" | "ja" | "other";
export type VerdictTier = "dim" | "suspect" | "keep";
export type SensitivityPreset = "conservative" | "balanced" | "strict";
export type VerifiedType = "blue" | "business" | "government" | "none" | "unknown_verified";
export type MediaKind = "photo" | "video" | "animated_gif";
export type Stage2StatusTag =
  | "disabled"
  | "available"
  | "downloadable"
  | "downloading"
  | "unavailable";

export type SignalFamily =
  | "relational"
  | "embedding"
  | "style"
  | "crosslang"
  | "summon"
  | "meta"
  | "author"
  | "rule";

export interface AuthorStats {
  readonly spamCount: number;
  readonly humanCount: number;
  readonly autoFlagCount: number;
  readonly lastUpdatedAtMs?: number | undefined;
  readonly autoFlagDecayed?: number | undefined;
}

export interface GraphqlTweetMeta {
  readonly tweetId?: TweetId | undefined;
  readonly authorRestId?: string | null | undefined;
  readonly screenName?: AuthorId | null | undefined;
  readonly displayName: string | null;
  readonly authorCreatedAtMs: number | null;
  readonly tweetCreatedAtMs: number | null;
  readonly inReplyToStatusId: string | null;
  readonly conversationId: string | null;
  readonly description: string | null;
  readonly isBlueVerified: boolean | null;
  readonly verifiedType: VerifiedType;
  readonly following: boolean | null;
  readonly followedBy: boolean | null;
  readonly followersCount: number | null;
  readonly friendsCount: number | null;
  readonly statusesCount: number | null;
  readonly defaultProfileImage: boolean | null;
  readonly favoriteCount: number | null;
  readonly replyCount: number | null;
  readonly retweetCount: number | null;
  readonly mediaTypes: ReadonlyArray<MediaKind>;
}

export interface PolicyToggles {
  readonly collapseBlueCheckReplyGuys: boolean;
  readonly collapseMediaOnlyReplies: boolean;
  readonly collapseBotSummons: boolean;
  readonly showSuspectBadges: boolean;
  readonly learnFromFeedback: boolean;
  readonly enableStage2: boolean;
}

export interface FilterSettings {
  readonly preset: SensitivityPreset;
  readonly toggles: PolicyToggles;
  readonly revision: number;
}

export const DEFAULT_POLICY_TOGGLES: PolicyToggles = {
  collapseBlueCheckReplyGuys: true,
  collapseMediaOnlyReplies: true,
  collapseBotSummons: true,
  showSuspectBadges: true,
  learnFromFeedback: true,
  enableStage2: false,
};

export const DEFAULT_FILTER_SETTINGS: FilterSettings = {
  preset: "balanced",
  toggles: DEFAULT_POLICY_TOGGLES,
  revision: 1,
};

export interface PresetThresholds {
  readonly dim: number;
  readonly suspect: number;
  readonly g_thr: number;
  readonly pc: number;
  readonly cc: number;
  readonly ct: number;
  readonly kb: number;
  readonly sg: number;
  readonly nf: number;
}

export interface SignalContribution {
  readonly id: string;
  readonly family: SignalFamily;
  readonly value: number;
  readonly contribution: number;
  readonly evidence?: string | undefined;
}

export interface Stage2Verdict {
  readonly provider: string;
  readonly verdict: "bot" | "human" | "uncertain";
  readonly confidence: number;
  readonly ms: number;
  readonly reason?: string | undefined;
}

export interface VerdictExplanation {
  readonly schemaVersion: 2;
  readonly tier: VerdictTier;
  readonly ruleId: string;
  readonly probability: number;
  readonly logit: number;
  readonly thresholds: { readonly dim: number; readonly suspect: number };
  readonly preset: SensitivityPreset;
  readonly top: ReadonlyArray<SignalContribution>;
  readonly counter: ReadonlyArray<SignalContribution>;
  readonly families: ReadonlyArray<SignalFamily>;
  readonly shields: ReadonlyArray<string>;
  readonly caps: ReadonlyArray<string>;
  readonly metaSource: "graphql" | "dom";
  readonly lang: { readonly reply: LangV2; readonly op: LangV2 | null };
  readonly stage2?: Stage2Verdict | undefined;
}

export interface FeedbackReplayEntry {
  readonly tweetId: TweetId;
  readonly authorId?: AuthorId | undefined;
  readonly z: ReadonlyArray<number>;
  readonly fixed: number;
  readonly y: 0 | 1;
  readonly lang: LangBiasKey;
  readonly tsMs: number;
  readonly topicHash: number;
  readonly source: "user_spam" | "user_human";
}

export interface UndoRecord {
  readonly threadId: ThreadId;
  readonly tweetId: TweetId;
  readonly authorId: AuthorId | undefined;
  readonly label: "spam" | "human";
  readonly prevAuthorStats: AuthorStats | undefined;
  readonly prevReplayEntry: FeedbackReplayEntry | undefined;
  readonly prevTheta: ReadonlyArray<number>;
  readonly prevLangBias: Readonly<Record<LangBiasKey, number>>;
  readonly prevUpdateCount: number;
  readonly prevRollbackCount: number;
  readonly prevVerdict: VerdictRecord | undefined;
  readonly tsMs: number;
}

export interface DomSignals {
  readonly hasVerifiedBadge: boolean;
  readonly hasMediaOrExternalLink: boolean;
  readonly endsWithBaitQuestion: boolean;
  readonly emojiToCharRatio: number;
  readonly cjkCharRatio: number;
  readonly matchedSpamKeyword: string | null;
}

export interface CandidateTweet {
  readonly tweetId: TweetId;
  readonly authorId: AuthorId;
  readonly rawText: string;
  readonly normalizedText: string;
  readonly isOp: boolean;
  readonly domSignals: DomSignals;
  readonly graphqlMeta?: GraphqlTweetMeta | null | undefined;
  readonly hasPrimaryMedia?: boolean | undefined;
  readonly hasQuote?: boolean | undefined;
  readonly isContextAncestor?: boolean | undefined;
  readonly opFavoriteCount?: number | null | undefined;
}

export interface EmbeddedTweet {
  readonly candidate: CandidateTweet;
  readonly embedding: UnitVec256;
  readonly tokenCount: TokenCount;
}

export type ThreadAnchorState =
  | { readonly kind: "unanchored" }
  | {
      readonly kind: "media_only";
      readonly tweetId: TweetId;
      readonly authorId: AuthorId;
      readonly tokenCount: TokenCount;
      readonly rawText?: string | undefined;
      readonly favoriteCount?: number | null | undefined;
    }
  | {
      readonly kind: "text_anchored";
      readonly tweetId: TweetId;
      readonly authorId: AuthorId;
      readonly tokenCount: TokenCount;
      readonly embedding: UnitVec256;
      readonly cjkCharRatio?: number | undefined;
      readonly rawText?: string | undefined;
      readonly favoriteCount?: number | null | undefined;
    };

export type FilterReason =
  | { readonly kind: "clean"; readonly action: "keep" }
  | {
      readonly kind: "short_human_exempt";
      readonly action: "keep";
      readonly tokenCount: TokenCount;
    }
  | {
      readonly kind: "known_bot_author";
      readonly action: "dim";
      readonly authorId: AuthorId;
      readonly spamCount: number;
    }
  | {
      readonly kind: "keyword_spam";
      readonly action: "dim";
      readonly matchedPattern: string;
    }
  | {
      readonly kind: "bio_scam";
      readonly action: "dim";
      readonly matchedPattern: string;
    }
  | {
      readonly kind: "bot_summon";
      readonly action: "dim" | "suspect";
      readonly handle: string;
    }
  | {
      readonly kind: "bot_answer";
      readonly action: "dim" | "suspect";
      readonly authorId: AuthorId;
    }
  | {
      readonly kind: "blue_short_praise";
      readonly action: "dim" | "suspect";
      readonly confirmedUnfollowed: boolean;
    }
  | {
      readonly kind: "media_only_blue";
      readonly action: "dim" | "suspect";
      readonly confirmedUnfollowed: boolean;
    }
  | {
      readonly kind: "op_paraphrase";
      readonly action: "dim" | "suspect";
      readonly similarityToOp: UnitCosine;
    }
  | {
      readonly kind: "thread_clone";
      readonly action: "dim" | "suspect";
      readonly similarityToPeer: UnitCosine;
      readonly canonicalTweetId: TweetId;
      readonly clusterSize: number;
    }
  | {
      readonly kind: "siamese_bot";
      readonly action: "dim" | "suspect";
      readonly probability: Probability;
      readonly dominantLanguage: SupportedLanguage;
      readonly opFallback: boolean;
    }
  | {
      readonly kind: "model_suspect";
      readonly action: "suspect";
      readonly probability: Probability;
      readonly dominantLanguage: SupportedLanguage;
    }
  | {
      readonly kind: "user_override";
      readonly action: "keep" | "dim";
      readonly label: "spam" | "human";
    };

export interface VerdictDiagnostics {
  readonly opCosSim: UnitCosine;
  readonly maxCloneSim: UnitCosine;
  readonly siameseProbability: Probability;
  readonly authorSpamPrior: number;
  readonly authorSpamCount: number;
  readonly authorHumanCount: number;
}

export interface VerdictRecord {
  readonly tweetId: TweetId;
  readonly authorId?: AuthorId | undefined;
  readonly snippet?: string | undefined;
  readonly action: FilterReason["action"];
  readonly tier?: VerdictTier | undefined;
  readonly score: number;
  readonly tokenCount: TokenCount;
  readonly reason: FilterReason;
  readonly diagnostics?: VerdictDiagnostics | undefined;
  readonly explanation?: VerdictExplanation | undefined;
}

export interface ThresholdConfig {
  readonly minOpTokens: TokenCount;
  readonly minReplyTokens: TokenCount;
  readonly paraphraseCosineMin: UnitCosine;
  readonly exactQuoteCeiling: UnitCosine;
  readonly cloneCosineMin: UnitCosine;
  readonly siameseScoreThreshold: Probability;
  readonly passiveAggressiveC: number;
  readonly authorKnownBotMinSpam: number;
}

export const DEFAULT_THRESHOLDS: ThresholdConfig = {
  minOpTokens: OP_MIN_TOKENS as TokenCount,
  minReplyTokens: REPLY_MIN_TOKENS as TokenCount,
  paraphraseCosineMin: OP_PARAPHRASE_MIN as UnitCosine,
  exactQuoteCeiling: EXACT_QUOTE_CEILING as UnitCosine,
  cloneCosineMin: THREAD_CLONE_MIN as UnitCosine,
  siameseScoreThreshold: SIAMESE_BOT_THRESHOLD as Probability,
  passiveAggressiveC: 0.5,
  authorKnownBotMinSpam: AUTHOR_KNOWN_BOT_MIN_SPAM,
};

const AUTO_FLAG_HALF_LIFE_MS = 30 * 24 * 3600 * 1000;

export function getDecayedAutoFlags(
  stats: AuthorStats | undefined,
  nowMs: number = Date.now(),
): number {
  if (!stats) {
    return 0;
  }
  const base = stats.autoFlagDecayed ?? stats.autoFlagCount;
  if (base <= 0) {
    return 0;
  }
  if (stats.lastUpdatedAtMs === undefined || nowMs <= stats.lastUpdatedAtMs) {
    return base;
  }
  const elapsed = nowMs - stats.lastUpdatedAtMs;
  return base * Math.pow(0.5, elapsed / AUTO_FLAG_HALF_LIFE_MS);
}

export function computeAuthorSpamPrior(
  stats: AuthorStats | undefined,
  nowMs: number = Date.now(),
): number {
  if (!stats) {
    return 0;
  }
  const decayedAuto = getDecayedAutoFlags(stats, nowMs);
  const pos = stats.spamCount * 2.0 + decayedAuto * 0.15;
  const neg = stats.humanCount * 2.0;
  const denom = pos + neg + 2.0;
  return (pos - neg) / denom;
}

export interface AuthorModelFeatures {
  readonly spamRate: number;
  readonly spamVolume: number;
  readonly humanRate: number;
  readonly handleBotScore: number;
}

export function computeHandleBotPatternScore(authorId: AuthorId): number {
  const trailingMatch = /(\d+)$/.exec(authorId);
  const trailingDigits = trailingMatch?.[1]?.length ?? 0;
  let score = 0;
  if (trailingDigits >= 6) {
    score = 0.85;
  } else if (trailingDigits >= 4) {
    score = 0.55;
  } else if (trailingDigits === 3) {
    score = 0.35;
  }
  const allDigits = (authorId.match(/\d/g) ?? []).length;
  if (authorId.length > 0 && allDigits / authorId.length > 0.4) {
    score = Math.min(1, score + 0.15);
  }
  return score;
}

export function extractAuthorModelFeatures(
  authorId: AuthorId,
  stats: AuthorStats | undefined,
  nowMs: number = Date.now(),
): AuthorModelFeatures {
  const spam = stats?.spamCount ?? 0;
  const human = stats?.humanCount ?? 0;
  const decayedAuto = getDecayedAutoFlags(stats, nowMs);
  const effectiveSpam = spam + 0.15 * decayedAuto;
  const spamRate = effectiveSpam / (effectiveSpam + human + 1);
  const spamVolume = Math.tanh(0.75 * (spam + 0.1 * decayedAuto));
  const humanRate = human / (spam + human + 1);
  const handleBotScore = computeHandleBotPatternScore(authorId);
  return {
    spamRate,
    spamVolume,
    humanRate,
    handleBotScore,
  };
}

const TWEET_ID_PATTERN = /^\d{1,32}$/;
const STATUS_PATH_PATTERN = /\/status\/(\d{1,32})/;

export function parseTweetId(raw: string): TweetId | null {
  const trimmed = raw.trim();
  if (!TWEET_ID_PATTERN.test(trimmed)) {
    return null;
  }
  return trimmed as TweetId;
}

export function parseAuthorId(raw: string): AuthorId | null {
  const cleaned = raw.trim().replace(/^@+/, "").toLowerCase();
  if (cleaned.length === 0 || cleaned.length > 64 || !/^[a-z0-9_]+$/.test(cleaned)) {
    return null;
  }
  return cleaned as AuthorId;
}

export function parseThreadId(pathnameOrId: string): ThreadId | null {
  const trimmed = pathnameOrId.trim();
  if (TWEET_ID_PATTERN.test(trimmed)) {
    return trimmed as ThreadId;
  }
  const match = STATUS_PATH_PATTERN.exec(trimmed);
  if (!match || !match[1]) {
    return null;
  }
  return match[1] as ThreadId;
}

export function asTokenCount(nonPaddingTokens: number): TokenCount {
  const clamped = Math.max(0, Math.floor(Number.isFinite(nonPaddingTokens) ? nonPaddingTokens : 0));
  return clamped as TokenCount;
}

export function asUnitCosine(value: number): UnitCosine {
  const safe = Number.isFinite(value) ? Math.max(-1, Math.min(1, value)) : 0;
  return safe as UnitCosine;
}

export function asProbability(value: number): Probability {
  const safe = Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : 0;
  return safe as Probability;
}

export function formatClassificationPrompt(normalizedText: string): string {
  return `${CLASSIFICATION_TASK_PREFIX}${normalizedText.trim()}`;
}

export function truncateAndNormalize256(
  rawOutput: Float32Array,
  rowOffset = 0,
  rawHiddenDim: number = rawOutput.length,
): UnitVec256 {
  const base = rowOffset * rawHiddenDim;
  if (base + EMBEDDING_DIM > rawOutput.length) {
    throw new RangeError(
      `Expected at least ${base + EMBEDDING_DIM} floats in rawOutput, received ${rawOutput.length}`,
    );
  }
  const out = new Float32Array(EMBEDDING_DIM);
  let sumSq = 0;
  for (let i = 0; i < EMBEDDING_DIM; i++) {
    const val = rawOutput[base + i] ?? 0;
    out[i] = val;
    sumSq += val * val;
  }
  if (sumSq > 1e-12) {
    const invNorm = 1 / Math.sqrt(sumSq);
    for (let i = 0; i < EMBEDDING_DIM; i++) {
      out[i] = (out[i] ?? 0) * invNorm;
    }
  }
  return out as UnitVec256;
}

export function truncateAndNormalizeBatch256(
  rawBatch: Float32Array,
  rowCount: number,
  rawHiddenDim: number = RAW_EMBEDDING_DIM,
): UnitVec256[] {
  const results: UnitVec256[] = [];
  for (let row = 0; row < rowCount; row++) {
    results.push(truncateAndNormalize256(rawBatch, row, rawHiddenDim));
  }
  return results;
}
