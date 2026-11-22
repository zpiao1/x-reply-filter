import { CALIBRATED_WEIGHTS_V2 } from "./calibrated-data.ts";
import {
  SPAM_PATTERNS_V2,
  detectDominantLanguage,
  extractDomSignalsFromText,
  matchMultilingualSpamKeyword,
} from "./stylometrics.ts";
import {
  type AuthorId,
  type AuthorStats,
  DOM_SIGNAL_DIM,
  EMBEDDING_DIM,
  type FeedbackReplayEntry,
  type LangBiasKey,
  MAX_AUTO_FLAGGED_TWEET_IDS,
  SIAMESE_RELATIONAL_DIM,
  SIAMESE_TOTAL_DIM,
  type SupportedLanguage,
  type TweetId,
  type UndoRecord,
  type UnitVec256,
  getDecayedAutoFlags,
  truncateAndNormalize256,
} from "./types.ts";

export {
  SPAM_PATTERNS_V2 as SPAM_PATTERNS,
  detectDominantLanguage,
  extractDomSignalsFromText,
  matchMultilingualSpamKeyword,
};

export interface SeedExample {
  readonly lang: SupportedLanguage;
  readonly langV2?: "en" | "zh-Hans" | "zh-Hant" | "ja" | undefined;
  readonly label: 1 | -1;
  readonly text: string;
}

export interface SeedPrototype {
  readonly lang: SupportedLanguage;
  readonly label: 1 | -1;
  readonly embedding: UnitVec256;
}

export interface OnlineClassifierWeights {
  readonly weights: Float32Array;
  readonly bias: number;
  readonly seedPrototypes: ReadonlyArray<SeedPrototype>;
  readonly updateCount: number;
  readonly authorStats: ReadonlyMap<AuthorId, AuthorStats>;
  // v2 fields (optional on interface for v1 IndexedDB / gpu-host compatibility)
  readonly schemaVersion?: 2 | undefined;
  readonly theta0Hash?: string | undefined;
  readonly theta?: Float32Array | undefined;
  readonly theta0?: Float32Array | undefined;
  readonly langBias?: Readonly<Record<LangBiasKey, number>> | undefined;
  readonly fixedCoefs?: { readonly protoMargin: number; readonly generic: number } | undefined;
  readonly genericCentroid?: UnitVec256 | undefined;
  readonly replayBuffer?: ReadonlyArray<FeedbackReplayEntry> | undefined;
  readonly rollbackCount?: number | undefined;
  readonly weightsRevision?: number | undefined;
  readonly undoHistory?: ReadonlyArray<UndoRecord> | undefined;
  readonly recordedAutoFlagIds?: ReadonlySet<TweetId> | undefined;
  readonly dimsReviewedCount?: number | undefined;
  readonly dimsOverriddenHumanCount?: number | undefined;
}

export const SEED_CORPUS: ReadonlyArray<SeedExample> = CALIBRATED_WEIGHTS_V2.prototypes.map((p) => ({
  lang: p.lang,
  langV2: p.langV2,
  label: p.label,
  text: p.text,
}));

export function createCalibratedSeedPrototypes(): ReadonlyArray<SeedPrototype> {
  return CALIBRATED_WEIGHTS_V2.prototypes.map((p) => ({
    lang: p.lang,
    label: p.label,
    embedding: truncateAndNormalize256(new Float32Array(p.embedding), 0, EMBEDDING_DIM),
  }));
}

export function createDeterministicSeedPrototypes(): ReadonlyArray<SeedPrototype> {
  return createCalibratedSeedPrototypes();
}

export function createCalibratedGenericCentroid(): UnitVec256 {
  return truncateAndNormalize256(
    new Float32Array(CALIBRATED_WEIGHTS_V2.genericCentroid),
    0,
    EMBEDDING_DIM,
  );
}

export function createInitialClassifierWeights(
  seedPrototypes: ReadonlyArray<SeedPrototype> = createCalibratedSeedPrototypes(),
  authorStats: ReadonlyMap<AuthorId, AuthorStats> = new Map(),
): OnlineClassifierWeights {
  // Legacy v1 778d weights preserved for backward compatibility with buildSiameseFeatures/scoreSiameseHead
  const weights = new Float32Array(SIAMESE_TOTAL_DIM);
  weights[SIAMESE_RELATIONAL_DIM + 0] = 0.35;
  weights[SIAMESE_RELATIONAL_DIM + 1] = 0.25;
  weights[SIAMESE_RELATIONAL_DIM + 2] = 0.95;
  weights[SIAMESE_RELATIONAL_DIM + 3] = 0.85;
  weights[SIAMESE_RELATIONAL_DIM + 4] = 0.0;
  weights[SIAMESE_RELATIONAL_DIM + 5] = 1.8;

  const authorBase = SIAMESE_RELATIONAL_DIM + DOM_SIGNAL_DIM;
  weights[authorBase + 0] = 0.75;
  weights[authorBase + 1] = 0.55;
  weights[authorBase + 2] = -1.1;
  weights[authorBase + 3] = 0.45;

  const theta0 = new Float32Array(CALIBRATED_WEIGHTS_V2.theta0);
  const theta = new Float32Array(theta0);
  const langBias: Record<LangBiasKey, number> = {
    en: CALIBRATED_WEIGHTS_V2.langBias.en,
    "zh-Hans": CALIBRATED_WEIGHTS_V2.langBias["zh-Hans"],
    "zh-Hant": CALIBRATED_WEIGHTS_V2.langBias["zh-Hant"],
    ja: CALIBRATED_WEIGHTS_V2.langBias.ja,
    other: CALIBRATED_WEIGHTS_V2.langBias.other,
  };

  return {
    weights,
    bias: -0.35,
    seedPrototypes,
    updateCount: 0,
    authorStats,
    schemaVersion: 2,
    theta0Hash: CALIBRATED_WEIGHTS_V2.theta0Hash,
    theta,
    theta0,
    langBias,
    fixedCoefs: { ...CALIBRATED_WEIGHTS_V2.fixedCoefs },
    genericCentroid: createCalibratedGenericCentroid(),
    replayBuffer: [],
    rollbackCount: 0,
    weightsRevision: 1,
    undoHistory: [],
    recordedAutoFlagIds: new Set<TweetId>(),
  };
}

export interface RecordAuthorOutcomeOptions {
  readonly tweetId?: TweetId | undefined;
  readonly nowMs?: number | undefined;
}

export function recordAuthorOutcome(
  current: OnlineClassifierWeights,
  authorId: AuthorId,
  outcome: "user_spam" | "user_human" | "auto_flag",
  options?: RecordAuthorOutcomeOptions,
): OnlineClassifierWeights {
  const nowMs = options?.nowMs ?? Date.now();
  const tweetId = options?.tweetId;

  // Deduplicate auto-flags per tweetId (capped at 5,000 entries per #6 §6)
  let nextAutoFlagIds = current.recordedAutoFlagIds;
  if (outcome === "auto_flag" && tweetId !== undefined) {
    if (nextAutoFlagIds?.has(tweetId)) {
      return current;
    }
    const updatedSet = new Set<TweetId>(nextAutoFlagIds ?? []);
    if (updatedSet.size >= MAX_AUTO_FLAGGED_TWEET_IDS) {
      const oldest = updatedSet.values().next().value;
      if (oldest !== undefined) {
        updatedSet.delete(oldest);
      }
    }
    updatedSet.add(tweetId);
    nextAutoFlagIds = updatedSet;
  }

  const nextMap = new Map(current.authorStats);
  const prev = nextMap.get(authorId) ?? {
    spamCount: 0,
    humanCount: 0,
    autoFlagCount: 0,
    lastUpdatedAtMs: nowMs,
    autoFlagDecayed: 0,
  };
  const decayedPrev = getDecayedAutoFlags(prev, nowMs);

  const updated: AuthorStats =
    outcome === "user_spam"
      ? {
          spamCount: prev.spamCount + 1,
          humanCount: prev.humanCount,
          autoFlagCount: prev.autoFlagCount,
          lastUpdatedAtMs: nowMs,
          autoFlagDecayed: decayedPrev,
        }
      : outcome === "user_human"
        ? {
            spamCount: prev.spamCount,
            humanCount: prev.humanCount + 1,
            autoFlagCount: prev.autoFlagCount,
            lastUpdatedAtMs: nowMs,
            autoFlagDecayed: decayedPrev,
          }
        : {
            spamCount: prev.spamCount,
            humanCount: prev.humanCount,
            autoFlagCount: prev.autoFlagCount + 1,
            lastUpdatedAtMs: nowMs,
            autoFlagDecayed: decayedPrev + 1,
          };

  nextMap.set(authorId, updated);
  return {
    ...current,
    authorStats: nextMap,
    recordedAutoFlagIds: nextAutoFlagIds,
  };
}
