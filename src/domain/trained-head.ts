import { ANCHOR_SET_V2, type AnchorSetV2, CALIBRATED_WEIGHTS_V2 } from "./calibrated-data.ts";
import { type OnlineClassifierWeights, createCalibratedGenericCentroid } from "./seeds.ts";
import {
  type CompactFeatureExtraction,
  FEATURE_FAMILIES,
  FEATURE_IDS,
  FEATURE_SIGNS,
  type PrototypeBankV2,
  toLangBiasKey,
} from "./stylometrics.ts";
import {
  type AuthorId,
  type AuthorStats,
  COMPACT_V2_DIM,
  type CompactFeatureVec,
  type FeedbackReplayEntry,
  type LangBiasKey,
  MAX_REPLAY_BUFFER_ENTRIES,
  type PresetThresholds,
  type Probability,
  type SensitivityPreset,
  type SignalContribution,
  type SignalFamily,
  asProbability,
  parseAuthorId,
  parseTweetId,
} from "./types.ts";

const LANG_BIAS_KEYS: ReadonlyArray<LangBiasKey> = ["en", "zh-Hans", "zh-Hant", "ja", "other"];
const REPLAY_HALF_LIFE_MS = 60 * 24 * 3600 * 1000;

export function sigmoid(x: number): number {
  if (x >= 0) {
    const z = Math.exp(-x);
    return 1 / (1 + z);
  }
  const z = Math.exp(x);
  return z / (1 + z);
}

export function getPresetThresholds(preset: SensitivityPreset): PresetThresholds {
  return CALIBRATED_WEIGHTS_V2.thresholds[preset];
}

export interface EffectiveV2Params {
  readonly theta: Float32Array;
  readonly theta0: Float32Array;
  readonly langBias: Readonly<Record<LangBiasKey, number>>;
  readonly langBias0: Readonly<Record<LangBiasKey, number>>;
  readonly fixedCoefs: { readonly protoMargin: number; readonly generic: number };
  readonly protoBank: PrototypeBankV2;
}

export function getEffectiveV2Params(classifier: OnlineClassifierWeights): EffectiveV2Params {
  const theta0 =
    classifier.theta0 && classifier.theta0.length === COMPACT_V2_DIM
      ? classifier.theta0
      : new Float32Array(CALIBRATED_WEIGHTS_V2.theta0);
  const theta =
    classifier.theta && classifier.theta.length === COMPACT_V2_DIM
      ? classifier.theta
      : new Float32Array(theta0);
  const langBias0 = CALIBRATED_WEIGHTS_V2.langBias;
  const langBias = classifier.langBias ?? langBias0;
  const fixedCoefs = classifier.fixedCoefs ?? CALIBRATED_WEIGHTS_V2.fixedCoefs;

  const slop: Array<ArrayLike<number>> = [];
  const human: Array<ArrayLike<number>> = [];
  if (classifier.seedPrototypes.length > 0) {
    for (const p of classifier.seedPrototypes) {
      if (p.label === 1) slop.push(p.embedding);
      else human.push(p.embedding);
    }
  }
  if (slop.length === 0 || human.length === 0) {
    for (const p of CALIBRATED_WEIGHTS_V2.prototypes) {
      if (p.label === 1) slop.push(p.embedding);
      else human.push(p.embedding);
    }
  }

  const genericCentroid = classifier.genericCentroid ?? createCalibratedGenericCentroid();

  return {
    theta,
    theta0,
    langBias,
    langBias0,
    fixedCoefs,
    protoBank: { slop, human, genericCentroid },
  };
}

export function standardizeCompactV2(vec: CompactFeatureVec): Float32Array {
  const mu = CALIBRATED_WEIGHTS_V2.mu;
  const sigma = CALIBRATED_WEIGHTS_V2.sigma;
  const z = new Float32Array(COMPACT_V2_DIM);
  for (let j = 0; j < COMPACT_V2_DIM; j++) {
    const s = sigma[j] ?? 1;
    z[j] = ((vec[j] ?? 0) - (mu[j] ?? 0)) / (s > 1e-6 ? s : 1);
  }
  return z;
}

export interface CompactV2ScoreResult {
  readonly probability: Probability;
  readonly logit: number;
  readonly baseHumanLogit: number;
  readonly fixedTerm: number;
  readonly z: Float32Array;
  readonly familyLogits: Readonly<Record<SignalFamily, number>>;
  readonly corroboratingFamilies: ReadonlyArray<SignalFamily>;
  readonly top: ReadonlyArray<SignalContribution>;
  readonly counter: ReadonlyArray<SignalContribution>;
}

export function scoreCompactV2(
  feat: CompactFeatureExtraction,
  classifier: OnlineClassifierWeights,
): CompactV2ScoreResult {
  const params = getEffectiveV2Params(classifier);
  const z = standardizeCompactV2(feat.vector);
  const lk = toLangBiasKey(feat.replyLang);
  const mu = CALIBRATED_WEIGHTS_V2.mu;
  const sigma = CALIBRATED_WEIGHTS_V2.sigma;
  const zHuman = CALIBRATED_WEIGHTS_V2.zHuman;

  const fixedTerm =
    params.fixedCoefs.protoMargin * (feat.vector[4] ?? 0) +
    params.fixedCoefs.generic * (feat.vector[5] ?? 0);

  const xHuman4 = (mu[4] ?? 0) + (zHuman[4] ?? 0) * (sigma[4] ?? 1);
  const xHuman5 = (mu[5] ?? 0) + (zHuman[5] ?? 0) * (sigma[5] ?? 1);

  let logit = (params.langBias[lk] ?? 0) + fixedTerm;
  let baseHumanLogit =
    (params.langBias[lk] ?? 0) +
    params.fixedCoefs.protoMargin * xHuman4 +
    params.fixedCoefs.generic * xHuman5;

  const familyLogits: Record<SignalFamily, number> = {
    relational: 0,
    embedding: 0,
    style: 0,
    crosslang: 0,
    summon: 0,
    meta: 0,
    author: 0,
    rule: 0,
  };

  const positives: SignalContribution[] = [];
  const negatives: SignalContribution[] = [];

  for (let j = 0; j < COMPACT_V2_DIM; j++) {
    const id = FEATURE_IDS[j]!;
    const family = FEATURE_FAMILIES[j]!;
    const val = feat.vector[j] ?? 0;
    let contrib = 0;

    if (j === 4) {
      contrib = params.fixedCoefs.protoMargin * (val - xHuman4);
    } else if (j === 5) {
      contrib = params.fixedCoefs.generic * (val - xHuman5);
    } else {
      const wj = params.theta[j] ?? 0;
      const zj = z[j] ?? 0;
      const zhj = zHuman[j] ?? 0;
      logit += wj * zj;
      baseHumanLogit += wj * zhj;
      contrib = wj * (zj - zhj);
    }

    familyLogits[family] += contrib;

    if (contrib >= 0.15) {
      const ev = feat.evidence[id];
      positives.push(
        ev !== undefined
          ? { id, family, value: val, contribution: contrib, evidence: ev }
          : { id, family, value: val, contribution: contrib },
      );
    } else if (contrib <= -0.15) {
      const ev = feat.evidence[id];
      negatives.push(
        ev !== undefined
          ? { id, family, value: val, contribution: contrib, evidence: ev }
          : { id, family, value: val, contribution: contrib },
      );
    }
  }

  positives.sort((a, b) => b.contribution - a.contribution);
  negatives.sort((a, b) => a.contribution - b.contribution);

  const corroboratingFamilies = (
    Object.entries(familyLogits) as Array<[SignalFamily, number]>
  )
    .filter(([fam, sum]) => fam !== "rule" && sum >= 0.5)
    .map(([fam]) => fam);

  return {
    probability: asProbability(sigmoid(logit)),
    logit,
    baseHumanLogit,
    fixedTerm,
    z,
    familyLogits,
    corroboratingFamilies,
    top: positives.slice(0, 5),
    counter: negatives.slice(0, 3),
  };
}

// ================= Anchor-Set Rollback Acceptance Gate (§6) =================
export interface AnchorEvaluation {
  readonly precisionDim: number;
  readonly humanFprDim: number;
  readonly recallDim: number;
  readonly passed: boolean;
}

export function evaluateAnchorSet(
  theta: Float32Array,
  langBias: Readonly<Record<LangBiasKey, number>>,
  anchorSet: AnchorSetV2 = ANCHOR_SET_V2,
): AnchorEvaluation {
  const tau = anchorSet.dimThreshold;
  let tp = 0;
  let fp = 0;
  let fn = 0;
  let tn = 0;

  for (const item of anchorSet.items) {
    let logit = (langBias[item.lang] ?? 0) + item.fixed;
    for (let j = 0; j < COMPACT_V2_DIM; j++) {
      logit += (theta[j] ?? 0) * (item.z[j] ?? 0);
    }
    const p = sigmoid(logit);
    if (p >= tau) {
      if (item.y === 1) tp += 1;
      else fp += 1;
    } else {
      if (item.y === 1) fn += 1;
      else tn += 1;
    }
  }

  const precisionDim = tp + fp > 0 ? tp / (tp + fp) : 1;
  const humanFprDim = fp + tn > 0 ? fp / (fp + tn) : 0;
  const recallDim = tp + fn > 0 ? tp / (tp + fn) : 0;
  const passed =
    precisionDim >= anchorSet.targetPrecisionDim - 0.02 &&
    humanFprDim <= anchorSet.baselineHumanFprDim + 0.02;

  return { precisionDim, humanFprDim, recallDim, passed };
}

// ================= Replay-Buffer Regularized Online Learner (§6) =================
export function insertIntoReplayBuffer(
  existing: ReadonlyArray<FeedbackReplayEntry>,
  entry: FeedbackReplayEntry,
  maxCapacity: number = MAX_REPLAY_BUFFER_ENTRIES,
): FeedbackReplayEntry[] {
  // Replace prior label for the exact same tweetId if present
  const next = existing.filter((item) => item.tweetId !== entry.tweetId);
  next.push(entry);

  // Enforce max 8% per topicHash bucket cap when buffer exceeds bucket limit
  const maxPerBucket = Math.max(1, Math.floor(maxCapacity * 0.08));
  const bucketIndices = new Map<number, number[]>();
  for (let i = 0; i < next.length; i++) {
    const h = next[i]!.topicHash & 0xff;
    const list = bucketIndices.get(h) ?? [];
    list.push(i);
    bucketIndices.set(h, list);
  }

  const toRemove = new Set<number>();
  for (const [, indices] of bucketIndices.entries()) {
    if (indices.length > maxPerBucket) {
      const sorted = [...indices].sort((a, b) => next[a]!.tsMs - next[b]!.tsMs);
      const excess = indices.length - maxPerBucket;
      for (let k = 0; k < excess; k++) {
        // Never evict the freshly inserted entry at the tail
        const victim = sorted[k]!;
        if (victim !== next.length - 1) {
          toRemove.add(victim);
        }
      }
    }
  }

  let pruned = toRemove.size > 0 ? next.filter((_, i) => !toRemove.has(i)) : next;

  // On capacity overflow, evict the oldest item in the largest topic bucket
  while (pruned.length > maxCapacity) {
    const counts = new Map<number, number>();
    for (const item of pruned) {
      const h = item.topicHash & 0xff;
      counts.set(h, (counts.get(h) ?? 0) + 1);
    }
    let largestBucket = -1;
    let largestCount = -1;
    for (const [h, cnt] of counts.entries()) {
      if (cnt > largestCount) {
        largestCount = cnt;
        largestBucket = h;
      }
    }
    let oldestIdx = -1;
    let oldestTs = Infinity;
    for (let i = 0; i < pruned.length - 1; i++) {
      if ((pruned[i]!.topicHash & 0xff) === largestBucket && pruned[i]!.tsMs < oldestTs) {
        oldestTs = pruned[i]!.tsMs;
        oldestIdx = i;
      }
    }
    if (oldestIdx < 0) oldestIdx = 0;
    pruned = pruned.filter((_, i) => i !== oldestIdx);
  }

  return pruned;
}

export interface ReplayUpdateResult {
  readonly nextWeights: OnlineClassifierWeights;
  readonly rolledBack: boolean;
  readonly anchorEval: AnchorEvaluation;
}

export function trainFromReplayBuffer(
  current: OnlineClassifierWeights,
  replayBuffer: ReadonlyArray<FeedbackReplayEntry>,
  options?: {
    readonly newestTweetId?: string | undefined;
    readonly nowMs?: number | undefined;
    readonly anchorSet?: AnchorSetV2 | undefined;
    readonly steps?: number | undefined;
    readonly lr?: number | undefined;
    readonly maxDrift?: number | undefined;
  },
): ReplayUpdateResult {
  const nowMs = options?.nowMs ?? Date.now();
  const anchorSet = options?.anchorSet ?? ANCHOR_SET_V2;
  const steps = options?.steps ?? 16;
  const lr = options?.lr ?? 0.1;
  const maxDrift = options?.maxDrift ?? 1.5;

  const params = getEffectiveV2Params(current);
  const theta0 = params.theta0;
  const langBias0 = params.langBias0;

  if (replayBuffer.length === 0) {
    const resetTheta = new Float32Array(theta0);
    const resetLangBias: Record<LangBiasKey, number> = {
      en: langBias0.en,
      "zh-Hans": langBias0["zh-Hans"],
      "zh-Hant": langBias0["zh-Hant"],
      ja: langBias0.ja,
      other: langBias0.other,
    };
    const anchorEval = evaluateAnchorSet(resetTheta, resetLangBias, anchorSet);
    return {
      nextWeights: {
        ...current,
        schemaVersion: 2,
        theta0Hash: CALIBRATED_WEIGHTS_V2.theta0Hash,
        theta: resetTheta,
        theta0,
        langBias: resetLangBias,
        fixedCoefs: params.fixedCoefs,
        replayBuffer: [],
        weightsRevision: (current.weightsRevision ?? 1) + 1,
      },
      rolledBack: false,
      anchorEval,
    };
  }

  // Class counts and topicHash bucket counts for sample weighting ω_i (§6)
  let posCount = 0;
  let negCount = 0;
  const bucketCounts = new Map<number, number>();
  for (const item of replayBuffer) {
    if (item.y === 1) posCount += 1;
    else negCount += 1;
    const h = item.topicHash & 0xff;
    bucketCounts.set(h, (bucketCounts.get(h) ?? 0) + 1);
  }
  const totalN = replayBuffer.length;
  const posBalance = posCount > 0 ? totalN / (2 * posCount) : 1;
  const negBalance = negCount > 0 ? totalN / (2 * negCount) : 1;

  const sampleWeights = replayBuffer.map((item) => {
    const cb = item.y === 1 ? posBalance : negBalance;
    const bCnt = bucketCounts.get(item.topicHash & 0xff) ?? 1;
    const ageMs = Math.max(0, nowMs - item.tsMs);
    const ageDecay = Math.pow(0.5, ageMs / REPLAY_HALF_LIFE_MS);
    return cb * (1 / Math.sqrt(bCnt)) * ageDecay;
  });

  // Build deterministic class-balanced batch indices (up to 32, newest always included)
  const posIdx: number[] = [];
  const negIdx: number[] = [];
  let newestIdx = replayBuffer.length - 1;
  for (let i = 0; i < replayBuffer.length; i++) {
    if (options?.newestTweetId !== undefined && replayBuffer[i]!.tweetId === options.newestTweetId) {
      newestIdx = i;
    }
    if (replayBuffer[i]!.y === 1) posIdx.push(i);
    else negIdx.push(i);
  }

  // Start optimization from current theta anchored toward theta0
  const theta = new Float32Array(params.theta);
  const langBias: Record<LangBiasKey, number> = {
    en: params.langBias.en,
    "zh-Hans": params.langBias["zh-Hans"],
    "zh-Hant": params.langBias["zh-Hant"],
    ja: params.langBias.ja,
    other: params.langBias.other,
  };

  const lambda = Math.max(0.05, 50 / (50 + replayBuffer.length));
  const accumG = new Float32Array(COMPACT_V2_DIM);
  const accumLb: Record<LangBiasKey, number> = {
    en: 0,
    "zh-Hans": 0,
    "zh-Hant": 0,
    ja: 0,
    other: 0,
  };
  const gradW = new Float32Array(COMPACT_V2_DIM);
  const eps = 1e-6;

  for (let step = 0; step < steps; step++) {
    // Select batch of up to 32 balanced items, always including newestIdx
    let batch: number[];
    if (replayBuffer.length <= 32) {
      batch = replayBuffer.map((_, i) => i);
    } else {
      const picked = new Set<number>([newestIdx]);
      const half = 16;
      for (let k = 0; k < half && posIdx.length > 0; k++) {
        picked.add(posIdx[(step * 7 + k) % posIdx.length]!);
      }
      for (let k = 0; k < half && negIdx.length > 0; k++) {
        picked.add(negIdx[(step * 7 + k) % negIdx.length]!);
      }
      batch = Array.from(picked);
    }

    gradW.fill(0);
    const gradLb: Record<LangBiasKey, number> = {
      en: 0,
      "zh-Hans": 0,
      "zh-Hant": 0,
      ja: 0,
      other: 0,
    };
    let weightSum = 0;

    for (const idx of batch) {
      const item = replayBuffer[idx]!;
      const wi = sampleWeights[idx]!;
      weightSum += wi;

      let logit = (langBias[item.lang] ?? 0) + item.fixed;
      for (let j = 0; j < COMPACT_V2_DIM; j++) {
        logit += (theta[j] ?? 0) * (item.z[j] ?? 0);
      }
      const err = wi * (sigmoid(logit) - item.y);
      gradLb[item.lang] += err;
      for (let j = 0; j < COMPACT_V2_DIM; j++) {
        if (j === 4 || j === 5) continue; // fixedCoefs frozen
        gradW[j] = (gradW[j] ?? 0) + err * (item.z[j] ?? 0);
      }
    }

    const invW = weightSum > 1e-8 ? 1 / weightSum : 1;

    for (let j = 0; j < COMPACT_V2_DIM; j++) {
      if (j === 4 || j === 5) {
        theta[j] = 0;
        continue;
      }
      const baseJ = theta0[j] ?? 0;
      const g = (gradW[j] ?? 0) * invW + lambda * ((theta[j] ?? 0) - baseJ);
      accumG[j] = (accumG[j] ?? 0) + g * g;
      let next = (theta[j] ?? 0) - (lr / Math.sqrt((accumG[j] ?? 0) + eps)) * g;

      // Project onto §3b sign constraints
      const sign = FEATURE_SIGNS[j] ?? 0;
      if (sign === 1 && next < 0) next = 0;
      if (sign === -1 && next > 0) next = 0;

      // Clip ||theta - theta0||_inf <= 1.5
      const delta = next - baseJ;
      if (delta > maxDrift) next = baseJ + maxDrift;
      else if (delta < -maxDrift) next = baseJ - maxDrift;

      theta[j] = next;
    }

    const lbLr = lr * 0.3;
    for (const lk of LANG_BIAS_KEYS) {
      const baseB = langBias0[lk] ?? 0;
      const g = gradLb[lk] * invW + lambda * (langBias[lk] - baseB);
      accumLb[lk] += g * g;
      let nextB = langBias[lk] - (lbLr / Math.sqrt(accumLb[lk] + eps)) * g;
      const delta = nextB - baseB;
      if (delta > maxDrift) nextB = baseB + maxDrift;
      else if (delta < -maxDrift) nextB = baseB - maxDrift;
      langBias[lk] = nextB;
    }
  }

  // Anchor-set acceptance gate (§6)
  const anchorEval = evaluateAnchorSet(theta, langBias, anchorSet);
  if (!anchorEval.passed) {
    return {
      nextWeights: {
        ...current,
        schemaVersion: 2,
        theta0Hash: CALIBRATED_WEIGHTS_V2.theta0Hash,
        theta: params.theta,
        theta0,
        langBias: params.langBias,
        fixedCoefs: params.fixedCoefs,
        rollbackCount: (current.rollbackCount ?? 0) + 1,
        updateCount: current.updateCount + 1,
      },
      rolledBack: true,
      anchorEval,
    };
  }

  return {
    nextWeights: {
      ...current,
      schemaVersion: 2,
      theta0Hash: CALIBRATED_WEIGHTS_V2.theta0Hash,
      theta,
      theta0,
      langBias,
      fixedCoefs: params.fixedCoefs,
      replayBuffer,
      updateCount: current.updateCount + 1,
      weightsRevision: (current.weightsRevision ?? 1) + 1,
    },
    rolledBack: false,
    anchorEval,
  };
}

export function updateOnlineWeightsReplay(
  current: OnlineClassifierWeights,
  entry: FeedbackReplayEntry,
  options?: {
    readonly nowMs?: number | undefined;
    readonly anchorSet?: AnchorSetV2 | undefined;
    readonly steps?: number | undefined;
    readonly lr?: number | undefined;
    readonly maxDrift?: number | undefined;
  },
): ReplayUpdateResult {
  const nextBuffer = insertIntoReplayBuffer(current.replayBuffer ?? [], entry);
  return trainFromReplayBuffer(current, nextBuffer, {
    ...options,
    newestTweetId: entry.tweetId,
  });
}

// ================= Feedback Export / Import Portability =================
export interface FeedbackExportPayload {
  readonly schemaVersion: 2;
  readonly exportedAt: string;
  readonly theta0Hash: string;
  readonly replayBuffer: ReadonlyArray<FeedbackReplayEntry>;
  readonly authorStatsEntries: ReadonlyArray<readonly [AuthorId, AuthorStats]>;
}

export function serializeFeedbackExport(weights: OnlineClassifierWeights): string {
  const payload: FeedbackExportPayload = {
    schemaVersion: 2,
    exportedAt: new Date().toISOString(),
    theta0Hash: weights.theta0Hash ?? CALIBRATED_WEIGHTS_V2.theta0Hash,
    replayBuffer: weights.replayBuffer ?? [],
    authorStatsEntries: Array.from(weights.authorStats.entries()),
  };
  return JSON.stringify(payload, null, 2);
}

export function parseFeedbackImport(rawJson: string): {
  readonly replayBuffer: FeedbackReplayEntry[];
  readonly authorStats: Map<AuthorId, AuthorStats>;
} | null {
  let parsed: unknown;
  try {
    parsed = JSON.parse(rawJson);
  } catch {
    return null;
  }
  if (typeof parsed !== "object" || parsed === null) return null;
  const obj = parsed as Record<string, unknown>;
  if (obj.schemaVersion !== 2 || !Array.isArray(obj.replayBuffer)) return null;

  const validLangs = new Set<string>(LANG_BIAS_KEYS);
  const replayBuffer: FeedbackReplayEntry[] = [];
  for (const rawEntry of obj.replayBuffer) {
    if (typeof rawEntry !== "object" || rawEntry === null) return null;
    const e = rawEntry as Record<string, unknown>;
    const tweetId = typeof e.tweetId === "string" ? parseTweetId(e.tweetId) : null;
    if (
      tweetId === null ||
      !Array.isArray(e.z) ||
      e.z.length !== COMPACT_V2_DIM ||
      !e.z.every((v) => typeof v === "number" && Number.isFinite(v)) ||
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
      return null;
    }
    const authorId =
      typeof e.authorId === "string" ? (parseAuthorId(e.authorId) ?? undefined) : undefined;
    replayBuffer.push({
      tweetId,
      ...(authorId !== undefined ? { authorId } : {}),
      z: e.z as number[],
      fixed: e.fixed,
      y: e.y,
      lang: e.lang as LangBiasKey,
      tsMs: e.tsMs,
      topicHash: Math.floor(e.topicHash) & 0xff,
      source: e.source,
    });
  }

  const authorStats = new Map<AuthorId, AuthorStats>();
  if (Array.isArray(obj.authorStatsEntries)) {
    for (const pair of obj.authorStatsEntries) {
      if (!Array.isArray(pair) || pair.length !== 2) continue;
      const [rawId, rawStats] = pair as [unknown, unknown];
      const authorId = typeof rawId === "string" ? parseAuthorId(rawId) : null;
      if (authorId === null || typeof rawStats !== "object" || rawStats === null) continue;
      const s = rawStats as Record<string, unknown>;
      if (typeof s.spamCount !== "number" || typeof s.humanCount !== "number") continue;
      authorStats.set(authorId, {
        spamCount: Math.max(0, Math.floor(s.spamCount)),
        humanCount: Math.max(0, Math.floor(s.humanCount)),
        autoFlagCount:
          typeof s.autoFlagCount === "number" ? Math.max(0, Math.floor(s.autoFlagCount)) : 0,
        ...(typeof s.lastUpdatedAtMs === "number" ? { lastUpdatedAtMs: s.lastUpdatedAtMs } : {}),
        ...(typeof s.autoFlagDecayed === "number" ? { autoFlagDecayed: s.autoFlagDecayed } : {}),
      });
    }
  }

  return {
    replayBuffer: replayBuffer.slice(-MAX_REPLAY_BUFFER_ENTRIES),
    authorStats,
  };
}
