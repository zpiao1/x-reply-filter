import type { BenchmarkCase } from "./dataset-humans.ts";
import {
  COMPACT_V2_DIM,
  type CompactFeatureExtraction,
  FEATURE_FAMILIES,
  FEATURE_IDS,
  FEATURE_SIGNS,
  type LangBiasKey,
  type PolicyToggles,
  type PresetName,
  type PresetThresholds,
  type SignalFamily,
  type VerdictV2Result,
  DEFAULT_TOGGLES,
  evaluateVerdictRulesV2,
  toLangBiasKey,
} from "./features-v2.ts";

export interface StandardizedScaler {
  readonly mu: Float32Array;
  readonly sigma: Float32Array;
  readonly zHuman: Float32Array;
  readonly xHuman: Float32Array;
}

export interface TrainedLinearModelV2 {
  readonly scaler: StandardizedScaler;
  readonly theta: Float32Array; // length 58; indices 4 & 5 moved to fixedCoefs when split
  readonly rawWeights: Float32Array; // full standardized weights before splitting fixedCoefs
  readonly langBias: Record<LangBiasKey, number>;
  readonly fixedCoefs: {
    readonly protoMargin: number;
    readonly generic: number;
  };
  readonly embeddingBlock?: {
    readonly dim: number;
    readonly frozen: true;
    readonly weights: Float32Array;
    readonly bias: number;
    readonly alpha: number;
  };
}

const LANG_KEYS: ReadonlyArray<LangBiasKey> = ["en", "zh-Hans", "zh-Hant", "ja", "other"];

export function sigmoid(x: number): number {
  if (x >= 0) {
    const z = Math.exp(-x);
    return 1 / (1 + z);
  }
  const z = Math.exp(x);
  return z / (1 + z);
}

/**
 * Deterministic Stratified Group K-Fold split grouped by `threadId`
 * and stratified by `(expectedLabel, lang)`.
 * Guarantees ZERO thread leakage and keeps thread-clone clusters intact inside a single fold.
 */
export function makeStratifiedGroupKFold(
  cases: ReadonlyArray<BenchmarkCase>,
  k = 5,
): number[][] {
  // Group indices by threadId
  const groups = new Map<string, number[]>();
  for (let i = 0; i < cases.length; i++) {
    const tid = cases[i]!.threadId;
    const list = groups.get(tid) ?? [];
    list.push(i);
    groups.set(tid, list);
  }

  // Sort groups: largest first, then deterministic by stratum + threadId
  const groupList = Array.from(groups.entries()).map(([tid, idxs]) => {
    const first = cases[idxs[0]!]!;
    return {
      tid,
      idxs,
      stratum: `${first.expectedLabel}:${first.lang}`,
      label: first.expectedLabel,
    };
  });
  groupList.sort((a, b) => {
    if (b.idxs.length !== a.idxs.length) return b.idxs.length - a.idxs.length;
    if (a.stratum !== b.stratum) return a.stratum.localeCompare(b.stratum);
    return a.tid.localeCompare(b.tid);
  });

  const folds: number[][] = Array.from({ length: k }, () => []);
  const foldStratumCounts: Array<Record<string, number>> = Array.from({ length: k }, () => ({}));
  const foldLabelCounts: Array<Record<string, number>> = Array.from({ length: k }, () => ({
    bot: 0,
    human: 0,
  }));

  for (const g of groupList) {
    let bestFold = 0;
    let bestScore = Infinity;
    for (let f = 0; f < k; f++) {
      const sc = foldStratumCounts[f]![g.stratum] ?? 0;
      const lc = foldLabelCounts[f]![g.label] ?? 0;
      const total = folds[f]!.length;
      // Prioritize stratum balance, then label balance, then total size
      const score = sc * 1000 + lc * 20 + total;
      if (score < bestScore) {
        bestScore = score;
        bestFold = f;
      }
    }
    folds[bestFold]!.push(...g.idxs);
    foldStratumCounts[bestFold]![g.stratum] =
      (foldStratumCounts[bestFold]![g.stratum] ?? 0) + g.idxs.length;
    foldLabelCounts[bestFold]![g.label] =
      (foldLabelCounts[bestFold]![g.label] ?? 0) + g.idxs.length;
  }

  return folds;
}

export function fitScaler(
  feats: ReadonlyArray<CompactFeatureExtraction>,
  labels: ReadonlyArray<number>,
  indices: ReadonlyArray<number>,
): StandardizedScaler {
  const d = COMPACT_V2_DIM;
  const mu = new Float32Array(d);
  const sigma = new Float32Array(d);
  const xHuman = new Float32Array(d);
  const zHuman = new Float32Array(d);
  const n = Math.max(1, indices.length);

  let nHuman = 0;
  for (const idx of indices) {
    const v = feats[idx]!.vector;
    const isH = labels[idx] === 0;
    if (isH) nHuman += 1;
    for (let j = 0; j < d; j++) {
      mu[j] = mu[j]! + v[j]!;
      if (isH) xHuman[j] = xHuman[j]! + v[j]!;
    }
  }
  for (let j = 0; j < d; j++) {
    mu[j] = mu[j]! / n;
    xHuman[j] = nHuman > 0 ? xHuman[j]! / nHuman : mu[j]!;
  }

  for (const idx of indices) {
    const v = feats[idx]!.vector;
    for (let j = 0; j < d; j++) {
      const diff = v[j]! - mu[j]!;
      sigma[j] = sigma[j]! + diff * diff;
    }
  }
  for (let j = 0; j < d; j++) {
    const std = Math.sqrt(sigma[j]! / n);
    sigma[j] = std > 1e-3 ? std : 1.0;
    zHuman[j] = (xHuman[j]! - mu[j]!) / sigma[j]!;
  }

  return { mu, sigma, zHuman, xHuman };
}

export function standardizeVector(v: Float32Array, scaler: StandardizedScaler): Float32Array {
  const z = new Float32Array(v.length);
  for (let j = 0; j < v.length; j++) {
    z[j] = (v[j]! - scaler.mu[j]!) / scaler.sigma[j]!;
  }
  return z;
}

/**
 * Fits L2-regularized Logistic Regression on arbitrary dense matrix X (n x d)
 * via Nesterov-accelerated projected gradient descent (exact convex optimum).
 */
export function fitDenseLogistic(
  X: ReadonlyArray<Float32Array>,
  y: ReadonlyArray<number>,
  indices: ReadonlyArray<number>,
  lambda: number,
  signs?: ReadonlyArray<1 | -1 | 0>,
  langKeys?: ReadonlyArray<LangBiasKey>,
  fixedLogits?: ReadonlyArray<number>,
  maxIters = 500,
): {
  readonly weights: Float32Array;
  readonly bias: number;
  readonly langBias: Record<LangBiasKey, number>;
} {
  const d = X[0]!.length;
  const n = indices.length;
  const w = new Float32Array(d);
  const wPrev = new Float32Array(d);
  let b = 0;
  let bPrev = 0;
  const lb: Record<LangBiasKey, number> = { en: 0, "zh-Hans": 0, "zh-Hant": 0, ja: 0, other: 0 };
  const lbPrev: Record<LangBiasKey, number> = { en: 0, "zh-Hans": 0, "zh-Hant": 0, ja: 0, other: 0 };

  // Estimate Lipschitz constant L ~ (0.25/n) * max||x||^2 + lambda
  let maxNormSq = 1;
  for (const idx of indices) {
    const row = X[idx]!;
    let s = 1;
    for (let j = 0; j < d; j++) s += row[j]! * row[j]!;
    if (s > maxNormSq) maxNormSq = s;
  }
  const lr = 1.0 / (0.25 * maxNormSq + lambda + 1.0);
  const lbReg = Math.max(0.5, lambda * 2.0); // shrink per-lang offset toward global bias

  const gradW = new Float32Array(d);
  const yW = new Float32Array(d);

  for (let iter = 0; iter < maxIters; iter++) {
    const mom = iter > 0 ? (iter - 1) / (iter + 2) : 0;
    for (let j = 0; j < d; j++) {
      yW[j] = w[j]! + mom * (w[j]! - wPrev[j]!);
    }
    const yB = b + mom * (b - bPrev);
    const yLb: Record<LangBiasKey, number> = {
      en: lb.en + mom * (lb.en - lbPrev.en),
      "zh-Hans": lb["zh-Hans"] + mom * (lb["zh-Hans"] - lbPrev["zh-Hans"]),
      "zh-Hant": lb["zh-Hant"] + mom * (lb["zh-Hant"] - lbPrev["zh-Hant"]),
      ja: lb.ja + mom * (lb.ja - lbPrev.ja),
      other: lb.other + mom * (lb.other - lbPrev.other),
    };

    gradW.fill(0);
    let gradB = 0;
    const gradLb: Record<LangBiasKey, number> = { en: 0, "zh-Hans": 0, "zh-Hant": 0, ja: 0, other: 0 };

    for (const idx of indices) {
      const row = X[idx]!;
      const lk = langKeys ? langKeys[idx]! : "en";
      let logit = yB + (langKeys ? yLb[lk] : 0) + (fixedLogits ? fixedLogits[idx]! : 0);
      for (let j = 0; j < d; j++) {
        logit += yW[j]! * row[j]!;
      }
      const err = (sigmoid(logit) - y[idx]!) / n;
      gradB += err;
      if (langKeys) gradLb[lk] += err;
      for (let j = 0; j < d; j++) {
        gradW[j] = gradW[j]! + err * row[j]!;
      }
    }

    wPrev.set(w);
    bPrev = b;
    for (const k of LANG_KEYS) lbPrev[k] = lb[k];

    let maxStep = 0;
    for (let j = 0; j < d; j++) {
      let next = yW[j]! - lr * (gradW[j]! + lambda * yW[j]!);
      if (signs) {
        const s = signs[j]!;
        if (s === 1 && next < 0) next = 0;
        if (s === -1 && next > 0) next = 0;
      }
      const diff = Math.abs(next - w[j]!);
      if (diff > maxStep) maxStep = diff;
      w[j] = next;
    }
    b = yB - lr * gradB;
    if (langKeys) {
      for (const k of LANG_KEYS) {
        lb[k] = yLb[k] - lr * (gradLb[k] + (lbReg / n) * yLb[k]);
      }
    }
    if (maxStep < 1e-6 && iter > 30) break;
  }

  // Fold global bias b into each langBias
  const finalLangBias: Record<LangBiasKey, number> = {
    en: b + lb.en,
    "zh-Hans": b + lb["zh-Hans"],
    "zh-Hant": b + lb["zh-Hant"],
    ja: b + lb.ja,
    other: b + lb.other,
  };

  return { weights: w, bias: b, langBias: finalLangBias };
}

export function fitCompactStage1Model(args: {
  readonly feats: ReadonlyArray<CompactFeatureExtraction>;
  readonly labels: ReadonlyArray<number>;
  readonly trainIndices: ReadonlyArray<number>;
  readonly lambda: number;
  readonly useSignConstraints?: boolean;
  readonly frozenEmbLogitsTrain?: ReadonlyArray<number>;
  readonly frozenEmbModel?: {
    readonly dim: number;
    readonly weights: Float32Array;
    readonly bias: number;
  };
}): TrainedLinearModelV2 {
  const { feats, labels, trainIndices, lambda } = args;
  const scaler = fitScaler(feats, labels, trainIndices);
  const Z = feats.map((f) => standardizeVector(f.vector, scaler));
  const langKeys = feats.map((f) => toLangBiasKey(f.replyLang));

  let embAlpha = 0;
  let XforFit: Float32Array[] = Z;
  let signsForFit: Array<1 | -1 | 0> | undefined =
    args.useSignConstraints !== false ? [...FEATURE_SIGNS] : undefined;

  // If a frozen embedding block is attached, append its inner-OOF logit as an extra positive feature
  if (args.frozenEmbLogitsTrain && args.frozenEmbModel) {
    XforFit = Z.map((z, idx) => {
      const ext = new Float32Array(COMPACT_V2_DIM + 1);
      ext.set(z, 0);
      ext[COMPACT_V2_DIM] = args.frozenEmbLogitsTrain![idx] ?? 0;
      return ext;
    });
    if (signsForFit) signsForFit.push(1);
  }

  const fitted = fitDenseLogistic(
    XforFit,
    labels,
    trainIndices,
    lambda,
    signsForFit,
    langKeys,
    undefined,
    600,
  );

  const rawWeights = new Float32Array(COMPACT_V2_DIM);
  for (let j = 0; j < COMPACT_V2_DIM; j++) {
    rawWeights[j] = fitted.weights[j]!;
  }
  if (args.frozenEmbLogitsTrain && args.frozenEmbModel) {
    embAlpha = fitted.weights[COMPACT_V2_DIM]!;
  }

  // Extract fixedCoefs for protoMargin (idx 4) and generic (idx 5) cleanly with ZERO double-counting
  const protoCoef = rawWeights[4]! / scaler.sigma[4]!;
  const genericCoef = rawWeights[5]! / scaler.sigma[5]!;
  const biasShift =
    -(rawWeights[4]! * scaler.mu[4]!) / scaler.sigma[4]! -
    (rawWeights[5]! * scaler.mu[5]!) / scaler.sigma[5]!;

  const theta = new Float32Array(rawWeights);
  theta[4] = 0;
  theta[5] = 0;

  const langBias: Record<LangBiasKey, number> = {
    en: fitted.langBias.en + biasShift,
    "zh-Hans": fitted.langBias["zh-Hans"] + biasShift,
    "zh-Hant": fitted.langBias["zh-Hant"] + biasShift,
    ja: fitted.langBias.ja + biasShift,
    other: fitted.langBias.other + biasShift,
  };

  return {
    scaler,
    theta,
    rawWeights,
    langBias,
    fixedCoefs: {
      protoMargin: protoCoef,
      generic: genericCoef,
    },
    ...(args.frozenEmbModel
      ? {
          embeddingBlock: {
            dim: args.frozenEmbModel.dim,
            frozen: true as const,
            weights: args.frozenEmbModel.weights,
            bias: args.frozenEmbModel.bias,
            alpha: embAlpha,
          },
        }
      : {}),
  };
}

export function scoreCompactStage1(
  feat: CompactFeatureExtraction,
  model: TrainedLinearModelV2,
  replyEmb?: Float32Array,
): {
  readonly probability: number;
  readonly logit: number;
  readonly fixedTerm: number;
  readonly familyLogits: Record<SignalFamily, number>;
  readonly z: Float32Array;
} {
  const z = standardizeVector(feat.vector, model.scaler);
  const lk = toLangBiasKey(feat.replyLang);
  const fixedTerm =
    model.fixedCoefs.protoMargin * feat.vector[4]! +
    model.fixedCoefs.generic * feat.vector[5]!;

  let embLogit = 0;
  if (model.embeddingBlock && replyEmb) {
    const eb = model.embeddingBlock;
    let s = eb.bias;
    for (let i = 0; i < eb.dim; i++) {
      s += eb.weights[i]! * (replyEmb[i] ?? 0);
    }
    embLogit = eb.alpha * s;
  }

  let logit = model.langBias[lk] + fixedTerm + embLogit;
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

  for (let j = 0; j < COMPACT_V2_DIM; j++) {
    logit += model.theta[j]! * z[j]!;
    const contrib = model.rawWeights[j]! * (z[j]! - model.scaler.zHuman[j]!);
    const fam = FEATURE_FAMILIES[j]!;
    familyLogits[fam] += contrib;
  }
  if (model.embeddingBlock) {
    // Center embedding block contribution approximately around 0
    familyLogits.embedding += Math.max(0, embLogit);
  }

  return {
    probability: sigmoid(logit),
    logit,
    fixedTerm: fixedTerm + embLogit,
    familyLogits,
    z,
  };
}

// ================= Frozen Embedding Block Helper (Strict Inner-OOF Stacking) =================
export function fitFrozenEmbeddingHead(
  embeddings: ReadonlyArray<Float32Array>,
  labels: ReadonlyArray<number>,
  trainIndices: ReadonlyArray<number>,
  dim: number,
  lambda = 0.05,
): { readonly dim: number; readonly weights: Float32Array; readonly bias: number } {
  const trunc = embeddings.map((e) => {
    const out = new Float32Array(dim);
    let sumSq = 0;
    for (let i = 0; i < dim; i++) {
      const val = e[i] ?? 0;
      out[i] = val;
      sumSq += val * val;
    }
    if (sumSq > 1e-12) {
      const inv = 1 / Math.sqrt(sumSq);
      for (let i = 0; i < dim; i++) out[i] = out[i]! * inv;
    }
    return out;
  });
  const fit = fitDenseLogistic(trunc, labels, trainIndices, lambda, undefined, undefined, undefined, 400);
  return { dim, weights: fit.weights, bias: fit.bias };
}

export function scoreEmbeddingHead(
  emb: Float32Array,
  head: { readonly dim: number; readonly weights: Float32Array; readonly bias: number },
): number {
  let sumSq = 0;
  for (let i = 0; i < head.dim; i++) {
    const v = emb[i] ?? 0;
    sumSq += v * v;
  }
  const inv = sumSq > 1e-12 ? 1 / Math.sqrt(sumSq) : 1;
  let s = head.bias;
  for (let i = 0; i < head.dim; i++) {
    s += head.weights[i]! * (emb[i] ?? 0) * inv;
  }
  return s;
}

// ================= Compact MLP (58 -> 32 ReLU -> 1) =================
export interface CompactMLP {
  readonly scaler: StandardizedScaler;
  readonly hiddenDim: number;
  readonly w1: Float32Array; // hiddenDim * COMPACT_V2_DIM
  readonly b1: Float32Array; // hiddenDim
  readonly w2: Float32Array; // hiddenDim
  readonly b2: number;
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function fitCompactMLP(
  feats: ReadonlyArray<CompactFeatureExtraction>,
  labels: ReadonlyArray<number>,
  trainIndices: ReadonlyArray<number>,
  hiddenDim = 32,
  weightDecay = 0.05,
  epochs = 180,
  seed = 42,
): CompactMLP {
  const scaler = fitScaler(feats, labels, trainIndices);
  const Z = feats.map((f) => standardizeVector(f.vector, scaler));
  const d = COMPACT_V2_DIM;
  const rng = mulberry32(seed);
  const scale1 = Math.sqrt(2 / d);
  const scale2 = Math.sqrt(2 / hiddenDim);

  const w1 = new Float32Array(hiddenDim * d);
  const b1 = new Float32Array(hiddenDim);
  const w2 = new Float32Array(hiddenDim);
  let b2 = 0;

  for (let i = 0; i < w1.length; i++) w1[i] = (rng() * 2 - 1) * scale1;
  for (let i = 0; i < w2.length; i++) w2[i] = (rng() * 2 - 1) * scale2;

  // Adam state
  const mW1 = new Float32Array(w1.length);
  const vW1 = new Float32Array(w1.length);
  const mB1 = new Float32Array(hiddenDim);
  const vB1 = new Float32Array(hiddenDim);
  const mW2 = new Float32Array(hiddenDim);
  const vW2 = new Float32Array(hiddenDim);
  let mB2 = 0;
  let vB2 = 0;

  const gW1 = new Float32Array(w1.length);
  const gB1 = new Float32Array(hiddenDim);
  const gW2 = new Float32Array(hiddenDim);
  const h = new Float32Array(hiddenDim);

  const lr = 0.01;
  const beta1 = 0.9;
  const beta2 = 0.999;
  const eps = 1e-8;
  const n = trainIndices.length;

  for (let ep = 1; ep <= epochs; ep++) {
    gW1.fill(0);
    gB1.fill(0);
    gW2.fill(0);
    let gB2 = 0;

    for (const idx of trainIndices) {
      const x = Z[idx]!;
      let logit = b2;
      for (let k = 0; k < hiddenDim; k++) {
        let sum = b1[k]!;
        const rowOff = k * d;
        for (let j = 0; j < d; j++) sum += w1[rowOff + j]! * x[j]!;
        const act = sum > 0 ? sum : 0;
        h[k] = act;
        logit += w2[k]! * act;
      }
      const err = (sigmoid(logit) - labels[idx]!) / n;
      gB2 += err;
      for (let k = 0; k < hiddenDim; k++) {
        gW2[k] = gW2[k]! + err * h[k]!;
        if (h[k]! > 0) {
          const dh = err * w2[k]!;
          gB1[k] = gB1[k]! + dh;
          const rowOff = k * d;
          for (let j = 0; j < d; j++) {
            gW1[rowOff + j] = gW1[rowOff + j]! + dh * x[j]!;
          }
        }
      }
    }

    const bc1 = 1 - beta1 ** ep;
    const bc2 = 1 - beta2 ** ep;
    for (let i = 0; i < w1.length; i++) {
      const g = gW1[i]! + weightDecay * w1[i]!;
      mW1[i] = beta1 * mW1[i]! + (1 - beta1) * g;
      vW1[i] = beta2 * vW1[i]! + (1 - beta2) * g * g;
      w1[i] = w1[i]! - (lr * (mW1[i]! / bc1)) / (Math.sqrt(vW1[i]! / bc2) + eps);
    }
    for (let k = 0; k < hiddenDim; k++) {
      mB1[k] = beta1 * mB1[k]! + (1 - beta1) * gB1[k]!;
      vB1[k] = beta2 * vB1[k]! + (1 - beta2) * gB1[k]! * gB1[k]!;
      b1[k] = b1[k]! - (lr * (mB1[k]! / bc1)) / (Math.sqrt(vB1[k]! / bc2) + eps);

      const g2 = gW2[k]! + weightDecay * w2[k]!;
      mW2[k] = beta1 * mW2[k]! + (1 - beta1) * g2;
      vW2[k] = beta2 * vW2[k]! + (1 - beta2) * g2 * g2;
      w2[k] = w2[k]! - (lr * (mW2[k]! / bc1)) / (Math.sqrt(vW2[k]! / bc2) + eps);
    }
    mB2 = beta1 * mB2 + (1 - beta1) * gB2;
    vB2 = beta2 * vB2 + (1 - beta2) * gB2 * gB2;
    b2 -= (lr * (mB2 / bc1)) / (Math.sqrt(vB2 / bc2) + eps);
  }

  return { scaler, hiddenDim, w1, b1, w2, b2 };
}

export function scoreCompactMLP(
  feat: CompactFeatureExtraction,
  mlp: CompactMLP,
): { readonly probability: number; readonly logit: number } {
  const x = standardizeVector(feat.vector, mlp.scaler);
  const d = COMPACT_V2_DIM;
  let logit = mlp.b2;
  for (let k = 0; k < mlp.hiddenDim; k++) {
    let sum = mlp.b1[k]!;
    const rowOff = k * d;
    for (let j = 0; j < d; j++) sum += mlp.w1[rowOff + j]! * x[j]!;
    if (sum > 0) logit += mlp.w2[k]! * sum;
  }
  return { probability: sigmoid(logit), logit };
}

// ================= Threshold Calibration (Training Fold Inner-OOF Only!) =================
export function calibrateThresholdsFromOOF(
  oofProbs: ReadonlyArray<number>,
  labels: ReadonlyArray<number>,
  feats: ReadonlyArray<CompactFeatureExtraction>,
  trainIndices: ReadonlyArray<number>,
): Record<PresetName, PresetThresholds> {
  // Compute g_thr (92nd percentile of e_generic on training humans) and pc (paraphrase cos)
  const humanGeneric: number[] = [];
  const humanOpCos: number[] = [];
  for (const idx of trainIndices) {
    if (labels[idx] === 0) {
      const v = feats[idx]!.vector;
      if (v[5]! > 0) humanGeneric.push(v[5]!);
      if (v[1]! > 0 && v[8]! <= 0.90 && feats[idx]!.specificityHits === 0) {
        humanOpCos.push(v[0]!);
      }
    }
  }
  humanGeneric.sort((a, b) => a - b);
  const gThr =
    humanGeneric.length > 0
      ? Number(
          Math.max(
            0.55,
            Math.min(0.85, humanGeneric[Math.floor(humanGeneric.length * 0.92)]! + 0.02),
          ).toFixed(3),
        )
      : 0.72;

  humanOpCos.sort((a, b) => a - b);
  const maxHumanLowNovCos =
    humanOpCos.length > 0 ? humanOpCos[humanOpCos.length - 1]! : 0.80;
  const pcBalanced = Number(Math.max(0.86, Math.min(0.90, maxHumanLowNovCos + 0.04)).toFixed(3));

  const findDimThreshold = (targetPrec: number, fallback: number): number => {
    const candidates = Array.from({ length: 81 }, (_, i) => Number((0.15 + i * 0.01).toFixed(2)));
    for (const tau of candidates) {
      let tp = 0;
      let fp = 0;
      for (let i = 0; i < trainIndices.length; i++) {
        const y = labels[trainIndices[i]!]!;
        const p = oofProbs[i]!;
        if (p >= tau) {
          if (y === 1) tp += 1;
          else fp += 1;
        }
      }
      // Small conservative Laplace smoothing (+0.5 FP) so finite train sets don't pick overly aggressive cliffs
      const smoothedPrec = tp > 0 ? tp / (tp + fp + 0.5) : 0;
      const rawPrec = tp + fp > 0 ? tp / (tp + fp) : 0;
      if (tp >= 8 && rawPrec >= targetPrec && smoothedPrec >= targetPrec - 0.02) {
        return tau;
      }
    }
    return fallback;
  };

  const findSuspectThreshold = (maxHumanFpr: number, dimTau: number, fallback: number): number => {
    const candidates = Array.from({ length: 75 }, (_, i) => Number((0.15 + i * 0.01).toFixed(2)));
    let totalHumans = 0;
    for (const idx of trainIndices) {
      if (labels[idx] === 0) totalHumans += 1;
    }
    for (const tau of candidates) {
      if (tau >= dimTau - 0.05) break;
      let humanFlagged = 0;
      for (let i = 0; i < trainIndices.length; i++) {
        if (labels[trainIndices[i]!] === 0 && oofProbs[i]! >= tau) {
          humanFlagged += 1;
        }
      }
      if (humanFlagged / Math.max(1, totalHumans) <= maxHumanFpr) {
        return tau;
      }
    }
    return Math.min(fallback, Number((dimTau - 0.1).toFixed(2)));
  };

  const dimCons = Math.max(0.82, findDimThreshold(0.98, 0.92));
  const dimBal = Math.min(dimCons - 0.03, Math.max(0.60, findDimThreshold(0.95, 0.80)));
  const dimStr = Math.min(dimBal - 0.05, Math.max(0.45, findDimThreshold(0.90, 0.65)));

  const susCons = findSuspectThreshold(0.03, dimCons, 0.68);
  const susBal = findSuspectThreshold(0.08, dimBal, 0.50);
  const susStr = findSuspectThreshold(0.15, dimStr, 0.35);

  return {
    conservative: {
      dim: Number(dimCons.toFixed(3)),
      suspect: Number(susCons.toFixed(3)),
      g_thr: Number(Math.min(0.88, gThr + 0.04).toFixed(3)),
      pc: Number(Math.min(0.92, pcBalanced + 0.04).toFixed(3)),
      cc: 0.95,
      ct: 10,
      kb: 3,
      sg: 8,
      nf: 2,
    },
    balanced: {
      dim: Number(dimBal.toFixed(3)),
      suspect: Number(susBal.toFixed(3)),
      g_thr: gThr,
      pc: pcBalanced,
      cc: 0.92,
      ct: 8,
      kb: 2,
      sg: 6,
      nf: 2,
    },
    strict: {
      dim: Number(dimStr.toFixed(3)),
      suspect: Number(susStr.toFixed(3)),
      g_thr: Number(Math.max(0.50, gThr - 0.04).toFixed(3)),
      pc: Number(Math.max(0.78, pcBalanced - 0.03).toFixed(3)),
      cc: 0.90,
      ct: 6,
      kb: 1,
      sg: 4,
      nf: 2,
    },
  };
}

// ================= Metrics & Reporting =================
export interface EvalMetrics {
  readonly total: number;
  readonly bots: number;
  readonly humans: number;
  readonly tpDim: number;
  readonly fpDim: number;
  readonly fnDim: number;
  readonly tnDim: number;
  readonly precisionDim: number;
  readonly recallDim: number;
  readonly f1Dim: number;
  readonly humanFprDim: number;
  readonly suspectBots: number;
  readonly suspectHumans: number;
  readonly suspectCapture: number; // bots in (dim U suspect) / bots
  readonly humanSuspectRate: number; // humans in (dim U suspect) / humans
  readonly precAt30Prev: number; // precision calibrated to 30% bot prevalence
  readonly auc: number;
}

export function computeRocAuc(probs: ReadonlyArray<number>, labels: ReadonlyArray<number>): number {
  const pairs = probs.map((p, i) => ({ p, y: labels[i]! })).sort((a, b) => b.p - a.p);
  let pos = 0;
  let neg = 0;
  for (const x of pairs) {
    if (x.y === 1) pos += 1;
    else neg += 1;
  }
  if (pos === 0 || neg === 0) return 0.5;
  let tp = 0;
  let fp = 0;
  let prevTp = 0;
  let prevFp = 0;
  let auc = 0;
  for (let i = 0; i < pairs.length; i++) {
    if (pairs[i]!.y === 1) tp += 1;
    else fp += 1;
    if (i === pairs.length - 1 || pairs[i + 1]!.p !== pairs[i]!.p) {
      auc += ((fp - prevFp) * (tp + prevTp)) / 2;
      prevTp = tp;
      prevFp = fp;
    }
  }
  return auc / (pos * neg);
}

export function computeMetricsFromVerdicts(
  verdicts: ReadonlyArray<{ readonly tier: "dim" | "suspect" | "keep"; readonly probability: number }>,
  labels: ReadonlyArray<number>,
): EvalMetrics {
  let bots = 0;
  let humans = 0;
  let tpDim = 0;
  let fpDim = 0;
  let fnDim = 0;
  let tnDim = 0;
  let suspectBots = 0;
  let suspectHumans = 0;

  for (let i = 0; i < verdicts.length; i++) {
    const v = verdicts[i]!;
    const y = labels[i]!;
    if (y === 1) {
      bots += 1;
      if (v.tier === "dim") tpDim += 1;
      else {
        fnDim += 1;
        if (v.tier === "suspect") suspectBots += 1;
      }
    } else {
      humans += 1;
      if (v.tier === "dim") fpDim += 1;
      else {
        tnDim += 1;
        if (v.tier === "suspect") suspectHumans += 1;
      }
    }
  }

  const precisionDim = tpDim + fpDim > 0 ? tpDim / (tpDim + fpDim) : 1;
  const recallDim = bots > 0 ? tpDim / bots : 0;
  const f1Dim =
    precisionDim + recallDim > 0 ? (2 * precisionDim * recallDim) / (precisionDim + recallDim) : 0;
  const humanFprDim = humans > 0 ? fpDim / humans : 0;
  const suspectCapture = bots > 0 ? (tpDim + suspectBots) / bots : 0;
  const humanSuspectRate = humans > 0 ? (fpDim + suspectHumans) / humans : 0;
  // Precision adjusted to 30% real-world bot prevalence: (0.3*TPR) / (0.3*TPR + 0.7*FPR)
  const precAt30Prev =
    0.3 * recallDim + 0.7 * humanFprDim > 0
      ? (0.3 * recallDim) / (0.3 * recallDim + 0.7 * humanFprDim)
      : 1;
  const auc = computeRocAuc(
    verdicts.map((v) => v.probability),
    labels,
  );

  return {
    total: verdicts.length,
    bots,
    humans,
    tpDim,
    fpDim,
    fnDim,
    tnDim,
    precisionDim,
    recallDim,
    f1Dim,
    humanFprDim,
    suspectBots,
    suspectHumans,
    suspectCapture,
    humanSuspectRate,
    precAt30Prev,
    auc,
  };
}

export interface PerRuleStats {
  readonly ruleId: string;
  readonly dimBots: number;
  readonly dimHumans: number;
  readonly dimPrecision: number;
  readonly suspectBots: number;
  readonly suspectHumans: number;
  readonly keepBots: number;
  readonly keepHumans: number;
}

export function computePerRuleBreakdown(
  verdicts: ReadonlyArray<VerdictV2Result>,
  labels: ReadonlyArray<number>,
): PerRuleStats[] {
  const map = new Map<
    string,
    { dimB: number; dimH: number; susB: number; susH: number; keepB: number; keepH: number }
  >();
  for (let i = 0; i < verdicts.length; i++) {
    const v = verdicts[i]!;
    const y = labels[i]!;
    const rec = map.get(v.ruleId) ?? { dimB: 0, dimH: 0, susB: 0, susH: 0, keepB: 0, keepH: 0 };
    if (v.tier === "dim") {
      if (y === 1) rec.dimB += 1;
      else rec.dimH += 1;
    } else if (v.tier === "suspect") {
      if (y === 1) rec.susB += 1;
      else rec.susH += 1;
    } else {
      if (y === 1) rec.keepB += 1;
      else rec.keepH += 1;
    }
    map.set(v.ruleId, rec);
  }
  return Array.from(map.entries())
    .map(([ruleId, r]) => ({
      ruleId,
      dimBots: r.dimB,
      dimHumans: r.dimH,
      dimPrecision: r.dimB + r.dimH > 0 ? r.dimB / (r.dimB + r.dimH) : 1,
      suspectBots: r.susB,
      suspectHumans: r.susH,
      keepBots: r.keepB,
      keepHumans: r.keepH,
    }))
    .sort((a, b) => b.dimBots + b.suspectBots - (a.dimBots + a.suspectBots));
}

export { DEFAULT_TOGGLES, FEATURE_IDS, evaluateVerdictRulesV2, type PolicyToggles };
