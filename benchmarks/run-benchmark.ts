import { AutoModel, AutoTokenizer, type Tensor } from "@huggingface/transformers";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { SEED_CORPUS } from "../src/domain/seeds.ts";
import { formatClassificationPrompt } from "../src/domain/types.ts";
import { evaluateBaselineEngine } from "./baseline-eval.ts";
import {
  type EvalMetrics,
  type PerRuleStats,
  type TrainedLinearModelV2,
  calibrateThresholdsFromOOF,
  computeMetricsFromVerdicts,
  computePerRuleBreakdown,
  computeRocAuc,
  fitCompactMLP,
  fitCompactStage1Model,
  fitFrozenEmbeddingHead,
  makeStratifiedGroupKFold,
  scoreCompactMLP,
  scoreCompactStage1,
  scoreEmbeddingHead,
  standardizeVector,
} from "./cv-engine.ts";
import type { BenchmarkCase } from "./dataset-humans.ts";
import {
  COMPACT_V2_DIM,
  type CompactFeatureExtraction,
  FEATURE_IDS,
  FEATURE_SIGNS,
  GENERIC_PRAISE_CORPUS,
  PROTOTYPES_V2,
  type PresetName,
  type PresetThresholds,
  type PrototypeEmbeddings,
  type ThreadCloneInfo,
  type VerdictV2Result,
  cosineSim,
  evaluateVerdictRulesV2,
  extractCompactFeaturesV2,
  toLangBiasKey,
} from "./features-v2.ts";
import {
  type Stage2JudgeReport,
  getVramMiB,
  percentile,
  runCausalStage2Judge,
  runNliStage2Judge,
} from "./stage2-judges.ts";

interface CachedEmbeddingPack {
  readonly modelId: string;
  readonly dtype: string;
  readonly device: string;
  readonly rawDim: number;
  readonly loadMs: number;
  readonly vramDeltaMiB: number;
  readonly singleP50Ms: number;
  readonly singleP95Ms: number;
  readonly batch8PerItemMs: number;
  readonly opEmbeddings: number[][];
  readonly replyEmbeddings: number[][];
  readonly replyNoEmojiEmbeddings?: number[][];
  readonly baselineSeedEmbeddings?: number[][];
  readonly protoEmbeddings: number[][];
  readonly genericEmbeddings: number[][];
}

function l2NormalizeTrunc(raw: Float32Array, dim: number): Float32Array {
  const n = Math.min(dim, raw.length);
  const out = new Float32Array(n);
  let sumSq = 0;
  for (let i = 0; i < n; i++) {
    const v = raw[i] ?? 0;
    out[i] = v;
    sumSq += v * v;
  }
  if (sumSq > 1e-12) {
    const inv = 1 / Math.sqrt(sumSq);
    for (let i = 0; i < n; i++) out[i] = out[i]! * inv;
  }
  return out;
}

function meanAndNormalize(vecs: ReadonlyArray<Float32Array>, dim: number): Float32Array {
  const acc = new Float32Array(dim);
  for (const v of vecs) {
    for (let i = 0; i < dim; i++) acc[i] = acc[i]! + (v[i] ?? 0);
  }
  return l2NormalizeTrunc(acc, dim);
}

function extractPooledEmbeddings(
  modelOut: Record<string, Tensor>,
  attentionMask: Tensor | undefined,
  batchSize: number,
  pooling: "sentence" | "last_token",
): Float32Array[] {
  if (pooling === "sentence" && modelOut.sentence_embedding) {
    const t = modelOut.sentence_embedding;
    const dim = Number(t.dims[t.dims.length - 1]);
    const data = t.data as Float32Array;
    const out: Float32Array[] = [];
    for (let r = 0; r < batchSize; r++) {
      out.push(l2NormalizeTrunc(data.subarray(r * dim, (r + 1) * dim), dim));
    }
    return out;
  }

  const lhs = modelOut.last_hidden_state ?? modelOut.sentence_embedding;
  if (!lhs) throw new Error("No hidden state or sentence_embedding returned");
  const seqLen = Number(lhs.dims[1]);
  const dim = Number(lhs.dims[2]);
  const data = lhs.data as Float32Array;
  const maskData = attentionMask?.data as ArrayLike<number | bigint> | undefined;

  const out: Float32Array[] = [];
  for (let r = 0; r < batchSize; r++) {
    let lastIdx = seqLen - 1;
    if (maskData) {
      for (let t = seqLen - 1; t >= 0; t--) {
        if (Number(maskData[r * seqLen + t] ?? 0) > 0) {
          lastIdx = t;
          break;
        }
      }
    }
    const start = (r * seqLen + lastIdx) * dim;
    out.push(l2NormalizeTrunc(data.subarray(start, start + dim), dim));
  }
  return out;
}

async function computeOrLoadEmbeddings(args: {
  readonly cacheKey: string;
  readonly modelId: string;
  readonly dtype: "q8" | "q4" | "q4f16";
  readonly device: "cpu" | "dml";
  readonly pooling: "sentence" | "last_token";
  readonly formatPrompt: (t: string) => string;
  readonly cases: ReadonlyArray<BenchmarkCase>;
  readonly includeBaselineExtras?: boolean;
}): Promise<CachedEmbeddingPack | null> {
  const cacheDir = resolve("benchmarks/.cache-embeddings");
  mkdirSync(cacheDir, { recursive: true });
  const cachePath = resolve(cacheDir, `${args.cacheKey}.json`);
  if (existsSync(cachePath)) {
    console.log(`[cache hit] ${args.cacheKey}`);
    return JSON.parse(readFileSync(cachePath, "utf8")) as CachedEmbeddingPack;
  }

  console.log(`[embed] Loading ${args.modelId} (${args.dtype} on ${args.device})...`);
  const v0 = getVramMiB();
  const tLoad = performance.now();
  try {
    const tokenizer = await AutoTokenizer.from_pretrained(args.modelId);
    const model = await AutoModel.from_pretrained(args.modelId, {
      device: args.device,
      dtype: args.dtype,
    });
    const loadMs = performance.now() - tLoad;

    const embedBatch = async (texts: string[]): Promise<Float32Array[]> => {
      const prompts = texts.map((t) => args.formatPrompt(t.trim().length > 0 ? t : " "));
      const tok = await tokenizer(prompts, {
        padding: true,
        truncation: true,
        max_length: 256,
      });
      const out = (await model(tok)) as Record<string, Tensor>;
      return extractPooledEmbeddings(out, tok.attention_mask, prompts.length, args.pooling);
    };

    // Warmup + single latency benchmark (12 representative replies)
    await embedBatch([args.cases[0]!.replyText, args.cases[100]!.replyText]);
    let peakVram = getVramMiB();
    const singleLatencies: number[] = [];
    for (let i = 0; i < 12; i++) {
      const sample = args.cases[(i * 17) % args.cases.length]!.replyText;
      const s0 = performance.now();
      await embedBatch([sample]);
      singleLatencies.push(performance.now() - s0);
      const curV = getVramMiB();
      if (curV > peakVram) peakVram = curV;
    }

    // Batch-8 throughput benchmark
    const batch8Sample = args.cases.slice(0, 8).map((c) => c.replyText);
    const b0 = performance.now();
    await embedBatch(batch8Sample);
    const batch8Ms = performance.now() - b0;
    const curV2 = getVramMiB();
    if (curV2 > peakVram) peakVram = curV2;

    const embedAll = async (texts: string[], batchSize = 8): Promise<number[][]> => {
      const res: number[][] = [];
      for (let i = 0; i < texts.length; i += batchSize) {
        const chunk = texts.slice(i, i + batchSize);
        const embs = await embedBatch(chunk);
        for (const e of embs) res.push(Array.from(e));
      }
      return res;
    };

    // Deduplicate OPs to save compute
    const uniqueOps: string[] = [];
    const opIdxMap = new Map<string, number>();
    const caseOpMapIdx: number[] = [];
    for (const c of args.cases) {
      let uIdx = opIdxMap.get(c.opText);
      if (uIdx === undefined) {
        uIdx = uniqueOps.length;
        uniqueOps.push(c.opText);
        opIdxMap.set(c.opText, uIdx);
      }
      caseOpMapIdx.push(uIdx);
    }

    const uniqueOpEmbs = await embedAll(uniqueOps, 8);
    const opEmbeddings = caseOpMapIdx.map((idx) => uniqueOpEmbs[idx]!);
    const replyEmbeddings = await embedAll(
      args.cases.map((c) => c.replyText),
      8,
    );
    const protoEmbeddings = await embedAll(
      PROTOTYPES_V2.map((p) => p.text),
      8,
    );
    const genericEmbeddings = await embedAll([...GENERIC_PRAISE_CORPUS], 8);

    let replyNoEmojiEmbeddings: number[][] | undefined;
    let baselineSeedEmbeddings: number[][] | undefined;
    if (args.includeBaselineExtras) {
      const emojiRe = /\p{Extended_Pictographic}|\p{Regional_Indicator}/gu;
      replyNoEmojiEmbeddings = await embedAll(
        args.cases.map((c) => c.replyText.replace(emojiRe, "").trim()),
        8,
      );
      baselineSeedEmbeddings = await embedAll(
        SEED_CORPUS.map((s) => s.text),
        8,
      );
    }

    const endVram = getVramMiB();
    if (endVram > peakVram) peakVram = endVram;
    await model.dispose();

    const pack: CachedEmbeddingPack = {
      modelId: args.modelId,
      dtype: args.dtype,
      device: args.device,
      rawDim: replyEmbeddings[0]!.length,
      loadMs: Math.round(loadMs),
      vramDeltaMiB: Math.max(0, peakVram - v0),
      singleP50Ms: Number(percentile(singleLatencies, 50).toFixed(1)),
      singleP95Ms: Number(percentile(singleLatencies, 95).toFixed(1)),
      batch8PerItemMs: Number((batch8Ms / 8).toFixed(1)),
      opEmbeddings,
      replyEmbeddings,
      ...(replyNoEmojiEmbeddings ? { replyNoEmojiEmbeddings } : {}),
      ...(baselineSeedEmbeddings ? { baselineSeedEmbeddings } : {}),
      protoEmbeddings,
      genericEmbeddings,
    };

    writeFileSync(cachePath, JSON.stringify(pack), "utf8");
    console.log(
      `[embed done] ${args.cacheKey}: dim=${pack.rawDim}, load=${pack.loadMs}ms, single p50=${pack.singleP50Ms}ms p95=${pack.singleP95Ms}ms, batch8/item=${pack.batch8PerItemMs}ms, vramDelta=${pack.vramDeltaMiB}MiB`,
    );
    return pack;
  } catch (err) {
    console.error(
      `[embed FAILED] ${args.cacheKey}:`,
      err instanceof Error ? err.message : String(err),
    );
    return null;
  }
}

function buildFeaturesForDim(args: {
  readonly cases: ReadonlyArray<BenchmarkCase>;
  readonly pack: CachedEmbeddingPack;
  readonly dim: number;
  readonly overrideGraphqlNull?: boolean;
}): {
  readonly feats: CompactFeatureExtraction[];
  readonly replyVecs: Float32Array[];
  readonly opVecs: Float32Array[];
  readonly protoEmb: PrototypeEmbeddings;
} {
  const { cases, pack, dim } = args;
  const replyVecs = pack.replyEmbeddings.map((arr) => l2NormalizeTrunc(new Float32Array(arr), dim));
  const opVecs = pack.opEmbeddings.map((arr) => l2NormalizeTrunc(new Float32Array(arr), dim));

  const slopProtos: Float32Array[] = [];
  const humanProtos: Float32Array[] = [];
  for (let i = 0; i < PROTOTYPES_V2.length; i++) {
    const v = l2NormalizeTrunc(new Float32Array(pack.protoEmbeddings[i]!), dim);
    if (PROTOTYPES_V2[i]!.label === 1) slopProtos.push(v);
    else humanProtos.push(v);
  }
  const genericVecs = pack.genericEmbeddings.map((arr) =>
    l2NormalizeTrunc(new Float32Array(arr), dim),
  );
  const genericCentroid = meanAndNormalize(genericVecs, dim);
  const protoEmb: PrototypeEmbeddings = { slop: slopProtos, human: humanProtos, genericCentroid };

  // Compute unsupervised within-thread clone info per case (distinct non-OP, non-followed authors)
  const byThread = new Map<string, number[]>();
  for (let i = 0; i < cases.length; i++) {
    const tid = cases[i]!.threadId;
    const list = byThread.get(tid) ?? [];
    list.push(i);
    byThread.set(tid, list);
  }

  const cloneInfos: ThreadCloneInfo[] = cases.map((c, idx) => {
    const peers = byThread.get(c.threadId) ?? [];
    if (peers.length <= 1 || c.replyText.trim().length === 0) {
      return { maxCloneSim: 0, clusterSize: 1, canonicalTokenCount: 0 };
    }
    let maxSim = 0;
    let clusterSize = 1;
    let bestPeerLen = 0;
    for (const pIdx of peers) {
      if (pIdx === idx) continue;
      const peer = cases[pIdx]!;
      if (peer.replyAuthorId === c.replyAuthorId || peer.replyAuthorId === c.opAuthorId) continue;
      if (!args.overrideGraphqlNull && peer.graphqlMeta?.following === true) continue;
      if (peer.replyText.trim().length === 0) continue;
      const sim = cosineSim(replyVecs[idx]!, replyVecs[pIdx]!);
      if (sim > maxSim) {
        maxSim = sim;
        bestPeerLen = peer.replyText.trim().length;
      }
      if (sim >= 0.90) clusterSize += 1;
    }
    return { maxCloneSim: maxSim, clusterSize, canonicalTokenCount: bestPeerLen };
  });

  const feats = cases.map((c, idx) =>
    extractCompactFeaturesV2(
      c,
      replyVecs[idx]!,
      opVecs[idx]!,
      protoEmb,
      cloneInfos[idx]!,
      undefined,
      args.overrideGraphqlNull ?? false,
    ),
  );

  return { feats, replyVecs, opVecs, protoEmb };
}

interface CvRunOutput {
  readonly name: string;
  readonly rawModelMetrics: EvalMetrics;
  readonly rawProbs: number[];
  readonly verdictsByPreset: Record<PresetName, VerdictV2Result[]>;
  readonly metricsByPreset: Record<PresetName, EvalMetrics>;
  readonly byLangBalanced: Record<string, EvalMetrics>;
  readonly perRuleBalanced: PerRuleStats[];
  readonly avgThresholds: Record<PresetName, PresetThresholds>;
  readonly selectedLambdas: number[];
  readonly foldModels: TrainedLinearModelV2[];
  readonly foldThresholds: Array<Record<PresetName, PresetThresholds>>;
}

function selectBestLambdaInnerCV(
  cases: ReadonlyArray<BenchmarkCase>,
  feats: ReadonlyArray<CompactFeatureExtraction>,
  labels: ReadonlyArray<number>,
  trainIndices: ReadonlyArray<number>,
  useSignConstraints: boolean,
): { bestLambda: number; innerOOFProbs: number[] } {
  const subCases = trainIndices.map((i) => cases[i]!);
  const innerFolds = makeStratifiedGroupKFold(subCases, 4);
  const lambdas = [0.03, 0.1, 0.3, 1.0, 3.0];

  let bestLambda = 0.3;
  let bestLoss = Infinity;
  let bestOOF: number[] = new Array(trainIndices.length).fill(0.5);

  for (const lam of lambdas) {
    const oof = new Array<number>(trainIndices.length).fill(0.5);
    let lossSum = 0;
    for (let f = 0; f < innerFolds.length; f++) {
      const valLocal = innerFolds[f]!;
      const trLocal: number[] = [];
      for (let g = 0; g < innerFolds.length; g++) {
        if (g !== f) trLocal.push(...innerFolds[g]!);
      }
      const trGlobal = trLocal.map((li) => trainIndices[li]!);
      const model = fitCompactStage1Model({
        feats,
        labels,
        trainIndices: trGlobal,
        lambda: lam,
        useSignConstraints,
      });
      for (const li of valLocal) {
        const gi = trainIndices[li]!;
        const p = scoreCompactStage1(feats[gi]!, model).probability;
        oof[li] = p;
        const pc = Math.max(1e-6, Math.min(1 - 1e-6, p));
        lossSum += -(labels[gi]! * Math.log(pc) + (1 - labels[gi]!) * Math.log(1 - pc));
      }
    }
    const avgLoss = lossSum / trainIndices.length;
    if (avgLoss < bestLoss) {
      bestLoss = avgLoss;
      bestLambda = lam;
      bestOOF = oof;
    }
  }

  return { bestLambda, innerOOFProbs: bestOOF };
}

function runStrict5FoldCV(args: {
  readonly name: string;
  readonly cases: ReadonlyArray<BenchmarkCase>;
  readonly feats: ReadonlyArray<CompactFeatureExtraction>;
  readonly replyVecs: ReadonlyArray<Float32Array>;
  readonly folds: number[][];
  readonly frozenEmbDim?: number;
  readonly overrideGraphqlNull?: boolean;
}): CvRunOutput {
  const { name, cases, feats, replyVecs, folds, frozenEmbDim } = args;
  const labels = cases.map((c) => (c.expectedLabel === "bot" ? 1 : 0));
  const rawProbs = new Array<number>(cases.length).fill(0.5);
  const verdictsByPreset: Record<PresetName, VerdictV2Result[]> = {
    conservative: new Array<VerdictV2Result>(cases.length),
    balanced: new Array<VerdictV2Result>(cases.length),
    strict: new Array<VerdictV2Result>(cases.length),
  };
  const foldThresholds: Array<Record<PresetName, PresetThresholds>> = [];
  const foldModels: TrainedLinearModelV2[] = [];
  const selectedLambdas: number[] = [];

  for (let f = 0; f < folds.length; f++) {
    const testIdx = folds[f]!;
    const trainIdx: number[] = [];
    for (let g = 0; g < folds.length; g++) {
      if (g !== f) trainIdx.push(...folds[g]!);
    }

    // Inner OOF embedding logits if frozenEmbDim is requested (ZERO stacking leakage!)
    let frozenEmbLogitsTrain: number[] | undefined;
    let frozenEmbModel: { readonly dim: number; readonly weights: Float32Array; readonly bias: number } | undefined;
    if (frozenEmbDim !== undefined) {
      const subCases = trainIdx.map((i) => cases[i]!);
      const innerFolds = makeStratifiedGroupKFold(subCases, 4);
      frozenEmbLogitsTrain = new Array<number>(cases.length).fill(0);
      for (let inf = 0; inf < innerFolds.length; inf++) {
        const valLocal = innerFolds[inf]!;
        const trLocal: number[] = [];
        for (let g = 0; g < innerFolds.length; g++) {
          if (g !== inf) trLocal.push(...innerFolds[g]!);
        }
        const trGlobal = trLocal.map((li) => trainIdx[li]!);
        const subHead = fitFrozenEmbeddingHead(replyVecs, labels, trGlobal, frozenEmbDim, 0.15);
        for (const li of valLocal) {
          const gi = trainIdx[li]!;
          frozenEmbLogitsTrain[gi] = scoreEmbeddingHead(replyVecs[gi]!, subHead);
        }
      }
      frozenEmbModel = fitFrozenEmbeddingHead(replyVecs, labels, trainIdx, frozenEmbDim, 0.15);
    }

    // Tune lambda & get inner OOF probabilities strictly inside trainIdx
    const { bestLambda, innerOOFProbs } = selectBestLambdaInnerCV(
      cases,
      feats,
      labels,
      trainIdx,
      true,
    );
    selectedLambdas.push(bestLambda);

    // If frozenEmbDim is used, also compute inner-OOF stacked probabilities for threshold calibration
    let calibOOF = innerOOFProbs;
    if (frozenEmbDim !== undefined && frozenEmbLogitsTrain && frozenEmbModel) {
      const subCases = trainIdx.map((i) => cases[i]!);
      const innerFolds = makeStratifiedGroupKFold(subCases, 4);
      calibOOF = new Array<number>(trainIdx.length).fill(0.5);
      for (let inf = 0; inf < innerFolds.length; inf++) {
        const valLocal = innerFolds[inf]!;
        const trLocal: number[] = [];
        for (let g = 0; g < innerFolds.length; g++) {
          if (g !== inf) trLocal.push(...innerFolds[g]!);
        }
        const trGlobal = trLocal.map((li) => trainIdx[li]!);
        const subHead = fitFrozenEmbeddingHead(replyVecs, labels, trGlobal, frozenEmbDim, 0.15);
        const subModel = fitCompactStage1Model({
          feats,
          labels,
          trainIndices: trGlobal,
          lambda: bestLambda,
          useSignConstraints: true,
          frozenEmbLogitsTrain,
          frozenEmbModel: subHead,
        });
        for (const li of valLocal) {
          const gi = trainIdx[li]!;
          calibOOF[li] = scoreCompactStage1(feats[gi]!, subModel, replyVecs[gi]!).probability;
        }
      }
    }

    const thresholds = calibrateThresholdsFromOOF(calibOOF, labels, feats, trainIdx);
    foldThresholds.push(thresholds);

    const foldModel = fitCompactStage1Model({
      feats,
      labels,
      trainIndices: trainIdx,
      lambda: bestLambda,
      useSignConstraints: true,
      ...(frozenEmbLogitsTrain && frozenEmbModel
        ? { frozenEmbLogitsTrain, frozenEmbModel }
        : {}),
    });
    foldModels.push(foldModel);

    for (const ti of testIdx) {
      const scored = scoreCompactStage1(feats[ti]!, foldModel, replyVecs[ti]!);
      rawProbs[ti] = scored.probability;
      for (const preset of ["conservative", "balanced", "strict"] as const) {
        verdictsByPreset[preset][ti] = evaluateVerdictRulesV2({
          item: cases[ti]!,
          feat: feats[ti]!,
          probability: scored.probability,
          logit: scored.logit,
          familyLogits: scored.familyLogits,
          preset,
          thresholds: thresholds[preset],
          overrideGraphqlNull: args.overrideGraphqlNull ?? false,
        });
      }
    }
  }

  // Average calibrated thresholds across folds for summary reporting
  const avgPreset = (p: PresetName): PresetThresholds => {
    const k = foldThresholds.length;
    const avg = (fn: (t: PresetThresholds) => number) =>
      Number((foldThresholds.reduce((s, ft) => s + fn(ft[p]), 0) / k).toFixed(3));
    return {
      dim: avg((t) => t.dim),
      suspect: avg((t) => t.suspect),
      g_thr: avg((t) => t.g_thr),
      pc: avg((t) => t.pc),
      cc: foldThresholds[0]![p].cc,
      ct: foldThresholds[0]![p].ct,
      kb: foldThresholds[0]![p].kb,
      sg: foldThresholds[0]![p].sg,
      nf: foldThresholds[0]![p].nf,
    };
  };
  const avgThresholds: Record<PresetName, PresetThresholds> = {
    conservative: avgPreset("conservative"),
    balanced: avgPreset("balanced"),
    strict: avgPreset("strict"),
  };

  // Raw model metrics (at balanced fold thresholds without rule overrides)
  const rawVerdicts = rawProbs.map((p, idx) => {
    const fIdx = folds.findIndex((fl) => fl.includes(idx));
    const thr = foldThresholds[fIdx]!.balanced;
    return {
      tier: (p >= thr.dim ? "dim" : p >= thr.suspect ? "suspect" : "keep") as "dim" | "suspect" | "keep",
      probability: p,
    };
  });
  const rawModelMetrics = computeMetricsFromVerdicts(rawVerdicts, labels);

  const metricsByPreset: Record<PresetName, EvalMetrics> = {
    conservative: computeMetricsFromVerdicts(verdictsByPreset.conservative, labels),
    balanced: computeMetricsFromVerdicts(verdictsByPreset.balanced, labels),
    strict: computeMetricsFromVerdicts(verdictsByPreset.strict, labels),
  };

  const byLangBalanced: Record<string, EvalMetrics> = {};
  for (const lang of ["en", "zh-hans", "zh-hant", "ja"] as const) {
    const subV: VerdictV2Result[] = [];
    const subY: number[] = [];
    for (let i = 0; i < cases.length; i++) {
      if (cases[i]!.lang === lang) {
        subV.push(verdictsByPreset.balanced[i]!);
        subY.push(labels[i]!);
      }
    }
    byLangBalanced[lang] = computeMetricsFromVerdicts(subV, subY);
  }

  const perRuleBalanced = computePerRuleBreakdown(verdictsByPreset.balanced, labels);

  return {
    name,
    rawModelMetrics,
    rawProbs,
    verdictsByPreset,
    metricsByPreset,
    byLangBalanced,
    perRuleBalanced,
    avgThresholds,
    selectedLambdas,
    foldModels,
    foldThresholds,
  };
}

function runMlp5FoldCV(
  cases: ReadonlyArray<BenchmarkCase>,
  feats: ReadonlyArray<CompactFeatureExtraction>,
  folds: number[][],
): EvalMetrics {
  const labels = cases.map((c) => (c.expectedLabel === "bot" ? 1 : 0));
  const rawVerdicts = new Array<{ tier: "dim" | "suspect" | "keep"; probability: number }>(cases.length);

  for (let f = 0; f < folds.length; f++) {
    const testIdx = folds[f]!;
    const trainIdx: number[] = [];
    for (let g = 0; g < folds.length; g++) {
      if (g !== f) trainIdx.push(...folds[g]!);
    }

    // Inner OOF threshold calibration for MLP
    const subCases = trainIdx.map((i) => cases[i]!);
    const innerFolds = makeStratifiedGroupKFold(subCases, 4);
    const innerOOF = new Array<number>(trainIdx.length).fill(0.5);
    for (let inf = 0; inf < innerFolds.length; inf++) {
      const valLocal = innerFolds[inf]!;
      const trLocal: number[] = [];
      for (let g = 0; g < innerFolds.length; g++) {
        if (g !== inf) trLocal.push(...innerFolds[g]!);
      }
      const trGlobal = trLocal.map((li) => trainIdx[li]!);
      const subMlp = fitCompactMLP(feats, labels, trGlobal, 32, 0.08, 160, 42 + f * 10 + inf);
      for (const li of valLocal) {
        innerOOF[li] = scoreCompactMLP(feats[trainIdx[li]!]!, subMlp).probability;
      }
    }
    const thr = calibrateThresholdsFromOOF(innerOOF, labels, feats, trainIdx).balanced;
    const mlp = fitCompactMLP(feats, labels, trainIdx, 32, 0.08, 180, 100 + f);
    for (const ti of testIdx) {
      const p = scoreCompactMLP(feats[ti]!, mlp).probability;
      rawVerdicts[ti] = {
        tier: p >= thr.dim ? "dim" : p >= thr.suspect ? "suspect" : "keep",
        probability: p,
      };
    }
  }

  return computeMetricsFromVerdicts(rawVerdicts, labels);
}

function evaluateStage2CascadeCV(args: {
  readonly cases: ReadonlyArray<BenchmarkCase>;
  readonly folds: number[][];
  readonly stage1Verdicts: ReadonlyArray<VerdictV2Result>;
  readonly borderlineIds: Set<string>;
  readonly judge: Stage2JudgeReport;
}): {
  readonly borderlineAuc: number;
  readonly borderlinePrecAt50: number;
  readonly borderlineRecAt50: number;
  readonly borderlineF1At50: number;
  readonly borderlineFprAt50: number;
  readonly cascadeMetrics: EvalMetrics;
  readonly promotedBots: number;
  readonly promotedHumans: number;
} {
  const { cases, folds, stage1Verdicts, borderlineIds, judge } = args;
  const labels = cases.map((c) => (c.expectedLabel === "bot" ? 1 : 0));
  const scoreMap = new Map(judge.scores.map((s) => [s.id, s.probBot]));

  // Borderline standalone metrics
  const bProbs: number[] = [];
  const bLabels: number[] = [];
  let bTp = 0;
  let bFp = 0;
  let bFn = 0;
  let bTn = 0;
  for (let i = 0; i < cases.length; i++) {
    const id = cases[i]!.id;
    if (!borderlineIds.has(id)) continue;
    const p = scoreMap.get(id) ?? 0.5;
    const y = labels[i]!;
    bProbs.push(p);
    bLabels.push(y);
    if (p >= 0.5) {
      if (y === 1) bTp += 1;
      else bFp += 1;
    } else {
      if (y === 1) bFn += 1;
      else bTn += 1;
    }
  }
  const borderlineAuc = computeRocAuc(bProbs, bLabels);
  const borderlinePrecAt50 = bTp + bFp > 0 ? bTp / (bTp + bFp) : 1;
  const borderlineRecAt50 = bTp + bFn > 0 ? bTp / (bTp + bFn) : 0;
  const borderlineF1At50 =
    borderlinePrecAt50 + borderlineRecAt50 > 0
      ? (2 * borderlinePrecAt50 * borderlineRecAt50) / (borderlinePrecAt50 + borderlineRecAt50)
      : 0;
  const borderlineFprAt50 = bFp + bTn > 0 ? bFp / (bFp + bTn) : 0;

  // Leak-free 5-fold CV cascade: tune Stage-2 promotion threshold strictly on train folds
  const cascadeVerdicts = stage1Verdicts.map((v) => ({ tier: v.tier, probability: v.probability }));
  let promotedBots = 0;
  let promotedHumans = 0;

  for (let f = 0; f < folds.length; f++) {
    const testIdx = folds[f]!;
    const trainIdx: number[] = [];
    for (let g = 0; g < folds.length; g++) {
      if (g !== f) trainIdx.push(...folds[g]!);
    }

    // Choose lowest Stage-2 threshold on train borderline items keeping Stage-2 promotion precision >= 95%
    let tauS2 = 0.99;
    const cands = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98];
    for (const t of cands) {
      let tp = 0;
      let fp = 0;
      for (const idx of trainIdx) {
        if (!borderlineIds.has(cases[idx]!.id)) continue;
        if (stage1Verdicts[idx]!.tier === "dim") continue;
        const p2 = scoreMap.get(cases[idx]!.id) ?? 0;
        if (p2 >= t) {
          if (labels[idx] === 1) tp += 1;
          else fp += 1;
        }
      }
      if (tp >= 2 && fp === 0) {
        tauS2 = t;
        break;
      }
    }

    for (const ti of testIdx) {
      const id = cases[ti]!.id;
      if (!borderlineIds.has(id)) continue;
      if (stage1Verdicts[ti]!.tier === "dim") continue;
      const p2 = scoreMap.get(id) ?? 0;
      if (p2 >= tauS2 && stage1Verdicts[ti]!.caps.length === 0) {
        cascadeVerdicts[ti] = {
          tier: "dim",
          probability: Math.max(stage1Verdicts[ti]!.probability, p2),
        };
        if (labels[ti] === 1) promotedBots += 1;
        else promotedHumans += 1;
      }
    }
  }

  return {
    borderlineAuc,
    borderlinePrecAt50,
    borderlineRecAt50,
    borderlineF1At50,
    borderlineFprAt50,
    cascadeMetrics: computeMetricsFromVerdicts(cascadeVerdicts, labels),
    promotedBots,
    promotedHumans,
  };
}

function fmtPct(x: number): string {
  return `${(x * 100).toFixed(1)}%`;
}

async function main() {
  const cases = JSON.parse(
    readFileSync(resolve("benchmarks/dataset.json"), "utf8"),
  ) as BenchmarkCase[];
  const labels = cases.map((c) => (c.expectedLabel === "bot" ? 1 : 0));
  const folds = makeStratifiedGroupKFold(cases, 5);

  console.log("=== Fold Distribution (StratifiedGroupKFold by threadId) ===");
  for (let f = 0; f < folds.length; f++) {
    const fl = folds[f]!;
    const b = fl.filter((i) => labels[i] === 1).length;
    const h = fl.length - b;
    console.log(`  Fold ${f}: n=${fl.length} (bot=${b}, human=${h})`);
  }

  // Verify ZERO threadId overlap across folds
  const seenThreads = new Set<string>();
  for (const fl of folds) {
    const foldThreads = new Set(fl.map((i) => cases[i]!.threadId));
    for (const tid of foldThreads) {
      if (seenThreads.has(tid)) throw new Error(`Thread leakage detected: ${tid}`);
      seenThreads.add(tid);
    }
  }

  // ================= 1. Real ONNX Embeddings =================
  const egQ8 = await computeOrLoadEmbeddings({
    cacheKey: "embeddinggemma-300m-q8",
    modelId: "onnx-community/embeddinggemma-300m-ONNX",
    dtype: "q8",
    device: "cpu", // 0 MiB VRAM, fast on Node ORT; we separately report DML VRAM & speed from probe
    pooling: "sentence",
    formatPrompt: formatClassificationPrompt,
    cases,
    includeBaselineExtras: true,
  });
  if (!egQ8) throw new Error("Failed to embed with EmbeddingGemma-300M q8");

  const egQ4 = await computeOrLoadEmbeddings({
    cacheKey: "embeddinggemma-300m-q4",
    modelId: "onnx-community/embeddinggemma-300m-ONNX",
    dtype: "q4",
    device: "cpu",
    pooling: "sentence",
    formatPrompt: formatClassificationPrompt,
    cases,
  });

  const qwenEmb = await computeOrLoadEmbeddings({
    cacheKey: "qwen3-embedding-0.6b-q8",
    modelId: "onnx-community/Qwen3-Embedding-0.6B-ONNX",
    dtype: "q8",
    device: "cpu",
    pooling: "last_token",
    formatPrompt: (t) =>
      `Instruct: Classify whether a social media reply is AI bot slop or a genuine human reply\nQuery:${t}`,
    cases,
  });

  // ================= 2. Baseline Evaluation =================
  const baselineWithTwemojiBug = evaluateBaselineEngine({
    cases,
    seedEmbeddings: egQ8.baselineSeedEmbeddings!.map((a) => new Float32Array(a)),
    opEmbeddings: egQ8.opEmbeddings.map((a) => new Float32Array(a)),
    replyEmbeddings: egQ8.replyEmbeddings.map((a) => new Float32Array(a)),
    replyEmbeddingsNoEmoji: egQ8.replyNoEmojiEmbeddings!.map((a) => new Float32Array(a)),
    simulateTwemojiBug: true,
  });

  const baselineFixedTwemoji = evaluateBaselineEngine({
    cases,
    seedEmbeddings: egQ8.baselineSeedEmbeddings!.map((a) => new Float32Array(a)),
    opEmbeddings: egQ8.opEmbeddings.map((a) => new Float32Array(a)),
    replyEmbeddings: egQ8.replyEmbeddings.map((a) => new Float32Array(a)),
    simulateTwemojiBug: false,
  });

  console.log("\n=== BASELINE (src/domain/engine.ts + 9 seeds) ===");
  console.log("  Live DOM (Twemoji bug):", {
    P: fmtPct(baselineWithTwemojiBug.metrics.precisionDim),
    R: fmtPct(baselineWithTwemojiBug.metrics.recallDim),
    F1: fmtPct(baselineWithTwemojiBug.metrics.f1Dim),
    FPR: fmtPct(baselineWithTwemojiBug.metrics.humanFprDim),
    AUC: baselineWithTwemojiBug.metrics.auc.toFixed(3),
    reasons: baselineWithTwemojiBug.byReason,
  });
  console.log("  Twemoji intact:", {
    P: fmtPct(baselineFixedTwemoji.metrics.precisionDim),
    R: fmtPct(baselineFixedTwemoji.metrics.recallDim),
    F1: fmtPct(baselineFixedTwemoji.metrics.f1Dim),
    FPR: fmtPct(baselineFixedTwemoji.metrics.humanFprDim),
    AUC: baselineFixedTwemoji.metrics.auc.toFixed(3),
  });

  // ================= 3. Stage-1 Strict 5-Fold CV Ablations =================
  const eg256 = buildFeaturesForDim({ cases, pack: egQ8, dim: 256 });
  const eg512 = buildFeaturesForDim({ cases, pack: egQ8, dim: 512 });
  const eg768 = buildFeaturesForDim({ cases, pack: egQ8, dim: 768 });
  const eg256DomOnly = buildFeaturesForDim({
    cases,
    pack: egQ8,
    dim: 256,
    overrideGraphqlNull: true,
  });

  const cvEg256Compact = runStrict5FoldCV({
    name: "EG-300M q8 (256d) + COMPACT_V2 (58d) + Rules v2",
    cases,
    feats: eg256.feats,
    replyVecs: eg256.replyVecs,
    folds,
  });

  const cvEg256PlusFrozen = runStrict5FoldCV({
    name: "EG-300M q8 (256d) + COMPACT_V2 + Frozen 256d",
    cases,
    feats: eg256.feats,
    replyVecs: eg256.replyVecs,
    folds,
    frozenEmbDim: 256,
  });

  const cvEg512Compact = runStrict5FoldCV({
    name: "EG-300M q8 (512d) + COMPACT_V2 (58d) + Rules v2",
    cases,
    feats: eg512.feats,
    replyVecs: eg512.replyVecs,
    folds,
  });

  const cvEg512PlusFrozen = runStrict5FoldCV({
    name: "EG-300M q8 (512d) + COMPACT_V2 + Frozen 512d",
    cases,
    feats: eg512.feats,
    replyVecs: eg512.replyVecs,
    folds,
    frozenEmbDim: 512,
  });

  const cvEg768PlusFrozen = runStrict5FoldCV({
    name: "EG-300M q8 (768d) + COMPACT_V2 + Frozen 768d",
    cases,
    feats: eg768.feats,
    replyVecs: eg768.replyVecs,
    folds,
    frozenEmbDim: 768,
  });

  const cvEg256DomOnly = runStrict5FoldCV({
    name: "EG-300M q8 (256d) + COMPACT_V2 + Rules v2 [DOM-ONLY]",
    cases,
    feats: eg256DomOnly.feats,
    replyVecs: eg256DomOnly.replyVecs,
    folds,
    overrideGraphqlNull: true,
  });

  const mlpMetrics = runMlp5FoldCV(cases, eg256.feats, folds);

  let cvEgQ4: CvRunOutput | null = null;
  let q4OnQ8Metrics: Record<PresetName, EvalMetrics> | null = null;
  if (egQ4) {
    const q4Data = buildFeaturesForDim({ cases, pack: egQ4, dim: 256 });
    cvEgQ4 = runStrict5FoldCV({
      name: "EG-300M q4 (256d) + COMPACT_V2 + Rules v2",
      cases,
      feats: q4Data.feats,
      replyVecs: q4Data.replyVecs,
      folds,
    });

    // Cross-quantization check: evaluate q4 embeddings against q8-trained fold models & q8 thresholds
    const crossVerdicts: Record<PresetName, VerdictV2Result[]> = {
      conservative: new Array<VerdictV2Result>(cases.length),
      balanced: new Array<VerdictV2Result>(cases.length),
      strict: new Array<VerdictV2Result>(cases.length),
    };
    for (let f = 0; f < folds.length; f++) {
      const q8Model = cvEg256Compact.foldModels[f]!;
      const q8Thr = cvEg256Compact.foldThresholds[f]!;
      for (const ti of folds[f]!) {
        const scored = scoreCompactStage1(q4Data.feats[ti]!, q8Model, q4Data.replyVecs[ti]!);
        for (const preset of ["conservative", "balanced", "strict"] as const) {
          crossVerdicts[preset][ti] = evaluateVerdictRulesV2({
            item: cases[ti]!,
            feat: q4Data.feats[ti]!,
            probability: scored.probability,
            logit: scored.logit,
            familyLogits: scored.familyLogits,
            preset,
            thresholds: q8Thr[preset],
          });
        }
      }
    }
    q4OnQ8Metrics = {
      conservative: computeMetricsFromVerdicts(crossVerdicts.conservative, labels),
      balanced: computeMetricsFromVerdicts(crossVerdicts.balanced, labels),
      strict: computeMetricsFromVerdicts(crossVerdicts.strict, labels),
    };
  }

  // Safety-margin sensitivity check: Balanced CV at fixed tau_dim = 0.65 and 0.70
  const evalBalancedAtDimFloor = (dimFloor: number): EvalMetrics => {
    const vs = new Array<VerdictV2Result>(cases.length);
    for (let f = 0; f < folds.length; f++) {
      const model = cvEg256Compact.foldModels[f]!;
      const baseThr = cvEg256Compact.foldThresholds[f]!.balanced;
      const thr: PresetThresholds = { ...baseThr, dim: Math.max(baseThr.dim, dimFloor) };
      for (const ti of folds[f]!) {
        const scored = scoreCompactStage1(eg256.feats[ti]!, model, eg256.replyVecs[ti]!);
        vs[ti] = evaluateVerdictRulesV2({
          item: cases[ti]!,
          feat: eg256.feats[ti]!,
          probability: scored.probability,
          logit: scored.logit,
          familyLogits: scored.familyLogits,
          preset: "balanced",
          thresholds: thr,
        });
      }
    }
    return computeMetricsFromVerdicts(vs, labels);
  };
  const balancedAt65 = evalBalancedAtDimFloor(0.65);
  const balancedAt70 = evalBalancedAtDimFloor(0.70);

  let cvQwenEmb: CvRunOutput | null = null;
  if (qwenEmb) {
    const qData = buildFeaturesForDim({ cases, pack: qwenEmb, dim: 512 });
    cvQwenEmb = runStrict5FoldCV({
      name: "Qwen3-Embedding-0.6B q8 (512d) + COMPACT_V2 + Rules v2",
      cases,
      feats: qData.feats,
      replyVecs: qData.replyVecs,
      folds,
    });
  }

  console.log("\n=== STAGE-1 5-FOLD CV SUMMARY (Balanced Preset) ===");
  const stage1Table = [
    { label: "EG-300M q8 256d RAW L2-LR (no rules)", m: cvEg256Compact.rawModelMetrics },
    { label: "EG-300M q8 256d RAW MLP (no rules)", m: mlpMetrics },
    { label: "EG-300M q8 256d + COMPACT_V2 + Rules v2", m: cvEg256Compact.metricsByPreset.balanced },
    { label: "EG-300M q8 256d + Rules v2 (tau_dim>=0.65 safety)", m: balancedAt65 },
    { label: "EG-300M q8 256d + Rules v2 (tau_dim>=0.70 safety)", m: balancedAt70 },
    { label: "EG-300M q8 256d + COMPACT_V2 + Frozen 256d + Rules v2", m: cvEg256PlusFrozen.metricsByPreset.balanced },
    { label: "EG-300M q8 512d + COMPACT_V2 + Rules v2", m: cvEg512Compact.metricsByPreset.balanced },
    { label: "EG-300M q8 512d + COMPACT_V2 + Frozen 512d + Rules v2", m: cvEg512PlusFrozen.metricsByPreset.balanced },
    { label: "EG-300M q8 768d + COMPACT_V2 + Frozen 768d + Rules v2", m: cvEg768PlusFrozen.metricsByPreset.balanced },
    { label: "EG-300M q8 256d + Rules v2 [DOM-ONLY]", m: cvEg256DomOnly.metricsByPreset.balanced },
    ...(cvEgQ4 ? [{ label: "EG-300M q4 256d (own weights) + Rules v2", m: cvEgQ4.metricsByPreset.balanced }] : []),
    ...(q4OnQ8Metrics ? [{ label: "EG-300M q4 256d ON q8-WEIGHTS (cross-quant)", m: q4OnQ8Metrics.balanced }] : []),
    ...(cvQwenEmb ? [{ label: "Qwen3-Embedding-0.6B q8 512d + Rules v2", m: cvQwenEmb.metricsByPreset.balanced }] : []),
  ];
  for (const row of stage1Table) {
    console.log(
      `  ${row.label.padEnd(56)} P@dim=${fmtPct(row.m.precisionDim)} R@dim=${fmtPct(row.m.recallDim)} F1=${fmtPct(row.m.f1Dim)} FPR=${fmtPct(row.m.humanFprDim)} SusCap=${fmtPct(row.m.suspectCapture)} HumSus=${fmtPct(row.m.humanSuspectRate)} AUC=${row.m.auc.toFixed(3)}`,
    );
  }

  console.log("\n=== PRESET SUMMARY (EG-300M q8 256d) ===");
  for (const p of ["conservative", "balanced", "strict"] as const) {
    const m = cvEg256Compact.metricsByPreset[p];
    const t = cvEg256Compact.avgThresholds[p];
    console.log(
      `  ${p.padEnd(14)} tau=(dim=${t.dim}, sus=${t.suspect}, g=${t.g_thr}, pc=${t.pc}) | P@dim=${fmtPct(m.precisionDim)} R@dim=${fmtPct(m.recallDim)} F1=${fmtPct(m.f1Dim)} FPR=${fmtPct(m.humanFprDim)} SusCap=${fmtPct(m.suspectCapture)} HumSus=${fmtPct(m.humanSuspectRate)}`,
    );
  }

  if (q4OnQ8Metrics) {
    console.log("\n=== CROSS-QUANTIZATION: q4 embeddings ON q8-calibrated weights ===");
    for (const p of ["conservative", "balanced", "strict"] as const) {
      const m = q4OnQ8Metrics[p];
      console.log(
        `  ${p.padEnd(14)} P@dim=${fmtPct(m.precisionDim)} R@dim=${fmtPct(m.recallDim)} F1=${fmtPct(m.f1Dim)} FPR=${fmtPct(m.humanFprDim)} SusCap=${fmtPct(m.suspectCapture)} HumSus=${fmtPct(m.humanSuspectRate)} AUC=${m.auc.toFixed(3)}`,
      );
    }
  }

  console.log("\n=== LANGUAGE BREAKDOWN (EG-300M q8 256d Balanced) ===");
  for (const [lang, m] of Object.entries(cvEg256Compact.byLangBalanced)) {
    console.log(
      `  ${lang.padEnd(10)} P@dim=${fmtPct(m.precisionDim)} R@dim=${fmtPct(m.recallDim)} F1=${fmtPct(m.f1Dim)} FPR=${fmtPct(m.humanFprDim)} SusCap=${fmtPct(m.suspectCapture)} HumSus=${fmtPct(m.humanSuspectRate)} AUC=${m.auc.toFixed(3)}`,
    );
  }

  console.log("\n=== PER-RULE BREAKDOWN (EG-300M q8 256d Balanced) ===");
  for (const r of cvEg256Compact.perRuleBalanced) {
    console.log(
      `  ${r.ruleId.padEnd(24)} dim=(${r.dimBots}B/${r.dimHumans}H, P=${fmtPct(r.dimPrecision)}) suspect=(${r.suspectBots}B/${r.suspectHumans}H) keep=(${r.keepBots}B/${r.keepHumans}H)`,
    );
  }

  console.log("\n=== ERROR INSPECTION (EG-300M q8 256d Balanced) ===");
  for (let i = 0; i < cases.length; i++) {
    const v = cvEg256Compact.verdictsByPreset.balanced[i]!;
    const c = cases[i]!;
    if (c.expectedLabel === "human" && v.tier !== "keep") {
      console.log(`  [HUMAN ${v.tier.toUpperCase()}] ${c.id} (${c.category}) rule=${v.ruleId} p=${v.probability.toFixed(3)} caps=${v.caps.join(",")}`);
    } else if (c.expectedLabel === "bot" && v.tier !== "dim") {
      console.log(`  [BOT ${v.tier.toUpperCase()}] ${c.id} (${c.category}) rule=${v.ruleId} p=${v.probability.toFixed(3)} meta=${c.graphqlMeta ? "gql" : "dom"}`);
    }
  }

  // ================= 4. Stage-2 Borderline Judges =================
  const skipStage2 = process.argv.includes("--skip-stage2");
  const borderlineCases: BenchmarkCase[] = [];
  const borderlineIds = new Set<string>();
  for (let i = 0; i < cases.length; i++) {
    const v = cvEg256Compact.verdictsByPreset.balanced[i]!;
    const c = cases[i]!;
    if (v.shields.length > 0) continue;
    if (v.tier === "dim") continue;
    if (c.replyText.trim().length === 0) continue;
    if (v.tier === "suspect" || v.probability >= 0.20) {
      borderlineCases.push(c);
      borderlineIds.add(c.id);
    }
  }
  const bBots = borderlineCases.filter((c) => c.expectedLabel === "bot").length;
  const bHumans = borderlineCases.length - bBots;
  console.log(`\n=== STAGE-2 BORDERLINE SLICE: n=${borderlineCases.length} (${bBots} bots, ${bHumans} humans) ===`);

  const stage2Reports: Array<{
    readonly report: Stage2JudgeReport;
    readonly evalSummary?: ReturnType<typeof evaluateStage2CascadeCV>;
  }> = [];
  let preservedStage2Json: unknown = null;

  if (skipStage2) {
    const prevPath = resolve("benchmarks/benchmark-results.json");
    if (existsSync(prevPath)) {
      const prev = JSON.parse(readFileSync(prevPath, "utf8")) as { stage2?: unknown };
      preservedStage2Json = prev.stage2 ?? [];
    }
    console.log("  [--skip-stage2] Preserving previously measured Stage-2 judge results.");
  } else {
    const judgesToRun: Array<() => Promise<Stage2JudgeReport>> = [
      () =>
        runNliStage2Judge({
          judgeName: "mDeBERTa-v3-base-xnli (q8, cpu)",
          modelId: "onnx-community/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7-ONNX",
          dtype: "q8",
          device: "cpu",
          cases: borderlineCases,
        }),
      () =>
        runCausalStage2Judge({
          judgeName: "Qwen3-Reranker-0.6B (q4, cpu)",
          kind: "qwen3_reranker",
          modelId: "onnx-community/Qwen3-Reranker-0.6B-ONNX",
          dtype: "q4",
          device: "cpu",
          cases: borderlineCases,
        }),
      () =>
        runCausalStage2Judge({
          judgeName: "Qwen3-0.6B (q8, cpu)",
          kind: "qwen3_06b",
          modelId: "onnx-community/Qwen3-0.6B-ONNX",
          dtype: "q8",
          device: "cpu",
          cases: borderlineCases,
        }),
      () =>
        runCausalStage2Judge({
          judgeName: "Gemma-3-1B-IT (int8, cpu)",
          kind: "gemma3_1b",
          modelId: "onnx-community/gemma-3-1b-it-ONNX",
          dtype: "int8",
          device: "cpu",
          useExternalDataFormat: false,
          cases: borderlineCases,
        }),
    ];

    for (const runJudge of judgesToRun) {
      const rep = await runJudge();
      if (rep.supported) {
        const ev = evaluateStage2CascadeCV({
          cases,
          folds,
          stage1Verdicts: cvEg256Compact.verdictsByPreset.balanced,
          borderlineIds,
          judge: rep,
        });
        stage2Reports.push({ report: rep, evalSummary: ev });
        console.log(
          `  [S2 OK] ${rep.judgeName}: p50=${rep.p50Ms}ms p95=${rep.p95Ms}ms VRAM+${rep.vramDeltaMiB}MiB | sliceAUC=${ev.borderlineAuc.toFixed(3)} sliceP@0.5=${fmtPct(ev.borderlinePrecAt50)} sliceR@0.5=${fmtPct(ev.borderlineRecAt50)} | cascade P@dim=${fmtPct(ev.cascadeMetrics.precisionDim)} R@dim=${fmtPct(ev.cascadeMetrics.recallDim)} (+${ev.promotedBots}B/+${ev.promotedHumans}H)`,
        );
      } else {
        stage2Reports.push({ report: rep });
        console.log(`  [S2 UNSUPPORTED] ${rep.judgeName}: ${rep.error}`);
      }
    }
  }

  // ================= 5. Export Production Calibrated Weights & Anchor Set =================
  // Choose median selected lambda across the 5 folds for final production fit
  const sortedLam = [...cvEg256Compact.selectedLambdas].sort((a, b) => a - b);
  const prodLambda = sortedLam[Math.floor(sortedLam.length / 2)]!;
  const allIndices = cases.map((_, i) => i);

  // Pick whether embeddingBlock should be included based on CV performance
  const baseF1 = cvEg256Compact.metricsByPreset.balanced.f1Dim;
  const froz256F1 = cvEg256PlusFrozen.metricsByPreset.balanced.f1Dim;
  const includeFrozenBlock =
    froz256F1 > baseF1 + 0.015 &&
    cvEg256PlusFrozen.metricsByPreset.balanced.precisionDim >= 0.95;

  let prodModel: TrainedLinearModelV2;
  if (includeFrozenBlock) {
    // Compute OOF emb logits over the 5 folds to fit meta-weight without stacking bias
    const oofEmbLogits = new Array<number>(cases.length).fill(0);
    for (let f = 0; f < folds.length; f++) {
      const testIdx = folds[f]!;
      const trainIdx: number[] = [];
      for (let g = 0; g < folds.length; g++) if (g !== f) trainIdx.push(...folds[g]!);
      const subHead = fitFrozenEmbeddingHead(eg256.replyVecs, labels, trainIdx, 256, 0.15);
      for (const ti of testIdx) {
        oofEmbLogits[ti] = scoreEmbeddingHead(eg256.replyVecs[ti]!, subHead);
      }
    }
    const fullHead = fitFrozenEmbeddingHead(eg256.replyVecs, labels, allIndices, 256, 0.15);
    prodModel = fitCompactStage1Model({
      feats: eg256.feats,
      labels,
      trainIndices: allIndices,
      lambda: prodLambda,
      useSignConstraints: true,
      frozenEmbLogitsTrain: oofEmbLogits,
      frozenEmbModel: fullHead,
    });
  } else {
    prodModel = fitCompactStage1Model({
      feats: eg256.feats,
      labels,
      trainIndices: allIndices,
      lambda: prodLambda,
      useSignConstraints: true,
    });
  }

  const roundArr = (arr: Float32Array, digits = 5) =>
    Array.from(arr, (x) => Number(x.toFixed(digits)));

  const calibratedWeights = {
    schemaVersion: 2,
    generatedAt: new Date().toISOString(),
    modelId: "onnx-community/embeddinggemma-300m-ONNX",
    modelDtype: "q8",
    embeddingDim: 256,
    regularizationLambda: prodLambda,
    featureIds: [...FEATURE_IDS],
    featureSigns: [...FEATURE_SIGNS],
    mu: roundArr(prodModel.scaler.mu, 5),
    sigma: roundArr(prodModel.scaler.sigma, 5),
    zHuman: roundArr(prodModel.scaler.zHuman, 5),
    theta0: roundArr(prodModel.theta, 5),
    rawWeights: roundArr(prodModel.rawWeights, 5),
    langBias: Object.fromEntries(
      Object.entries(prodModel.langBias).map(([k, v]) => [k, Number(v.toFixed(5))]),
    ),
    fixedCoefs: {
      protoMargin: Number(prodModel.fixedCoefs.protoMargin.toFixed(5)),
      generic: Number(prodModel.fixedCoefs.generic.toFixed(5)),
    },
    ...(prodModel.embeddingBlock
      ? {
          embeddingBlock: {
            dim: prodModel.embeddingBlock.dim,
            frozen: true,
            alpha: Number(prodModel.embeddingBlock.alpha.toFixed(5)),
            bias: Number(prodModel.embeddingBlock.bias.toFixed(5)),
            weights: roundArr(prodModel.embeddingBlock.weights, 5),
          },
        }
      : {}),
    prototypes: PROTOTYPES_V2.map((p, idx) => ({
      lang: p.lang,
      label: p.label,
      text: p.text,
      embedding: roundArr(
        l2NormalizeTrunc(new Float32Array(egQ8.protoEmbeddings[idx]!), 256),
        5,
      ),
    })),
    genericCentroid: roundArr(eg256.protoEmb.genericCentroid, 5),
    thresholds: cvEg256Compact.avgThresholds,
    cvSummary: {
      folds: 5,
      grouping: "threadId",
      conservative: cvEg256Compact.metricsByPreset.conservative,
      balanced: cvEg256Compact.metricsByPreset.balanced,
      strict: cvEg256Compact.metricsByPreset.strict,
    },
  };

  writeFileSync(
    resolve("benchmarks/calibrated-weights.json"),
    JSON.stringify(calibratedWeights, null, 2) + "\n",
    "utf8",
  );

  // Anchor set (~200 items with precomputed z, fixed, y, lang for online learner rollback gate)
  const anchorSet = {
    schemaVersion: 2,
    generatedAt: new Date().toISOString(),
    targetPrecisionDim: 0.95,
    baselinePrecisionDim: Number(cvEg256Compact.metricsByPreset.balanced.precisionDim.toFixed(4)),
    baselineHumanFprDim: Number(cvEg256Compact.metricsByPreset.balanced.humanFprDim.toFixed(4)),
    dimThreshold: cvEg256Compact.avgThresholds.balanced.dim,
    items: cases.map((c, idx) => {
      const f = eg256.feats[idx]!;
      const scored = scoreCompactStage1(f, prodModel, eg256.replyVecs[idx]!);
      return {
        id: c.id,
        category: c.category,
        lang: toLangBiasKey(f.replyLang),
        y: labels[idx]!,
        z: roundArr(standardizeVector(f.vector, prodModel.scaler), 4),
        fixed: Number(scored.fixedTerm.toFixed(5)),
        oofProb: Number(cvEg256Compact.rawProbs[idx]!.toFixed(4)),
        oofRuleId: cvEg256Compact.verdictsByPreset.balanced[idx]!.ruleId,
      };
    }),
  };
  writeFileSync(
    resolve("benchmarks/anchor-set.json"),
    JSON.stringify(anchorSet, null, 2) + "\n",
    "utf8",
  );

  // Save full structured results for report generation
  const fullResults = {
    generatedAt: new Date().toISOString(),
    datasetSize: cases.length,
    embeddings: {
      egQ8: {
        rawDim: egQ8.rawDim,
        loadMs: egQ8.loadMs,
        vramDeltaMiB: egQ8.vramDeltaMiB,
        singleP50Ms: egQ8.singleP50Ms,
        singleP95Ms: egQ8.singleP95Ms,
        batch8PerItemMs: egQ8.batch8PerItemMs,
      },
      egQ4: egQ4
        ? {
            rawDim: egQ4.rawDim,
            loadMs: egQ4.loadMs,
            vramDeltaMiB: egQ4.vramDeltaMiB,
            singleP50Ms: egQ4.singleP50Ms,
            singleP95Ms: egQ4.singleP95Ms,
            batch8PerItemMs: egQ4.batch8PerItemMs,
          }
        : null,
      qwenEmb: qwenEmb
        ? {
            rawDim: qwenEmb.rawDim,
            loadMs: qwenEmb.loadMs,
            vramDeltaMiB: qwenEmb.vramDeltaMiB,
            singleP50Ms: qwenEmb.singleP50Ms,
            singleP95Ms: qwenEmb.singleP95Ms,
            batch8PerItemMs: qwenEmb.batch8PerItemMs,
          }
        : null,
    },
    baselineWithTwemojiBug: {
      metrics: baselineWithTwemojiBug.metrics,
      byLang: baselineWithTwemojiBug.byLang,
      byReason: baselineWithTwemojiBug.byReason,
    },
    baselineFixedTwemoji: {
      metrics: baselineFixedTwemoji.metrics,
      byLang: baselineFixedTwemoji.byLang,
      byReason: baselineFixedTwemoji.byReason,
    },
    stage1: {
      rawMlpMetrics: mlpMetrics,
      eg256Compact: {
        rawModelMetrics: cvEg256Compact.rawModelMetrics,
        metricsByPreset: cvEg256Compact.metricsByPreset,
        balancedAt65,
        balancedAt70,
        byLangBalanced: cvEg256Compact.byLangBalanced,
        perRuleBalanced: cvEg256Compact.perRuleBalanced,
        avgThresholds: cvEg256Compact.avgThresholds,
        selectedLambdas: cvEg256Compact.selectedLambdas,
      },
      q4OnQ8Weights: q4OnQ8Metrics,
      eg256PlusFrozen: {
        rawModelMetrics: cvEg256PlusFrozen.rawModelMetrics,
        metricsByPreset: cvEg256PlusFrozen.metricsByPreset,
        byLangBalanced: cvEg256PlusFrozen.byLangBalanced,
        perRuleBalanced: cvEg256PlusFrozen.perRuleBalanced,
      },
      eg512Compact: {
        rawModelMetrics: cvEg512Compact.rawModelMetrics,
        metricsByPreset: cvEg512Compact.metricsByPreset,
      },
      eg512PlusFrozen: {
        rawModelMetrics: cvEg512PlusFrozen.rawModelMetrics,
        metricsByPreset: cvEg512PlusFrozen.metricsByPreset,
      },
      eg768PlusFrozen: {
        rawModelMetrics: cvEg768PlusFrozen.rawModelMetrics,
        metricsByPreset: cvEg768PlusFrozen.metricsByPreset,
      },
      eg256DomOnly: {
        rawModelMetrics: cvEg256DomOnly.rawModelMetrics,
        metricsByPreset: cvEg256DomOnly.metricsByPreset,
        perRuleBalanced: cvEg256DomOnly.perRuleBalanced,
      },
      egQ4: cvEgQ4
        ? {
            rawModelMetrics: cvEgQ4.rawModelMetrics,
            metricsByPreset: cvEgQ4.metricsByPreset,
          }
        : null,
      qwenEmb: cvQwenEmb
        ? {
            rawModelMetrics: cvQwenEmb.rawModelMetrics,
            metricsByPreset: cvQwenEmb.metricsByPreset,
            byLangBalanced: cvQwenEmb.byLangBalanced,
          }
        : null,
    },
    borderlineSlice: {
      total: borderlineCases.length,
      bots: bBots,
      humans: bHumans,
      ids: borderlineCases.map((c) => c.id),
    },
    stage2: skipStage2
      ? preservedStage2Json
      : stage2Reports.map((s) => ({
          judgeName: s.report.judgeName,
          modelId: s.report.modelId,
          dtype: s.report.dtype,
          device: s.report.device,
          supported: s.report.supported,
          error: s.report.error,
          loadMs: s.report.loadMs,
          vramDeltaMiB: s.report.vramDeltaMiB,
          p50Ms: s.report.p50Ms,
          p95Ms: s.report.p95Ms,
          meanMs: s.report.meanMs,
          evalSummary: s.evalSummary ?? null,
        })),
    topFeatureWeights: FEATURE_IDS.map((id, i) => ({
      id,
      rawWeight: Number(prodModel.rawWeights[i]!.toFixed(4)),
      theta0: Number(prodModel.theta[i]!.toFixed(4)),
    })).sort((a, b) => Math.abs(b.rawWeight) - Math.abs(a.rawWeight)),
  };

  writeFileSync(
    resolve("benchmarks/benchmark-results.json"),
    JSON.stringify(fullResults, null, 2) + "\n",
    "utf8",
  );
  console.log("\nExported calibrated-weights.json, anchor-set.json, benchmark-results.json!");
}

await main();
