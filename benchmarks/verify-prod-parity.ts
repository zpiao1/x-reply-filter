import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { CALIBRATED_WEIGHTS_V2, ANCHOR_SET_V2 } from "../src/domain/calibrated-data.ts";
import { ThreadFilterEngine } from "../src/domain/engine.ts";
import { createInitialClassifierWeights, extractDomSignalsFromText } from "../src/domain/seeds.ts";
import { extractCompactFeaturesV2 as prodExtractCompactV2 } from "../src/domain/stylometrics.ts";
import { evaluateAnchorSet, getEffectiveV2Params } from "../src/domain/trained-head.ts";
import {
  type AuthorId,
  type CandidateTweet,
  DEFAULT_FILTER_SETTINGS,
  type EmbeddedTweet,
  type FilterSettings,
  type GraphqlTweetMeta,
  type SensitivityPreset,
  type ThreadId,
  type TweetId,
  type VerdictRecord,
  asTokenCount,
  truncateAndNormalize256,
} from "../src/domain/types.ts";
import { type EvalMetrics, computeMetricsFromVerdicts } from "./cv-engine.ts";
import type { BenchmarkCase } from "./dataset-humans.ts";
import {
  COMPACT_V2_DIM,
  FEATURE_IDS,
  PROTOTYPES_V2,
  type PrototypeEmbeddings,
  cosineSim,
  estimateContentTokenCount,
  extractCompactFeaturesV2 as benchExtractCompactV2,
} from "./features-v2.ts";

interface CachedEmbeddingPack {
  readonly opEmbeddings: number[][];
  readonly replyEmbeddings: number[][];
  readonly protoEmbeddings: number[][];
  readonly genericEmbeddings: number[][];
}

function toVec256(arr: ReadonlyArray<number>): ReturnType<typeof truncateAndNormalize256> {
  return truncateAndNormalize256(new Float32Array(arr), 0, arr.length);
}

function fmtPct(x: number): string {
  return `${(x * 100).toFixed(1)}%`;
}

function runProdEngine(args: {
  readonly cases: ReadonlyArray<BenchmarkCase>;
  readonly opEmbeddings: ReadonlyArray<ReadonlyArray<number>>;
  readonly replyEmbeddings: ReadonlyArray<ReadonlyArray<number>>;
  readonly preset: SensitivityPreset;
  readonly forceDomOnly?: boolean;
}): {
  readonly verdicts: VerdictRecord[];
  readonly metrics: EvalMetrics;
  readonly byLang: Record<string, EvalMetrics>;
  readonly byRule: Record<string, { dimB: number; dimH: number; susB: number; susH: number; keepB: number; keepH: number }>;
  readonly byCat: Record<string, { total: number; dim: number; sus: number; keep: number }>;
} {
  const { cases, opEmbeddings, replyEmbeddings, preset, forceDomOnly = false } = args;
  const classifier = createInitialClassifierWeights();
  const settings: FilterSettings = {
    ...DEFAULT_FILTER_SETTINGS,
    preset,
    revision: 1,
  };

  // Group cases by threadId so ThreadFilterEngine exercises thread_clone + OP anchoring faithfully
  const byThread = new Map<string, number[]>();
  for (let i = 0; i < cases.length; i++) {
    const tid = cases[i]!.threadId;
    const list = byThread.get(tid) ?? [];
    list.push(i);
    byThread.set(tid, list);
  }

  const verdictByIdx = new Array<VerdictRecord>(cases.length);

  for (const [threadId, indices] of byThread.entries()) {
    const engine = new ThreadFilterEngine(512, undefined, 16, settings);
    const first = cases[indices[0]!]!;
    const opVec = toVec256(opEmbeddings[indices[0]!]!);
    const opLikes = forceDomOnly ? null : (first.graphqlMeta?.opFavoriteCount ?? null);

    const opCandidate: CandidateTweet = {
      tweetId: threadId as TweetId,
      authorId: first.opAuthorId as AuthorId,
      rawText: first.opText,
      normalizedText: first.opText,
      isOp: true,
      domSignals: extractDomSignalsFromText(first.opText, false, false),
      opFavoriteCount: opLikes,
    };

    const batch: EmbeddedTweet[] = [
      {
        candidate: opCandidate,
        embedding: opVec,
        tokenCount: asTokenCount(estimateContentTokenCount(first.opText)),
      },
    ];

    for (let k = 0; k < indices.length; k++) {
      const idx = indices[k]!;
      const c = cases[idx]!;
      const replyTweetId = `${threadId}${String(k + 1).padStart(2, "0")}` as TweetId;
      const gMeta: GraphqlTweetMeta | null =
        forceDomOnly || c.graphqlMeta === null
          ? null
          : {
              tweetId: replyTweetId,
              screenName: c.replyAuthorId,
              displayName: c.graphqlMeta.displayName,
              authorCreatedAtMs: c.graphqlMeta.authorCreatedAtMs,
              tweetCreatedAtMs: c.graphqlMeta.tweetCreatedAtMs,
              inReplyToStatusId: c.graphqlMeta.inReplyToStatusId,
              conversationId: c.graphqlMeta.conversationId,
              description: c.graphqlMeta.description,
              isBlueVerified: c.graphqlMeta.isBlueVerified,
              verifiedType: c.graphqlMeta.verifiedType,
              following: c.graphqlMeta.following,
              followedBy: c.graphqlMeta.followedBy,
              followersCount: c.graphqlMeta.followersCount,
              friendsCount: c.graphqlMeta.friendsCount,
              statusesCount: c.graphqlMeta.statusesCount,
              defaultProfileImage: c.graphqlMeta.defaultProfileImage,
              favoriteCount: c.graphqlMeta.favoriteCount,
              replyCount: c.graphqlMeta.replyCount,
              retweetCount: c.graphqlMeta.retweetCount,
              mediaTypes: c.graphqlMeta.mediaTypes,
            };

      const candidate: CandidateTweet = {
        tweetId: replyTweetId,
        authorId: c.replyAuthorId as AuthorId,
        rawText: c.replyText,
        normalizedText: c.replyText,
        isOp: false,
        domSignals: extractDomSignalsFromText(
          c.replyText,
          c.domMeta.hasVerifiedBadge,
          c.domMeta.hasPrimaryMedia || c.domMeta.hasExternalLink,
        ),
        graphqlMeta: gMeta,
        hasPrimaryMedia: c.domMeta.hasPrimaryMedia,
        hasQuote: c.domMeta.hasQuote,
        isContextAncestor: false,
        opFavoriteCount: forceDomOnly ? null : (c.graphqlMeta?.opFavoriteCount ?? null),
      };

      batch.push({
        candidate,
        embedding: toVec256(replyEmbeddings[idx]!),
        tokenCount: asTokenCount(estimateContentTokenCount(c.replyText)),
      });
    }

    // Use deterministic timestamp matching dataset BASE_TS (2026-11-18T12:00:00Z)
    const nowMs = first.graphqlMeta?.tweetCreatedAtMs ?? first.opCreatedAtMs;
    const res = engine.ingestEmbeddedBatch(
      threadId as ThreadId,
      batch,
      classifier,
      settings,
      nowMs,
    );

    const finalById = new Map<TweetId, VerdictRecord>(res.verdicts.map((v) => [v.tweetId, v]));
    for (const retro of res.retroactiveUpdates) {
      finalById.set(retro.tweetId, retro);
    }

    for (let k = 0; k < indices.length; k++) {
      const idx = indices[k]!;
      const replyTweetId = `${threadId}${String(k + 1).padStart(2, "0")}` as TweetId;
      verdictByIdx[idx] = finalById.get(replyTweetId)!;
    }
  }

  const labels = cases.map((c) => (c.expectedLabel === "bot" ? 1 : 0));
  const scoreObjs = verdictByIdx.map((v) => ({
    tier: v.action,
    probability: v.explanation?.probability ?? v.diagnostics?.siameseProbability ?? v.score,
  }));
  const metrics = computeMetricsFromVerdicts(scoreObjs, labels);

  const byLang: Record<string, EvalMetrics> = {};
  for (const lang of ["en", "zh-hans", "zh-hant", "ja"] as const) {
    const subV: typeof scoreObjs = [];
    const subY: number[] = [];
    for (let i = 0; i < cases.length; i++) {
      if (cases[i]!.lang === lang) {
        subV.push(scoreObjs[i]!);
        subY.push(labels[i]!);
      }
    }
    byLang[lang] = computeMetricsFromVerdicts(subV, subY);
  }

  const byRule: Record<
    string,
    { dimB: number; dimH: number; susB: number; susH: number; keepB: number; keepH: number }
  > = {};
  const byCat: Record<string, { total: number; dim: number; sus: number; keep: number }> = {};

  for (let i = 0; i < cases.length; i++) {
    const v = verdictByIdx[i]!;
    const y = labels[i]!;
    const ruleId = v.explanation?.ruleId ?? v.reason.kind;
    const rr = byRule[ruleId] ?? { dimB: 0, dimH: 0, susB: 0, susH: 0, keepB: 0, keepH: 0 };
    if (v.action === "dim") {
      if (y === 1) rr.dimB += 1;
      else rr.dimH += 1;
    } else if (v.action === "suspect") {
      if (y === 1) rr.susB += 1;
      else rr.susH += 1;
    } else {
      if (y === 1) rr.keepB += 1;
      else rr.keepH += 1;
    }
    byRule[ruleId] = rr;

    const cat = cases[i]!.category;
    const cc = byCat[cat] ?? { total: 0, dim: 0, sus: 0, keep: 0 };
    cc.total += 1;
    if (v.action === "dim") cc.dim += 1;
    else if (v.action === "suspect") cc.sus += 1;
    else cc.keep += 1;
    byCat[cat] = cc;
  }

  return { verdicts: verdictByIdx, metrics, byLang, byRule, byCat };
}

function auditWeightAndFeatureParity(
  cases: ReadonlyArray<BenchmarkCase>,
  pack: CachedEmbeddingPack,
) {
  const benchWeights = JSON.parse(
    readFileSync(resolve("benchmarks/calibrated-weights.json"), "utf8"),
  ) as {
    featureIds: string[];
    mu: number[];
    sigma: number[];
    zHuman: number[];
    theta0: number[];
    rawWeights: number[];
    langBias: Record<string, number>;
    fixedCoefs: { protoMargin: number; generic: number };
    genericCentroid: number[];
    thresholds: Record<string, Record<string, number>>;
  };

  console.log("=== 1. CALIBRATED WEIGHTS PARITY (src/domain/calibrated-data.ts vs benchmarks/calibrated-weights.json) ===");
  const diffs: string[] = [];
  const checkArr = (name: string, a: ReadonlyArray<number>, b: ReadonlyArray<number>) => {
    if (a.length !== b.length) {
      diffs.push(`${name}: len ${a.length} != ${b.length}`);
      return;
    }
    let maxD = 0;
    for (let i = 0; i < a.length; i++) {
      const d = Math.abs(a[i]! - b[i]!);
      if (d > maxD) maxD = d;
    }
    if (maxD > 1e-6) diffs.push(`${name}: maxDiff=${maxD}`);
  };

  if (CALIBRATED_WEIGHTS_V2.featureIds.join(",") !== benchWeights.featureIds.join(",")) {
    diffs.push("featureIds mismatch!");
  }
  checkArr("mu", CALIBRATED_WEIGHTS_V2.mu, benchWeights.mu);
  checkArr("sigma", CALIBRATED_WEIGHTS_V2.sigma, benchWeights.sigma);
  checkArr("zHuman", CALIBRATED_WEIGHTS_V2.zHuman, benchWeights.zHuman);
  checkArr("theta0", CALIBRATED_WEIGHTS_V2.theta0, benchWeights.theta0);
  checkArr("rawWeights", CALIBRATED_WEIGHTS_V2.rawWeights, benchWeights.rawWeights);
  checkArr("genericCentroid", CALIBRATED_WEIGHTS_V2.genericCentroid, benchWeights.genericCentroid);
  for (const k of ["en", "zh-Hans", "zh-Hant", "ja", "other"] as const) {
    const d = Math.abs(CALIBRATED_WEIGHTS_V2.langBias[k] - benchWeights.langBias[k]!);
    if (d > 1e-6) diffs.push(`langBias.${k}: ${d}`);
  }
  if (Math.abs(CALIBRATED_WEIGHTS_V2.fixedCoefs.protoMargin - benchWeights.fixedCoefs.protoMargin) > 1e-6) {
    diffs.push("fixedCoefs.protoMargin mismatch");
  }
  for (const preset of ["conservative", "balanced", "strict"] as const) {
    const pt = CALIBRATED_WEIGHTS_V2.thresholds[preset];
    const bt = benchWeights.thresholds[preset]!;
    for (const k of Object.keys(bt)) {
      const pv = (pt as unknown as Record<string, number>)[k]!;
      const bv = bt[k]!;
      if (Math.abs(pv - bv) > 1e-6) {
        diffs.push(`thresholds.${preset}.${k}: prod=${pv} vs bench=${bv}`);
      }
    }
  }
  console.log("  Weight/threshold diffs:", diffs.length === 0 ? "NONE (exact 0 diff)" : diffs);

  // Anchor set check
  const initW = createInitialClassifierWeights();
  const anchorEval = evaluateAnchorSet(initW.theta!, initW.langBias!, ANCHOR_SET_V2);
  console.log("  Anchor-set eval at shipped theta0:", {
    dimThreshold: ANCHOR_SET_V2.dimThreshold,
    precisionDim: fmtPct(anchorEval.precisionDim),
    recallDim: fmtPct(anchorEval.recallDim),
    humanFprDim: fmtPct(anchorEval.humanFprDim),
    passed: anchorEval.passed,
  });

  // Feature-by-feature numerical parity across ALL 200 cases x 58 features!
  console.log("\n=== 2. COMPACT_V2 FEATURE VECTOR PARITY (src/domain/stylometrics.ts vs benchmarks/features-v2.ts) ===");
  const params = getEffectiveV2Params(initW);
  const benchSlop: Float32Array[] = [];
  const benchHuman: Float32Array[] = [];
  for (let i = 0; i < PROTOTYPES_V2.length; i++) {
    const v = toVec256(pack.protoEmbeddings[i]!);
    if (PROTOTYPES_V2[i]!.label === 1) benchSlop.push(v);
    else benchHuman.push(v);
  }
  const benchProtoEmb: PrototypeEmbeddings = {
    slop: benchSlop,
    human: benchHuman,
    genericCentroid: toVec256(CALIBRATED_WEIGHTS_V2.genericCentroid),
  };

  const maxDiffByFeature = new Float64Array(COMPACT_V2_DIM);
  const worstCaseByFeature: string[] = new Array(COMPACT_V2_DIM).fill("");

  for (let i = 0; i < cases.length; i++) {
    const c = cases[i]!;
    const replyVec = toVec256(pack.replyEmbeddings[i]!);
    const opVec = toVec256(pack.opEmbeddings[i]!);
    const benchF = benchExtractCompactV2(c, replyVec, opVec, benchProtoEmb, {
      maxCloneSim: 0,
      clusterSize: 1,
      canonicalTokenCount: 0,
    });

    const gMeta: GraphqlTweetMeta | null =
      c.graphqlMeta === null
        ? null
        : {
            tweetId: c.threadId,
            screenName: c.replyAuthorId,
            displayName: c.graphqlMeta.displayName,
            authorCreatedAtMs: c.graphqlMeta.authorCreatedAtMs,
            tweetCreatedAtMs: c.graphqlMeta.tweetCreatedAtMs,
            inReplyToStatusId: c.graphqlMeta.inReplyToStatusId,
            conversationId: c.graphqlMeta.conversationId,
            description: c.graphqlMeta.description,
            isBlueVerified: c.graphqlMeta.isBlueVerified,
            verifiedType: c.graphqlMeta.verifiedType,
            following: c.graphqlMeta.following,
            followedBy: c.graphqlMeta.followedBy,
            followersCount: c.graphqlMeta.followersCount,
            friendsCount: c.graphqlMeta.friendsCount,
            statusesCount: c.graphqlMeta.statusesCount,
            defaultProfileImage: c.graphqlMeta.defaultProfileImage,
            favoriteCount: c.graphqlMeta.favoriteCount,
            replyCount: c.graphqlMeta.replyCount,
            retweetCount: c.graphqlMeta.retweetCount,
            mediaTypes: c.graphqlMeta.mediaTypes,
          };

    const prodF = prodExtractCompactV2({
      candidate: {
        tweetId: `${c.threadId}01` as TweetId,
        authorId: c.replyAuthorId as AuthorId,
        rawText: c.replyText,
        normalizedText: c.replyText,
        isOp: false,
        domSignals: extractDomSignalsFromText(
          c.replyText,
          c.domMeta.hasVerifiedBadge,
          c.domMeta.hasPrimaryMedia || c.domMeta.hasExternalLink,
        ),
        graphqlMeta: gMeta,
        hasPrimaryMedia: c.domMeta.hasPrimaryMedia,
        hasQuote: c.domMeta.hasQuote,
        opFavoriteCount: c.graphqlMeta?.opFavoriteCount ?? null,
      },
      replyVec,
      anchor: {
        kind: "text_anchored",
        tweetId: c.threadId as TweetId,
        authorId: c.opAuthorId as AuthorId,
        tokenCount: asTokenCount(estimateContentTokenCount(c.opText)),
        embedding: opVec,
        rawText: c.opText,
        favoriteCount: c.graphqlMeta?.opFavoriteCount ?? null,
      },
      protoBank: params.protoBank,
      cloneCtx: { maxCloneSim: 0, clusterSize: 1 },
      nowMs: c.graphqlMeta?.tweetCreatedAtMs ?? c.opCreatedAtMs,
    });

    for (let j = 0; j < COMPACT_V2_DIM; j++) {
      const diff = Math.abs(benchF.vector[j]! - prodF.vector[j]!);
      if (diff > maxDiffByFeature[j]!) {
        maxDiffByFeature[j] = diff;
        worstCaseByFeature[j] = `${c.id} (bench=${benchF.vector[j]!.toFixed(4)} vs prod=${prodF.vector[j]!.toFixed(4)})`;
      }
    }
  }

  let anyFeatureDrift = false;
  for (let j = 0; j < COMPACT_V2_DIM; j++) {
    if (maxDiffByFeature[j]! > 1e-4) {
      anyFeatureDrift = true;
      console.log(`  [DRIFT] idx=${j} ${FEATURE_IDS[j]}: maxDiff=${maxDiffByFeature[j]!.toFixed(5)} @ ${worstCaseByFeature[j]}`);
    }
  }
  if (!anyFeatureDrift) {
    console.log("  EXACT PARITY across all 200 x 58 = 11,600 feature values (max abs diff <= 1e-4)!");
  }
}

function main() {
  const cases = JSON.parse(
    readFileSync(resolve("benchmarks/dataset.json"), "utf8"),
  ) as BenchmarkCase[];
  const q8Pack = JSON.parse(
    readFileSync(resolve("benchmarks/.cache-embeddings/embeddinggemma-300m-q8.json"), "utf8"),
  ) as CachedEmbeddingPack;
  const q4Pack = JSON.parse(
    readFileSync(resolve("benchmarks/.cache-embeddings/embeddinggemma-300m-q4.json"), "utf8"),
  ) as CachedEmbeddingPack;

  auditWeightAndFeatureParity(cases, q8Pack);

  console.log("\n=== 3. PRODUCTION ThreadFilterEngine METRICS (src/domain/engine.ts) ===");
  const runs = [
    { label: "Balanced (default, tau_dim=0.65)", res: runProdEngine({ cases, opEmbeddings: q8Pack.opEmbeddings, replyEmbeddings: q8Pack.replyEmbeddings, preset: "balanced" }) },
    { label: "Conservative", res: runProdEngine({ cases, opEmbeddings: q8Pack.opEmbeddings, replyEmbeddings: q8Pack.replyEmbeddings, preset: "conservative" }) },
    { label: "Strict", res: runProdEngine({ cases, opEmbeddings: q8Pack.opEmbeddings, replyEmbeddings: q8Pack.replyEmbeddings, preset: "strict" }) },
    { label: "Balanced [100% DOM-ONLY]", res: runProdEngine({ cases, opEmbeddings: q8Pack.opEmbeddings, replyEmbeddings: q8Pack.replyEmbeddings, preset: "balanced", forceDomOnly: true }) },
    { label: "Balanced [q4 embeddings on prod q8 weights]", res: runProdEngine({ cases, opEmbeddings: q4Pack.opEmbeddings, replyEmbeddings: q4Pack.replyEmbeddings, preset: "balanced" }) },
  ];

  for (const r of runs) {
    const m = r.res.metrics;
    console.log(
      `  ${r.label.padEnd(44)} P@dim=${fmtPct(m.precisionDim)} R@dim=${fmtPct(m.recallDim)} F1=${fmtPct(m.f1Dim)} FPR=${fmtPct(m.humanFprDim)} SusCap=${fmtPct(m.suspectCapture)} HumSus=${fmtPct(m.humanSuspectRate)} AUC=${m.auc.toFixed(3)}`,
    );
  }

  const bal = runs[0]!.res;
  console.log("\n=== 4. LANGUAGE BREAKDOWN (Production Balanced) ===");
  for (const [lang, m] of Object.entries(bal.byLang)) {
    console.log(
      `  ${lang.padEnd(10)} P@dim=${fmtPct(m.precisionDim)} R@dim=${fmtPct(m.recallDim)} F1=${fmtPct(m.f1Dim)} FPR=${fmtPct(m.humanFprDim)} SusCap=${fmtPct(m.suspectCapture)} HumSus=${fmtPct(m.humanSuspectRate)} AUC=${m.auc.toFixed(3)}`,
    );
  }

  console.log("\n=== 5. PER-RULE BREAKDOWN (Production Balanced) ===");
  for (const [ruleId, r] of Object.entries(bal.byRule)) {
    const p = r.dimB + r.dimH > 0 ? fmtPct(r.dimB / (r.dimB + r.dimH)) : "n/a";
    console.log(
      `  ${ruleId.padEnd(22)} dim=(${r.dimB}B/${r.dimH}H, P=${p}) sus=(${r.susB}B/${r.susH}H) keep=(${r.keepB}B/${r.keepH}H)`,
    );
  }

  console.log("\n=== 6. SAFETY SHIELD & HARD-NEGATIVE CHECKS (Production Balanced) ===");
  for (const cat of [
    "human_op_followup",
    "human_followed_account",
    "human_short_unverified",
    "human_grok_discussion",
    "human_agreement_nuance",
    "human_sarcasm_banter",
    "human_cross_lingual",
  ]) {
    const c = bal.byCat[cat]!;
    console.log(`  ${cat.padEnd(26)} total=${c.total} dim=${c.dim} (false-dim=${fmtPct(c.dim / c.total)}) sus=${c.sus} keep=${c.keep}`);
  }

  console.log("\n=== 7. #16/#17 R9 NOVELTY GUARD INSPECTION (bot_op_paraphrase) ===");
  for (let i = 0; i < cases.length; i++) {
    const c = cases[i]!;
    if (c.category !== "bot_op_paraphrase") continue;
    const v = bal.verdicts[i]!;
    console.log(
      `  ${c.id.padEnd(14)} tier=${v.action.padEnd(7)} rule=${(v.explanation?.ruleId ?? v.reason.kind).padEnd(18)} opSim=${(v.diagnostics?.opCosSim ?? 0).toFixed(3)} p=${(v.explanation?.probability ?? 0).toFixed(3)} fams=[${(v.explanation?.families ?? []).join(",")}]`,
    );
  }

  console.log("\n=== 8. ALL HUMAN SUSPECT / DIM & BOT NON-DIM IN PRODUCTION BALANCED ===");
  for (let i = 0; i < cases.length; i++) {
    const c = cases[i]!;
    const v = bal.verdicts[i]!;
    if (c.expectedLabel === "human" && v.action !== "keep") {
      console.log(`  [HUMAN ${v.action.toUpperCase()}] ${c.id} (${c.category}) rule=${v.explanation?.ruleId} p=${v.explanation?.probability.toFixed(3)} caps=[${v.explanation?.caps.join(",")}]`);
    } else if (c.expectedLabel === "bot" && v.action !== "dim") {
      console.log(`  [BOT ${v.action.toUpperCase()}] ${c.id} (${c.category}) rule=${v.explanation?.ruleId} p=${v.explanation?.probability.toFixed(3)} meta=${v.explanation?.metaSource}`);
    }
  }
}

main();
