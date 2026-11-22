import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { ANCHOR_SET_V2, CALIBRATED_WEIGHTS_V2 } from "../src/domain/calibrated-data.ts";
import {
  ThreadFilterEngine,
  buildSiameseFeatures,
  dotProduct256,
  evaluateVerdictRulesV2,
  scoreSiameseHead,
  updateOnlineWeightsPassiveAggressive,
} from "../src/domain/engine.ts";
import {
  createInitialClassifierWeights,
  extractDomSignalsFromText,
  recordAuthorOutcome,
} from "../src/domain/seeds.ts";
import {
  FEATURE_IDS,
  FEATURE_SIGNS,
  computeTopicHash8,
  detectLangV2,
  extractCompactFeaturesV2,
  matchKeywordSpamV2,
} from "../src/domain/stylometrics.ts";
import {
  getEffectiveV2Params,
  insertIntoReplayBuffer,
  scoreCompactV2,
  trainFromReplayBuffer,
} from "../src/domain/trained-head.ts";
import {
  type AuthorId,
  COMPACT_V2_DIM,
  type CandidateTweet,
  DEFAULT_FILTER_SETTINGS,
  EMBEDDING_DIM,
  type FeedbackReplayEntry,
  type FilterSettings,
  type GraphqlTweetMeta,
  RAW_EMBEDDING_DIM,
  type ThreadId,
  type TweetId,
  type UnitVec256,
  asProbability,
  asTokenCount,
  getDecayedAutoFlags,
  parseAuthorId,
  parseThreadId,
  parseTweetId,
  truncateAndNormalize256,
  truncateAndNormalizeBatch256,
} from "../src/domain/types.ts";
import {
  parseGraphqlTweetMeta,
  parseRpcRequestV2,
  parseStage2VerdictUpdate,
} from "../src/shared/rpc.ts";

function makeUnitVec(seedIndex: number, noise = 0): UnitVec256 {
  const raw = new Float32Array(RAW_EMBEDDING_DIM);
  raw[seedIndex % EMBEDDING_DIM] = 1;
  if (noise !== 0) {
    raw[(seedIndex + 1) % EMBEDDING_DIM] = noise;
  }
  return truncateAndNormalize256(raw, 0, RAW_EMBEDDING_DIM);
}

function makeGqlMeta(overrides: Partial<GraphqlTweetMeta> = {}): GraphqlTweetMeta {
  return {
    displayName: "Regular User",
    authorCreatedAtMs: Date.parse("2022-01-01T00:00:00Z"),
    tweetCreatedAtMs: Date.parse("2026-11-20T12:00:00Z"),
    inReplyToStatusId: "1000",
    conversationId: "1000",
    description: "Software engineer and coffee drinker.",
    isBlueVerified: false,
    verifiedType: "none",
    following: false,
    followedBy: false,
    followersCount: 600,
    friendsCount: 400,
    statusesCount: 2500,
    defaultProfileImage: false,
    favoriteCount: 2,
    replyCount: 0,
    retweetCount: 0,
    mediaTypes: [],
    ...overrides,
  };
}

function makeCandidate(
  id: string,
  author: string,
  text: string,
  opts: {
    isOp?: boolean;
    verified?: boolean;
    hasMedia?: boolean;
    hasQuote?: boolean;
    isContextAncestor?: boolean;
    opFavoriteCount?: number | null;
    graphqlMeta?: GraphqlTweetMeta | null;
  } = {},
): CandidateTweet {
  const tweetId = parseTweetId(id);
  const authorId = parseAuthorId(author);
  assert(tweetId !== null && authorId !== null, "valid test IDs");
  return {
    tweetId,
    authorId,
    rawText: text,
    normalizedText: text.trim(),
    isOp: opts.isOp ?? false,
    domSignals: extractDomSignalsFromText(text, opts.verified ?? false, opts.hasMedia ?? false),
    ...(opts.graphqlMeta !== undefined ? { graphqlMeta: opts.graphqlMeta } : {}),
    ...(opts.hasMedia !== undefined ? { hasPrimaryMedia: opts.hasMedia } : {}),
    ...(opts.hasQuote !== undefined ? { hasQuote: opts.hasQuote } : {}),
    ...(opts.isContextAncestor !== undefined ? { isContextAncestor: opts.isContextAncestor } : {}),
    ...(opts.opFavoriteCount !== undefined ? { opFavoriteCount: opts.opFavoriteCount } : {}),
  };
}

function getCalibratedSlopVec(): UnitVec256 {
  const firstSlop = CALIBRATED_WEIGHTS_V2.prototypes.find((p) => p.label === 1)!;
  return truncateAndNormalize256(new Float32Array(firstSlop.embedding), 0, EMBEDDING_DIM);
}

function getCalibratedHumanVec(offset = 0): UnitVec256 {
  const humans = CALIBRATED_WEIGHTS_V2.prototypes.filter((p) => p.label === -1);
  const picked = humans[offset % humans.length]!;
  return truncateAndNormalize256(new Float32Array(picked.embedding), 0, EMBEDDING_DIM);
}

describe("domain vector truncation and boundary parsing", () => {
  it("slices 768d output to 256d and normalizes to unit length", () => {
    const raw = new Float32Array(RAW_EMBEDDING_DIM * 2);
    raw[0] = 3;
    raw[1] = 4;
    raw[300] = 999;
    raw[RAW_EMBEDDING_DIM + 2] = 6;
    raw[RAW_EMBEDDING_DIM + 3] = 8;

    const batch = truncateAndNormalizeBatch256(raw, 2, RAW_EMBEDDING_DIM);
    assert.equal(batch.length, 2);
    const first = batch[0]!;
    const second = batch[1]!;
    assert.equal(first.length, 256);
    assert(Math.abs(first[0]! - 0.6) < 1e-6);
    assert(Math.abs(first[1]! - 0.8) < 1e-6);
    assert(Math.abs(dotProduct256(first, first) - 1) < 1e-6);
    assert(Math.abs(dotProduct256(second, second) - 1) < 1e-6);
  });

  it("parses thread, tweet, and author identifiers at the boundary", () => {
    assert.equal(parseThreadId("/elonmusk/status/189900112233"), "189900112233");
    assert.equal(parseThreadId("/home"), null);
    assert.equal(parseTweetId("189900112233"), "189900112233");
    assert.equal(parseTweetId("not-a-id"), null);
    assert.equal(parseAuthorId("@Alice_99"), "alice_99");
    assert.equal(parseAuthorId("invalid handle!"), null);
  });
});

describe("COMPACT_V2 stylometrics, detectLangV2, and exact additive scoring", () => {
  it("matches calibrated 58-dim schema and detects EN/ZH-Hans/ZH-Hant/JA/cjk-ambiguous/other", () => {
    assert.equal(FEATURE_IDS.length, COMPACT_V2_DIM);
    assert.equal(FEATURE_SIGNS.length, COMPACT_V2_DIM);
    assert.equal(CALIBRATED_WEIGHTS_V2.theta0.length, COMPACT_V2_DIM);
    assert.equal(CALIBRATED_WEIGHTS_V2.thresholds.balanced.dim, 0.65);

    assert.equal(detectLangV2("Local WebGPU inference runs cleanly."), "en");
    assert.equal(detectLangV2("这个地方债置换方案确实能缓解城投短债压力。"), "zh-Hans");
    assert.equal(detectLangV2("這不只是技術導入的問題，更是組織思維的重塑。"), "zh-Hant");
    assert.equal(detectLangV2("手元の環境で試してみましたが再現しませんでした。"), "ja");
    assert.equal(detectLangV2("草"), "cjk-ambiguous");
    assert.equal(detectLangV2("了解"), "cjk-ambiguous");
    assert.equal(detectLangV2("안녕하세요 반갑습니다"), "other");
  });

  it("extracts LLM cadence, excludes Chinese —— from em-dash, and preserves exact additive logit", () => {
    const weights = createInitialClassifierWeights();
    const params = getEffectiveV2Params(weights);
    const slopVec = getCalibratedSlopVec();
    const opVec = makeUnitVec(12);

    const enSlop = makeCandidate(
      "2001",
      "growth_guru_99",
      "Spot on! This isn't just a tool — it's a paradigm shift for the entire landscape. What do you think? 🚀🔥",
      {
        verified: true,
        graphqlMeta: makeGqlMeta({
          isBlueVerified: true,
          verifiedType: "blue",
          following: false,
          displayName: "Alex | AI 🚀",
        }),
      },
    );

    const featEn = extractCompactFeaturesV2({
      candidate: enSlop,
      replyVec: slopVec,
      anchor: {
        kind: "text_anchored",
        tweetId: "2000" as TweetId,
        authorId: "op_user" as AuthorId,
        tokenCount: asTokenCount(14),
        embedding: opVec,
        rawText: "We shipped local browser classification using WebGPU today.",
      },
      protoBank: params.protoBank,
      cloneCtx: { maxCloneSim: 0, clusterSize: 1 },
    });

    assert.equal(featEn.vector.length, 58);
    assert(featEn.vector[13]! > 0, "s_contrast fires on not just X — it's Y");
    assert.equal(featEn.vector[14], 1, "s_syco_open fires on Spot on!");
    assert.equal(featEn.vector[15], 1, "s_bait_close fires on What do you think?");
    assert(featEn.vector[16]! > 0, "s_slop_lex fires on paradigm/landscape");
    assert(featEn.vector[17]! > 0, "s_em_dash fires on Latin em-dash");
    assert.equal(featEn.vector[23], 1, "s_emoji_tail fires on trailing 🚀🔥");
    assert.equal(featEn.vector[37], 1, "m_blue = 1 for confirmed blue");
    assert.equal(featEn.vector[39], 1, "m_unfollowed = 1 for following===false");
    assert.equal(featEn.vector[48], 1, "m_name_spam fires on | AI 🚀");

    const scored = scoreCompactV2(featEn, weights);
    assert(scored.probability > 0.8, "calibrated Stage-1 scores obvious AI slop high");
    assert(scored.top.length >= 2, "produces explainable top signal contributions");
    assert(scored.corroboratingFamilies.length >= 2, "corroborates across multiple signal families");

    // Verify exact logit additivity: baseHumanLogit + sum(all contributions) === logit
    let totalContrib = 0;
    const zHuman = CALIBRATED_WEIGHTS_V2.zHuman;
    const mu = CALIBRATED_WEIGHTS_V2.mu;
    const sigma = CALIBRATED_WEIGHTS_V2.sigma;
    for (let j = 0; j < COMPACT_V2_DIM; j++) {
      if (j === 4) {
        totalContrib += params.fixedCoefs.protoMargin * (featEn.vector[4]! - (mu[4]! + zHuman[4]! * sigma[4]!));
      } else if (j === 5) {
        totalContrib += params.fixedCoefs.generic * (featEn.vector[5]! - (mu[5]! + zHuman[5]! * sigma[5]!));
      } else {
        totalContrib += params.theta[j]! * (scored.z[j]! - zHuman[j]!);
      }
    }
    assert(
      Math.abs(scored.baseHumanLogit + totalContrib - scored.logit) < 1e-4,
      "exact additivity holds between baseHumanLogit + contributions and final logit",
    );

    // Chinese —— must NOT trigger s_em_dash, and ZH-Hans <-> ZH-Hant is never x_lang_mismatch
    const zhReply = makeCandidate(
      "2002",
      "zh_user",
      "这不仅仅是工具效率——更是工作流的重塑。",
      { graphqlMeta: makeGqlMeta() },
    );
    const featZh = extractCompactFeaturesV2({
      candidate: zhReply,
      replyVec: slopVec,
      anchor: {
        kind: "text_anchored",
        tweetId: "2000" as TweetId,
        authorId: "op_user" as AuthorId,
        tokenCount: asTokenCount(14),
        embedding: opVec,
        rawText: "這是一個關於繁體中文與簡體中文混合討論的帖子。",
      },
      protoBank: params.protoBank,
      cloneCtx: { maxCloneSim: 0, clusterSize: 1 },
    });
    assert.equal(featZh.vector[17], 0, "Chinese —— excluded from Latin s_em_dash");
    assert.equal(featZh.vector[32], 0, "zh-Hans reply to zh-Hant OP is not x_lang_mismatch");
  });

  it("tightens t.me keyword spam so bare links require a bait verb", () => {
    assert.equal(matchKeywordSpamV2("Release changelog mirrored at t.me/oss_updates"), null);
    assert.equal(matchKeywordSpamV2("Join t.me/sol_alpha_99 for 100x calls"), "en_crypto_scam");
  });
});

describe("Rule Table v2: shields, summons, blue-check short/media, clones, paraphrases, caps", () => {
  it("protects OP follow-ups, followed accounts, and context ancestors via R1 shields", () => {
    const engine = new ThreadFilterEngine();
    const weights = createInitialClassifierWeights();
    const threadId = "1000" as ThreadId;
    const slopVec = getCalibratedSlopVec();

    const op = makeCandidate("1000", "op_author", "Published our v2 benchmark results today.", {
      isOp: true,
    });
    const opFollowup = makeCandidate("1001", "op_author", "Link in bio for full PDF! 🔥🔥", {
      verified: true,
      graphqlMeta: makeGqlMeta({ isBlueVerified: true, verifiedType: "blue", following: false }),
    });
    const followedFriend = makeCandidate("1002", "friend_alex", "Fire 🔥🔥🔥", {
      verified: true,
      graphqlMeta: makeGqlMeta({ isBlueVerified: true, verifiedType: "blue", following: true }),
    });
    const followedGif = makeCandidate("1003", "friend_gif", "", {
      verified: true,
      hasMedia: true,
      graphqlMeta: makeGqlMeta({
        isBlueVerified: true,
        verifiedType: "blue",
        following: true,
        mediaTypes: ["animated_gif"],
      }),
    });
    const ancestorTweet = makeCandidate(
      "1004",
      "parent_user",
      "Spot on! Game-changer! What do you think?",
      { isContextAncestor: true },
    );

    const res = engine.ingestEmbeddedBatch(
      threadId,
      [
        { candidate: op, embedding: makeUnitVec(1), tokenCount: asTokenCount(10) },
        { candidate: opFollowup, embedding: slopVec, tokenCount: asTokenCount(6) },
        { candidate: followedFriend, embedding: slopVec, tokenCount: asTokenCount(2) },
        { candidate: followedGif, embedding: slopVec, tokenCount: asTokenCount(0) },
        { candidate: ancestorTweet, embedding: slopVec, tokenCount: asTokenCount(8) },
      ],
      weights,
    );

    for (const v of res.verdicts) {
      assert.equal(v.action, "keep", `shielded tweet ${v.tweetId} must stay keep`);
      assert((v.explanation?.shields.length ?? 0) >= 1, `shield recorded on ${v.tweetId}`);
    }
    assert.equal(res.verdicts[0]?.explanation?.ruleId, "shield_op_author");
    assert.equal(res.verdicts[1]?.explanation?.ruleId, "shield_followed");
    assert.equal(res.verdicts[2]?.explanation?.ruleId, "shield_followed");
    assert.equal(res.verdicts[3]?.explanation?.ruleId, "shield_context_ancestor");
  });

  it("spares short casual human reactions while catching unfollowed blue-check Facts/W/emoji & DOM-only -> suspect", () => {
    const engine = new ThreadFilterEngine();
    const weights = createInitialClassifierWeights();
    const threadId = "1100" as ThreadId;
    const opVec = makeUnitVec(1);
    const humanVec = getCalibratedHumanVec();

    const op = makeCandidate("1100", "op_user", "New kernel latency benchmarks landed today.", {
      isOp: true,
    });
    const shortJp = makeCandidate("1101", "jp_human", "草www", { graphqlMeta: makeGqlMeta() });
    const shortZh = makeCandidate("1102", "zh_human", "哈哈哈哈笑死", {
      graphqlMeta: makeGqlMeta(),
    });
    const shortEn = makeCandidate("1103", "en_human", "wtf lol", { graphqlMeta: makeGqlMeta() });
    const shortSpamZh = makeCandidate("1104", "zh_spammer", "看主页置顶自取", {
      verified: true,
      graphqlMeta: makeGqlMeta({ isBlueVerified: true, verifiedType: "blue", following: false }),
    });
    const blueEmojiConfirmed = makeCandidate("1105", "rankida600", "🙏🙏🙏🙏", {
      verified: true,
      graphqlMeta: makeGqlMeta({ isBlueVerified: true, verifiedType: "blue", following: false }),
    });
    const blueEmojiDomOnly = makeCandidate("1106", "dom_only_blue", "🙏🙏🙏🙏", {
      verified: true,
      graphqlMeta: null,
    });
    // #13 micro-fix #1: bare EN "Facts" and "W" from unfollowed blue checks must NOT be exempted!
    const blueFacts = makeCandidate("1107", "reply_guy_1", "Facts 💯🔥", {
      verified: true,
      graphqlMeta: makeGqlMeta({ isBlueVerified: true, verifiedType: "blue", following: false }),
    });
    const blueWTake = makeCandidate("1108", "reply_guy_2", "W take 🔥🔥", {
      verified: true,
      graphqlMeta: makeGqlMeta({ isBlueVerified: true, verifiedType: "blue", following: false }),
    });

    const res = engine.ingestEmbeddedBatch(
      threadId,
      [
        { candidate: op, embedding: opVec, tokenCount: asTokenCount(12) },
        { candidate: shortJp, embedding: opVec, tokenCount: asTokenCount(2) },
        { candidate: shortZh, embedding: humanVec, tokenCount: asTokenCount(3) },
        { candidate: shortEn, embedding: humanVec, tokenCount: asTokenCount(2) },
        { candidate: shortSpamZh, embedding: makeUnitVec(4), tokenCount: asTokenCount(4) },
        { candidate: blueEmojiConfirmed, embedding: makeUnitVec(5), tokenCount: asTokenCount(2) },
        { candidate: blueEmojiDomOnly, embedding: makeUnitVec(6), tokenCount: asTokenCount(2) },
        { candidate: blueFacts, embedding: makeUnitVec(7), tokenCount: asTokenCount(2) },
        { candidate: blueWTake, embedding: makeUnitVec(8), tokenCount: asTokenCount(2) },
      ],
      weights,
    );

    assert.equal(res.verdicts[0]?.action, "keep");
    assert.equal(res.verdicts[0]?.reason.kind, "short_human_exempt");
    assert.equal(res.verdicts[1]?.action, "keep");
    assert.equal(res.verdicts[2]?.action, "keep");
    assert.equal(res.verdicts[3]?.action, "dim");
    assert.equal(res.verdicts[3]?.reason.kind, "keyword_spam");
    assert.equal(res.verdicts[4]?.action, "dim");
    assert.equal(res.verdicts[4]?.reason.kind, "blue_short_praise");
    assert.equal(
      res.verdicts[5]?.action,
      "suspect",
      "DOM-only verified emoji reply (following===null) safely routes to suspect",
    );
    assert.equal(res.verdicts[5]?.reason.kind, "blue_short_praise");
    assert.equal(res.verdicts[6]?.action, "dim", "Facts 💯🔥 from unfollowed blue check is dimmed");
    assert.equal(res.verdicts[7]?.action, "dim", "W take 🔥🔥 from unfollowed blue check is dimmed");
  });

  it("handles media_only_blue (dim confirmed, suspect DOM-only, keep unverified human GIF)", () => {
    const engine = new ThreadFilterEngine();
    const weights = createInitialClassifierWeights();
    const threadId = "1200" as ThreadId;
    const op = makeCandidate("1200", "op_user", "Friday deployment alert triggered at 5:30pm.", {
      isOp: true,
    });
    const blueGifConfirmed = makeCandidate("1201", "gif_bot_1", "", {
      verified: true,
      hasMedia: true,
      graphqlMeta: makeGqlMeta({
        isBlueVerified: true,
        verifiedType: "blue",
        following: false,
        mediaTypes: ["animated_gif"],
      }),
    });
    const blueGifDomOnly = makeCandidate("1202", "gif_bot_2", "", {
      verified: true,
      hasMedia: true,
      graphqlMeta: null,
    });
    const unverifiedHumanGif = makeCandidate("1203", "human_gif", "", {
      verified: false,
      hasMedia: true,
      graphqlMeta: makeGqlMeta({
        isBlueVerified: false,
        verifiedType: "none",
        following: false,
        mediaTypes: ["animated_gif"],
      }),
    });

    const res = engine.ingestEmbeddedBatch(
      threadId,
      [
        { candidate: op, embedding: makeUnitVec(1), tokenCount: asTokenCount(10) },
        { candidate: blueGifConfirmed, embedding: makeUnitVec(2), tokenCount: asTokenCount(0) },
        { candidate: blueGifDomOnly, embedding: makeUnitVec(3), tokenCount: asTokenCount(0) },
        { candidate: unverifiedHumanGif, embedding: makeUnitVec(4), tokenCount: asTokenCount(0) },
      ],
      weights,
    );

    assert.equal(res.verdicts[0]?.action, "dim");
    assert.equal(res.verdicts[0]?.reason.kind, "media_only_blue");
    assert.equal(res.verdicts[1]?.action, "suspect");
    assert.equal(res.verdicts[1]?.reason.kind, "media_only_blue");
    assert.equal(res.verdicts[2]?.action, "keep", "unverified zero-text reaction GIF is kept");
  });

  it("detects @grok pure summons and @grok bot_answer while sparing substantive Grok discussions", () => {
    const engine = new ThreadFilterEngine();
    const weights = createInitialClassifierWeights();
    const threadId = "1300" as ThreadId;

    const op = makeCandidate(
      "1300",
      "news_op",
      "Container shipping rates dropped 24% since September according to port authority data.",
      { isOp: true },
    );
    const summonEn = makeCandidate("1301", "ask_1", "@grok is this true?", {
      graphqlMeta: makeGqlMeta(),
    });
    const summonZh = makeCandidate("1302", "ask_2", "@grok 这是真的吗？帮我核实一下", {
      graphqlMeta: makeGqlMeta(),
    });
    const summonJa = makeCandidate("1303", "ask_3", "@grok これ本当？解説して", {
      graphqlMeta: makeGqlMeta(),
    });
    const grokAnswer = makeCandidate(
      "1304",
      "grok",
      "Yes, container freight indices declined roughly 24% since September.",
      { graphqlMeta: makeGqlMeta() },
    );
    const humanGrokCritique = makeCandidate(
      "1305",
      "hannah_journo",
      "What drives me crazy is people yelling @grok is this true under Onion satire instead of reading the Reuters link.",
      { graphqlMeta: makeGqlMeta() },
    );
    const humanGrokBench = makeCandidate(
      "1306",
      "victor_ml",
      "Grok pulls breaking X posts fast during earnings calls, but its 10-Q table accuracy still trails Claude and Gemini.",
      { graphqlMeta: makeGqlMeta() },
    );

    const res = engine.ingestEmbeddedBatch(
      threadId,
      [
        { candidate: op, embedding: makeUnitVec(1), tokenCount: asTokenCount(14) },
        { candidate: summonEn, embedding: makeUnitVec(2), tokenCount: asTokenCount(3) },
        { candidate: summonZh, embedding: makeUnitVec(3), tokenCount: asTokenCount(6) },
        { candidate: summonJa, embedding: makeUnitVec(4), tokenCount: asTokenCount(6) },
        { candidate: grokAnswer, embedding: makeUnitVec(5), tokenCount: asTokenCount(10) },
        { candidate: humanGrokCritique, embedding: getCalibratedHumanVec(0), tokenCount: asTokenCount(20) },
        { candidate: humanGrokBench, embedding: getCalibratedHumanVec(1), tokenCount: asTokenCount(20) },
      ],
      weights,
    );

    assert.equal(res.verdicts[0]?.action, "dim");
    assert.equal(res.verdicts[0]?.reason.kind, "bot_summon");
    assert.equal(res.verdicts[1]?.action, "dim");
    assert.equal(res.verdicts[1]?.reason.kind, "bot_summon");
    assert.equal(res.verdicts[2]?.action, "dim");
    assert.equal(res.verdicts[2]?.reason.kind, "bot_summon");
    assert.equal(res.verdicts[3]?.action, "dim");
    assert.equal(res.verdicts[3]?.reason.kind, "bot_answer");
    assert.equal(res.verdicts[4]?.action, "keep", "substantive critique mentioning @grok is kept");
    assert.equal(res.verdicts[5]?.action, "keep", "substantive discussion mentioning Grok is kept");
  });

  it("never clone-dims short human reactions (lol / 草) while retroactively dimming 8+ token bot clones", () => {
    const engine = new ThreadFilterEngine();
    const weights = createInitialClassifierWeights();
    const threadId = "4000" as ThreadId;
    const op = makeCandidate("4000", "op_user", "Major earthquake reported near the coast.", {
      isOp: true,
    });

    // Two distinct humans saying "lol" with identical vector -> MUST stay keep!
    const shortHuman1 = makeCandidate("4001", "human_a", "lol", { graphqlMeta: makeGqlMeta() });
    const shortHuman2 = makeCandidate("4002", "human_b", "lol", { graphqlMeta: makeGqlMeta() });
    const shortVec = makeUnitVec(15);

    const firstZombie = makeCandidate(
      "4003",
      "zombie_1",
      "現地の方々が無事であることを心よりお祈り申し上げます。",
      { graphqlMeta: makeGqlMeta() },
    );
    const secondZombie = makeCandidate(
      "4004",
      "zombie_2",
      "現地の方々が無事であることを心からお祈り申し上げます。",
      { graphqlMeta: makeGqlMeta() },
    );
    const zombieVec1 = makeUnitVec(20, 0);
    const zombieVec2 = makeUnitVec(20, 0.1);

    const batch1 = engine.ingestEmbeddedBatch(
      threadId,
      [
        { candidate: op, embedding: makeUnitVec(1), tokenCount: asTokenCount(10) },
        { candidate: shortHuman1, embedding: shortVec, tokenCount: asTokenCount(1) },
        { candidate: shortHuman2, embedding: shortVec, tokenCount: asTokenCount(1) },
        { candidate: firstZombie, embedding: zombieVec1, tokenCount: asTokenCount(14) },
      ],
      weights,
    );
    assert.equal(batch1.verdicts[0]?.action, "keep");
    assert.equal(batch1.verdicts[1]?.action, "keep", "identical 'lol' is NEVER clone-dimmed");
    assert.equal(batch1.verdicts[2]?.action, "keep");
    assert.equal(batch1.retroactiveUpdates.length, 0);

    const batch2 = engine.ingestEmbeddedBatch(
      threadId,
      [{ candidate: secondZombie, embedding: zombieVec2, tokenCount: asTokenCount(14) }],
      weights,
    );
    assert.equal(batch2.verdicts[0]?.action, "dim");
    assert.equal(batch2.verdicts[0]?.reason.kind, "thread_clone");
    assert.equal(batch2.retroactiveUpdates.length, 1);
    assert.equal(batch2.retroactiveUpdates[0]?.tweetId, "4003");
    assert.equal(batch2.retroactiveUpdates[0]?.action, "dim");
  });

  it("applies #16/#17 R9 op_paraphrase novelty guard (<=0.5 needs 1 family, 0.5-0.85 needs 2 families else R9b suspect)", () => {
    const weights = createInitialClassifierWeights();
    const params = getEffectiveV2Params(weights);
    const opVec = getCalibratedSlopVec();
    // cos ~ 0.94 with opVec
    const paraRaw = new Float32Array(opVec);
    paraRaw[0] = (paraRaw[0] ?? 0) + 0.3;
    const paraVec = truncateAndNormalize256(paraRaw, 0, EMBEDDING_DIM);

    const anchor = {
      kind: "text_anchored" as const,
      tweetId: "3000" as TweetId,
      authorId: "author_a" as AuthorId,
      tokenCount: asTokenCount(16),
      embedding: opVec,
      rawText:
        "Deploying quantized models inside an offscreen document keeps browser extension service workers completely stateless and fast.",
    };

    // High lexical overlap (l_novelty <= 0.5) + meta corroboration -> R9 dim
    const closePara = makeCandidate(
      "3001",
      "bot_para_close",
      "Deploying quantized models inside an offscreen document keeps extension service workers stateless.",
      {
        verified: true,
        graphqlMeta: makeGqlMeta({
          isBlueVerified: true,
          verifiedType: "blue",
          following: false,
          followersCount: 80,
          friendsCount: 2400,
          statusesCount: 15000,
          description: "Builder | AI daily frameworks 🚀",
        }),
      },
    );
    const featClose = extractCompactFeaturesV2({
      candidate: closePara,
      replyVec: paraVec,
      anchor,
      protoBank: params.protoBank,
      cloneCtx: { maxCloneSim: 0, clusterSize: 1 },
    });
    assert(featClose.vector[8]! <= 0.5, "close paraphrase has l_novelty <= 0.5");
    const scoreClose = scoreCompactV2(featClose, weights);
    const vClose = evaluateVerdictRulesV2({
      candidate: closePara,
      tokenCount: asTokenCount(12),
      anchor,
      feat: featClose,
      scoreResult: scoreClose,
      canonicalCloneTweetId: null,
      canonicalTokenCount: 0,
      cloneClusterSize: 1,
      authorStats: undefined,
      settings: DEFAULT_FILTER_SETTINGS,
    });
    assert.equal(vClose.action, "dim");
    assert.equal(vClose.reason.kind, "op_paraphrase");

    // Synonym paraphrase (0.5 < l_novelty <= 0.85) from neutral account (only 1 family: embedding) -> R9b suspect!
    const synParaNeutral = makeCandidate(
      "3002",
      "neutral_reword",
      "Running quantized models in a background document allows service workers to avoid holding persistent state.",
      {
        graphqlMeta: makeGqlMeta({
          isBlueVerified: false,
          verifiedType: "none",
          following: false,
          followersCount: 1200,
          friendsCount: 400,
          favoriteCount: 5,
        }),
      },
    );
    const featSyn = extractCompactFeaturesV2({
      candidate: synParaNeutral,
      replyVec: paraVec,
      anchor,
      protoBank: params.protoBank,
      cloneCtx: { maxCloneSim: 0, clusterSize: 1 },
    });
    assert(featSyn.vector[8]! > 0.5 && featSyn.vector[8]! <= 0.85, "synonym paraphrase novelty in (0.5, 0.85]");
    const scoreSyn = scoreCompactV2(featSyn, weights);
    // Single-family corroboration + borderline probability -> falls through R9 and R10 to R9b suspect!
    const vSynOneFamily = evaluateVerdictRulesV2({
      candidate: synParaNeutral,
      tokenCount: asTokenCount(14),
      anchor,
      feat: featSyn,
      scoreResult: {
        ...scoreSyn,
        probability: asProbability(0.55),
        corroboratingFamilies: ["embedding"],
        familyLogits: { ...scoreSyn.familyLogits, style: 0.1, meta: 0.1, embedding: 0.8 },
      },
      canonicalCloneTweetId: null,
      canonicalTokenCount: 0,
      cloneClusterSize: 1,
      authorStats: undefined,
      settings: DEFAULT_FILTER_SETTINGS,
    });
    assert.equal(vSynOneFamily.action, "suspect", "0.5 < novelty <= 0.85 with only 1 family falls to R9b suspect");
    assert.equal(vSynOneFamily.explanation?.ruleId, "op_paraphrase_soft");

    // #30/#33 MAJOR: When high-confidence model score (p >= tau_dim, >=2 families) also satisfies R9b,
    // R10 siamese_bot (dim) MUST take precedence over R9b op_paraphrase_soft (suspect)!
    const vR10BeatsR9b = evaluateVerdictRulesV2({
      candidate: synParaNeutral,
      tokenCount: asTokenCount(14),
      anchor,
      feat: featSyn,
      scoreResult: {
        ...scoreSyn,
        probability: asProbability(0.88),
        corroboratingFamilies: ["relational", "embedding"],
        familyLogits: { ...scoreSyn.familyLogits, relational: 0.9, style: 0.1, meta: 0.1, embedding: 0.8 },
      },
      canonicalCloneTweetId: null,
      canonicalTokenCount: 0,
      cloneClusterSize: 1,
      authorStats: undefined,
      settings: DEFAULT_FILTER_SETTINGS,
    });
    assert.equal(vR10BeatsR9b.action, "dim", "R10 siamese_bot runs BEFORE R9b op_paraphrase_soft");
    assert.equal(vR10BeatsR9b.reason.kind, "siamese_bot");

    // Same synonym paraphrase with 2 corroborating families -> R9 dim!
    const vSynTwoFamilies = evaluateVerdictRulesV2({
      candidate: synParaNeutral,
      tokenCount: asTokenCount(14),
      anchor,
      feat: featSyn,
      scoreResult: {
        ...scoreSyn,
        familyLogits: { ...scoreSyn.familyLogits, style: 0.7, meta: 0.6, embedding: 0.8 },
      },
      canonicalCloneTweetId: null,
      canonicalTokenCount: 0,
      cloneClusterSize: 1,
      authorStats: undefined,
      settings: DEFAULT_FILTER_SETTINGS,
    });
    assert.equal(vSynTwoFamilies.action, "dim", "0.5 < novelty <= 0.85 with >=2 families dims via R9");
    assert.equal(vSynTwoFamilies.reason.kind, "op_paraphrase");
  });

  it("transitions to media_only anchor when OP has fewer than 5 tokens and zeroes relational features", () => {
    const engine = new ThreadFilterEngine();
    const weights = createInitialClassifierWeights();
    const threadId = "2000" as ThreadId;
    const mediaOp = makeCandidate("2000", "artist", "wip", { isOp: true });
    const reply = makeCandidate(
      "2001",
      "viewer",
      "The lighting on the background clouds looks super clean here.",
    );
    const sharedVec = getCalibratedHumanVec();

    const result = engine.ingestEmbeddedBatch(
      threadId,
      [
        { candidate: mediaOp, embedding: sharedVec, tokenCount: asTokenCount(2) },
        { candidate: reply, embedding: sharedVec, tokenCount: asTokenCount(14) },
      ],
      weights,
    );

    assert.equal(result.anchorKind, "media_only");
    assert.equal(result.verdicts[0]?.action, "keep");
    assert.notEqual(result.verdicts[0]?.reason.kind, "op_paraphrase");

    const legacyFeatures = buildSiameseFeatures(
      sharedVec,
      {
        kind: "media_only",
        tweetId: "2000" as TweetId,
        authorId: "artist" as AuthorId,
        tokenCount: asTokenCount(2),
      },
      reply.domSignals,
    );
    for (let i = 256; i < 768; i++) {
      assert.equal(legacyFeatures[i], 0);
    }
  });

  it("enforces caps C1–C5 and zero-GPU preset/toggle re-evaluation", () => {
    const engine = new ThreadFilterEngine();
    const weights = createInitialClassifierWeights();
    const threadId = "1500" as ThreadId;
    const slopVec = getCalibratedSlopVec();
    const op = makeCandidate("1500", "op_user", "Announcing our new developer platform release.", {
      isOp: true,
    });

    const orgReply = makeCandidate(
      "1501",
      "corp_account",
      "Spot on! This isn't just an upgrade — it's a game-changer for the entire ecosystem. What do you think?",
      {
        verified: true,
        graphqlMeta: makeGqlMeta({ isBlueVerified: true, verifiedType: "business" }),
      },
    );
    const highLikeReply = makeCandidate(
      "1502",
      "popular_reply",
      "Spot on! This isn't just an upgrade — it's a game-changer for the entire ecosystem. What do you think?",
      {
        verified: true,
        graphqlMeta: makeGqlMeta({
          isBlueVerified: true,
          verifiedType: "blue",
          favoriteCount: 250,
        }),
      },
    );
    const followedByReply = makeCandidate(
      "1503",
      "my_follower",
      "Spot on! This isn't just an upgrade — it's a game-changer for the entire ecosystem. What do you think?",
      {
        verified: true,
        graphqlMeta: makeGqlMeta({
          isBlueVerified: true,
          verifiedType: "blue",
          followedBy: true,
        }),
      },
    );
    const summonReply = makeCandidate("1504", "summoner", "@grok explain this", {
      graphqlMeta: makeGqlMeta(),
    });

    const res = engine.ingestEmbeddedBatch(
      threadId,
      [
        { candidate: op, embedding: makeUnitVec(1), tokenCount: asTokenCount(10) },
        { candidate: orgReply, embedding: slopVec, tokenCount: asTokenCount(18) },
        { candidate: highLikeReply, embedding: slopVec, tokenCount: asTokenCount(18) },
        { candidate: followedByReply, embedding: slopVec, tokenCount: asTokenCount(18) },
        { candidate: summonReply, embedding: makeUnitVec(2), tokenCount: asTokenCount(2) },
      ],
      weights,
    );

    assert.equal(res.verdicts[0]?.action, "suspect", "C1 org-verified capped to suspect");
    assert(res.verdicts[0]?.explanation?.caps.includes("C1_org_verified"));
    assert.equal(res.verdicts[1]?.action, "suspect", "C2 high-likes (>=100) capped to suspect");
    assert(res.verdicts[1]?.explanation?.caps.includes("C2_high_likes"));
    assert.equal(res.verdicts[2]?.action, "suspect", "C3 followedBy=true capped to suspect");
    assert(res.verdicts[2]?.explanation?.caps.includes("C3_followed_by"));
    assert.equal(res.verdicts[3]?.action, "dim", "R4 bot_summon dims at Balanced");

    // Switch to Conservative + hide suspect badges -> ZERO GPU re-embedding via filterUnseenCandidates!
    const conservativeHidden: FilterSettings = {
      preset: "conservative",
      toggles: { ...DEFAULT_FILTER_SETTINGS.toggles, showSuspectBadges: false },
      revision: 2,
    };
    const remount = engine.filterUnseenCandidates(
      threadId,
      [op, orgReply, highLikeReply, followedByReply, summonReply],
      weights,
      conservativeHidden,
    );
    assert.equal(remount.toEmbed.length, 0, "settings change re-evaluates with ZERO GPU calls");
    assert.equal(remount.cached.length, 4);
    assert.equal(remount.cached[0]?.action, "keep", "C5 hides suspect when showSuspectBadges=false");
    assert(remount.cached[0]?.explanation?.caps.includes("C5_hide_suspect"));
    assert.equal(remount.cached[3]?.action, "keep", "Conservative + C5 renders summon as keep");
  });
});

describe("Online replay learner, rollback gate, undo/export/import/reset, and Stage-2 policy", () => {
  it("enforces 8% topicHash bucket cap, sign constraints, drift clip, and anchor-set rollback gate", () => {
    const weights = createInitialClassifierWeights();
    const baseVec = makeUnitVec(7);
    const hash = computeTopicHash8(baseVec);
    let buf: FeedbackReplayEntry[] = [];

    // Flood 50 items into identical topicHash bucket -> capped at floor(512 * 0.08) = 40
    for (let i = 0; i < 50; i++) {
      buf = insertIntoReplayBuffer(buf, {
        tweetId: `${90000 + i}` as TweetId,
        z: new Array(COMPACT_V2_DIM).fill(0),
        fixed: 0,
        y: i % 2 === 0 ? 1 : 0,
        lang: "en",
        tsMs: 1_700_000_000_000 + i * 1000,
        topicHash: hash,
        source: i % 2 === 0 ? "user_spam" : "user_human",
      });
    }
    assert.equal(buf.length, 40, "8% topicHash bucket cap enforces max 40 entries per topic");

    // Train against an impossibly strict anchor gate (targetPrecisionDim = 1.05) -> must roll back!
    const impossibleGate = { ...ANCHOR_SET_V2, targetPrecisionDim: 1.05 };
    const poisoned = trainFromReplayBuffer(weights, buf.slice(0, 5), {
      anchorSet: impossibleGate,
    });
    assert.equal(poisoned.rolledBack, true, "failing anchor gate triggers rollback");
    assert.equal(poisoned.nextWeights.rollbackCount, 1);
    assert.deepEqual(
      Array.from(poisoned.nextWeights.theta ?? []),
      Array.from(weights.theta ?? []),
      "rolled-back update leaves theta completely unchanged",
    );

    // Normal valid update passes anchor gate, preserves sign constraints and ||theta - theta0||_inf <= 1.5
    const validUpdate = trainFromReplayBuffer(weights, buf.slice(0, 8));
    assert.equal(validUpdate.rolledBack, false);
    const nextTheta = validUpdate.nextWeights.theta!;
    const theta0 = validUpdate.nextWeights.theta0!;
    for (let j = 0; j < COMPACT_V2_DIM; j++) {
      assert(Math.abs(nextTheta[j]! - theta0[j]!) <= 1.5 + 1e-5, `drift clip holds at dim ${j}`);
      if (FEATURE_SIGNS[j] === 1) assert(nextTheta[j]! >= 0, `positive sign holds at dim ${j}`);
      if (FEATURE_SIGNS[j] === -1) assert(nextTheta[j]! <= 0, `negative sign holds at dim ${j}`);
    }
  });

  it("supports feedback after slab LRU eviction, undo, export/import round-trip, reset, and auto-flag decay", () => {
    const engine = new ThreadFilterEngine(64, undefined, 2);
    let weights = createInitialClassifierWeights();

    const op1 = makeCandidate("5000", "op_1", "Thread 1 about compiler optimization.", { isOp: true });
    const r1 = makeCandidate(
      "5001",
      "subtle_bot_1",
      "This is a subtle synthetic reply without obvious bait keywords.",
      { graphqlMeta: makeGqlMeta() },
    );

    engine.ingestEmbeddedBatch(
      "5000" as ThreadId,
      [
        { candidate: op1, embedding: makeUnitVec(0), tokenCount: asTokenCount(9) },
        { candidate: r1, embedding: makeUnitVec(42), tokenCount: asTokenCount(12) },
      ],
      weights,
    );

    // Evict thread 5000 from 2-slot LRU slab by visiting two newer threads
    engine.filterUnseenCandidates("5100" as ThreadId, []);
    engine.filterUnseenCandidates("5200" as ThreadId, []);
    assert.equal(engine.getCachedThreadCount(), 2);
    assert(engine.getCachedFeatureCount() >= 1);

    // Feedback on evicted thread 5000 STILL works via 2000-entry featureCacheByTweetId!
    const fb = engine.applyUserFeedback("5000" as ThreadId, "5001" as TweetId, "spam", weights);
    assert(fb !== null, "feedback succeeds even after thread slab LRU eviction");
    assert.equal(fb.updatedVerdict.action, "dim");
    assert.equal(fb.updatedVerdict.reason.kind, "user_override");
    weights = fb.nextWeights;
    assert.equal(weights.replayBuffer?.length, 1);
    assert.equal(weights.authorStats.get("subtle_bot_1" as AuthorId)?.spamCount, 1);

    // Export -> Reset -> Import round-trip
    const exported = engine.exportFeedbackJson(weights);
    const resetW = engine.resetFeedback(weights);
    assert.equal(resetW.replayBuffer?.length, 0);
    assert.equal(resetW.authorStats.size, 0);

    const imported = engine.importFeedbackJson(exported, resetW);
    assert(imported !== null);
    assert.equal(imported.importedReplayCount, 1);
    assert.equal(imported.importedAuthorCount, 1);
    assert.equal(engine.importFeedbackJson("{not valid json", resetW), null);

    // Undo last feedback restores exact previous state
    const undone = engine.undoLastFeedback(weights);
    assert(undone !== null);
    assert.equal(undone.undoneTweetId, "5001");
    assert.equal(undone.undoneLabel, "spam");
    assert.equal(undone.nextWeights.replayBuffer?.length, 0);
    assert.equal(undone.nextWeights.authorStats.get("subtle_bot_1" as AuthorId), undefined);

    // Auto-flag dedup per tweetId & 30-day half-life decay
    const t0 = Date.parse("2026-10-01T00:00:00Z");
    const af1 = recordAuthorOutcome(resetW, "spammer_x" as AuthorId, "auto_flag", {
      tweetId: "8801" as TweetId,
      nowMs: t0,
    });
    const afDup = recordAuthorOutcome(af1, "spammer_x" as AuthorId, "auto_flag", {
      tweetId: "8801" as TweetId,
      nowMs: t0 + 1000,
    });
    assert.equal(afDup.authorStats.get("spammer_x" as AuthorId)?.autoFlagCount, 1, "dedups same tweetId");
    const after30d = getDecayedAutoFlags(
      afDup.authorStats.get("spammer_x" as AuthorId),
      t0 + 30 * 24 * 3600 * 1000,
    );
    assert(Math.abs(after30d - 0.5) < 1e-4, "30-day half-life decays auto-flag weight to 0.5");

    // Legacy PA updater still works for backward compatibility
    const legacyFeat = buildSiameseFeatures(
      makeUnitVec(42),
      {
        kind: "text_anchored",
        tweetId: op1.tweetId,
        authorId: op1.authorId,
        tokenCount: asTokenCount(9),
        embedding: makeUnitVec(0),
      },
      r1.domSignals,
    );
    const before = scoreSiameseHead(legacyFeat, makeUnitVec(42), r1.rawText, resetW).probability;
    const afterPa = updateOnlineWeightsPassiveAggressive(resetW, legacyFeat, 1);
    const after = scoreSiameseHead(legacyFeat, makeUnitVec(42), r1.rawText, afterPa).probability;
    assert(after > before);
  });

  it("tracks per-author reputation and retroactively dims peers after 2 spam marks", () => {
    const engine = new ThreadFilterEngine();
    let weights = createInitialClassifierWeights();
    const threadId = "6000" as ThreadId;
    const humanVec = getCalibratedHumanVec();

    const op = makeCandidate("6000", "op_user", "Testing account-level spam memory across replies.", {
      isOp: true,
    });
    const r1 = makeCandidate("6001", "repeat_spammer", "First reply that looks superficially normal.", {
      graphqlMeta: makeGqlMeta(),
    });
    const r2 = makeCandidate("6002", "repeat_spammer", "Second reply from the same account in thread.", {
      graphqlMeta: makeGqlMeta(),
    });
    const r3 = makeCandidate("6003", "repeat_spammer", "Third reply arriving in the same thread.", {
      graphqlMeta: makeGqlMeta(),
    });
    const r4 = makeCandidate("6004", "repeat_spammer", "ok", { graphqlMeta: makeGqlMeta() });

    const res1 = engine.ingestEmbeddedBatch(
      threadId,
      [
        { candidate: op, embedding: makeUnitVec(0), tokenCount: asTokenCount(10) },
        { candidate: r1, embedding: humanVec, tokenCount: asTokenCount(10) },
        { candidate: r2, embedding: humanVec, tokenCount: asTokenCount(10) },
        { candidate: r3, embedding: humanVec, tokenCount: asTokenCount(10) },
      ],
      weights,
    );
    assert.equal(res1.verdicts[0]?.action, "keep");
    assert.equal(res1.verdicts[1]?.action, "keep");
    assert.equal(res1.verdicts[2]?.action, "keep");

    const fb1 = engine.applyUserFeedback(threadId, r1.tweetId, "spam", weights);
    assert(fb1 !== null);
    assert.equal(fb1.retroactiveUpdates.length, 0);
    weights = fb1.nextWeights;

    const fb2 = engine.applyUserFeedback(threadId, r2.tweetId, "spam", weights);
    assert(fb2 !== null);
    weights = fb2.nextWeights;
    assert.equal(fb2.retroactiveUpdates.length, 1);
    assert.equal(fb2.retroactiveUpdates[0]?.tweetId, "6003");
    assert.equal(fb2.retroactiveUpdates[0]?.reason.kind, "known_bot_author");

    const res2 = engine.ingestEmbeddedBatch(
      threadId,
      [{ candidate: r4, embedding: humanVec, tokenCount: asTokenCount(1) }],
      weights,
    );
    assert.equal(res2.verdicts[0]?.action, "dim");
    assert.equal(res2.verdicts[0]?.reason.kind, "known_bot_author");

    const diag = engine.getDiagnosticsSummary(weights);
    assert.equal(diag.totalScoredReplies, 4);
    assert.equal(diag.dimmedReplies, 4);
    assert.equal(diag.topFlaggedAuthors[0]?.authorId, "repeat_spammer");
    assert.equal(diag.topFlaggedAuthors[0]?.spamCount, 2);
  });

  it("enforces Stage-2 asymmetric safety policy (human>=0.85 -> keep, bot>=0.92 -> dim, never overrides shields/caps)", () => {
    const engine = new ThreadFilterEngine();
    const weights = createInitialClassifierWeights();
    const threadId = "7000" as ThreadId;
    const slopVec = getCalibratedSlopVec();
    const op = makeCandidate("7000", "op_user", "Benchmarking browser AI models locally.", {
      isOp: true,
    });

    // 1. Borderline model_suspect reply (p >= tau_suspect, < tau_dim or 1 family, no caps)
    const suspectModel = makeCandidate(
      "7001",
      "borderline_model",
      "Spot on! This really clarifies how modern software teams balance reliability against deployment speed.",
      {
        graphqlMeta: makeGqlMeta({
          isBlueVerified: false,
          verifiedType: "none",
          following: false,
          followersCount: 600,
          friendsCount: 300,
          statusesCount: 800,
          favoriteCount: 2,
        }),
      },
    );
    // 2. DOM-only blue_short_praise suspect (following=null -> Stage-2 MUST NOT override DOM-only demotion to dim,
    //    but Stage-2 human >= 0.85 CAN clear it to keep!)
    const suspectDomBlue = makeCandidate(
      "7002",
      "borderline_dom_blue",
      "Spot on! Really resonates!",
      { verified: true, graphqlMeta: null },
    );
    const cappedOrg = makeCandidate(
      "7003",
      "org_acct",
      "Spot on! This isn't just a win — it's a paradigm shift for the entire landscape. What do you think?",
      {
        verified: true,
        graphqlMeta: makeGqlMeta({ isBlueVerified: true, verifiedType: "business" }),
      },
    );
    const shieldedFollowed = makeCandidate("7004", "my_friend", "Fire 🔥🔥🔥", {
      verified: true,
      graphqlMeta: makeGqlMeta({ isBlueVerified: true, verifiedType: "blue", following: true }),
    });

    const res = engine.ingestEmbeddedBatch(
      threadId,
      [
        { candidate: op, embedding: makeUnitVec(1), tokenCount: asTokenCount(8) },
        { candidate: suspectModel, embedding: slopVec, tokenCount: asTokenCount(14) },
        { candidate: suspectDomBlue, embedding: slopVec, tokenCount: asTokenCount(4) },
        { candidate: cappedOrg, embedding: slopVec, tokenCount: asTokenCount(16) },
        { candidate: shieldedFollowed, embedding: slopVec, tokenCount: asTokenCount(2) },
      ],
      weights,
    );

    assert.equal(res.verdicts[0]?.action, "suspect");
    assert.equal(res.verdicts[0]?.reason.kind, "model_suspect");
    assert.equal(res.verdicts[1]?.action, "suspect");
    assert.equal(res.verdicts[1]?.reason.kind, "blue_short_praise");
    assert.equal(res.verdicts[2]?.action, "suspect");
    assert.equal(res.verdicts[3]?.action, "keep");

    // 1. #33 Minor: DOM-only blue_short_praise (following=null) MUST NOT be promoted to dim by Stage-2
    const domBlueBlocked = engine.applyStage2Verdict(threadId, "7002" as TweetId, {
      provider: "chrome_gemini_nano",
      verdict: "bot",
      confidence: 0.99,
      ms: 150,
    });
    assert(domBlueBlocked !== null && domBlueBlocked.changedTier === false);
    assert.equal(
      domBlueBlocked.updatedVerdict.action,
      "suspect",
      "Stage-2 bot promotion restricted to model_suspect/op_paraphrase_soft (never overrides DOM-only demotion)",
    );

    // 2. Human confidence >= 0.85 clears suspect -> keep
    const cleared = engine.applyStage2Verdict(threadId, "7002" as TweetId, {
      provider: "chrome_gemini_nano",
      verdict: "human",
      confidence: 0.9,
      ms: 180,
    });
    assert(cleared !== null && cleared.changedTier === true);
    assert.equal(cleared.updatedVerdict.action, "keep");
    assert.equal(cleared.updatedVerdict.explanation?.stage2?.provider, "chrome_gemini_nano");

    // 3. Bot confidence 0.80 (< 0.92) must NOT promote model_suspect -> dim
    const lowConfBot = engine.applyStage2Verdict(threadId, "7001" as TweetId, {
      provider: "chrome_gemini_nano",
      verdict: "bot",
      confidence: 0.8,
      ms: 160,
    });
    assert(lowConfBot !== null && lowConfBot.changedTier === false);
    assert.equal(lowConfBot.updatedVerdict.action, "suspect");

    // 4. Bot confidence >= 0.92 promotes uncapped model_suspect with >=1 family -> dim
    const highConfBot = engine.applyStage2Verdict(threadId, "7001" as TweetId, {
      provider: "chrome_gemini_nano",
      verdict: "bot",
      confidence: 0.95,
      ms: 170,
    });
    assert(highConfBot !== null && highConfBot.changedTier === true);
    assert.equal(highConfBot.updatedVerdict.action, "dim");
    assert.equal(highConfBot.updatedVerdict.reason.kind, "siamese_bot");

    // 5. C1 org-verified cap blocks Stage-2 bot promotion even at confidence 0.99
    const orgBlocked = engine.applyStage2Verdict(threadId, "7003" as TweetId, {
      provider: "chrome_gemini_nano",
      verdict: "bot",
      confidence: 0.99,
      ms: 150,
    });
    assert(orgBlocked !== null && orgBlocked.changedTier === false);
    assert.equal(orgBlocked.updatedVerdict.action, "suspect");

    // 6. R1 followed shield is NEVER touched by Stage-2
    const shieldBlocked = engine.applyStage2Verdict(threadId, "7004" as TweetId, {
      provider: "chrome_gemini_nano",
      verdict: "bot",
      confidence: 0.99,
      ms: 150,
    });
    assert(shieldBlocked !== null && shieldBlocked.changedTier === false);
    assert.equal(shieldBlocked.updatedVerdict.action, "keep");
  });

  it("validates v2 RPC payloads, GraphqlTweetMeta caps, and Stage-2 update notifications", () => {
    const longBio = "x".repeat(700);
    const longName = "N".repeat(200);
    const meta = parseGraphqlTweetMeta({
      displayName: longName,
      description: longBio,
      verifiedType: "blue",
      isBlueVerified: true,
      following: null,
      followedBy: false,
      followersCount: 120,
      friendsCount: 300,
      statusesCount: 400,
      mediaTypes: ["photo", "invalid_kind"],
    });
    assert(meta !== null);
    assert.equal(meta.description?.length, 512, "description capped at 512 chars");
    assert.equal(meta.displayName?.length, 128, "displayName capped at 128 chars");
    assert.equal(meta.following, null, "strict tri-state null preserved");
    assert.deepEqual(meta.mediaTypes, ["photo"]);

    const settingsReq = parseRpcRequestV2({
      kind: "update_settings",
      settings: DEFAULT_FILTER_SETTINGS,
      target: "offscreen",
    });
    assert.equal(settingsReq?.kind, "update_settings");

    assert.equal(parseRpcRequestV2({ kind: "get_settings" })?.kind, "get_settings");
    assert.equal(parseRpcRequestV2({ kind: "undo_feedback" })?.kind, "undo_feedback");
    assert.equal(parseRpcRequestV2({ kind: "export_feedback" })?.kind, "export_feedback");
    assert.equal(
      parseRpcRequestV2({ kind: "import_feedback", payloadJson: "{}" })?.kind,
      "import_feedback",
    );
    assert.equal(parseRpcRequestV2({ kind: "reset_feedback" })?.kind, "reset_feedback");
    assert.equal(
      parseRpcRequestV2({ kind: "trigger_stage2_download", action: "status" })?.kind,
      "trigger_stage2_download",
    );

    const s2Msg = parseStage2VerdictUpdate({
      kind: "stage2_verdict_update",
      threadId: "7000",
      tweetId: "7001",
      verdict: {
        tweetId: "7001",
        action: "keep",
        score: 0.12,
        tokenCount: 8,
        reason: { kind: "clean", action: "keep" },
      },
      target: "content",
    });
    assert(s2Msg !== null && s2Msg.kind === "stage2_verdict_update");
  });
});

describe("Step-4 audit (#33/#30) defect regressions", () => {
  it("MAJOR #1: spares normal artist/office-worker bios (置顶/副業) and drops s_syco_open from R3 bio_scam", () => {
    const engine = new ThreadFilterEngine();
    const weights = createInitialClassifierWeights();
    const params = getEffectiveV2Params(weights);
    const humanVec = getCalibratedHumanVec();
    const threadId = "8100" as ThreadId;

    const op = makeCandidate("8100", "art_op", "Sharing the final color pass on my illustration.", {
      isOp: true,
    });
    // Exact repro #1 from #33
    const zhArtist = makeCandidate(
      "8101",
      "zh_artist",
      "说得好！这张的光影处理真的很厉害",
      {
        graphqlMeta: makeGqlMeta({
          description: "插画师｜置顶是约稿信息",
          isBlueVerified: false,
          verifiedType: "none",
          following: false,
        }),
      },
    );
    // Exact repro #2 from #33
    const jaWorker = makeCandidate(
      "8102",
      "ja_worker",
      "素晴らしいですね、色使いが本当に好きです",
      {
        graphqlMeta: makeGqlMeta({
          description: "会社員。副業でブログ書いてます",
          isBlueVerified: false,
          verifiedType: "none",
          following: false,
        }),
      },
    );
    // Strong scam bio + syco reply WITHOUT link or CTA must NOT trigger R3 bio_scam
    const strongBioNoCta = makeCandidate(
      "8103",
      "scam_bio_no_cta",
      "Spot on! Great breakdown.",
      {
        graphqlMeta: makeGqlMeta({
          description: "Join my free telegram signal group t.me/vip_signals_99",
          isBlueVerified: false,
          verifiedType: "none",
          following: false,
        }),
      },
    );
    // Strong scam bio + explicit CTA in reply DOES trigger R3 bio_scam
    const strongBioWithCta = makeCandidate(
      "8104",
      "scam_bio_cta",
      "DM me for instant access to the VIP channel.",
      {
        graphqlMeta: makeGqlMeta({
          description: "Join my free telegram signal group t.me/vip_signals_99",
          isBlueVerified: false,
          verifiedType: "none",
          following: false,
        }),
      },
    );

    const featArtist = extractCompactFeaturesV2({
      candidate: zhArtist,
      replyVec: humanVec,
      anchor: { kind: "unanchored" },
      protoBank: params.protoBank,
      cloneCtx: { maxCloneSim: 0, clusterSize: 1 },
    });
    assert.equal(featArtist.vector[47], 0.5, "置顶 downgraded to weak bio spam (0.5)");

    const featJa = extractCompactFeaturesV2({
      candidate: jaWorker,
      replyVec: humanVec,
      anchor: { kind: "unanchored" },
      protoBank: params.protoBank,
      cloneCtx: { maxCloneSim: 0, clusterSize: 1 },
    });
    assert.equal(featJa.vector[47], 0.5, "副業 downgraded to weak bio spam (0.5)");

    const res = engine.ingestEmbeddedBatch(
      threadId,
      [
        { candidate: op, embedding: makeUnitVec(1), tokenCount: asTokenCount(10) },
        { candidate: zhArtist, embedding: getCalibratedHumanVec(0), tokenCount: asTokenCount(9) },
        { candidate: jaWorker, embedding: getCalibratedHumanVec(1), tokenCount: asTokenCount(11) },
        { candidate: strongBioNoCta, embedding: getCalibratedHumanVec(2), tokenCount: asTokenCount(4) },
        { candidate: strongBioWithCta, embedding: getCalibratedHumanVec(3), tokenCount: asTokenCount(8) },
      ],
      weights,
    );

    assert.equal(res.verdicts[0]?.action, "keep", "ZH artist with 置顶 bio kept");
    assert.notEqual(res.verdicts[0]?.reason.kind, "bio_scam");
    assert.equal(res.verdicts[1]?.action, "keep", "JA office worker with 副業 bio kept");
    assert.notEqual(res.verdicts[1]?.reason.kind, "bio_scam");
    assert.notEqual(
      res.verdicts[2]?.reason.kind,
      "bio_scam",
      "s_syco_open alone no longer triggers R3 bio_scam",
    );
    assert.equal(res.verdicts[3]?.action, "dim");
    assert.equal(res.verdicts[3]?.reason.kind, "bio_scam");
  });

  it("MAJOR #2 & Minor #2: anchors s_casual, enforces R7 !syco && slop_lex==0, and excludes 'Absolutely not'", () => {
    const weights = createInitialClassifierWeights();
    const params = getEffectiveV2Params(weights);
    const vec = makeUnitVec(5);
    const extract = (text: string, opts?: { verified?: boolean; meta?: GraphqlTweetMeta | null }) =>
      extractCompactFeaturesV2({
        candidate: makeCandidate("8201", "tester", text, {
          verified: opts?.verified ?? false,
          graphqlMeta: opts?.meta ?? makeGqlMeta(),
        }),
        replyVec: vec,
        anchor: { kind: "unanchored" },
        protoBank: params.protoBank,
        cloneCtx: { maxCloneSim: 0, clusterSize: 1 },
      });

    // All #33 false-positive repros must have s_casual === 0
    for (const bad of [
      "Based on the data, Q3 margins compressed by 4 points.",
      "The real question is whether enterprise demand holds.",
      "Check www.example.com for the full specification.",
      "shhh don't spoil the ending yet",
      "Huge 66% upside 🚀🔥",
      "确实，这个观点非常有洞察力，值得深思",
      "A ratio of 3:1 is typical for mid-tier accounts.",
    ]) {
      assert.equal(extract(bad).vector[28], 0, `s_casual must be 0 for: ${bad}`);
    }

    // Genuine casual reactions STILL match s_casual === 1
    for (const good of ["lol", "草", "666", "23333", "hhh", "まじかよww", "for real!", "based"]) {
      assert.equal(extract(good).vector[28], 1, `s_casual must be 1 for: ${good}`);
    }

    // Minor #2: "Absolutely not, that's wrong" must NOT trigger s_syco_open
    assert.equal(extract("Absolutely not, that's completely wrong.").vector[14], 0);
    assert.equal(extract("Absolutely love this breakdown!").vector[14], 1);

    // Blue unfollowed "Huge 66% upside 🚀🔥" must NOT be exempted by R7
    const engine = new ThreadFilterEngine();
    const blue66 = makeCandidate("8202", "reply_guy_66", "Huge 66% upside 🚀🔥", {
      verified: true,
      graphqlMeta: makeGqlMeta({ isBlueVerified: true, verifiedType: "blue", following: false }),
    });
    const zhSlop = makeCandidate(
      "8203",
      "zh_llm_bot",
      "确实，这个观点非常有洞察力，值得深思",
      {
        verified: true,
        graphqlMeta: makeGqlMeta({ isBlueVerified: true, verifiedType: "blue", following: false }),
      },
    );
    const batch = engine.ingestEmbeddedBatch(
      "8200" as ThreadId,
      [
        { candidate: blue66, embedding: getCalibratedSlopVec(), tokenCount: asTokenCount(4) },
        { candidate: zhSlop, embedding: getCalibratedSlopVec(), tokenCount: asTokenCount(9) },
      ],
      weights,
    );
    assert.notEqual(batch.verdicts[0]?.reason.kind, "short_human_exempt");
    assert.equal(batch.verdicts[0]?.action, "dim");
    assert.notEqual(batch.verdicts[1]?.reason.kind, "short_human_exempt");
    assert.equal(batch.verdicts[1]?.action, "dim");
  });

  it("MAJOR #3: makes m_reply_likes_rel shield-only and imputes missing meta at zHuman with 0 familyLogits", () => {
    const weights = createInitialClassifierWeights();
    const params = getEffectiveV2Params(weights);
    const humanVec = getCalibratedHumanVec();

    // 1. Fresh reply with 0 likes against 1000-like OP -> m_reply_likes_rel contrib MUST be 0 (never +0.70!)
    const freshZeroLikes = makeCandidate(
      "8301",
      "fresh_human",
      "We saw the exact same latency spike in our staging cluster yesterday.",
      {
        graphqlMeta: makeGqlMeta({
          isBlueVerified: false,
          verifiedType: "none",
          following: false,
          followersCount: 800,
          friendsCount: 400,
          statusesCount: 1500,
          favoriteCount: 0,
        }),
        opFavoriteCount: 1000,
      },
    );
    const featFresh = extractCompactFeaturesV2({
      candidate: freshZeroLikes,
      replyVec: humanVec,
      anchor: { kind: "unanchored" },
      protoBank: params.protoBank,
      cloneCtx: { maxCloneSim: 0, clusterSize: 1 },
    });
    const scoreFresh = scoreCompactV2(featFresh, weights);
    assert(
      !scoreFresh.top.some((c) => c.id === "m_reply_likes_rel"),
      "0-like fresh reply never receives positive bot contribution from m_reply_likes_rel",
    );
    assert(
      !scoreFresh.corroboratingFamilies.includes("meta"),
      "0-like fresh reply does not grant free 'meta' corroboration",
    );

    // Exact additivity still holds after shield-only clamp!
    const zHuman = CALIBRATED_WEIGHTS_V2.zHuman;
    const mu = CALIBRATED_WEIGHTS_V2.mu;
    const sigma = CALIBRATED_WEIGHTS_V2.sigma;
    let sumFresh = 0;
    for (let j = 0; j < COMPACT_V2_DIM; j++) {
      if (j === 4) {
        sumFresh +=
          CALIBRATED_WEIGHTS_V2.fixedCoefs.protoMargin *
          ((featFresh.vector[4] ?? 0) - ((mu[4] ?? 0) + (zHuman[4] ?? 0) * (sigma[4] ?? 1)));
      } else if (j === 5) {
        sumFresh +=
          CALIBRATED_WEIGHTS_V2.fixedCoefs.generic *
          ((featFresh.vector[5] ?? 0) - ((mu[5] ?? 0) + (zHuman[5] ?? 0) * (sigma[5] ?? 1)));
      } else {
        sumFresh += (weights.theta![j] ?? 0) * ((scoreFresh.z[j] ?? 0) - (zHuman[j] ?? 0));
      }
    }
    assert(
      Math.abs(scoreFresh.baseHumanLogit + sumFresh - scoreFresh.logit) < 1e-4,
      "exact additivity preserved under shield-only m_reply_likes_rel clamp",
    );

    // 2. High-like reply (500 likes vs 1000 OP) -> negative shield contribution!
    const highLikes = makeCandidate("8302", "liked_human", freshZeroLikes.rawText, {
      graphqlMeta: makeGqlMeta({ favoriteCount: 500 }),
      opFavoriteCount: 1000,
    });
    const featLiked = extractCompactFeaturesV2({
      candidate: highLikes,
      replyVec: humanVec,
      anchor: { kind: "unanchored" },
      protoBank: params.protoBank,
      cloneCtx: { maxCloneSim: 0, clusterSize: 1 },
    });
    const scoreLiked = scoreCompactV2(featLiked, weights);
    assert(
      scoreLiked.counter.some((c) => c.id === "m_reply_likes_rel" && c.contribution < -0.5),
      "high-like reply keeps negative human-shield contribution from m_reply_likes_rel",
    );

    // 3. DOM-only reply (graphqlMeta === null, unverified) -> missing meta imputed at zHuman, 0 free meta family!
    const domOnly = makeCandidate("8303", "dom_human", freshZeroLikes.rawText, {
      graphqlMeta: null,
    });
    const featDom = extractCompactFeaturesV2({
      candidate: domOnly,
      replyVec: humanVec,
      anchor: { kind: "unanchored" },
      protoBank: params.protoBank,
      cloneCtx: { maxCloneSim: 0, clusterSize: 1 },
    });
    assert.equal(featDom.missingMask[36], 1, "m_meta_present marked missing in DOM-only mode");
    assert.equal(featDom.missingMask[50], 1, "m_reply_likes_rel marked missing in DOM-only mode");
    const scoreDom = scoreCompactV2(featDom, weights);
    assert(
      scoreDom.familyLogits.meta <= 0,
      `DOM-only unverified reply must not get positive meta logit (got ${scoreDom.familyLogits.meta})`,
    );
    assert(!scoreDom.top.some((c) => c.id === "m_meta_present"));
  });

  it("MAJOR #5: retroactively promotes suspect clone peers and persists across reevaluateThread without flip-flopping", () => {
    const engine = new ThreadFilterEngine();
    const weights = createInitialClassifierWeights();
    const threadId = "8400" as ThreadId;
    const slopVec = getCalibratedSlopVec();
    const op = makeCandidate("8400", "op_user", "Launching our open-source local inference demo.", {
      isOp: true,
    });

    // First zombie starts as model_suspect (borderline probability, single corroborating family)
    const zombieA = makeCandidate(
      "8401",
      "clone_bot_a",
      "Spot on! This really clarifies how modern software teams balance reliability against deployment speed.",
      {
        graphqlMeta: makeGqlMeta({
          isBlueVerified: false,
          verifiedType: "none",
          following: false,
          followersCount: 600,
          friendsCount: 300,
          statusesCount: 800,
          favoriteCount: 2,
        }),
      },
    );
    const zombieB = makeCandidate(
      "8402",
      "clone_bot_b",
      "Spot on! This really clarifies how modern software teams balance reliability against deployment speed.",
      {
        graphqlMeta: makeGqlMeta({
          isBlueVerified: false,
          verifiedType: "none",
          following: false,
          followersCount: 600,
          friendsCount: 300,
          statusesCount: 800,
          favoriteCount: 2,
        }),
      },
    );

    const b1 = engine.ingestEmbeddedBatch(
      threadId,
      [
        { candidate: op, embedding: makeUnitVec(1), tokenCount: asTokenCount(10) },
        { candidate: zombieA, embedding: slopVec, tokenCount: asTokenCount(14) },
      ],
      weights,
    );
    assert.equal(b1.verdicts[0]?.action, "suspect", "first zombie starts as suspect");

    const b2 = engine.ingestEmbeddedBatch(
      threadId,
      [{ candidate: zombieB, embedding: slopVec, tokenCount: asTokenCount(14) }],
      weights,
    );
    assert.equal(b2.verdicts[0]?.action, "dim");
    assert.equal(b2.verdicts[0]?.reason.kind, "thread_clone");
    assert.equal(b2.retroactiveUpdates.length, 1, "suspect peer is retroactively promoted to dim");
    assert.equal(b2.retroactiveUpdates[0]?.tweetId, "8401");
    assert.equal(b2.retroactiveUpdates[0]?.action, "dim");
    assert.equal(b2.retroactiveUpdates[0]?.reason.kind, "thread_clone");

    // Re-evaluating the thread (e.g. after settings or weightsRevision bump) MUST NOT flip zombieA back!
    const reeval = engine.reevaluateThread(threadId, weights, {
      ...DEFAULT_FILTER_SETTINGS,
      revision: 2,
    });
    const reevalA = reeval.find((v) => v.tweetId === "8401");
    assert.equal(reevalA?.action, "dim", "promoted canonical clone persists across re-evaluation");
    assert.equal(reevalA?.reason.kind, "thread_clone");
  });

  it("Minor #1, #4, #5 & #32-19: shields unanchored focalAuthorId, preserves replay on rollback, persists live precision, and merges imports", () => {
    const engine = new ThreadFilterEngine();
    let weights = createInitialClassifierWeights();
    const threadId = "8500" as ThreadId;
    const focalAuthor = "thread_author" as AuthorId;

    // No OP candidate in batch (OP scrolled off / virtualized away), only focalAuthorId from URL!
    const opFollowup = makeCandidate("8501", "thread_author", "@grok explain this chart", {
      verified: true,
      graphqlMeta: makeGqlMeta({ isBlueVerified: true, verifiedType: "blue", following: false }),
    });
    const filterRes = engine.filterUnseenCandidates(
      threadId,
      [opFollowup],
      weights,
      DEFAULT_FILTER_SETTINGS,
      focalAuthor,
    );
    assert.equal(filterRes.toEmbed.length, 1);

    const ingested = engine.ingestEmbeddedBatch(
      threadId,
      [{ candidate: opFollowup, embedding: getCalibratedSlopVec(), tokenCount: asTokenCount(4) }],
      weights,
    );
    assert.equal(ingested.anchorKind, "unanchored");
    assert.equal(ingested.verdicts[0]?.action, "keep", "focalAuthorId shields unanchored OP follow-up");
    assert(ingested.verdicts[0]?.explanation?.shields.includes("op_author"));

    // Minor #4: rollback preserves the new replay entry while reverting theta
    const entryA: FeedbackReplayEntry = {
      tweetId: "8510" as TweetId,
      z: new Array(COMPACT_V2_DIM).fill(0),
      fixed: 0,
      y: 1,
      lang: "en",
      tsMs: 1_700_000_000_000,
      topicHash: 11,
      source: "user_spam",
    };
    const impossibleGate = { ...ANCHOR_SET_V2, targetPrecisionDim: 1.05 };
    const rolled = trainFromReplayBuffer(weights, [entryA], { anchorSet: impossibleGate });
    assert.equal(rolled.rolledBack, true);
    assert.equal(rolled.nextWeights.replayBuffer?.length, 1, "rollback preserves replay entry");

    // Minor #5: live precision counters persist on weights & restore on undo
    const spamReply = makeCandidate(
      "8520",
      "bot_precision_test",
      "Check my pinned tweet for 100x crypto signals!",
      { graphqlMeta: makeGqlMeta() },
    );
    const bSpam = engine.ingestEmbeddedBatch(
      threadId,
      [{ candidate: spamReply, embedding: getCalibratedSlopVec(), tokenCount: asTokenCount(8) }],
      weights,
    );
    assert.equal(bSpam.verdicts[0]?.action, "dim");

    const fbHuman = engine.applyUserFeedback(threadId, "8520" as TweetId, "human", weights);
    assert(fbHuman !== null);
    weights = fbHuman.nextWeights;
    assert.equal(weights.dimsReviewedCount, 1);
    assert.equal(weights.dimsOverriddenHumanCount, 1);
    // Fresh engine instance reads persisted counters from weights!
    const freshEngine = new ThreadFilterEngine();
    assert.equal(freshEngine.getDiagnosticsSummary(weights).livePrecisionEstimate, 0);

    // #32 Minor #19: importFeedbackJson merges imported replayBuffer into existing replayBuffer
    const exportedJson = engine.exportFeedbackJson(weights);
    const otherWeights: typeof weights = {
      ...createInitialClassifierWeights(),
      replayBuffer: [entryA],
    };
    const merged = engine.importFeedbackJson(exportedJson, otherWeights);
    assert(merged !== null);
    assert.equal(
      merged.nextWeights.replayBuffer?.length,
      2,
      "import merges existing and imported replay entries instead of wiping existing ones",
    );

    const undone = engine.undoLastFeedback(weights);
    assert(undone !== null);
    assert.equal(undone.nextWeights.dimsReviewedCount, 0);
    assert.equal(undone.nextWeights.dimsOverriddenHumanCount, 0);
  });

  it("MAJOR #6: rpc.ts degrades malformed graphqlMeta, skips invalid candidates, rejects NaN ratios, caps sizes, and parses focalAuthorId", () => {
    const validCand = makeCandidate("8601", "good_user", "A".repeat(4500));
    const malformedMetaCand = {
      ...makeCandidate("8602", "drift_user", "Normal reply with drifted schema"),
      graphqlMeta: { favoriteCount: "not_a_number", verifiedType: "bogus" },
    };
    const invalidCand = {
      ...makeCandidate("8603", "bad_ratio_user", "NaN ratio"),
      domSignals: { ...validCand.domSignals, emojiToCharRatio: Number.NaN },
    };

    const req = parseRpcRequestV2({
      kind: "evaluate_batch",
      threadId: "8600",
      focalAuthorId: "@Thread_OP",
      candidates: [validCand, malformedMetaCand, invalidCand],
    });
    assert(req !== null && req.kind === "evaluate_batch");
    assert.equal(req.focalAuthorId, "thread_op");
    assert.equal(req.candidates.length, 2, "invalid candidate skipped instead of dropping whole batch");
    assert.equal(req.candidates[0]?.rawText.length, 4000, "rawText capped to 4000 chars");
    assert.equal(req.candidates[1]?.tweetId, "8602");
    assert.equal(req.candidates[1]?.graphqlMeta, null, "malformed graphqlMeta degraded to null");

    // Batch cap at 200 candidates
    const oversized = Array.from({ length: 225 }, (_, i) =>
      makeCandidate(`${9000 + i}`, `u_${i}`, "hello world"),
    );
    const cappedReq = parseRpcRequestV2({
      kind: "evaluate_batch",
      threadId: "8600",
      candidates: oversized,
    });
    assert(cappedReq !== null && cappedReq.kind === "evaluate_batch");
    assert.equal(cappedReq.candidates.length, 200, "candidates capped at 200");

    // parseStage2VerdictUpdate validates explanation shape and caps snippet
    assert.equal(
      parseStage2VerdictUpdate({
        kind: "stage2_verdict_update",
        threadId: "8600",
        tweetId: "8601",
        verdict: {
          tweetId: "8601",
          action: "keep",
          score: 0.1,
          tokenCount: 5,
          reason: { kind: "clean", action: "keep" },
          explanation: { schemaVersion: 999 },
        },
      }),
      null,
      "malformed explanation rejected",
    );
    const cappedS2 = parseStage2VerdictUpdate({
      kind: "stage2_verdict_update",
      threadId: "8600",
      tweetId: "8601",
      verdict: {
        tweetId: "8601",
        action: "keep",
        score: 0.1,
        tokenCount: 5,
        snippet: "S".repeat(300),
        reason: { kind: "clean", action: "keep" },
      },
    });
    assert(cappedS2 !== null);
    assert.equal(cappedS2.verdict.snippet?.length, 160, "Stage-2 snippet capped at 160 chars");
  });
});
