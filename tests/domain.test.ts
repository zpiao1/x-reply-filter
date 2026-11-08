import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  ThreadFilterEngine,
  buildSiameseFeatures,
  dotProduct256,
  scoreSiameseHead,
  updateOnlineWeightsPassiveAggressive,
} from "../src/domain/engine.ts";
import {
  createInitialClassifierWeights,
  extractDomSignalsFromText,
} from "../src/domain/seeds.ts";
import {
  type AuthorId,
  type CandidateTweet,
  EMBEDDING_DIM,
  RAW_EMBEDDING_DIM,
  type ThreadId,
  type TweetId,
  type UnitVec256,
  asTokenCount,
  parseAuthorId,
  parseThreadId,
  parseTweetId,
  truncateAndNormalize256,
  truncateAndNormalizeBatch256,
} from "../src/domain/types.ts";

function makeUnitVec(seedIndex: number, noise = 0): UnitVec256 {
  const raw = new Float32Array(RAW_EMBEDDING_DIM);
  raw[seedIndex % EMBEDDING_DIM] = 1;
  if (noise !== 0) {
    raw[(seedIndex + 1) % EMBEDDING_DIM] = noise;
  }
  return truncateAndNormalize256(raw, 0, RAW_EMBEDDING_DIM);
}

function makeCandidate(
  id: string,
  author: string,
  text: string,
  isOp = false,
  verified = false,
): CandidateTweet {
  const tweetId = parseTweetId(id);
  const authorId = parseAuthorId(author);
  assert(tweetId !== null && authorId !== null, "valid test IDs");
  return {
    tweetId,
    authorId,
    rawText: text,
    normalizedText: text.trim(),
    isOp,
    domSignals: extractDomSignalsFromText(text, verified, false),
  };
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
    assert(Math.abs(first[0]! - 0.6) < 1e-6, "first component normalized");
    assert(Math.abs(first[1]! - 0.8) < 1e-6, "second component normalized");
    assert(Math.abs(dotProduct256(first, first) - 1) < 1e-6, "unit norm on row 0");
    assert(Math.abs(dotProduct256(second, second) - 1) < 1e-6, "unit norm on row 1");
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

describe("ThreadFilterEngine 3-in-1 detection and edge-case invariants", () => {
  it("spares short CJK and English human reactions under 6 tokens unless cloned or keyword spam", () => {
    const engine = new ThreadFilterEngine();
    const weights = createInitialClassifierWeights();
    const threadId = "1000" as ThreadId;

    const op = makeCandidate("1000", "op_user", "New kernel latency benchmarks landed today.", true);
    const shortJp = makeCandidate("1001", "jp_human", "草www", false, true);
    const shortZh = makeCandidate("1002", "zh_human", "太牛逼了", false, true);
    const shortEn = makeCandidate("1003", "en_human", "lol true", false, true);
    const shortSpamZh = makeCandidate("1004", "zh_spammer", "看主页置顶自取", false, true);
    const verifiedEmojiZombie = makeCandidate("1005", "rankida600", "🙏🙏🙏🙏", false, true);

    const opVec = makeUnitVec(1);
    const result = engine.ingestEmbeddedBatch(
      threadId,
      [
        { candidate: op, embedding: opVec, tokenCount: asTokenCount(12) },
        { candidate: shortJp, embedding: opVec, tokenCount: asTokenCount(2) },
        { candidate: shortZh, embedding: makeUnitVec(2), tokenCount: asTokenCount(3) },
        { candidate: shortEn, embedding: makeUnitVec(3), tokenCount: asTokenCount(2) },
        { candidate: shortSpamZh, embedding: makeUnitVec(4), tokenCount: asTokenCount(4) },
        { candidate: verifiedEmojiZombie, embedding: makeUnitVec(5), tokenCount: asTokenCount(2) },
      ],
      weights,
    );

    assert.equal(result.anchorKind, "text_anchored");
    assert.equal(result.verdicts.length, 5);
    assert.equal(result.verdicts[0]?.action, "keep");
    assert.equal(result.verdicts[0]?.reason.kind, "short_human_exempt");
    assert.equal(result.verdicts[1]?.action, "keep");
    assert.equal(result.verdicts[1]?.reason.kind, "short_human_exempt");
    assert.equal(result.verdicts[2]?.action, "keep");
    assert.equal(result.verdicts[2]?.reason.kind, "short_human_exempt");
    assert.equal(result.verdicts[3]?.action, "dim");
    assert.equal(result.verdicts[3]?.reason.kind, "keyword_spam");
    assert.equal(
      result.verdicts[4]?.action,
      "dim",
      "verified emoji-only reply (🙏🙏🙏🙏) must be dimmed as bot, not exempted",
    );
  });

  it("transitions to media_only anchor when OP has fewer than 5 tokens and zeroes relational features", () => {
    const engine = new ThreadFilterEngine();
    const weights = createInitialClassifierWeights();
    const threadId = "2000" as ThreadId;

    const mediaOp = makeCandidate("2000", "artist", "wip", true);
    const reply = makeCandidate(
      "2001",
      "viewer",
      "The lighting on the background clouds looks super clean here.",
      false,
    );
    const sharedVec = makeUnitVec(5);

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

    const features = buildSiameseFeatures(
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
      assert.equal(features[i], 0, "relational delta and product blocks must be zeroed");
    }
  });

  it("flags OP paraphrases in the high cosine band while sparing OP self-replies and exact quotes", () => {
    const engine = new ThreadFilterEngine();
    const weights = createInitialClassifierWeights();
    const threadId = "3000" as ThreadId;

    const op = makeCandidate(
      "3000",
      "author_a",
      "Deploying quantized models in an offscreen document keeps service workers stateless.",
      true,
    );
    const opSelfReply = makeCandidate(
      "3001",
      "author_a",
      "Adding benchmarks from our Windows desktop run below.",
      false,
    );
    const paraphraseBot = makeCandidate(
      "3002",
      "bot_reply_guy",
      "Running quantized models inside an offscreen document really keeps the worker stateless.",
      false,
    );
    const exactQuoteReply = makeCandidate(
      "3003",
      "quoter",
      "Deploying quantized models in an offscreen document keeps service workers stateless.",
      false,
    );

    const opVec = makeUnitVec(10, 0);
    const paraphraseVec = makeUnitVec(10, 0.45);
    const selfVec = makeUnitVec(10, 0.35);

    const res = engine.ingestEmbeddedBatch(
      threadId,
      [
        { candidate: op, embedding: opVec, tokenCount: asTokenCount(15) },
        { candidate: opSelfReply, embedding: selfVec, tokenCount: asTokenCount(12) },
        { candidate: paraphraseBot, embedding: paraphraseVec, tokenCount: asTokenCount(16) },
        { candidate: exactQuoteReply, embedding: opVec, tokenCount: asTokenCount(15) },
      ],
      weights,
    );

    assert.equal(res.verdicts[0]?.action, "keep", "OP self-reply is never marked op_paraphrase");
    assert.equal(res.verdicts[1]?.action, "dim");
    assert.equal(res.verdicts[1]?.reason.kind, "op_paraphrase");
    assert.equal(res.verdicts[2]?.action, "keep", "exact quote >= 0.985 ceiling is spared");
  });

  it("detects cross-author thread clones and emits retroactive updates for the first clone peer", () => {
    const engine = new ThreadFilterEngine();
    const weights = createInitialClassifierWeights();
    const threadId = "4000" as ThreadId;

    const op = makeCandidate("4000", "op_user", "Major earthquake reported near the coast.", true);
    const firstZombie = makeCandidate(
      "4001",
      "zombie_1",
      "現地の方々が無事であることを心よりお祈り申し上げます。",
      false,
    );
    const secondZombie = makeCandidate(
      "4002",
      "zombie_2",
      "現地の方々が無事であることを心からお祈り申し上げます。",
      false,
    );

    const opVec = makeUnitVec(1);
    const zombieVec1 = makeUnitVec(20, 0);
    const zombieVec2 = makeUnitVec(20, 0.1);

    const batch1 = engine.ingestEmbeddedBatch(
      threadId,
      [
        { candidate: op, embedding: opVec, tokenCount: asTokenCount(10) },
        { candidate: firstZombie, embedding: zombieVec1, tokenCount: asTokenCount(14) },
      ],
      weights,
    );
    assert.equal(batch1.verdicts[0]?.action, "keep");
    assert.equal(batch1.retroactiveUpdates.length, 0);

    const batch2 = engine.ingestEmbeddedBatch(
      threadId,
      [{ candidate: secondZombie, embedding: zombieVec2, tokenCount: asTokenCount(14) }],
      weights,
    );
    assert.equal(batch2.verdicts[0]?.action, "dim");
    assert.equal(batch2.verdicts[0]?.reason.kind, "thread_clone");
    assert.equal(batch2.retroactiveUpdates.length, 1);
    assert.equal(batch2.retroactiveUpdates[0]?.tweetId, "4001");
    assert.equal(batch2.retroactiveUpdates[0]?.action, "dim");
  });

  it("deduplicates virtual-scroller remounts in filterUnseenCandidates and updates weights on user feedback", () => {
    const engine = new ThreadFilterEngine(512, undefined, 2);
    let weights = createInitialClassifierWeights();
    const threadId = "5000" as ThreadId;

    const op = makeCandidate("5000", "op_user", "Testing WebGPU inference latency on Turing.", true);
    const reply = makeCandidate(
      "5001",
      "subtle_bot",
      "This is a subtle synthetic reply without obvious bait keywords.",
      false,
    );
    const replyVec = makeUnitVec(42);

    engine.ingestEmbeddedBatch(
      threadId,
      [
        { candidate: op, embedding: makeUnitVec(0), tokenCount: asTokenCount(9) },
        { candidate: reply, embedding: replyVec, tokenCount: asTokenCount(12) },
      ],
      weights,
    );

    const remountCheck = engine.filterUnseenCandidates(threadId, [op, reply]);
    assert.equal(remountCheck.toEmbed.length, 0, "seen OP and reply are not re-embedded");
    assert.equal(remountCheck.cached.length, 1);

    const features = buildSiameseFeatures(
      replyVec,
      {
        kind: "text_anchored",
        tweetId: op.tweetId,
        authorId: op.authorId,
        tokenCount: asTokenCount(9),
        embedding: makeUnitVec(0),
      },
      reply.domSignals,
    );
    const beforeScore = scoreSiameseHead(features, replyVec, reply.rawText, weights).probability;

    const feedback = engine.applyUserFeedback(threadId, reply.tweetId, "spam", weights);
    assert(feedback !== null);
    assert.equal(feedback.updatedVerdict.action, "dim");
    assert.equal(feedback.updatedVerdict.reason.kind, "user_override");
    weights = feedback.nextWeights;

    const afterScore = scoreSiameseHead(features, replyVec, reply.rawText, weights).probability;
    assert(afterScore > beforeScore, "passive-aggressive spam update increases bot probability");

    const directUpdate = updateOnlineWeightsPassiveAggressive(weights, features, -1);
    assert.equal(directUpdate.updateCount, 2);
  });

  it("evicts the least-recently-used thread slab when exceeding maxCachedThreads", () => {
    const engine = new ThreadFilterEngine(64, undefined, 2);
    engine.filterUnseenCandidates("10" as ThreadId, []);
    engine.filterUnseenCandidates("20" as ThreadId, []);
    engine.filterUnseenCandidates("10" as ThreadId, []);
    engine.filterUnseenCandidates("30" as ThreadId, []);

    assert.equal(engine.getCachedThreadCount(), 2);
  });

  it("tracks per-author spam reputation, boosts prior after 1 mark, and auto-dims + retroactively dims after 2 marks", () => {
    const engine = new ThreadFilterEngine();
    let weights = createInitialClassifierWeights();
    const threadId = "6000" as ThreadId;

    const op = makeCandidate("6000", "op_user", "Testing account-level spam memory across replies.", true);
    const botReply1 = makeCandidate(
      "6001",
      "repeat_spammer",
      "First reply from this account that looks superficially benign.",
      false,
    );
    const botReply2 = makeCandidate(
      "6002",
      "repeat_spammer",
      "Second reply from the same account in the same thread.",
      false,
    );
    const botReply3 = makeCandidate(
      "6003",
      "repeat_spammer",
      "Third reply arriving later in the same thread.",
      false,
    );
    const botReply4 = makeCandidate(
      "6004",
      "repeat_spammer",
      "ok",
      false,
    );

    const res1 = engine.ingestEmbeddedBatch(
      threadId,
      [
        { candidate: op, embedding: makeUnitVec(0), tokenCount: asTokenCount(10) },
        { candidate: botReply1, embedding: makeUnitVec(50), tokenCount: asTokenCount(10) },
        { candidate: botReply2, embedding: makeUnitVec(51), tokenCount: asTokenCount(10) },
        { candidate: botReply3, embedding: makeUnitVec(52), tokenCount: asTokenCount(10) },
      ],
      weights,
    );
    assert.equal(res1.verdicts[0]?.action, "keep");
    assert.equal(res1.verdicts[1]?.action, "keep");
    assert.equal(res1.verdicts[2]?.action, "keep");

    const fb1 = engine.applyUserFeedback(threadId, botReply1.tweetId, "spam", weights);
    assert(fb1 !== null);
    assert.equal(fb1.retroactiveUpdates.length, 0, "1 spam mark boosts prior but does not yet hard-ban");
    weights = fb1.nextWeights;
    assert.equal(weights.authorStats.get("repeat_spammer" as AuthorId)?.spamCount, 1);

    const anchorState = {
      kind: "text_anchored" as const,
      tweetId: op.tweetId,
      authorId: op.authorId,
      tokenCount: asTokenCount(10),
      embedding: makeUnitVec(0),
    };
    const probeVec = makeUnitVec(99);
    const cleanAuthorFeatures = buildSiameseFeatures(
      probeVec,
      anchorState,
      botReply1.domSignals,
      "clean_human" as AuthorId,
      undefined,
    );
    const repeatAuthorFeatures = buildSiameseFeatures(
      probeVec,
      anchorState,
      botReply1.domSignals,
      "repeat_spammer" as AuthorId,
      weights.authorStats.get("repeat_spammer" as AuthorId),
    );
    const botHandleFeatures = buildSiameseFeatures(
      probeVec,
      anchorState,
      botReply1.domSignals,
      "tianata3005" as AuthorId,
      undefined,
    );
    assert.equal(repeatAuthorFeatures.length, 778);
    assert(repeatAuthorFeatures[774]! > 0, "author spamRate feature populated in SiameseFeatureVec[774]");
    assert(repeatAuthorFeatures[775]! > 0, "author spamVolume feature populated in SiameseFeatureVec[775]");
    assert(botHandleFeatures[777]! > 0, "handleBotScore feature populated in SiameseFeatureVec[777]");

    const cleanProb = scoreSiameseHead(cleanAuthorFeatures, probeVec, botReply1.rawText, weights).probability;
    const repeatProb = scoreSiameseHead(repeatAuthorFeatures, probeVec, botReply1.rawText, weights).probability;
    const botHandleProb = scoreSiameseHead(botHandleFeatures, probeVec, botReply1.rawText, weights).probability;
    assert(repeatProb > cleanProb, "Siamese model scores repeat_spammer higher via learned author feature slots");
    assert(botHandleProb > cleanProb, "Siamese model scores digit-suffixed bot handles higher via feature slot 777");

    const fb2 = engine.applyUserFeedback(threadId, botReply2.tweetId, "spam", weights);
    assert(fb2 !== null);
    weights = fb2.nextWeights;
    assert.equal(weights.authorStats.get("repeat_spammer" as AuthorId)?.spamCount, 2);
    assert.equal(
      fb2.retroactiveUpdates.length,
      1,
      "2nd spam mark retroactively dims other visible replies from the same account",
    );
    assert.equal(fb2.retroactiveUpdates[0]?.tweetId, "6003");
    assert.equal(fb2.retroactiveUpdates[0]?.reason.kind, "known_bot_author");

    const res2 = engine.ingestEmbeddedBatch(
      threadId,
      [{ candidate: botReply4, embedding: makeUnitVec(53), tokenCount: asTokenCount(1) }],
      weights,
    );
    assert.equal(res2.verdicts[0]?.action, "dim");
    assert.equal(
      res2.verdicts[0]?.reason.kind,
      "known_bot_author",
      "future replies from repeat_spammer (even 1-token replies) are automatically dimmed",
    );

    const diag = engine.getDiagnosticsSummary(weights);
    assert.equal(diag.totalScoredReplies, 4);
    assert.equal(diag.dimmedReplies, 4);
    assert.equal(diag.topFlaggedAuthors[0]?.authorId, "repeat_spammer");
    assert.equal(diag.topFlaggedAuthors[0]?.spamCount, 2);
  });
});
