import assert from "node:assert/strict";
import path from "node:path";
import { describe, it } from "node:test";
import { fileURLToPath } from "node:url";
import { build } from "esbuild";
import { parseHTML } from "linkedom";
import {
  XThreadDomAdapter,
  extractTweetTextWithTwemoji,
  formatDiagnosticsTooltip,
  formatFilterReasonBadge,
  parseFocalAuthorIdFromPathname,
} from "../src/content/dom-adapter.ts";
import {
  GRAPHQL_CONTENT_SOURCE,
  GRAPHQL_TAP_PROTOCOL_VERSION,
  GRAPHQL_TAP_SOURCE,
} from "../src/content/graphql-bridge.ts";
import {
  extractGraphqlTweetMetasFromJson,
  installGraphqlTap,
} from "../src/content/graphql-tap.ts";
import {
  BoundedGraphqlCache,
  handleRuntimeMessage,
  mountThreadObserver,
  runScrollPass,
} from "../src/content/index.ts";
import {
  type AuthorId,
  type FilterReason,
  type ThreadId,
  type TweetId,
  type VerdictRecord,
  asProbability,
  asTokenCount,
  asUnitCosine,
} from "../src/domain/types.ts";
import type { OffscreenRpcRequest, OffscreenRpcResponse } from "../src/shared/rpc.ts";

function buildMockXThreadDocument(dividerText = "Discover more"): Document {
  const { document } = parseHTML(`
    <!doctype html>
    <html>
      <head></head>
      <body>
        <main>
          <div data-testid="cellInnerDiv">
            <article data-testid="tweet">
              <a href="/op_author/status/1000"><time datetime="2026-11-08T12:00:00Z"></time></a>
              <div data-testid="tweetText">Original post text about local WebGPU inference.</div>
              <div data-testid="quoteTweet">
                <a href="/quoted_author/status/8888"></a>
                <div data-testid="tweetText">Quoted tweet text that must not overwrite OP text.</div>
              </div>
            </article>
          </div>
          <div data-testid="cellInnerDiv">
            <article data-testid="tweet">
              <div class="tweet-body">
                <svg data-testid="icon-verified"></svg>
                <a href="/bot_account/status/1001"><time datetime="2026-11-08T12:00:12Z"></time></a>
                <div data-testid="tweetText">Spot on! Local WebGPU inference really changes everything. What do you think?</div>
              </div>
            </article>
          </div>
          <div data-testid="cellInnerDiv">
            <article data-testid="tweet">
              <div class="tweet-body">
                <a href="/jp_user/status/1002"><time datetime="2026-11-08T12:01:00Z"></time></a>
                <div data-testid="tweetText">草www</div>
              </div>
            </article>
          </div>
          <div data-testid="cellInnerDiv">
            <h2><span>${dividerText}</span></h2>
          </div>
          <div data-testid="cellInnerDiv">
            <article data-testid="tweet">
              <a href="/random_user/status/9999"><time datetime="2026-11-08T11:00:00Z"></time></a>
              <div data-testid="tweetText">Unrelated algorithmic recommendation below the cutoff.</div>
            </article>
          </div>
        </main>
      </body>
    </html>
  `);
  return document as unknown as Document;
}

describe("XThreadDomAdapter extraction, cutoff, and zero-jitter dimming", () => {
  it("extracts thread tweets, ignores nested quote-tweet links, and stops at Discover more in EN/JA", () => {
    const adapter = new XThreadDomAdapter();

    for (const heading of ["Discover more", "さらに表示", "发现更多", "探索更多"]) {
      const doc = buildMockXThreadDocument(heading);
      const snapshot = adapter.extractThreadSnapshot(doc, "/op_author/status/1000");
      assert(snapshot !== null);
      assert.equal(snapshot.threadId, "1000");
      assert.equal(snapshot.candidates.length, 3, `stops before 9999 on heading ${heading}`);

      const op = snapshot.candidates[0]!;
      assert.equal(op.tweetId, "1000");
      assert.equal(op.authorId, "op_author");
      assert.equal(op.isOp, true);
      assert.equal(op.rawText, "Original post text about local WebGPU inference.");
      assert.equal(op.hasQuote, true);

      const botReply = snapshot.candidates[1]!;
      assert.equal(botReply.tweetId, "1001");
      assert.equal(botReply.isOp, false);
      assert.equal(botReply.domSignals.hasVerifiedBadge, true);
      assert.equal(botReply.domSignals.endsWithBaitQuestion, true);
    }
  });

  it("dims flagged replies in place, respects Show/Hide toggles, and routes 1-click feedback", async () => {
    const doc = buildMockXThreadDocument();
    const adapter = new XThreadDomAdapter();
    const requests: OffscreenRpcRequest[] = [];

    let seenFocalAuthorId: string | undefined;
    const sendRpc = async (req: OffscreenRpcRequest): Promise<OffscreenRpcResponse> => {
      requests.push(req);
      if (req.kind === "evaluate_batch") {
        seenFocalAuthorId = (req as { focalAuthorId?: string }).focalAuthorId;
        const verdicts: VerdictRecord[] = [
          {
            tweetId: "1001" as TweetId,
            action: "dim",
            score: 0.91,
            tokenCount: asTokenCount(14),
            reason: {
              kind: "op_paraphrase",
              action: "dim",
              similarityToOp: asUnitCosine(0.91),
            },
          },
          {
            tweetId: "1002" as TweetId,
            action: "keep",
            score: 0.05,
            tokenCount: asTokenCount(2),
            reason: {
              kind: "short_human_exempt",
              action: "keep",
              tokenCount: asTokenCount(2),
            },
          },
        ];
        return {
          kind: "evaluate_batch_ok",
          result: {
            threadId: req.threadId,
            anchorKind: "text_anchored",
            verdicts,
            retroactiveUpdates: [],
          },
        };
      }

      if (req.kind === "record_feedback") {
        return {
          kind: "record_feedback_ok",
          updatedVerdict: {
            tweetId: req.tweetId,
            action: req.label === "spam" ? "dim" : "keep",
            score: req.label === "spam" ? 1 : 0,
            tokenCount: asTokenCount(14),
            reason: {
              kind: "user_override",
              action: req.label === "spam" ? "dim" : "keep",
              label: req.label,
            },
          },
          updateCount: 1,
        };
      }

      return { kind: "health_ok", gpuStatus: "ready", deviceEpoch: 1 };
    };

    await runScrollPass(doc, "/op_author/status/1000", adapter, sendRpc);
    assert.equal(seenFocalAuthorId, "op_author");

    const botArticle = doc.querySelector('article[data-xrf-tweet-id="1001"]');
    const jpArticle = doc.querySelector('article[data-xrf-tweet-id="1002"]');
    assert(botArticle !== null && jpArticle !== null);

    assert.equal(botArticle.getAttribute("data-xrf-state"), "dimmed");
    assert.equal(botArticle.getAttribute("data-xrf-reason"), "op_paraphrase");
    assert.equal(
      botArticle.querySelector(".xrf-badge")?.textContent,
      "@bot_account · AI paraphrase (91%)",
      "collapsed 1-line bar includes the reply author handle alongside the filter reason",
    );
    assert.equal(jpArticle.getAttribute("data-xrf-state"), "clean");
    assert.equal(
      jpArticle.querySelector(".xrf-score-hint"),
      null,
      "clean replies omit Bot NN% clutter",
    );
    assert.equal(jpArticle.querySelector(".xrf-feedback-spam")?.textContent, "Mark bot");

    const toggleBtn = botArticle.querySelector(".xrf-toggle-btn") as HTMLButtonElement | null;
    assert(toggleBtn !== null);
    toggleBtn.click();
    assert.equal(botArticle.getAttribute("data-xrf-state"), "revealed");
    assert.equal(botArticle.getAttribute("data-xrf-user-revealed"), "true");

    await runScrollPass(doc, "/op_author/status/1000", adapter, sendRpc);
    assert.equal(seenFocalAuthorId, "op_author");
    assert.equal(
      botArticle.getAttribute("data-xrf-state"),
      "revealed",
      "user reveal state persists across idempotent scroll passes",
    );

    const notSpamBtn = botArticle.querySelector(".xrf-feedback-human") as HTMLButtonElement | null;
    assert(notSpamBtn !== null);
    notSpamBtn.click();

    await new Promise((r) => setTimeout(r, 5));
    assert.equal(requests.some((r) => r.kind === "record_feedback" && r.label === "human"), true);
    assert.equal(botArticle.getAttribute("data-xrf-state"), "clean");
    assert.equal(botArticle.getAttribute("data-xrf-reason"), "user_override");

    const rpcCountBeforeIdlePass = requests.length;
    await runScrollPass(doc, "/op_author/status/1000", adapter, sendRpc);
    assert.equal(seenFocalAuthorId, "op_author");
    assert.equal(
      requests.length,
      rpcCountBeforeIdlePass,
      "subsequent scroll pass with no new unscored tweets must not fire RPC or flicker status",
    );
    assert.equal(
      botArticle.children[botArticle.children.length - 1]?.classList.contains("xrf-control-bar"),
      false,
      "control bar must mount inside the tweet content column, not as a direct flex-row child of article",
    );
  });

  it("preserves Twemoji <img alt> icons & <br> newlines and isolates quote-tweet media/badges", () => {
    const { document } = parseHTML(`
      <!doctype html>
      <html><body><main>
        <div data-testid="cellInnerDiv">
          <article data-testid="tweet">
            <a href="/op_user/status/5000"></a>
            <div data-testid="tweetText">Launching v2 today!</div>
          </article>
        </div>
        <div data-testid="cellInnerDiv">
          <article data-testid="tweet">
            <a href="/emoji_bot/status/5001"></a>
            <svg data-testid="icon-verified"></svg>
            <div data-testid="tweetText"><img alt="🔥" src="https://abs-0.twimg.com/emoji/v2/svg/1f525.svg" /><img alt="🚀" src="https://abs-0.twimg.com/emoji/v2/svg/1f680.svg" /><img alt="💯" src="https://abs-0.twimg.com/emoji/v2/svg/1f4af.svg" /></div>
          </article>
        </div>
        <div data-testid="cellInnerDiv">
          <article data-testid="tweet">
            <a href="/human_quoter/status/5002"></a>
            <div data-testid="tweetText">Check this<br>out <img alt="👍" src="x.svg" /></div>
            <div data-testid="quoteTweet">
              <svg data-testid="icon-verified"></svg>
              <div data-testid="tweetPhoto"></div>
              <a href="https://spam.example.com" target="_blank">link</a>
              <div data-testid="tweetText">Quoted verified photo tweet</div>
            </div>
          </article>
        </div>
        <div data-testid="cellInnerDiv">
          <article data-testid="tweet">
            <div class="tweet-col">
              <a href="/gif_reply/status/5003"></a>
              <div data-testid="videoPlayer"></div>
              <div role="group"></div>
            </div>
          </article>
        </div>
      </main></body></html>
    `);
    const doc = document as unknown as Document;
    const adapter = new XThreadDomAdapter();

    const textNode = doc.querySelector('article:nth-of-type(1) [data-testid="tweetText"]');
    assert.equal(extractTweetTextWithTwemoji(textNode), "Launching v2 today!");

    const snap = adapter.extractThreadSnapshot(doc, "/op_user/status/5000");
    assert(snap !== null);
    assert.equal(snap.candidates.length, 4);

    const emojiBot = snap.candidates[1]!;
    assert.equal(emojiBot.rawText, "🔥🚀💯");
    assert.equal(emojiBot.domSignals.hasVerifiedBadge, true);
    assert.equal(emojiBot.domSignals.emojiToCharRatio, 1);

    const quoter = snap.candidates[2]!;
    assert.equal(quoter.rawText, "Check this\nout 👍");
    assert.equal(quoter.hasQuote, true);
    assert.equal(quoter.hasPrimaryMedia, false, "quoted photo must not set hasPrimaryMedia");
    assert.equal(
      quoter.domSignals.hasVerifiedBadge,
      false,
      "quoted verified icon must not bleed onto replying human",
    );
    assert.equal(
      quoter.domSignals.hasMediaOrExternalLink,
      false,
      "quoted external link/photo must not bleed onto replying human",
    );

    const gifReply = snap.candidates[3]!;
    assert.equal(gifReply.rawText, "");
    assert.equal(gifReply.hasPrimaryMedia, true);
    assert.equal(gifReply.hasQuote, false);
  });

  it("tags context ancestors before the focal tweet and avoids false ancestor tagging when OP scrolled off", () => {
    const adapter = new XThreadDomAdapter();
    const { document: docWithAncestors } = parseHTML(`
      <!doctype html><html><body><main>
        <div data-testid="cellInnerDiv"><article data-testid="tweet"><a href="/root/status/100"></a><div data-testid="tweetText">Root post</div></article></div>
        <div data-testid="cellInnerDiv"><article data-testid="tweet"><a href="/mid/status/150"></a><div data-testid="tweetText">Parent context</div></article></div>
        <div data-testid="cellInnerDiv"><article data-testid="tweet"><a href="/focal/status/200"></a><div data-testid="tweetText">Focal tweet</div></article></div>
        <div data-testid="cellInnerDiv"><article data-testid="tweet"><a href="/rep/status/201"></a><div data-testid="tweetText">Reply below focal</div></article></div>
      </main></body></html>
    `);
    const snap1 = adapter.extractThreadSnapshot(
      docWithAncestors as unknown as Document,
      "/focal/status/200",
    );
    assert(snap1 !== null);
    assert.equal(snap1.candidates[0]!.isContextAncestor, true);
    assert.equal(snap1.candidates[1]!.isContextAncestor, true);
    assert.equal(snap1.candidates[2]!.isOp, true);
    assert.equal(snap1.candidates[2]!.isContextAncestor, false);
    assert.equal(snap1.candidates[3]!.isContextAncestor, false);

    // Virtual-scrolled thread where focal tweet 200 unmounted from DOM
    const { document: docScrolled } = parseHTML(`
      <!doctype html><html><body><main>
        <div data-testid="cellInnerDiv"><article data-testid="tweet"><a href="/rep1/status/201"></a><div data-testid="tweetText">Reply 1</div></article></div>
        <div data-testid="cellInnerDiv"><article data-testid="tweet"><a href="/rep2/status/202"></a><div data-testid="tweetText">Reply 2</div></article></div>
      </main></body></html>
    `);
    const snap2 = adapter.extractThreadSnapshot(
      docScrolled as unknown as Document,
      "/focal/status/200",
    );
    assert(snap2 !== null);
    assert.equal(snap2.candidates[0]!.isContextAncestor, false);
    assert.equal(snap2.candidates[1]!.isContextAncestor, false);
  });

  it("clears stale attributes when virtual scroller recycles an article for a new tweetId", () => {
    const adapter = new XThreadDomAdapter();
    const { document } = parseHTML(`
      <!doctype html><html><body><main>
        <div data-testid="cellInnerDiv">
          <article data-testid="tweet" data-xrf-tweet-id="1001" data-xrf-state="dimmed" data-xrf-reason="keyword_spam" data-xrf-user-revealed="true">
            <div class="col">
              <a href="/fresh_human/status/3003"></a>
              <div data-testid="tweetText">Brand new recycled tweet</div>
              <div class="xrf-control-bar">old bar</div>
            </div>
          </article>
        </div>
      </main></body></html>
    `);
    const doc = document as unknown as Document;
    const snap = adapter.extractThreadSnapshot(doc, "/op/status/1000");
    assert(snap !== null);
    assert.equal(snap.hasUnscoredReplies, true);
    const article = doc.querySelector("article")!;
    assert.equal(article.getAttribute("data-xrf-tweet-id"), "3003");
    assert.equal(article.getAttribute("data-xrf-state"), null);
    assert.equal(article.getAttribute("data-xrf-user-revealed"), null);
    assert.equal(article.querySelector(".xrf-control-bar"), null);
  });

  it("renders 3-tier states (dimmed, suspect readable, clean, revealed), single-line CSS, and all 14 FilterReason badges", () => {
    const adapter = new XThreadDomAdapter();
    const { document } = parseHTML(`
      <!doctype html><html><head></head><body><main>
        <div data-testid="cellInnerDiv">
          <article data-testid="tweet">
            <div class="col">
              <a href="/op/status/7000"></a>
              <div data-testid="tweetText">OP text</div>
            </div>
          </article>
        </div>
        <div data-testid="cellInnerDiv">
          <article data-testid="tweet">
            <div class="col">
              <a href="/borderline_guy/status/7001"></a>
              <div data-testid="tweetText">Interesting point, worth thinking about!</div>
              <div role="group"></div>
            </div>
          </article>
        </div>
        <div data-testid="cellInnerDiv">
          <article data-testid="tweet">
            <div class="col">
              <a href="/media_bot/status/7002"></a>
              <div data-testid="tweetPhoto"></div>
            </div>
          </article>
        </div>
      </main></body></html>
    `);
    const doc = document as unknown as Document;
    const snap = adapter.extractThreadSnapshot(doc, "/op/status/7000");
    assert(snap !== null);

    const suspectVerdict: VerdictRecord = {
      tweetId: "7001" as TweetId,
      authorId: "borderline_guy" as AuthorId,
      action: "suspect",
      tier: "suspect",
      score: 0.56,
      tokenCount: asTokenCount(6),
      reason: {
        kind: "model_suspect",
        action: "suspect",
        probability: asProbability(0.56),
        dominantLanguage: "en",
      },
      explanation: {
        schemaVersion: 2,
        tier: "suspect",
        ruleId: "model_suspect",
        probability: 0.56,
        logit: 0.24,
        thresholds: { dim: 0.65, suspect: 0.42 },
        preset: "balanced",
        top: [{ id: "s_syco_open", family: "style", value: 1, contribution: 0.9, evidence: "Interesting" }],
        counter: [{ id: "m_age", family: "meta", value: 0.8, contribution: -0.35 }],
        families: ["style"],
        shields: [],
        caps: ["C2_high_likes"],
        metaSource: "graphql",
        lang: { reply: "en", op: "en" },
      },
    };
    const mediaDimVerdict: VerdictRecord = {
      tweetId: "7002" as TweetId,
      authorId: "media_bot" as AuthorId,
      action: "dim",
      tier: "dim",
      score: 0.9,
      tokenCount: asTokenCount(0),
      reason: { kind: "media_only_blue", action: "dim", confirmedUnfollowed: true },
    };

    const feedbacks: Array<[string, string]> = [];
    adapter.applyVerdictsInPlace(doc, [suspectVerdict, mediaDimVerdict], (id, label) => {
      feedbacks.push([id, label]);
    });

    const suspectArticle = doc.querySelector('article[data-xrf-tweet-id="7001"]')!;
    assert.equal(suspectArticle.getAttribute("data-xrf-state"), "suspect");
    const suspectBar = suspectArticle.querySelector(".xrf-control-bar")!;
    assert.equal(suspectBar.getAttribute("data-xrf-bar-action"), "suspect");
    assert.equal(suspectBar.querySelector(".xrf-badge")?.textContent, "Borderline (56%)");
    assert.equal(suspectBar.querySelector(".xrf-feedback-spam")?.textContent, "Mark bot");
    assert.equal(suspectBar.querySelector(".xrf-feedback-human")?.textContent, "Not spam");
    const tooltip = suspectBar.getAttribute("title") ?? "";
    assert.equal(tooltip.includes("rule=model_suspect"), true);
    assert.equal(tooltip.includes("s_syco_open+0.90(Interesting)"), true);
    assert.equal(tooltip.includes("caps=C2_high_likes"), true);

    (suspectBar.querySelector(".xrf-feedback-spam") as HTMLButtonElement).click();
    assert.deepEqual(feedbacks, [["7001", "spam"]]);

    // Verify media-only bar did NOT mount directly as a child of <article>
    const mediaArticle = doc.querySelector('article[data-xrf-tweet-id="7002"]')!;
    assert.equal(mediaArticle.getAttribute("data-xrf-state"), "dimmed");
    const mediaBar = mediaArticle.querySelector(".xrf-control-bar")!;
    assert.notEqual(mediaBar.parentElement?.tagName, "ARTICLE");

    // Verify stylesheet keeps suspect text readable and bar strictly single-line
    const styleText = doc.getElementById("xrf-injected-styles")?.textContent ?? "";
    assert.equal(styleText.includes('article[data-xrf-state="suspect"]'), true);
    assert.equal(styleText.includes("white-space: nowrap !important"), true);
    assert.equal(styleText.includes("text-overflow: ellipsis"), true);
    assert.equal(styleText.includes("flex-shrink: 0"), true);

    // Verify exhaustive FilterReason formatting across all 14 kinds
    const allReasons: FilterReason[] = [
      { kind: "known_bot_author", action: "dim", authorId: "b" as AuthorId, spamCount: 3 },
      { kind: "keyword_spam", action: "dim", matchedPattern: "en_crypto_scam" },
      { kind: "bio_scam", action: "dim", matchedPattern: "bio_spam_strong" },
      { kind: "bot_summon", action: "dim", handle: "@grok" },
      { kind: "bot_answer", action: "dim", authorId: "grok" as AuthorId },
      { kind: "blue_short_praise", action: "dim", confirmedUnfollowed: true },
      { kind: "media_only_blue", action: "suspect", confirmedUnfollowed: false },
      { kind: "op_paraphrase", action: "dim", similarityToOp: asUnitCosine(0.89) },
      { kind: "thread_clone", action: "dim", similarityToPeer: asUnitCosine(0.95), canonicalTweetId: "1" as TweetId, clusterSize: 4 },
      { kind: "siamese_bot", action: "dim", probability: asProbability(0.88), dominantLanguage: "en", opFallback: false },
      { kind: "model_suspect", action: "suspect", probability: asProbability(0.55), dominantLanguage: "ja" },
      { kind: "user_override", action: "keep", label: "human" },
      { kind: "short_human_exempt", action: "keep", tokenCount: asTokenCount(2) },
      { kind: "clean", action: "keep" },
    ];
    for (const r of allReasons) {
      assert.equal(formatFilterReasonBadge(r).length > 0, true);
    }
    assert.equal(formatDiagnosticsTooltip({ ...mediaDimVerdict, diagnostics: undefined }).includes("media_only_blue"), true);
  });
});

describe("GraphQL passive recursive walker and MAIN/isolated window bridge", () => {
  it("unwraps TweetWithVisibilityResults, handles migrated user schemas, isolates nested quotes, and preserves null following", () => {
    const cyclicQuote: Record<string, unknown> = {
      __typename: "Tweet",
      rest_id: "9009",
      core: {
        user_results: {
          result: {
            __typename: "User",
            rest_id: "777",
            is_blue_verified: false,
            core: {
              screen_name: "QuotedOrg",
              name: "Quoted Org",
              created_at: "Wed Oct 10 20:19:24 +0000 2018",
            },
            verification: { verified_type: "Business" },
            relationship_perspectives: { following: true, followed_by: false },
            legacy: {
              description: "Official business org",
              followers_count: 50000,
              friends_count: 10,
              statuses_count: 400,
              default_profile_image: false,
            },
          },
        },
      },
      legacy: {
        id_str: "9009",
        created_at: "Sun Nov 22 08:00:00 +0000 2026",
        conversation_id_str: "9009",
        favorite_count: 420,
        reply_count: 15,
        retweet_count: 30,
        extended_entities: { media: [{ type: "photo" }] },
      },
    };
    cyclicQuote.selfCycle = cyclicQuote;

    const payload = {
      data: {
        threaded_conversation_with_injections_v2: {
          instructions: [
            {
              type: "TimelineAddEntries",
              entries: [
                {
                  content: {
                    itemContent: {
                      __typename: "TimelineTweet",
                      tweet_results: {
                        result: {
                          __typename: "TweetWithVisibilityResults",
                          tweet: {
                            rest_id: "8001",
                            core: {
                              user_results: {
                                result: {
                                  __typename: "User",
                                  rest_id: "111",
                                  is_blue_verified: true,
                                  legacy: {
                                    screen_name: "BlueReplyGuy",
                                    name: "Reply 🚀🔥💎",
                                    created_at: "Mon Jan 01 00:00:00 +0000 2024",
                                    description: "DM for promo | t.me/scam",
                                    verified_type: "Blue",
                                    followers_count: 120,
                                    friends_count: 4500,
                                    statuses_count: 9000,
                                    default_profile_image: true,
                                    // following omitted -> must remain null, NOT false!
                                  },
                                },
                              },
                            },
                            legacy: {
                              id_str: "8001",
                              created_at: "Sun Nov 22 08:05:00 +0000 2026",
                              in_reply_to_status_id_str: "8000",
                              conversation_id_str: "8000",
                              favorite_count: 1,
                              reply_count: 0,
                              retweet_count: 0,
                              extended_entities: { media: [{ type: "animated_gif" }] },
                            },
                            quoted_status_result: { result: cyclicQuote },
                          },
                        },
                      },
                    },
                  },
                },
                {
                  content: {
                    itemContent: {
                      __typename: "TimelineTweet",
                      promotedMetadata: { impressionId: "ad-1" },
                      tweet_results: {
                        result: {
                          __typename: "Tweet",
                          rest_id: "8999",
                          core: { user_results: { result: { legacy: { screen_name: "ad_bot" } } } },
                          legacy: { id_str: "8999" },
                        },
                      },
                    },
                  },
                },
                {
                  content: {
                    itemContent: {
                      tweet_results: { result: { __typename: "TweetTombstone" } },
                    },
                  },
                },
              ],
            },
          ],
        },
      },
    };

    const items = extractGraphqlTweetMetasFromJson(payload);
    assert.equal(items.length, 2, "extracts outer tweet + quoted tweet while skipping ad/tombstone");

    const outer = items.find((m) => m.tweetId === "8001")!;
    const quoted = items.find((m) => m.tweetId === "9009")!;
    assert(outer !== undefined && quoted !== undefined);

    assert.equal(outer.screenName, "bluereplyguy");
    assert.equal(outer.verifiedType, "blue");
    assert.equal(outer.isBlueVerified, true);
    assert.equal(outer.following, null, "omitted following must stay null, never false");
    assert.equal(outer.followedBy, null);
    assert.equal(outer.defaultProfileImage, true);
    assert.equal(outer.inReplyToStatusId, "8000");
    assert.equal(outer.conversationId, "8000");
    assert.deepEqual(outer.mediaTypes, ["animated_gif"]);
    assert.equal(outer.followersCount, 120);
    assert.equal(outer.friendsCount, 4500);
    assert.equal(outer.statusesCount, 9000);
    assert.equal(outer.favoriteCount, 1);

    assert.equal(quoted.screenName, "quotedorg");
    assert.equal(quoted.verifiedType, "business");
    assert.equal(quoted.following, true);
    assert.equal(quoted.followedBy, false);
    assert.deepEqual(quoted.mediaTypes, ["photo"]);
  });

  it("installs passive fetch hook + tap_ready replay buffer on mock Window without interfering with Response.json()", async () => {
    const messageListeners: Array<(ev: MessageEvent) => void> = [];
    const posted: Array<{ data: unknown; origin: string }> = [];

    const graphqlBody = {
      data: {
        tweetResult: {
          result: {
            __typename: "Tweet",
            rest_id: "6001",
            core: {
              user_results: {
                result: {
                  rest_id: "42",
                  is_blue_verified: true,
                  legacy: {
                    screen_name: "tap_user",
                    name: "Tap User",
                    following: false,
                    followers_count: 100,
                    friends_count: 200,
                    statuses_count: 50,
                  },
                },
              },
            },
            legacy: { id_str: "6001", favorite_count: 5, reply_count: 1, retweet_count: 0 },
          },
        },
      },
    };

    const mockFetch = async (): Promise<Response> =>
      new Response(JSON.stringify(graphqlBody), {
        status: 200,
        headers: { "content-type": "application/json" },
      });

    class MockXhr {
      public status = 200;
      public responseType: XMLHttpRequestResponseType = "";
      public responseText = "";
      public response: unknown = null;
      public shouldThrowOnSend = false;
      public readonly loadListeners: Array<() => void> = [];
      public open(_method: string, _url: string | URL): void {}
      public send(): void {
        if (this.shouldThrowOnSend) {
          throw new Error("InvalidStateError");
        }
        for (const cb of this.loadListeners.splice(0)) {
          cb.call(this);
        }
      }
      public addEventListener(type: string, cb: () => void): void {
        if (type === "load") this.loadListeners.push(cb);
      }
      public removeEventListener(type: string, cb: () => void): void {
        if (type === "load") {
          const idx = this.loadListeners.indexOf(cb);
          if (idx >= 0) this.loadListeners.splice(idx, 1);
        }
      }
    }

    const mockWin = {
      location: { origin: "https://x.com" },
      fetch: mockFetch,
      XMLHttpRequest: MockXhr,
      postMessage(data: unknown, origin: string) {
        posted.push({ data, origin });
      },
      addEventListener(type: string, cb: (ev: MessageEvent) => void) {
        if (type === "message") messageListeners.push(cb);
      },
    } as unknown as Window;

    installGraphqlTap(mockWin);
    installGraphqlTap(mockWin); // idempotent second install

    const pageRes = await mockWin.fetch("https://x.com/i/api/graphql/xyz/TweetDetail?variables=%7B%7D");
    const pageJson = (await pageRes.json()) as { data: { tweetResult: { result: { rest_id: string } } } };
    assert.equal(pageJson.data.tweetResult.result.rest_id, "6001");

    await new Promise((r) => setTimeout(r, 15));
    assert.equal(posted.length, 1);
    assert.equal(posted[0]!.origin, "https://x.com");

    // XHR passive hook test
    const xhr = new MockXhr();
    xhr.responseType = "text";
    xhr.responseText = JSON.stringify({
      data: {
        tweetResult: {
          result: {
            __typename: "Tweet",
            rest_id: "6002",
            core: { user_results: { result: { __typename: "User", legacy: { screen_name: "xhr_user" } } } },
            legacy: { id_str: "6002", favorite_count: 12 },
          },
        },
      },
    });
    xhr.open("GET", "https://x.com/i/api/graphql/abc/TweetResultByRestId");
    xhr.send();
    assert.equal(posted.length, 2);

    // Simulate tap_ready replay handshake (replays both 6001 and 6002)
    for (const cb of messageListeners) {
      cb({
        source: mockWin,
        origin: "https://x.com",
        data: { source: GRAPHQL_CONTENT_SOURCE, type: "tap_ready" },
      } as unknown as MessageEvent);
    }
    assert.equal(posted.length, 3);
    assert.equal(
      (posted[2]!.data as { items: Array<{ tweetId: string }> }).items.length,
      2,
    );
  });

  it("validates origins, prunes BoundedGraphqlCache LRU, re-enriches late GraphQL once, and handles runtime Stage-2 + settings updates", async () => {
    const cache = new BoundedGraphqlCache(2);
    cache.ingestMessageData({
      source: GRAPHQL_TAP_SOURCE,
      v: GRAPHQL_TAP_PROTOCOL_VERSION,
      items: [
        { tweetId: "1", verifiedType: "blue", following: false },
        { tweetId: "2", verifiedType: "none", following: true },
        { tweetId: "3", verifiedType: "business", following: null },
      ],
    });
    assert.equal(cache.size, 2);
    assert.equal(cache.get("1" as TweetId), undefined, "oldest entry evicted from LRU");
    assert.equal(cache.get("3" as TweetId)?.verifiedType, "business");

    const doc = buildMockXThreadDocument();
    const adapter = new XThreadDomAdapter();
    const messageListeners: Array<(ev: MessageEvent) => void> = [];
    const runtimeListeners: Array<(msg: unknown) => void> = [];
    const postedToWin: unknown[] = [];
    const evalCalls: Array<{ tweetId: string; hasGql: boolean }> = [];

    const mockWin = {
      location: { origin: "https://x.com" },
      postMessage(data: unknown) {
        postedToWin.push(data);
      },
      addEventListener(type: string, cb: (ev: MessageEvent) => void) {
        if (type === "message") messageListeners.push(cb);
      },
      removeEventListener() {},
    } as unknown as Window;

    let currentScoreAction: "suspect" | "keep" = "suspect";
    const sendRpc = async (req: OffscreenRpcRequest): Promise<OffscreenRpcResponse> => {
      if (req.kind === "evaluate_batch") {
        for (const c of req.candidates) {
          evalCalls.push({ tweetId: c.tweetId, hasGql: c.graphqlMeta != null });
        }
        return {
          kind: "evaluate_batch_ok",
          result: {
            threadId: req.threadId,
            anchorKind: "text_anchored",
            verdicts: [
              {
                tweetId: "1001" as TweetId,
                authorId: "bot_account" as AuthorId,
                action: currentScoreAction,
                tier: currentScoreAction,
                score: 0.55,
                tokenCount: asTokenCount(14),
                reason:
                  currentScoreAction === "suspect"
                    ? {
                        kind: "model_suspect",
                        action: "suspect",
                        probability: asProbability(0.55),
                        dominantLanguage: "en",
                      }
                    : { kind: "clean", action: "keep" },
              },
            ],
            retroactiveUpdates: [],
          },
        };
      }
      return { kind: "health_ok", gpuStatus: "ready", deviceEpoch: 1 };
    };

    const handle = mountThreadObserver(doc, sendRpc, () => "/op_author/status/1000", 10, {
      win: mockWin,
      graphqlCache: new BoundedGraphqlCache(10),
      runtimeOnMessage: {
        addListener: (cb) => runtimeListeners.push(cb),
        removeListener: () => {},
      },
    });

    assert.equal(postedToWin.length, 1, "sends tap_ready handshake on boot");
    await handle.flushNow();
    assert.equal(evalCalls.some((e) => e.tweetId === "1001" && !e.hasGql), true);

    // Evil origin must be ignored
    for (const cb of messageListeners) {
      cb({
        source: mockWin,
        origin: "https://evil.example",
        data: {
          source: GRAPHQL_TAP_SOURCE,
          v: GRAPHQL_TAP_PROTOCOL_VERSION,
          items: [{ tweetId: "1001", verifiedType: "blue", following: false }],
        },
      } as unknown as MessageEvent);
    }

    // Valid x.com origin enriches 1001 and propagates OP favoriteCount
    const countBeforeValidGql = evalCalls.length;
    for (const cb of messageListeners) {
      cb({
        source: mockWin,
        origin: "https://x.com",
        data: {
          source: GRAPHQL_TAP_SOURCE,
          v: GRAPHQL_TAP_PROTOCOL_VERSION,
          items: [
            { tweetId: "1000", verifiedType: "none", favoriteCount: 250 },
            { tweetId: "1001", verifiedType: "blue", following: false },
          ],
        },
      } as unknown as MessageEvent);
    }
    await new Promise((r) => setTimeout(r, 30));
    assert.equal(
      evalCalls.slice(countBeforeValidGql).some((e) => e.tweetId === "1001" && e.hasGql),
      true,
      "late GraphQL metadata triggers single enrichment re-evaluation",
    );
    const snapEnriched = adapter.extractThreadSnapshot(doc, "/op_author/status/1000", new Map([
      ["1000" as TweetId, { tweetId: "1000" as TweetId, displayName: null, authorCreatedAtMs: null, tweetCreatedAtMs: null, inReplyToStatusId: null, conversationId: null, description: null, isBlueVerified: null, verifiedType: "none", following: null, followedBy: null, followersCount: null, friendsCount: null, statusesCount: null, defaultProfileImage: null, favoriteCount: 250, replyCount: null, retweetCount: null, mediaTypes: [] }],
    ]));
    assert.equal(snapEnriched?.candidates[1]?.opFavoriteCount, 250);

    // Repeating same GraphQL message must NOT trigger another RPC loop
    const countAfterEnrich = evalCalls.length;
    for (const cb of messageListeners) {
      cb({
        source: mockWin,
        origin: "https://x.com",
        data: {
          source: GRAPHQL_TAP_SOURCE,
          v: GRAPHQL_TAP_PROTOCOL_VERSION,
          items: [{ tweetId: "1001", verifiedType: "blue", following: false }],
        },
      } as unknown as MessageEvent);
    }
    await new Promise((r) => setTimeout(r, 30));
    assert.equal(evalCalls.length, countAfterEnrich, "duplicate GraphQL payload must not loop");

    // Stage-2 verdict update pushes DOM transition from suspect -> dimmed
    const handledStage2 = handleRuntimeMessage(
      {
        kind: "stage2_verdict_update",
        threadId: "1000" as ThreadId,
        tweetId: "1001" as TweetId,
        verdict: {
          tweetId: "1001" as TweetId,
          authorId: "bot_account" as AuthorId,
          action: "dim",
          tier: "dim",
          score: 0.94,
          tokenCount: asTokenCount(14),
          reason: {
            kind: "siamese_bot",
            action: "dim",
            probability: asProbability(0.94),
            dominantLanguage: "en",
            opFallback: false,
          },
          explanation: {
            schemaVersion: 2,
            tier: "dim",
            ruleId: "stage2_nano",
            probability: 0.94,
            logit: 2.75,
            thresholds: { dim: 0.65, suspect: 0.42 },
            preset: "balanced",
            top: [],
            counter: [],
            families: ["style"],
            shields: [],
            caps: [],
            metaSource: "graphql",
            lang: { reply: "en", op: "en" },
            stage2: { provider: "chrome_builtin", verdict: "bot", confidence: 0.94, ms: 120 },
          },
        },
      },
      doc,
      () => "/op_author/status/1000",
      adapter,
      sendRpc,
    );
    assert.equal(handledStage2, true);
    const updatedArticle = doc.querySelector('article[data-xrf-tweet-id="1001"]')!;
    assert.equal(updatedArticle.getAttribute("data-xrf-state"), "dimmed");
    assert.equal(
      updatedArticle.querySelector(".xrf-control-bar")?.getAttribute("title")?.includes("stage2=bot(94%)"),
      true,
    );

    // Mismatched threadId Stage-2 update is ignored
    assert.equal(
      handleRuntimeMessage(
        {
          kind: "stage2_verdict_update",
          threadId: "9999" as ThreadId,
          tweetId: "1001" as TweetId,
          verdict: {
            tweetId: "1001" as TweetId,
            action: "keep",
            score: 0.1,
            tokenCount: asTokenCount(14),
            reason: { kind: "clean", action: "keep" },
          },
        },
        doc,
        () => "/op_author/status/1000",
        adapter,
        sendRpc,
      ),
      false,
    );

    // Settings change broadcast forces re-evaluation cleanly
    currentScoreAction = "keep";
    for (const cb of runtimeListeners) {
      cb({ kind: "xrf_settings_changed" });
    }
    await new Promise((r) => setTimeout(r, 30));
    assert.equal(updatedArticle.getAttribute("data-xrf-state"), "clean");

    handle.dispose();
  });

  it("cleans stale below-cutoff and SPA-navigated articles, blocks forged screenName, updates dynamic badges, and uses inset box-shadow", () => {
    const adapter = new XThreadDomAdapter();
    const { document } = parseHTML(`
      <!doctype html><html><head></head><body><main>
        <div data-testid="cellInnerDiv"><article data-testid="tweet"><a href="/op_user/status/100"></a><div data-testid="tweetText">OP</div></article></div>
        <div data-testid="cellInnerDiv"><article data-testid="tweet"><div class="col"><a href="/bot_a/status/101"></a><div data-testid="tweetText">Same reply</div></div></article></div>
        <div data-testid="cellInnerDiv"><article data-testid="tweet"><div class="col"><a href="/bot_a/status/101"></a><div data-testid="tweetText">Same reply duplicate</div></div></article></div>
        <div data-testid="cellInnerDiv"><h2><span>Discover more</span></h2></div>
        <div data-testid="cellInnerDiv">
          <article data-testid="tweet" data-xrf-tweet-id="999" data-xrf-state="dimmed" data-xrf-reason="keyword_spam">
            <div class="col"><a href="/below/status/999"></a><div data-testid="tweetText">Below cutoff recycled</div><div class="xrf-control-bar">old</div></div>
          </article>
        </div>
      </main></body></html>
    `);
    const doc = document as unknown as Document;

    // Cross-check forged screenName vs DOM authorId
    const forgedCache = new Map([
      [
        "101" as TweetId,
        {
          tweetId: "101" as TweetId,
          screenName: "different_followed_account" as AuthorId,
          displayName: null,
          authorCreatedAtMs: null,
          tweetCreatedAtMs: null,
          inReplyToStatusId: null,
          conversationId: null,
          description: null,
          isBlueVerified: null,
          verifiedType: "none" as const,
          following: true,
          followedBy: null,
          followersCount: null,
          friendsCount: null,
          statusesCount: null,
          defaultProfileImage: null,
          favoriteCount: null,
          replyCount: null,
          retweetCount: null,
          mediaTypes: [],
        },
      ],
    ]);

    const snap = adapter.extractThreadSnapshot(doc, "/op_user/status/100", forgedCache);
    assert(snap !== null);
    assert.equal(snap.focalAuthorId, "op_user");
    assert.equal(parseFocalAuthorIdFromPathname("/i/status/100"), null);
    assert.equal(parseFocalAuthorIdFromPathname("/home"), null);
    assert.equal(snap.candidates[1]!.graphqlMeta, undefined, "mismatched screenName must drop forged graphqlMeta");

    // Below-cutoff stale state & bar must be swept clean
    const belowArticle = doc.querySelector('a[href="/below/status/999"]')?.closest("article")!;
    assert.equal(belowArticle.getAttribute("data-xrf-state"), null);
    assert.equal(belowArticle.getAttribute("data-xrf-tweet-id"), null);
    assert.equal(belowArticle.querySelector(".xrf-control-bar"), null);

    // Malformed tweetId ignored safely; duplicate 101 renders both updated; dynamic badge text updates in place
    adapter.applyVerdictsInPlace(
      doc,
      [
        {
          tweetId: 'bad"]' as TweetId,
          action: "dim",
          score: 1,
          tokenCount: asTokenCount(4),
          reason: { kind: "clean", action: "keep" },
        },
        {
          tweetId: "101" as TweetId,
          authorId: "bot_a" as AuthorId,
          action: "dim",
          score: 0.94,
          tokenCount: asTokenCount(4),
          reason: {
            kind: "thread_clone",
            action: "dim",
            similarityToPeer: asUnitCosine(0.94),
            canonicalTweetId: "100" as TweetId,
            clusterSize: 2,
          },
        },
      ],
      () => {},
    );
    const dupArticles = Array.from(doc.querySelectorAll('article[data-xrf-tweet-id="101"]'));
    assert.equal(dupArticles.length, 2);
    for (const a of dupArticles) {
      assert.equal(a.querySelector(".xrf-badge")?.textContent, "@bot_a · Bot clone (2x)");
    }

    // Updated clusterSize (2x -> 3x) with same action & reason kind must refresh badge text
    adapter.applyVerdictsInPlace(
      doc,
      [
        {
          tweetId: "101" as TweetId,
          authorId: "bot_a" as AuthorId,
          action: "dim",
          score: 0.96,
          tokenCount: asTokenCount(4),
          reason: {
            kind: "thread_clone",
            action: "dim",
            similarityToPeer: asUnitCosine(0.96),
            canonicalTweetId: "100" as TweetId,
            clusterSize: 3,
          },
        },
      ],
      () => {},
    );
    assert.equal(dupArticles[0]!.querySelector(".xrf-badge")?.textContent, "@bot_a · Bot clone (3x)");

    const styleText = doc.getElementById("xrf-injected-styles")?.textContent ?? "";
    assert.equal(styleText.includes("box-shadow: inset 2px 0 0 rgba(245, 158, 11, 0.55)"), true);
    assert.equal(styleText.includes("border-left:"), false);

    // SPA navigation away from thread (/home -> threadId === null) clears remaining stale articles
    assert.equal(adapter.extractThreadSnapshot(doc, "/home"), null);
    assert.equal(dupArticles[0]!.getAttribute("data-xrf-state"), null);
    assert.equal(dupArticles[0]!.querySelector(".xrf-control-bar"), null);
  });

  it("enforces tap safety (no legacy.verified blue coercion, null origin skip, XHR cleanup on throw) and keeps content.js bundle free of installGraphqlTap & calibrated data", async () => {
    // 1. legacy.verified: false without is_blue_verified must leave isBlueVerified === null
    const legacyOnly = extractGraphqlTweetMetasFromJson({
      data: {
        tweet: {
          __typename: "Tweet",
          rest_id: "7701",
          core: {
            user_results: {
              result: {
                __typename: "User",
                rest_id: "55",
                legacy: { screen_name: "legacy_user", verified: false },
              },
            },
          },
          legacy: { id_str: "7701" },
        },
      },
    });
    assert.equal(legacyOnly.length, 1);
    assert.equal(legacyOnly[0]!.isBlueVerified, null);

    // 2. Null origin skips postMessage (never falls back to "*") and XHR send throw removes load listener
    const posted: unknown[] = [];
    let removedCount = 0;
    class ThrowXhr {
      public status = 200;
      public responseType: XMLHttpRequestResponseType = "text";
      public responseText = "{}";
      public response: unknown = null;
      public listeners: Array<() => void> = [];
      public open(): void {}
      public send(): void {
        throw new Error("InvalidStateError");
      }
      public addEventListener(_t: string, cb: () => void): void {
        this.listeners.push(cb);
      }
      public removeEventListener(_t: string, cb: () => void): void {
        const i = this.listeners.indexOf(cb);
        if (i >= 0) {
          this.listeners.splice(i, 1);
          removedCount += 1;
        }
      }
    }
    const nullOriginWin = {
      location: { origin: "null" },
      XMLHttpRequest: ThrowXhr,
      postMessage(d: unknown) {
        posted.push(d);
      },
      addEventListener() {},
    } as unknown as Window;
    installGraphqlTap(nullOriginWin);
    const xhr = new ThrowXhr();
    xhr.open();
    assert.throws(() => xhr.send(), /InvalidStateError/);
    // Even if URL matched, nothing lingers on ThrowXhr; test matching URL too:
    const xhrProto = (nullOriginWin as unknown as { XMLHttpRequest: { prototype: ThrowXhr } }).XMLHttpRequest.prototype;
    (xhrProto.open as (m: string, u: string) => void).call(xhr, "GET", "https://x.com/i/api/graphql/123/TweetDetail");
    assert.throws(() => xhr.send(), /InvalidStateError/);
    assert.equal(removedCount, 1);
    assert.equal(xhr.listeners.length, 0);
    assert.equal(posted.length, 0);

    // 3. Bundle isolation check: content.js must NOT include installGraphqlTap, XMLHttpRequest, or CALIBRATED_WEIGHTS_V2
    const rootDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
    const contentBundle = await build({
      entryPoints: [path.join(rootDir, "src/content/index.ts")],
      bundle: true,
      write: false,
      format: "iife",
      target: "es2022",
      platform: "browser",
    });
    const contentJs = contentBundle.outputFiles[0]!.text;
    assert.equal(contentJs.includes("installGraphqlTap"), false);
    assert.equal(contentJs.includes("XMLHttpRequest"), false);
    assert.equal(contentJs.includes("CALIBRATED_WEIGHTS_V2"), false);
    assert.equal(contentJs.includes("ANCHOR_SET_V2"), false);

    const tapBundle = await build({
      entryPoints: [path.join(rootDir, "src/content/graphql-tap.ts")],
      bundle: true,
      write: false,
      format: "iife",
      target: "es2022",
      platform: "browser",
    });
    const tapJs = tapBundle.outputFiles[0]!.text;
    assert.equal(tapJs.includes("installGraphqlTap"), true);
    assert.equal(tapJs.includes("chrome.runtime"), false);
  });

  it("ignores XRF-only childList bar insertion mutations in mountThreadObserver", async () => {
    const doc = buildMockXThreadDocument();
    let observerCb: ((records: MutationRecord[]) => void) | null = null;
    class MockObserver {
      constructor(cb: (records: MutationRecord[]) => void) {
        observerCb = cb;
      }
      observe() {}
      disconnect() {}
    }
    const mockWin = {
      location: { origin: "https://x.com" },
      MutationObserver: MockObserver,
      postMessage() {},
      addEventListener() {},
      removeEventListener() {},
    } as unknown as Window;

    let rpcCount = 0;
    const sendRpc = async (req: OffscreenRpcRequest): Promise<OffscreenRpcResponse> => {
      rpcCount += 1;
      if (req.kind === "evaluate_batch") {
        return {
          kind: "evaluate_batch_ok",
          result: {
            threadId: req.threadId,
            anchorKind: "text_anchored",
            verdicts: [
              {
                tweetId: "1001" as TweetId,
                authorId: "bot_account" as AuthorId,
                action: "dim",
                score: 0.9,
                tokenCount: asTokenCount(14),
                reason: { kind: "keyword_spam", action: "dim", matchedPattern: "test" },
              },
            ],
            retroactiveUpdates: [],
          },
        };
      }
      return { kind: "health_ok", gpuStatus: "ready", deviceEpoch: 1 };
    };

    const handle = mountThreadObserver(doc, sendRpc, () => "/op_author/status/1000", 10, {
      win: mockWin,
    });
    await handle.flushNow();
    assert.equal(rpcCount, 1);

    const bar = doc.querySelector(".xrf-control-bar")!;
    const col = bar.parentElement!;
    assert(observerCb !== null);
    (observerCb as (records: MutationRecord[]) => void)([
      {
        type: "childList",
        target: col,
        addedNodes: [bar] as unknown as NodeList,
        removedNodes: [] as unknown as NodeList,
      } as unknown as MutationRecord,
    ]);
    await new Promise((r) => setTimeout(r, 30));
    assert.equal(rpcCount, 1, "XRF-only childList bar insertion must not trigger extra RPC pass");

    handle.dispose();
  });
});
