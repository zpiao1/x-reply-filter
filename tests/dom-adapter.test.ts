import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { parseHTML } from "linkedom";
import { XThreadDomAdapter } from "../src/content/dom-adapter.ts";
import { runScrollPass } from "../src/content/index.ts";
import {
  type TweetId,
  type VerdictRecord,
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
              <a href="/jp_user/status/1002"><time datetime="2026-11-08T12:01:00Z"></time></a>
              <div data-testid="tweetText">草www</div>
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

    const sendRpc = async (req: OffscreenRpcRequest): Promise<OffscreenRpcResponse> => {
      requests.push(req);
      if (req.kind === "evaluate_batch") {
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

    const toggleBtn = botArticle.querySelector(".xrf-toggle-btn") as HTMLButtonElement | null;
    assert(toggleBtn !== null);
    toggleBtn.click();
    assert.equal(botArticle.getAttribute("data-xrf-state"), "revealed");
    assert.equal(botArticle.getAttribute("data-xrf-user-revealed"), "true");

    await runScrollPass(doc, "/op_author/status/1000", adapter, sendRpc);
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
});
