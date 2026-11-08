import { extractDomSignalsFromText } from "../domain/seeds.ts";
import {
  type AuthorId,
  type CandidateTweet,
  type FilterReason,
  type ThreadId,
  type TweetId,
  type VerdictRecord,
  parseAuthorId,
  parseThreadId,
  parseTweetId,
} from "../domain/types.ts";

const DISCOVER_MORE_HEADINGS: ReadonlyArray<RegExp> = [
  /^discover more$/i,
  /^more tweets$/i,
  /^more posts$/i,
  /^发现更多$/,
  /^更多帖子$/,
  /^さらに表示$/,
  /^他のポスト$/,
];

const STYLE_ELEMENT_ID = "xrf-injected-styles" as const;

const XRF_STYLESHEET = `
article[data-xrf-state="dimmed"] [data-testid="tweetText"],
article[data-xrf-state="dimmed"] [data-testid="tweetPhoto"],
article[data-xrf-state="dimmed"] [data-testid="videoPlayer"],
article[data-xrf-state="dimmed"] [data-testid="card.wrapper"] {
  opacity: 0.22 !important;
  filter: blur(3.5px) !important;
  transition: opacity 120ms ease, filter 120ms ease !important;
}
article[data-xrf-state="revealed"] [data-testid="tweetText"],
article[data-xrf-state="revealed"] [data-testid="tweetPhoto"],
article[data-xrf-state="revealed"] [data-testid="videoPlayer"],
article[data-xrf-state="revealed"] [data-testid="card.wrapper"] {
  opacity: 1 !important;
  filter: none !important;
}
.xrf-control-bar {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  margin-top: 4px;
  padding: 2px 8px;
  border-radius: 9999px;
  font-size: 11px;
  line-height: 16px;
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: rgba(120, 120, 128, 0.14);
  color: inherit;
  user-select: none;
}
.xrf-control-bar[data-xrf-bar-action="dim"] {
  background: rgba(244, 33, 46, 0.14);
}
.xrf-btn {
  border: none;
  background: transparent;
  color: inherit;
  cursor: pointer;
  padding: 0 4px;
  font-size: 11px;
  font-weight: 600;
  text-decoration: underline;
  opacity: 0.85;
}
.xrf-btn:hover {
  opacity: 1;
}
`;

export function ensureStylesInjected(doc: Document): void {
  if (doc.getElementById(STYLE_ELEMENT_ID)) {
    return;
  }
  const style = doc.createElement("style");
  style.id = STYLE_ELEMENT_ID;
  style.textContent = XRF_STYLESHEET;
  (doc.head ?? doc.documentElement).appendChild(style);
}

export function isDiscoverMoreDivider(cell: Element): boolean {
  if (cell.querySelector('article[data-testid="tweet"]') !== null) {
    return false;
  }
  const headings = Array.from(cell.querySelectorAll('h2, [role="heading"], span'));
  for (const node of headings) {
    const text = (node.textContent ?? "").trim();
    if (text.length === 0 || text.length > 40) {
      continue;
    }
    if (DISCOVER_MORE_HEADINGS.some((rx) => rx.test(text))) {
      return true;
    }
  }
  return false;
}

function isInsideQuoteTweet(node: Element, article: Element): boolean {
  let curr: Element | null = node.parentElement;
  while (curr !== null && curr !== article) {
    const testId = curr.getAttribute("data-testid");
    if (testId === "quoteTweet" || testId === "card.wrapper") {
      return true;
    }
    curr = curr.parentElement;
  }
  return false;
}

function extractStatusIdentity(
  article: Element,
): { readonly tweetId: TweetId; readonly authorId: AuthorId } | null {
  const links = Array.from(article.querySelectorAll('a[href*="/status/"]'));
  for (const link of links) {
    if (isInsideQuoteTweet(link, article)) {
      continue;
    }
    const href = link.getAttribute("href") ?? "";
    const match = /^\/?([^/]+)\/status\/(\d{1,32})/.exec(href);
    if (!match || !match[1] || !match[2]) {
      continue;
    }
    const authorId = parseAuthorId(match[1]);
    const tweetId = parseTweetId(match[2]);
    if (authorId !== null && tweetId !== null) {
      return { tweetId, authorId };
    }
  }
  return null;
}

function extractPrimaryTweetText(article: Element): string {
  const textNodes = Array.from(article.querySelectorAll('[data-testid="tweetText"]'));
  for (const node of textNodes) {
    if (!isInsideQuoteTweet(node, article)) {
      return (node.textContent ?? "").trim();
    }
  }
  return "";
}

export function formatFilterReasonBadge(reason: FilterReason): string {
  switch (reason.kind) {
    case "op_paraphrase":
      return `AI paraphrase (${Math.round(reason.similarityToOp * 100)}%)`;
    case "thread_clone":
      return `Bot clone (${reason.clusterSize}x)`;
    case "keyword_spam":
      return `Spam (${reason.matchedPattern})`;
    case "siamese_bot":
      return `Likely AI bot (${Math.round(reason.probability * 100)}%)`;
    case "user_override":
      return reason.label === "spam" ? "Marked bot" : "Marked human";
    case "short_human_exempt":
    case "clean":
      return "Clean";
  }
}

export class XThreadDomAdapter {
  public extractThreadSnapshot(
    root: ParentNode,
    locationPathname: string,
  ): {
    readonly threadId: ThreadId;
    readonly candidates: ReadonlyArray<CandidateTweet>;
  } | null {
    const threadId = parseThreadId(locationPathname);
    if (threadId === null) {
      return null;
    }

    const cells = Array.from(root.querySelectorAll('[data-testid="cellInnerDiv"]'));
    const articles: Element[] = [];

    if (cells.length > 0) {
      for (const cell of cells) {
        if (isDiscoverMoreDivider(cell)) {
          break;
        }
        const article = cell.querySelector('article[data-testid="tweet"]');
        if (article !== null) {
          articles.push(article);
        }
      }
    } else {
      articles.push(...Array.from(root.querySelectorAll('article[data-testid="tweet"]')));
    }

    const candidates: CandidateTweet[] = [];
    for (const article of articles) {
      const identity = extractStatusIdentity(article);
      if (identity === null) {
        continue;
      }

      article.setAttribute("data-xrf-tweet-id", identity.tweetId);

      const rawText = extractPrimaryTweetText(article);
      const hasVerifiedBadge =
        article.querySelector('[data-testid="icon-verified"]') !== null;
      const hasMediaOrExternalLink =
        article.querySelector(
          '[data-testid="tweetPhoto"], [data-testid="videoPlayer"], [data-testid="card.wrapper"], a[target="_blank"]',
        ) !== null;

      const domSignals = extractDomSignalsFromText(
        rawText,
        hasVerifiedBadge,
        hasMediaOrExternalLink,
      );

      candidates.push({
        tweetId: identity.tweetId,
        authorId: identity.authorId,
        rawText,
        normalizedText: rawText,
        isOp: identity.tweetId === (threadId as unknown as TweetId),
        domSignals,
      });
    }

    return { threadId, candidates };
  }

  public applyVerdictsInPlace(
    root: ParentNode,
    verdicts: ReadonlyArray<VerdictRecord>,
    onFeedback: (tweetId: TweetId, label: "spam" | "human") => void,
  ): void {
    const ownerDoc = "createElement" in root ? (root as Document) : root.ownerDocument;
    if (ownerDoc !== null) {
      ensureStylesInjected(ownerDoc);
    }

    for (const verdict of verdicts) {
      const article = root.querySelector(
        `article[data-xrf-tweet-id="${verdict.tweetId}"]`,
      );
      if (article === null || ownerDoc === null) {
        continue;
      }

      const targetState =
        verdict.action === "dim"
          ? article.getAttribute("data-xrf-user-revealed") === "true"
            ? "revealed"
            : "dimmed"
          : "clean";

      const currentState = article.getAttribute("data-xrf-state");
      const currentReason = article.getAttribute("data-xrf-reason");

      if (
        currentState === targetState &&
        currentReason === verdict.reason.kind &&
        article.querySelector(".xrf-control-bar") !== null
      ) {
        continue;
      }

      article.setAttribute("data-xrf-state", targetState);
      article.setAttribute("data-xrf-reason", verdict.reason.kind);

      let bar = article.querySelector(".xrf-control-bar");
      if (bar === null) {
        bar = ownerDoc.createElement("div");
        bar.className = "xrf-control-bar";
        article.appendChild(bar);
      }

      bar.setAttribute("data-xrf-bar-action", verdict.action);
      bar.textContent = "";

      if (verdict.action === "dim") {
        const badge = ownerDoc.createElement("span");
        badge.className = "xrf-badge";
        badge.textContent = formatFilterReasonBadge(verdict.reason);
        bar.appendChild(badge);

        const toggleBtn = ownerDoc.createElement("button");
        toggleBtn.type = "button";
        toggleBtn.className = "xrf-btn xrf-toggle-btn";
        toggleBtn.textContent = targetState === "dimmed" ? "Show" : "Hide";
        toggleBtn.addEventListener("click", (ev) => {
          ev.stopPropagation();
          const nowRevealed = article.getAttribute("data-xrf-state") === "dimmed";
          article.setAttribute("data-xrf-user-revealed", nowRevealed ? "true" : "false");
          article.setAttribute("data-xrf-state", nowRevealed ? "revealed" : "dimmed");
          toggleBtn.textContent = nowRevealed ? "Hide" : "Show";
        });
        bar.appendChild(toggleBtn);

        const humanBtn = ownerDoc.createElement("button");
        humanBtn.type = "button";
        humanBtn.className = "xrf-btn xrf-feedback-human";
        humanBtn.textContent = "Not spam";
        humanBtn.addEventListener("click", (ev) => {
          ev.stopPropagation();
          article.removeAttribute("data-xrf-user-revealed");
          onFeedback(verdict.tweetId, "human");
        });
        bar.appendChild(humanBtn);
      } else {
        const spamBtn = ownerDoc.createElement("button");
        spamBtn.type = "button";
        spamBtn.className = "xrf-btn xrf-feedback-spam";
        spamBtn.textContent = "Mark bot";
        spamBtn.addEventListener("click", (ev) => {
          ev.stopPropagation();
          article.removeAttribute("data-xrf-user-revealed");
          onFeedback(verdict.tweetId, "spam");
        });
        bar.appendChild(spamBtn);
      }
    }
  }
}
