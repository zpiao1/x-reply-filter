import { extractDomSignalsFromText } from "../domain/stylometrics.ts";
import {
  type AuthorId,
  type CandidateTweet,
  type FilterReason,
  type GraphqlTweetMeta,
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
  /^探索更多$/,
  /^更多貼文$/,
  /^來自整個\s*X$/i,
  /^来自整个\s*X$/i,
  /^さらに表示$/,
  /^他のポスト$/,
];

const STYLE_ELEMENT_ID = "xrf-injected-styles" as const;

const XRF_STYLESHEET = `
article[data-xrf-state="dimmed"] *:not(:has(.xrf-control-bar)):not(.xrf-control-bar):not(.xrf-control-bar *) {
  display: none !important;
}
article[data-xrf-state="dimmed"],
article[data-xrf-state="dimmed"] *:has(.xrf-control-bar) {
  padding-top: 2px !important;
  padding-bottom: 2px !important;
  margin-top: 0 !important;
  margin-bottom: 0 !important;
  min-height: 0 !important;
}
article[data-xrf-state="revealed"] [data-testid="tweetText"],
article[data-xrf-state="revealed"] [data-testid="tweetPhoto"],
article[data-xrf-state="revealed"] [data-testid="videoPlayer"],
article[data-xrf-state="revealed"] [data-testid="card.wrapper"] {
  opacity: 1 !important;
  filter: none !important;
}
article[data-xrf-state="suspect"] {
  box-shadow: inset 2px 0 0 rgba(245, 158, 11, 0.55) !important;
}
article[data-xrf-state="suspect"] [data-testid="tweetText"],
article[data-xrf-state="suspect"] [data-testid="tweetPhoto"],
article[data-xrf-state="suspect"] [data-testid="videoPlayer"],
article[data-xrf-state="suspect"] [data-testid="card.wrapper"] {
  opacity: 1 !important;
  filter: none !important;
}
.xrf-control-bar {
  display: inline-flex !important;
  flex-direction: row !important;
  align-items: center !important;
  align-self: flex-start !important;
  flex: 0 0 auto !important;
  width: fit-content !important;
  max-width: 100% !important;
  height: auto !important;
  box-sizing: border-box !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  gap: 6px;
  margin-top: 6px;
  padding: 2px 8px;
  border-radius: 6px;
  font-size: 11px;
  line-height: 16px;
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: rgba(120, 120, 128, 0.14);
  color: rgb(139, 152, 165);
  user-select: none;
}
article[data-xrf-state="dimmed"] .xrf-control-bar {
  margin-top: 2px !important;
  margin-bottom: 2px !important;
}
.xrf-control-bar[data-xrf-bar-action="dim"] {
  background: rgba(244, 33, 46, 0.14);
  color: rgb(244, 113, 120);
}
.xrf-control-bar[data-xrf-bar-action="suspect"] {
  background: rgba(245, 158, 11, 0.14);
  color: rgb(251, 191, 36);
  border: 1px solid rgba(245, 158, 11, 0.28);
}
.xrf-control-bar[data-xrf-bar-action="keep"] {
  opacity: 0;
  transition: opacity 120ms ease;
}
article[data-testid="tweet"]:hover .xrf-control-bar[data-xrf-bar-action="keep"] {
  opacity: 0.85;
}
.xrf-badge {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  min-width: 0;
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
  flex-shrink: 0;
  white-space: nowrap;
}
.xrf-btn:hover {
  opacity: 1;
}
`;

export function ensureStylesInjected(doc: Document): void {
  let style = doc.getElementById(STYLE_ELEMENT_ID);
  if (style === null) {
    style = doc.createElement("style");
    style.id = STYLE_ELEMENT_ID;
    (doc.head ?? doc.documentElement).appendChild(style);
  }
  if (style.textContent !== XRF_STYLESHEET) {
    style.textContent = XRF_STYLESHEET;
  }
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

function isInsideQuoteOnly(node: Element, article: Element): boolean {
  let curr: Element | null = node.parentElement;
  while (curr !== null && curr !== article) {
    if (curr.tagName.toUpperCase() === "ARTICLE") {
      return true;
    }
    if (curr.getAttribute("data-testid") === "quoteTweet") {
      return true;
    }
    curr = curr.parentElement;
  }
  return false;
}

function isInsideQuoteTweet(node: Element, article: Element): boolean {
  let curr: Element | null = node.parentElement;
  while (curr !== null && curr !== article) {
    if (curr.tagName.toUpperCase() === "ARTICLE") {
      return true;
    }
    const testId = curr.getAttribute("data-testid");
    if (testId === "quoteTweet" || testId === "card.wrapper") {
      return true;
    }
    curr = curr.parentElement;
  }
  return false;
}

/**
 * Recursively extracts tweet text including X's <img alt="..."> Twemoji icons
 * and <br> newlines without relying on browser-only instanceof constructors.
 */
export function extractTweetTextWithTwemoji(node: Node | null): string {
  if (node === null) {
    return "";
  }
  if (node.nodeType === 3) {
    return node.nodeValue ?? "";
  }
  if (node.nodeType !== 1) {
    return "";
  }
  const el = node as Element;
  const tag = el.tagName.toUpperCase();
  if (tag === "IMG") {
    return el.getAttribute("alt") ?? "";
  }
  if (tag === "BR") {
    return "\n";
  }
  let out = "";
  for (let child = el.firstChild; child !== null; child = child.nextSibling) {
    out += extractTweetTextWithTwemoji(child);
  }
  return out;
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

function extractPrimaryTweetTextNode(article: Element): Element | null {
  const textNodes = Array.from(article.querySelectorAll('[data-testid="tweetText"]'));
  for (const node of textNodes) {
    if (!isInsideQuoteTweet(node, article)) {
      return node;
    }
  }
  return null;
}

function findControlMountContainer(article: Element): Element {
  const textNode = extractPrimaryTweetTextNode(article);
  if (textNode?.parentElement && textNode.parentElement !== article) {
    return textNode.parentElement;
  }
  const actionGroups = Array.from(article.querySelectorAll('div[role="group"]'));
  for (let i = actionGroups.length - 1; i >= 0; i--) {
    const group = actionGroups[i]!;
    if (!isInsideQuoteTweet(group, article) && group.parentElement && group.parentElement !== article) {
      return group.parentElement;
    }
  }
  if (article.firstElementChild && article.firstElementChild !== article) {
    return article.firstElementChild;
  }
  return article;
}

function clearRecycledArticleAttributes(article: Element): void {
  article.removeAttribute("data-xrf-state");
  article.removeAttribute("data-xrf-reason");
  article.removeAttribute("data-xrf-score");
  article.removeAttribute("data-xrf-tokens");
  article.removeAttribute("data-xrf-siamese");
  article.removeAttribute("data-xrf-op-sim");
  article.removeAttribute("data-xrf-clone-sim");
  article.removeAttribute("data-xrf-author-prior");
  article.removeAttribute("data-xrf-meta-source");
  article.removeAttribute("data-xrf-user-revealed");
  article.querySelector(".xrf-control-bar")?.remove();
}

export function formatFilterReasonBadge(reason: FilterReason): string {
  switch (reason.kind) {
    case "known_bot_author":
      return `Known bot (@${reason.authorId}, ${reason.spamCount}x)`;
    case "keyword_spam":
      return `Spam (${reason.matchedPattern})`;
    case "bio_scam":
      return `Bio scam (${reason.matchedPattern})`;
    case "bot_summon":
      return `Bot summon (${reason.handle})`;
    case "bot_answer":
      return `Bot reply (@${reason.authorId})`;
    case "blue_short_praise":
      return `Blue praise${reason.confirmedUnfollowed ? "" : " ?"}`;
    case "media_only_blue":
      return `Media-only blue${reason.confirmedUnfollowed ? "" : " ?"}`;
    case "op_paraphrase":
      return `AI paraphrase (${Math.round(reason.similarityToOp * 100)}%)`;
    case "thread_clone":
      return `Bot clone (${reason.clusterSize}x)`;
    case "siamese_bot":
      return `Likely AI bot (${Math.round(reason.probability * 100)}%)`;
    case "model_suspect":
      return `Borderline (${Math.round(reason.probability * 100)}%)`;
    case "user_override":
      return reason.label === "spam" ? "Marked bot" : "Marked human";
    case "short_human_exempt":
    case "clean":
      return "Clean";
  }
  const _exhaustive: never = reason;
  return _exhaustive;
}

export function formatDiagnosticsTooltip(verdict: VerdictRecord): string {
  const exp = verdict.explanation;
  if (exp) {
    const parts: string[] = [
      `tier=${exp.tier}`,
      `rule=${exp.ruleId}`,
      `p=${Math.round(exp.probability * 100)}% (${exp.preset})`,
    ];
    if (exp.top.length > 0) {
      const topStr = exp.top
        .map((s) => `${s.id}+${s.contribution.toFixed(2)}${s.evidence ? `(${s.evidence})` : ""}`)
        .join(", ");
      parts.push(`top=[${topStr}]`);
    }
    if (exp.counter.length > 0) {
      const counterStr = exp.counter
        .map((s) => `${s.id}${s.contribution.toFixed(2)}`)
        .join(", ");
      parts.push(`counter=[${counterStr}]`);
    }
    parts.push(`lang=${exp.lang.reply}${exp.lang.op ? `/${exp.lang.op}` : ""}`);
    parts.push(`meta=${exp.metaSource}`);
    if (exp.shields.length > 0) {
      parts.push(`shields=${exp.shields.join(",")}`);
    }
    if (exp.caps.length > 0) {
      parts.push(`caps=${exp.caps.join(",")}`);
    }
    if (exp.stage2) {
      parts.push(
        `stage2=${exp.stage2.verdict}(${Math.round(exp.stage2.confidence * 100)}%)`,
      );
    }
    return parts.join(" | ");
  }

  const d = verdict.diagnostics;
  if (!d) {
    return `reason=${verdict.reason.kind} | score=${verdict.score.toFixed(2)} | tokens=${verdict.tokenCount}`;
  }
  const priorSign = d.authorSpamPrior >= 0 ? "+" : "";
  return [
    `reason=${verdict.reason.kind}`,
    `siamese=${Math.round(d.siameseProbability * 100)}%`,
    `opSim=${d.opCosSim.toFixed(2)}`,
    `cloneSim=${d.maxCloneSim.toFixed(2)}`,
    `authorPrior=${priorSign}${d.authorSpamPrior.toFixed(2)} (${d.authorSpamCount} spam / ${d.authorHumanCount} human)`,
    `tokens=${verdict.tokenCount}`,
  ].join(" | ");
}

export function parseFocalAuthorIdFromPathname(pathname: string): AuthorId | null {
  const match = /^\/?([^/]+)\/status\/\d{1,32}(?:\/|$|\?)/i.exec(pathname.trim());
  if (!match || !match[1]) {
    return null;
  }
  const candidate = match[1].trim().replace(/^@+/, "").toLowerCase();
  if (candidate === "i" || candidate === "intent") {
    return null;
  }
  return parseAuthorId(candidate);
}

export interface ThreadDomSnapshot {
  readonly threadId: ThreadId;
  readonly focalAuthorId?: AuthorId | null | undefined;
  readonly candidates: ReadonlyArray<CandidateTweet>;
  readonly hasUnscoredReplies: boolean;
}

export class XThreadDomAdapter {
  public extractThreadSnapshot(
    root: ParentNode,
    locationPathname: string,
    graphqlCache?: ReadonlyMap<TweetId, GraphqlTweetMeta>,
  ): ThreadDomSnapshot | null {
    const threadId = parseThreadId(locationPathname);
    if (threadId === null) {
      for (const article of Array.from(
        root.querySelectorAll('article[data-xrf-state], article[data-xrf-tweet-id]'),
      )) {
        const identity = extractStatusIdentity(article);
        const prevId = article.getAttribute("data-xrf-tweet-id");
        if (
          article.hasAttribute("data-xrf-state") ||
          (prevId !== null && (identity === null || prevId !== identity.tweetId))
        ) {
          article.removeAttribute("data-xrf-tweet-id");
          clearRecycledArticleAttributes(article);
        }
      }
      return null;
    }

    const focalAuthorId = parseFocalAuthorIdFromPathname(locationPathname);
    const cells = Array.from(root.querySelectorAll('[data-testid="cellInnerDiv"]'));
    const articles: Element[] = [];

    if (cells.length > 0) {
      let hitCutoff = false;
      for (const cell of cells) {
        if (!hitCutoff && isDiscoverMoreDivider(cell)) {
          hitCutoff = true;
          continue;
        }
        if (hitCutoff) {
          for (const belowArticle of Array.from(
            cell.querySelectorAll('article[data-testid="tweet"]'),
          )) {
            if (
              belowArticle.hasAttribute("data-xrf-state") ||
              belowArticle.hasAttribute("data-xrf-tweet-id") ||
              belowArticle.querySelector(".xrf-control-bar") !== null
            ) {
              belowArticle.removeAttribute("data-xrf-tweet-id");
              clearRecycledArticleAttributes(belowArticle);
            }
          }
          continue;
        }
        const article = cell.querySelector('article[data-testid="tweet"]');
        if (article !== null) {
          articles.push(article);
        }
      }
    } else {
      articles.push(...Array.from(root.querySelectorAll('article[data-testid="tweet"]')));
    }

    interface ExtractedEntry {
      readonly article: Element;
      readonly identity: { readonly tweetId: TweetId; readonly authorId: AuthorId };
      readonly rawText: string;
      readonly hasVerifiedBadge: boolean;
      readonly hasPrimaryMedia: boolean;
      readonly hasExternalLink: boolean;
      readonly hasQuote: boolean;
      readonly graphqlMeta: GraphqlTweetMeta | null;
      readonly isOp: boolean;
    }

    const entries: ExtractedEntry[] = [];
    let hasUnscoredReplies = false;
    let focalIdx = -1;

    for (const article of articles) {
      const identity = extractStatusIdentity(article);
      if (identity === null) {
        continue;
      }

      const prevTweetId = article.getAttribute("data-xrf-tweet-id");
      if (prevTweetId !== identity.tweetId) {
        if (prevTweetId !== null) {
          clearRecycledArticleAttributes(article);
        }
        article.setAttribute("data-xrf-tweet-id", identity.tweetId);
      }

      const isOp = identity.tweetId === (threadId as unknown as TweetId);
      if (isOp && focalIdx === -1) {
        focalIdx = entries.length;
      }
      if (!isOp && article.getAttribute("data-xrf-state") === null) {
        hasUnscoredReplies = true;
      }

      const textNode = extractPrimaryTweetTextNode(article);
      const rawText = extractTweetTextWithTwemoji(textNode).trim();
      const rawMeta = graphqlCache?.get(identity.tweetId) ?? null;
      const graphqlMeta =
        rawMeta !== null &&
        ((rawMeta.screenName != null && rawMeta.screenName !== identity.authorId) ||
          (rawMeta.tweetId != null && rawMeta.tweetId !== identity.tweetId))
          ? null
          : rawMeta;

      const hasVerifiedBadge = Array.from(
        article.querySelectorAll('[data-testid="icon-verified"]'),
      ).some((el) => !isInsideQuoteTweet(el, article));

      const domPrimaryMedia = Array.from(
        article.querySelectorAll(
          '[data-testid="tweetPhoto"], [data-testid="videoPlayer"], [data-testid="videoComponent"], video',
        ),
      ).some((el) => !isInsideQuoteTweet(el, article));
      const hasPrimaryMedia =
        domPrimaryMedia || (graphqlMeta !== null && graphqlMeta.mediaTypes.length > 0);

      const hasExternalLink = Array.from(
        article.querySelectorAll('[data-testid="card.wrapper"], a[target="_blank"]'),
      ).some((el) => !isInsideQuoteOnly(el, article));

      const hasQuote = Array.from(
        article.querySelectorAll('[data-testid="quoteTweet"]'),
      ).some((el) => !isInsideQuoteOnly(el, article));

      entries.push({
        article,
        identity,
        rawText,
        hasVerifiedBadge,
        hasPrimaryMedia,
        hasExternalLink,
        hasQuote,
        graphqlMeta,
        isOp,
      });
    }

    const opMeta =
      (focalIdx >= 0 ? entries[focalIdx]?.graphqlMeta : null) ??
      graphqlCache?.get(threadId as unknown as TweetId) ??
      null;
    const opFavoriteCount = opMeta?.favoriteCount ?? null;

    const candidates: CandidateTweet[] = entries.map((entry, idx) => {
      const hasMediaOrExternalLink = entry.hasPrimaryMedia || entry.hasExternalLink;
      const domSignals = extractDomSignalsFromText(
        entry.rawText,
        entry.hasVerifiedBadge,
        hasMediaOrExternalLink,
      );
      const isContextAncestor = focalIdx >= 0 && idx < focalIdx;

      return {
        tweetId: entry.identity.tweetId,
        authorId: entry.identity.authorId,
        rawText: entry.rawText,
        normalizedText: entry.rawText,
        isOp: entry.isOp,
        domSignals,
        hasPrimaryMedia: entry.hasPrimaryMedia,
        hasQuote: entry.hasQuote,
        isContextAncestor,
        ...(entry.graphqlMeta !== null ? { graphqlMeta: entry.graphqlMeta } : {}),
        ...(opFavoriteCount !== null ? { opFavoriteCount } : {}),
      };
    });

    return { threadId, focalAuthorId, candidates, hasUnscoredReplies };
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
      const safeTweetId = parseTweetId(verdict.tweetId);
      if (safeTweetId === null || ownerDoc === null) {
        continue;
      }
      const matchedArticles = Array.from(
        root.querySelectorAll(`article[data-xrf-tweet-id="${safeTweetId}"]`),
      );
      if (matchedArticles.length === 0) {
        continue;
      }

      for (const article of matchedArticles) {
        const targetState =
          verdict.action === "dim"
            ? article.getAttribute("data-xrf-user-revealed") === "true"
              ? "revealed"
              : "dimmed"
            : verdict.action === "suspect"
              ? "suspect"
              : "clean";

        if (verdict.action !== "dim" && article.hasAttribute("data-xrf-user-revealed")) {
          article.removeAttribute("data-xrf-user-revealed");
        }

        const currentState = article.getAttribute("data-xrf-state");
        const currentReason = article.getAttribute("data-xrf-reason");
        const existingBar = article.querySelector(".xrf-control-bar");
        const stage2Tag = verdict.explanation?.stage2
          ? `${verdict.explanation.stage2.verdict}:${verdict.explanation.stage2.confidence.toFixed(2)}`
          : "";
        const expTag = verdict.explanation
          ? `${verdict.explanation.preset}:${verdict.explanation.metaSource}:${verdict.explanation.ruleId}:${verdict.explanation.top.map((s) => s.id).join(",")}`
          : "";

        const targetTitle = formatDiagnosticsTooltip(verdict);
        const targetBadgeText =
          verdict.action === "dim"
            ? (() => {
                const resolvedAuthor =
                  verdict.authorId ?? extractStatusIdentity(article)?.authorId;
                const baseBadge = formatFilterReasonBadge(verdict.reason);
                return resolvedAuthor &&
                  verdict.reason.kind !== "known_bot_author" &&
                  verdict.reason.kind !== "bot_answer"
                  ? `@${resolvedAuthor} · ${baseBadge}`
                  : baseBadge;
              })()
            : verdict.action === "suspect"
              ? (() => {
                  const baseBadge = formatFilterReasonBadge(verdict.reason);
                  return verdict.reason.kind === "model_suspect"
                    ? baseBadge
                    : `Suspect · ${baseBadge}`;
                })()
              : "";

        const currentBadgeText = existingBar?.querySelector(".xrf-badge")?.textContent ?? "";
        if (
          currentState === targetState &&
          currentReason === verdict.reason.kind &&
          existingBar !== null &&
          existingBar.getAttribute("data-xrf-bar-action") === verdict.action &&
          currentBadgeText === targetBadgeText &&
          (existingBar.getAttribute("title") ?? "") === targetTitle &&
          (existingBar.getAttribute("data-xrf-stage2") ?? "") === stage2Tag &&
          (existingBar.getAttribute("data-xrf-exp") ?? "") === expTag
        ) {
          continue;
        }

        article.setAttribute("data-xrf-state", targetState);
        article.setAttribute("data-xrf-reason", verdict.reason.kind);
        article.setAttribute("data-xrf-score", verdict.score.toFixed(3));
        article.setAttribute("data-xrf-tokens", String(verdict.tokenCount));
        if (verdict.explanation) {
          article.setAttribute("data-xrf-meta-source", verdict.explanation.metaSource);
        }
        if (verdict.diagnostics) {
          article.setAttribute(
            "data-xrf-siamese",
            verdict.diagnostics.siameseProbability.toFixed(3),
          );
          article.setAttribute("data-xrf-op-sim", verdict.diagnostics.opCosSim.toFixed(3));
          article.setAttribute(
            "data-xrf-clone-sim",
            verdict.diagnostics.maxCloneSim.toFixed(3),
          );
          article.setAttribute(
            "data-xrf-author-prior",
            verdict.diagnostics.authorSpamPrior.toFixed(3),
          );
        }

        const mountTarget = findControlMountContainer(article);
        let bar = existingBar;
        if (bar === null) {
          bar = ownerDoc.createElement("div");
          bar.className = "xrf-control-bar";
          mountTarget.appendChild(bar);
        } else if (bar.parentElement !== mountTarget) {
          mountTarget.appendChild(bar);
        }

        bar.setAttribute("data-xrf-bar-action", verdict.action);
        if (stage2Tag !== "") {
          bar.setAttribute("data-xrf-stage2", stage2Tag);
        } else {
          bar.removeAttribute("data-xrf-stage2");
        }
        if (expTag !== "") {
          bar.setAttribute("data-xrf-exp", expTag);
        } else {
          bar.removeAttribute("data-xrf-exp");
        }
        bar.setAttribute("title", targetTitle);
        bar.textContent = "";

        if (verdict.action === "dim") {
          const badge = ownerDoc.createElement("span");
          badge.className = "xrf-badge";
          badge.textContent = targetBadgeText;
          bar.appendChild(badge);

          const toggleBtn = ownerDoc.createElement("button");
          toggleBtn.type = "button";
          toggleBtn.className = "xrf-btn xrf-toggle-btn";
          toggleBtn.textContent = targetState === "dimmed" ? "Show" : "Hide";
          toggleBtn.addEventListener("click", (ev) => {
            ev.stopPropagation();
            ev.preventDefault?.();
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
            ev.preventDefault?.();
            article.removeAttribute("data-xrf-user-revealed");
            onFeedback(safeTweetId, "human");
          });
          bar.appendChild(humanBtn);
        } else if (verdict.action === "suspect") {
          const badge = ownerDoc.createElement("span");
          badge.className = "xrf-badge";
          badge.textContent = targetBadgeText;
          bar.appendChild(badge);

          const spamBtn = ownerDoc.createElement("button");
          spamBtn.type = "button";
          spamBtn.className = "xrf-btn xrf-feedback-spam";
          spamBtn.textContent = "Mark bot";
          spamBtn.addEventListener("click", (ev) => {
            ev.stopPropagation();
            ev.preventDefault?.();
            article.removeAttribute("data-xrf-user-revealed");
            onFeedback(safeTweetId, "spam");
          });
          bar.appendChild(spamBtn);

          const humanBtn = ownerDoc.createElement("button");
          humanBtn.type = "button";
          humanBtn.className = "xrf-btn xrf-feedback-human";
          humanBtn.textContent = "Not spam";
          humanBtn.addEventListener("click", (ev) => {
            ev.stopPropagation();
            ev.preventDefault?.();
            article.removeAttribute("data-xrf-user-revealed");
            onFeedback(safeTweetId, "human");
          });
          bar.appendChild(humanBtn);
        } else {
          const spamBtn = ownerDoc.createElement("button");
          spamBtn.type = "button";
          spamBtn.className = "xrf-btn xrf-feedback-spam";
          spamBtn.textContent = "Mark bot";
          spamBtn.addEventListener("click", (ev) => {
            ev.stopPropagation();
            ev.preventDefault?.();
            article.removeAttribute("data-xrf-user-revealed");
            onFeedback(safeTweetId, "spam");
          });
          bar.appendChild(spamBtn);
        }
      }
    }
  }
}
