async () => {
  const cutoffRegexes = [
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

  let xrfMutationsInWindow = 0;
  const observer = new MutationObserver((records) => {
    for (const rec of records) {
      const target = rec.target;
      if (
        target instanceof Element &&
        (target.classList.contains("xrf-control-bar") ||
          target.closest(".xrf-control-bar") !== null ||
          rec.attributeName?.startsWith("data-xrf-"))
      ) {
        xrfMutationsInWindow += 1;
      }
    }
  });

  observer.observe(document.body, {
    childList: true,
    subtree: true,
    attributes: true,
  });
  await new Promise((resolve) => setTimeout(resolve, 400));
  observer.disconnect();

  const bars = Array.from(document.querySelectorAll(".xrf-control-bar"));
  const layoutViolations = [];
  for (const bar of bars) {
    const rect = bar.getBoundingClientRect();
    const isDirectChildOfArticle = bar.parentElement?.tagName === "ARTICLE";
    const state =
      bar.closest("article")?.getAttribute("data-xrf-state") ?? "unknown";
    if (isDirectChildOfArticle || rect.height > 36) {
      layoutViolations.push({
        state,
        text: bar.textContent,
        width: Math.round(rect.width),
        height: Math.round(rect.height),
        isDirectChildOfArticle,
      });
    }
  }

  let dimmedCount = 0;
  let suspectCount = 0;
  let keptCount = 0;
  let revealedCount = 0;
  let graphqlEnrichedCount = 0;
  for (const article of Array.from(document.querySelectorAll("article[data-xrf-state]"))) {
    const state = article.getAttribute("data-xrf-state");
    if (state === "dimmed") {
      dimmedCount += 1;
    } else if (state === "suspect") {
      suspectCount += 1;
    } else if (state === "clean") {
      keptCount += 1;
    } else if (state === "revealed") {
      revealedCount += 1;
    }
    const bar = article.querySelector(".xrf-control-bar");
    const isGraphql =
      article.getAttribute("data-xrf-meta-source") === "graphql" ||
      /\b(?:metaSource|meta)\s*=\s*(?:graphql|GQL)\b/i.test(bar?.getAttribute("title") ?? "");
    if (isGraphql) {
      graphqlEnrichedCount += 1;
    }
  }

  const cells = Array.from(document.querySelectorAll('[data-testid="cellInnerDiv"]'));
  let seenCutoff = false;
  let scoredBelowCutoff = 0;
  for (const cell of cells) {
    const hasTweet = cell.querySelector('article[data-testid="tweet"]') !== null;
    if (!hasTweet) {
      const headings = Array.from(cell.querySelectorAll('h2, [role="heading"], span'));
      if (
        headings.some((h) => {
          const text = (h.textContent ?? "").trim();
          return text.length > 0 && text.length <= 40 && cutoffRegexes.some((rx) => rx.test(text));
        })
      ) {
        seenCutoff = true;
      }
    } else if (seenCutoff) {
      const article = cell.querySelector('article[data-testid="tweet"]');
      if (article?.getAttribute("data-xrf-state") !== null) {
        scoredBelowCutoff += 1;
      }
    }
  }

  return {
    ok:
      xrfMutationsInWindow === 0 &&
      layoutViolations.length === 0 &&
      scoredBelowCutoff === 0,
    url: window.location.href,
    barCount: bars.length,
    dimmedCount,
    suspectCount,
    keptCount,
    revealedCount,
    graphqlEnrichedCount,
    xrfMutationsInWindow,
    layoutViolations,
    seenCutoff,
    scoredBelowCutoff,
  };
}
