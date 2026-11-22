import {
  type GraphqlTweetMeta,
  SCROLL_DEBOUNCE_MS,
  type ThreadId,
  type TweetId,
  parseThreadId,
} from "../domain/types.ts";
import {
  type OffscreenRpcRequest,
  type OffscreenRpcResponse,
  parseGraphqlTweetMeta,
  parseStage2VerdictUpdate,
} from "../shared/rpc.ts";
import { XThreadDomAdapter } from "./dom-adapter.ts";
import {
  GRAPHQL_CONTENT_SOURCE,
  GRAPHQL_TAP_PROTOCOL_VERSION,
  GRAPHQL_TAP_SOURCE,
} from "./graphql-bridge.ts";

export const MAX_GQL_CACHE = 1000 as const;
const MAX_EVALUATED_TRACKED = 2000 as const;
const ALLOWED_X_ORIGINS: ReadonlySet<string> = new Set([
  "https://x.com",
  "https://twitter.com",
]);

export interface ThreadObserverHandle {
  readonly flushNow: () => Promise<void>;
  readonly dispose: () => void;
}

export interface ScrollPassOptions {
  readonly graphqlCache?: ReadonlyMap<TweetId, GraphqlTweetMeta> | undefined;
  readonly evaluatedWithGraphql?: Map<TweetId, boolean> | undefined;
  readonly forceReevaluate?: boolean | undefined;
}

export interface ThreadObserverOptions {
  readonly win?: Window | undefined;
  readonly graphqlCache?: BoundedGraphqlCache | undefined;
  readonly runtimeOnMessage?:
    | {
        readonly addListener: (cb: (message: unknown) => void) => void;
        readonly removeListener: (cb: (message: unknown) => void) => void;
      }
    | undefined;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isXrfInternalNode(node: Node | null): boolean {
  let curr: Node | null = node;
  while (curr !== null) {
    if (curr.nodeType === 1) {
      const el = curr as Element;
      if (el.classList?.contains("xrf-control-bar") || el.id === "xrf-injected-styles") {
        return true;
      }
    }
    curr = curr.parentNode;
  }
  return false;
}

function isXrfOnlyMutation(mutation: MutationRecord): boolean {
  if (isXrfInternalNode(mutation.target)) {
    return true;
  }
  if (mutation.type === "childList") {
    const added = Array.from(mutation.addedNodes);
    const removed = Array.from(mutation.removedNodes);
    if (added.length + removed.length > 0) {
      return added.every(isXrfInternalNode) && removed.every(isXrfInternalNode);
    }
  }
  return false;
}

function isExtensionContextValid(): boolean {
  if (typeof chrome === "undefined" || !chrome.runtime) {
    return true;
  }
  return Boolean(chrome.runtime.id);
}

/**
 * Bounded insertion-order LRU cache holding parsed GraphQL metadata keyed by TweetId.
 */
export class BoundedGraphqlCache {
  private readonly map = new Map<TweetId, GraphqlTweetMeta>();
  private readonly maxEntries: number;

  constructor(maxEntries: number = MAX_GQL_CACHE) {
    this.maxEntries = maxEntries;
  }

  public get size(): number {
    return this.map.size;
  }

  public asMap(): ReadonlyMap<TweetId, GraphqlTweetMeta> {
    return this.map;
  }

  public get(tweetId: TweetId): GraphqlTweetMeta | undefined {
    const existing = this.map.get(tweetId);
    if (existing !== undefined) {
      this.map.delete(tweetId);
      this.map.set(tweetId, existing);
    }
    return existing;
  }

  public set(tweetId: TweetId, meta: GraphqlTweetMeta): void {
    if (this.map.has(tweetId)) {
      this.map.delete(tweetId);
    } else if (this.map.size >= this.maxEntries) {
      const oldest = this.map.keys().next().value;
      if (oldest !== undefined) {
        this.map.delete(oldest);
      }
    }
    this.map.set(tweetId, meta);
  }

  public ingestMessageData(data: unknown): TweetId[] {
    if (
      !isRecord(data) ||
      data.source !== GRAPHQL_TAP_SOURCE ||
      data.v !== GRAPHQL_TAP_PROTOCOL_VERSION ||
      !Array.isArray(data.items)
    ) {
      return [];
    }

    const ingested: TweetId[] = [];
    for (const rawItem of data.items) {
      const parsed = parseGraphqlTweetMeta(rawItem);
      if (parsed === null || parsed.tweetId === undefined) {
        continue;
      }
      this.set(parsed.tweetId, parsed);
      ingested.push(parsed.tweetId);
    }
    return ingested;
  }
}

export function createFeedbackHandler(
  doc: Document,
  threadId: ThreadId,
  adapter: XThreadDomAdapter,
  sendRpc: (req: OffscreenRpcRequest) => Promise<OffscreenRpcResponse>,
): (tweetId: TweetId, label: "spam" | "human") => void {
  const onFeedback = (tweetId: TweetId, label: "spam" | "human"): void => {
    if (!isExtensionContextValid()) {
      return;
    }
    void sendRpc({
      kind: "record_feedback",
      threadId,
      tweetId,
      label,
      target: "background",
    })
      .then((feedbackRes) => {
        if (feedbackRes.kind === "record_feedback_ok" && feedbackRes.updatedVerdict !== null) {
          const updates = [
            feedbackRes.updatedVerdict,
            ...(feedbackRes.retroactiveUpdates ?? []),
          ];
          adapter.applyVerdictsInPlace(doc, updates, onFeedback);
        }
      })
      .catch(() => {});
  };
  return onFeedback;
}

export function handleRuntimeMessage(
  rawMessage: unknown,
  doc: Document,
  getPathname: () => string,
  adapter: XThreadDomAdapter,
  sendRpc: (req: OffscreenRpcRequest) => Promise<OffscreenRpcResponse>,
  onSettingsChanged?: () => void,
): boolean {
  const stage2Msg = parseStage2VerdictUpdate(rawMessage);
  if (stage2Msg !== null) {
    const activeThreadId = parseThreadId(getPathname());
    if (activeThreadId !== stage2Msg.threadId) {
      return false;
    }
    const onFeedback = createFeedbackHandler(doc, stage2Msg.threadId, adapter, sendRpc);
    adapter.applyVerdictsInPlace(doc, [stage2Msg.verdict], onFeedback);
    return true;
  }

  if (isRecord(rawMessage) && rawMessage.kind === "xrf_settings_changed") {
    onSettingsChanged?.();
    return true;
  }

  return false;
}

export async function runScrollPass(
  doc: Document,
  pathname: string,
  adapter: XThreadDomAdapter,
  sendRpc: (req: OffscreenRpcRequest) => Promise<OffscreenRpcResponse>,
  options?: ScrollPassOptions,
): Promise<void> {
  if (!isExtensionContextValid()) {
    return;
  }
  const snapshot = adapter.extractThreadSnapshot(doc, pathname, options?.graphqlCache);
  if (snapshot === null || snapshot.candidates.length === 0) {
    return;
  }

  const evaluatedMap = options?.evaluatedWithGraphql;
  const hasNewlyEnriched =
    evaluatedMap !== undefined &&
    snapshot.candidates.some(
      (c) => c.graphqlMeta != null && evaluatedMap.get(c.tweetId) === false,
    );

  if (!options?.forceReevaluate && !snapshot.hasUnscoredReplies && !hasNewlyEnriched) {
    return;
  }

  let response: OffscreenRpcResponse;
  try {
    response = await sendRpc({
      kind: "evaluate_batch",
      threadId: snapshot.threadId,
      candidates: snapshot.candidates,
      ...(snapshot.focalAuthorId != null ? { focalAuthorId: snapshot.focalAuthorId } : {}),
      target: "background",
    });
  } catch {
    return;
  }

  if (response.kind !== "evaluate_batch_ok") {
    return;
  }

  if (evaluatedMap !== undefined) {
    for (const candidate of snapshot.candidates) {
      if (!evaluatedMap.has(candidate.tweetId) && evaluatedMap.size >= MAX_EVALUATED_TRACKED) {
        const oldest = evaluatedMap.keys().next().value;
        if (oldest !== undefined) {
          evaluatedMap.delete(oldest);
        }
      }
      evaluatedMap.set(candidate.tweetId, candidate.graphqlMeta != null);
    }
  }

  const handleFeedback = createFeedbackHandler(doc, snapshot.threadId, adapter, sendRpc);
  const allUpdates = [...response.result.verdicts, ...response.result.retroactiveUpdates];
  adapter.applyVerdictsInPlace(doc, allUpdates, handleFeedback);
}

export function mountThreadObserver(
  doc: Document,
  sendRpc: (req: OffscreenRpcRequest) => Promise<OffscreenRpcResponse>,
  getPathname: () => string = () => window.location.pathname,
  debounceMs: number = SCROLL_DEBOUNCE_MS,
  options?: ThreadObserverOptions,
): ThreadObserverHandle {
  const adapter = new XThreadDomAdapter();
  const graphqlCache = options?.graphqlCache ?? new BoundedGraphqlCache(MAX_GQL_CACHE);
  const evaluatedWithGraphql = new Map<TweetId, boolean>();
  const win = options?.win ?? (typeof window !== "undefined" ? window : undefined);
  const runtimeOnMessage =
    options?.runtimeOnMessage ??
    (typeof chrome !== "undefined" && chrome.runtime?.onMessage
      ? chrome.runtime.onMessage
      : undefined);

  let timer: ReturnType<typeof setTimeout> | null = null;
  let pendingForceReevaluate = false;
  let inFlightPass: Promise<void> | null = null;
  let rerunAfterInFlight = false;

  const executePass = async (force: boolean): Promise<void> => {
    if (inFlightPass !== null) {
      if (force) {
        pendingForceReevaluate = true;
      }
      rerunAfterInFlight = true;
      await inFlightPass;
      return;
    }

    const run = async (initialForce: boolean): Promise<void> => {
      let nextForce = initialForce;
      do {
        rerunAfterInFlight = false;
        await runScrollPass(doc, getPathname(), adapter, sendRpc, {
          graphqlCache: graphqlCache.asMap(),
          evaluatedWithGraphql,
          forceReevaluate: nextForce,
        });
        nextForce = pendingForceReevaluate;
        pendingForceReevaluate = false;
      } while (rerunAfterInFlight);
    };

    inFlightPass = run(force).finally(() => {
      inFlightPass = null;
    });
    await inFlightPass;
  };

  const schedulePass = (force = false): void => {
    if (!isExtensionContextValid()) {
      observer.disconnect();
      return;
    }
    if (force) {
      pendingForceReevaluate = true;
    }
    if (timer !== null) {
      clearTimeout(timer);
    }
    timer = setTimeout(() => {
      timer = null;
      if (!isExtensionContextValid()) {
        observer.disconnect();
        return;
      }
      const runForce = pendingForceReevaluate;
      pendingForceReevaluate = false;
      void executePass(runForce);
    }, debounceMs);
  };

  const onWindowMessage = (event: MessageEvent): void => {
    if (!isExtensionContextValid()) {
      win?.removeEventListener("message", onWindowMessage);
      return;
    }
    if (win === undefined || event.source !== win || !ALLOWED_X_ORIGINS.has(event.origin)) {
      return;
    }
    const ingested = graphqlCache.ingestMessageData(event.data);
    if (ingested.length === 0) {
      return;
    }
    const needsEnrichment = ingested.some((tid) => {
      if (evaluatedWithGraphql.get(tid) === false) {
        return true;
      }
      const article = doc.querySelector(`article[data-xrf-tweet-id="${tid}"]`);
      return article !== null && evaluatedWithGraphql.get(tid) !== true;
    });
    if (needsEnrichment) {
      schedulePass(false);
    }
  };

  if (win !== undefined && typeof win.addEventListener === "function") {
    win.addEventListener("message", onWindowMessage);
    try {
      const origin = win.location?.origin;
      if (typeof origin === "string" && ALLOWED_X_ORIGINS.has(origin)) {
        win.postMessage({ source: GRAPHQL_CONTENT_SOURCE, type: "tap_ready" }, origin);
      }
    } catch {
      // Ignore postMessage failure in restricted environments
    }
  }

  const onRuntimeMessage = (rawMessage: unknown): void => {
    if (!isExtensionContextValid()) {
      return;
    }
    handleRuntimeMessage(rawMessage, doc, getPathname, adapter, sendRpc, () => {
      schedulePass(true);
    });
  };

  if (runtimeOnMessage !== undefined) {
    runtimeOnMessage.addListener(onRuntimeMessage);
  }

  const ObserverCtor =
    (win as unknown as { readonly MutationObserver?: typeof MutationObserver } | undefined)
      ?.MutationObserver ??
    (typeof MutationObserver !== "undefined" ? MutationObserver : undefined);

  const observer: Pick<MutationObserver, "disconnect"> =
    ObserverCtor !== undefined
      ? new ObserverCtor((mutations) => {
          if (!isExtensionContextValid()) {
            observer.disconnect();
            return;
          }
          const hasExternalMutation = mutations.some((m) => !isXrfOnlyMutation(m));
          if (hasExternalMutation) {
            schedulePass(false);
          }
        })
      : { disconnect: () => {} };

  if ("observe" in observer && typeof observer.observe === "function") {
    observer.observe(doc.body ?? doc.documentElement, {
      childList: true,
      subtree: true,
    });
  }

  schedulePass(false);

  return {
    async flushNow(): Promise<void> {
      if (timer !== null) {
        clearTimeout(timer);
        timer = null;
      }
      const runForce = pendingForceReevaluate;
      pendingForceReevaluate = false;
      await executePass(runForce);
    },
    dispose(): void {
      if (timer !== null) {
        clearTimeout(timer);
        timer = null;
      }
      observer.disconnect();
      if (win !== undefined && typeof win.removeEventListener === "function") {
        win.removeEventListener("message", onWindowMessage);
      }
      if (runtimeOnMessage !== undefined) {
        runtimeOnMessage.removeListener(onRuntimeMessage);
      }
    },
  };
}

if (
  typeof chrome !== "undefined" &&
  typeof chrome.runtime?.sendMessage === "function" &&
  typeof document !== "undefined"
) {
  for (const article of Array.from(document.querySelectorAll("article[data-xrf-state]"))) {
    article.removeAttribute("data-xrf-state");
    article.removeAttribute("data-xrf-reason");
  }
  const sendRpc = (req: OffscreenRpcRequest): Promise<OffscreenRpcResponse> =>
    chrome.runtime.sendMessage(req) as Promise<OffscreenRpcResponse>;
  mountThreadObserver(document, sendRpc);
}
