import { SCROLL_DEBOUNCE_MS } from "../domain/types.ts";
import type { OffscreenRpcRequest, OffscreenRpcResponse } from "../shared/rpc.ts";
import { XThreadDomAdapter } from "./dom-adapter.ts";

export interface ThreadObserverHandle {
  readonly flushNow: () => Promise<void>;
  readonly dispose: () => void;
}

function isXrfInternalNode(node: Node | null): boolean {
  let curr: Node | null = node;
  while (curr !== null) {
    if (curr instanceof Element) {
      if (curr.classList.contains("xrf-control-bar") || curr.id === "xrf-injected-styles") {
        return true;
      }
    }
    curr = curr.parentNode;
  }
  return false;
}

function isExtensionContextValid(): boolean {
  if (typeof chrome === "undefined" || !chrome.runtime) {
    return true;
  }
  return Boolean(chrome.runtime.id);
}

export async function runScrollPass(
  doc: Document,
  pathname: string,
  adapter: XThreadDomAdapter,
  sendRpc: (req: OffscreenRpcRequest) => Promise<OffscreenRpcResponse>,
): Promise<void> {
  if (!isExtensionContextValid()) {
    return;
  }
  const snapshot = adapter.extractThreadSnapshot(doc, pathname);
  if (snapshot === null || snapshot.candidates.length === 0 || !snapshot.hasUnscoredReplies) {
    return;
  }

  let response: OffscreenRpcResponse;
  try {
    response = await sendRpc({
      kind: "evaluate_batch",
      threadId: snapshot.threadId,
      candidates: snapshot.candidates,
      target: "background",
    });
  } catch {
    return;
  }

  if (response.kind !== "evaluate_batch_ok") {
    return;
  }

  const handleFeedback = (
    tweetId: Parameters<typeof adapter.applyVerdictsInPlace>[2] extends (
      id: infer T,
      l: "spam" | "human",
    ) => void
      ? T
      : never,
    label: "spam" | "human",
  ): void => {
    if (!isExtensionContextValid()) {
      return;
    }
    void sendRpc({
      kind: "record_feedback",
      threadId: snapshot.threadId,
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
          adapter.applyVerdictsInPlace(doc, updates, handleFeedback);
        }
      })
      .catch(() => {});
  };

  const allUpdates = [...response.result.verdicts, ...response.result.retroactiveUpdates];
  adapter.applyVerdictsInPlace(doc, allUpdates, handleFeedback);
}

export function mountThreadObserver(
  doc: Document,
  sendRpc: (req: OffscreenRpcRequest) => Promise<OffscreenRpcResponse>,
  getPathname: () => string = () => window.location.pathname,
  debounceMs: number = SCROLL_DEBOUNCE_MS,
): ThreadObserverHandle {
  const adapter = new XThreadDomAdapter();
  let timer: ReturnType<typeof setTimeout> | null = null;

  const schedulePass = (): void => {
    if (!isExtensionContextValid()) {
      observer.disconnect();
      return;
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
      void runScrollPass(doc, getPathname(), adapter, sendRpc);
    }, debounceMs);
  };

  const observer = new MutationObserver((mutations) => {
    if (!isExtensionContextValid()) {
      observer.disconnect();
      return;
    }
    const hasExternalMutation = mutations.some((m) => !isXrfInternalNode(m.target));
    if (hasExternalMutation) {
      schedulePass();
    }
  });

  observer.observe(doc.body ?? doc.documentElement, {
    childList: true,
    subtree: true,
  });

  schedulePass();

  return {
    async flushNow(): Promise<void> {
      if (timer !== null) {
        clearTimeout(timer);
        timer = null;
      }
      await runScrollPass(doc, getPathname(), adapter, sendRpc);
    },
    dispose(): void {
      if (timer !== null) {
        clearTimeout(timer);
        timer = null;
      }
      observer.disconnect();
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
