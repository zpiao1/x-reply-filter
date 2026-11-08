import { SCROLL_DEBOUNCE_MS } from "../domain/types.ts";
import type { OffscreenRpcRequest, OffscreenRpcResponse } from "../shared/rpc.ts";
import { XThreadDomAdapter } from "./dom-adapter.ts";

export interface ThreadObserverHandle {
  readonly flushNow: () => Promise<void>;
  readonly dispose: () => void;
}

export async function runScrollPass(
  doc: Document,
  pathname: string,
  adapter: XThreadDomAdapter,
  sendRpc: (req: OffscreenRpcRequest) => Promise<OffscreenRpcResponse>,
): Promise<void> {
  const snapshot = adapter.extractThreadSnapshot(doc, pathname);
  if (snapshot === null || snapshot.candidates.length === 0) {
    return;
  }

  const response = await sendRpc({
    kind: "evaluate_batch",
    threadId: snapshot.threadId,
    candidates: snapshot.candidates,
    target: "background",
  });

  if (response.kind !== "evaluate_batch_ok") {
    return;
  }

  const allUpdates = [...response.result.verdicts, ...response.result.retroactiveUpdates];
  adapter.applyVerdictsInPlace(doc, allUpdates, (tweetId, label) => {
    void sendRpc({
      kind: "record_feedback",
      threadId: snapshot.threadId,
      tweetId,
      label,
      target: "background",
    }).then((feedbackRes) => {
      if (feedbackRes.kind === "record_feedback_ok" && feedbackRes.updatedVerdict !== null) {
        adapter.applyVerdictsInPlace(doc, [feedbackRes.updatedVerdict], () => {});
      }
    });
  });
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
    if (timer !== null) {
      clearTimeout(timer);
    }
    timer = setTimeout(() => {
      timer = null;
      void runScrollPass(doc, getPathname(), adapter, sendRpc);
    }, debounceMs);
  };

  const observer = new MutationObserver(() => {
    schedulePass();
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

if (typeof chrome !== "undefined" && chrome.runtime?.sendMessage && typeof document !== "undefined") {
  const sendRpc = (req: OffscreenRpcRequest): Promise<OffscreenRpcResponse> =>
    chrome.runtime.sendMessage(req) as Promise<OffscreenRpcResponse>;
  mountThreadObserver(document, sendRpc);
}
