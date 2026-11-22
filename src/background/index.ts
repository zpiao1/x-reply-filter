import type { ThreadId } from "../domain/types.ts";
import {
  type OffscreenChromeApi,
  type OffscreenRpcRequestV2,
  type OffscreenRpcResponse,
  type Stage2VerdictUpdateMessage,
  ensureOffscreenDocument,
  parseRpcRequestV2,
  parseStage2VerdictUpdate,
} from "../shared/rpc.ts";

const OFFSCREEN_DOCUMENT_PATH = "offscreen.html" as const;
const X_TAB_URL_PATTERNS = ["https://x.com/*", "https://twitter.com/*"] as const;
const MAX_TRACKED_THREADS = 64 as const;

export interface SettingsChangedBroadcastMessage {
  readonly kind: "xrf_settings_changed";
}

export interface BackgroundTabInfo {
  readonly id?: number | undefined;
  readonly url?: string | undefined;
}

export interface BackgroundTabsApi {
  readonly query: (queryInfo: {
    readonly url: string[];
  }) => Promise<ReadonlyArray<BackgroundTabInfo>>;
  readonly sendMessage: (tabId: number, message: unknown) => Promise<unknown>;
}

export interface BackgroundScriptingApi {
  readonly executeScript: (injection: {
    readonly target: { readonly tabId: number };
    readonly files: string[];
    readonly world?: "MAIN" | "ISOLATED";
  }) => Promise<unknown>;
}

export function shouldBroadcastSettingsChanged(
  req: OffscreenRpcRequestV2,
  res: OffscreenRpcResponse,
): boolean {
  if (req.kind === "update_settings" && res.kind === "settings_ok") {
    return true;
  }
  if (req.kind === "undo_feedback" && res.kind === "undo_feedback_ok" && res.undone) {
    return true;
  }
  if (req.kind === "import_feedback" && res.kind === "import_feedback_ok") {
    return true;
  }
  if (req.kind === "reset_feedback" && res.kind === "reset_feedback_ok") {
    return true;
  }
  return false;
}

export async function broadcastSettingsChangedToXTabs(
  tabsApi: BackgroundTabsApi,
): Promise<number> {
  let delivered = 0;
  try {
    const tabs = await tabsApi.query({ url: [...X_TAB_URL_PATTERNS] });
    const msg: SettingsChangedBroadcastMessage = { kind: "xrf_settings_changed" };
    await Promise.all(
      tabs.map(async (tab) => {
        if (tab.id === undefined) return;
        try {
          await tabsApi.sendMessage(tab.id, msg);
          delivered += 1;
        } catch {
          // Tab may not have content script ready yet
        }
      }),
    );
  } catch {}
  return delivered;
}

export async function reinjectOpenXTabs(
  tabsApi: BackgroundTabsApi,
  scriptingApi: BackgroundScriptingApi,
): Promise<number> {
  let injectedTabs = 0;
  try {
    const tabs = await tabsApi.query({ url: [...X_TAB_URL_PATTERNS] });
    await Promise.all(
      tabs.map(async (tab) => {
        if (tab.id === undefined) return;
        try {
          await scriptingApi.executeScript({
            target: { tabId: tab.id },
            files: ["graphql-tap.js"],
            world: "MAIN",
          });
        } catch {}
        try {
          await scriptingApi.executeScript({
            target: { tabId: tab.id },
            files: ["content.js"],
          });
          injectedTabs += 1;
        } catch {}
      }),
    );
  } catch {}
  return injectedTabs;
}

export async function forwardRpcToOffscreen(
  rawMessage: unknown,
  offscreenApi: OffscreenChromeApi,
  sendMessageFn: (req: OffscreenRpcRequestV2) => Promise<OffscreenRpcResponse>,
): Promise<OffscreenRpcResponse | null> {
  const parsed = parseRpcRequestV2(rawMessage);
  if (parsed === null || parsed.target === "offscreen") {
    return null;
  }

  try {
    await ensureOffscreenDocument(OFFSCREEN_DOCUMENT_PATH, offscreenApi);
    const forwarded: OffscreenRpcRequestV2 = {
      ...parsed,
      target: "offscreen",
    };
    return await sendMessageFn(forwarded);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return {
      kind: "rpc_error",
      message,
    };
  }
}

export class BackgroundMessageRouter {
  private readonly threadTabs = new Map<ThreadId, Set<number>>();
  private readonly offscreenApi: OffscreenChromeApi;
  private readonly sendToOffscreen: (req: OffscreenRpcRequestV2) => Promise<OffscreenRpcResponse>;
  private readonly tabsApi: BackgroundTabsApi | undefined;

  constructor(
    offscreenApi: OffscreenChromeApi,
    sendToOffscreen: (req: OffscreenRpcRequestV2) => Promise<OffscreenRpcResponse>,
    tabsApi?: BackgroundTabsApi,
  ) {
    this.offscreenApi = offscreenApi;
    this.sendToOffscreen = sendToOffscreen;
    this.tabsApi = tabsApi;
  }

  public trackThreadTab(threadId: ThreadId, tabId: number): void {
    const existing = this.threadTabs.get(threadId);
    if (existing !== undefined) {
      this.threadTabs.delete(threadId);
      existing.add(tabId);
      this.threadTabs.set(threadId, existing);
      return;
    }

    if (this.threadTabs.size >= MAX_TRACKED_THREADS) {
      const oldest = this.threadTabs.keys().next().value;
      if (oldest !== undefined) {
        this.threadTabs.delete(oldest);
      }
    }
    this.threadTabs.set(threadId, new Set<number>([tabId]));
  }

  public removeTab(tabId: number): void {
    for (const [threadId, tabs] of this.threadTabs.entries()) {
      tabs.delete(tabId);
      if (tabs.size === 0) {
        this.threadTabs.delete(threadId);
      }
    }
  }

  public getTrackedTabs(threadId: ThreadId): ReadonlyArray<number> {
    return Array.from(this.threadTabs.get(threadId) ?? []);
  }

  public async relayStage2Update(update: Stage2VerdictUpdateMessage): Promise<number> {
    if (this.tabsApi === undefined) {
      return 0;
    }

    const tracked = this.threadTabs.get(update.threadId);
    let targetTabIds: number[] = tracked ? Array.from(tracked) : [];

    // Fallback across MV3 service-worker restarts: match open X status tabs by threadId
    if (targetTabIds.length === 0) {
      try {
        const openTabs = await this.tabsApi.query({ url: [...X_TAB_URL_PATTERNS] });
        const statusNeedle = `/status/${update.threadId}`;
        targetTabIds = openTabs
          .filter((t) => t.id !== undefined && typeof t.url === "string" && t.url.includes(statusNeedle))
          .map((t) => t.id!);
      } catch {}
    }

    if (targetTabIds.length === 0) {
      return 0;
    }

    const outgoing: Stage2VerdictUpdateMessage = {
      kind: "stage2_verdict_update",
      threadId: update.threadId,
      tweetId: update.tweetId,
      verdict: update.verdict,
      target: "content",
    };

    let delivered = 0;
    await Promise.all(
      targetTabIds.map(async (tabId) => {
        try {
          await this.tabsApi!.sendMessage(tabId, outgoing);
          delivered += 1;
        } catch {
          // Prune closed/unreachable tab
          tracked?.delete(tabId);
          if (tracked && tracked.size === 0) {
            this.threadTabs.delete(update.threadId);
          }
        }
      }),
    );

    return delivered;
  }

  public async handleMessage(
    rawMessage: unknown,
    senderTabId?: number,
  ): Promise<OffscreenRpcResponse | null> {
    // 1. Relay Stage-2 async verdict updates from offscreen -> matching X tabs
    const stage2Update = parseStage2VerdictUpdate(rawMessage);
    if (stage2Update !== null) {
      if (senderTabId === undefined) {
        await this.relayStage2Update(stage2Update);
      }
      return null;
    }

    // 2. Forward RPC v2 requests to offscreen document
    const parsed = parseRpcRequestV2(rawMessage);
    if (parsed === null || parsed.target === "offscreen") {
      return null;
    }

    if (parsed.kind === "evaluate_batch" && senderTabId !== undefined) {
      this.trackThreadTab(parsed.threadId, senderTabId);
    }

    const res = await forwardRpcToOffscreen(parsed, this.offscreenApi, this.sendToOffscreen);
    if (res !== null && this.tabsApi !== undefined && shouldBroadcastSettingsChanged(parsed, res)) {
      await broadcastSettingsChangedToXTabs(this.tabsApi);
    }
    return res;
  }
}

export function registerBackgroundRouter(): BackgroundMessageRouter {
  const tabsApi: BackgroundTabsApi | undefined = chrome.tabs
    ? {
        query: (queryInfo) => chrome.tabs.query(queryInfo),
        sendMessage: (tabId, message) => chrome.tabs.sendMessage(tabId, message),
      }
    : undefined;

  const router = new BackgroundMessageRouter(
    chrome.offscreen as unknown as OffscreenChromeApi,
    (req) => chrome.runtime.sendMessage(req) as Promise<OffscreenRpcResponse>,
    tabsApi,
  );

  chrome.runtime.onMessage.addListener((rawMessage, sender, sendResponse) => {
    const stage2Update = parseStage2VerdictUpdate(rawMessage);
    if (stage2Update !== null) {
      if (sender.tab?.id === undefined) {
        void router.relayStage2Update(stage2Update);
      }
      return undefined;
    }

    const parsed = parseRpcRequestV2(rawMessage);
    if (parsed === null || parsed.target === "offscreen") {
      return undefined;
    }

    void router.handleMessage(parsed, sender.tab?.id).then((res) => {
      if (res !== null) {
        sendResponse(res);
      }
    });

    return true;
  });

  if (chrome.tabs?.onRemoved) {
    chrome.tabs.onRemoved.addListener((tabId) => {
      router.removeTab(tabId);
    });
  }

  if (chrome.runtime.onInstalled && tabsApi && chrome.scripting) {
    const scriptingApi: BackgroundScriptingApi = {
      executeScript: (injection) => chrome.scripting.executeScript(injection),
    };
    chrome.runtime.onInstalled.addListener(() => {
      void reinjectOpenXTabs(tabsApi, scriptingApi);
    });
  }

  return router;
}

if (typeof chrome !== "undefined" && chrome.runtime?.onMessage && chrome.offscreen) {
  registerBackgroundRouter();
}
