import { type FilterSettings, type ThreadId, parseThreadId } from "../domain/types.ts";
import {
  type OffscreenChromeApi,
  type OffscreenRpcRequestV2,
  type OffscreenRpcResponse,
  type Stage2VerdictUpdateMessage,
  ensureOffscreenDocument,
  parseFilterSettings,
  parseRpcRequestV2,
  parseStage2VerdictUpdate,
} from "../shared/rpc.ts";

const OFFSCREEN_DOCUMENT_PATH = "offscreen.html" as const;
const X_TAB_URL_PATTERNS = ["https://x.com/*", "https://twitter.com/*"] as const;
const MAX_TRACKED_THREADS = 64 as const;

export const SETTINGS_SYNC_STORAGE_KEY = "xrf_settings_v2" as const;
export const THREAD_TABS_SESSION_KEY = "xrf_thread_tabs_v1" as const;

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

export interface BackgroundStorageArea {
  readonly get: (keys: string | string[]) => Promise<Record<string, unknown>>;
  readonly set: (items: Record<string, unknown>) => Promise<void>;
}

export interface BackgroundStorageChange {
  readonly oldValue?: unknown;
  readonly newValue?: unknown;
}

export interface BackgroundRouterOptions {
  readonly syncStorage?: BackgroundStorageArea | undefined;
  readonly sessionStorage?: BackgroundStorageArea | undefined;
}

export function shouldReinjectOnInstalled(reason: string): boolean {
  return reason === "install" || reason === "update";
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

function areSettingsIdentical(a: FilterSettings | null, b: FilterSettings): boolean {
  if (a === null) return false;
  return (
    a.preset === b.preset &&
    a.revision === b.revision &&
    a.toggles.collapseBlueCheckReplyGuys === b.toggles.collapseBlueCheckReplyGuys &&
    a.toggles.collapseMediaOnlyReplies === b.toggles.collapseMediaOnlyReplies &&
    a.toggles.collapseBotSummons === b.toggles.collapseBotSummons &&
    a.toggles.showSuspectBadges === b.toggles.showSuspectBadges &&
    a.toggles.learnFromFeedback === b.toggles.learnFromFeedback &&
    a.toggles.enableStage2 === b.toggles.enableStage2
  );
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
  private readonly syncStorage: BackgroundStorageArea | undefined;
  private readonly sessionStorage: BackgroundStorageArea | undefined;

  private settingsSynced = false;
  private settingsSyncPromise: Promise<void> | null = null;
  private lastSyncedSettings: FilterSettings | null = null;

  private threadTabsHydrated = false;
  private threadTabsHydratePromise: Promise<void> | null = null;

  constructor(
    offscreenApi: OffscreenChromeApi,
    sendToOffscreen: (req: OffscreenRpcRequestV2) => Promise<OffscreenRpcResponse>,
    tabsApi?: BackgroundTabsApi,
    options?: BackgroundRouterOptions,
  ) {
    this.offscreenApi = offscreenApi;
    this.sendToOffscreen = sendToOffscreen;
    this.tabsApi = tabsApi;
    this.syncStorage = options?.syncStorage;
    this.sessionStorage = options?.sessionStorage;
  }

  private persistThreadTabsToSession(): void {
    if (this.sessionStorage === undefined) {
      return;
    }
    const serialized: Record<string, number[]> = {};
    for (const [threadId, tabSet] of this.threadTabs.entries()) {
      if (tabSet.size > 0) {
        serialized[threadId] = Array.from(tabSet);
      }
    }
    void this.sessionStorage.set({ [THREAD_TABS_SESSION_KEY]: serialized }).catch(() => {});
  }

  private async ensureThreadTabsHydrated(): Promise<void> {
    if (this.threadTabsHydrated || this.sessionStorage === undefined) {
      this.threadTabsHydrated = true;
      return;
    }
    if (this.threadTabsHydratePromise !== null) {
      return this.threadTabsHydratePromise;
    }

    this.threadTabsHydratePromise = (async () => {
      try {
        const stored = await this.sessionStorage!.get(THREAD_TABS_SESSION_KEY);
        const raw = stored[THREAD_TABS_SESSION_KEY];
        if (typeof raw === "object" && raw !== null) {
          for (const [rawThreadId, rawTabIds] of Object.entries(raw as Record<string, unknown>)) {
            const threadId = parseThreadId(rawThreadId);
            if (threadId === null || !Array.isArray(rawTabIds)) continue;
            for (const tabId of rawTabIds) {
              if (typeof tabId === "number" && Number.isInteger(tabId) && tabId >= 0) {
                this.insertThreadTabMemory(threadId, tabId);
              }
            }
          }
        }
      } catch {}
      this.threadTabsHydrated = true;
    })().finally(() => {
      this.threadTabsHydratePromise = null;
    });

    return this.threadTabsHydratePromise;
  }

  private insertThreadTabMemory(threadId: ThreadId, tabId: number): void {
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

  public trackThreadTab(threadId: ThreadId, tabId: number): void {
    this.insertThreadTabMemory(threadId, tabId);
    this.persistThreadTabsToSession();
  }

  private deleteTabFromMemory(tabId: number): boolean {
    let changed = false;
    for (const [threadId, tabs] of this.threadTabs.entries()) {
      if (tabs.delete(tabId)) {
        changed = true;
      }
      if (tabs.size === 0) {
        this.threadTabs.delete(threadId);
      }
    }
    return changed;
  }

  public removeTab(tabId: number): void {
    const changed = this.deleteTabFromMemory(tabId);
    if (changed) {
      this.persistThreadTabsToSession();
    } else if (!this.threadTabsHydrated && this.sessionStorage !== undefined) {
      void this.ensureThreadTabsHydrated().then(() => {
        if (this.deleteTabFromMemory(tabId)) {
          this.persistThreadTabsToSession();
        }
      });
    }
  }

  public getTrackedTabs(threadId: ThreadId): ReadonlyArray<number> {
    return Array.from(this.threadTabs.get(threadId) ?? []);
  }

  /**
   * Reconciles chrome.storage.sync (`xrf_settings_v2`) with offscreen IDB cache on SW boot
   * (single-flight; higher revision wins, equal-revision divergence converges to sync).
   * Only marks `settingsSynced = true` when offscreen actually returned `settings_ok`.
   */
  public async syncSettingsOnBoot(): Promise<void> {
    if (this.settingsSynced || this.syncStorage === undefined) {
      if (this.syncStorage === undefined) {
        this.settingsSynced = true;
      }
      return;
    }
    if (this.settingsSyncPromise !== null) {
      return this.settingsSyncPromise;
    }

    this.settingsSyncPromise = (async () => {
      try {
        const stored = await this.syncStorage!.get(SETTINGS_SYNC_STORAGE_KEY);
        const syncSettings = parseFilterSettings(stored[SETTINGS_SYNC_STORAGE_KEY]);

        const offscreenRes = await forwardRpcToOffscreen(
          { kind: "get_settings" },
          this.offscreenApi,
          this.sendToOffscreen,
        );
        const offscreenSettings =
          offscreenRes !== null && offscreenRes.kind === "settings_ok"
            ? offscreenRes.settings
            : null;

        if (syncSettings !== null) {
          // Guard against a fresher revision already committed concurrently
          if (
            this.lastSyncedSettings !== null &&
            this.lastSyncedSettings.revision > syncSettings.revision
          ) {
            if (offscreenSettings !== null) {
              this.settingsSynced = true;
            }
            return;
          }

          if (
            offscreenSettings === null ||
            syncSettings.revision > offscreenSettings.revision ||
            (syncSettings.revision === offscreenSettings.revision &&
              !areSettingsIdentical(syncSettings, offscreenSettings))
          ) {
            const pushed = await forwardRpcToOffscreen(
              { kind: "update_settings", settings: syncSettings },
              this.offscreenApi,
              this.sendToOffscreen,
            );
            if (pushed !== null && pushed.kind === "settings_ok") {
              this.lastSyncedSettings = pushed.settings;
              this.settingsSynced = true;
              // If equal-revision divergence bumped offscreen revision, converge syncStorage too
              if (pushed.settings.revision > syncSettings.revision) {
                await this.syncStorage!.set({ [SETTINGS_SYNC_STORAGE_KEY]: pushed.settings });
              }
              if (this.tabsApi !== undefined) {
                await broadcastSettingsChangedToXTabs(this.tabsApi);
              }
            }
          } else if (offscreenSettings.revision > syncSettings.revision) {
            this.lastSyncedSettings = offscreenSettings;
            this.settingsSynced = true;
            await this.syncStorage!.set({ [SETTINGS_SYNC_STORAGE_KEY]: offscreenSettings });
          } else {
            this.lastSyncedSettings = syncSettings;
            this.settingsSynced = true;
          }
        } else if (offscreenSettings !== null) {
          this.lastSyncedSettings = offscreenSettings;
          this.settingsSynced = true;
          if (offscreenSettings.revision > 1) {
            await this.syncStorage!.set({ [SETTINGS_SYNC_STORAGE_KEY]: offscreenSettings });
          }
        }
      } catch (err) {
        console.warn("[x-reply-filter] chrome.storage.sync boot reconciliation failed:", err);
      }
    })().finally(() => {
      this.settingsSyncPromise = null;
    });

    return this.settingsSyncPromise;
  }

  /**
   * Handles cross-device chrome.storage.onChanged updates to `xrf_settings_v2`.
   * Serializes behind any in-flight boot sync so stale boot reads can never clobber newer revisions.
   */
  public async handleSyncStorageChange(
    changes: Readonly<Record<string, BackgroundStorageChange>>,
    areaName: string,
  ): Promise<boolean> {
    if (areaName !== "sync") {
      return false;
    }
    const entry = changes[SETTINGS_SYNC_STORAGE_KEY];
    if (entry === undefined) {
      return false;
    }
    const incoming = parseFilterSettings(entry.newValue);
    if (incoming === null) {
      return false;
    }

    if (this.settingsSyncPromise !== null) {
      await this.settingsSyncPromise;
    }

    if (
      areSettingsIdentical(this.lastSyncedSettings, incoming) ||
      (this.lastSyncedSettings !== null && incoming.revision < this.lastSyncedSettings.revision)
    ) {
      return false;
    }

    const pushed = await forwardRpcToOffscreen(
      { kind: "update_settings", settings: incoming },
      this.offscreenApi,
      this.sendToOffscreen,
    );
    if (pushed !== null && pushed.kind === "settings_ok") {
      this.lastSyncedSettings = pushed.settings;
      this.settingsSynced = true;
      if (this.tabsApi !== undefined) {
        await broadcastSettingsChangedToXTabs(this.tabsApi);
      }
      return true;
    }
    return false;
  }

  public async relayStage2Update(update: Stage2VerdictUpdateMessage): Promise<number> {
    if (this.tabsApi === undefined) {
      return 0;
    }

    await this.ensureThreadTabsHydrated();

    const tracked = this.threadTabs.get(update.threadId);
    const targetTabSet = new Set<number>(tracked ?? []);

    // Union tracked session/memory tabs with open X status tabs matching `/status/<threadId>`
    // so tabs opened before an SW restart never miss Stage-2 updates.
    try {
      const openTabs = await this.tabsApi.query({ url: [...X_TAB_URL_PATTERNS] });
      const statusNeedle = `/status/${update.threadId}`;
      for (const tab of openTabs) {
        if (tab.id !== undefined && typeof tab.url === "string" && tab.url.includes(statusNeedle)) {
          targetTabSet.add(tab.id);
        }
      }
    } catch {}

    if (targetTabSet.size === 0) {
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
    let prunedAny = false;
    await Promise.all(
      Array.from(targetTabSet).map(async (tabId) => {
        try {
          await this.tabsApi!.sendMessage(tabId, outgoing);
          delivered += 1;
        } catch {
          // Prune closed/unreachable tab
          if (tracked?.delete(tabId)) {
            prunedAny = true;
          }
          if (tracked && tracked.size === 0) {
            this.threadTabs.delete(update.threadId);
          }
        }
      }),
    );

    if (prunedAny) {
      this.persistThreadTabsToSession();
    }

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

    // Ensure chrome.storage.sync settings are reconciled before processing requests after SW wake
    if (this.syncStorage !== undefined && !this.settingsSynced) {
      await this.syncSettingsOnBoot();
    }

    if (parsed.kind === "evaluate_batch" && senderTabId !== undefined) {
      await this.ensureThreadTabsHydrated();
      this.trackThreadTab(parsed.threadId, senderTabId);
    }

    const res = await forwardRpcToOffscreen(parsed, this.offscreenApi, this.sendToOffscreen);
    if (res !== null) {
      if (parsed.kind === "update_settings" && res.kind === "settings_ok") {
        this.lastSyncedSettings = res.settings;
        if (this.syncStorage !== undefined) {
          try {
            await this.syncStorage.set({ [SETTINGS_SYNC_STORAGE_KEY]: res.settings });
          } catch (err) {
            console.warn("[x-reply-filter] chrome.storage.sync write failed:", err);
          }
        }
      }
      if (this.tabsApi !== undefined && shouldBroadcastSettingsChanged(parsed, res)) {
        await broadcastSettingsChangedToXTabs(this.tabsApi);
      }
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

  const syncStorage: BackgroundStorageArea | undefined = chrome.storage?.sync
    ? {
        get: (keys) => chrome.storage.sync.get(keys),
        set: (items) => chrome.storage.sync.set(items),
      }
    : undefined;

  const sessionStorage: BackgroundStorageArea | undefined = chrome.storage?.session
    ? {
        get: (keys) => chrome.storage.session.get(keys),
        set: (items) => chrome.storage.session.set(items),
      }
    : undefined;

  const router = new BackgroundMessageRouter(
    chrome.offscreen as unknown as OffscreenChromeApi,
    (req) => chrome.runtime.sendMessage(req) as Promise<OffscreenRpcResponse>,
    tabsApi,
    { syncStorage, sessionStorage },
  );

  // Reconcile chrome.storage.sync settings immediately on SW startup
  void router.syncSettingsOnBoot();

  if (chrome.storage?.onChanged) {
    chrome.storage.onChanged.addListener((changes, areaName) => {
      void router.handleSyncStorageChange(changes, areaName);
    });
  }

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
    chrome.runtime.onInstalled.addListener((details) => {
      if (shouldReinjectOnInstalled(details.reason)) {
        void reinjectOpenXTabs(tabsApi, scriptingApi);
      }
    });
  }

  return router;
}

if (typeof chrome !== "undefined" && chrome.runtime?.onMessage && chrome.offscreen) {
  registerBackgroundRouter();
}
