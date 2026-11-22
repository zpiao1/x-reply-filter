import type {
  LangV2,
  Stage2StatusTag,
  Stage2Verdict,
  ThreadId,
  TweetId,
} from "../domain/types.ts";
import { OFFSCREEN_IDLE_TIMEOUT_MS } from "../domain/types.ts";

export const STAGE2_PROVIDER_ID = "chrome_gemini_nano" as const;
export const STAGE2_DEFAULT_TIMEOUT_MS = 1_500 as const;
export const STAGE2_MAX_PENDING_PER_THREAD = 3 as const;

export type RawLanguageModelAvailability =
  | "available"
  | "readily"
  | "downloadable"
  | "after-download"
  | "downloading"
  | "unavailable"
  | "no";

export interface LanguageModelDownloadProgressEvent {
  readonly loaded?: number | undefined;
  readonly total?: number | undefined;
}

export interface LanguageModelCreateMonitor {
  addEventListener: (
    type: "downloadprogress",
    listener: (event: LanguageModelDownloadProgressEvent) => void,
  ) => void;
}

export interface LanguageModelExpectedIo {
  readonly type: "text";
  readonly languages?: ReadonlyArray<string>;
}

export interface LanguageModelAvailabilityOptions {
  readonly expectedInputs?: ReadonlyArray<LanguageModelExpectedIo>;
  readonly expectedOutputs?: ReadonlyArray<LanguageModelExpectedIo>;
}

export interface LanguageModelCreateOptions extends LanguageModelAvailabilityOptions {
  readonly initialPrompts?: ReadonlyArray<{
    readonly role: "system" | "user" | "assistant";
    readonly content: string;
  }>;
  readonly temperature?: number;
  readonly topK?: number;
  readonly signal?: AbortSignal;
  readonly monitor?: (monitor: LanguageModelCreateMonitor) => void;
}

export interface LanguageModelPromptOptions {
  readonly responseConstraint?: Record<string, unknown>;
  readonly signal?: AbortSignal;
}

export interface LanguageModelSession {
  readonly prompt: (input: string, options?: LanguageModelPromptOptions) => Promise<string>;
  readonly clone: (options?: { readonly signal?: AbortSignal }) => Promise<LanguageModelSession>;
  readonly destroy: () => void | Promise<void>;
}

export interface LanguageModelProvider {
  readonly availability: (
    options?: LanguageModelAvailabilityOptions,
  ) => Promise<RawLanguageModelAvailability>;
  readonly create: (options?: LanguageModelCreateOptions) => Promise<LanguageModelSession>;
}

export const STAGE2_RESPONSE_CONSTRAINT = {
  type: "object",
  properties: {
    verdict: { type: "string", enum: ["bot", "human", "uncertain"] },
    confidence: { type: "number", minimum: 0, maximum: 1 },
    reason: { type: "string", maxLength: 80 },
  },
  required: ["verdict", "confidence", "reason"],
  additionalProperties: false,
} as const;

const STAGE2_SYSTEM_PROMPT = [
  "You classify borderline X (Twitter) replies as 'bot' (low-effort AI slop, sycophantic engagement-farming, generic praise/antithesis filler, spam) or 'human' (authentic personal reaction, concrete point, sarcasm, humor, banter, disagreement).",
  "Be conservative: when unsure, output 'uncertain'. Never treat sarcasm, slang, political debate, or concrete critique as bot.",
  "Respond strictly with JSON matching the schema: { verdict, confidence, reason }.",
].join(" ");

export interface Stage2Task {
  readonly threadId: ThreadId;
  readonly tweetId: TweetId;
  readonly replyText: string;
  readonly opText?: string | undefined;
  readonly replyLang: LangV2;
  readonly signalSummary?: string | undefined;
}

export interface Stage2Completion {
  readonly task: Stage2Task;
  readonly verdict: Stage2Verdict;
}

export interface Stage2DownloadResult {
  readonly status: Stage2StatusTag;
  readonly downloadProgress?: number | undefined;
}

export interface Stage2JudgeOptions {
  readonly provider?: LanguageModelProvider | null | (() => LanguageModelProvider | null);
  readonly nowFn?: () => number;
  readonly timeoutMs?: number;
  readonly idleTimeoutMs?: number;
  readonly maxPendingPerThread?: number;
  readonly onResult?: (completion: Stage2Completion) => void | Promise<void>;
}

export function getDefaultLanguageModelProvider(): LanguageModelProvider | null {
  const g = globalThis as unknown as { readonly LanguageModel?: unknown };
  const lm = g.LanguageModel;
  if (
    typeof lm === "object" &&
    lm !== null &&
    typeof (lm as { availability?: unknown }).availability === "function" &&
    typeof (lm as { create?: unknown }).create === "function"
  ) {
    return lm as LanguageModelProvider;
  }
  return null;
}

export function normalizeStage2Availability(
  raw: RawLanguageModelAvailability | string | null | undefined,
): Stage2StatusTag {
  switch (raw) {
    case "available":
    case "readily":
      return "available";
    case "downloadable":
    case "after-download":
      return "downloadable";
    case "downloading":
      return "downloading";
    default:
      return "unavailable";
  }
}

export function mapLangV2ToBcp47(lang: LangV2 | undefined): string | null {
  switch (lang) {
    case "en":
      return "en";
    case "ja":
      return "ja";
    case "zh-Hans":
      return "zh-Hans";
    case "zh-Hant":
      return "zh-Hant";
    default:
      return null;
  }
}

function buildIoOptions(replyBcp47: string): LanguageModelAvailabilityOptions {
  return {
    expectedInputs: [{ type: "text", languages: [replyBcp47] }],
    expectedOutputs: [{ type: "text", languages: ["en"] }],
  };
}

function buildUserPrompt(task: Stage2Task): string {
  const lines: string[] = [];
  if (task.opText && task.opText.trim().length > 0) {
    lines.push(`OP: ${task.opText.trim().slice(0, 280)}`);
  }
  lines.push(`Reply (${task.replyLang}): ${task.replyText.trim().slice(0, 400)}`);
  if (task.signalSummary && task.signalSummary.trim().length > 0) {
    lines.push(`Stage-1 notes: ${task.signalSummary.trim().slice(0, 160)}`);
  }
  lines.push("Classify this reply.");
  return lines.join("\n");
}

function parseStage2JsonResponse(
  rawText: string,
  elapsedMs: number,
): Stage2Verdict | null {
  let parsed: unknown;
  try {
    parsed = JSON.parse(rawText);
  } catch {
    return null;
  }
  if (typeof parsed !== "object" || parsed === null) {
    return null;
  }
  const obj = parsed as Record<string, unknown>;
  const verdict = obj.verdict;
  const confidence = obj.confidence;
  const reason = obj.reason;

  if (verdict !== "bot" && verdict !== "human" && verdict !== "uncertain") {
    return null;
  }
  if (
    typeof confidence !== "number" ||
    !Number.isFinite(confidence) ||
    confidence < 0 ||
    confidence > 1
  ) {
    return null;
  }
  if (typeof reason !== "string") {
    return null;
  }

  const trimmedReason = reason.trim().slice(0, 80);
  return {
    provider: STAGE2_PROVIDER_ID,
    verdict,
    confidence,
    ms: Math.max(0, Math.round(elapsedMs)),
    ...(trimmedReason.length > 0 ? { reason: trimmedReason } : {}),
  };
}

function raceWithAbortSignal<T>(promise: Promise<T>, signal: AbortSignal): Promise<T> {
  if (signal.aborted) {
    return Promise.reject(signal.reason ?? new Error("Stage-2 request aborted"));
  }
  return new Promise<T>((resolve, reject) => {
    const onAbort = () => {
      reject(signal.reason ?? new Error("Stage-2 request timed out"));
    };
    signal.addEventListener("abort", onAbort, { once: true });
    promise.then(
      (val) => {
        signal.removeEventListener("abort", onAbort);
        resolve(val);
      },
      (err) => {
        signal.removeEventListener("abort", onAbort);
        reject(err);
      },
    );
  });
}

export class Stage2JudgeService {
  private readonly providerResolver: () => LanguageModelProvider | null;
  private readonly nowFn: () => number;
  private readonly timeoutMs: number;
  private readonly idleTimeoutMs: number;
  private readonly maxPendingPerThread: number;
  private onResultCallback: ((completion: Stage2Completion) => void | Promise<void>) | undefined;

  private readonly pendingQueue: Stage2Task[] = [];
  private readonly queuedOrActiveTweetIds = new Set<TweetId>();
  private readonly pendingCountByThread = new Map<ThreadId, number>();
  private drainScheduled = false;
  private draining = false;
  private idleResolvers: Array<() => void> = [];

  private baseSession: LanguageModelSession | null = null;
  private baseSessionLang: string | null = null;
  private lastUsedAtMs = 0;

  private downloading = false;
  private downloadProgress: number | undefined;
  private activeDownloadPromise: Promise<Stage2DownloadResult> | null = null;

  constructor(options?: Stage2JudgeOptions) {
    if (options && "provider" in options) {
      const p = options.provider;
      this.providerResolver = typeof p === "function" ? p : () => p ?? null;
    } else {
      this.providerResolver = getDefaultLanguageModelProvider;
    }
    this.nowFn = options?.nowFn ?? (() => Date.now());
    this.timeoutMs = options?.timeoutMs ?? STAGE2_DEFAULT_TIMEOUT_MS;
    this.idleTimeoutMs = options?.idleTimeoutMs ?? OFFSCREEN_IDLE_TIMEOUT_MS;
    this.maxPendingPerThread = options?.maxPendingPerThread ?? STAGE2_MAX_PENDING_PER_THREAD;
    this.onResultCallback = options?.onResult;
  }

  public setOnResult(
    cb: ((completion: Stage2Completion) => void | Promise<void>) | undefined,
  ): void {
    this.onResultCallback = cb;
  }

  public getPendingCount(threadId?: ThreadId): number {
    if (threadId !== undefined) {
      return this.pendingCountByThread.get(threadId) ?? 0;
    }
    return this.pendingQueue.length;
  }

  public hasActiveSession(): boolean {
    return this.baseSession !== null;
  }

  public async checkAvailability(lang: string = "en"): Promise<Stage2DownloadResult> {
    if (this.downloading) {
      return {
        status: "downloading",
        ...(this.downloadProgress !== undefined ? { downloadProgress: this.downloadProgress } : {}),
      };
    }

    const provider = this.providerResolver();
    if (provider === null) {
      return { status: "unavailable" };
    }

    try {
      const raw = await provider.availability(buildIoOptions(lang));
      const status = normalizeStage2Availability(raw);
      return {
        status,
        ...(status === "available"
          ? { downloadProgress: 1 }
          : status === "downloading" && this.downloadProgress !== undefined
            ? { downloadProgress: this.downloadProgress }
            : {}),
      };
    } catch {
      return { status: "unavailable" };
    }
  }

  public async getStatus(enabled: boolean): Promise<Stage2StatusTag> {
    if (!enabled) {
      return "disabled";
    }
    const res = await this.checkAvailability("en");
    return res.status;
  }

  /**
   * Must ONLY be called in response to an explicit popup user gesture.
   * Never invoked automatically during background batch evaluation.
   */
  public async triggerDownload(lang: string = "en"): Promise<Stage2DownloadResult> {
    if (this.activeDownloadPromise !== null) {
      return this.activeDownloadPromise;
    }

    const provider = this.providerResolver();
    if (provider === null) {
      return { status: "unavailable" };
    }

    const ioOptions = buildIoOptions(lang);
    let initialStatus: Stage2StatusTag;
    try {
      initialStatus = normalizeStage2Availability(await provider.availability(ioOptions));
    } catch {
      return { status: "unavailable" };
    }

    if (initialStatus === "unavailable") {
      return { status: "unavailable" };
    }
    if (initialStatus === "available") {
      this.downloadProgress = 1;
      return { status: "available", downloadProgress: 1 };
    }

    this.downloading = true;
    this.downloadProgress = 0;

    this.activeDownloadPromise = (async (): Promise<Stage2DownloadResult> => {
      try {
        const session = await provider.create({
          ...ioOptions,
          initialPrompts: [{ role: "system", content: STAGE2_SYSTEM_PROMPT }],
          temperature: 0,
          topK: 1,
          monitor: (m) => {
            m.addEventListener("downloadprogress", (ev) => {
              if (typeof ev.loaded === "number" && Number.isFinite(ev.loaded)) {
                const ratio =
                  typeof ev.total === "number" && ev.total > 0
                    ? ev.loaded / ev.total
                    : ev.loaded;
                this.downloadProgress = Math.max(0, Math.min(1, ratio));
              }
            });
          },
        });
        await this.destroyBaseSession();
        this.baseSession = session;
        this.baseSessionLang = lang;
        this.lastUsedAtMs = this.nowFn();
        this.downloadProgress = 1;
        return { status: "available", downloadProgress: 1 };
      } catch {
        const after = await this.checkAvailability(lang);
        return after;
      } finally {
        this.downloading = false;
        this.activeDownloadPromise = null;
      }
    })();

    return this.activeDownloadPromise;
  }

  public enqueue(task: Stage2Task): boolean {
    if (task.replyText.trim().length === 0) {
      return false;
    }
    const bcp47 = mapLangV2ToBcp47(task.replyLang);
    if (bcp47 === null) {
      return false;
    }
    if (this.providerResolver() === null) {
      return false;
    }
    if (this.queuedOrActiveTweetIds.has(task.tweetId)) {
      return false;
    }

    const threadPending = this.pendingCountByThread.get(task.threadId) ?? 0;
    if (threadPending >= this.maxPendingPerThread) {
      return false;
    }

    this.pendingQueue.push(task);
    this.queuedOrActiveTweetIds.add(task.tweetId);
    this.pendingCountByThread.set(task.threadId, threadPending + 1);
    this.scheduleDrain();
    return true;
  }

  public clearPending(): void {
    for (const task of this.pendingQueue) {
      this.queuedOrActiveTweetIds.delete(task.tweetId);
    }
    this.pendingQueue.length = 0;
    this.pendingCountByThread.clear();
  }

  public waitForIdle(): Promise<void> {
    if (!this.draining && !this.drainScheduled && this.pendingQueue.length === 0) {
      return Promise.resolve();
    }
    return new Promise<void>((resolve) => {
      this.idleResolvers.push(resolve);
    });
  }

  public async unloadIfIdle(nowMs: number = this.nowFn()): Promise<boolean> {
    if (this.baseSession === null || this.draining) {
      return false;
    }
    if (nowMs - this.lastUsedAtMs < this.idleTimeoutMs) {
      return false;
    }
    await this.destroyBaseSession();
    return true;
  }

  public async dispose(): Promise<void> {
    this.clearPending();
    await this.destroyBaseSession();
  }

  private async destroyBaseSession(): Promise<void> {
    const existing = this.baseSession;
    this.baseSession = null;
    this.baseSessionLang = null;
    if (existing !== null) {
      try {
        await existing.destroy();
      } catch {}
    }
  }

  private scheduleDrain(): void {
    if (this.drainScheduled || this.draining) {
      return;
    }
    this.drainScheduled = true;
    queueMicrotask(() => {
      this.drainScheduled = false;
      void this.drainQueue();
    });
  }

  private notifyIdleIfDone(): void {
    if (!this.draining && !this.drainScheduled && this.pendingQueue.length === 0) {
      const resolvers = this.idleResolvers;
      this.idleResolvers = [];
      for (const resolve of resolvers) {
        resolve();
      }
    }
  }

  private async drainQueue(): Promise<void> {
    if (this.draining) {
      return;
    }
    this.draining = true;
    try {
      while (this.pendingQueue.length > 0) {
        const task = this.pendingQueue.shift()!;
        const prevCount = this.pendingCountByThread.get(task.threadId) ?? 1;
        if (prevCount <= 1) {
          this.pendingCountByThread.delete(task.threadId);
        } else {
          this.pendingCountByThread.set(task.threadId, prevCount - 1);
        }

        try {
          const verdict = await this.evaluateSingleTask(task);
          if (verdict !== null && this.onResultCallback !== undefined) {
            await this.onResultCallback({ task, verdict });
          }
        } catch {
          // Silent fallback to Stage-1 suspect on any runtime failure
        } finally {
          this.queuedOrActiveTweetIds.delete(task.tweetId);
        }
      }
    } finally {
      this.draining = false;
      if (this.pendingQueue.length > 0) {
        this.scheduleDrain();
      } else {
        this.notifyIdleIfDone();
      }
    }
  }

  private async ensureBaseSession(
    provider: LanguageModelProvider,
    bcp47: string,
    signal: AbortSignal,
  ): Promise<LanguageModelSession | null> {
    const ioOptions = buildIoOptions(bcp47);

    // Non-blocking availability check — NEVER trigger downloads in background
    let rawAvail: RawLanguageModelAvailability;
    try {
      rawAvail = await raceWithAbortSignal(provider.availability(ioOptions), signal);
    } catch {
      return null;
    }

    if (normalizeStage2Availability(rawAvail) !== "available") {
      return null;
    }

    if (this.baseSession !== null && this.baseSessionLang === bcp47) {
      this.lastUsedAtMs = this.nowFn();
      return this.baseSession;
    }

    await this.destroyBaseSession();
    try {
      const created = await raceWithAbortSignal(
        provider.create({
          ...ioOptions,
          initialPrompts: [{ role: "system", content: STAGE2_SYSTEM_PROMPT }],
          temperature: 0,
          topK: 1,
          signal,
        }),
        signal,
      );
      this.baseSession = created;
      this.baseSessionLang = bcp47;
      this.lastUsedAtMs = this.nowFn();
      return created;
    } catch {
      return null;
    }
  }

  private async evaluateSingleTask(task: Stage2Task): Promise<Stage2Verdict | null> {
    const provider = this.providerResolver();
    if (provider === null) {
      return null;
    }
    const bcp47 = mapLangV2ToBcp47(task.replyLang);
    if (bcp47 === null || task.replyText.trim().length === 0) {
      return null;
    }

    const startMs = this.nowFn();
    const signal = AbortSignal.timeout(this.timeoutMs);

    const base = await this.ensureBaseSession(provider, bcp47, signal);
    if (base === null) {
      return null;
    }

    let child: LanguageModelSession | null = null;
    try {
      child = await raceWithAbortSignal(base.clone({ signal }), signal);
      const userPrompt = buildUserPrompt(task);
      const rawOutput = await raceWithAbortSignal(
        child.prompt(userPrompt, {
          responseConstraint: STAGE2_RESPONSE_CONSTRAINT as unknown as Record<string, unknown>,
          signal,
        }),
        signal,
      );
      this.lastUsedAtMs = this.nowFn();
      const elapsedMs = Math.max(0, this.nowFn() - startMs);
      return parseStage2JsonResponse(rawOutput, elapsedMs);
    } catch {
      return null;
    } finally {
      if (child !== null) {
        try {
          await child.destroy();
        } catch {}
      }
    }
  }
}
