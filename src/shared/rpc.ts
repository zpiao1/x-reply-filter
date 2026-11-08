import type { BatchEvaluationResult } from "../domain/engine.ts";
import {
  type CandidateTweet,
  type DomSignals,
  type ThreadId,
  type TweetId,
  type VerdictRecord,
  parseAuthorId,
  parseThreadId,
  parseTweetId,
} from "../domain/types.ts";

export type GpuStatusTag = "unloaded" | "loading" | "ready" | "recovering";

export type OffscreenRpcRequest =
  | {
      readonly kind: "evaluate_batch";
      readonly threadId: ThreadId;
      readonly candidates: ReadonlyArray<CandidateTweet>;
      readonly target?: "background" | "offscreen";
    }
  | {
      readonly kind: "record_feedback";
      readonly threadId: ThreadId;
      readonly tweetId: TweetId;
      readonly label: "spam" | "human";
      readonly target?: "background" | "offscreen";
    }
  | {
      readonly kind: "get_health";
      readonly target?: "background" | "offscreen";
    };

export type OffscreenRpcResponse =
  | {
      readonly kind: "evaluate_batch_ok";
      readonly result: BatchEvaluationResult;
    }
  | {
      readonly kind: "record_feedback_ok";
      readonly updatedVerdict: VerdictRecord | null;
      readonly updateCount: number;
    }
  | {
      readonly kind: "health_ok";
      readonly gpuStatus: GpuStatusTag;
      readonly deviceEpoch: number;
    }
  | {
      readonly kind: "rpc_error";
      readonly message: string;
    };

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function parseDomSignals(raw: unknown): DomSignals | null {
  if (!isRecord(raw)) {
    return null;
  }
  if (
    typeof raw.hasVerifiedBadge !== "boolean" ||
    typeof raw.hasMediaOrExternalLink !== "boolean" ||
    typeof raw.endsWithBaitQuestion !== "boolean" ||
    typeof raw.emojiToCharRatio !== "number" ||
    typeof raw.cjkCharRatio !== "number" ||
    (raw.matchedSpamKeyword !== null && typeof raw.matchedSpamKeyword !== "string")
  ) {
    return null;
  }

  return {
    hasVerifiedBadge: raw.hasVerifiedBadge,
    hasMediaOrExternalLink: raw.hasMediaOrExternalLink,
    endsWithBaitQuestion: raw.endsWithBaitQuestion,
    emojiToCharRatio: Math.max(0, Math.min(1, raw.emojiToCharRatio)),
    cjkCharRatio: Math.max(0, Math.min(1, raw.cjkCharRatio)),
    matchedSpamKeyword: raw.matchedSpamKeyword,
  };
}

function parseCandidateTweet(raw: unknown): CandidateTweet | null {
  if (!isRecord(raw)) {
    return null;
  }
  if (
    typeof raw.tweetId !== "string" ||
    typeof raw.authorId !== "string" ||
    typeof raw.rawText !== "string" ||
    typeof raw.normalizedText !== "string" ||
    typeof raw.isOp !== "boolean"
  ) {
    return null;
  }

  const tweetId = parseTweetId(raw.tweetId);
  const authorId = parseAuthorId(raw.authorId);
  const domSignals = parseDomSignals(raw.domSignals);
  if (tweetId === null || authorId === null || domSignals === null) {
    return null;
  }

  return {
    tweetId,
    authorId,
    rawText: raw.rawText,
    normalizedText: raw.normalizedText,
    isOp: raw.isOp,
    domSignals,
  };
}

export function parseRpcRequest(raw: unknown): OffscreenRpcRequest | null {
  if (!isRecord(raw) || typeof raw.kind !== "string") {
    return null;
  }

  const target =
    raw.target === "background" || raw.target === "offscreen" ? raw.target : undefined;

  if (raw.kind === "get_health") {
    return target !== undefined ? { kind: "get_health", target } : { kind: "get_health" };
  }

  if (raw.kind === "record_feedback") {
    if (
      typeof raw.threadId !== "string" ||
      typeof raw.tweetId !== "string" ||
      (raw.label !== "spam" && raw.label !== "human")
    ) {
      return null;
    }
    const threadId = parseThreadId(raw.threadId);
    const tweetId = parseTweetId(raw.tweetId);
    if (threadId === null || tweetId === null) {
      return null;
    }
    return target !== undefined
      ? { kind: "record_feedback", threadId, tweetId, label: raw.label, target }
      : { kind: "record_feedback", threadId, tweetId, label: raw.label };
  }

  if (raw.kind === "evaluate_batch") {
    if (typeof raw.threadId !== "string" || !Array.isArray(raw.candidates)) {
      return null;
    }
    const threadId = parseThreadId(raw.threadId);
    if (threadId === null) {
      return null;
    }
    const candidates: CandidateTweet[] = [];
    for (const item of raw.candidates) {
      const parsed = parseCandidateTweet(item);
      if (parsed === null) {
        return null;
      }
      candidates.push(parsed);
    }
    return target !== undefined
      ? { kind: "evaluate_batch", threadId, candidates, target }
      : { kind: "evaluate_batch", threadId, candidates };
  }

  return null;
}

export interface OffscreenChromeApi {
  readonly hasDocument: () => Promise<boolean>;
  readonly createDocument: (params: {
    readonly url: string;
    readonly reasons: string[];
    readonly justification: string;
  }) => Promise<void>;
}

let inFlightOffscreenPromise: Promise<void> | null = null;

export async function ensureOffscreenDocument(
  offscreenUrl: string,
  offscreenApi: OffscreenChromeApi = chrome.offscreen as unknown as OffscreenChromeApi,
): Promise<void> {
  if (await offscreenApi.hasDocument()) {
    return;
  }
  if (inFlightOffscreenPromise !== null) {
    return inFlightOffscreenPromise;
  }

  inFlightOffscreenPromise = (async () => {
    try {
      if (!(await offscreenApi.hasDocument())) {
        await offscreenApi.createDocument({
          url: offscreenUrl,
          reasons: ["WORKERS", "BLOBS"],
          justification: "Run local WebGPU EmbeddingGemma-300M inference for X reply filtering",
        });
      }
    } finally {
      inFlightOffscreenPromise = null;
    }
  })();

  return inFlightOffscreenPromise;
}
