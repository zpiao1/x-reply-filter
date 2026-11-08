import type { BatchEvaluationResult, DiagnosticsSummary } from "../domain/engine.ts";
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
    }
  | {
      readonly kind: "get_diagnostics";
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
      readonly retroactiveUpdates?: ReadonlyArray<VerdictRecord> | undefined;
      readonly updateCount: number;
    }
  | {
      readonly kind: "health_ok";
      readonly gpuStatus: GpuStatusTag;
      readonly deviceEpoch: number;
    }
  | {
      readonly kind: "diagnostics_ok";
      readonly gpuStatus: GpuStatusTag;
      readonly deviceEpoch: number;
      readonly summary: DiagnosticsSummary;
    }
  | {
      readonly kind: "rpc_error";
      readonly message: string;
    };

function isRecord(value: unknown): boolean {
  return typeof value === "object" && value !== null;
}

function parseDomSignals(raw: unknown): DomSignals | null {
  if (!isRecord(raw)) {
    return null;
  }
  const obj = raw as Record<string, unknown>;
  if (
    typeof obj.hasVerifiedBadge !== "boolean" ||
    typeof obj.hasMediaOrExternalLink !== "boolean" ||
    typeof obj.endsWithBaitQuestion !== "boolean" ||
    typeof obj.emojiToCharRatio !== "number" ||
    typeof obj.cjkCharRatio !== "number" ||
    (obj.matchedSpamKeyword !== null && typeof obj.matchedSpamKeyword !== "string")
  ) {
    return null;
  }

  return {
    hasVerifiedBadge: obj.hasVerifiedBadge,
    hasMediaOrExternalLink: obj.hasMediaOrExternalLink,
    endsWithBaitQuestion: obj.endsWithBaitQuestion,
    emojiToCharRatio: Math.max(0, Math.min(1, obj.emojiToCharRatio)),
    cjkCharRatio: Math.max(0, Math.min(1, obj.cjkCharRatio)),
    matchedSpamKeyword: obj.matchedSpamKeyword,
  };
}

function parseCandidateTweet(raw: unknown): CandidateTweet | null {
  if (!isRecord(raw)) {
    return null;
  }
  const obj = raw as Record<string, unknown>;
  if (
    typeof obj.tweetId !== "string" ||
    typeof obj.authorId !== "string" ||
    typeof obj.rawText !== "string" ||
    typeof obj.normalizedText !== "string" ||
    typeof obj.isOp !== "boolean"
  ) {
    return null;
  }

  const tweetId = parseTweetId(obj.tweetId);
  const authorId = parseAuthorId(obj.authorId);
  const domSignals = parseDomSignals(obj.domSignals);
  if (tweetId === null || authorId === null || domSignals === null) {
    return null;
  }

  return {
    tweetId,
    authorId,
    rawText: obj.rawText,
    normalizedText: obj.normalizedText,
    isOp: obj.isOp,
    domSignals,
  };
}

export function parseRpcRequest(raw: unknown): OffscreenRpcRequest | null {
  if (!isRecord(raw)) {
    return null;
  }
  const obj = raw as Record<string, unknown>;
  if (typeof obj.kind !== "string") {
    return null;
  }

  const target =
    obj.target === "background" || obj.target === "offscreen" ? obj.target : undefined;

  if (obj.kind === "get_health") {
    return target !== undefined ? { kind: "get_health", target } : { kind: "get_health" };
  }

  if (obj.kind === "get_diagnostics") {
    return target !== undefined
      ? { kind: "get_diagnostics", target }
      : { kind: "get_diagnostics" };
  }

  if (obj.kind === "record_feedback") {
    if (
      typeof obj.threadId !== "string" ||
      typeof obj.tweetId !== "string" ||
      (obj.label !== "spam" && obj.label !== "human")
    ) {
      return null;
    }
    const threadId = parseThreadId(obj.threadId);
    const tweetId = parseTweetId(obj.tweetId);
    if (threadId === null || tweetId === null) {
      return null;
    }
    return target !== undefined
      ? { kind: "record_feedback", threadId, tweetId, label: obj.label, target }
      : { kind: "record_feedback", threadId, tweetId, label: obj.label };
  }

  if (obj.kind === "evaluate_batch") {
    if (typeof obj.threadId !== "string" || !Array.isArray(obj.candidates)) {
      return null;
    }
    const threadId = parseThreadId(obj.threadId);
    if (threadId === null) {
      return null;
    }
    const candidates: CandidateTweet[] = [];
    for (const item of obj.candidates) {
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
