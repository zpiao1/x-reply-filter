import type { BatchEvaluationResult, DiagnosticsSummary } from "../domain/engine.ts";
import {
  type AuthorId,
  type CandidateTweet,
  type DomSignals,
  type FilterSettings,
  type GraphqlTweetMeta,
  type MediaKind,
  type PolicyToggles,
  type SensitivityPreset,
  type Stage2StatusTag,
  type ThreadId,
  type TweetId,
  type VerdictRecord,
  type VerifiedType,
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
      readonly focalAuthorId?: AuthorId | undefined;
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

export type ExtendedRpcRequest =
  | {
      readonly kind: "get_settings";
      readonly target?: "background" | "offscreen";
    }
  | {
      readonly kind: "update_settings";
      readonly settings: FilterSettings;
      readonly target?: "background" | "offscreen";
    }
  | {
      readonly kind: "undo_feedback";
      readonly target?: "background" | "offscreen";
    }
  | {
      readonly kind: "export_feedback";
      readonly target?: "background" | "offscreen";
    }
  | {
      readonly kind: "import_feedback";
      readonly payloadJson: string;
      readonly target?: "background" | "offscreen";
    }
  | {
      readonly kind: "reset_feedback";
      readonly target?: "background" | "offscreen";
    }
  | {
      readonly kind: "trigger_stage2_download";
      readonly action?: "status" | "download";
      readonly target?: "background" | "offscreen";
    };

export type OffscreenRpcRequestV2 = OffscreenRpcRequest | ExtendedRpcRequest;

export interface Stage2VerdictUpdateMessage {
  readonly kind: "stage2_verdict_update";
  readonly threadId: ThreadId;
  readonly tweetId: TweetId;
  readonly verdict: VerdictRecord;
  readonly target?: "content" | "background";
}

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
      readonly rolledBack?: boolean | undefined;
      readonly rollbackCount?: number | undefined;
    }
  | {
      readonly kind: "health_ok";
      readonly gpuStatus: GpuStatusTag;
      readonly deviceEpoch: number;
      readonly stage2Status?: Stage2StatusTag | undefined;
    }
  | {
      readonly kind: "diagnostics_ok";
      readonly gpuStatus: GpuStatusTag;
      readonly deviceEpoch: number;
      readonly summary: DiagnosticsSummary;
      readonly stage2Status?: Stage2StatusTag | undefined;
    }
  | {
      readonly kind: "settings_ok";
      readonly settings: FilterSettings;
      readonly reevaluatedThreads?: number | undefined;
    }
  | {
      readonly kind: "undo_feedback_ok";
      readonly undone: boolean;
      readonly restoredVerdict?: VerdictRecord | null | undefined;
      readonly undoneTweetId?: TweetId | undefined;
      readonly updateCount: number;
    }
  | {
      readonly kind: "export_feedback_ok";
      readonly payloadJson: string;
    }
  | {
      readonly kind: "import_feedback_ok";
      readonly importedReplayCount: number;
      readonly importedAuthorCount: number;
      readonly rolledBack: boolean;
    }
  | {
      readonly kind: "reset_feedback_ok";
      readonly updateCount: number;
    }
  | {
      readonly kind: "stage2_status_ok";
      readonly status: Stage2StatusTag;
      readonly downloadProgress?: number | undefined;
    }
  | {
      readonly kind: "rpc_error";
      readonly message: string;
    };

const MAX_DESCRIPTION_LEN = 512;
const MAX_DISPLAY_NAME_LEN = 128;
const MAX_TWEET_TEXT_LEN = 4000;
const MAX_BATCH_CANDIDATES = 200;
const MAX_SNIPPET_LEN = 160;
const MAX_IMPORT_JSON_LEN = 2_000_000;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function parseNullableBool(val: unknown): boolean | null | undefined {
  if (val === null) return null;
  if (typeof val === "boolean") return val;
  return undefined;
}

function parseNullableNonNegInt(val: unknown): number | null | undefined {
  if (val === null) return null;
  if (typeof val === "number" && Number.isFinite(val) && val >= 0) {
    return Math.floor(val);
  }
  return undefined;
}

function parseNullableFiniteMs(val: unknown): number | null | undefined {
  if (val === null) return null;
  if (typeof val === "number" && Number.isFinite(val) && val >= 0) {
    return Math.floor(val);
  }
  return undefined;
}

function parseNullableCappedString(val: unknown, maxLen: number): string | null | undefined {
  if (val === null) return null;
  if (typeof val === "string") {
    return val.length > maxLen ? val.slice(0, maxLen) : val;
  }
  return undefined;
}

function parseVerifiedType(val: unknown): VerifiedType | null {
  if (
    val === "blue" ||
    val === "business" ||
    val === "government" ||
    val === "none" ||
    val === "unknown_verified"
  ) {
    return val;
  }
  return null;
}

export function parseGraphqlTweetMeta(raw: unknown): GraphqlTweetMeta | null {
  if (!isRecord(raw)) {
    return null;
  }

  const verifiedType = parseVerifiedType(raw.verifiedType ?? "none");
  const displayName = parseNullableCappedString(raw.displayName ?? null, MAX_DISPLAY_NAME_LEN);
  const description = parseNullableCappedString(raw.description ?? null, MAX_DESCRIPTION_LEN);
  const authorCreatedAtMs = parseNullableFiniteMs(raw.authorCreatedAtMs ?? null);
  const tweetCreatedAtMs = parseNullableFiniteMs(raw.tweetCreatedAtMs ?? null);
  const inReplyToStatusId = parseNullableCappedString(raw.inReplyToStatusId ?? null, 32);
  const conversationId = parseNullableCappedString(raw.conversationId ?? null, 32);
  const isBlueVerified = parseNullableBool(raw.isBlueVerified ?? null);
  const following = parseNullableBool(raw.following ?? null);
  const followedBy = parseNullableBool(raw.followedBy ?? null);
  const followersCount = parseNullableNonNegInt(raw.followersCount ?? null);
  const friendsCount = parseNullableNonNegInt(raw.friendsCount ?? null);
  const statusesCount = parseNullableNonNegInt(raw.statusesCount ?? null);
  const defaultProfileImage = parseNullableBool(raw.defaultProfileImage ?? null);
  const favoriteCount = parseNullableNonNegInt(raw.favoriteCount ?? null);
  const replyCount = parseNullableNonNegInt(raw.replyCount ?? null);
  const retweetCount = parseNullableNonNegInt(raw.retweetCount ?? null);

  if (
    verifiedType === null ||
    displayName === undefined ||
    description === undefined ||
    authorCreatedAtMs === undefined ||
    tweetCreatedAtMs === undefined ||
    inReplyToStatusId === undefined ||
    conversationId === undefined ||
    isBlueVerified === undefined ||
    following === undefined ||
    followedBy === undefined ||
    followersCount === undefined ||
    friendsCount === undefined ||
    statusesCount === undefined ||
    defaultProfileImage === undefined ||
    favoriteCount === undefined ||
    replyCount === undefined ||
    retweetCount === undefined
  ) {
    return null;
  }

  const mediaTypes: MediaKind[] = [];
  if (raw.mediaTypes !== undefined) {
    if (!Array.isArray(raw.mediaTypes)) {
      return null;
    }
    for (const item of raw.mediaTypes) {
      if (item === "photo" || item === "video" || item === "animated_gif") {
        mediaTypes.push(item);
      }
    }
  }

  const parsedTweetId =
    typeof raw.tweetId === "string" ? parseTweetId(raw.tweetId) : null;
  const parsedScreenName =
    typeof raw.screenName === "string" ? parseAuthorId(raw.screenName) : null;
  const authorRestId =
    raw.authorRestId === null
      ? null
      : typeof raw.authorRestId === "string"
        ? raw.authorRestId.slice(0, 32)
        : undefined;

  return {
    ...(parsedTweetId !== null ? { tweetId: parsedTweetId } : {}),
    ...(authorRestId !== undefined ? { authorRestId } : {}),
    ...(parsedScreenName !== null
      ? { screenName: parsedScreenName }
      : raw.screenName === null
        ? { screenName: null }
        : {}),
    displayName,
    authorCreatedAtMs,
    tweetCreatedAtMs,
    inReplyToStatusId,
    conversationId,
    description,
    isBlueVerified,
    verifiedType,
    following,
    followedBy,
    followersCount,
    friendsCount,
    statusesCount,
    defaultProfileImage,
    favoriteCount,
    replyCount,
    retweetCount,
    mediaTypes,
  };
}

export function parseFilterSettings(raw: unknown): FilterSettings | null {
  if (!isRecord(raw)) {
    return null;
  }
  const preset = raw.preset;
  if (preset !== "conservative" && preset !== "balanced" && preset !== "strict") {
    return null;
  }
  if (!isRecord(raw.toggles)) {
    return null;
  }
  const t = raw.toggles;
  if (
    typeof t.collapseBlueCheckReplyGuys !== "boolean" ||
    typeof t.collapseMediaOnlyReplies !== "boolean" ||
    typeof t.collapseBotSummons !== "boolean" ||
    typeof t.showSuspectBadges !== "boolean" ||
    typeof t.learnFromFeedback !== "boolean" ||
    typeof t.enableStage2 !== "boolean"
  ) {
    return null;
  }
  if (typeof raw.revision !== "number" || !Number.isFinite(raw.revision) || raw.revision < 0) {
    return null;
  }

  const toggles: PolicyToggles = {
    collapseBlueCheckReplyGuys: t.collapseBlueCheckReplyGuys,
    collapseMediaOnlyReplies: t.collapseMediaOnlyReplies,
    collapseBotSummons: t.collapseBotSummons,
    showSuspectBadges: t.showSuspectBadges,
    learnFromFeedback: t.learnFromFeedback,
    enableStage2: t.enableStage2,
  };

  return {
    preset: preset as SensitivityPreset,
    toggles,
    revision: Math.floor(raw.revision),
  };
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
    !Number.isFinite(raw.emojiToCharRatio) ||
    typeof raw.cjkCharRatio !== "number" ||
    !Number.isFinite(raw.cjkCharRatio) ||
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
    matchedSpamKeyword:
      raw.matchedSpamKeyword !== null ? raw.matchedSpamKeyword.slice(0, 64) : null,
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

  // #33 MAJOR: Degrade malformed graphqlMeta (schema drift / NaN) to null instead of rejecting the candidate/batch
  let graphqlMeta: GraphqlTweetMeta | null | undefined;
  if (raw.graphqlMeta === null) {
    graphqlMeta = null;
  } else if (raw.graphqlMeta !== undefined) {
    graphqlMeta = parseGraphqlTweetMeta(raw.graphqlMeta);
  }

  if (raw.hasPrimaryMedia !== undefined && typeof raw.hasPrimaryMedia !== "boolean") {
    return null;
  }
  if (raw.hasQuote !== undefined && typeof raw.hasQuote !== "boolean") {
    return null;
  }
  if (raw.isContextAncestor !== undefined && typeof raw.isContextAncestor !== "boolean") {
    return null;
  }

  let opFavoriteCount: number | null | undefined;
  if (raw.opFavoriteCount !== undefined) {
    const parsedLikes = parseNullableNonNegInt(raw.opFavoriteCount);
    opFavoriteCount = parsedLikes === undefined ? null : parsedLikes;
  }

  const rawText =
    raw.rawText.length > MAX_TWEET_TEXT_LEN
      ? raw.rawText.slice(0, MAX_TWEET_TEXT_LEN)
      : raw.rawText;
  const normalizedText =
    raw.normalizedText.length > MAX_TWEET_TEXT_LEN
      ? raw.normalizedText.slice(0, MAX_TWEET_TEXT_LEN)
      : raw.normalizedText;

  return {
    tweetId,
    authorId,
    rawText,
    normalizedText,
    isOp: raw.isOp,
    domSignals,
    ...(graphqlMeta !== undefined ? { graphqlMeta } : {}),
    ...(raw.hasPrimaryMedia !== undefined ? { hasPrimaryMedia: raw.hasPrimaryMedia } : {}),
    ...(raw.hasQuote !== undefined ? { hasQuote: raw.hasQuote } : {}),
    ...(raw.isContextAncestor !== undefined ? { isContextAncestor: raw.isContextAncestor } : {}),
    ...(opFavoriteCount !== undefined ? { opFavoriteCount } : {}),
  };
}

export function parseRpcRequestV2(raw: unknown): OffscreenRpcRequestV2 | null {
  if (!isRecord(raw) || typeof raw.kind !== "string") {
    return null;
  }

  const target =
    raw.target === "background" || raw.target === "offscreen" ? raw.target : undefined;

  if (raw.kind === "get_health") {
    return target !== undefined ? { kind: "get_health", target } : { kind: "get_health" };
  }

  if (raw.kind === "get_diagnostics") {
    return target !== undefined
      ? { kind: "get_diagnostics", target }
      : { kind: "get_diagnostics" };
  }

  if (raw.kind === "get_settings") {
    return target !== undefined ? { kind: "get_settings", target } : { kind: "get_settings" };
  }

  if (raw.kind === "update_settings") {
    const settings = parseFilterSettings(raw.settings);
    if (settings === null) return null;
    return target !== undefined
      ? { kind: "update_settings", settings, target }
      : { kind: "update_settings", settings };
  }

  if (raw.kind === "undo_feedback") {
    return target !== undefined ? { kind: "undo_feedback", target } : { kind: "undo_feedback" };
  }

  if (raw.kind === "export_feedback") {
    return target !== undefined
      ? { kind: "export_feedback", target }
      : { kind: "export_feedback" };
  }

  if (raw.kind === "import_feedback") {
    if (
      typeof raw.payloadJson !== "string" ||
      raw.payloadJson.length === 0 ||
      raw.payloadJson.length > MAX_IMPORT_JSON_LEN
    ) {
      return null;
    }
    return target !== undefined
      ? { kind: "import_feedback", payloadJson: raw.payloadJson, target }
      : { kind: "import_feedback", payloadJson: raw.payloadJson };
  }

  if (raw.kind === "reset_feedback") {
    return target !== undefined ? { kind: "reset_feedback", target } : { kind: "reset_feedback" };
  }

  if (raw.kind === "trigger_stage2_download") {
    const action =
      raw.action === "status" || raw.action === "download" ? raw.action : undefined;
    return {
      kind: "trigger_stage2_download",
      ...(action !== undefined ? { action } : {}),
      ...(target !== undefined ? { target } : {}),
    };
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
    const focalAuthorId =
      typeof raw.focalAuthorId === "string"
        ? (parseAuthorId(raw.focalAuthorId) ?? undefined)
        : undefined;
    // #33 MAJOR: Cap candidate count and skip individual invalid candidates instead of rejecting the entire batch
    const limit = Math.min(raw.candidates.length, MAX_BATCH_CANDIDATES);
    const candidates: CandidateTweet[] = [];
    for (let i = 0; i < limit; i++) {
      const parsed = parseCandidateTweet(raw.candidates[i]);
      if (parsed === null) {
        continue;
      }
      candidates.push(parsed);
    }
    return {
      kind: "evaluate_batch",
      threadId,
      candidates,
      ...(focalAuthorId !== undefined ? { focalAuthorId } : {}),
      ...(target !== undefined ? { target } : {}),
    };
  }

  return null;
}

export function parseRpcRequest(raw: unknown): OffscreenRpcRequest | null {
  const parsed = parseRpcRequestV2(raw);
  if (parsed === null) {
    return null;
  }
  if (
    parsed.kind === "evaluate_batch" ||
    parsed.kind === "record_feedback" ||
    parsed.kind === "get_health" ||
    parsed.kind === "get_diagnostics"
  ) {
    return parsed;
  }
  return null;
}

function isValidExplanationShape(raw: unknown): boolean {
  if (!isRecord(raw)) return false;
  if (
    raw.schemaVersion !== 2 ||
    (raw.tier !== "keep" && raw.tier !== "suspect" && raw.tier !== "dim") ||
    typeof raw.ruleId !== "string" ||
    typeof raw.probability !== "number" ||
    !Number.isFinite(raw.probability) ||
    typeof raw.logit !== "number" ||
    !Number.isFinite(raw.logit) ||
    !isRecord(raw.thresholds) ||
    typeof raw.thresholds.dim !== "number" ||
    !Number.isFinite(raw.thresholds.dim) ||
    typeof raw.thresholds.suspect !== "number" ||
    !Number.isFinite(raw.thresholds.suspect) ||
    (raw.preset !== "conservative" && raw.preset !== "balanced" && raw.preset !== "strict") ||
    !Array.isArray(raw.top) ||
    !Array.isArray(raw.counter) ||
    !Array.isArray(raw.families) ||
    !Array.isArray(raw.shields) ||
    !Array.isArray(raw.caps) ||
    (raw.metaSource !== "graphql" && raw.metaSource !== "dom") ||
    !isRecord(raw.lang) ||
    typeof raw.lang.reply !== "string"
  ) {
    return false;
  }
  return true;
}

export function parseStage2VerdictUpdate(raw: unknown): Stage2VerdictUpdateMessage | null {
  if (!isRecord(raw) || raw.kind !== "stage2_verdict_update") {
    return null;
  }
  if (
    typeof raw.threadId !== "string" ||
    typeof raw.tweetId !== "string" ||
    !isRecord(raw.verdict)
  ) {
    return null;
  }
  const threadId = parseThreadId(raw.threadId);
  const tweetId = parseTweetId(raw.tweetId);
  if (threadId === null || tweetId === null) {
    return null;
  }
  const v = raw.verdict;
  if (
    v.tweetId !== tweetId ||
    (v.action !== "keep" && v.action !== "suspect" && v.action !== "dim") ||
    typeof v.score !== "number" ||
    !Number.isFinite(v.score) ||
    typeof v.tokenCount !== "number" ||
    !Number.isFinite(v.tokenCount) ||
    v.tokenCount < 0 ||
    !isRecord(v.reason) ||
    typeof v.reason.kind !== "string"
  ) {
    return null;
  }
  if (v.explanation !== undefined && !isValidExplanationShape(v.explanation)) {
    return null;
  }
  const sanitizedVerdict =
    typeof v.snippet === "string" && v.snippet.length > MAX_SNIPPET_LEN
      ? { ...v, snippet: v.snippet.slice(0, MAX_SNIPPET_LEN) }
      : v;
  const target =
    raw.target === "content" || raw.target === "background" ? raw.target : undefined;
  return {
    kind: "stage2_verdict_update",
    threadId,
    tweetId,
    verdict: sanitizedVerdict as unknown as VerdictRecord,
    ...(target !== undefined ? { target } : {}),
  };
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
