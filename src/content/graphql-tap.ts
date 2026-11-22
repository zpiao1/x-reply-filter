import type {
  AuthorId,
  GraphqlTweetMeta,
  MediaKind,
  TweetId,
  VerifiedType,
} from "../domain/types.ts";

export const GRAPHQL_TAP_SOURCE = "xrf-graphql-tap" as const;
export const GRAPHQL_CONTENT_SOURCE = "xrf-content" as const;
export const GRAPHQL_TAP_PROTOCOL_VERSION = 1 as const;
export const MAX_TAP_BUFFER = 256 as const;
export const TARGET_GRAPHQL_URL_RE = /\/graphql\/[^/]+\/(TweetDetail|TweetResultByRestId)\b/;

const INSTALLED_SYMBOL = Symbol.for("xrf.graphqlTapInstalled");
const MAX_WALK_DEPTH = 32;
const MAX_DESCRIPTION_LEN = 512;
const MAX_DISPLAY_NAME_LEN = 128;
const TWEET_ID_RE = /^\d{1,32}$/;
const AUTHOR_HANDLE_RE = /^[a-z0-9_]{1,64}$/;

interface TapBatchMessage {
  readonly source: typeof GRAPHQL_TAP_SOURCE;
  readonly v: typeof GRAPHQL_TAP_PROTOCOL_VERSION;
  readonly items: ReadonlyArray<GraphqlTweetMeta>;
}

function isRecord(val: unknown): val is Record<string, unknown> {
  return typeof val === "object" && val !== null && !Array.isArray(val);
}

function readString(val: unknown): string | null {
  return typeof val === "string" ? val : null;
}

function readCappedString(val: unknown, maxLen: number): string | null {
  if (typeof val !== "string") {
    return null;
  }
  return val.length > maxLen ? val.slice(0, maxLen) : val;
}

function readNullableBool(val: unknown): boolean | null {
  return typeof val === "boolean" ? val : null;
}

function parseIdStr(val: unknown): string | null {
  if (typeof val !== "string") {
    return null;
  }
  const trimmed = val.trim();
  return TWEET_ID_RE.test(trimmed) ? trimmed : null;
}

function parseHandle(val: unknown): AuthorId | null {
  if (typeof val !== "string") {
    return null;
  }
  const cleaned = val.trim().replace(/^@+/, "").toLowerCase();
  return AUTHOR_HANDLE_RE.test(cleaned) ? (cleaned as AuthorId) : null;
}

function parseNonNegInt(val: unknown): number | null {
  if (typeof val !== "number" || !Number.isFinite(val) || val < 0) {
    return null;
  }
  return Math.floor(val);
}

function parseXDateMs(val: unknown): number | null {
  if (typeof val !== "string" || val.trim().length === 0) {
    return null;
  }
  const ms = Date.parse(val);
  if (!Number.isFinite(ms) || ms < 0) {
    return null;
  }
  return Math.floor(ms);
}

function unwrapUserNode(tweet: Record<string, unknown>): Record<string, unknown> | null {
  const core = isRecord(tweet.core) ? tweet.core : null;
  const userResults = core && isRecord(core.user_results) ? core.user_results : null;
  let result = userResults && isRecord(userResults.result) ? userResults.result : null;
  if (result === null) {
    return null;
  }
  if (result.__typename === "UserUnavailable") {
    return null;
  }
  if (result.__typename === "UserWithVisibilityResults" && isRecord(result.user)) {
    result = result.user;
  }
  return result;
}

function extractMediaTypes(tweetLegacy: Record<string, unknown> | null): MediaKind[] {
  if (tweetLegacy === null) {
    return [];
  }
  const extEntities = isRecord(tweetLegacy.extended_entities)
    ? tweetLegacy.extended_entities
    : null;
  const entities = isRecord(tweetLegacy.entities) ? tweetLegacy.entities : null;
  const mediaList = Array.isArray(extEntities?.media)
    ? extEntities.media
    : Array.isArray(entities?.media)
      ? entities.media
      : null;

  if (mediaList === null) {
    return [];
  }

  const seen = new Set<MediaKind>();
  for (const item of mediaList) {
    if (!isRecord(item)) {
      continue;
    }
    if (item.type === "photo" || item.type === "video" || item.type === "animated_gif") {
      seen.add(item.type);
    }
  }
  return Array.from(seen);
}

function isTweetNode(obj: Record<string, unknown>): boolean {
  if (obj.__typename === "Tweet") {
    return true;
  }
  if (typeof obj.__typename === "string" && obj.__typename !== "") {
    return false;
  }
  if (typeof obj.rest_id !== "string" || !isRecord(obj.legacy)) {
    return false;
  }
  const core = isRecord(obj.core) ? obj.core : null;
  return (
    (core !== null && isRecord(core.user_results)) ||
    "conversation_id_str" in obj.legacy ||
    "full_text" in obj.legacy
  );
}

function extractSingleTweetMeta(tweet: Record<string, unknown>): GraphqlTweetMeta | null {
  const tweetLegacy = isRecord(tweet.legacy) ? tweet.legacy : null;
  const tweetIdStr = parseIdStr(tweet.rest_id) ?? parseIdStr(tweetLegacy?.id_str);
  if (tweetIdStr === null) {
    return null;
  }

  const userNode = unwrapUserNode(tweet);
  const userLegacy = userNode && isRecord(userNode.legacy) ? userNode.legacy : null;
  const userCore = userNode && isRecord(userNode.core) ? userNode.core : null;
  const userVerification =
    userNode && isRecord(userNode.verification) ? userNode.verification : null;
  const userRel =
    userNode && isRecord(userNode.relationship_perspectives)
      ? userNode.relationship_perspectives
      : null;
  const userBio = userNode && isRecord(userNode.profile_bio) ? userNode.profile_bio : null;
  const userAvatar = userNode && isRecord(userNode.avatar) ? userNode.avatar : null;

  const authorRestId =
    parseIdStr(userNode?.rest_id) ?? parseIdStr(tweetLegacy?.user_id_str) ?? null;
  const screenName =
    parseHandle(userCore?.screen_name) ?? parseHandle(userLegacy?.screen_name) ?? null;
  const displayName =
    readCappedString(userCore?.name, MAX_DISPLAY_NAME_LEN) ??
    readCappedString(userLegacy?.name, MAX_DISPLAY_NAME_LEN);
  const authorCreatedAtMs = parseXDateMs(userCore?.created_at ?? userLegacy?.created_at);
  const tweetCreatedAtMs = parseXDateMs(tweetLegacy?.created_at);
  const inReplyToStatusId = parseIdStr(tweetLegacy?.in_reply_to_status_id_str);
  const conversationId = parseIdStr(tweetLegacy?.conversation_id_str);
  const description =
    readCappedString(userLegacy?.description, MAX_DESCRIPTION_LEN) ??
    readCappedString(userBio?.description, MAX_DESCRIPTION_LEN);

  const isBlueVerified =
    userNode === null
      ? null
      : readNullableBool(userNode.is_blue_verified) ??
        readNullableBool(userVerification?.is_blue_verified) ??
        readNullableBool(userLegacy?.verified);

  const rawVerifiedType = (
    readString(userVerification?.verified_type) ??
    readString(userLegacy?.verified_type) ??
    ""
  ).toLowerCase();

  let verifiedType: VerifiedType = "none";
  if (rawVerifiedType === "business") {
    verifiedType = "business";
  } else if (rawVerifiedType === "government") {
    verifiedType = "government";
  } else if (rawVerifiedType === "blue" || isBlueVerified === true) {
    verifiedType = "blue";
  } else if (
    userNode !== null &&
    (userVerification?.verified === true || userLegacy?.verified === true)
  ) {
    verifiedType = "unknown_verified";
  }

  const following =
    readNullableBool(userRel?.following) ?? readNullableBool(userLegacy?.following);
  const followedBy =
    readNullableBool(userRel?.followed_by) ?? readNullableBool(userLegacy?.followed_by);

  const followersCount = parseNonNegInt(userLegacy?.followers_count);
  const friendsCount = parseNonNegInt(userLegacy?.friends_count);
  const statusesCount = parseNonNegInt(userLegacy?.statuses_count);

  const avatarUrl =
    readString(userAvatar?.image_url) ?? readString(userLegacy?.profile_image_url_https);
  const defaultProfileImage =
    userLegacy !== null && typeof userLegacy.default_profile_image === "boolean"
      ? userLegacy.default_profile_image
      : avatarUrl !== null
        ? avatarUrl.includes("default_profile_images/")
        : null;

  const favoriteCount = parseNonNegInt(tweetLegacy?.favorite_count);
  const replyCount = parseNonNegInt(tweetLegacy?.reply_count);
  const retweetCount = parseNonNegInt(tweetLegacy?.retweet_count);
  const mediaTypes = extractMediaTypes(tweetLegacy);

  return {
    tweetId: tweetIdStr as TweetId,
    authorRestId,
    screenName,
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

/**
 * Schema-defensive recursive GraphQL TweetDetail / TweetResultByRestId walker.
 * Unwraps Tweet and TweetWithVisibilityResults independently by tweetId so
 * nested quoted_status_result / retweeted_status_result never pollute outer tweets.
 */
export function extractGraphqlTweetMetasFromJson(root: unknown): GraphqlTweetMeta[] {
  const byTweetId = new Map<TweetId, GraphqlTweetMeta>();
  const visited = new WeakSet<object>();

  const recordMeta = (meta: GraphqlTweetMeta | null): void => {
    if (meta === null || meta.tweetId === undefined) {
      return;
    }
    const existing = byTweetId.get(meta.tweetId);
    if (existing === undefined || (existing.screenName == null && meta.screenName != null)) {
      byTweetId.set(meta.tweetId, meta);
    }
  };

  const walk = (node: unknown, depth: number): void => {
    if (depth > MAX_WALK_DEPTH || typeof node !== "object" || node === null) {
      return;
    }
    if (visited.has(node)) {
      return;
    }
    visited.add(node);

    if (Array.isArray(node)) {
      for (const item of node) {
        walk(item, depth + 1);
      }
      return;
    }

    const obj = node as Record<string, unknown>;
    const typename = obj.__typename;
    if (typename === "TweetUnavailable" || typename === "TweetTombstone") {
      return;
    }
    if (
      isRecord(obj.promotedMetadata) &&
      (typename === "TimelineTweet" || "tweet_results" in obj)
    ) {
      return;
    }

    if (typename === "TweetWithVisibilityResults" && isRecord(obj.tweet)) {
      if (!visited.has(obj.tweet)) {
        visited.add(obj.tweet);
        recordMeta(extractSingleTweetMeta(obj.tweet));
        for (const val of Object.values(obj.tweet)) {
          walk(val, depth + 1);
        }
      }
      for (const [key, val] of Object.entries(obj)) {
        if (key !== "tweet") {
          walk(val, depth + 1);
        }
      }
      return;
    }

    if (isTweetNode(obj)) {
      recordMeta(extractSingleTweetMeta(obj));
    }

    for (const val of Object.values(obj)) {
      walk(val, depth + 1);
    }
  };

  walk(root, 0);
  return Array.from(byTweetId.values());
}

function resolveFetchUrl(input: RequestInfo | URL): string | null {
  try {
    if (typeof input === "string") {
      return input;
    }
    if (input instanceof URL) {
      return input.href;
    }
    if (typeof Request !== "undefined" && input instanceof Request) {
      return input.url;
    }
  } catch {
    return null;
  }
  return null;
}

function resolveTargetOrigin(win: Window): string {
  try {
    const origin = win.location?.origin;
    if (typeof origin === "string" && origin !== "" && origin !== "null") {
      return origin;
    }
  } catch {
    // Fallback below
  }
  return "*";
}

export function installGraphqlTap(targetWindow: Window): void {
  const markerHost = targetWindow as unknown as Record<symbol, unknown>;
  if (markerHost[INSTALLED_SYMBOL] === true) {
    return;
  }

  try {
    Object.defineProperty(targetWindow, INSTALLED_SYMBOL, {
      value: true,
      configurable: true,
    });
  } catch {
    return;
  }

  const tapBuffer = new Map<TweetId, GraphqlTweetMeta>();

  const postItems = (items: ReadonlyArray<GraphqlTweetMeta>): void => {
    if (items.length === 0) {
      return;
    }
    const msg: TapBatchMessage = {
      source: GRAPHQL_TAP_SOURCE,
      v: GRAPHQL_TAP_PROTOCOL_VERSION,
      items,
    };
    try {
      targetWindow.postMessage(msg, resolveTargetOrigin(targetWindow));
    } catch {
      // Ignore postMessage failure
    }
  };

  const emitFromJson = (json: unknown): void => {
    try {
      const items = extractGraphqlTweetMetasFromJson(json);
      if (items.length === 0) {
        return;
      }
      for (const item of items) {
        if (item.tweetId === undefined) {
          continue;
        }
        if (tapBuffer.has(item.tweetId)) {
          tapBuffer.delete(item.tweetId);
        } else if (tapBuffer.size >= MAX_TAP_BUFFER) {
          const oldest = tapBuffer.keys().next().value;
          if (oldest !== undefined) {
            tapBuffer.delete(oldest);
          }
        }
        tapBuffer.set(item.tweetId, item);
      }
      postItems(items);
    } catch {
      // Passive tap must never throw
    }
  };

  targetWindow.addEventListener("message", (event: MessageEvent) => {
    try {
      if (event.source !== targetWindow || !isRecord(event.data)) {
        return;
      }
      if (
        event.data.source === GRAPHQL_CONTENT_SOURCE &&
        event.data.type === "tap_ready" &&
        tapBuffer.size > 0
      ) {
        postItems(Array.from(tapBuffer.values()));
      }
    } catch {
      // Ignore message replay errors
    }
  });

  if (typeof targetWindow.fetch === "function") {
    const origFetch = targetWindow.fetch;
    const wrappedFetch: typeof targetWindow.fetch = function (
      this: unknown,
      input: RequestInfo | URL,
      init?: RequestInit,
    ): Promise<Response> {
      const url = resolveFetchUrl(input);
      const promise = origFetch.apply(
        this ?? targetWindow,
        init !== undefined ? [input, init] : [input],
      );

      if (url !== null && TARGET_GRAPHQL_URL_RE.test(url)) {
        void promise
          .then((response) => {
            try {
              if (!response || !response.ok || response.bodyUsed) {
                return;
              }
              const contentType = response.headers?.get("content-type") ?? "";
              if (contentType !== "" && !contentType.toLowerCase().includes("json")) {
                return;
              }
              const cloned = response.clone();
              void cloned
                .json()
                .then((json: unknown) => {
                  emitFromJson(json);
                })
                .catch(() => {});
            } catch {
              // Never disturb page fetch
            }
          })
          .catch(() => {});
      }

      return promise;
    };

    try {
      targetWindow.fetch = wrappedFetch;
    } catch {
      try {
        Object.defineProperty(targetWindow, "fetch", {
          value: wrappedFetch,
          configurable: true,
          writable: true,
        });
      } catch {
        // Non-writable fetch
      }
    }
  }

  const xhrProto = (
    targetWindow as unknown as {
      readonly XMLHttpRequest?: { readonly prototype?: XMLHttpRequest };
    }
  ).XMLHttpRequest?.prototype;
  if (xhrProto && typeof xhrProto.open === "function" && typeof xhrProto.send === "function") {
    const xhrUrls = new WeakMap<XMLHttpRequest, string>();
    const origOpen = xhrProto.open;
    const origSend = xhrProto.send;

    xhrProto.open = function (
      this: XMLHttpRequest,
      _method: string,
      url: string | URL,
    ): void {
      try {
        const urlStr = typeof url === "string" ? url : url instanceof URL ? url.href : String(url);
        xhrUrls.set(this, urlStr);
      } catch {
        // Ignore URL tracking failure
      }
      origOpen.apply(this, arguments as unknown as Parameters<typeof origOpen>);
    };

    xhrProto.send = function (
      this: XMLHttpRequest,
    ): void {
      try {
        const url = xhrUrls.get(this);
        if (url !== undefined && TARGET_GRAPHQL_URL_RE.test(url)) {
          this.addEventListener(
            "load",
            function onXrfXhrLoad() {
              try {
                if (this.status < 200 || this.status >= 300) {
                  return;
                }
                if (this.responseType === "json") {
                  if (this.response !== null && typeof this.response === "object") {
                    emitFromJson(this.response);
                  }
                } else if (this.responseType === "" || this.responseType === "text") {
                  const text = this.responseText;
                  if (typeof text === "string" && text.length > 0) {
                    emitFromJson(JSON.parse(text));
                  }
                }
              } catch {
                // Passive tap must never throw
              }
            },
            { once: true },
          );
        }
      } catch {
        // Ignore XHR hook setup error
      }
      origSend.apply(this, arguments as unknown as Parameters<typeof origSend>);
    };
  }
}

if (
  typeof window !== "undefined" &&
  typeof window.document !== "undefined" &&
  typeof window.postMessage === "function"
) {
  installGraphqlTap(window);
}
