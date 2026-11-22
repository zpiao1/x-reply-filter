import type { GraphqlTweetMeta } from "../domain/types.ts";

export const GRAPHQL_TAP_SOURCE = "xrf-graphql-tap" as const;
export const GRAPHQL_CONTENT_SOURCE = "xrf-content" as const;
export const GRAPHQL_TAP_PROTOCOL_VERSION = 1 as const;
export const MAX_TAP_BUFFER = 256 as const;
export const TARGET_GRAPHQL_URL_RE = /\/graphql\/[^/]+\/(TweetDetail|TweetResultByRestId)\b/;

export interface TapBatchMessage {
  readonly source: typeof GRAPHQL_TAP_SOURCE;
  readonly v: typeof GRAPHQL_TAP_PROTOCOL_VERSION;
  readonly items: ReadonlyArray<GraphqlTweetMeta>;
}
