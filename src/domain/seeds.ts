import {
  type AuthorId,
  type AuthorStats,
  type DomSignals,
  EMBEDDING_DIM,
  SIAMESE_RELATIONAL_DIM,
  SIAMESE_TOTAL_DIM,
  type SupportedLanguage,
  type UnitVec256,
  truncateAndNormalize256,
} from "./types.ts";

export interface SeedExample {
  readonly lang: SupportedLanguage;
  readonly label: 1 | -1;
  readonly text: string;
}

export interface SeedPrototype {
  readonly lang: SupportedLanguage;
  readonly label: 1 | -1;
  readonly embedding: UnitVec256;
}

export interface OnlineClassifierWeights {
  readonly weights: Float32Array;
  readonly bias: number;
  readonly seedPrototypes: ReadonlyArray<SeedPrototype>;
  readonly updateCount: number;
  readonly authorStats: ReadonlyMap<AuthorId, AuthorStats>;
}

const SPAM_PATTERNS: ReadonlyArray<{ readonly name: string; readonly regex: RegExp }> = [
  { name: "en_bio_bait", regex: /\b(check\s+my\s+(bio|pinned)|link\s+in\s+bio|onlyfans\.com)\b/i },
  { name: "en_crypto_scam", regex: /\b(claim\s+your\s+airdrop|send\s+\d+\s*(sol|eth|btc)|whatsapp\s*\+?\d{5,}|t\.me\/[a-z0-9_]+)/i },
  { name: "zh_profile_bait", regex: /(看主页|点我头像|置顶有惊喜|置顶自取|私信领|同城可约|全国空降|外围女|福利姬|电报群|飞机群)/ },
  { name: "ja_profile_bait", regex: /(プロフ(ィール)?(見て|確認)|固ツイ(見て|確認)|LINE追加で|裏アカ女子.*固ツイ)/ },
];

const BAIT_QUESTION_REGEX =
  /(what do you think|do you agree|thoughts on this|how do you see this|大家怎么看|你怎么看呢|对此你怎么看|皆さんはどう思いますか|どう思いますか)[?？!！\s]*$/i;

const CJK_CHAR_REGEX = /[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]/g;
const EMOJI_CHAR_REGEX = /\p{Extended_Pictographic}/gu;

export const SEED_CORPUS: ReadonlyArray<SeedExample> = [
  {
    lang: "en",
    label: 1,
    text: "Spot on! This is such a game-changer for the ecosystem. Most people overlook how important this really is. What do you think?",
  },
  {
    lang: "en",
    label: 1,
    text: "Great breakdown! Let that sink in. Here are 3 key takeaways everyone needs to remember going forward.",
  },
  {
    lang: "zh",
    label: 1,
    text: "总结得很到位！这确实是一个非常关键的转折点，很多人都忽略了背后的深层逻辑，大家怎么看？",
  },
  {
    lang: "zh",
    label: 1,
    text: "太真实了！看我主页置顶有更多干货福利，懂的来私信领取。",
  },
  {
    lang: "ja",
    label: 1,
    text: "本当にその通りですね！非常に素晴らしい視点だと思います。今後の展開がとても楽しみです。皆さんはどう思いますか？",
  },
  {
    lang: "ja",
    label: 1,
    text: "すごく共感しました！詳しい情報はプロフと固ツイにまとめているのでぜひ見てください！",
  },
  {
    lang: "en",
    label: -1,
    text: "We tried this on our staging cluster yesterday and hit a 12ms regression whenever the ring buffer wrapped.",
  },
  {
    lang: "zh",
    label: -1,
    text: "昨天升级驱动之后实测了一下，显存占用大概在220MB左右，比上一版稳定多了。",
  },
  {
    lang: "ja",
    label: -1,
    text: "手元の環境で試してみましたが、設定ファイルのパスを相対パスに変えると再現しなくなりました。",
  },
];

export function matchMultilingualSpamKeyword(text: string): string | null {
  for (const entry of SPAM_PATTERNS) {
    if (entry.regex.test(text)) {
      return entry.name;
    }
  }
  return null;
}

export function detectDominantLanguage(text: string): SupportedLanguage {
  if (/[\u3040-\u30ff]/.test(text)) {
    return "ja";
  }
  if (/[\u3400-\u4dbf\u4e00-\u9fff]/.test(text)) {
    return "zh";
  }
  return "en";
}

export function extractDomSignalsFromText(
  rawText: string,
  hasVerifiedBadge: boolean,
  hasMediaOrExternalLink: boolean,
): DomSignals {
  const trimmed = rawText.trim();
  const charCount = Math.max(1, Array.from(trimmed).length);
  const cjkMatches = trimmed.match(CJK_CHAR_REGEX);
  const emojiMatches = trimmed.match(EMOJI_CHAR_REGEX);
  const cjkCharRatio = Math.min(1, (cjkMatches ? cjkMatches.length : 0) / charCount);
  const emojiToCharRatio = Math.min(1, (emojiMatches ? emojiMatches.length : 0) / charCount);
  const endsWithBaitQuestion = BAIT_QUESTION_REGEX.test(trimmed);
  const matchedSpamKeyword = matchMultilingualSpamKeyword(trimmed);

  return {
    hasVerifiedBadge,
    hasMediaOrExternalLink,
    endsWithBaitQuestion,
    emojiToCharRatio,
    cjkCharRatio,
    matchedSpamKeyword,
  };
}

export function createDeterministicSeedPrototypes(): ReadonlyArray<SeedPrototype> {
  return SEED_CORPUS.map((item, idx) => {
    const raw = new Float32Array(EMBEDDING_DIM);
    for (let i = 0; i < EMBEDDING_DIM; i++) {
      const phase = (idx + 1) * 0.37 + i * 0.11;
      raw[i] = item.label * Math.sin(phase);
    }
    return {
      lang: item.lang,
      label: item.label,
      embedding: truncateAndNormalize256(raw, 0, EMBEDDING_DIM),
    };
  });
}

export function createInitialClassifierWeights(
  seedPrototypes: ReadonlyArray<SeedPrototype> = createDeterministicSeedPrototypes(),
  authorStats: ReadonlyMap<AuthorId, AuthorStats> = new Map(),
): OnlineClassifierWeights {
  const weights = new Float32Array(SIAMESE_TOTAL_DIM);
  weights[SIAMESE_RELATIONAL_DIM + 0] = 0.35;
  weights[SIAMESE_RELATIONAL_DIM + 1] = 0.25;
  weights[SIAMESE_RELATIONAL_DIM + 2] = 0.95;
  weights[SIAMESE_RELATIONAL_DIM + 3] = 0.85;
  weights[SIAMESE_RELATIONAL_DIM + 4] = 0.0;
  weights[SIAMESE_RELATIONAL_DIM + 5] = 1.8;

  return {
    weights,
    bias: -0.35,
    seedPrototypes,
    updateCount: 0,
    authorStats,
  };
}

export function recordAuthorOutcome(
  current: OnlineClassifierWeights,
  authorId: AuthorId,
  outcome: "user_spam" | "user_human" | "auto_flag",
): OnlineClassifierWeights {
  const nextMap = new Map(current.authorStats);
  const prev = nextMap.get(authorId) ?? {
    spamCount: 0,
    humanCount: 0,
    autoFlagCount: 0,
  };

  const updated: AuthorStats =
    outcome === "user_spam"
      ? { ...prev, spamCount: prev.spamCount + 1 }
      : outcome === "user_human"
        ? { ...prev, humanCount: prev.humanCount + 1 }
        : { ...prev, autoFlagCount: prev.autoFlagCount + 1 };

  nextMap.set(authorId, updated);
  return {
    ...current,
    authorStats: nextMap,
  };
}
