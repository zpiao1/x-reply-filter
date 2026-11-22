import {
  type AuthorStats,
  COMPACT_V2_DIM,
  type CandidateTweet,
  type CompactFeatureVec,
  type DomSignals,
  type LangBiasKey,
  type LangV2,
  type SignalFamily,
  type SupportedLanguage,
  type ThreadAnchorState,
  type UnitVec256,
  extractAuthorModelFeatures,
} from "./types.ts";

export type FeatureSign = 1 | -1 | 0;

export const FEATURE_IDS = [
  "e_op_cos",
  "e_has_anchor",
  "e_proto_slop",
  "e_proto_human",
  "e_proto_margin",
  "e_generic",
  "e_clone_max",
  "e_clone_cluster",
  "l_novelty",
  "l_jaccard",
  "l_char3_contain",
  "l_specificity",
  "l_first_person",
  "s_contrast",
  "s_syco_open",
  "s_bait_close",
  "s_slop_lex",
  "s_em_dash",
  "s_colon_reveal",
  "s_listicle",
  "s_uniform",
  "s_emoji_density",
  "s_emoji_only",
  "s_emoji_tail",
  "s_hashtags",
  "s_exclaim",
  "s_caps",
  "s_punct",
  "s_casual",
  "s_profanity",
  "s_informal_en",
  "s_log_tokens",
  "x_lang_mismatch",
  "x_mismatch_generic",
  "b_summon_pure",
  "b_summon_any",
  "m_meta_present",
  "m_blue",
  "m_org_verified",
  "m_unfollowed",
  "m_blue_unfollowed_short",
  "m_followed_by",
  "m_age",
  "m_followers",
  "m_ff_ratio",
  "m_rate",
  "m_default_avatar",
  "m_bio_spam",
  "m_name_spam",
  "m_handle_digits",
  "m_reply_likes_rel",
  "a_spam_rate",
  "a_human_rate",
  "a_spam_volume",
  "c_media_only",
  "c_has_media",
  "c_ext_link",
  "c_has_quote",
] as const;

export type FeatureId = (typeof FEATURE_IDS)[number];

export const FEATURE_SIGNS: ReadonlyArray<FeatureSign> = [
  0, // 0 e_op_cos
  0, // 1 e_has_anchor
  1, // 2 e_proto_slop
  -1, // 3 e_proto_human
  1, // 4 e_proto_margin
  1, // 5 e_generic
  1, // 6 e_clone_max
  1, // 7 e_clone_cluster
  -1, // 8 l_novelty
  0, // 9 l_jaccard
  1, // 10 l_char3_contain
  -1, // 11 l_specificity
  -1, // 12 l_first_person
  1, // 13 s_contrast
  1, // 14 s_syco_open
  1, // 15 s_bait_close
  1, // 16 s_slop_lex
  1, // 17 s_em_dash
  1, // 18 s_colon_reveal
  1, // 19 s_listicle
  1, // 20 s_uniform
  0, // 21 s_emoji_density
  1, // 22 s_emoji_only
  1, // 23 s_emoji_tail
  1, // 24 s_hashtags
  1, // 25 s_exclaim
  0, // 26 s_caps
  0, // 27 s_punct
  -1, // 28 s_casual
  -1, // 29 s_profanity
  -1, // 30 s_informal_en
  0, // 31 s_log_tokens
  1, // 32 x_lang_mismatch
  1, // 33 x_mismatch_generic
  1, // 34 b_summon_pure
  0, // 35 b_summon_any
  0, // 36 m_meta_present
  1, // 37 m_blue
  -1, // 38 m_org_verified
  1, // 39 m_unfollowed
  1, // 40 m_blue_unfollowed_short
  -1, // 41 m_followed_by
  -1, // 42 m_age
  0, // 43 m_followers
  1, // 44 m_ff_ratio
  1, // 45 m_rate
  1, // 46 m_default_avatar
  1, // 47 m_bio_spam
  1, // 48 m_name_spam
  1, // 49 m_handle_digits
  -1, // 50 m_reply_likes_rel
  1, // 51 a_spam_rate
  -1, // 52 a_human_rate
  1, // 53 a_spam_volume
  1, // 54 c_media_only
  0, // 55 c_has_media
  0, // 56 c_ext_link
  -1, // 57 c_has_quote
];

export const FEATURE_FAMILIES: ReadonlyArray<SignalFamily> = FEATURE_IDS.map((_, i) => {
  if (i === 0 || (i >= 6 && i <= 10)) return "relational";
  if (i >= 2 && i <= 5) return "embedding";
  if ((i >= 13 && i <= 30) || i === 12 || i === 11 || i === 31) return "style";
  if (i === 32 || i === 33) return "crosslang";
  if (i === 34 || i === 35) return "summon";
  if ((i >= 36 && i <= 50) || i === 54 || i === 55 || i === 56 || i === 57 || i === 1) return "meta";
  return "author";
});

// ================= Text Normalization & Language Detection (§3a) =================
const GRAPHEME_SEGMENTER = new Intl.Segmenter("und", { granularity: "grapheme" });
const EMOJI_GRAPHEME_RE = /\p{Extended_Pictographic}|\p{Regional_Indicator}/u;

const TRAD_CHARS = new Set(
  Array.from("這們說國會為個來時對麼發與後應實體區當過還經開關點頭東車長門問間見"),
);
const SIMP_CHARS = new Set(
  Array.from("这们说国会为个来时对么发与后应实体区当过还经开关点头东车长门问间见"),
);

export function normalizeTextForRegex(text: string): string {
  return text
    .normalize("NFKC")
    .replace(/[\u2018\u2019\u02bc]/g, "'")
    .replace(/[\u201c\u201d]/g, '"');
}

export function splitGraphemes(text: string): string[] {
  const out: string[] = [];
  for (const seg of GRAPHEME_SEGMENTER.segment(text)) {
    out.push(seg.segment);
  }
  return out;
}

export function detectLangV2(text: string, opLangHint?: LangV2 | null): LangV2 {
  const cleaned = text.replace(/https?:\/\/\S+/gi, "").replace(/@[a-zA-Z0-9_]+/g, "").trim();
  const kanaCount = (cleaned.match(/[\u3040-\u30ff\uff66-\uff9f]/g) ?? []).length;
  const hanMatches = cleaned.match(/[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]/g) ?? [];
  const hanCount = hanMatches.length;

  if (kanaCount >= 1 && kanaCount / Math.max(1, kanaCount + hanCount) >= 0.05) {
    return "ja";
  }

  if (hanCount > 0 && kanaCount === 0) {
    if (hanCount <= 3) {
      const latinCount = (cleaned.match(/[A-Za-z]/g) ?? []).length;
      if (latinCount >= 6) {
        return "en";
      }
      return "cjk-ambiguous";
    }
    let tradScore = 0;
    let simpScore = 0;
    for (const ch of hanMatches) {
      if (TRAD_CHARS.has(ch)) tradScore += 1;
      if (SIMP_CHARS.has(ch)) simpScore += 1;
    }
    if (tradScore > simpScore) return "zh-Hant";
    if (simpScore > tradScore) return "zh-Hans";
    return opLangHint === "zh-Hant" ? "zh-Hant" : "zh-Hans";
  }

  const allLetters = (cleaned.match(/\p{L}/gu) ?? []).length;
  const latinLetters = (cleaned.match(/[A-Za-z]/g) ?? []).length;
  if (allLetters > 0 && latinLetters / allLetters >= 0.5) {
    return "en";
  }
  return "other";
}

export function toLangBiasKey(lang: LangV2): LangBiasKey {
  if (lang === "en" || lang === "zh-Hans" || lang === "zh-Hant" || lang === "ja") {
    return lang;
  }
  return "other";
}

export function toSupportedLanguage(lang: LangV2): SupportedLanguage {
  if (lang === "ja") return "ja";
  if (lang === "zh-Hans" || lang === "zh-Hant" || lang === "cjk-ambiguous") return "zh";
  return "en";
}

// ================= Summon List & Lexicons (§3c, §3d, #13 micro-fix #1) =================
export const SUMMON_HANDLES = new Set<string>([
  "grok",
  "askperplexity",
  "threadreaderapp",
  "unrollhelper",
  "savetonotion",
  "readwiseio",
  "memdotai",
  "remindme_ofthis",
  "pikaso_me",
  "dearassistant",
]);

const SUMMON_QUERY_RES: ReadonlyArray<RegExp> = [
  /\bis (this|that|it) (true|real|legit|accurate|fake)\b/i,
  /\bexplain( this| that| it)?\b/i,
  /\bwhat('s| is) (this|that|happening|going on)\b/i,
  /\bsummari[sz]e\b/i,
  /\btrue\?/i,
  /\bfact[- ]?check\b/i,
  /\bunroll\b/i,
  /\bsave\b/i,
  /\btranslate\b/i,
  /\bverify\b/i,
  /(这是真的吗|是真的吗|真的假的|解释一下|总结一下|翻译|核实一下)/,
  /(這是真的嗎|是真的嗎|真的假的|解釋一下|總結一下|這是真的還是假)/,
  /(本当(ですか|？|\?)?|これ(本当|マジ)|解説して|要約して|ファクトチェック)/,
  /\?+$/,
];

const EN_STOPWORDS = new Set([
  "the", "a", "an", "and", "or", "but", "if", "in", "on", "at", "to", "for", "of", "with",
  "by", "from", "up", "about", "into", "over", "after", "is", "are", "was", "were", "be",
  "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "shall",
  "should", "can", "could", "may", "might", "must", "this", "that", "these", "those",
  "it", "its", "it's", "i", "you", "he", "she", "we", "they", "me", "him", "her", "us",
  "them", "my", "your", "his", "our", "their", "not", "no", "so", "very", "just", "too",
  "as", "than", "then", "now", "here", "there", "when", "where", "why", "how", "all",
  "any", "both", "each", "more", "most", "other", "some", "such", "only", "own", "same",
]);

const CONTRAST_RES: ReadonlyArray<RegExp> = [
  // EN
  /\b(it's|this is|that's|we're|they're|you're)?\s*not (just|only|merely|simply) (about )?[^.?!]{1,65}[,;—–-]+\s*(it's|but|this is|that's|we're) (about )?/gi,
  /\bisn't (just |merely |about )?[^.?!]{1,50}[—–,;]\s*it's\b/gi,
  /\baren't [^.?!]{1,45}[—–,;]\s*they're\b/gi,
  /\bless about [^.?!]{1,45},?\s*(and )?more about\b/gi,
  /\b[^.?!]{1,45} isn't the (problem|point|issue)[,;—–]\s*[^.?!]{1,45} is\b/gi,
  /\bthe real (question|issue|story|lesson|shift|revolution) (is|isn't|here)\b/gi,
  /\bthis isn't [^.?!]{1,45}\.\s*(it's|this is)\b/gi,
  /\bwas never about [^.?!]{1,45}[;—–,]\s*it's about\b/gi,
  /\bdoesn't just [^.?!]{1,35}[;—–,]\s*it\b/gi,
  // ZH-Hans
  /不(仅|只)(仅)?是.{1,35}(更|而)是/g,
  /这不是.{1,35}而是/g,
  /与其说.{1,35}不如说/g,
  /重点不是.{1,35}而是/g,
  /从来不是.{1,35}而是/g,
  /不再是.{1,30}而是/g,
  /真正的(问题|关键|壁垒)(是|在于|从来不是)/g,
  /表面上看.{1,30}本质上/g,
  // ZH-Hant
  /不(僅|只)(僅)?是.{1,35}(更|而)是/g,
  /這不是.{1,35}而是/g,
  /與其說.{1,35}不如說/g,
  /重點不是.{1,35}而是/g,
  /從來不是.{1,35}而是/g,
  /真正的(問題|關鍵|轉型關鍵)(是|在於|從來不是)/g,
  // JA
  /(だけ|のみ)(では|じゃ)な(く|い).{0,35}(も|こそ|そのもの)/g,
  /単なる.{1,25}(では|じゃ)な(く|い)/g,
  /.{1,30}というより(も)?.{1,30}/g,
  /(本当の|真の)(問題|ポイント)は/g,
  /(重要なのは|大切なのは).{1,30}(ではなく|じゃなく)/g,
];

const SYCO_OPEN_RES: ReadonlyArray<RegExp> = [
  /^\W*(great (point|take|thread|post|breakdown|insight|perspective)s?|spot on|so true|this is (so )?(true|spot on|brilliant|insightful|gold)|absolutely( love)?|couldn't (agree more|have said it better)|well said|(you )?nailed it|love this|fascinating|interesting (take|perspective)|this!+|exactly( this)?|what a (great|brilliant|powerful|fascinating)|powerful (words|message)|such an? (important|thoughtful|insightful)|100% (agree|truth)|massive w|amazing update|wild stat|facts\b)/i,
  /^\W*(说得(太|非常)?(好|透彻|对)|太真实了|总结得(很|非常|太)?到位|非常(赞同|有洞察力|有价值|认同)|确实如此|太对了|深有同感|讲得透彻|这个观点(很|非常|太)|精辟)/,
  /^\W*(說得(太|非常)?(好|透徹|對)|太真實了|總結得(很|非常|太)?到位|非常(贊同|精闢|認同)|確實如此|太對了|太精闢了|深有同感|講得透徹|很棒的觀點|這個觀點|認同)/,
  /^\W*((本当|まさ)に(その通り|本質|これ)|おっしゃる通り|素晴らしい(発信|視点)?|(すごく|とても|めちゃくちゃ)?共感(しました|します)|勉強になります|非常に(興味深い|参考になります|有益な)|大切な(視点|ポイント)|これは神知見)/,
];

const BAIT_CLOSE_RES: ReadonlyArray<RegExp> = [
  /(what do you think|thoughts( on this)?\?|agree( or disagree)?\?|do you agree|who else|let's discuss|drop (your|a) .{0,20} below|curious what others think|what's your take|follow for more|am i wrong\?|how do you see this|save this for later|bookmark this|read that again)[?!.\s\p{Extended_Pictographic}]*$/iu,
  /(大家怎么看|你怎么看|对此你怎么看|欢迎(讨论|留言)|评论区(聊聊|见)|你同意吗|建议收藏)[?!。？！\s\p{Extended_Pictographic}]*$/u,
  /(大家怎麼看|你怎麼看|對此你怎麼看|歡迎(討論|留言)|你同意嗎|按讚收藏)[?!。？！\s\p{Extended_Pictographic}]*$/u,
  /((皆さん|みなさん)は?どう(思|活用)|どう思いますか|ご意見(を|お)?聞かせ|皆さんの意見)[?!。？！\s\p{Extended_Pictographic}]*$/u,
];

const SLOP_LEX_RES: ReadonlyArray<RegExp> = [
  // EN
  /\bgame[- ]?changer\b/gi,
  /\bdelve\b/gi,
  /\btestament to\b/gi,
  /\btapestry\b/gi,
  /\blandscape\b/gi,
  /\bnavigat(e|ing) the\b/gi,
  /\bcrucial\b/gi,
  /\bpivotal\b/gi,
  /\bunderscores?\b/gi,
  /\bresonates?\b/gi,
  /\bunlock(s|ing)?\b/gi,
  /\belevate\b/gi,
  /\bin today's (fast-paced|competitive|world)\b/gi,
  /\blet that sink in\b/gi,
  /\bhere's the (thing|harsh truth)\b/gi,
  /\bat the end of the day\b/gi,
  /\bit's worth noting\b/gi,
  /\ba reminder that\b/gi,
  /\bthe power of\b/gi,
  /\bmindset\b/gi,
  /\bparadigm\b/gi,
  /\bdouble-edged sword\b/gi,
  /\bfood for thought\b/gi,
  /\bspot-on analysis\b/gi,
  /\btruly (remarkable|inspiring|appreciate)\b/gi,
  /\bcompound(s|ing)? (massively|daily|over time)\b/gi,
  /\blevel up\b/gi,
  /\bkeep building\b/gi,
  /\bstay winning\b/gi,
  /\bmost people (overlook|are completely sleeping|don't realize)\b/gi,
  // ZH-Hans
  /(赋能|底层逻辑|闭环|值得深思|至关重要|不可忽视|颗粒度|抓手|发人深省|深刻(的)?(洞察|见解)|一针见血|干货满满|长期主义|认知升级|受益匪浅|持续精进|深度输出)/g,
  // ZH-Hant
  /(賦能|底層邏輯|閉環|值得深思|至關重要|不可忽視|發人深省|深刻(的)?(洞察|見解)|一針見血|乾貨滿滿|長期主義|收穫滿滿|護城河|啟發性)/g,
  // JA
  /(非常に興味深い|本質的|示唆に富む|重要なポイント|今後の展開が(とても)?楽しみ|考えさせられ|素晴らしい視点|深い洞察|保存必須|学びの多い|大変参考に|本質を突いた)/g,
];

const COLON_REVEAL_RE =
  /(here's why:|the key:|bottom line:|the truth:|harsh truth[^:]*:|关键在于[:：]|说白了[:：]|核心在于[:：]|ポイントは[:：]|結論[:：])/i;

const LISTICLE_BULLET_RE = /(?:^|\n)\s*(?:\d+[.)]|[•\-–*]|[①②③④⑤]|✅|👉|🚀|・)\s+\S/gm;
const LISTICLE_PHRASE_RE =
  /(\b\d+\s+key takeaways\b|\bthree (things|reasons|takeaways)\b|首先.{1,40}其次|第一.{1,40}第二|三个核心重点|三個核心重點|まず.{1,40}次に)/i;

// #13 micro-fix #1: bare EN `w` and `facts` excluded so unfollowed blue-check hype replies
// ("Facts 💯🔥", "W take 🔥") aren't exempted as short_human_reaction.
const CASUAL_EN_RE =
  /\b(lol+|lmao+|lmfao|rofl|wtf|omg|bruh|fr|ngl|tbh|imo|idk|smh|based|ratio|l|real|yikes|oof|damn|nah|yep|kek|cope|mid|sus)\b/i;
const CASUAL_CJK_RE =
  /(哈{2,}|笑死|卧槽|臥槽|我靠|靠北|绝了|絕了|离谱|離譜|牛逼|草|6{2,}|2333*|h{3,}|啊这|好家伙|真的假的|真假|好扯|傻眼|蚌埠住了|破防了|确实|確實|无语|w{2,}|ｗ{2,}|ワロタ|わろた|それな|まじ(か|で)|マジ(か|で)|えぐい|草生え|ほんこれ|わかる|了解)/;

const PROFANITY_RE =
  /(\b(fuck|fucking|shit|bullshit|asshole|bitch|crap|hell|bs)\b|卧槽|臥槽|他妈|他媽|靠北|傻逼|牛逼|クソ|くそ|ゴミ)/i;

const INFORMAL_EN_TYPO_RE =
  /\b(u|ur|im|dont|cant|wont|didnt|doesnt|youre|thats|tbh|idk|lmao|lol|omg|bruh|rn|af|imo)\b/i;

const FIRST_PERSON_RE =
  /(\b(i tried|i've been|i spent|i tested|i read|i watched|i saw|we ran|we tried|we saw|we bind|we're still|my (team|kid|car|cat|umbrella|wallet|solo queue|2080|bambino|coat)|learned that the hard way|last (week|month|year|tuesday))\b|(我(试|测|以前|去年|上个月|之前|们团队|们后来|们去年|家|阳台|钱包)|实测|上周|昨天|去年の|先週|うちも|うちの|手元の|実際に|私も|試してみ|使ってます))/i;

const BIO_SPAM_STRONG_RE =
  /(dm\s+(for|me)\s+(promo|collab|'scale')|airdrop|100x|onlyfans|signal group|free telegram|forex|(trading|crypto) mentor|whatsapp\s*\+?\d|t\.me\/|带单|空投|兼职|福利|同城|全国空降|代理招募|副業|稼げる|LINE(追加|登録)|固ツイ見て|飙股|飆股|免費進群|私信领|置顶|置頂)/i;
const BIO_SPAM_WEAK_RE =
  /(\bcrypto\b|\bnft\b|\bweb3\b|ai enthusiast|\bbuilder\b|daily frameworks|scale smarter|saas builder|startup ops|币圈|投资|個人品牌|AI出海|一人公司|仮想通貨|投資|AI活用|毎日発信)/i;

const NAME_SPAM_RE =
  /(\$[A-Z]{2,6}\b|\bweb3\b|\|\s*(ai|growth|saas|startup|7-figure|indie|tech)|\bdm\b|🚀|\bnft\b|副業|币圈|合作|置顶|置頂|免費進群|プロフ必読|億り人)/i;

// Tightened t.me/ requiring bait verb per #6 §3c
export const SPAM_PATTERNS_V2: ReadonlyArray<{ readonly name: string; readonly regex: RegExp }> = [
  { name: "en_bio_bait", regex: /\b(check\s+my\s+(bio|pinned)|link\s+in\s+bio|onlyfans\.com)\b/i },
  {
    name: "en_crypto_scam",
    regex:
      /\b(claim\s+your\s+airdrop|send\s+\d+\s*(sol|eth|btc)|whatsapp\s*\+?\d{5,}|(join|check|visit|click|dm|message|加入|進入|参加).{0,30}t\.me\/[a-z0-9_]+)/i,
  },
  {
    name: "zh_profile_bait",
    regex:
      /(看主页|看主頁|点我头像|點我頭像|置顶有惊喜|置頂有驚喜|置顶自取|置頂自取|私信领|私信領|同城可约|全国空降|外围女|福利姬|电报群|飞机群)/,
  },
  {
    name: "ja_profile_bait",
    regex: /(プロフ(ィール)?(見て|確認|と固ツイ)|固ツイ(見て|確認)|LINE追加で|裏アカ女子.*固ツイ)/,
  },
];

export function matchKeywordSpamV2(text: string): string | null {
  const norm = normalizeTextForRegex(text);
  for (const p of SPAM_PATTERNS_V2) {
    if (p.regex.test(norm)) return p.name;
  }
  return null;
}

export const matchMultilingualSpamKeyword = matchKeywordSpamV2;

export function detectDominantLanguage(text: string): SupportedLanguage {
  return toSupportedLanguage(detectLangV2(text));
}

const CJK_CHAR_REGEX = /[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]/g;
const EMOJI_CHAR_REGEX = /\p{Extended_Pictographic}/gu;

export function extractDomSignalsFromText(
  rawText: string,
  hasVerifiedBadge: boolean,
  hasMediaOrExternalLink: boolean,
): DomSignals {
  const trimmed = rawText.trim();
  const norm = normalizeTextForRegex(trimmed);
  const charCount = Math.max(1, Array.from(trimmed).length);
  const cjkMatches = trimmed.match(CJK_CHAR_REGEX);
  const emojiMatches = trimmed.match(EMOJI_CHAR_REGEX);
  const cjkCharRatio = Math.min(1, (cjkMatches ? cjkMatches.length : 0) / charCount);
  const emojiToCharRatio = Math.min(1, (emojiMatches ? emojiMatches.length : 0) / charCount);
  const endsWithBaitQuestion = BAIT_CLOSE_RES.some((re) => re.test(norm));
  const matchedSpamKeyword = matchKeywordSpamV2(trimmed);

  return {
    hasVerifiedBadge,
    hasMediaOrExternalLink,
    endsWithBaitQuestion,
    emojiToCharRatio,
    cjkCharRatio,
    matchedSpamKeyword,
  };
}

export function stripLeadingMentionsAndUrls(text: string): string {
  let out = text.replace(/https?:\/\/\S+/gi, " ");
  out = out.replace(/^(?:\s*@[a-zA-Z0-9_]+)+/g, "");
  return out.trim();
}

export function extractContentUnits(text: string): string[] {
  const norm = normalizeTextForRegex(stripLeadingMentionsAndUrls(text));
  const units: string[] = [];

  const words = norm.toLowerCase().match(/[a-z0-9][a-z0-9'_/-]*/g) ?? [];
  for (const w of words) {
    if (w.length >= 2 && !EN_STOPWORDS.has(w)) {
      units.push(w);
    }
  }

  const cjkRuns = norm.match(/[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+/g) ?? [];
  for (const run of cjkRuns) {
    const chars = Array.from(run);
    if (chars.length === 1) {
      units.push(chars[0]!);
    } else {
      for (let i = 0; i < chars.length - 1; i++) {
        units.push(chars[i]! + chars[i + 1]!);
      }
    }
  }

  return units;
}

export function estimateContentTokenCount(text: string): number {
  const cleaned = stripLeadingMentionsAndUrls(text).replace(/@[a-zA-Z0-9_]+/g, " ").trim();
  if (cleaned.length === 0) return 0;
  const words = cleaned.match(/[a-zA-Z0-9]+(?:['’_-][a-zA-Z0-9]+)*/g) ?? [];
  const cjk = cleaned.match(/[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]/g) ?? [];
  const est = Math.round(words.length * 1.0 + cjk.length * 0.75);
  return Math.max(est, words.length > 0 || cjk.length > 0 ? 1 : 0);
}

function char3Grams(text: string): Set<string> {
  const clean = normalizeTextForRegex(stripLeadingMentionsAndUrls(text))
    .toLowerCase()
    .replace(/\s+/g, "");
  const chars = Array.from(clean);
  const grams = new Set<string>();
  for (let i = 0; i <= chars.length - 3; i++) {
    grams.add(chars[i]! + chars[i + 1]! + chars[i + 2]!);
  }
  return grams;
}

export function countSpecificity(replyText: string, opText: string): number {
  const normReply = normalizeTextForRegex(stripLeadingMentionsAndUrls(replyText));
  const normOpLower = normalizeTextForRegex(opText).toLowerCase();
  let hits = 0;

  const numMatches =
    normReply.match(
      /(\$[\d,.]+[KMBkmb]?|\b\d+(?:\.\d+)?\s*(?:ms|s|sec|min|hr|hrs|days?|months?|years?|gb|mb|kb|tb|mhz|ghz|fps|bps|%|k|m|b|x|mm|cm|km|ft|px|f|c|k|w|mw|gw|円|万円|億円|万|亿|元|港元|度|分|秒|時間|集|話|年|月|日|層|倍)\b|\bv\d+\.\d+(?:\.\d+)?\b|\bCVE-\d+-\d+\b|\b(?:0050|006208|PTM7950|N2|N3E?|H100|2080|9950X3D|10-Q|40mm)\b)/gi,
    ) ?? [];
  for (const m of numMatches) {
    if (!normOpLower.includes(m.toLowerCase()) && !/^(99%|3\s+key|100%)$/i.test(m.trim())) {
      hits += 1;
    }
  }

  const cashtags = normReply.match(/\$[A-Z]{2,5}\b/g) ?? [];
  for (const c of cashtags) {
    if (!normOpLower.includes(c.toLowerCase())) hits += 1;
  }

  const codeSpans = normReply.match(/(`[^`]{2,40}`|--[a-z][a-z0-9-]{3,}|\.[a-z]{2,5}\b)/g) ?? [];
  hits += codeSpans.length;

  const quotes = normReply.match(/["「『']([^"」』']{4,40})["」』']/g) ?? [];
  for (const q of quotes) {
    const inner = q.slice(1, -1).toLowerCase();
    if (!normOpLower.includes(inner)) hits += 1;
  }

  const midCap = normReply.match(/(?<=[a-z,]\s+)[A-Z][a-zA-Z0-9]{2,}\b/g) ?? [];
  const capBlacklist = new Set([
    "This", "That", "The", "Here", "Most", "True", "Great", "Spot", "Love", "What",
    "When", "Why", "How", "Save", "Read", "Follow", "Agree", "Facts", "Massive",
  ]);
  for (const w of midCap) {
    if (!capBlacklist.has(w) && !normOpLower.includes(w.toLowerCase())) {
      hits += 1;
    }
  }

  const katakana = normReply.match(/[\u30a0-\u30ffー]{4,}/g) ?? [];
  const kataGeneric = new Set([
    "ビジネス",
    "マーケティング",
    "ツール",
    "ポイント",
    "プロセス",
    "プロダクト",
    "グローバル",
    "ブックマーク",
    "テクニック",
    "メンタル",
    "エントリー",
    "キャッシュレス",
    "リマスター",
    "オンボーディング",
    "ローカル",
    "フォロー",
  ]);
  for (const k of katakana) {
    if (!kataGeneric.has(k) && !normOpLower.includes(k.toLowerCase())) {
      hits += 1;
    }
  }

  return hits;
}

function cap(x: number, k: number): number {
  return Math.min(Math.max(0, x), k) / k;
}

function clamp(x: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, x));
}

export function cosineSim256(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const n = Math.min(a.length, b.length);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    sum += (a[i] ?? 0) * (b[i] ?? 0);
  }
  return clamp(sum, -1, 1);
}

export function topKMeanCosine(
  vec: ArrayLike<number>,
  protos: ReadonlyArray<ArrayLike<number>>,
  k = 3,
): number {
  if (protos.length === 0) return 0;
  const sims = protos.map((p) => cosineSim256(vec, p)).sort((a, b) => b - a);
  const take = Math.min(k, sims.length);
  let sum = 0;
  for (let i = 0; i < take; i++) {
    sum += sims[i]!;
  }
  return sum / take;
}

/**
 * Deterministic 8-bit sign-LSH over the 256d embedding for replay-buffer topic bucketing (§6).
 */
export function computeTopicHash8(embedding: ArrayLike<number>): number {
  let hash = 0;
  for (let bit = 0; bit < 8; bit++) {
    let acc = 0;
    const stride = 32;
    const base = bit * stride;
    for (let j = 0; j < stride; j++) {
      const sign = ((bit * 7 + j * 13) & 1) === 0 ? 1 : -1;
      acc += sign * (embedding[base + j] ?? 0);
    }
    if (acc >= 0) {
      hash |= 1 << bit;
    }
  }
  return hash & 0xff;
}

export interface PrototypeBankV2 {
  readonly slop: ReadonlyArray<ArrayLike<number>>;
  readonly human: ReadonlyArray<ArrayLike<number>>;
  readonly genericCentroid: ArrayLike<number>;
}

export interface ThreadCloneContext {
  readonly maxCloneSim: number;
  readonly clusterSize: number;
}

export interface CompactFeatureExtraction {
  readonly vector: CompactFeatureVec;
  readonly replyLang: LangV2;
  readonly opLang: LangV2 | null;
  readonly contentTokenCount: number;
  readonly opTokenCount: number;
  readonly isPureSummon: boolean;
  readonly summonHandle: string | null;
  readonly isBotAuthor: boolean;
  readonly matchedSpamKeyword: string | null;
  readonly specificityHits: number;
  readonly evidence: Readonly<Record<string, string>>;
}

export function extractCompactFeaturesV2(args: {
  readonly candidate: CandidateTweet;
  readonly replyVec: UnitVec256;
  readonly anchor: ThreadAnchorState;
  readonly protoBank: PrototypeBankV2;
  readonly cloneCtx: ThreadCloneContext;
  readonly authorStats?: AuthorStats | undefined;
  readonly nowMs?: number | undefined;
}): CompactFeatureExtraction {
  const { candidate, replyVec, anchor, protoBank, cloneCtx, authorStats } = args;
  const nowMs = args.nowMs ?? Date.now();
  const vec = new Float32Array(COMPACT_V2_DIM);
  const evidence: Record<string, string> = {};
  const meta = candidate.graphqlMeta ?? null;
  const rawText = candidate.rawText.trim();
  const normText = normalizeTextForRegex(rawText);

  const opText = anchor.kind !== "unanchored" ? (anchor.rawText ?? "") : "";
  const opTokenCount =
    anchor.kind !== "unanchored"
      ? opText.trim().length > 0
        ? estimateContentTokenCount(opText)
        : anchor.tokenCount
      : 0;
  const opLang = opText.trim().length > 0 ? detectLangV2(opText) : null;
  const replyLang = rawText.length > 0 ? detectLangV2(rawText, opLang) : "other";
  const hasAnchor = anchor.kind === "text_anchored";

  // @summon detection (§3d)
  const mentionMatches = rawText.match(/@([a-zA-Z0-9_]+)/g) ?? [];
  const summonMentions = mentionMatches.filter((m) =>
    SUMMON_HANDLES.has(m.slice(1).toLowerCase()),
  );
  const hasSummonMention = summonMentions.length > 0;
  const summonHandle = summonMentions[0] ?? null;
  const isBotAuthor = SUMMON_HANDLES.has(candidate.authorId.toLowerCase());
  const residualText = normText
    .replace(/https?:\/\/\S+/gi, " ")
    .replace(/@[a-zA-Z0-9_]+/g, " ")
    .trim();
  const contentNoSummon = rawText
    .replace(/https?:\/\/\S+/gi, " ")
    .replace(/@([a-zA-Z0-9_]+)/g, (full, handle: string) =>
      SUMMON_HANDLES.has(handle.toLowerCase()) ? " " : full,
    )
    .trim();
  const contentTokenCount = estimateContentTokenCount(contentNoSummon);

  // Content units & overlap (8..12)
  const replyUnits = extractContentUnits(rawText);
  const opUnits = opText.trim().length > 0 ? extractContentUnits(opText) : [];
  const replySet = new Set(replyUnits);
  const opSet = new Set(opUnits);

  let intersection = 0;
  for (const u of replySet) {
    if (opSet.has(u)) intersection += 1;
  }
  const unionSize = replySet.size + opSet.size - intersection;
  const novelty =
    !hasAnchor || opSet.size === 0 || replySet.size === 0
      ? 1
      : (replySet.size - intersection) / replySet.size;
  const jaccard = !hasAnchor || opSet.size === 0 || unionSize === 0 ? 0 : intersection / unionSize;

  const r3 = char3Grams(rawText);
  const o3 = opText.trim().length > 0 ? char3Grams(opText) : new Set<string>();
  let c3Hit = 0;
  for (const g of r3) {
    if (o3.has(g)) c3Hit += 1;
  }
  const char3Contain = !hasAnchor || o3.size === 0 || r3.size === 0 ? 0 : c3Hit / r3.size;

  const specificityHits = countSpecificity(rawText, opText);
  const firstPerson = FIRST_PERSON_RE.test(normText) ? 1 : 0;

  // Pure summon guard (§3d)
  const residualEnWords = (residualText.match(/[a-zA-Z0-9']+/g) ?? []).length;
  const residualCjkGraphemes = (
    residualText.match(/[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]/g) ?? []
  ).length;
  const residualWithinLimit = residualEnWords <= 8 && residualCjkGraphemes <= 14;
  const residualMatchesTemplate =
    residualText.length === 0 || SUMMON_QUERY_RES.some((r) => r.test(residualText));
  const anchorAuthorId = anchor.kind !== "unanchored" ? anchor.authorId : null;
  const isShieldedAuthor =
    (anchorAuthorId !== null && candidate.authorId === anchorAuthorId) ||
    meta?.following === true;
  const isPureSummon =
    hasSummonMention &&
    residualWithinLimit &&
    residualMatchesTemplate &&
    specificityHits === 0 &&
    firstPerson === 0 &&
    !isShieldedAuthor &&
    estimateContentTokenCount(residualText) < 12;

  if (isPureSummon && summonHandle !== null) {
    evidence.b_summon_pure = summonHandle;
  }

  // Embedding features (0..7)
  const opCos =
    anchor.kind === "text_anchored" ? cosineSim256(replyVec, anchor.embedding) : 0;
  const protoSlop = rawText.length > 0 ? topKMeanCosine(replyVec, protoBank.slop, 3) : 0;
  const protoHuman = rawText.length > 0 ? topKMeanCosine(replyVec, protoBank.human, 3) : 0;
  const protoMargin = protoSlop - protoHuman;
  const genericCos =
    rawText.length > 0 ? cosineSim256(replyVec, protoBank.genericCentroid) : 0;

  vec[0] = opCos;
  vec[1] = hasAnchor ? 1 : 0;
  vec[2] = protoSlop;
  vec[3] = protoHuman;
  vec[4] = protoMargin;
  vec[5] = genericCos;
  vec[6] = cloneCtx.maxCloneSim;
  vec[7] = Math.log1p(Math.max(0, cloneCtx.clusterSize - 1)) / Math.log(16);

  vec[8] = novelty;
  vec[9] = jaccard;
  vec[10] = char3Contain;
  vec[11] = cap(specificityHits, 3);
  vec[12] = firstPerson;

  // Style features (13..31)
  let contrastMatches = 0;
  for (const re of CONTRAST_RES) {
    re.lastIndex = 0;
    const m = normText.match(re);
    if (m && m.length > 0) {
      contrastMatches += m.length;
      if (!evidence.s_contrast) evidence.s_contrast = m[0].slice(0, 32);
    }
  }
  vec[13] = cap(contrastMatches, 2);

  const graphemes = splitGraphemes(rawText);
  const first40 = normalizeTextForRegex(graphemes.slice(0, 40).join(""));
  let sycoOpen = 0;
  for (const re of SYCO_OPEN_RES) {
    const m = re.exec(first40);
    if (m) {
      sycoOpen = 1;
      evidence.s_syco_open = m[0].slice(0, 32);
      break;
    }
  }
  vec[14] = sycoOpen;

  let baitClose = 0;
  for (const re of BAIT_CLOSE_RES) {
    const m = re.exec(normText);
    if (m) {
      baitClose = 1;
      evidence.s_bait_close = m[0].slice(0, 32);
      break;
    }
  }
  vec[15] = baitClose;

  let slopHits = 0;
  for (const re of SLOP_LEX_RES) {
    re.lastIndex = 0;
    const m = normText.match(re);
    if (m) {
      slopHits += m.length;
      if (!evidence.s_slop_lex) evidence.s_slop_lex = m[0].slice(0, 32);
    }
  }
  const unitDenom = Math.max(10, replyUnits.length);
  vec[16] = cap((slopHits / unitDenom) * 50, 2);

  // Latin-context em-dash only (ZH/JA "——" and "…" excluded)
  const latinEmDashes =
    replyLang === "en"
      ? (
          rawText.match(/(?:[A-Za-z0-9]\s*[—]\s*[A-Za-z0-9]|[A-Za-z0-9]\s+–\s+[A-Za-z0-9])/g) ??
          []
        ).length
      : 0;
  vec[17] = cap(latinEmDashes, 3);

  vec[18] = COLON_REVEAL_RE.test(normText) ? 1 : 0;

  LISTICLE_BULLET_RE.lastIndex = 0;
  const bulletCount = (normText.match(LISTICLE_BULLET_RE) ?? []).length;
  vec[19] = bulletCount >= 2 || LISTICLE_PHRASE_RE.test(normText) ? 1 : 0;

  // Sentence-length uniformity
  const sentences = normText
    .split(/[.!?。！？\n]+/)
    .map((s) => s.trim())
    .filter((s) => s.length >= 4);
  if (sentences.length >= 3) {
    const lens = sentences.map((s) => s.length);
    const mean = lens.reduce((a, b) => a + b, 0) / lens.length;
    const variance = lens.reduce((a, b) => a + (b - mean) ** 2, 0) / lens.length;
    const cv = mean > 0 ? Math.sqrt(variance) / mean : 1;
    vec[20] = 1 - Math.min(1, cv);
  } else {
    vec[20] = 0;
  }

  const gLen = Math.max(1, graphemes.length);
  let emojiCount = 0;
  for (const g of graphemes) {
    if (EMOJI_GRAPHEME_RE.test(g)) emojiCount += 1;
  }
  vec[21] = rawText.length > 0 ? Math.min(1, emojiCount / gLen) : 0;

  const nonEmojiText = graphemes
    .filter((g) => !EMOJI_GRAPHEME_RE.test(g))
    .join("")
    .replace(/[\s\p{P}\p{S}]/gu, "");
  vec[22] = rawText.length > 0 && emojiCount > 0 && nonEmojiText.length === 0 ? 1 : 0;
  if (vec[22] > 0) evidence.s_emoji_only = rawText.slice(0, 16);

  let trailingEmojis = 0;
  for (let i = graphemes.length - 1; i >= 0; i--) {
    const g = graphemes[i]!;
    if (/^\s+$/.test(g)) continue;
    if (EMOJI_GRAPHEME_RE.test(g)) trailingEmojis += 1;
    else break;
  }
  vec[23] = trailingEmojis >= 2 ? 1 : 0;

  const hashtags = (rawText.match(/#[A-Za-z0-9_\u3040-\u9fff]+/g) ?? []).length;
  vec[24] = cap(hashtags, 3);

  const exclaims = (rawText.match(/[!！]/g) ?? []).length;
  vec[25] = rawText.length > 0 ? Math.min(1, (exclaims / gLen) * 10) : 0;

  const latinChars = rawText.match(/[A-Za-z]/g) ?? [];
  const upperChars = rawText.match(/[A-Z]/g) ?? [];
  vec[26] = latinChars.length >= 20 ? upperChars.length / latinChars.length : 0;

  const punctChars = (rawText.match(/[\p{P}]/gu) ?? []).length;
  vec[27] = rawText.length > 0 ? Math.min(1, punctChars / gLen) : 0;

  const casualMatch = CASUAL_EN_RE.exec(normText) ?? CASUAL_CJK_RE.exec(normText);
  vec[28] = casualMatch !== null ? 1 : 0;
  if (casualMatch) evidence.s_casual = casualMatch[0].slice(0, 24);

  vec[29] = PROFANITY_RE.test(normText) ? 1 : 0;

  const startsLowerEn =
    replyLang === "en" && /^[a-z]/.test(rawText) && !/[.!?]$/.test(rawText);
  const hasTypoEn = replyLang === "en" && INFORMAL_EN_TYPO_RE.test(normText);
  vec[30] = startsLowerEn || hasTypoEn ? 1 : 0;

  vec[31] = Math.log1p(contentTokenCount) / Math.log(128);

  // Cross-language features (32..33)
  const isChinese = (l: LangV2 | null) => l === "zh-Hans" || l === "zh-Hant";
  const langMismatch =
    opLang !== null &&
    replyLang !== "cjk-ambiguous" &&
    replyLang !== "other" &&
    opLang !== "cjk-ambiguous" &&
    opLang !== "other" &&
    !(isChinese(replyLang) && isChinese(opLang)) &&
    replyLang !== opLang
      ? 1
      : 0;
  vec[32] = langMismatch;
  vec[33] = langMismatch * Math.max(0, genericCos);

  // Summon features (34..35)
  vec[34] = isPureSummon ? 1 : 0;
  vec[35] = hasSummonMention ? 1 : 0;

  // Account / GraphQL metadata features (36..50)
  vec[36] = meta !== null ? 1 : 0;

  const mBlue =
    meta !== null
      ? meta.verifiedType === "blue"
        ? 1
        : meta.verifiedType === "unknown_verified"
          ? 0.5
          : 0
      : candidate.domSignals.hasVerifiedBadge
        ? 0.5
        : 0;
  vec[37] = mBlue;
  vec[38] =
    meta !== null && (meta.verifiedType === "business" || meta.verifiedType === "government")
      ? 1
      : 0;
  const mUnfollowed = meta !== null && meta.following === false ? 1 : 0;
  vec[39] = mUnfollowed;
  vec[40] = mBlue * mUnfollowed * (contentTokenCount <= 12 ? 1 : 0);
  vec[41] = meta !== null && meta.followedBy === true ? 1 : 0;

  const refTs = meta?.tweetCreatedAtMs ?? nowMs;
  const ageDays =
    meta?.authorCreatedAtMs !== null && meta?.authorCreatedAtMs !== undefined
      ? Math.max(1, (refTs - meta.authorCreatedAtMs) / (24 * 3600 * 1000))
      : null;
  vec[42] = ageDays !== null ? Math.log1p(ageDays) / Math.log1p(5000) : 0.5;
  vec[43] =
    meta?.followersCount !== null && meta?.followersCount !== undefined
      ? Math.min(1, Math.log10(1 + Math.max(0, meta.followersCount)) / 7)
      : 0.3;
  vec[44] =
    meta?.followersCount !== null &&
    meta?.followersCount !== undefined &&
    meta?.friendsCount !== null &&
    meta?.friendsCount !== undefined
      ? clamp(Math.log10((meta.friendsCount + 1) / (meta.followersCount + 1)), -3, 3) / 3
      : 0;
  vec[45] =
    ageDays !== null && meta?.statusesCount !== null && meta?.statusesCount !== undefined
      ? Math.min(1, Math.log1p(meta.statusesCount / ageDays) / Math.log1p(500))
      : 0.3;
  vec[46] = meta?.defaultProfileImage === true ? 1 : 0;

  const bioText = normalizeTextForRegex(meta?.description ?? "");
  const bioSpam = BIO_SPAM_STRONG_RE.test(bioText)
    ? 1
    : BIO_SPAM_WEAK_RE.test(bioText)
      ? 0.5
      : 0;
  vec[47] = bioSpam;
  if (bioSpam > 0) evidence.m_bio_spam = bioText.slice(0, 32);

  const dispName = normalizeTextForRegex(meta?.displayName ?? "");
  const dispEmojis = splitGraphemes(dispName).filter((g) => EMOJI_GRAPHEME_RE.test(g)).length;
  vec[48] = dispEmojis >= 3 || NAME_SPAM_RE.test(dispName) ? 1 : 0;
  if (vec[48] > 0) evidence.m_name_spam = dispName.slice(0, 32);

  const authorFeatures = extractAuthorModelFeatures(candidate.authorId, authorStats, nowMs);
  vec[49] = authorFeatures.handleBotScore;

  const opLikes =
    candidate.opFavoriteCount ??
    (anchor.kind !== "unanchored" ? (anchor.favoriteCount ?? null) : null);
  vec[50] =
    meta?.favoriteCount !== null &&
    meta?.favoriteCount !== undefined &&
    opLikes !== null &&
    opLikes !== undefined
      ? Math.min(1, Math.log1p(meta.favoriteCount) / Math.log1p(1 + Math.max(0, opLikes)))
      : 0;

  // Author memory features (51..53)
  vec[51] = authorFeatures.spamRate;
  vec[52] = authorFeatures.humanRate;
  vec[53] = authorFeatures.spamVolume;

  // Context features (54..57)
  const hasPrimaryMedia =
    candidate.hasPrimaryMedia ??
    (meta !== null ? meta.mediaTypes.length > 0 : false);
  vec[54] = rawText.length === 0 && hasPrimaryMedia ? 1 : 0;
  vec[55] = hasPrimaryMedia ? 1 : 0;
  vec[56] = candidate.domSignals.hasMediaOrExternalLink && !hasPrimaryMedia ? 1 : 0;
  vec[57] = candidate.hasQuote === true ? 1 : 0;

  const matchedSpamKeyword = matchKeywordSpamV2(rawText) ?? candidate.domSignals.matchedSpamKeyword;

  return {
    vector: vec as CompactFeatureVec,
    replyLang,
    opLang,
    contentTokenCount,
    opTokenCount,
    isPureSummon,
    summonHandle,
    isBotAuthor,
    matchedSpamKeyword,
    specificityHits,
    evidence,
  };
}
