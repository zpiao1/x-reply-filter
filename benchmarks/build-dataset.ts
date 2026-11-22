import { writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { BOT_CASES } from "./dataset-bots.ts";
import { type BenchmarkCase, HUMAN_CASES } from "./dataset-humans.ts";

// Apply targeted edge-case overrides so DOM-only fallback (graphqlMeta: null),
// org-verified cap (C1), high-likes cap (C2), followedBy cap (C3), quote tweet,
// and unverified human reaction GIF are all empirically represented in dataset.json.
const DOM_ONLY_IDS = new Set<string>([
  // 10 diverse humans (never on human_followed_account)
  "h_pol_en_03",
  "h_fin_en_02",
  "h_gam_zhs_01",
  "h_ani_ja_01",
  "h_tech_en_02",
  "h_agr_en_02",
  "h_sar_zht_01",
  "h_sht_en_01",
  "h_sht_ja_01",
  "h_grk_zhs_01",
  // 10 diverse bots (including blue-check short praise & media-only to exercise following===null -> suspect)
  "b_praise_en_02",
  "b_praise_zhs_01",
  "b_anti_en_04",
  "b_anti_ja_01",
  "b_para_en_02",
  "b_rg_en_04",
  "b_shp_en_04",
  "b_shp_zhs_01",
  "b_med_en_02",
  "b_grk_en_02",
]);

function applyEdgeOverrides(item: BenchmarkCase): BenchmarkCase {
  let out: BenchmarkCase = item;

  // Unverified human reaction GIF (tests R6 exemption for unverified media-only)
  if (item.id === "h_sht_en_04") {
    out = {
      ...out,
      replyText: "",
      domMeta: { ...out.domMeta, hasPrimaryMedia: true },
      graphqlMeta: out.graphqlMeta
        ? { ...out.graphqlMeta, mediaTypes: ["animated_gif"] }
        : null,
    };
  }

  // Org-verified (business) human reply (tests C1 cap)
  if (item.id === "h_tech_en_01" && out.graphqlMeta) {
    out = {
      ...out,
      graphqlMeta: {
        ...out.graphqlMeta,
        isBlueVerified: true,
        verifiedType: "business",
      },
    };
  }

  // High-liked human replies (tests C2 cap)
  if ((item.id === "h_agr_en_01" || item.id === "h_pol_en_01") && out.graphqlMeta) {
    out = {
      ...out,
      graphqlMeta: {
        ...out.graphqlMeta,
        favoriteCount: 240,
        opFavoriteCount: 380,
      },
    };
  }

  // Human reply with quote tweet (tests c_has_quote)
  if (item.id === "h_fin_en_01") {
    out = {
      ...out,
      domMeta: { ...out.domMeta, hasQuote: true },
    };
  }

  // Human followedBy=true (tests C3 cap)
  if (item.id === "h_agr_ja_01" && out.graphqlMeta) {
    out = {
      ...out,
      graphqlMeta: {
        ...out.graphqlMeta,
        followedBy: true,
      },
    };
  }

  // DOM-only fallback cases (graphqlMeta: null)
  if (DOM_ONLY_IDS.has(item.id)) {
    out = {
      ...out,
      graphqlMeta: null,
    };
  }

  return out;
}

const allCases: BenchmarkCase[] = [...HUMAN_CASES, ...BOT_CASES].map(applyEdgeOverrides);

// Validation assertions
const seenIds = new Set<string>();
const counts = {
  total: allCases.length,
  human: 0,
  bot: 0,
  domOnlyHuman: 0,
  domOnlyBot: 0,
  blueHuman: 0,
  blueBot: 0,
  byLang: {} as Record<string, { human: number; bot: number }>,
  byCat: {} as Record<string, number>,
};

for (const c of allCases) {
  if (seenIds.has(c.id)) {
    throw new Error(`Duplicate id: ${c.id}`);
  }
  seenIds.add(c.id);
  counts[c.expectedLabel] += 1;
  if (c.graphqlMeta === null) {
    if (c.expectedLabel === "human") counts.domOnlyHuman += 1;
    else counts.domOnlyBot += 1;
  }
  if (c.domMeta.hasVerifiedBadge) {
    if (c.expectedLabel === "human") counts.blueHuman += 1;
    else counts.blueBot += 1;
  }
  const lc = counts.byLang[c.lang] ?? { human: 0, bot: 0 };
  lc[c.expectedLabel] += 1;
  counts.byLang[c.lang] = lc;
  counts.byCat[c.category] = (counts.byCat[c.category] ?? 0) + 1;
}

if (counts.total !== 200 || counts.human !== 100 || counts.bot !== 100) {
  throw new Error(`Unexpected totals: ${JSON.stringify(counts)}`);
}

const outPath = resolve("benchmarks/dataset.json");
writeFileSync(outPath, JSON.stringify(allCases, null, 2) + "\n", "utf8");
console.log("Wrote", outPath);
console.log(JSON.stringify(counts, null, 2));
