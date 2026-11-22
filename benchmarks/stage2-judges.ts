import {
  AutoModelForCausalLM,
  AutoModelForSequenceClassification,
  AutoTokenizer,
  Tensor,
} from "@huggingface/transformers";
import { execSync } from "node:child_process";
import type { BenchmarkCase } from "./dataset-humans.ts";

export interface Stage2JudgeScore {
  readonly id: string;
  readonly probBot: number;
  readonly inferMs: number;
}

export interface Stage2JudgeReport {
  readonly judgeName: string;
  readonly modelId: string;
  readonly dtype: string;
  readonly device: string;
  readonly supported: boolean;
  readonly error?: string;
  readonly loadMs: number;
  readonly vramDeltaMiB: number;
  readonly p50Ms: number;
  readonly p95Ms: number;
  readonly meanMs: number;
  readonly scores: ReadonlyArray<Stage2JudgeScore>;
}

export function getVramMiB(): number {
  try {
    const out = execSync("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits", {
      encoding: "utf8",
    }).trim();
    return Number(out);
  } catch {
    return -1;
  }
}

export function percentile(vals: ReadonlyArray<number>, p: number): number {
  if (vals.length === 0) return 0;
  const sorted = [...vals].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1));
  return sorted[idx]!;
}

function makeEmptyPastKV(config: Record<string, unknown>, batchSize = 1): Record<string, Tensor> {
  const numLayers = Number(config.num_hidden_layers ?? 0);
  const numKvHeads = Number(config.num_key_value_heads ?? config.num_attention_heads ?? 0);
  const headDim = Number(
    config.head_dim ??
      Number(config.hidden_size ?? 0) / Math.max(1, Number(config.num_attention_heads ?? 1)),
  );
  const kv: Record<string, Tensor> = {};
  for (let i = 0; i < numLayers; i++) {
    kv[`past_key_values.${i}.key`] = new Tensor("float32", new Float32Array(0), [
      batchSize,
      numKvHeads,
      0,
      headDim,
    ]);
    kv[`past_key_values.${i}.value`] = new Tensor("float32", new Float32Array(0), [
      batchSize,
      numKvHeads,
      0,
      headDim,
    ]);
  }
  return kv;
}

function formatMetaBrief(c: BenchmarkCase): string {
  const meta = c.graphqlMeta;
  if (!meta) {
    return `verified=${c.domMeta.hasVerifiedBadge}, following=unknown, media=${c.domMeta.hasPrimaryMedia}`;
  }
  const bioShort = (meta.description ?? "").slice(0, 80);
  return `verified=${meta.verifiedType}, following=${meta.following}, followers=${meta.followersCount}, bio="${bioShort}"`;
}

function buildCausalPrompt(
  kind: "qwen3_reranker" | "qwen3_06b" | "gemma3_1b",
  c: BenchmarkCase,
): string {
  const op = c.opText.slice(0, 260);
  const reply = c.replyText.trim().length > 0 ? c.replyText.slice(0, 320) : "[empty text / media-only]";
  const meta = formatMetaBrief(c);

  if (kind === "qwen3_reranker") {
    const sys =
      'Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".';
    const instruct =
      "Determine whether the social media reply (Document) is AI-generated slop, generic sycophantic engagement farming, paraphrase spam, or a bot reply to the original post (Query). Answer yes for AI/bot slop, no for a genuine human reply with personal voice, sarcasm, concrete details, or normal banter.";
    return (
      `<|im_start|>system\n${sys}<|im_end|>\n` +
      `<|im_start|>user\n<Instruct>: ${instruct}\n\n<Query>: ${op}\n\n<Document>: ${reply}\nAuthor metadata: ${meta}<|im_end|>\n` +
      `<|im_start|>assistant\n<think>\n\n</think>\n`
    );
  }

  if (kind === "qwen3_06b") {
    const sys =
      'You are an X/Twitter anti-bot classifier. Decide if the reply is AI-generated slop / engagement-farming bot / generic sycophancy / empty paraphrase ("YES") or a genuine human reply ("NO"). Genuine humans can be political, sarcastic, brief, or warmly agreeing with concrete experience. Output ONLY YES or NO.';
    return (
      `<|im_start|>system\n${sys}<|im_end|>\n` +
      `<|im_start|>user\nOriginal Post: ${op}\nReply: ${reply}\nAuthor info: ${meta}\nIs this reply an AI bot or engagement-farming slop? Answer YES or NO:<|im_end|>\n` +
      `<|im_start|>assistant\n<think>\n\n</think>\n\n`
    );
  }

  // gemma3_1b
  return (
    `<bos><start_of_turn>user\n` +
    `Classify whether this X/Twitter reply is AI-generated slop / engagement-farming bot / generic sycophantic praise / paraphrase spam (answer YES) or a genuine human reply with real opinion, nuance, sarcasm, banter, or concrete detail (answer NO). Answer with a single word: YES or NO.\n\n` +
    `Original Post: ${op}\n` +
    `Reply: ${reply}\n` +
    `Author info: ${meta}\n` +
    `Answer (YES or NO):<end_of_turn>\n<start_of_turn>model\n`
  );
}

export async function runCausalStage2Judge(args: {
  readonly judgeName: string;
  readonly kind: "qwen3_reranker" | "qwen3_06b" | "gemma3_1b";
  readonly modelId: string;
  readonly dtype: "q4" | "q4f16" | "q8" | "int8";
  readonly device: "cpu" | "dml";
  readonly useExternalDataFormat?: boolean;
  readonly cases: ReadonlyArray<BenchmarkCase>;
}): Promise<Stage2JudgeReport> {
  const { judgeName, kind, modelId, dtype, device, cases } = args;
  const v0 = getVramMiB();
  const tLoad = performance.now();
  try {
    const tokenizer = await AutoTokenizer.from_pretrained(modelId);
    const model = await AutoModelForCausalLM.from_pretrained(modelId, {
      device,
      dtype,
      ...(args.useExternalDataFormat !== undefined
        ? { use_external_data_format: args.useExternalDataFormat }
        : {}),
    });
    const loadMs = performance.now() - tLoad;

    const yesCandidates = kind === "qwen3_reranker" ? ["yes", "Yes"] : ["YES", "Yes", "yes"];
    const noCandidates = kind === "qwen3_reranker" ? ["no", "No"] : ["NO", "No", "no"];
    const pickTokenIds = (cands: string[]) => {
      const ids = new Set<number>();
      for (const s of cands) {
        const enc = tokenizer.encode(s, { add_special_tokens: false });
        if (enc.length === 1 && enc[0] !== undefined) ids.add(enc[0]);
      }
      return Array.from(ids);
    };
    const yesIds = pickTokenIds(yesCandidates);
    const noIds = pickTokenIds(noCandidates);
    if (yesIds.length === 0 || noIds.length === 0) {
      await model.dispose();
      throw new Error(`Could not resolve yes/no single token IDs (${yesIds}, ${noIds})`);
    }

    const cfg = model.config as unknown as Record<string, unknown>;
    let useManualFeed = true;

    const inferPrompt = async (prompt: string): Promise<number> => {
      const inputs = await tokenizer(prompt, { truncation: true, max_length: 512 });
      const seqLen = Number(inputs.input_ids.dims[1]);
      let out: { logits: Tensor };
      if (useManualFeed) {
        const posData = BigInt64Array.from({ length: seqLen }, (_, i) => BigInt(i));
        const feed: Record<string, Tensor> = {
          input_ids: inputs.input_ids,
          attention_mask: inputs.attention_mask,
          position_ids: new Tensor("int64", posData, [1, seqLen]),
          ...makeEmptyPastKV(cfg, 1),
        };
        try {
          out = (await model(feed)) as { logits: Tensor };
        } catch {
          useManualFeed = false;
          out = (await model(inputs)) as { logits: Tensor };
        }
      } else {
        out = (await model(inputs)) as { logits: Tensor };
      }

      const dims = out.logits.dims;
      const outSeq = Number(dims[1]);
      const vocab = Number(dims[2]);
      const data = out.logits.data as Float32Array;
      const off = (outSeq - 1) * vocab;

      // Log-sum-exp over yes variants vs no variants
      const lse = (ids: number[]) => {
        let m = -Infinity;
        for (const id of ids) {
          const val = Number(data[off + id]);
          if (val > m) m = val;
        }
        let s = 0;
        for (const id of ids) s += Math.exp(Number(data[off + id]) - m);
        return m + Math.log(s);
      };
      const yL = lse(yesIds);
      const nL = lse(noIds);
      const maxL = Math.max(yL, nL);
      return Math.exp(yL - maxL) / (Math.exp(yL - maxL) + Math.exp(nL - maxL));
    };

    // Warmup on first case
    if (cases.length > 0) {
      await inferPrompt(buildCausalPrompt(kind, cases[0]!));
    }

    let peakVram = getVramMiB();
    const scores: Stage2JudgeScore[] = [];
    const latencies: number[] = [];

    for (const c of cases) {
      const prompt = buildCausalPrompt(kind, c);
      const s0 = performance.now();
      const probBot = await inferPrompt(prompt);
      const inferMs = performance.now() - s0;
      latencies.push(inferMs);
      scores.push({ id: c.id, probBot, inferMs });
      const curV = getVramMiB();
      if (curV > peakVram) peakVram = curV;
    }

    await model.dispose();

    const meanMs = latencies.reduce((a, b) => a + b, 0) / Math.max(1, latencies.length);
    return {
      judgeName,
      modelId,
      dtype,
      device,
      supported: true,
      loadMs: Math.round(loadMs),
      vramDeltaMiB: Math.max(0, peakVram - v0),
      p50Ms: Number(percentile(latencies, 50).toFixed(1)),
      p95Ms: Number(percentile(latencies, 95).toFixed(1)),
      meanMs: Number(meanMs.toFixed(1)),
      scores,
    };
  } catch (err) {
    return {
      judgeName,
      modelId,
      dtype,
      device,
      supported: false,
      error: err instanceof Error ? err.message.slice(0, 350) : String(err).slice(0, 350),
      loadMs: Math.round(performance.now() - tLoad),
      vramDeltaMiB: 0,
      p50Ms: 0,
      p95Ms: 0,
      meanMs: 0,
      scores: [],
    };
  }
}

export async function runNliStage2Judge(args: {
  readonly judgeName: string;
  readonly modelId: string;
  readonly dtype: "q8" | "fp16";
  readonly device: "cpu" | "dml";
  readonly cases: ReadonlyArray<BenchmarkCase>;
}): Promise<Stage2JudgeReport> {
  const { judgeName, modelId, dtype, device, cases } = args;
  const v0 = getVramMiB();
  const tLoad = performance.now();
  try {
    const tokenizer = await AutoTokenizer.from_pretrained(modelId);
    const model = await AutoModelForSequenceClassification.from_pretrained(modelId, {
      device,
      dtype,
    });
    const loadMs = performance.now() - tLoad;
    const hypothesis =
      "This social media reply is generic AI-generated sycophantic praise, empty platitude, or engagement-farming spam rather than a specific human response.";

    const inferCase = async (c: BenchmarkCase): Promise<number> => {
      const premise = `Post: ${c.opText.slice(0, 180)}\nReply: ${c.replyText.slice(0, 260)}`;
      const inputs = await tokenizer(premise, {
        text_pair: hypothesis,
        truncation: true,
        max_length: 256,
      });
      const out = (await model(inputs)) as { logits: Tensor };
      const logits = out.logits.data as Float32Array;
      // id2label: { 0: 'entailment', 1: 'neutral', 2: 'contradiction' }
      const ent = Number(logits[0]);
      const con = Number(logits[2]);
      const m = Math.max(ent, con);
      return Math.exp(ent - m) / (Math.exp(ent - m) + Math.exp(con - m));
    };

    if (cases.length > 0) {
      await inferCase(cases[0]!);
    }
    let peakVram = getVramMiB();
    const scores: Stage2JudgeScore[] = [];
    const latencies: number[] = [];

    for (const c of cases) {
      const s0 = performance.now();
      const probBot = await inferCase(c);
      const inferMs = performance.now() - s0;
      latencies.push(inferMs);
      scores.push({ id: c.id, probBot, inferMs });
      const curV = getVramMiB();
      if (curV > peakVram) peakVram = curV;
    }

    await model.dispose();
    const meanMs = latencies.reduce((a, b) => a + b, 0) / Math.max(1, latencies.length);
    return {
      judgeName,
      modelId,
      dtype,
      device,
      supported: true,
      loadMs: Math.round(loadMs),
      vramDeltaMiB: Math.max(0, peakVram - v0),
      p50Ms: Number(percentile(latencies, 50).toFixed(1)),
      p95Ms: Number(percentile(latencies, 95).toFixed(1)),
      meanMs: Number(meanMs.toFixed(1)),
      scores,
    };
  } catch (err) {
    return {
      judgeName,
      modelId,
      dtype,
      device,
      supported: false,
      error: err instanceof Error ? err.message.slice(0, 350) : String(err).slice(0, 350),
      loadMs: Math.round(performance.now() - tLoad),
      vramDeltaMiB: 0,
      p50Ms: 0,
      p95Ms: 0,
      meanMs: 0,
      scores: [],
    };
  }
}
