import {
  AutoModel,
  AutoModelForCausalLM,
  AutoModelForSequenceClassification,
  AutoTokenizer,
  Tensor,
} from "@huggingface/transformers";
import { execSync } from "node:child_process";

function vramMiB(): number {
  try {
    const out = execSync("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits", {
      encoding: "utf8",
    }).trim();
    return Number(out);
  } catch {
    return -1;
  }
}

function makeEmptyPastKV(config: Record<string, unknown>, batchSize = 1): Record<string, Tensor> {
  const numLayers = Number(config.num_hidden_layers ?? 0);
  const numKvHeads = Number(config.num_key_value_heads ?? config.num_attention_heads ?? 0);
  const headDim = Number(
    config.head_dim ??
      (Number(config.hidden_size ?? 0) / Math.max(1, Number(config.num_attention_heads ?? 1))),
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

async function probeEmbedding(modelId: string, dtype: "q8" | "q4" | "q4f16", device: "cpu" | "dml") {
  const v0 = vramMiB();
  const t0 = performance.now();
  try {
    const tok = await AutoTokenizer.from_pretrained(modelId);
    const model = await AutoModel.from_pretrained(modelId, { device, dtype });
    const loadMs = performance.now() - t0;
    const inputs = await tok(
      [
        "Spot on! This is such a game-changer for the ecosystem.",
        "We tested this patch on our staging cluster yesterday and hit a 12ms regression.",
      ],
      { padding: true, truncation: true, max_length: 256 },
    );
    await model(inputs); // warmup
    const t1 = performance.now();
    const out = (await model(inputs)) as Record<string, Tensor>;
    const batchMs = performance.now() - t1;
    const v1 = vramMiB();
    const keys = Object.keys(out);
    const firstKey = out.sentence_embedding ? "sentence_embedding" : keys[0]!;
    console.log(
      JSON.stringify({
        kind: "embedding",
        modelId,
        dtype,
        device,
        ok: true,
        loadMs: Math.round(loadMs),
        batch2Ms: Math.round(batchMs),
        vramDeltaMiB: v1 - v0,
        outKeys: keys,
        dims: out[firstKey]?.dims,
      }),
    );
    await model.dispose();
  } catch (err) {
    console.log(
      JSON.stringify({
        kind: "embedding",
        modelId,
        dtype,
        device,
        ok: false,
        error: err instanceof Error ? err.message.slice(0, 300) : String(err).slice(0, 300),
      }),
    );
  }
}

async function probeCausalJudge(
  modelId: string,
  dtype: "q4" | "q4f16" | "q8",
  device: "cpu" | "dml",
  prompt: string,
  yesTokenStr: string,
  noTokenStr: string,
) {
  const v0 = vramMiB();
  const t0 = performance.now();
  try {
    const tok = await AutoTokenizer.from_pretrained(modelId);
    const model = await AutoModelForCausalLM.from_pretrained(modelId, { device, dtype });
    const loadMs = performance.now() - t0;
    const inputs = await tok(prompt, { truncation: true, max_length: 512 });
    const seqLen = Number(inputs.input_ids.dims[1]);
    const posData = BigInt64Array.from({ length: seqLen }, (_, i) => BigInt(i));
    const position_ids = new Tensor("int64", posData, [1, seqLen]);
    const pastKV = makeEmptyPastKV(model.config as unknown as Record<string, unknown>, 1);

    // Check which inputs the session expects
    const feed: Record<string, Tensor> = {
      input_ids: inputs.input_ids,
      attention_mask: inputs.attention_mask,
      position_ids,
      ...pastKV,
    };

    const runOne = async () => {
      try {
        return (await model(feed)) as { logits: Tensor };
      } catch {
        // Fallback: let transformers.js construct its own inputs without manual pastKV
        return (await model(inputs)) as { logits: Tensor };
      }
    };

    await runOne(); // warmup
    const t1 = performance.now();
    const out = await runOne();
    const inferMs = performance.now() - t1;
    const v1 = vramMiB();

    const yesIds = tok.encode(yesTokenStr, { add_special_tokens: false });
    const noIds = tok.encode(noTokenStr, { add_special_tokens: false });
    const yesId = yesIds[yesIds.length - 1]!;
    const noId = noIds[noIds.length - 1]!;
    const dims = out.logits.dims;
    const outSeq = Number(dims[1]);
    const vocab = Number(dims[2]);
    const data = out.logits.data as Float32Array;
    const off = (outSeq - 1) * vocab;
    const yL = Number(data[off + yesId]);
    const nL = Number(data[off + noId]);
    const m = Math.max(yL, nL);
    const pYes = Math.exp(yL - m) / (Math.exp(yL - m) + Math.exp(nL - m));

    console.log(
      JSON.stringify({
        kind: "causal_judge",
        modelId,
        dtype,
        device,
        ok: true,
        loadMs: Math.round(loadMs),
        inferMs: Math.round(inferMs),
        vramDeltaMiB: v1 - v0,
        yesId,
        noId,
        pYes: Number(pYes.toFixed(4)),
      }),
    );
    await model.dispose();
  } catch (err) {
    console.log(
      JSON.stringify({
        kind: "causal_judge",
        modelId,
        dtype,
        device,
        ok: false,
        error: err instanceof Error ? err.message.slice(0, 350) : String(err).slice(0, 350),
      }),
    );
  }
}

async function probeNli(modelId: string, dtype: "q8" | "fp16", device: "cpu" | "dml") {
  const v0 = vramMiB();
  const t0 = performance.now();
  try {
    const tok = await AutoTokenizer.from_pretrained(modelId);
    const model = await AutoModelForSequenceClassification.from_pretrained(modelId, {
      device,
      dtype,
    });
    const loadMs = performance.now() - t0;
    const premise =
      "Reply: Spot on! This is such a game-changer for the ecosystem. What do you think?";
    const hypothesis = "This reply is generic AI-generated engagement-farming spam.";
    const inputs = await tok(premise, { text_pair: hypothesis, truncation: true, max_length: 256 });
    await model(inputs); // warmup
    const t1 = performance.now();
    const out = (await model(inputs)) as { logits: Tensor };
    const inferMs = performance.now() - t1;
    const v1 = vramMiB();
    console.log(
      JSON.stringify({
        kind: "nli",
        modelId,
        dtype,
        device,
        ok: true,
        loadMs: Math.round(loadMs),
        inferMs: Math.round(inferMs),
        vramDeltaMiB: v1 - v0,
        logits: Array.from(out.logits.data as Float32Array).map((x) => Number(x.toFixed(3))),
        id2label: (model.config as unknown as Record<string, unknown>).id2label,
      }),
    );
    await model.dispose();
  } catch (err) {
    console.log(
      JSON.stringify({
        kind: "nli",
        modelId,
        dtype,
        device,
        ok: false,
        error: err instanceof Error ? err.message.slice(0, 300) : String(err).slice(0, 300),
      }),
    );
  }
}

async function main() {
  console.log("Initial VRAM:", vramMiB(), "MiB");

  // 1. EmbeddingGemma q4
  await probeEmbedding("onnx-community/embeddinggemma-300m-ONNX", "q4", "dml");
  await probeEmbedding("onnx-community/embeddinggemma-300m-ONNX", "q4", "cpu");

  // 2. Qwen3-Embedding-0.6B
  await probeEmbedding("onnx-community/Qwen3-Embedding-0.6B-ONNX", "q8", "cpu");
  await probeEmbedding("onnx-community/Qwen3-Embedding-0.6B-ONNX", "q4f16", "dml");

  // 3. mDeBERTa-v3-base-xnli
  await probeNli("onnx-community/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7-ONNX", "q8", "cpu");
  await probeNli("onnx-community/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7-ONNX", "fp16", "dml");

  // 4. Qwen3-Reranker-0.6B-ONNX (tests transformers.js v3.8.1 vs v4 GroupQueryAttention compatibility!)
  const rerankerPrompt =
    `<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n` +
    `<|im_start|>user\n<Instruct>: Determine if the social media reply is AI-generated slop, sycophantic praise, or engagement-farming spam rather than a genuine human reply.\n\n<Query>: We migrated our billing ledger off MongoDB to Postgres.\n\n<Document>: Spot on! This is such a game-changer for the ecosystem. What do you think?<|im_end|>\n` +
    `<|im_start|>assistant\n<think>\n\n</think>\n`;
  await probeCausalJudge("onnx-community/Qwen3-Reranker-0.6B-ONNX", "q4", "cpu", rerankerPrompt, "yes", "no");

  // 5. Qwen3-0.6B-ONNX
  const qwenPrompt =
    `<|im_start|>system\nYou classify social media replies. Answer ONLY "YES" if the reply is AI-generated slop, generic sycophantic engagement farming, or a bot; answer ONLY "NO" if it is a genuine human reply.<|im_end|>\n` +
    `<|im_start|>user\nOP: We migrated our billing ledger off MongoDB to Postgres.\nReply: Spot on! This is such a game-changer for the ecosystem. What do you think?\nIs this reply an AI bot/slop? Answer YES or NO:<|im_end|>\n` +
    `<|im_start|>assistant\n<think>\n\n</think>\n\n`;
  await probeCausalJudge("onnx-community/Qwen3-0.6B-ONNX", "q4f16", "dml", qwenPrompt, "YES", "NO");
  await probeCausalJudge("onnx-community/Qwen3-0.6B-ONNX", "q8", "cpu", qwenPrompt, "YES", "NO");

  // 6. gemma-3-1b-it-ONNX
  const gemmaPrompt =
    `<bos><start_of_turn>user\nClassify if this X/Twitter reply is AI-generated bot slop / engagement farming (YES) or a genuine human reply (NO). Reply with a single word: YES or NO.\nOP: We migrated our billing ledger off MongoDB to Postgres.\nReply: Spot on! This is such a game-changer for the ecosystem. What do you think?<end_of_turn>\n<start_of_turn>model\n`;
  await probeCausalJudge("onnx-community/gemma-3-1b-it-ONNX", "q4f16", "dml", gemmaPrompt, "YES", "NO");
  await probeCausalJudge("onnx-community/gemma-3-1b-it-ONNX", "q4", "cpu", gemmaPrompt, "YES", "NO");

  console.log("Final VRAM:", vramMiB(), "MiB");
}

await main();
