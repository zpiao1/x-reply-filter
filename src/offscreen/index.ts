import { ThreadFilterEngine } from "../domain/engine.ts";
import {
  type OffscreenRpcResponse,
  type Stage2VerdictUpdateMessage,
  parseRpcRequestV2,
} from "../shared/rpc.ts";
import {
  IndexedDbWeightStore,
  OffscreenFilterService,
  WebGpuEmbeddingHost,
} from "./gpu-host.ts";
import { Stage2JudgeService } from "./stage2-judge.ts";

const IDLE_CHECK_INTERVAL_MS = 60_000 as const;

export function mountOffscreenRuntime(
  service: OffscreenFilterService,
  gpuHost: WebGpuEmbeddingHost,
  stage2Judge?: Stage2JudgeService,
): () => void {
  const timer = setInterval(() => {
    const now = Date.now();
    void gpuHost.unloadIfIdle(now);
    void stage2Judge?.unloadIfIdle(now);
  }, IDLE_CHECK_INTERVAL_MS);

  const listener = (
    rawMessage: unknown,
    _sender: chrome.runtime.MessageSender,
    sendResponse: (response: OffscreenRpcResponse) => void,
  ): boolean | undefined => {
    const parsed = parseRpcRequestV2(rawMessage);
    if (parsed === null || parsed.target !== "offscreen") {
      return undefined;
    }

    void service.handleRequest(parsed).then(sendResponse);
    return true;
  };

  chrome.runtime.onMessage.addListener(listener);

  return () => {
    clearInterval(timer);
    chrome.runtime.onMessage.removeListener(listener);
  };
}

if (typeof chrome !== "undefined" && chrome.runtime?.onMessage) {
  const engine = new ThreadFilterEngine();
  const gpuHost = new WebGpuEmbeddingHost();
  const weightStore = new IndexedDbWeightStore();
  const stage2Judge = new Stage2JudgeService();
  const service = new OffscreenFilterService(engine, gpuHost, weightStore, {
    stage2Judge,
    onStage2Update: (msg: Stage2VerdictUpdateMessage) => {
      void Promise.resolve(chrome.runtime.sendMessage(msg)).catch(() => {});
    },
  });
  mountOffscreenRuntime(service, gpuHost, stage2Judge);
}
