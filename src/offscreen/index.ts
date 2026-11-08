import { ThreadFilterEngine } from "../domain/engine.ts";
import { type OffscreenRpcResponse, parseRpcRequest } from "../shared/rpc.ts";
import {
  IndexedDbWeightStore,
  OffscreenFilterService,
  WebGpuEmbeddingHost,
} from "./gpu-host.ts";

const IDLE_CHECK_INTERVAL_MS = 60_000 as const;

export function mountOffscreenRuntime(
  service: OffscreenFilterService,
  gpuHost: WebGpuEmbeddingHost,
): () => void {
  const timer = setInterval(() => {
    void gpuHost.unloadIfIdle(Date.now());
  }, IDLE_CHECK_INTERVAL_MS);

  const listener = (
    rawMessage: unknown,
    _sender: chrome.runtime.MessageSender,
    sendResponse: (response: OffscreenRpcResponse) => void,
  ): boolean | undefined => {
    const parsed = parseRpcRequest(rawMessage);
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
  const service = new OffscreenFilterService(engine, gpuHost, weightStore);
  mountOffscreenRuntime(service, gpuHost);
}
