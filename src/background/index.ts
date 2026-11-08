import {
  type OffscreenChromeApi,
  type OffscreenRpcRequest,
  type OffscreenRpcResponse,
  ensureOffscreenDocument,
  parseRpcRequest,
} from "../shared/rpc.ts";

const OFFSCREEN_DOCUMENT_PATH = "offscreen.html" as const;

export async function forwardRpcToOffscreen(
  rawMessage: unknown,
  offscreenApi: OffscreenChromeApi,
  sendMessageFn: (req: OffscreenRpcRequest) => Promise<OffscreenRpcResponse>,
): Promise<OffscreenRpcResponse | null> {
  const parsed = parseRpcRequest(rawMessage);
  if (parsed === null || parsed.target === "offscreen") {
    return null;
  }

  try {
    await ensureOffscreenDocument(OFFSCREEN_DOCUMENT_PATH, offscreenApi);
    const forwarded: OffscreenRpcRequest = {
      ...parsed,
      target: "offscreen",
    };
    return await sendMessageFn(forwarded);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return {
      kind: "rpc_error",
      message,
    };
  }
}

export function registerBackgroundRouter(): void {
  chrome.runtime.onMessage.addListener((rawMessage, _sender, sendResponse) => {
    const parsed = parseRpcRequest(rawMessage);
    if (parsed === null || parsed.target === "offscreen") {
      return undefined;
    }

    void forwardRpcToOffscreen(
      parsed,
      chrome.offscreen as unknown as OffscreenChromeApi,
      (req) => chrome.runtime.sendMessage(req) as Promise<OffscreenRpcResponse>,
    ).then((res) => {
      if (res !== null) {
        sendResponse(res);
      }
    });

    return true;
  });
}

if (typeof chrome !== "undefined" && chrome.runtime?.onMessage && chrome.offscreen) {
  registerBackgroundRouter();
}
