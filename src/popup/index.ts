import type { DiagnosticsSummary } from "../domain/engine.ts";
import {
  DEFAULT_FILTER_SETTINGS,
  type FilterSettings,
  MAX_REPLAY_BUFFER_ENTRIES,
  type PolicyToggles,
  type SensitivityPreset,
  type SignalContribution,
  type Stage2StatusTag,
  type VerdictRecord,
} from "../domain/types.ts";
import type { OffscreenRpcRequestV2, OffscreenRpcResponse } from "../shared/rpc.ts";

export interface PopupRenderOptions {
  readonly stage2Status?: Stage2StatusTag | undefined;
  readonly stage2Progress?: number | undefined;
  readonly stage2Hint?: string | undefined;
  readonly resetConfirmArmed?: boolean | undefined;
  readonly actionFeedback?: string | undefined;
}

interface ChromeLanguageModelMonitor {
  addEventListener(
    type: "downloadprogress",
    listener: (ev: { readonly loaded?: number; readonly total?: number }) => void,
  ): void;
}

interface ChromeLanguageModelSession {
  readonly destroy?: () => void;
}

interface ChromeLanguageModelApi {
  readonly availability?: (options?: Record<string, unknown>) => Promise<string>;
  readonly create?: (options?: {
    readonly expectedInputs?: ReadonlyArray<{
      readonly type: string;
      readonly languages?: ReadonlyArray<string>;
    }>;
    readonly expectedOutputs?: ReadonlyArray<{
      readonly type: string;
      readonly languages?: ReadonlyArray<string>;
    }>;
    readonly monitor?: (monitor: ChromeLanguageModelMonitor) => void;
  }) => Promise<ChromeLanguageModelSession>;
}

const STAGE2_MODEL_OPTIONS = {
  expectedInputs: [{ type: "text", languages: ["en"] }],
  expectedOutputs: [{ type: "text", languages: ["en"] }],
} as const;

const POLICY_TOGGLE_ITEMS: ReadonlyArray<{
  readonly key: keyof PolicyToggles;
  readonly label: string;
}> = [
  { key: "collapseBlueCheckReplyGuys", label: "Collapse blue-check reply guys" },
  { key: "collapseMediaOnlyReplies", label: "Collapse blue-check media-only replies" },
  { key: "collapseBotSummons", label: "Collapse @grok / bot summons" },
  { key: "showSuspectBadges", label: "Show amber suspect badges" },
  { key: "learnFromFeedback", label: "Learn online from feedback" },
  { key: "enableStage2", label: "Enable Stage-2 Gemini Nano judge" },
];

const VALID_TOGGLE_KEYS: ReadonlySet<string> = new Set(
  POLICY_TOGGLE_ITEMS.map((item) => item.key),
);

const LANG_DISPLAY_KEYS = ["en", "zh-Hans", "zh-Hant", "ja", "other"] as const;

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function getGpuPillClass(gpuStatus: string): string {
  if (gpuStatus === "ready") return "pill-ok";
  if (gpuStatus === "loading" || gpuStatus === "recovering") return "pill-warn";
  return "pill-muted";
}

function formatStage2Pill(
  stage2Status: Stage2StatusTag | undefined,
  enableStage2: boolean,
): { readonly label: string; readonly cls: string } {
  if (stage2Status === "downloading") {
    return { label: "downloading", cls: "pill-warn" };
  }
  if (stage2Status === "downloadable") {
    return { label: "downloadable", cls: "pill-warn" };
  }
  if (stage2Status === "unavailable") {
    return enableStage2
      ? { label: "unavailable", cls: "pill-bad" }
      : { label: "off", cls: "pill-muted" };
  }
  if (!enableStage2 || stage2Status === "disabled" || stage2Status === undefined) {
    return { label: "off", cls: "pill-muted" };
  }
  if (stage2Status === "available") {
    return { label: "ready", cls: "pill-ok" };
  }
  return { label: "off", cls: "pill-muted" };
}

function formatPrecision(livePrecisionEstimate: number | null | undefined): string {
  if (livePrecisionEstimate === null || livePrecisionEstimate === undefined) {
    return "—";
  }
  return `${Math.round(livePrecisionEstimate * 100)}%`;
}

function formatSignalList(signals: ReadonlyArray<SignalContribution>, arrow: "↑" | "↓"): string {
  return signals
    .map((s) => {
      const sign = s.contribution >= 0 ? "+" : "";
      const ev = s.evidence ? `: ${s.evidence}` : "";
      return `${arrow}${s.id} (${sign}${s.contribution.toFixed(2)}${ev})`;
    })
    .join(", ");
}

function aggregateLanguageCounts(
  countsByLang:
    | Readonly<Record<string, { readonly dim: number; readonly suspect: number; readonly keep: number }>>
    | undefined,
): Record<(typeof LANG_DISPLAY_KEYS)[number], { dim: number; suspect: number; keep: number }> {
  const out: Record<
    (typeof LANG_DISPLAY_KEYS)[number],
    { dim: number; suspect: number; keep: number }
  > = {
    en: { dim: 0, suspect: 0, keep: 0 },
    "zh-Hans": { dim: 0, suspect: 0, keep: 0 },
    "zh-Hant": { dim: 0, suspect: 0, keep: 0 },
    ja: { dim: 0, suspect: 0, keep: 0 },
    other: { dim: 0, suspect: 0, keep: 0 },
  };
  if (!countsByLang) {
    return out;
  }
  for (const [key, counts] of Object.entries(countsByLang)) {
    const bucket =
      key === "en" || key === "zh-Hans" || key === "zh-Hant" || key === "ja" ? key : "other";
    out[bucket].dim += counts.dim;
    out[bucket].suspect += counts.suspect;
    out[bucket].keep += counts.keep;
  }
  return out;
}

function renderVerdictRow(v: VerdictRecord): string {
  const rowClass =
    v.action === "dim"
      ? "verdict-dim"
      : v.action === "suspect"
        ? "verdict-suspect"
        : "verdict-keep";
  const tagClass =
    v.action === "dim" ? "tag-dim" : v.action === "suspect" ? "tag-suspect" : "tag-keep";
  const authorLabel = v.authorId ? `@${escapeHtml(v.authorId)}` : `#${escapeHtml(v.tweetId)}`;
  const exp = v.explanation;
  const d = v.diagnostics;

  const metaBadge = exp
    ? `<span class="meta-pill ${exp.metaSource === "graphql" ? "meta-graphql" : "meta-dom"}">${exp.metaSource === "graphql" ? "GQL" : "DOM"}</span>`
    : "";

  const stage2Badge = exp?.stage2
    ? `<span class="stage2-pill stage2-${escapeHtml(exp.stage2.verdict)}">S2: ${escapeHtml(exp.stage2.verdict)} ${Math.round(exp.stage2.confidence * 100)}%</span>`
    : "";

  let signalLine = "";
  if (exp) {
    const parts: string[] = [];
    if (exp.top.length > 0) {
      parts.push(`<span class="sig-up">${escapeHtml(formatSignalList(exp.top, "↑"))}</span>`);
    }
    if (exp.counter.length > 0) {
      parts.push(
        `<span class="sig-down">${escapeHtml(formatSignalList(exp.counter, "↓"))}</span>`,
      );
    }
    if (exp.shields.length > 0) {
      parts.push(
        `<span class="sig-down">${escapeHtml(`shields=${exp.shields.join(",")}`)}</span>`,
      );
    }
    if (exp.caps.length > 0) {
      parts.push(`<span class="sig-cap">${escapeHtml(`caps=${exp.caps.join(",")}`)}</span>`);
    }
    if (parts.length > 0) {
      signalLine = `<div class="verdict-signals">${parts.join("")}</div>`;
    }
  }

  const metrics = exp
    ? `p=${Math.round(exp.probability * 100)}% | lang=${exp.lang.reply} | opSim=${d ? d.opCosSim.toFixed(2) : "—"} | cloneSim=${d ? d.maxCloneSim.toFixed(2) : "—"} | tok=${v.tokenCount}`
    : d
      ? `siamese=${Math.round(d.siameseProbability * 100)}% | opSim=${d.opCosSim.toFixed(2)} | cloneSim=${d.maxCloneSim.toFixed(2)} | prior=${d.authorSpamPrior >= 0 ? "+" : ""}${d.authorSpamPrior.toFixed(2)} | tok=${v.tokenCount}`
      : `score=${v.score.toFixed(2)} | tok=${v.tokenCount}`;

  return `
    <div class="verdict-row ${rowClass}">
      <div class="verdict-head">
        <span>${authorLabel}</span>
        <span class="verdict-tags">
          ${metaBadge}
          ${stage2Badge}
          <span class="${tagClass}">${escapeHtml(v.reason.kind)} (${escapeHtml(v.action)})</span>
        </span>
      </div>
      <div class="verdict-snippet">${escapeHtml(v.snippet && v.snippet.length > 0 ? v.snippet : "(media / no text)")}</div>
      ${signalLine}
      <div class="verdict-metrics">${escapeHtml(metrics)}</div>
    </div>`;
}

export function renderDiagnosticsHtml(
  gpuStatus: string,
  deviceEpoch: number,
  summary: DiagnosticsSummary,
  options: PopupRenderOptions = {},
): string {
  const settings: FilterSettings = summary.settings ?? DEFAULT_FILTER_SETTINGS;
  const stage2Pill = formatStage2Pill(options.stage2Status, settings.toggles.enableStage2);
  const suspectCount = summary.suspectReplies ?? 0;
  const precisionText = formatPrecision(summary.livePrecisionEstimate);
  const rollbackCount = summary.rollbackCount ?? 0;
  const replayBufferSize = summary.replayBufferSize ?? 0;

  const reasonEntries = Object.entries(summary.countsByReason).sort(
    (a, b) => b[1] - a[1] || a[0].localeCompare(b[0]),
  );
  const reasonChips = reasonEntries
    .map(
      ([reason, count]) =>
        `<span class="chip">${escapeHtml(reason)}: <strong>${count}</strong></span>`,
    )
    .join("");

  const signalEntries = Object.entries(summary.topSignalHistogram ?? {})
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .slice(0, 12);
  const signalChips = signalEntries
    .map(
      ([sigId, count]) =>
        `<span class="chip chip-signal">${escapeHtml(sigId)}: <strong>${count}</strong></span>`,
    )
    .join("");

  const langCounts = aggregateLanguageCounts(summary.countsByLang);
  const langRows = LANG_DISPLAY_KEYS.map((lang) => {
    const row = langCounts[lang];
    return `
      <tr>
        <td>${escapeHtml(lang)}</td>
        <td class="tag-dim">${row.dim}</td>
        <td class="tag-suspect">${row.suspect}</td>
        <td class="tag-keep">${row.keep}</td>
      </tr>`;
  }).join("");

  const presetButtons = (
    [
      { id: "conservative", label: "Conservative" },
      { id: "balanced", label: "Balanced" },
      { id: "strict", label: "Strict" },
    ] as ReadonlyArray<{ readonly id: SensitivityPreset; readonly label: string }>
  )
    .map((p) => {
      const isActive = settings.preset === p.id;
      return `<button type="button" class="seg-btn${isActive ? " active" : ""}" data-preset="${p.id}" aria-pressed="${isActive ? "true" : "false"}">${p.label}</button>`;
    })
    .join("");

  const toggleRows = POLICY_TOGGLE_ITEMS.map((item) => {
    const checked = settings.toggles[item.key];
    return `
      <label class="toggle-row">
        <span>${escapeHtml(item.label)}</span>
        <input type="checkbox" data-toggle="${item.key}" ${checked ? "checked" : ""} />
      </label>`;
  }).join("");

  const showPrepareStage2 =
    settings.toggles.enableStage2 ||
    options.stage2Status === "downloadable" ||
    options.stage2Status === "downloading";

  const progressText =
    options.stage2Progress !== undefined
      ? `Download: ${Math.max(0, Math.min(100, Math.round(options.stage2Progress)))}%`
      : options.stage2Hint;

  const prepareStage2Html = showPrepareStage2
    ? `
      <div class="stage2-prepare-box">
        <button type="button" id="xrf-prepare-stage2" class="btn btn-secondary">Prepare Gemini Nano</button>
        ${progressText ? `<span class="stage2-hint">${escapeHtml(progressText)}</span>` : ""}
      </div>`
    : "";

  const resetButtonsHtml = options.resetConfirmArmed
    ? `
      <button type="button" id="xrf-reset-feedback" class="btn btn-confirm">Confirm Reset</button>
      <button type="button" id="xrf-reset-cancel" class="btn">Cancel</button>`
    : `<button type="button" id="xrf-reset-feedback" class="btn btn-danger">Reset Learned</button>`;

  const authorRows =
    summary.topFlaggedAuthors.length === 0
      ? `<tr><td colspan="4" class="empty">No author history recorded yet</td></tr>`
      : summary.topFlaggedAuthors
          .map(
            (a) => `
          <tr>
            <td>@${escapeHtml(a.authorId)}</td>
            <td>${a.spamCount}</td>
            <td>${a.humanCount}</td>
            <td>${a.prior >= 0 ? "+" : ""}${a.prior.toFixed(2)}</td>
          </tr>`,
          )
          .join("");

  const recentItems =
    summary.recentVerdicts.length === 0
      ? `<div class="empty">Open an X thread to see scored replies.</div>`
      : summary.recentVerdicts.map(renderVerdictRow).join("");

  return `
    <h1>
      <span class="brand"><img src="./icons/icon-32.png" alt="" /><span>X Reply Filter</span></span>
      <span class="pills">
        <span class="status-pill ${getGpuPillClass(gpuStatus)}">GPU: ${escapeHtml(gpuStatus)} (e${deviceEpoch})</span>
        <span class="status-pill ${stage2Pill.cls}">Stage-2: ${escapeHtml(stage2Pill.label)}</span>
      </span>
    </h1>
    <div class="grid-5">
      <div class="card">
        <div class="card-label">Scored</div>
        <div class="card-value">${summary.totalScoredReplies}</div>
      </div>
      <div class="card">
        <div class="card-label">Dimmed</div>
        <div class="card-value val-dim">${summary.dimmedReplies}</div>
      </div>
      <div class="card">
        <div class="card-label">Suspect</div>
        <div class="card-value val-suspect">${suspectCount}</div>
      </div>
      <div class="card">
        <div class="card-label">Kept</div>
        <div class="card-value val-keep">${summary.keptReplies}</div>
      </div>
      <div class="card">
        <div class="card-label">Precision</div>
        <div class="card-value">${escapeHtml(precisionText)}</div>
      </div>
    </div>
    <div class="telemetry-sub">
      Threads: ${summary.cachedThreadCount} · Feedback updates: ${summary.updateCount} · Anchor guard rollbacks: ${rollbackCount}
    </div>

    <h2>Sensitivity Preset</h2>
    <div class="segmented" role="group" aria-label="Sensitivity preset">
      ${presetButtons}
    </div>

    <h2>Policy Toggles</h2>
    <div class="panel">
      <div class="toggle-list">${toggleRows}</div>
      ${prepareStage2Html}
    </div>

    <h2>Feedback &amp; Learner Controls</h2>
    <div class="panel">
      <div class="learner-stats">
        <div class="learner-stat">
          <div class="learner-stat-label">Feedback updates</div>
          <div class="learner-stat-val">${summary.updateCount}</div>
        </div>
        <div class="learner-stat">
          <div class="learner-stat-label">Replay buffer size</div>
          <div class="learner-stat-val">${replayBufferSize}/${MAX_REPLAY_BUFFER_ENTRIES}</div>
        </div>
        <div class="learner-stat">
          <div class="learner-stat-label">Anchor guard rollbacks</div>
          <div class="learner-stat-val">${rollbackCount}</div>
        </div>
      </div>
      <div class="btn-row">
        <button type="button" id="xrf-undo-feedback" class="btn">Undo Last</button>
        <button type="button" id="xrf-export-feedback" class="btn">Export JSON</button>
        <button type="button" id="xrf-import-feedback" class="btn">Import JSON</button>
        <input type="file" id="xrf-import-file" class="hidden-input" accept=".json,application/json" />
        ${resetButtonsHtml}
      </div>
      ${options.actionFeedback ? `<div class="action-note">${escapeHtml(options.actionFeedback)}</div>` : ""}
    </div>

    <h2>Filter Breakdown &amp; Languages</h2>
    <table>
      <thead>
        <tr>
          <th>Language</th>
          <th>Dim</th>
          <th>Suspect</th>
          <th>Keep</th>
        </tr>
      </thead>
      <tbody>${langRows}</tbody>
    </table>
    <div class="subhead">Reasons</div>
    <div class="chips">${reasonChips || '<span class="empty">None yet</span>'}</div>
    <div class="subhead">Top Signals</div>
    <div class="chips">${signalChips || '<span class="empty">None yet</span>'}</div>

    <h2>Account Reputation</h2>
    <table>
      <thead>
        <tr>
          <th>Account</th>
          <th>Bot marks</th>
          <th>Human marks</th>
          <th>Prior</th>
        </tr>
      </thead>
      <tbody>${authorRows}</tbody>
    </table>

    <h2>Recent Scored Replies</h2>
    <div>${recentItems}</div>
  `;
}

function normalizeLanguageModelAvailability(raw: string): Stage2StatusTag {
  if (raw === "available" || raw === "readily") return "available";
  if (raw === "downloadable" || raw === "after-download") return "downloadable";
  if (raw === "downloading") return "downloading";
  return "unavailable";
}

function getChromeLanguageModelApi(): ChromeLanguageModelApi | undefined {
  return (globalThis as { readonly LanguageModel?: ChromeLanguageModelApi }).LanguageModel;
}

async function sendPopupRpc(req: OffscreenRpcRequestV2): Promise<OffscreenRpcResponse> {
  try {
    const raw = (await chrome.runtime.sendMessage(req)) as OffscreenRpcResponse | undefined;
    if (!raw || typeof raw !== "object" || typeof raw.kind !== "string") {
      return { kind: "rpc_error", message: "No response from background worker." };
    }
    return raw;
  } catch (err) {
    return {
      kind: "rpc_error",
      message: err instanceof Error ? err.message : String(err),
    };
  }
}

async function probeLocalStage2Availability(): Promise<Stage2StatusTag | undefined> {
  const lm = getChromeLanguageModelApi();
  if (!lm || typeof lm.availability !== "function") {
    return undefined;
  }
  try {
    const raw = await lm.availability(STAGE2_MODEL_OPTIONS);
    return normalizeLanguageModelAvailability(raw);
  } catch {
    return "unavailable";
  }
}

function downloadJsonFile(filename: string, content: string): void {
  const blob = new Blob([content], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

interface PopupRuntimeState {
  gpuStatus: string;
  deviceEpoch: number;
  summary: DiagnosticsSummary | null;
  pendingSettings: FilterSettings | null;
  pendingSettingsInFlight: number;
  stage2Status: Stage2StatusTag | undefined;
  stage2Progress: number | undefined;
  stage2Hint: string | undefined;
  resetConfirmArmed: boolean;
  actionFeedback: string | undefined;
}

export async function bootPopup(): Promise<void> {
  const app = document.getElementById("app");
  if (!app) {
    return;
  }

  const state: PopupRuntimeState = {
    gpuStatus: "unloaded",
    deviceEpoch: 0,
    summary: null,
    pendingSettings: null,
    pendingSettingsInFlight: 0,
    stage2Status: undefined,
    stage2Progress: undefined,
    stage2Hint: undefined,
    resetConfirmArmed: false,
    actionFeedback: undefined,
  };
  let settingsQueue: Promise<void> = Promise.resolve();

  const render = (): void => {
    if (!state.summary) return;
    app.innerHTML = renderDiagnosticsHtml(state.gpuStatus, state.deviceEpoch, state.summary, {
      stage2Status: state.stage2Status,
      stage2Progress: state.stage2Progress,
      stage2Hint: state.stage2Hint,
      resetConfirmArmed: state.resetConfirmArmed,
      actionFeedback: state.actionFeedback,
    });
  };

  const refreshDiagnostics = async (): Promise<void> => {
    const res = await sendPopupRpc({ kind: "get_diagnostics", target: "background" });
    if (res.kind === "diagnostics_ok") {
      state.gpuStatus = res.gpuStatus;
      state.deviceEpoch = res.deviceEpoch;
      if (state.pendingSettingsInFlight > 0 && state.pendingSettings !== null) {
        state.summary = { ...res.summary, settings: state.pendingSettings };
      } else {
        state.pendingSettings = null;
        state.summary = res.summary;
      }
      if (res.stage2Status !== undefined) {
        state.stage2Status = res.stage2Status;
      }
      if (
        state.stage2Status === undefined ||
        state.stage2Status === "disabled" ||
        state.summary.settings?.toggles.enableStage2
      ) {
        const localAvail = await probeLocalStage2Availability();
        if (localAvail === "downloadable" || localAvail === "downloading") {
          state.stage2Status = localAvail;
        } else if (
          state.summary.settings?.toggles.enableStage2 &&
          localAvail !== undefined &&
          (state.stage2Status === undefined || state.stage2Status === "disabled")
        ) {
          state.stage2Status = localAvail;
        }
      }
      render();
    } else if (res.kind === "rpc_error") {
      if (state.summary === null) {
        app.textContent = `Diagnostics error: ${res.message}`;
      } else {
        state.actionFeedback = `Diagnostics error: ${res.message}`;
        render();
      }
    }
  };

  const getCurrentSettings = (): FilterSettings =>
    state.pendingSettings ?? state.summary?.settings ?? DEFAULT_FILTER_SETTINGS;

  const enqueueSettingsUpdate = (
    buildNext: (cur: FilterSettings) => FilterSettings,
    successNote: string,
    onSuccess?: () => Promise<void>,
  ): void => {
    const cur = getCurrentSettings();
    const nextSettings = buildNext(cur);
    state.resetConfirmArmed = false;
    state.pendingSettings = nextSettings;
    state.pendingSettingsInFlight += 1;
    if (state.summary) {
      state.summary = { ...state.summary, settings: nextSettings };
    }
    render();

    settingsQueue = settingsQueue.then(async () => {
      try {
        const res = await sendPopupRpc({
          kind: "update_settings",
          settings: nextSettings,
          target: "background",
        });
        if (res.kind === "rpc_error") {
          state.actionFeedback = `Settings update failed: ${res.message}`;
          if (state.pendingSettingsInFlight <= 1) {
            state.pendingSettings = null;
          }
        } else {
          state.actionFeedback = successNote;
          if (onSuccess) {
            await onSuccess();
          }
        }
      } finally {
        state.pendingSettingsInFlight = Math.max(0, state.pendingSettingsInFlight - 1);
        if (state.pendingSettingsInFlight === 0) {
          await refreshDiagnostics();
        } else {
          render();
        }
      }
    });
  };

  app.addEventListener("click", (ev) => {
    const target = ev.target;
    if (!(target instanceof Element)) return;

    const presetBtn = target.closest("[data-preset]");
    if (presetBtn) {
      const preset = presetBtn.getAttribute("data-preset");
      if (preset === "conservative" || preset === "balanced" || preset === "strict") {
        enqueueSettingsUpdate(
          (cur) => ({
            preset,
            toggles: cur.toggles,
            revision: cur.revision + 1,
          }),
          `Switched preset to ${preset}.`,
        );
      }
      return;
    }

    if (target.closest("#xrf-prepare-stage2")) {
      state.resetConfirmArmed = false;
      void (async () => {
        const lm = getChromeLanguageModelApi();
        if (!lm || typeof lm.availability !== "function" || typeof lm.create !== "function") {
          state.stage2Status = "unavailable";
          state.stage2Hint =
            "Chrome Built-in AI (LanguageModel API) is unavailable in this browser.";
          render();
          const rpcRes = await sendPopupRpc({
            kind: "trigger_stage2_download",
            action: "download",
            target: "background",
          });
          if (rpcRes.kind === "stage2_status_ok" && rpcRes.status !== "disabled") {
            state.stage2Status = rpcRes.status;
          }
          render();
          return;
        }

        try {
          const rawAvail = await lm.availability(STAGE2_MODEL_OPTIONS);
          const normAvail = normalizeLanguageModelAvailability(rawAvail);
          if (normAvail === "unavailable") {
            state.stage2Status = "unavailable";
            state.stage2Hint =
              "Gemini Nano is unavailable on this device or Prompt API flags are disabled.";
            render();
            return;
          }

          state.stage2Status = "downloading";
          state.stage2Progress = 0;
          state.stage2Hint = "Preparing Gemini Nano...";
          render();

          let session: ChromeLanguageModelSession | undefined;
          try {
            session = await lm.create({
              ...STAGE2_MODEL_OPTIONS,
              monitor(m) {
                m.addEventListener("downloadprogress", (pEv) => {
                  const loaded = pEv.loaded ?? 0;
                  const total = pEv.total;
                  const frac = total !== undefined && total > 1 ? loaded / total : loaded;
                  state.stage2Progress = Math.max(0, Math.min(100, frac * 100));
                  state.stage2Status = "downloading";
                  render();
                });
              },
            });
            state.stage2Progress = undefined;
            state.stage2Status = "available";
            state.stage2Hint = "Gemini Nano ready.";
          } finally {
            session?.destroy?.();
          }

          const rpcRes = await sendPopupRpc({
            kind: "trigger_stage2_download",
            action: "download",
            target: "background",
          });
          if (rpcRes.kind === "stage2_status_ok" && rpcRes.status !== "disabled") {
            state.stage2Status = rpcRes.status;
          }
          render();
        } catch (err) {
          state.stage2Progress = undefined;
          state.stage2Hint = `Stage-2 setup failed: ${err instanceof Error ? err.message : String(err)}`;
          render();
        }
      })();
      return;
    }

    if (target.closest("#xrf-undo-feedback")) {
      state.resetConfirmArmed = false;
      void (async () => {
        const res = await sendPopupRpc({ kind: "undo_feedback", target: "background" });
        if (res.kind === "undo_feedback_ok") {
          state.actionFeedback = res.undone
            ? "Undid last feedback update."
            : "No feedback updates left to undo.";
          await refreshDiagnostics();
        } else if (res.kind === "rpc_error") {
          state.actionFeedback = `Undo failed: ${res.message}`;
          render();
        }
      })();
      return;
    }

    if (target.closest("#xrf-export-feedback")) {
      state.resetConfirmArmed = false;
      void (async () => {
        const res = await sendPopupRpc({ kind: "export_feedback", target: "background" });
        if (res.kind === "export_feedback_ok") {
          downloadJsonFile("xrf-feedback-backup.json", res.payloadJson);
          state.actionFeedback = "Downloaded xrf-feedback-backup.json.";
          render();
        } else if (res.kind === "rpc_error") {
          state.actionFeedback = `Export failed: ${res.message}`;
          render();
        }
      })();
      return;
    }

    if (target.closest("#xrf-import-feedback")) {
      state.resetConfirmArmed = false;
      const input = document.getElementById("xrf-import-file") as HTMLInputElement | null;
      input?.click();
      return;
    }

    if (target.closest("#xrf-reset-cancel")) {
      state.resetConfirmArmed = false;
      render();
      return;
    }

    if (target.closest("#xrf-reset-feedback")) {
      if (!state.resetConfirmArmed) {
        state.resetConfirmArmed = true;
        render();
        return;
      }
      state.resetConfirmArmed = false;
      void (async () => {
        const res = await sendPopupRpc({ kind: "reset_feedback", target: "background" });
        if (res.kind === "reset_feedback_ok") {
          state.actionFeedback = "Reset learned weights, replay buffer, and author history.";
          await refreshDiagnostics();
        } else if (res.kind === "rpc_error") {
          state.actionFeedback = `Reset failed: ${res.message}`;
          render();
        }
      })();
    }
  });

  app.addEventListener("change", (ev) => {
    const target = ev.target;
    if (!(target instanceof HTMLInputElement)) return;

    const rawToggleKey = target.getAttribute("data-toggle");
    if (rawToggleKey !== null && VALID_TOGGLE_KEYS.has(rawToggleKey)) {
      const toggleKey = rawToggleKey as keyof PolicyToggles;
      const checked = target.checked;
      enqueueSettingsUpdate(
        (cur) => ({
          preset: cur.preset,
          toggles: {
            ...cur.toggles,
            [toggleKey]: checked,
          },
          revision: cur.revision + 1,
        }),
        `Updated ${toggleKey}.`,
        toggleKey === "enableStage2"
          ? async () => {
              if (checked) {
                const statusRes = await sendPopupRpc({
                  kind: "trigger_stage2_download",
                  action: "status",
                  target: "background",
                });
                if (statusRes.kind === "stage2_status_ok") {
                  state.stage2Status = statusRes.status;
                }
              } else {
                state.stage2Status = "disabled";
                state.stage2Hint = undefined;
              }
            }
          : undefined,
      );
      return;
    }

    if (target.id === "xrf-import-file") {
      const file = target.files?.[0];
      target.value = "";
      if (!file) return;
      void (async () => {
        try {
          const payloadJson = await file.text();
          if (payloadJson.length === 0 || payloadJson.length > 2_000_000) {
            state.actionFeedback = "Import rejected: file is empty or exceeds 2 MB limit.";
            render();
            return;
          }
          const res = await sendPopupRpc({
            kind: "import_feedback",
            payloadJson,
            target: "background",
          });
          if (res.kind === "import_feedback_ok") {
            const rollbackNote = res.rolledBack ? " (anchor guard rolled back weight update)" : "";
            state.actionFeedback = `Imported ${res.importedReplayCount} replay items and ${res.importedAuthorCount} authors${rollbackNote}.`;
            await refreshDiagnostics();
          } else if (res.kind === "rpc_error") {
            state.actionFeedback = `Import failed: ${res.message}`;
            render();
          }
        } catch (err) {
          state.actionFeedback = `Import read error: ${err instanceof Error ? err.message : String(err)}`;
          render();
        }
      })();
    }
  });

  try {
    await refreshDiagnostics();
  } catch (err) {
    app.textContent = `Failed to load diagnostics: ${err instanceof Error ? err.message : String(err)}`;
  }
}

if (
  typeof chrome !== "undefined" &&
  typeof chrome.runtime?.sendMessage === "function" &&
  typeof document !== "undefined"
) {
  void bootPopup();
}
