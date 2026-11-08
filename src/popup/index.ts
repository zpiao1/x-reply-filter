import type { DiagnosticsSummary } from "../domain/engine.ts";
import type { OffscreenRpcResponse } from "../shared/rpc.ts";

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

export function renderDiagnosticsHtml(
  gpuStatus: string,
  deviceEpoch: number,
  summary: DiagnosticsSummary,
): string {
  const reasonChips = Object.entries(summary.countsByReason)
    .map(
      ([reason, count]) =>
        `<span class="chip">${escapeHtml(reason)}: <strong>${count}</strong></span>`,
    )
    .join("");

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
      : summary.recentVerdicts
          .map((v) => {
            const d = v.diagnostics;
            const metrics = d
              ? `siamese=${Math.round(d.siameseProbability * 100)}% | opSim=${d.opCosSim.toFixed(2)} | cloneSim=${d.maxCloneSim.toFixed(2)} | prior=${d.authorSpamPrior >= 0 ? "+" : ""}${d.authorSpamPrior.toFixed(2)} | tok=${v.tokenCount}`
              : `score=${v.score.toFixed(2)} | tok=${v.tokenCount}`;
            const tagClass = v.action === "dim" ? "tag-dim" : "tag-keep";
            const authorLabel = v.authorId ? `@${escapeHtml(v.authorId)}` : `#${escapeHtml(v.tweetId)}`;
            return `
          <div class="verdict-row">
            <div class="verdict-head">
              <span>${authorLabel}</span>
              <span class="${tagClass}">${escapeHtml(v.reason.kind)} (${v.action})</span>
            </div>
            <div class="verdict-snippet">${escapeHtml(v.snippet ?? "(media / no text)")}</div>
            <div class="verdict-metrics">${escapeHtml(metrics)}</div>
          </div>`;
          })
          .join("");

  return `
    <h1>
      <span>X Reply Filter</span>
      <span class="status-pill">GPU: ${escapeHtml(gpuStatus)} (e${deviceEpoch})</span>
    </h1>
    <div class="grid">
      <div class="card">
        <div class="card-label">Scored</div>
        <div class="card-value">${summary.totalScoredReplies}</div>
      </div>
      <div class="card">
        <div class="card-label">Dimmed</div>
        <div class="card-value">${summary.dimmedReplies}</div>
      </div>
      <div class="card">
        <div class="card-label">Kept</div>
        <div class="card-value">${summary.keptReplies}</div>
      </div>
      <div class="card">
        <div class="card-label">Feedback</div>
        <div class="card-value">${summary.updateCount}</div>
      </div>
    </div>
    <h2>Filter Breakdown</h2>
    <div class="chips">${reasonChips || '<span class="empty">None yet</span>'}</div>
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

async function bootPopup(): Promise<void> {
  const app = document.getElementById("app");
  if (!app) {
    return;
  }
  try {
    const res = (await chrome.runtime.sendMessage({
      kind: "get_diagnostics",
      target: "background",
    })) as OffscreenRpcResponse;

    if (res.kind === "diagnostics_ok") {
      app.innerHTML = renderDiagnosticsHtml(res.gpuStatus, res.deviceEpoch, res.summary);
    } else if (res.kind === "rpc_error") {
      app.textContent = `Diagnostics error: ${res.message}`;
    }
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
