import { timeSeriesChart, type Marker, type Series } from "../chart";
import { loadFirm } from "../data";
import { escapeHtml, money, num, pct, revenueMatches } from "../format";
import type { FilingEvent, Firm, Report } from "../types";

const EVENT_CLASS: Record<string, string> = {
  "bankruptcy": "ev-bankruptcy",
  "restatement": "ev-restatement",
  "auditor change": "ev-auditor",
  "acquisition/disposition": "ev-deal",
};

const eventClass = (e: FilingEvent) => EVENT_CLASS[e.kinds[0] ?? ""] ?? "ev-deal";

export function similaritySeries(reports: Report[]): Series[] {
  const originals = reports.filter((r) => r.version_no === 1);
  const make = (key: "sim_rf" | "sim_mdna", name: string, cls: string): Series => ({
    name,
    cls,
    points: originals
      .filter((r) => r[key] != null)
      .map((r) => ({
        t: Date.parse(r.period_end),
        y: r[key] as number,
        label: `FY${r.fy} ${name}: ${num(r[key])} similar to the prior year`,
      })),
  });
  return [make("sim_rf", "Risk Factors", "s-rf"), make("sim_mdna", "MD&A", "s-mdna")];
}

export function eventMarkers(events: FilingEvent[]): Marker[] {
  return events.map((e) => ({
    t: Date.parse(e.date),
    cls: eventClass(e),
    label: `${e.date}: ${e.kinds.join(", ")} (8-K items ${e.items})`,
  }));
}

export function edgarUrl(cik: number, accession: string): string {
  return `https://www.sec.gov/Archives/edgar/data/${cik}/${accession.replace(/-/g, "")}/`;
}

function extraction(v: number | null | undefined, truth: number | null): string {
  if (v == null) return `<span class="muted">unknown</span>`;
  return `<span class="${revenueMatches(v, truth) ? "pos" : "neg"}">${money(v)}</span>`;
}

function reportRow(cik: number, r: Report): string {
  const amended = r.is_amendment ? ` <span class="tag">${escapeHtml(r.form)} v${r.version_no}</span>` : "";
  const late = r.dated_next_business_day
    ? ` <span class="tag" title="Accepted after 5:30pm ET, so EDGAR dates it the next business day">after hours</span>`
    : "";
  const risk = r.risk_pct == null
    ? ""
    : `<span class="${r.risk_pct > 0.9 ? "neg" : ""}">${pct(r.risk_pct, 0)}</span>`
      + (r.event_next_12m ? ` <span class="tag warn">event followed</span>` : "");
  const tested = r.qlora_usd !== undefined || r.zero_shot_usd !== undefined;
  const llm = tested
    ? `LLM ${extraction(r.qlora_usd, r.revenue)} · zero-shot ${extraction(r.zero_shot_usd, r.revenue)}`
    : "";
  return `<tr${r.is_amendment ? ' class="amend"' : ""}>
    <td>FY${r.fy}${amended}</td>
    <td class="small">${escapeHtml(r.accepted_et ?? "")}${late}</td>
    <td>${escapeHtml(r.ticker ?? "—")}</td>
    <td class="num">${money(r.public_float)}</td>
    <td class="num">${num(r.sim_rf)}</td>
    <td class="num">${num(r.sim_mdna)}</td>
    <td class="num">${risk}</td>
    <td class="num">${money(r.revenue)}</td>
    <td class="small">${llm}</td>
    <td><a class="small" href="${edgarUrl(cik, r.accession)}" target="_blank" rel="noopener">EDGAR</a></td></tr>`;
}

const LEGEND = `<div class="legend">
  <span><i class="sw s-rf"></i>Risk Factors</span><span><i class="sw s-mdna"></i>MD&amp;A</span>
  <span><i class="sw ev-bankruptcy"></i>bankruptcy</span><span><i class="sw ev-restatement"></i>restatement</span>
  <span><i class="sw ev-auditor"></i>auditor change</span><span><i class="sw ev-deal"></i>acquisition/disposal</span></div>`;

export function firmHtml(f: Firm): string {
  const status = f.current_tickers
    ? `trades today as ${escapeHtml(f.current_tickers)}`
    : "no current ticker: delisted, acquired or never listed";
  const events = f.events.length
    ? `<ul class="events">${f.events.map((e) => `<li><i class="sw ${eventClass(e)}"></i>
        ${escapeHtml(e.date)} · ${escapeHtml(e.kinds.join(", "))} <span class="muted">items ${escapeHtml(e.items)}</span></li>`).join("")}</ul>`
    : `<p class="muted">No bankruptcy, restatement, auditor-change or deal 8-Ks.</p>`;
  return `
    <p><a href="#/firms">← All firms</a></p>
    <section class="card">
      <h2>${escapeHtml(f.name)} ${f.ticker ? `<span class="muted">${escapeHtml(f.ticker)}</span>` : ""}</h2>
      <p class="muted">${escapeHtml(f.industry ?? "")} · CIK ${f.cik} · ${status}</p>
      <h3>How much did the text change each year?</h3>
      <p class="small muted">Point-in-time TF-IDF cosine similarity to the prior year's section (1 = unchanged).
      Marks under the axis are 8-K events; hover for details.</p>
      ${timeSeriesChart(similaritySeries(f.reports), eventMarkers(f.events))}
      ${LEGEND}
    </section>
    <section class="card">
      <h3>Annual reports</h3>
      <div class="scroll"><table class="reports"><thead><tr><th>Fiscal year</th><th>Accepted (ET)</th><th>Ticker in filing</th>
        <th class="num">Public float</th><th class="num">RF sim.</th><th class="num">MD&amp;A sim.</th>
        <th class="num" title="Phase 3a TF-IDF risk score: percentile among 2020-24 test reports">Risk pct.</th>
        <th class="num">Revenue (XBRL)</th><th>Phase 3b extraction</th><th></th></tr></thead>
        <tbody>${f.reports.map((r) => reportRow(f.cik, r)).join("")}</tbody></table></div>
    </section>
    <section class="card">
      <h3>8-K events</h3>
      ${events}
    </section>`;
}

export async function renderFirm(root: HTMLElement, cik: number): Promise<void> {
  root.innerHTML = firmHtml(await loadFirm(cik));
}
