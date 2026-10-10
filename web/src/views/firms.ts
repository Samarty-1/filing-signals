import { loadIndex } from "../data";
import { escapeHtml, num } from "../format";
import type { FirmIndexRow } from "../types";

export type SortKey = "name" | "events" | "rewrite" | "reports";

export function filterFirms(rows: FirmIndexRow[], query: string, sort: SortKey): FirmIndexRow[] {
  const q = query.trim().toLowerCase();
  const hit = q
    ? rows.filter((r) => r.name.toLowerCase().includes(q) || (r.ticker ?? "").toLowerCase() === q
        || String(r.cik) === q || (r.industry ?? "").toLowerCase().includes(q))
    : rows.slice();
  const by: Record<SortKey, (a: FirmIndexRow, b: FirmIndexRow) => number> = {
    name: (a, b) => a.name.localeCompare(b.name),
    events: (a, b) => b.events - a.events,
    reports: (a, b) => b.reports - a.reports,
    // biggest single-year Risk Factors rewrite first; firms without pairs last
    rewrite: (a, b) => (a.min_sim_rf ?? 2) - (b.min_sim_rf ?? 2),
  };
  return hit.sort(by[sort]);
}

const LIMIT = 150;

export async function renderFirms(root: HTMLElement): Promise<void> {
  const rows = await loadIndex();
  root.innerHTML = `
    <section class="card">
      <h2>Firms</h2>
      <p class="muted">${rows.length.toLocaleString()} filers drawn at random from EDGAR, including the ones that
      have since gone bankrupt, been acquired or stopped filing.</p>
      <div class="controls">
        <input id="q" type="search" placeholder="Name, ticker, CIK or industry" aria-label="Search firms">
        <select id="sort" aria-label="Sort by">
          <option value="name">Name</option>
          <option value="rewrite">Biggest Risk Factors rewrite</option>
          <option value="events">Most 8-K events</option>
          <option value="reports">Most reports</option>
        </select>
      </div>
      <div id="list"></div>
    </section>`;
  const q = root.querySelector<HTMLInputElement>("#q")!;
  const sort = root.querySelector<HTMLSelectElement>("#sort")!;
  const list = root.querySelector<HTMLDivElement>("#list")!;
  const draw = () => {
    const hits = filterFirms(rows, q.value, sort.value as SortKey);
    const body = hits.slice(0, LIMIT).map((r) => `<tr>
        <td><a href="#/firm/${r.cik}">${escapeHtml(r.name)}</a></td>
        <td>${escapeHtml(r.ticker ?? "—")}</td>
        <td class="muted">${escapeHtml(r.industry ?? "")}</td>
        <td class="num">${r.first_fy ?? ""}–${r.last_fy ?? ""}</td>
        <td class="num">${r.events}</td>
        <td class="num">${num(r.min_sim_rf)}</td></tr>`).join("");
    const footer = hits.length > LIMIT
      ? `Showing ${LIMIT} of ${hits.length.toLocaleString()}; refine the search.`
      : `${hits.length} firms`;
    list.innerHTML = `<div class="scroll"><table class="firms"><thead><tr><th>Firm</th><th>Ticker</th><th>Industry</th>
      <th class="num">Years</th><th class="num">8-K events</th><th class="num">Lowest RF similarity</th></tr></thead>
      <tbody>${body}</tbody></table></div><p class="muted">${footer}</p>`;
  };
  q.addEventListener("input", draw);
  sort.addEventListener("change", draw);
  draw();
  q.focus();
}
