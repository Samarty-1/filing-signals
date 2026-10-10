// A small SVG time-series chart: year-over-year text similarity per section,
// with 8-K events marked underneath. No chart library; the scale and tick
// helpers are pure and unit-tested.

import { escapeHtml } from "./format";

export interface Point { t: number; y: number; label: string }
export interface Series { name: string; cls: string; points: Point[] }
export interface Marker { t: number; cls: string; label: string }

export function linear(d0: number, d1: number, r0: number, r1: number): (v: number) => number {
  const span = d1 - d0 || 1;
  return (v) => r0 + ((v - d0) / span) * (r1 - r0);
}

/** About `count` round tick values covering [lo, hi]. */
export function niceTicks(lo: number, hi: number, count = 5): number[] {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) {
    out.push(Number(v.toFixed(10)));
  }
  return out;
}

/** The y-range to show: similarity lives near 1, so zoom to the data but
 * never hide 1.0 and never draw a range thinner than `minSpan`. */
export function yDomain(values: number[], minSpan = 0.1): [number, number] {
  const finite = values.filter(Number.isFinite);
  if (!finite.length) return [1 - minSpan, 1];
  let lo = Math.min(...finite);
  lo = Math.min(lo - (1 - lo) * 0.1, 1 - minSpan);
  return [Math.max(0, lo), 1];
}

/** Fewest decimals that print every tick exactly (0.975 must not read 0.97). */
export function tickDecimals(ticks: number[]): number {
  for (let d = 0; d < 4; d++) {
    if (ticks.every((v) => Math.abs(Number(v.toFixed(d)) - v) < 1e-9)) return d;
  }
  return 4;
}

export function yearStart(year: number): number {
  return Date.UTC(year, 0, 1);
}

export function timeSeriesChart(series: Series[], markers: Marker[], opts: { width?: number; height?: number } = {}): string {
  const W = opts.width ?? 760;
  const H = opts.height ?? 280;
  const m = { top: 16, right: 16, bottom: 52, left: 48 };
  const pointTs = series.flatMap((s) => s.points.map((p) => p.t));
  // events are context: keep the ones within the years the reports cover
  const lo0 = Math.min(...pointTs), hi0 = Math.max(...pointTs);
  const shown = pointTs.length
    ? markers.filter((k) => k.t >= lo0 - 366 * 864e5 && k.t <= hi0 + 366 * 864e5)
    : markers;
  const ts = [...pointTs, ...shown.map((k) => k.t)];
  if (!ts.length) return `<p class="muted">No year-over-year pairs for this firm.</p>`;
  const y0 = new Date(Math.min(...ts)).getUTCFullYear();
  const y1 = new Date(Math.max(...ts)).getUTCFullYear() + 1;
  const x = linear(yearStart(y0), yearStart(y1), m.left, W - m.right);
  const [lo, hi] = yDomain(series.flatMap((s) => s.points.map((p) => p.y)));
  const plotBottom = H - m.bottom;
  const y = linear(lo, hi, plotBottom, m.top);

  const parts: string[] = [];
  const yTicks = niceTicks(lo, hi, 4);
  const dec = tickDecimals(yTicks);
  for (const v of yTicks) {
    parts.push(`<line class="grid" x1="${m.left}" x2="${W - m.right}" y1="${y(v)}" y2="${y(v)}"/>`,
      `<text class="tick" x="${m.left - 6}" y="${y(v) + 4}" text-anchor="end">${v.toFixed(dec)}</text>`);
  }
  const step = y1 - y0 > 12 ? 2 : 1;
  for (let yr = y0; yr <= y1; yr += step) {
    parts.push(`<text class="tick" x="${x(yearStart(yr))}" y="${plotBottom + 16}" text-anchor="middle">${yr}</text>`);
  }
  parts.push(`<line class="axis" x1="${m.left}" x2="${W - m.right}" y1="${plotBottom}" y2="${plotBottom}"/>`);

  for (const s of series) {
    const pts = [...s.points].sort((a, b) => a.t - b.t);
    if (pts.length > 1) {
      parts.push(`<polyline class="line ${s.cls}" points="${pts.map((p) => `${x(p.t)},${y(p.y)}`).join(" ")}"/>`);
    }
    for (const p of pts) {
      parts.push(`<circle class="dot ${s.cls}" cx="${x(p.t)}" cy="${y(p.y)}" r="4"><title>${escapeHtml(p.label)}</title></circle>`);
    }
  }
  // events: a lane under the axis, one glyph per 8-K
  const lane = plotBottom + 32;
  for (const k of shown) {
    parts.push(`<line class="event-rule ${k.cls}" x1="${x(k.t)}" x2="${x(k.t)}" y1="${m.top}" y2="${plotBottom}"/>`,
      `<rect class="event ${k.cls}" x="${x(k.t) - 4}" y="${lane - 4}" width="8" height="8" rx="1.5"><title>${escapeHtml(k.label)}</title></rect>`);
  }
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Year-over-year text similarity">${parts.join("")}</svg>`;
}
