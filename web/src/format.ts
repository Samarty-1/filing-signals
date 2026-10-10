// Number formatting shared by every view. Pure functions, unit-tested.

export function money(usd: number | null | undefined): string {
  if (usd == null || !Number.isFinite(usd)) return "—";
  const abs = Math.abs(usd);
  const sign = usd < 0 ? "-" : "";
  if (abs >= 1e9) return `${sign}$${(abs / 1e9).toFixed(abs >= 1e10 ? 1 : 2)}B`;
  if (abs >= 1e6) return `${sign}$${(abs / 1e6).toFixed(abs >= 1e7 ? 0 : 1)}M`;
  if (abs >= 1e3) return `${sign}$${(abs / 1e3).toFixed(0)}K`;
  return `${sign}$${abs.toFixed(0)}`;
}

export function pct(x: number | null | undefined, digits = 1): string {
  if (x == null || !Number.isFinite(x)) return "—";
  return `${(x * 100).toFixed(digits)}%`;
}

export function num(x: number | null | undefined, digits = 3): string {
  if (x == null || !Number.isFinite(x)) return "—";
  return x.toFixed(digits);
}

export function ci(pair: [number, number] | undefined, digits = 3): string {
  if (!pair) return "";
  return `[${pair[0].toFixed(digits)}, ${pair[1].toFixed(digits)}]`;
}

/** A confidence interval that excludes zero, for highlighting. */
export function significant(pair: [number, number] | undefined): boolean {
  return !!pair && (pair[0] > 0 || pair[1] < 0);
}

/** True when an extracted revenue is within 0.5% of the XBRL figure (the Phase 3b rule). */
export function revenueMatches(pred: number | null | undefined, truth: number | null | undefined): boolean {
  if (pred == null || truth == null || truth <= 0) return false;
  return Math.abs(pred - truth) <= 0.005 * truth;
}

export function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);
}

/** Share of given answers that were correct (Phase 3b "right when it answers"). */
export function precisionWhenAnswering(x: {
  answerable_n: number; unanswerable_n: number; answered_answerable: number;
  abstain_unanswerable: number; accuracy_all: number;
}): number {
  const answered = x.answered_answerable * x.answerable_n + (1 - x.abstain_unanswerable) * x.unanswerable_n;
  return answered > 0 ? (x.accuracy_all * (x.answerable_n + x.unanswerable_n)) / answered : NaN;
}
