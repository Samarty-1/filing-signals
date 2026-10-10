import { describe, expect, it } from "vitest";
import { linear, niceTicks, timeSeriesChart, yDomain } from "./chart";
import { escapeHtml, money, pct, precisionWhenAnswering, revenueMatches, significant } from "./format";
import { parseRoute } from "./router";
import type { Firm, FirmIndexRow, Report } from "./types";
import { edgarUrl, eventMarkers, firmHtml, similaritySeries } from "./views/firm";
import { filterFirms } from "./views/firms";

describe("format", () => {
  it("formats money at sensible scales", () => {
    expect(money(68_636_146_000)).toBe("$68.6B");
    expect(money(2_227_559)).toBe("$2.2M");
    expect(money(37_637_910)).toBe("$38M");
    expect(money(null)).toBe("—");
    expect(money(-4_500)).toBe("-$5K");
  });

  it("formats percentages", () => {
    expect(pct(0.4841)).toBe("48.4%");
    expect(pct(undefined)).toBe("—");
  });

  it("flags intervals that exclude zero", () => {
    expect(significant([-0.041, -0.003])).toBe(true);
    expect(significant([-0.016, 0.014])).toBe(false);
  });

  it("uses the Phase 3b 0.5% revenue tolerance", () => {
    expect(revenueMatches(100.4e6, 100e6)).toBe(true);
    expect(revenueMatches(100.6e6, 100e6)).toBe(false);
    expect(revenueMatches(null, 100e6)).toBe(false);
  });

  it("reproduces the README's 91.4% precision for the QLoRA model", () => {
    const qlora = { answerable_n: 874, unanswerable_n: 320, answered_answerable: 0.515,
      abstain_unanswerable: 0.953, accuracy_all: 0.356 };
    expect(precisionWhenAnswering(qlora)).toBeCloseTo(0.914, 2);
  });

  it("escapes HTML from filing text", () => {
    expect(escapeHtml(`AT&T <b>"x"</b>`)).toBe("AT&amp;T &lt;b&gt;&quot;x&quot;&lt;/b&gt;");
  });
});

describe("chart helpers", () => {
  it("maps linearly, including inverted ranges for SVG y", () => {
    const y = linear(0.9, 1, 200, 0);
    expect(y(0.9)).toBe(200);
    expect(y(1)).toBe(0);
    expect(y(0.95)).toBeCloseTo(100);
  });

  it("produces round ticks inside the domain", () => {
    expect(niceTicks(0.8, 1, 4)).toEqual([0.8, 0.85, 0.9, 0.95, 1]);
    expect(niceTicks(0, 1, 5)).toEqual([0, 0.2, 0.4, 0.6, 0.8, 1]);
  });

  it("keeps 1.0 in view and never zooms below a minimum span", () => {
    expect(yDomain([0.995, 0.999])).toEqual([0.9, 1]);
    const [lo, hi] = yDomain([0.4, 0.99]);
    expect(hi).toBe(1);
    expect(lo).toBeLessThan(0.4);
    expect(lo).toBeGreaterThanOrEqual(0);
  });

  it("renders an empty-state message instead of an empty SVG", () => {
    expect(timeSeriesChart([], [])).toContain("No year-over-year pairs");
  });
});

const report = (over: Partial<Report>): Report => ({
  accession: "0000000000-20-000001", form: "10-K", is_amendment: false, version_no: 1, fy: 2020,
  period_end: "2020-12-31", accepted_et: "2021-02-20 16:05", dated_next_business_day: false,
  ticker: "ABC", public_float: 5e8, revenue: 1e9, sim_rf: 0.98, sim_mdna: 0.95, words_rf: 1000,
  words_mdna: 2000, ...over,
});

describe("firm view", () => {
  it("plots originals only, never amendments", () => {
    const s = similaritySeries([report({}), report({ version_no: 2, is_amendment: true, sim_rf: 0.1 })]);
    expect(s[0]!.points.map((p) => p.y)).toEqual([0.98]);
  });

  it("colours events by their first kind", () => {
    const m = eventMarkers([{ date: "2021-05-01", items: "4.02,9.01", kinds: ["restatement"] }]);
    expect(m[0]!.cls).toBe("ev-restatement");
  });

  it("links to the filing's EDGAR folder", () => {
    expect(edgarUrl(320193, "0000320193-24-000123"))
      .toBe("https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/");
  });

  it("escapes firm names in the page", () => {
    const firm: Firm = { cik: 1, name: "<script>x</script> Corp", industry: null, current_tickers: null,
      ticker: null, reports: [report({})], events: [] };
    const html = firmHtml(firm);
    expect(html).not.toContain("<script>x");
    expect(html).toContain("no current ticker");
  });
});

describe("firm search", () => {
  const rows: FirmIndexRow[] = [
    { cik: 1, name: "Apple Inc.", industry: "Electronic Computers", ticker: "AAPL", first_fy: 2010, last_fy: 2024,
      reports: 15, events: 2, min_sim_rf: 0.97, has_current_ticker: true },
    { cik: 2, name: "Sears Holdings", industry: "Retail-Department Stores", ticker: "SHLD", first_fy: 2010, last_fy: 2018,
      reports: 9, events: 6, min_sim_rf: 0.61, has_current_ticker: false },
    { cik: 3, name: "Shell Co", industry: null, ticker: null, first_fy: 2015, last_fy: 2016,
      reports: 2, events: 0, min_sim_rf: null, has_current_ticker: false },
  ];

  it("matches names, exact tickers, CIKs and industries", () => {
    expect(filterFirms(rows, "sears", "name").map((r) => r.cik)).toEqual([2]);
    expect(filterFirms(rows, "aapl", "name").map((r) => r.cik)).toEqual([1]);
    expect(filterFirms(rows, "3", "name").map((r) => r.cik)).toEqual([3]);
    expect(filterFirms(rows, "retail", "name").map((r) => r.cik)).toEqual([2]);
  });

  it("sorts the biggest rewrite first and firms without pairs last", () => {
    expect(filterFirms(rows, "", "rewrite").map((r) => r.cik)).toEqual([2, 1, 3]);
  });
});

describe("router", () => {
  it("parses hash routes", () => {
    expect(parseRoute("#/firm/320193")).toEqual({ view: "firm", cik: 320193 });
    expect(parseRoute("#/firms")).toEqual({ view: "firms" });
    expect(parseRoute("")).toEqual({ view: "overview" });
    expect(parseRoute("#/firm/abc")).toEqual({ view: "overview" });
  });
});

describe("tick labels", () => {
  it("prints every tick exactly", async () => {
    const { niceTicks, tickDecimals } = await import("./chart");
    for (const [lo, hi] of [[0.9, 1], [0.92, 1], [0.8, 1], [0.95, 1], [0, 1], [0.5, 1]] as const) {
      const t = niceTicks(lo, hi, 4);
      const labels = t.map((v) => v.toFixed(tickDecimals(t)));
      labels.forEach((l, i) => expect(Number(l)).toBeCloseTo(t[i]!, 9));
      expect(new Set(labels).size).toBe(labels.length);
    }
  });

  it("drops events far outside the reports' years", async () => {
    const { timeSeriesChart } = await import("./chart");
    const svg = timeSeriesChart(
      [{ name: "RF", cls: "s-rf", points: [{ t: Date.UTC(2015, 11, 31), y: 0.97, label: "" }, { t: Date.UTC(2018, 11, 31), y: 0.99, label: "" }] }],
      [{ t: Date.UTC(2005, 5, 1), cls: "ev-deal", label: "old" }, { t: Date.UTC(2017, 5, 1), cls: "ev-deal", label: "in range" }]);
    expect(svg).toContain("in range");
    expect(svg).not.toContain(">old<");
  });
});
