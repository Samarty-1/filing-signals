import { loadSummary } from "../data";
import { ci, num, pct, precisionWhenAnswering, significant } from "../format";
import type { Diff, Metric, Summary } from "../types";

const REPO = "https://github.com/Samarty-1/filing-signals";

function metricRow(label: string, m: Metric | undefined, note = ""): string {
  if (!m) return "";
  return `<tr><td>${label}${note ? ` <span class="muted">${note}</span>` : ""}</td>
    <td class="num">${num(m.auc)}</td><td class="num muted">${ci(m.auc_ci, 2)}</td>
    <td class="num">${num(m.avg_precision)}</td></tr>`;
}

function diffRow(label: string, d: Diff | undefined): string {
  if (!d) return "";
  const cls = significant(d.ci) ? (d.auc_diff > 0 ? "pos" : "neg") : "muted";
  return `<tr><td>${label}</td><td class="num ${cls}">${d.auc_diff > 0 ? "+" : ""}${num(d.auc_diff)}</td>
    <td class="num muted">${ci(d.ci)}</td><td>${significant(d.ci) ? "significant" : "within noise"}</td></tr>`;
}

function phase3a(s: Summary): string {
  const b = s.results.baseline, e = s.results.encoder;
  if (!b) return "";
  return `<section class="card"><h2>3a · Does Risk Factors text predict trouble?</h2>
    <p>Target: an 8-K for bankruptcy, restatement or auditor change within 12 months after the 10-K.
    Test years 2020–24, ${b.tfidf?.n ?? ""} reports, ${b.tfidf?.positives ?? ""} events. Intervals resample whole firms.</p>
    <table><thead><tr><th>Model</th><th class="num">AUC</th><th class="num">95% CI</th><th class="num">Avg precision</th></tr></thead><tbody>
    ${metricRow("Event in the prior year", b.prior_event)}
    ${metricRow("Company size + prior event", b["size+prior"])}
    ${metricRow("TF-IDF + logistic regression", b.tfidf)}
    ${metricRow("Fine-tuned FinBERT", e?.encoder)}
    ${metricRow("FinBERT + size + prior", e?.["encoder+size+prior"])}
    </tbody></table>
    <table class="diffs"><thead><tr><th>Comparison</th><th class="num">ΔAUC</th><th class="num">95% CI</th><th></th></tr></thead><tbody>
    ${diffRow("Text over size + prior event", b.text_vs_size)}
    ${diffRow("FinBERT − TF-IDF", e?.vs_tfidf)}
    </tbody></table>
    <p class="takeaway">The 110M-parameter transformer loses to a bag of words; most of the text signal identifies small companies.</p></section>`;
}

function phase3b(s: Summary): string {
  const r = s.results.llm_extract;
  if (!r) return "";
  const row = (label: string, x: typeof r.regex, strong = false) => `<tr${strong ? ' class="strong"' : ""}><td>${label}</td>
    <td class="num">${pct(x.accuracy_answerable)}</td><td class="num">${pct(x.abstain_unanswerable)}</td>
    <td class="num">${pct(precisionWhenAnswering(x))}</td></tr>`;
  return `<section class="card"><h2>3b · Reading revenue out of MD&amp;A with a fine-tuned 1.5B LLM</h2>
    <p>${r.model}, QLoRA (4-bit base, rank-16 adapters). Ground truth is the filing's own XBRL; correct means within 0.5%.
    Test years 2020–24: ${r.qlora.answerable_n} reports whose figure is in the passage, ${r.qlora.unanswerable_n} where it isn't.</p>
    <table><thead><tr><th>Method</th><th class="num">Correct when stated</th><th class="num">Says "unknown" when not</th><th class="num">Right when it answers</th></tr></thead><tbody>
    ${row("Regex", r.regex)}${row("Same model, zero-shot", r.zero_shot)}${row("QLoRA fine-tuned", r.qlora, true)}
    </tbody></table>
    <p class="takeaway">Fine-tuning roughly doubles accuracy and makes the model calibrated: it abstains instead of guessing.</p></section>`;
}

function phase3c(s: Summary): string {
  const r = s.results.embed_change;
  if (!r) return "";
  const rows = (["mdna", "risk_factors"] as const).map((k) => {
    const x = r[k];
    return `<tr><td>${k === "mdna" ? "MD&amp;A" : "Risk Factors"}</td><td class="num">${num(x.tfidf.auc)}</td>
      <td class="num">${num(x.novelty_base.auc)}</td><td class="num">${num(x.novelty_tuned.auc)}</td>
      <td class="num ${significant(x.tuned_vs_tfidf.ci) ? "neg" : "muted"}">${num(x.tuned_vs_tfidf.auc_diff)} <span class="muted">${ci(x.tuned_vs_tfidf.ci)}</span></td></tr>`;
  }).join("");
  return `<section class="card"><h2>3c · A learned paragraph-level change measure</h2>
    <p>Does the measure rise when a firm completed an acquisition or disposal between two reports? AUC, fiscal years 2020+.</p>
    <table><thead><tr><th>Section</th><th class="num">TF-IDF</th><th class="num">Embedder</th><th class="num">Fine-tuned</th><th class="num">Tuned − TF-IDF</th></tr></thead><tbody>${rows}</tbody></table>
    <p class="takeaway">Negative result: deals show up as new vocabulary, which TF-IDF weights and embeddings are built to look past.</p></section>`;
}

function phase4(s: Summary): string {
  const done = s.results.phase4;
  return `<section class="card"><h2>4 · Does unchanged text predict returns?</h2>
    ${done ? `<pre>${JSON.stringify(done, null, 1).slice(0, 2000)}</pre>`
      : `<p>Pre-registered and waiting on prices. The hypothesis, universe, portfolio rules and the pass/fail rule were
      <a href="${REPO}/blob/main/docs/PREREGISTRATION.md">committed before any price was downloaded</a>.</p>`}</section>`;
}

export async function renderOverview(root: HTMLElement): Promise<void> {
  const s = await loadSummary();
  const c = s.counts;
  root.innerHTML = `
    <header class="hero">
      <h1>What changes in a 10-K, and does it matter?</h1>
      <p>An end-to-end pipeline over SEC annual reports: point-in-time EDGAR warehouse, a Rust section parser,
      three fine-tuned language models and a pre-registered return test. <a href="${REPO}">Source on GitHub</a>.</p>
      <div class="stats">
        <div><b>${c.firms.toLocaleString()}</b><span>randomly sampled firms</span></div>
        <div><b>${c.reports.toLocaleString()}</b><span>annual reports</span></div>
        <div><b>${c.section_pairs.toLocaleString()}</b><span>year-over-year section pairs</span></div>
        <div><b>${c.amendments.toLocaleString()}</b><span>amendments kept as versions</span></div>
      </div>
      <p><a class="button" href="#/firms">Explore the firms →</a></p>
    </header>
    ${phase3a(s)}${phase3b(s)}${phase3c(s)}${phase4(s)}`;
}
