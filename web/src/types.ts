// Shapes of the JSON written by `filing-signals export` (src/filing_signals/export.py).

export interface Report {
  accession: string;
  form: string;
  is_amendment: boolean;
  version_no: number;
  fy: number;
  period_end: string;
  accepted_et: string | null;
  dated_next_business_day: boolean | null;
  ticker: string | null;
  public_float: number | null;
  revenue: number | null;
  sim_rf: number | null;
  sim_mdna: number | null;
  words_rf: number | null;
  words_mdna: number | null;
  // Phase 3, test years (2020+) only
  event_next_12m?: boolean | null;
  risk_pct?: number | null;
  answerable?: boolean | null;
  qlora_usd?: number | null;
  zero_shot_usd?: number | null;
}

export interface FilingEvent {
  date: string;
  items: string;
  kinds: string[];
}

export interface Firm {
  cik: number;
  name: string;
  industry: string | null;
  current_tickers: string | null;
  ticker: string | null;
  reports: Report[];
  events: FilingEvent[];
}

export interface FirmIndexRow {
  cik: number;
  name: string;
  industry: string | null;
  ticker: string | null;
  first_fy: number | null;
  last_fy: number | null;
  reports: number;
  events: number;
  min_sim_rf: number | null;
  has_current_ticker: boolean;
}

export interface Metric {
  n: number;
  positives: number;
  auc: number;
  auc_ci: [number, number];
  avg_precision: number;
}

export interface Diff {
  auc_diff: number;
  ci: [number, number];
}

export interface Extraction {
  answerable_n: number;
  accuracy_answerable: number;
  answered_answerable: number;
  unanswerable_n: number;
  abstain_unanswerable: number;
  accuracy_all: number;
}

export interface EmbedSection {
  pairs: number;
  deal_pairs: number;
  tfidf: Metric;
  novelty_base: Metric;
  novelty_tuned: Metric;
  tuned_vs_tfidf: Diff;
  tuned_vs_base: Diff;
}

export interface Summary {
  counts: {
    reports: number;
    firms: number;
    amendments: number;
    section_pairs: number;
    first_fy: number;
    last_fy: number;
  };
  results: {
    baseline?: Record<string, Metric> & { text_vs_size: Diff };
    encoder?: { encoder: Metric; "encoder+size+prior": Metric; vs_tfidf: Diff };
    llm_extract?: { model: string; regex: Extraction; zero_shot: Extraction; qlora: Extraction };
    embed_change?: { mdna: EmbedSection; risk_factors: EmbedSection };
    phase4?: Record<string, unknown> | null;
  };
}
