# Phase 4 pre-registration: does unchanged 10-K text predict returns?

Committed **before any stock price was downloaded**. The commit that adds
this file is the timestamp. Every choice below is fixed; anything decided
after prices arrive will be labelled exploratory in the results.

## Hypothesis

Cohen, Malloy & Nguyen (2020, *Lazy Prices*) find that firms whose 10-K
language changes a lot from one year to the next underperform firms whose
text is unchanged, because investors are slow to read the changes. Their
sample ends before publication. This sample (filings 2011–2025) is mostly
after it, so the prior for a surviving effect is modest.

**H1:** a monthly portfolio long the fifth of firms whose Risk Factors text is
most similar to last year's, and short the fifth whose text changed most,
earns a positive alpha against the Fama-French five factors plus momentum.

## Data, fixed before prices

- **Filings:** 999 randomly sampled SEC filers, original 10-Ks for fiscal
  years 2010–2024 (Phase 1).
- **Signal:** `sim_tfidf_pit`, the TF-IDF cosine similarity between this year's
  section and last year's version as it was public at filing. IDF is fitted
  only on documents accepted before 1 January of the filing's acceptance year,
  so no word weight uses the future (Phase 2, `features.py`). High means
  unchanged.
- **Tickers:** taken from the 10-K itself: the inline-XBRL `dei:TradingSymbol`
  tag, otherwise the Item 5 "under the symbol" sentence. A ticker is resolved
  to a Tiingo symbol in this order: the same ticker, its bankruptcy Q ticker,
  share-class variants, then the same CIK's current ticker. The symbol must
  still have been listed at filing.
- **Ticker validation:** a match is used only if the symbol has a price
  within 7 days of the filing, and if the cover page's public float divided
  by (price × cover shares outstanding) is between 0.05 and 2.0. A failed
  match drops that report. The drop count is reported.
- **Universe:** reports whose cover public float is ≥ $100M. This is set
  before prices partly because ticker coverage there is 92%, against 50–79%
  below it. Survivorship check (reported): coverage is 82% for firms that
  have stopped filing and 97% for firms still filing.
- **Prices:** Tiingo daily adjusted closes, which include dividends and splits.
- **Factors:** Ken French data library, FF5 + momentum, monthly.

## Portfolio

- **Formation:** at each month's last trading close. A firm enters with its
  most recent original 10-K, which must have been accepted on an earlier
  calendar day than that close and within the last 365 days.
- **Sort:** quintiles of the signal within the month. A month needs ≥ 50
  firms or it is skipped.
- **Holding:** the next calendar month, close to close.
- **Return:** Q5 (most similar) minus Q1 (most changed).
- **Weighting:** **equal-weighted (primary)**; weighted by public float
  (secondary).
- **Delisting:** a series that ends inside the holding month earns its
  return to the last close. If the firm filed an 8-K Item 1.03 (bankruptcy)
  in the year before, −30% is compounded on top (Shumway 1997), and that is
  the primary treatment. "No adjustment" is reported as a sensitivity.
  Delisting proceeds earn zero.

## Test and decision rule

**Primary:** Risk Factors, equal-weighted, Q5−Q1, with the Shumway delisting
treatment.

**H1 is CONFIRMED only if both hold:**
1. The FF5 + momentum alpha is positive, with a Newey-West (3 lags) t-stat
   ≥ 2.0.
2. The deflated Sharpe ratio is ≥ 0.95. It is computed against all **6
   reported trials**: {Risk Factors, MD&A, both (mean of the two sections'
   within-month percentile ranks)} × {equal, float-weighted}.

Otherwise the result is **NOT CONFIRMED**, whatever the secondary series show.

Also reported, whatever the outcome:
- All 6 trials: mean, Newey-West t, Sharpe, alpha and factor loadings.
- The quintile spread.
- The minimum detectable monthly effect at 80% power. About 180–240 firms a
  year qualify (36–48 per quintile), so the test is underpowered for
  effects much under ~1% a month. A null result is weak evidence of no
  effect, and the write-up will say so.

## What was seen before this was written

- **Phases 1–3:** text statistics, the 8-K deal and adverse-event
  validations, and model results. None of them involve returns or prices.
- **One ticker-resolution choice was made after looking at the data.** The
  date rule was relaxed because Tiingo's listed start dates proved
  unreliable. That was settled before any price was fetched.
- **No price, return or portfolio statistic for these firms has been
  computed.** The only market data loaded so far is the Fama-French factor
  file, used to test the parser.
