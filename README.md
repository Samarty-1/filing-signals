# filing-signals

Do changes in what companies write in their annual reports predict their
stock returns? This repo builds the full path from SEC filings to that
answer:

| Phase | | Status |
|---|---|---|
| 1 | **EDGAR ingestion → point-in-time filings warehouse** (Python, DuckDB) | ✅ done |
| 2 | Section extraction (Risk Factors, MD&A) with a Rust parser via PyO3, plus a TF-IDF baseline | next |
| 3 | Fine-tuned models: a FinBERT-class encoder, a QLoRA small LLM for extraction, an embedding model | |
| 4 | Pre-registered return test, judged with deflated Sharpe and PBO from [backtest-overfit-audit](https://github.com/Samarty-1/backtest-overfit-audit) | |

## Phase 1: what's in the warehouse

**8,082 annual reports** (10-K and amendments) from **1,000 companies**,
2010–2024. That's 18.8 GB of filing text, stored as 1.3 GB gzipped, along
with 752k filing records of every form type for the same companies.

**The sample is drawn without survivorship bias.** Candidates are every
company that filed an annual report anywhere in the window (16,923 of them),
so firms that later went bankrupt or were acquired are as likely to be drawn
as survivors. They're drawn in a seeded order until 1,000 qualify, and every
decision is recorded. The only exclusions were 82 asset-backed trusts, which
file 10-Ks but have no stock to price. Re-running with the same seed draws
the same firms.

## When did the text become public?

This is the column every later phase depends on. EDGAR offers three
candidates, and two of them are wrong in ways that matter.

**`filing_date` is wrong for 11.2% of annual reports.** EDGAR dates anything
accepted after 17:30 ET on the next business day, so 11.2% of these reports
were public the evening before their filing date. 60.5% are accepted after
the 16:00 close, and 9.4% before the 09:30 open.

**The submissions API's `acceptanceDateTime` is wrong for 11.5% of filings.**
It is labelled UTC (`…Z`). For 88.5% of filings it really is UTC. For the
rest it is 4 or 5 hours late: the Eastern-to-UTC shift applied twice. Apple's
2024 10-K was accepted at 06:01 ET (header), but the API implies 10:01 ET.

- **The error belongs to the filer, not the filing.** Of 823 filers with
  three or more reports, 750 are always right and 73 are always wrong, so it
  doesn't average out.
- **It's getting worse.** The error affects 6% of 2010 filings and 21% of
  2024 filings.
- **It changes the trading session.** For **3.7% of annual reports**,
  trusting the API puts the filing in the wrong session (before the open,
  during trading or after the close, on which date). For an event study
  that's a look-ahead or a missed day.

The warehouse therefore takes acceptance time from each filing's SGML header
(`.hdr.sgml`), keeps the API value alongside, and measures the disagreement
in `api_timestamp_check`.

One more trap: the friendlier `-index-headers.html` page that carries the
same header **404s for most filings before mid-2014**. That cost 3,041 of the
first run's header fetches until the pipeline switched to `.hdr.sgml`.

## Versions and point-in-time reads

13.8% of fiscal years get at least one 10-K/A, a median 52 days after the
original, and one has five. Each amendment is a **version** of the same
report:
- `annual_reports` numbers each version by when it became public.
- `annual_reports_as_of(ts)` returns what an investor could actually have
  read at `ts`.

Anything that joins text to returns goes through the `as_of` function, never
the raw table:

```sql
select cik, fiscal_period_end, accession, version_no
from annual_reports_as_of('2015-03-21 00:00:00+00');
```

## Engineering

- **Polite client.** It caps at 8 requests/second (the SEC limit is 10),
  backs off exponentially on 429/5xx and honours `Retry-After`. A 403
  (undeclared or over-limit) fails loudly instead of being retried. The full
  run made 17,512 requests with 8 retries.
- **Idempotent and resumable.** Every response is cached on disk, the
  warehouse upserts, and finished work is skipped. A URL that 404'd isn't
  retried, but a filing whose source URL changes is. The header fix above was
  applied by re-running the same command: 3,041 requests, nothing else
  refetched.
- **Provenance.** Every document is stored with its URL, SHA-256 and byte
  counts. Writes are atomic, so an interrupted run never leaves half a file.
- **Fast loads.** Bulk upserts go through Arrow. The first version used
  row-by-row `executemany` and took over two minutes per run on just a
  quarter's index.
- **Tests.** 19 tests run offline against a fake EDGAR. They cover the
  sampling decisions, amendment versions, point-in-time reads, after-hours
  dating, API offset measurement, missing documents and a no-op second run.
  CI runs them on Linux and Windows and never calls the SEC.

## Run it

```bash
pip install -e ".[dev]"
export EDGAR_USER_AGENT="your-project you@example.com"   # required by the SEC
filing-signals ingest --years 2010 2024 --firms 1000      # ~45 min, ~1.3 GB on disk
filing-signals report                                    # the findings above
pytest
```

The SEC rejects requests without a contact address in the User-Agent, so it
is read from the environment and never committed.

**Known limitation.** `filers.current_tickers` holds today's tickers. Using
them to map 2012 filings to prices would bring survivorship bias back in.
Phase 4 needs a point-in-time CIK-to-security map, and this is recorded so it
isn't forgotten.

Data: [SEC EDGAR](https://www.sec.gov/edgar), public domain.
