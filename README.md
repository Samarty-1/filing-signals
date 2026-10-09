# filing-signals

Do changes in what companies write in their annual reports predict their
stock returns? This repo builds the full path from SEC filings to that
answer:

| Phase | | Status |
|---|---|---|
| 1 | **EDGAR ingestion → point-in-time filings warehouse** (Python, DuckDB) | ✅ done |
| 2 | **Section extraction in Rust (via PyO3) → year-over-year change measures** | ✅ done |
| 3a | **Fine-tuned FinBERT: Risk Factors → adverse event in the next 12 months** | ✅ done |
| 3b | **QLoRA-tuned 1.5B LLM: read reported revenue out of MD&A (XBRL as ground truth)** | ✅ done |
| 3c | Fine-tuned embedder: paragraph-level change measure | code ready |
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

## Phase 2: sections and how much they change

### Extracting Item 1A (Risk Factors) and Item 7 (MD&A)

A 10-K mentions "Item 1A" several times: in the table of contents, as the
real heading, and in cross-references ("see Item 1A herein"). The parser
handles this as follows:
1. Converts HTML to clean lines.
2. Counts an Item heading only at the **start of a short line**, which skips
   mid-sentence cross-references.
3. Takes the occurrence with the **longest span** to the next later Item,
   which skips the table of contents.
4. Accepts headings without the word "Item" ("1A. RISK FACTORS"), which are
   common, but only when the item's standard title follows, so numbered lists
   don't match.
5. Classifies the result explicitly instead of guessing:
   - `found`
   - `omitted`: "not required for smaller reporting companies"
   - `by_reference`: "incorporated by reference" to an annual report
   - `too_short`, `unterminated` or `not_found`

The cleaning step decodes Windows-1252 bytes and `&#146;`-style entities
(older filings use both for curly quotes), and it drops inline-XBRL hidden
facts, scripts and comments.

On the 6,907 original 10-Ks:

| Section | Found | Legitimately absent | Not located |
|---|---|---|---|
| MD&A | **95.3%** | 1.4% (incorporated by reference / omitted) | 1.8% (+1.5% too short) |
| Risk Factors | **84.8%** | 9.7% (smaller companies, not required) | 4.2% (+1.3% too short) |

Most of the "not located" filings have no Item structure at all: some small
filers and banks write their 10-K as a cross-reference index into an annual
report. 10-K/A amendments are mostly "not found", which is correct, since a
typical amendment only restates Part III (executive pay and governance).

### Python spec, Rust implementation, identical output

[`textparse.py`](src/filing_signals/textparse.py) is the readable
specification. [`rust/src/core.rs`](rust/src/core.rs) implements the same
algorithm with hand-written byte scanners instead of a chain of regex passes.
Both run the same spec tests, and
[`scripts/bench_parser.py`](scripts/bench_parser.py) runs both over every
stored filing.

| | Time for all 8,082 filings (18.8 GB HTML) | Throughput | Speedup |
|---|---|---|---|
| Python reference | 416 s | 45 MB/s | 1× |
| Rust, one thread | 74 s | 253 MB/s | **5.6×** |
| Rust, 12 threads, gunzip included | 18 s | | **22.9×** |
| **Output mismatches** | **0 of 8,082** | | |

The first Rust port copied the regex-pass structure and was only **1.1×**
faster. Python's `re` is C, and the port allocated a new copy of each document
on every pass, plus one string per tag. The 5.6× came from rewriting the
passes as single-allocation scanners with exactly the same matching rules,
including leftmost match, earliest closing tag and identical entity length
limits. The parity check is what made that rewrite safe. Shared tables, such
as the named-entity list, are generated from Python's
([`scripts/gen_entities.py`](scripts/gen_entities.py)) so the two can't drift.

### Change measures

For every company, each section is compared with the same section a fiscal
year earlier. That gives **10,431 pairs from 865 firms**. The comparison is
point-in-time:
- **This year** is the original 10-K as filed.
- **Last year** is whichever version of the prior report was public at that
  moment, so a later amendment can't leak backwards (tested).

The three measures are raw term-frequency cosine, Jaccard of word sets and
TF-IDF cosine.

| | Pairs | Median cosine | Median Jaccard | Median TF-IDF | Near-verbatim copies |
|---|---|---|---|---|---|
| Risk Factors | 4,921 | 0.998 | 0.896 | 0.993 | **14.2%** |
| MD&A | 5,510 | 0.993 | 0.786 | 0.975 | 1.2% |

Raw cosine is **saturated**. Function words dominate it, so it barely moves.

**Do the measures track real change?** There are no returns until Phase 4,
but there is a check that doesn't need them. A company that completed an
acquisition or disposal during the year (8-K Item 2.01, already in the
warehouse) should rewrite more of its 10-K. AUC here is the probability that
a pair spanning such a deal is *less* similar than one that doesn't:

| | TF-IDF | Jaccard | Raw cosine |
|---|---|---|---|
| MD&A (865 vs 4,645 pairs) | **0.661** | 0.637 | 0.620 |
| Risk Factors (793 vs 4,128 pairs) | **0.635** | 0.630 | 0.614 |

All three measures detect it, and TF-IDF does best in both sections. This is
a check on the measures, not a causal claim, since firms that do deals differ
in other ways too.

## Phase 3a: does Risk Factors text predict trouble, and does fine-tuning help?

**Target:** within 12 months *after* a 10-K, the company files an 8-K for
bankruptcy (Item 1.03), a restatement (Item 4.02) or an auditor change
(Item 4.01). The labels are strictly point-in-time:
- Same-day 8-Ks don't count.
- Filings whose 12-month window hadn't elapsed by the download date are
  excluded (tested).

That gives 5,875 reports. Training covers fiscal years up to 2017,
validation 2018–19 and **test 2020–24** (1,560 reports, 163 events).
Intervals come from a bootstrap that resamples whole **firms**.

| Test years 2020–24 | AUC [95% CI] | Avg precision |
|---|---|---|
| Event in the prior year | 0.557 [0.52, 0.59] | 0.125 |
| **Company size** (public float from the 10-K's XBRL cover) + prior event | **0.697** [0.65, 0.74] | 0.183 |
| TF-IDF + logistic regression | 0.746 [0.71, 0.79] | 0.227 |
| Fine-tuned FinBERT | 0.724 [0.68, 0.77] | 0.210 |
| FinBERT + size + prior | 0.739 [0.70, 0.78] | 0.209 |

**What the text model actually reads.** TF-IDF's top features are "penny
stock", "broker-dealer", "going concern" and "limited operating history". The
strongest negative ones are "pension", "credit facility" and "collective
bargaining". Filing language is largely a **company-size detector**, and
small companies change auditors far more often. So the real question is
whether text beats size.

- **Text over size + prior event: +0.043 AUC, CI [0.024, 0.064].** It's real
  but modest. Text is strongest on **restatements** (0.76 against 0.61 for
  size), where disclosed material weaknesses are a classic warning sign.
  Size is better for bankruptcy.
- **Fine-tuned FinBERT − TF-IDF: −0.022, CI [−0.041, −0.003].** The
  110M-parameter transformer is significantly *worse* than a bag of words.
  Combined with size, the two tie (−0.001, CI [−0.016, 0.014]).

The FinBERT result is the honest kind of negative. There are 388 positive
training documents of ~7,500 tokens each, seen through 512-token windows that
all carry the document's label, and validation AUC peaked after one epoch.
One pre-planned configuration was run, chosen by validation AUC and scored
on the test years once. It was not tuned until it won.

## Phase 3b: can a small fine-tuned LLM read revenue out of MD&A?

**Task.** The model gets MD&A passages from a 10-K and the fiscal year end,
and must answer the year's total revenue in dollars, or `unknown`. The ground
truth is the filing's **own XBRL**: a full-year revenue fact carrying this
10-K's accession and ending on its fiscal period end, which excludes the
prior-year comparatives. That needs no hand labelling and covers 3,749
reports. MD&A runs to ~12k tokens, so the input is each revenue or sales line
plus the six lines after it. That window matters because tables often put
"Net sales" and "17,253" on separate lines.

The figure is in the passage for about 73% of reports, in some written form
(`$1.2 billion`, `1,234.5` in millions, `1,234,567` in thousands…). Those
reports are *answerable*. On the rest the right answer is `unknown`, and the
model was trained towards that, because training it to answer anyway teaches
it to invent numbers.

**Model.** Qwen2.5-1.5B-Instruct, loaded in 4-bit NF4 and frozen, with
rank-16 LoRA adapters on every attention and MLP projection (18.5M trainable
parameters). Loss is on the answer tokens only. It ran for one epoch over
2,006 reports from fiscal years ≤2017, taking 44 minutes on an RTX 4070.

| Test years 2020–24 (1,194 reports) | Correct on answerable (874) | Abstains on unanswerable (320) | Right when it answers |
|---|---|---|---|
| Regex (`revenue … $N [scale]`, table scale note) | 18.9% | 22.2% | 15.3% |
| Same model, zero-shot, same prompt | 24.6% | 17.5% | 19.6% |
| **QLoRA fine-tuned** | **48.4%** | **95.3%** | **91.4%** |

"Correct" means within 0.5% of the XBRL figure.

**Fine-tuning roughly doubles recall: +23.8 points over zero-shot on
answerable reports, 95% CI [19.9, 27.5].** The bigger change is calibration.
The zero-shot model answers 94% of the time and is right a fifth of the time.
The tuned model answers 39% of the time and is right **91%** of the time.
The zero-shot model's typical error is
the unit: 388 of its answers on answerable reports are off by exactly 1,000×
or 1,000,000×, because it copied a figure "in thousands" or "in millions"
without scaling it. Fine-tuning fixed that, and taught it when to say it
doesn't know.

The tuned model also abstains on about half the answerable reports, so its
recall is the open problem. For a data pipeline, high-precision answers with honest
abstentions are the useful operating point, since an abstention can fall back
to XBRL or a human.

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
- **Tests.** There are 49 tests, all offline:
  - The pipeline runs against a fake EDGAR. Tests cover the sampling
    decisions, amendment versions, point-in-time reads, after-hours dating,
    API offset measurement, missing documents and a no-op second run.
  - Every section-parsing spec runs against both the Python and Rust parsers.
  - The point-in-time pairing of change measures is checked.
  - Phase 3 label windows, the size join, the revenue passage and
    answerability logic, and the answer parsers are tested.
- **Resumable GPU training.** QLoRA training checkpoints its adapter,
  optimizer and schedule every 25 steps, and the seeded shuffle makes resuming
  exact. On Windows the driver silently spills VRAM overflow into system RAM,
  which made training **13× slower** halfway through. Capping PyTorch's share
  of the card makes it free its cache instead.

  CI builds the Rust extension and runs everything on Linux and Windows. It
  never calls the SEC.

## Run it

```bash
pip install -e ".[dev]"            # builds the Rust parser via maturin (needs a Rust toolchain)
export EDGAR_USER_AGENT="your-project you@example.com"   # required by the SEC
filing-signals ingest --years 2010 2024 --firms 1000      # ~45 min, ~1.3 GB on disk
filing-signals parse                                     # Rust section parser, ~20 s
filing-signals features                                  # year-over-year change measures
filing-signals report                                    # every finding above
python scripts/bench_parser.py                           # Python-vs-Rust parity and speed
pytest
```

The SEC rejects requests without a contact address in the User-Agent, so it
is read from the environment and never committed.

On Windows without Visual Studio's build tools, the GNU Rust toolchain works
(`rustup-init --default-host x86_64-pc-windows-gnu`). PyO3's import-library
step also needs `dlltool` on `PATH`, which MSYS2's binutils provides.

**Known limitation.** `filers.current_tickers` holds today's tickers. Using
them to map 2012 filings to prices would bring survivorship bias back in.
Phase 4 needs a point-in-time CIK-to-security map, and this is recorded so it
isn't forgotten.

Data: [SEC EDGAR](https://www.sec.gov/edgar), public domain.
