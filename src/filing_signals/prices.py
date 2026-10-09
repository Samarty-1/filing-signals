"""Phase 4 data: point-in-time tickers and daily prices, delisted firms included.

Tickers. The filer's current ticker (submissions API) is survivorship-biased:
half the firms in the sample no longer exist. Instead each 10-K supplies its
own ticker as of filing: the inline-XBRL cover tag dei:TradingSymbol when
present (2019+), otherwise the Item 5 sentence nearly every listed company
writes ("traded on the NASDAQ under the symbol 'ABCD'").

Prices. Tiingo keeps delisted securities, and moves a bankrupt company's
history to its Q ticker (SVB: SIVB -> SIVBQ, prices from 1990). A filing's
ticker resolves to the Tiingo symbol whose listed date range covers the
filing date; ticker, then ticker+Q, then share-class punctuation variants.
Free accounts get 500 unique symbols a month, so fetching is budgeted and
resumable: each symbol's history is cached on disk once.
"""
from __future__ import annotations

import collections
import gzip
import json
import os
import re
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

from . import textparse

TIINGO_LIST_URL = "https://apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip"
TIINGO_PRICES = "https://api.tiingo.com/tiingo/daily/{ticker}/prices"
PRICE_START = "2009-01-01"
MONTHLY_SYMBOLS = 480          # free tier allows 500; keep a margin for retries
HOURLY_REQUESTS = 45           # free tier allows 50

_TAG = re.compile(r'name="dei:TradingSymbol"[^>]*>(?:\s*<[^>]+>)*\s*([A-Za-z][A-Za-z.\-]{0,7})')
_SYMBOL = re.compile(
    r"(?:under\s+the\s+(?:ticker\s+|trading\s+)?symbols?|ticker\s+symbols?|trading\s+symbols?|symbol)"
    r"""\s*[:"“'‘(]*\s*([A-Z]{1,5}(?:[.\-][A-Z]{1,2})?)\b""")
_NOT_TICKERS = {"A", "I", "THE", "OUR", "NYSE", "OTC", "OTCBB", "OTCQB", "OTCQX", "USD", "AMEX",
                "NASDAQ", "MKT", "LLC", "INC", "CORP", "CO", "AND", "OF", "IS", "ON", "PINK", "QB"}

SCHEMA = """
create table if not exists report_tickers (
    accession varchar primary key,
    cik bigint,
    ticker varchar,           -- as written in the filing, upper case
    source varchar,           -- 'xbrl_tag' | 'item5_text' | null
    candidates varchar[]      -- every distinct symbol the text mentioned
);
"""

RESOLVED_SQL = """
-- the Tiingo symbol each report's ticker referred to when it was filed.
-- Candidates in order: the filing's own ticker, its bankruptcy Q ticker,
-- share-class punctuation variants, then the same CIK's current tickers
-- (renames move history to the new symbol: CenturyLink CTL -> LUMN).
-- Tiingo's listed start dates are unreliable (Linear Technology, trading
-- since 1986, is listed from 2016), so only "still listed at filing" is
-- required here; validate_matches() then checks the actual prices.
with cand as (
    select r.accession, r.cik, a.accepted_at::date as filed, r.ticker,
           unnest(list_concat(
               [r.ticker, r.ticker || 'Q', replace(r.ticker, '.', '-'), replace(r.ticker, '-', '.')],
               coalesce(string_split(f.current_tickers, ','), []))) as symbol,
           generate_subscripts(list_concat(
               [r.ticker, r.ticker || 'Q', replace(r.ticker, '.', '-'), replace(r.ticker, '-', '.')],
               coalesce(string_split(f.current_tickers, ','), [])), 1) as pref
    from report_tickers r
    join annual_reports a using (accession)
    join filers f on f.cik = r.cik
    where a.version_no = 1 and (r.ticker is not null or f.current_tickers is not null)
)
select c.accession, c.cik, c.filed, c.ticker, c.symbol, c.pref > 4 as via_current_ticker,
       t.exchange, t.end_date
from cand c
join tiingo_tickers t on t.ticker = upper(c.symbol)
where c.symbol is not null and t.asset_type = 'Stock' and t.currency = 'USD'
  and t.end_date >= c.filed
qualify row_number() over (partition by c.accession order by c.pref) = 1
"""


# --- tickers from the filings themselves ----------------------------------------

def extract_ticker(raw: bytes) -> tuple[str | None, str | None, list[str]]:
    """(ticker, source, candidates) from one 10-K document."""
    tag = _TAG.search(raw.decode("utf-8", "ignore"))
    text = " ".join(textparse.to_lines(raw))
    found = [m for m in _SYMBOL.findall(text) if m not in _NOT_TICKERS]
    candidates = list(dict.fromkeys(found))
    if tag:
        return tag.group(1).upper(), "xbrl_tag", candidates
    if found:
        # the most-mentioned symbol; ties go to the first mentioned
        counts = collections.Counter(found)
        return max(candidates, key=lambda s: counts[s]), "item5_text", candidates
    return None, None, candidates


def _extract_file(args):
    accession, cik, path = args
    ticker, source, candidates = extract_ticker(gzip.decompress(Path(path).read_bytes()))
    return {"accession": accession, "cik": cik, "ticker": ticker, "source": source,
            "candidates": candidates}


def build_ticker_map(con: duckdb.DuckDBPyConnection, data: Path, log=print) -> int:
    con.execute(SCHEMA)
    todo = con.execute("""
        select d.accession, a.cik, d.stored_path from documents d
        join annual_reports a using (accession)
        where d.accession not in (select accession from report_tickers)""").fetchall()
    log(f"extracting tickers from {len(todo):,} reports")
    jobs = [(acc, cik, str(data / "raw" / p)) for acc, cik, p in todo]
    with ProcessPoolExecutor() as pool:
        rows = list(pool.map(_extract_file, jobs, chunksize=32))
    if rows:
        df = pd.DataFrame(rows)  # noqa: F841 - read by duckdb below
        con.execute("insert or replace into report_tickers select accession, cik, ticker, source, candidates from df")
    return len(rows)


def load_tiingo_list(con: duckdb.DuckDBPyConnection, path: Path) -> int:
    with zipfile.ZipFile(path) as z:
        t = pd.read_csv(z.open("supported_tickers.csv"), dtype=str)
    t = t.dropna(subset=["ticker", "startDate", "endDate"])
    t["ticker"] = t.ticker.str.upper()
    con.execute("""
        create or replace table tiingo_tickers as
        select ticker, exchange, assetType as asset_type, priceCurrency as currency,
               startDate::date as start_date, endDate::date as end_date from t""")
    return len(t)


# --- Tiingo client ----------------------------------------------------------------

class TiingoBudget:
    """Remembers which symbols were fetched in which month, and request times,
    so the free tier's limits are never exceeded across runs."""

    def __init__(self, path: Path):
        self.path = path
        self.state = json.loads(path.read_text()) if path.exists() else {"months": {}, "requests": []}

    def month_used(self) -> int:
        return len(self.state["months"].get(date.today().strftime("%Y-%m"), []))

    def wait_for_request(self):
        now = time.time()
        recent = [t for t in self.state["requests"] if now - t < 3600]
        if len(recent) >= HOURLY_REQUESTS:
            time.sleep(3600 - (now - min(recent)) + 5)
        self.state["requests"] = recent + [time.time()]

    def record(self, symbol: str):
        self.state["months"].setdefault(date.today().strftime("%Y-%m"), []).append(symbol)
        self.path.write_text(json.dumps(self.state))


def fetch_prices(symbols: list[str], cache: Path, log=print, token: str | None = None) -> dict:
    """Fetch each symbol's full daily history into cache/<symbol>.json, within
    budget. Already-cached symbols cost nothing. Returns counts."""
    import requests

    token = token or os.environ.get("TIINGO_API_KEY")
    if not token:
        raise SystemExit("set TIINGO_API_KEY (free account at tiingo.com)")
    cache.mkdir(parents=True, exist_ok=True)
    budget = TiingoBudget(cache / "_budget.json")
    done = fetched = missing = 0
    for symbol in symbols:
        out = cache / f"{symbol}.json"
        if out.exists():
            done += 1
            continue
        if budget.month_used() >= MONTHLY_SYMBOLS:
            log(f"monthly symbol budget reached ({MONTHLY_SYMBOLS}); rerun next month")
            break
        budget.wait_for_request()
        r = requests.get(TIINGO_PRICES.format(ticker=symbol), timeout=60,
                         params={"startDate": PRICE_START, "token": token, "format": "json"})
        budget.record(symbol)
        if r.status_code == 404:
            out.write_text("[]")
            missing += 1
            continue
        r.raise_for_status()
        out.write_text(r.text)
        fetched += 1
        if fetched % 25 == 0:
            log(f"  fetched {fetched} ({budget.month_used()} symbols used this month)")
    remaining = sum(not (cache / f"{s}.json").exists() for s in symbols)
    return {"cached_before": done, "fetched": fetched, "not_found": missing, "remaining": remaining}


def load_prices(con: duckdb.DuckDBPyConnection, cache: Path) -> int:
    """Cached histories -> table daily_prices(symbol, day, adj_close, close, volume)."""
    files = [p for p in cache.glob("*.json") if not p.name.startswith("_")]
    frames = []
    for p in files:
        rows = json.loads(p.read_text())
        if rows:
            df = pd.DataFrame(rows)[["date", "adjClose", "close", "volume"]]
            df["symbol"] = p.stem
            frames.append(df)
    prices = pd.concat(frames) if frames else pd.DataFrame(  # noqa: F841
        columns=["date", "adjClose", "close", "volume", "symbol"])
    con.execute("""
        create or replace table daily_prices as
        select symbol, date::date as day, adjClose as adj_close, close, volume from prices""")
    return con.execute("select count(*) from daily_prices").fetchone()[0]
