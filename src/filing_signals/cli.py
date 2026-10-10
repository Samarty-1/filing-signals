"""filing-signals command line.

    export EDGAR_USER_AGENT="filing-signals research you@example.com"
    filing-signals ingest --years 2010 2024 --firms 400
    filing-signals status
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import features, sections
from .http import EdgarClient, user_agent_from_env
from .pipeline import Params, Pipeline
from .warehouse import RawStore, connect

DATA = Path("data")


def main(argv: list[str] | None = None) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="filing-signals")
    parser.add_argument("--data", type=Path, default=DATA)
    sub = parser.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest", help="index, sample, and download annual reports")
    ing.add_argument("--years", type=int, nargs=2, default=[2010, 2024], metavar=("FIRST", "LAST"))
    ing.add_argument("--firms", type=int, default=400)
    ing.add_argument("--seed", type=int, default=20261009)
    ing.add_argument("--workers", type=int, default=4)

    sub.add_parser("status", help="summarise what the warehouse holds")
    sub.add_parser("report", help="data-quality findings (sql/report.sql)")
    sub.add_parser("parse", help="extract Risk Factors and MD&A sections (Rust parser)")
    sub.add_parser("features", help="year-over-year section similarity (Lazy Prices)")
    sub.add_parser("xbrl", help="XBRL facts: public float (size) and revenue (ground truth)")
    sub.add_parser("tickers", help="point-in-time ticker for every report, resolved to Tiingo symbols")
    sub.add_parser("prices", help="fetch daily prices from Tiingo within the free-tier budget "
                                  "(needs TIINGO_API_KEY; resumable, rerun until done)")
    sub.add_parser("backtest", help="Phase 4: the pre-registered return test")
    exp = sub.add_parser("export", help="Phase 5: static JSON for the TypeScript explorer (web/)")
    exp.add_argument("--out", type=Path, default=Path("web/public/data"))
    args = parser.parse_args(argv)

    args.data.mkdir(parents=True, exist_ok=True)
    con = connect(args.data / "filings.duckdb")
    sections.register_view(con, args.data)
    if args.cmd == "ingest":
        client = EdgarClient(user_agent_from_env())
        pipe = Pipeline(con, client, RawStore(args.data / "raw"))
        pipe.run(Params(args.years[0], args.years[1], args.firms, args.seed, args.workers))
        s = client.stats
        print(f"done: {s.requests:,} requests, {s.retries} retries, {s.bytes / 1e9:.2f} GB")
    if args.cmd == "parse":
        sections.parse_all(con, args.data)
    if args.cmd == "features":
        features.compute(con)
    if args.cmd == "xbrl":
        from . import xbrl
        client = EdgarClient(user_agent_from_env())
        xbrl.load(Pipeline(con, client, RawStore(args.data / "raw")))
    if args.cmd == "tickers":
        from . import prices
        lst = args.data / "prices" / "supported_tickers.zip"
        if not lst.exists():
            import requests
            lst.parent.mkdir(parents=True, exist_ok=True)
            lst.write_bytes(requests.get(prices.TIINGO_LIST_URL, timeout=120).content)
        prices.build_ticker_map(con, args.data)
        prices.load_tiingo_list(con, lst)
        n = con.execute(f"select count(*), count(distinct symbol) from ({prices.RESOLVED_SQL})").fetchone()
        print(f"tickers: {n[0]:,} reports resolve to {n[1]:,} Tiingo symbols")
    if args.cmd == "prices":
        from . import prices
        # only symbols the test can use: reports with a public float >= $100M
        symbols = [s for (s,) in con.execute(f"""
            select distinct r.symbol from ({prices.RESOLVED_SQL}) r
            join xbrl_facts x on x.accession = r.accession
            where x.concept = 'EntityPublicFloat' and x.value >= 1e8
            order by 1""").fetchall()]
        symbols.append("SPY")       # market check for the factor loadings
        out = prices.fetch_prices(symbols, args.data / "prices" / "tiingo")
        print(f"prices: {out}")
        print(f"daily_prices rows: {prices.load_prices(con, args.data / 'prices' / 'tiingo'):,}")
    if args.cmd == "backtest":
        from . import returns
        returns.run(con, args.data)
    if args.cmd == "export":
        from . import export
        export.run(con, args.data, args.out)
    if args.cmd == "report":
        report(con)
    else:
        status(con)


def report(con) -> None:
    from importlib import resources
    sql_dir = resources.files("filing_signals").joinpath("sql")
    text = sql_dir.joinpath("report.sql").read_text()
    tables = {t for (t,) in con.execute("select table_name from information_schema.tables").fetchall()}
    if "section_changes" in tables:
        text += sql_dir.joinpath("validation.sql").read_text()
    for block in text.split("-- name: ")[1:]:
        name, sql = block.split("\n", 1)
        print(f"\n== {name.strip()}")
        con.sql(sql).show(max_width=160)


def status(con) -> None:
    for label, sql in [
        ("universe", "select status, count(*) from universe group by 1 order by 1"),
        ("annual reports with text (versions, firms, amendments)",
         "select count(*), count(distinct cik), count(*) filter (where is_amendment) "
         "from annual_reports where stored_path is not null"),
        ("documents", "select count(*), round(sum(bytes_raw) / 1e9, 2), "
                      "round(sum(bytes_stored) / 1e9, 2) from documents"),
        # failures whose filing still lacks a header or document; a 404 on an
        # old URL scheme that was later fetched another way is resolved
        ("unresolved fetch failures",
         "select count(distinct x.accession) from fetch_failures x "
         "left join filing_headers h using (accession) left join documents d using (accession) "
         "where h.accession is null or d.accession is null"),
    ]:
        print(f"{label}: {con.execute(sql).fetchall()}")
