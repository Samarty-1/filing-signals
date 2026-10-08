"""filing-signals command line.

    export EDGAR_USER_AGENT="filing-signals research you@example.com"
    filing-signals ingest --years 2010 2024 --firms 400
    filing-signals status
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    args = parser.parse_args(argv)

    args.data.mkdir(parents=True, exist_ok=True)
    con = connect(args.data / "filings.duckdb")
    if args.cmd == "ingest":
        client = EdgarClient(user_agent_from_env())
        pipe = Pipeline(con, client, RawStore(args.data / "raw"))
        pipe.run(Params(args.years[0], args.years[1], args.firms, args.seed, args.workers))
        s = client.stats
        print(f"done: {s.requests:,} requests, {s.retries} retries, {s.bytes / 1e9:.2f} GB")
    if args.cmd == "report":
        report(con)
    else:
        status(con)


def report(con) -> None:
    from importlib import resources
    text = resources.files("filing_signals").joinpath("sql", "report.sql").read_text()
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
