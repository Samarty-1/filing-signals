"""Phase 3b dataset: MD&A passages -> the year's total revenue, in USD.

Ground truth comes from the filing's own XBRL: a revenue fact carrying this
10-K's accession, covering a full year (350-380 days) that ends on the
report's fiscal period end, which excludes the prior-year comparatives every
10-K also tags. US GAAP renamed revenue over time, so concepts are tried in
order of preference.

Input: MD&A runs to ~12k tokens, so the passage is each line that mentions
revenue or sales plus the WINDOW lines after it, in document order, capped
at MAX_CHARS. The window matters: many filings put every table cell in its
own block, so "Net sales" and "17,253" land on separate lines, and keeping
only lines with both a keyword and a number found the answer in 61% of
reports when it is in the MD&A in 94%. A model can only be fairly scored on examples whose answer is in
its input, so `answerable` records whether the true figure appears in the
passage in any common written form ($1.2 billion, 1,234.5 million, 1,234,567
in thousands...). Metrics are reported on both the full set and the
answerable subset.
"""
from __future__ import annotations

import re
from pathlib import Path

import duckdb
import pandas as pd

REVENUE_CONCEPTS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueGoodsNet",
]
MAX_CHARS = 8000
WINDOW = 6
_MENTION = re.compile(r"\b(revenues?|net sales|total sales|sales)\b", re.I)
_NUMBER = re.compile(r"\d")
SPLITS = "case when fy <= 2017 then 'train' when fy <= 2019 then 'val' else 'test' end"

TRUTH_SQL = f"""
with rev as (
    select x.accession, x.concept, x.value,
           list_position({REVENUE_CONCEPTS}, x.concept) as pref
    from xbrl_facts x
    join annual_reports a on a.accession = x.accession
    where x.concept in (select unnest({REVENUE_CONCEPTS}))
      and x.unit = 'USD'
      and x.period_end = a.fiscal_period_end
      and date_diff('day', x.period_start, x.period_end) between 350 and 380
)
select a.accession, a.cik, a.fiscal_period_end, year(a.fiscal_period_end) as fy, {SPLITS} as split,
       r.concept, r.value as revenue_usd, s.text
from annual_reports a
join sections s using (accession)
join rev r using (accession)
where a.version_no = 1 and s.section = 'mdna' and s.status = 'found' and r.value > 0
qualify row_number() over (partition by a.accession order by r.pref, r.value desc) = 1
"""


def passage(mdna: str) -> str:
    lines = mdna.split("\n")
    wanted = set()
    for i, line in enumerate(lines):
        if _MENTION.search(line):
            wanted.update(range(i, min(i + WINDOW + 1, len(lines))))
    keep, size = [], 0
    for i in sorted(wanted):
        line = lines[i]
        # long prose without a number rarely carries the figure
        if not _NUMBER.search(line) and len(line) > 300:
            continue
        if size + len(line) > MAX_CHARS:
            break
        keep.append(line)
        size += len(line) + 1
    return "\n".join(keep)


def written_forms(usd: float) -> set[str]:
    """Strings a filing might use for this amount: scaled to units, thousands,
    millions or billions, at the precisions filings use."""
    forms = set()
    for scale in (1, 1e3, 1e6, 1e9):
        v = usd / scale
        for decimals in (0, 1, 2):
            text = f"{v:,.{decimals}f}"
            if text.replace(",", "").replace(".", "").strip("0") and v >= 1:
                forms.add(text)
    return forms


def _digits(s: str) -> str:
    return re.sub(r"[^\d.]", "", s)


def is_answerable(text: str, usd: float) -> bool:
    numbers = {_digits(m) for m in re.findall(r"\d[\d,]*(?:\.\d+)?", text)}
    return any(_digits(f) in numbers for f in written_forms(usd))


def build(con: duckdb.DuckDBPyConnection, out: Path, log=print) -> pd.DataFrame:
    df = con.execute(TRUTH_SQL).df()
    df["passage"] = df.text.map(passage)
    df = df.drop(columns="text")
    df["answerable"] = [is_answerable(p, v) for p, v in zip(df.passage, df.revenue_usd)]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    for split, g in df.groupby("split"):
        log(f"  {split}: {len(g)} reports, {g.answerable.mean():.0%} answerable, "
            f"median passage {int(g.passage.str.len().median()):,} chars")
    return df
