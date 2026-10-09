"""Phase 3a dataset: Risk Factors text -> adverse event in the next 12 months.

Unit: an original 10-K (version 1) whose Risk Factors section was found.

Label: within 365 days *after* the 10-K's acceptance date, the company files an
8-K reporting any of
    1.03  bankruptcy or receivership
    4.02  non-reliance on previously issued financial statements (a restatement)
    4.01  change in the company's certifying accountant
Only 8-Ks filed on a later date than the 10-K count, so a restatement
announced alongside the 10-K (which the 10-K itself discusses) is not a
"prediction". Components are kept as separate columns.

A filing gets a label only if its whole 365-day window had passed when the
filing lists were downloaded; later filings would be biased towards 0.

Feature available at filing time besides the text: whether the same events
happened in the 365 days *before* the 10-K (`prior_event`). It is the
baseline the text has to beat: auditor changes and restatements cluster.

Size: the public float the 10-K itself reports on its cover page (XBRL
dei:EntityPublicFloat, keyed by this filing's accession), so it is known at
filing time. Missing for filers without XBRL (mostly pre-2011 small
companies); `has_float` flags that instead of imputing silently.

Split by fiscal year: train <= 2017, validation 2018-2019, test >= 2020.
"""
from __future__ import annotations

from pathlib import Path

import duckdb

EVENTS = {"bankruptcy": "1.03", "restatement": "4.02", "auditor_change": "4.01"}
SPLITS = "case when fy <= 2017 then 'train' when fy <= 2019 then 'val' else 'test' end"


def _event_cols(window: str) -> str:
    return ",\n".join(
        f"""exists (select 1 from filings f
                    where f.cik = k.cik and f.form = '8-K' and f.items like '%{item}%'
                      and {window}) as {window_name(window)}{name}"""
        for name, item in EVENTS.items())


def window_name(window: str) -> str:
    return "prior_" if "- 365" in window else ""


AFTER = "f.filing_date between k.accepted_at::date + 1 and k.accepted_at::date + 365"
BEFORE = "f.filing_date between k.accepted_at::date - 365 and k.accepted_at::date - 1"

DATASET_SQL = f"""
with data_asof as (
    -- when the filing lists were downloaded: the label horizon
    select min(fetched_at) as t from filers
),
k as (
    select a.accession, a.cik, a.accepted_at, year(a.fiscal_period_end) as fy, s.text
    from annual_reports a
    join sections s using (accession)
    where a.version_no = 1 and s.section = 'risk_factors' and s.status = 'found'
      and a.accepted_at + interval 365 day <= (select t from data_asof)
),
labelled as (
    select k.*,
        {_event_cols(AFTER)},
        {_event_cols(BEFORE)}
    from k
)
select
    l.accession, l.cik, fy, {SPLITS} as split,
    f.public_float is not null                                          as has_float,
    ln(1 + coalesce(f.public_float, 0))                                 as log_float,
    (bankruptcy or restatement or auditor_change)                       as label,
    bankruptcy, restatement, auditor_change,
    (prior_bankruptcy or prior_restatement or prior_auditor_change)     as prior_event,
    text
from labelled l
left join (
    select accession, max(value) as public_float
    from xbrl_facts
    where concept = 'EntityPublicFloat' and unit = 'USD' and value > 0
    group by accession
) f using (accession)
order by l.accepted_at, l.accession
"""


def build(con: duckdb.DuckDBPyConnection, out: Path, log=print) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"copy ({DATASET_SQL}) to '{out.as_posix()}' (format parquet, compression zstd)")
    summary = con.execute(f"""
        select split, count(*) n, sum(label::int) pos, round(100 * avg(label::int), 1) pct,
               count(distinct cik) firms, min(fy), max(fy)
        from read_parquet('{out.as_posix()}') group by split order by min(fy)
    """).fetchall()
    for row in summary:
        log("  {}: {} reports, {} positive ({}%), {} firms, FY{}-{}".format(*row))
    return out
